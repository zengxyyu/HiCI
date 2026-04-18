# HiCI attention module — pre-training variant.
# Modified based on https://github.com/lm-sys/FastChat

import warnings
from typing import Optional, Tuple
from functools import partial

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
import transformers
from einops import rearrange
from flash_attn import __version__ as flash_attn_version
from flash_attn.bert_padding import pad_input, unpad_input
from flash_attn.flash_attn_interface import (
    flash_attn_func,
    flash_attn_varlen_kvpacked_func,
    flash_attn_varlen_qkvpacked_func,
)
from transformers.models.llama.modeling_llama import (
    apply_rotary_pos_emb,
    repeat_kv,
    rotate_half,
    LlamaRMSNorm,
)
import math
import random
import os
import json

# ============================================================================
# Mixed-group training configuration
# ============================================================================
# MIXED_GROUP_TRAINING = True: randomly choose 2/4/8 groups per batch
# MIXED_GROUP_TRAINING = False: use a fixed group_size_ratio
MIXED_GROUP_TRAINING = False
GROUP_SIZE_RATIOS = [1 / 2, 1 / 4, 1 / 8]  # corresponding to 2, 4, 8 groups

group_size_ratio = 1 / 4  # default (used when MIXED_GROUP_TRAINING=False)

# ============================================================================
# Fixed segment-size mode (for evaluation; matches training segment size)
# ============================================================================
# USE_FIXED_SEGMENT_SIZE = True: use a fixed segment_size (recommended for eval)
# USE_FIXED_SEGMENT_SIZE = False: use group_size_ratio (used during training)
USE_FIXED_SEGMENT_SIZE = False
FIXED_SEGMENT_SIZE = 1024  # tokens per segment

# ============================================================================
# Full Attention + HiCI mode (used to verify HiCI module behaviour)
# ============================================================================
# USE_FULL_ATTN_WITH_HICI = True: no chunking, but still applies HiCI
#   - extract local slots from full input -> aggregate to global
#   - all tokens attend to [global, all_tokens]
# USE_FULL_ATTN_WITH_HICI = False: use chunked attention (default)
USE_FULL_ATTN_WITH_HICI = True

# Global state: all layers share the same grouping within one forward pass
_mixed_group_current_ratio = None
_mixed_group_call_count = 0  # used to detect a new forward pass
rank = dist.get_rank() if dist.is_initialized() else 0

# ============================================================================
# Causal context mode
# ============================================================================
# "none"           - default: all segments share one G (non-causal)
# "causal_gi"      - option A: segment_i uses G_i=Agg(L_1..L_i) and L_i
#                    G is causal; L_i has bounded intra-segment leakage
# "causal_shift"   - option B: segment_i uses G_{i-1}=Agg(L_1..L_{i-1}) and L_{i-1}
#                    strictly causal, zero leakage; segment_0 has no G or L
# "causal_shift_g" - option C: segment_i uses only G_{i-1}, no L appended
#                    strictly causal, zero leakage
# "causal_gi_gonly"- option D: segment_i uses G_i=Agg(L_1..L_i), no L in KV
#                    G is causal (includes current segment); double bottleneck
CAUSAL_CONTEXT_MODE = "none"

# ============================================================================
# Attention visualization configuration (inference only)
# ============================================================================
COLLECT_ATTENTION_FOR_VIZ = False

# Global collector — stores per-layer attention statistics
attention_visualizer = {
    "enabled": False,
    "layer_attn_to_global": [],  # per-layer mean attention fraction to global slots
    "layer_attn_to_local": [],   # per-layer mean attention fraction to local slots
    "layer_attn_to_tokens": [],  # per-layer mean attention fraction to tokens
    "segment_attention_maps": [],  # per-segment attention heatmaps (sparse)
    "num_global_slots": 0,
    "num_local_slots": 0,
    "segment_len": 0,
}


def reset_attention_visualizer():
    """Reset the attention visualizer; call before each inference pass."""
    global attention_visualizer
    attention_visualizer = {
        "enabled": COLLECT_ATTENTION_FOR_VIZ,
        "layer_attn_to_global": [],
        "layer_attn_to_local": [],
        "layer_attn_to_tokens": [],
        "segment_attention_maps": [],
        "num_global_slots": 0,
        "num_local_slots": 0,
        "segment_len": 0,
    }


def save_attention_stats(save_path="attention_stats.json"):
    """Save collected attention statistics to a JSON file."""
    import json

    stats = {
        "num_global_slots": attention_visualizer["num_global_slots"],
        "num_local_slots": attention_visualizer["num_local_slots"],
        "segment_len": attention_visualizer["segment_len"],
        "layer_attn_to_global": attention_visualizer["layer_attn_to_global"],
        "layer_attn_to_local": attention_visualizer["layer_attn_to_local"],
        "layer_attn_to_tokens": attention_visualizer["layer_attn_to_tokens"],
        "segment_attention_maps": attention_visualizer.get(
            "segment_attention_maps", []
        ),
    }
    with open(save_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Attention stats saved to {save_path}")
    if stats["segment_attention_maps"]:
        print(f"   Includes {len(stats['segment_attention_maps'])} attention heatmaps")


# LocalConstructor v1 — single-head, no Flash Attention
class LocalConstructor(nn.Module):
    """
    Learnable query slots for local context construction (LocalConstructor).

    This module is registered as a sub-module of LlamaAttention, ensuring:
    1. Parameters are properly registered in model.parameters()
    2. Optimizer tracks and updates these parameters
    3. Saved/loaded with model checkpoints

    Args:
        hidden_size: Model hidden dimension (e.g., 4096 for Llama-2-7B)
        num_local_slots: Number of learnable query slots (default: 8)
    """

    def __init__(
        self,
        hidden_size,
        num_local_slots=8,
        num_heads: Optional[int] = None,
        init_from_embeddings=None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_local_slots = num_local_slots

        # Learnable slot vectors Lslot: [num_slots, hidden_size]
        # Embedding-based initialization is disabled; use standard normal with 1/sqrt(H) std.
        if False:
            indices = torch.randperm(init_from_embeddings.size(0))[:num_local_slots]
            self.memory_slots = nn.Parameter(init_from_embeddings[indices].clone())
            if rank == 0:
                print(
                    f"    Initialized memory_slots from pretrained embeddings (sampled {num_local_slots} tokens)"
                )
        else:
            std = 1.0 / math.sqrt(hidden_size)
            self.memory_slots = nn.Parameter(
                torch.randn(num_local_slots, hidden_size) * std
            )
            rank = dist.get_rank() if dist.is_initialized() else 0
            layer_idx = getattr(self, "layer_idx", 0)
            if rank == 0 and layer_idx == 0:
                print(
                    f"LocalConstructor v1: Initialized memory_slots with std={std}"
                )

        # Cross-attention projections for summarization
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, hidden_states):
        """
        Compute global context via cross-attention.

        Args:
            hidden_states: [bsz, seq_len, hidden_size] - full input sequence

        Returns:
            global_context: [bsz, num_slots, hidden_size] - global summary
        """
        bsz, seq_len, _ = hidden_states.shape

        # Expand slot vectors for batch
        slots_input = self.memory_slots.unsqueeze(0).expand(
            bsz, -1, -1
        )  # [bsz, num_slots, hidden_size]

        # Cross-attention: slot vectors attend over full segment
        Q_slots = self.q_proj(slots_input)  # [bsz, num_slots, hidden_size]
        K_seq = self.k_proj(hidden_states)  # [bsz, seq_len, hidden_size]
        V_seq = self.v_proj(hidden_states)  # [bsz, seq_len, hidden_size]

        # Compute attention scores
        scores = torch.matmul(Q_slots, K_seq.transpose(-2, -1)) / math.sqrt(
            self.hidden_size
        )
        attn_weights = torch.softmax(scores, dim=-1)  # [bsz, num_slots, seq_len]

        # Apply attention to get global context
        global_context = torch.matmul(
            attn_weights, V_seq
        )  # [bsz, num_slots, hidden_size]

        return global_context


# LocalConstructorMulti v2 — multi-head cross-attention with optional bottleneck
class LocalConstructorMulti(nn.Module):
    """
    Multi-head Local Construction module (standard PyTorch implementation, no Flash Attention).

    Extracts M learnable query slot representations from each input segment via
    multi-head cross-attention. Supports optional bottleneck compression.

    This module is registered as a sub-module of LlamaAttention, ensuring:
    1. Parameters are properly registered in model.parameters()
    2. Optimizer tracks and updates these parameters
    3. Saved/loaded with model checkpoints

    Args:
        hidden_size: Model hidden dimension (e.g., 4096 for Llama-2-7B)
        num_local_slots: Number of learnable query slots (default: 8)
        num_heads: Number of attention heads (default: 32)
        init_from_embeddings: Optional pretrained embeddings for memory_slots initialization
        init_from_attn: Optional LlamaAttention layer for Q/K/V projection initialization
        use_bottleneck: Whether to use bottleneck compression (default: True)
        bottleneck_dim: Bottleneck dimension (default: 2048)
    """

    # Class variable: print initialization message only once
    _init_msg_printed = False

    def __init__(
        self,
        hidden_size,
        num_local_slots=8,
        num_heads=8,
        init_from_embeddings=None,
        init_from_attn=None,
        use_bottleneck: Optional[bool] = True,
        bottleneck_dim: Optional[int] = 512,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_local_slots = num_local_slots
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.use_bottleneck = use_bottleneck
        self.bottleneck_dim = bottleneck_dim

        assert hidden_size % num_heads == 0, (
            f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})"
        )

        # Learnable slot vectors Lslot: [num_slots, hidden_size]
        # Embedding-based initialization is disabled; use standard normal with 1/sqrt(H) std.
        if False:
            indices = torch.randperm(init_from_embeddings.size(0))[:num_local_slots]
            self.memory_slots = nn.Parameter(init_from_embeddings[indices].clone())
            rank = dist.get_rank() if dist.is_initialized() else 0
            if rank == 0:
                print(
                    f"    Initialized memory_slots from pretrained embeddings (sampled {num_local_slots} tokens)"
                )
        else:
            std = 1.0 / math.sqrt(hidden_size)
            self.memory_slots = nn.Parameter(
                torch.randn(num_local_slots, hidden_size) * std
            )
            rank = dist.get_rank() if dist.is_initialized() else 0
            if rank == 0 and not LocalConstructorMulti._init_msg_printed:
                print(
                    f"LocalConstructorMulti: Initialized memory_slots with std={std}"
                )
                LocalConstructorMulti._init_msg_printed = True

        # Cross-attention projections with optional bottleneck
        if use_bottleneck:
            # Validate bottleneck_dim divisibility
            assert bottleneck_dim % num_heads == 0, (
                f"bottleneck_dim ({bottleneck_dim}) must be divisible by num_heads ({num_heads})"
            )
            rank = dist.get_rank() if dist.is_initialized() else 0
            if rank == 0:
                print(
                    f"LocalConstructorMulti: bottleneck_dim={bottleneck_dim}, num_heads={num_heads}"
                )

            # Direct projection: hidden_size -> bottleneck_dim
            self.q_proj = nn.Linear(hidden_size, bottleneck_dim, bias=False)
            self.k_proj = nn.Linear(hidden_size, bottleneck_dim, bias=False)
            self.v_proj = nn.Linear(hidden_size, bottleneck_dim, bias=False)

            # Output projection: bottleneck_dim -> hidden_size
            self.o_proj = nn.Linear(bottleneck_dim, hidden_size, bias=False)

            # Effective dimensions for attention computation
            self.effective_dim = bottleneck_dim
            self.effective_head_dim = bottleneck_dim // num_heads
        else:
            # Standard full-size projections: hidden_size -> hidden_size
            self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
            self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
            self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
            self.o_proj = None  # no separate output projection needed

            # Use original dimensions
            self.effective_dim = hidden_size
            self.effective_head_dim = self.head_dim

        # Warm initialization from LLaMA pretrained Q/K/V weights (only without bottleneck)
        if init_from_attn is not None and not use_bottleneck:
            rank = dist.get_rank() if dist.is_initialized() else 0
            layer_idx = getattr(self, "layer_idx", 0)
            with torch.no_grad():
                self.q_proj.weight.copy_(init_from_attn.q_proj.weight)
                self.k_proj.weight.copy_(init_from_attn.k_proj.weight)
                self.v_proj.weight.copy_(init_from_attn.v_proj.weight)
            if rank == 0 and layer_idx == 0:
                print(
                    f"[LocalConstructorMulti] Initialized Q/K/V projections from LLaMA pretrained weights"
                )

    def forward(self, hidden_states, attention_mask=None):
        """
        Compute local context via multi-head cross-attention (standard PyTorch, no Flash Attention).

        Args:
            hidden_states: [bsz, seq_len, hidden_size] - full input sequence
            attention_mask: [bsz, seq_len] - 1 for valid tokens, 0 for padding (optional)

        Returns:
            global_context: [bsz, num_slots, hidden_size] - local summary
        """
        bsz, seq_len, _ = hidden_states.shape

        # Expand slot vectors for batch
        slots_input = self.memory_slots.unsqueeze(0).expand(
            bsz, -1, -1
        )  # [bsz, num_slots, hidden_size]

        # Cross-attention projections: project to target dimension (bottleneck or full)
        Q_slots = self.q_proj(slots_input)  # [bsz, num_slots, effective_dim]
        K_seq = self.k_proj(hidden_states)  # [bsz, seq_len, effective_dim]
        V_seq = self.v_proj(hidden_states)  # [bsz, seq_len, effective_dim]

        # Reshape for multi-head attention: [bsz, seqlen, num_heads, effective_head_dim]
        Q_slots = Q_slots.view(
            bsz, self.num_local_slots, self.num_heads, self.effective_head_dim
        )
        K_seq = K_seq.view(bsz, seq_len, self.num_heads, self.effective_head_dim)
        V_seq = V_seq.view(bsz, seq_len, self.num_heads, self.effective_head_dim)

        # Transpose for attention: [bsz, num_heads, seqlen, head_dim]
        Q_slots = Q_slots.transpose(1, 2)  # [bsz, num_heads, num_slots, effective_head_dim]
        K_seq = K_seq.transpose(1, 2)  # [bsz, num_heads, seq_len, effective_head_dim]
        V_seq = V_seq.transpose(1, 2)  # [bsz, num_heads, seq_len, effective_head_dim]

        # Compute attention scores: Q @ K^T -> [bsz, num_heads, num_slots, seq_len]
        scores = torch.matmul(Q_slots, K_seq.transpose(-2, -1)) / math.sqrt(
            self.effective_head_dim
        )

        # Apply attention mask if provided
        if attention_mask is not None:
            # attention_mask: [bsz, seq_len] - 1 for valid, 0 for padding
            mask_expanded = attention_mask.unsqueeze(1).unsqueeze(
                2
            )  # [bsz, 1, 1, seq_len]
            mask_value = torch.finfo(scores.dtype).min
            scores = scores.masked_fill(mask_expanded == 0, mask_value)

        # Softmax: [bsz, num_heads, num_slots, seq_len]
        attn_weights = torch.softmax(scores, dim=-1)

        # Weighted sum: [bsz, num_heads, num_slots, effective_head_dim]
        attn_output = torch.matmul(attn_weights, V_seq)

        # Merge heads: [bsz, num_slots, effective_dim]
        attn_output = attn_output.transpose(1, 2).contiguous()
        global_context = attn_output.view(
            bsz, self.num_local_slots, self.effective_dim
        )

        # Apply output projection if using bottleneck: effective_dim -> hidden_size
        if self.o_proj is not None:
            global_context = self.o_proj(
                global_context
            )  # [bsz, num_slots, hidden_size]

        return global_context


# LocalConstructorFlash v3 — Flash Attention cross-attention with padding support
class LocalConstructorFlash(nn.Module):
    """
    Learnable query slots for local context construction using Flash Attention.

    Supports very long sequences (100k+) via O(N) memory complexity and
    correct padding handling via unpad_input.

    This module is registered as a sub-module of LlamaAttention, ensuring:
    1. Parameters are properly registered in model.parameters()
    2. Optimizer tracks and updates these parameters
    3. Saved/loaded with model checkpoints

    Args:
        hidden_size: Model hidden dimension (e.g., 4096 for Llama-2-7B)
        num_local_slots: Number of learnable query slots (default: 8)
        num_heads: Number of attention heads (default: 32)
        init_from_embeddings: Optional pretrained embeddings for memory_slots initialization
        init_from_attn: Optional LlamaAttention layer for Q/K/V projection initialization
    """

    # Class variable: print initialization message only once
    _init_msg_printed = False

    def __init__(
        self,
        hidden_size,
        num_local_slots=8,
        num_heads=32,
        init_from_embeddings=None,
        init_from_attn=None,
        use_bottleneck: Optional[bool] = True,
        bottleneck_dim: Optional[int] = 512,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_local_slots = num_local_slots
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.use_bottleneck = use_bottleneck
        self.bottleneck_dim = bottleneck_dim

        assert hidden_size % num_heads == 0, (
            f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})"
        )

        # Learnable slot vectors Lslot: [num_slots, hidden_size]
        # Embedding-based initialization is disabled; use standard normal with 1/sqrt(H) std.
        if False:
            indices = torch.randperm(init_from_embeddings.size(0))[:num_local_slots]
            self.memory_slots = nn.Parameter(init_from_embeddings[indices].clone())
            rank = dist.get_rank() if dist.is_initialized() else 0
            if rank == 0:
                print(
                    f"    Initialized memory_slots from pretrained embeddings (sampled {num_local_slots} tokens)"
                )
        else:
            std = 1.0 / math.sqrt(hidden_size)
            self.memory_slots = nn.Parameter(
                torch.randn(num_local_slots, hidden_size) * std
            )
            rank = dist.get_rank() if dist.is_initialized() else 0
            if rank == 0 and not LocalConstructorFlash._init_msg_printed:
                print(
                    f"LocalConstructorFlash: Initialized memory_slots with std={std}"
                )
                LocalConstructorFlash._init_msg_printed = True

        # Cross-attention projections for summarization
        if use_bottleneck:
            # Validate bottleneck_dim divisibility
            assert bottleneck_dim % num_heads == 0, (
                f"bottleneck_dim ({bottleneck_dim}) must be divisible by num_heads ({num_heads})"
            )
            rank = dist.get_rank() if dist.is_initialized() else 0
            if rank == 0:
                print(f"LocalConstructorFlash: bottleneck_dim={bottleneck_dim}")

            # Direct projection: hidden_size -> bottleneck_dim
            self.q_proj = nn.Linear(hidden_size, bottleneck_dim, bias=False)
            self.k_proj = nn.Linear(hidden_size, bottleneck_dim, bias=False)
            self.v_proj = nn.Linear(hidden_size, bottleneck_dim, bias=False)

            # Output projection: bottleneck_dim -> hidden_size
            self.o_proj = nn.Linear(bottleneck_dim, hidden_size, bias=False)

            # Effective dimensions for attention computation
            self.effective_dim = bottleneck_dim
            self.effective_head_dim = bottleneck_dim // num_heads
        else:
            # Standard full-size projections: hidden_size -> hidden_size
            self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
            self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
            self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
            self.o_proj = None  # no separate output projection needed

            # Use original dimensions
            self.effective_dim = hidden_size
            self.effective_head_dim = self.head_dim

        # Warm initialization from LLaMA pretrained Q/K/V weights
        if init_from_attn is not None:
            rank = dist.get_rank() if dist.is_initialized() else 0
            layer_idx = getattr(self, "layer_idx", 0)
            with torch.no_grad():
                self.q_proj.weight.copy_(init_from_attn.q_proj.weight)
                self.k_proj.weight.copy_(init_from_attn.k_proj.weight)
                self.v_proj.weight.copy_(init_from_attn.v_proj.weight)
            if rank == 0 and layer_idx == 0:
                print(
                    f"[LocalConstructorFlash] Initialized Q/K/V projections from LLaMA pretrained weights"
                )

    def forward(self, hidden_states, attention_mask=None):
        """
        Compute local context via Flash Attention cross-attention.

        Uses flash_attn_varlen_kvpacked_func:
        - Q: query slots (no padding), fixed length = num_slots
        - K/V: input sequence (potentially padded); padding removed via unpad_input

        Args:
            hidden_states: [bsz, seq_len, hidden_size] - full input sequence
            attention_mask: [bsz, seq_len] - 1 for valid, 0 for padding (optional)

        Returns:
            global_context: [bsz, num_slots, hidden_size] - local summary
        """
        bsz, seq_len, _ = hidden_states.shape

        # Expand slot vectors for batch
        slots_input = self.memory_slots.unsqueeze(0).expand(
            bsz, -1, -1
        )  # [bsz, num_slots, hidden_size]

        # Cross-attention projections: project to target dimension (bottleneck or full)
        Q_slots = self.q_proj(slots_input)  # [bsz, num_slots, effective_dim]
        K_seq = self.k_proj(hidden_states)  # [bsz, seq_len, effective_dim]
        V_seq = self.v_proj(hidden_states)  # [bsz, seq_len, effective_dim]

        # Reshape for multi-head attention: [bsz, seqlen, num_heads, effective_head_dim]
        Q_slots = Q_slots.view(
            bsz, self.num_local_slots, self.num_heads, self.effective_head_dim
        )
        K_seq = K_seq.view(bsz, seq_len, self.num_heads, self.effective_head_dim)
        V_seq = V_seq.view(bsz, seq_len, self.num_heads, self.effective_head_dim)

        if attention_mask is not None:
            # Flash Attention + unpad (correct padding handling)
            # 1. Remove padding tokens from K/V via unpad_input
            # 2. Q (slot vectors) has no padding — fixed num_slots per sample
            # 3. Use flash_attn_varlen_kvpacked_func for variable-length cross-attention

            # Pack K and V together: [bsz, seq_len, 2, num_heads, head_dim]
            kv = torch.stack([K_seq, V_seq], dim=2)

            # Reshape for unpad_input: [bsz, seq_len, 2 * num_heads * head_dim]
            kv_for_unpad = rearrange(kv, "b s two h d -> b s (two h d)")

            # Remove padding from K/V
            # kv_unpad: [total_valid_kv_tokens, 2 * num_heads * head_dim]
            # cu_seqlens_kv: [bsz + 1], e.g., [0, 500, 1300]
            kv_unpad, indices_kv, cu_seqlens_kv, max_seqlen_kv = unpad_input(
                kv_for_unpad, attention_mask
            )

            # Reshape back: [total_valid_kv_tokens, 2, num_heads, head_dim]
            kv_unpad = rearrange(
                kv_unpad, "nnz (two h d) -> nnz two h d", two=2, h=self.num_heads
            )

            # Q has no padding, flatten: [bsz, num_slots, h, d] -> [bsz * num_slots, h, d]
            q_unpad = rearrange(Q_slots, "b s h d -> (b s) h d")

            # cu_seqlens_q: Q length per sample is always num_slots
            # e.g., bsz=2, num_slots=16 -> cu_seqlens_q = [0, 16, 32]
            cu_seqlens_q = torch.arange(
                0,
                (bsz + 1) * self.num_local_slots,
                self.num_local_slots,
                device=hidden_states.device,
                dtype=torch.int32,
            )

            # Flash Attention variable-length cross-attention
            # Q: [total_q, num_heads, effective_head_dim] where total_q = bsz * num_slots
            # KV: [total_kv, 2, num_heads, effective_head_dim] where total_kv = sum of valid lengths
            output_unpad = flash_attn_varlen_kvpacked_func(
                q_unpad,  # [bsz * num_slots, num_heads, effective_head_dim]
                kv_unpad,  # [total_valid_kv, 2, num_heads, effective_head_dim]
                cu_seqlens_q,  # [bsz + 1]
                cu_seqlens_kv,  # [bsz + 1]
                self.num_local_slots,  # max_seqlen_q (fixed)
                max_seqlen_kv,  # max_seqlen_kv (longest valid length in batch)
                dropout_p=0.0,
                softmax_scale=None,  # default: 1/sqrt(head_dim)
                causal=False,  # cross-attention does not use causal mask
            )
            # output_unpad: [bsz * num_slots, num_heads, effective_head_dim]

            # Reshape back: [bsz, num_slots, effective_dim]
            global_context = rearrange(
                output_unpad, "(b s) h d -> b s (h d)", b=bsz, s=self.num_local_slots
            )
        else:
            # No padding: use the simpler flash_attn_func (most efficient path)
            global_context = flash_attn_func(
                Q_slots,  # [bsz, num_slots, num_heads, effective_head_dim]
                K_seq,  # [bsz, seq_len, num_heads, effective_head_dim]
                V_seq,  # [bsz, seq_len, num_heads, effective_head_dim]
                dropout_p=0.0,
                softmax_scale=None,
                causal=False,
            )
            # global_context: [bsz, num_slots, num_heads, effective_head_dim]

            # Reshape: [bsz, num_slots, effective_dim]
            global_context = rearrange(global_context, "b s h d -> b s (h d)")

        # Apply output projection if using bottleneck: effective_dim -> hidden_size
        if self.o_proj is not None:
            global_context = self.o_proj(
                global_context
            )  # [bsz, num_slots, hidden_size]

        return global_context


# ============================================================================
# GlobalIntegrator_new — statistical aggregation + lightweight attention
# ============================================================================
class GlobalIntegrator_new(nn.Module):
    """
    GlobalIntegrator with EMA prior (statistics + lightweight attention).

    Two-stage design:
        Stage 1 (stable foundation): statistical compression
            local_repr: [bsz, N, hidden_size]
            -> 5 statistics: [mean, max, min, std, norm_mean]
            -> separate compressors: each hidden_size -> compress_dim
            -> compressed_stats: [bsz, 5, compress_dim]

        Stage 2 (learned refinement): lightweight attention
            global_queries: [global_slots, compress_dim]  (learnable)
            compressed_stats: [bsz, 5, compress_dim]
            -> cross-attention in compress_dim space
            -> G_compressed: [bsz, global_slots, compress_dim]
            -> expand to hidden_size
            -> G: [bsz, global_slots, hidden_size]

    Input:  local_repr [bsz, num_chunks, local_slots, hidden_size]
    Output: G              [bsz, global_slots, hidden_size]
    """

    _init_msg_printed = False

    def __init__(
        self,
        hidden_size: int = 4096,
        global_slots: int = 4,
        compress_dim: int = 512,
        local_slots: int = 16,  # compatibility parameter
        use_bottleneck: bool = False,  # compatibility parameter
        bottleneck_dim: int = 4096,  # compatibility parameter
        init_from_embeddings=None,
        use_high_norm_init: Optional[bool] = True,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_global = global_slots
        self.compress_dim = compress_dim
        self.use_high_norm_init = use_high_norm_init

        # Stage 1: separate statistical compressors (with LayerNorm for stability)
        self.stat_compressors = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_size, compress_dim, bias=False),
                    nn.LayerNorm(compress_dim),
                )
                for _ in range(5)  # mean, max, min, std, norm_mean
            ]
        )
        # self.stat_compressors = nn.ModuleList(
        #     [
        #         nn.Linear(hidden_size, compress_dim, bias=False)
        #         for _ in range(5)  # mean, max, min, std, norm_mean
        #     ]
        # )

        # Stage 2: lightweight attention in compressed space
        self.global_queries = nn.Parameter(torch.zeros(global_slots, compress_dim))

        # Q/K/V projections in compressed space
        self.q_proj = nn.Linear(compress_dim, compress_dim, bias=False)
        self.k_proj = nn.Linear(compress_dim, compress_dim, bias=False)
        self.v_proj = nn.Linear(compress_dim, compress_dim, bias=False)

        # Stage 3: expand to hidden_size
        self.expand = nn.Linear(compress_dim, hidden_size, bias=False)
        std_init = 0.02 / math.sqrt(compress_dim)
        nn.init.normal_(self.expand.weight, mean=0.0, std=std_init)
        self.expand_scale = nn.Parameter(torch.tensor(0.1))

        # EMA buffer (long-term prior for global queries)
        self.register_buffer("ema_global", torch.zeros(1, global_slots, hidden_size))
        self.ema_decay = 0.95
        self.ema_weight = 0.1  # EMA influence weight on queries

        # Cache compressed EMA to avoid recomputing every forward pass
        self.register_buffer(
            "ema_compressed_cache", torch.zeros(global_slots, compress_dim)
        )
        self._ema_cache_valid = False  # marks whether the cache is valid

        # Cache first compressor reference (set in _init_weights)
        self._first_compressor = None

        # Initialize weights
        self._init_weights(
            init_from_embeddings, use_high_norm_init=self.use_high_norm_init
        )

        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank == 0 and not GlobalIntegrator_new._init_msg_printed:
            total_params = sum(p.numel() for p in self.parameters())
            print(f"   GlobalIntegrator_new initialized")
            print(f"       - Design: Statistical Aggregation + Lightweight Attention")
            print(f"       - Global slots: {global_slots}")
            print(f"       - Compress dim: {compress_dim}")
            print(
                f"       - Params/layer: {total_params:,} ({total_params / 1e6:.1f}M)"
            )
            print(
                f"       - 32 layers: {total_params * 32:,} ({total_params * 32 / 1e9:.2f}B)"
            )
            GlobalIntegrator_new._init_msg_printed = True

    def _init_weights(self, embed_weight, use_high_norm_init=False):
        """Initialize global queries.

        Args:
            embed_weight: pretrained embedding weights [vocab_size, hidden_size]
            use_high_norm_init: whether to select high-norm embeddings for initialization
        """
        # Cache the full compressor (Linear+LayerNorm) so EMA and stats use the same normalization
        self._first_compressor = self.stat_compressors[0]

        if embed_weight is not None:
            with torch.no_grad():
                if use_high_norm_init:
                    embed_norms = torch.norm(embed_weight, dim=-1)  # [vocab_size]
                    _, top_indices = torch.topk(embed_norms, k=self.num_global)
                    indices = top_indices
                else:
                    indices = torch.randperm(embed_weight.size(0))[: self.num_global]

                init_embeddings = embed_weight[indices]  # [global_slots, hidden_size]

                init_embeddings = init_embeddings.to(
                    self._first_compressor[0].weight.dtype
                )

                # Compress using the first stat compressor (queries and stats in the same space)
                init_compressed = self._first_compressor(
                    init_embeddings
                )  # [global_slots, compress_dim]
                self.global_queries.copy_(init_compressed)

                # EMA uses the full embedding
                self.ema_global.copy_(init_embeddings.unsqueeze(0))

                # Initialize EMA compressed cache
                self.ema_compressed_cache.copy_(init_compressed)
                self._ema_cache_valid = True
        else:
            nn.init.xavier_uniform_(self.global_queries)

    def forward(self, local_repr):
        """
        Two-stage forward pass.

        Args:
            local_repr: [bsz, num_chunks, local_slots, hidden_size]

        Returns:
            G: [bsz, global_slots, hidden_size]
        """
        bsz, num_chunks, local_slots, hidden_size = local_repr.shape

        # ========== Stage 1: statistical compression ==========
        # Flatten: [bsz, num_chunks * local_slots, hidden_size]
        all_local_flat = local_repr.reshape(bsz, -1, hidden_size)

        # Compute 5 statistics
        mean_pool = all_local_flat.mean(dim=1)
        max_pool, _ = all_local_flat.max(dim=1)
        min_pool, _ = all_local_flat.min(dim=1)

        # Compute std in fp32 for numerical stability
        with torch.cuda.amp.autocast(enabled=False):
            std_pool = all_local_flat.float().std(dim=1, unbiased=False)
        std_pool = std_pool.to(all_local_flat.dtype)

        norm_mean = F.normalize(mean_pool, dim=-1, p=2)

        # Separate compression: each statistic hidden_size -> compress_dim
        stats_list = [mean_pool, max_pool, min_pool, std_pool, norm_mean]
        compressed_stats = [
            self.stat_compressors[i](stat) for i, stat in enumerate(stats_list)
        ]

        # Stack: [bsz, 5, compress_dim]
        stats_stacked = torch.stack(compressed_stats, dim=1)

        # ========== Stage 2: lightweight attention ==========
        # Q: [bsz, global_slots, compress_dim]
        Q = self.global_queries.unsqueeze(0).expand(bsz, -1, -1)

        # Blend in EMA long-term prior using cached compressed result
        if hasattr(self, "ema_global") and hasattr(self, "ema_weight"):
            if self._ema_cache_valid:
                ema_compressed = self.ema_compressed_cache.unsqueeze(0).expand(
                    bsz, -1, -1
                )
            else:
                # Cache miss: recompress (only happens after EMA update)
                ema_compressed = self._first_compressor(self.ema_global.squeeze(0))
                self.ema_compressed_cache.copy_(ema_compressed)
                self._ema_cache_valid = True
                ema_compressed = ema_compressed.unsqueeze(0).expand(bsz, -1, -1)

            # Weighted blend: Q = learned + alpha * prior (detach EMA from gradient)
            Q = Q + self.ema_weight * ema_compressed.detach()

        # Project to attention space
        Q = self.q_proj(Q)

        # K, V: [bsz, 5, compress_dim]
        K = self.k_proj(stats_stacked)
        V = self.v_proj(stats_stacked)

        # Scaled dot-product attention
        scale = self.compress_dim**-0.5
        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) * scale
        # [bsz, global_slots, 5]

        attn_probs = F.softmax(attn_weights, dim=-1)

        # Weighted sum
        G_compressed = torch.matmul(attn_probs, V)
        # [bsz, global_slots, compress_dim]

        # ========== Stage 3: expand to hidden_size ==========
        G_unscaled = self.expand(G_compressed)  # [bsz, global_slots, hidden_size]
        G = G_unscaled * self.expand_scale

        # EMA update during training; invalidate compressed cache
        if self.training:
            with torch.no_grad():
                # Use unscaled G to avoid expand_scale changes corrupting EMA
                batch_mean_G = G_unscaled.mean(dim=0, keepdim=True)
                self.ema_global.copy_(
                    self.ema_decay * self.ema_global
                    + (1 - self.ema_decay) * batch_mean_G
                )
                self._ema_cache_valid = False

        return G


# ============================================================================
# GlobalIntegrator — simplified variant (no EMA)
# ============================================================================
class GlobalIntegrator(nn.Module):
    """
    GlobalIntegrator — statistical aggregation + multi-head attention (no EMA).

    Two-stage design:
        Stage 1: extract 5 statistics from local memories, compress to compress_dim
        Stage 2: global learned queries attend over compressed statistics

    Input:  local_repr [bsz, num_chunks, local_slots, hidden_size]
    Output: global_context  [bsz, global_slots, hidden_size]
    """

    _init_msg_printed = False

    def __init__(
        self,
        hidden_size: int = 4096,
        global_slots: int = 4,
        compress_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.0,
        local_slots: int = 16,  # compatibility parameter
        use_bottleneck: bool = False,  # compatibility parameter
        bottleneck_dim: int = 4096,  # compatibility parameter
        init_from_embeddings: Optional[torch.Tensor] = None,
        use_high_norm_init: bool = True,
        output_scale_init: float = 0.1,
    ):
        """
        Args:
            hidden_size: hidden dimension (typically 4096)
            global_slots: number of global context slots (typically 4-16)
            compress_dim: compression dimension (typically 512)
            num_heads: number of attention heads (compress_dim must be divisible)
            dropout: attention dropout probability
            init_from_embeddings: pretrained embeddings for initialization
            use_high_norm_init: whether to use high-norm token selection for initialization
            output_scale_init: initial value for output scale
        """
        super().__init__()

        # ============ Parameter validation ============
        assert compress_dim % num_heads == 0, (
            f"compress_dim ({compress_dim}) must be divisible by num_heads ({num_heads})"
        )
        assert output_scale_init > 0, "output_scale_init must be positive"

        # ============ Save configuration ============
        self.hidden_size = hidden_size
        self.num_global = global_slots  # compatibility alias
        self.global_slots = global_slots
        self.compress_dim = compress_dim
        self.num_heads = num_heads
        self.head_dim = compress_dim // num_heads
        self.dropout_p = dropout
        self.use_high_norm_init = use_high_norm_init
        self._output_scale_init = output_scale_init  # saved for softplus inverse

        # ============ Stage 1: statistical compressors ============
        self.stat_names = ["mean", "max", "min", "std", "norm_mean"]
        self.stat_compressors = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_size, compress_dim, bias=False),
                    nn.LayerNorm(compress_dim),
                )
                for _ in range(5)
            ]
        )

        # ============ Stage 2: lightweight multi-head attention ============
        self.global_queries = nn.Parameter(torch.zeros(global_slots, compress_dim))

        self.q_proj = nn.Linear(compress_dim, compress_dim, bias=False)
        self.k_proj = nn.Linear(compress_dim, compress_dim, bias=False)
        self.v_proj = nn.Linear(compress_dim, compress_dim, bias=False)
        # Add output projection (standard MHA design)
        self.o_proj = nn.Linear(compress_dim, compress_dim, bias=False)

        self.attn_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # ============ Stage 3: dimension expansion ============
        self.expand = nn.Linear(compress_dim, hidden_size, bias=False)
        # LLaMA-style small-norm initialization
        std_init = 0.02 / math.sqrt(compress_dim)
        nn.init.normal_(self.expand.weight, mean=0.0, std=std_init)

        # Use softplus to ensure scale is always positive
        # Store parameter in log-space; softplus ensures > 0
        # softplus(x) ≈ x when x > 0, softplus(0) ≈ 0.693
        # Invert softplus to get the initial parameter value
        init_param = math.log(math.exp(output_scale_init) - 1)  # inverse softplus
        self.expand_scale_param = nn.Parameter(torch.tensor(init_param))

        # ============ Weight initialization ============
        self._init_weights(init_from_embeddings)
        self._print_init_info()

    @property
    def expand_scale(self) -> torch.Tensor:
        """Scale via softplus to ensure it remains positive."""
        return F.softplus(self.expand_scale_param)

    def _init_weights(self, embed_weight: Optional[torch.Tensor] = None):
        """Initialize weights."""
        if embed_weight is not None:
            with torch.no_grad():
                if self.use_high_norm_init:
                    embed_norms = torch.norm(embed_weight, dim=-1)
                    _, top_indices = torch.topk(embed_norms, k=self.global_slots)
                    init_embeddings = embed_weight[top_indices]
                else:
                    indices = torch.randperm(embed_weight.size(0))[: self.global_slots]
                    init_embeddings = embed_weight[indices]

                # Ensure device and dtype match
                target_device = self.stat_compressors[0][0].weight.device
                target_dtype = self.stat_compressors[0][0].weight.dtype
                init_embeddings = init_embeddings.to(
                    device=target_device, dtype=target_dtype
                )

                init_compressed = self.stat_compressors[0](init_embeddings)
                self.global_queries.copy_(init_compressed)
        else:
            nn.init.xavier_uniform_(self.global_queries)

        # Initialize projection layers
        for proj in [self.q_proj, self.k_proj, self.v_proj, self.o_proj]:
            nn.init.xavier_uniform_(proj.weight)

    def _print_init_info(self):
        """Print initialization info."""
        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank == 0 and not GlobalIntegrator._init_msg_printed:
            total_params = sum(p.numel() for p in self.parameters())
            print(f"   GlobalIntegratorClean initialized (no-EMA simplified)")
            print(f"       - Design: Statistical Aggregation + Lightweight MHA")
            print(f"       - Global slots: {self.global_slots}")
            print(f"       - Compress dim: {self.compress_dim}")
            print(f"       - Num heads: {self.num_heads}")
            print(f"       - Output scale (init): {self._output_scale_init}")
            print(
                f"       - Params/layer: {total_params:,} ({total_params / 1e6:.1f}M)"
            )
            GlobalIntegrator._init_msg_printed = True

    def forward(self, local_repr: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            local_repr: [bsz, num_chunks, local_slots, hidden_size]

        Returns:
            G: [bsz, global_slots, hidden_size]

        Data flow:
            local_repr [bsz, C, L, H]
                ↓ reshape
            all_local [bsz, C*L, H]
                ↓ 5 statistics
            stats [bsz, H] × 5
                ↓ compress
            compressed_stats [bsz, 5, D]
                ↓ Multi-Head Attention
            G_compressed [bsz, G, D]
                ↓ expand + scale
            G [bsz, G, H]
        """
        bsz, num_chunks, local_slots, hidden_size = local_repr.shape

        # ========== Stage 1: extract and compress statistics ==========
        # Flatten: [bsz, num_chunks * local_slots, hidden_size]
        all_local = local_repr.reshape(bsz, -1, hidden_size)

        # Compute 5 statistics, each [bsz, hidden_size]
        mean_pool = all_local.mean(dim=1)
        max_pool, _ = all_local.max(dim=1)
        min_pool, _ = all_local.min(dim=1)

        # Compute std in fp32 with numerical stability protection
        with torch.amp.autocast(device_type="cuda", enabled=False):
            all_local_fp32 = all_local.float()
            # Add eps to guard against all-zero inputs
            std_pool = all_local_fp32.std(dim=1, unbiased=False).clamp(min=1e-6)
        std_pool = std_pool.to(all_local.dtype)

        # L2-normalized mean (direction vector)
        norm_mean = F.normalize(mean_pool, dim=-1, p=2, eps=1e-6)

        # Compress each statistic separately: [bsz, hidden_size] -> [bsz, compress_dim]
        stats_list = [mean_pool, max_pool, min_pool, std_pool, norm_mean]
        compressed_stats = torch.stack(
            [self.stat_compressors[i](stat) for i, stat in enumerate(stats_list)], dim=1
        )  # [bsz, 5, compress_dim]

        # ========== Stage 2: lightweight multi-head attention ==========
        # Q: [bsz, global_slots, compress_dim]
        Q = self.global_queries.unsqueeze(0).expand(bsz, -1, -1)
        Q = self.q_proj(Q)

        # K, V: [bsz, 5, compress_dim]
        K = self.k_proj(compressed_stats)
        V = self.v_proj(compressed_stats)

        # Split heads: [bsz, num_heads, seq_len, head_dim]
        Q = Q.view(bsz, self.global_slots, self.num_heads, self.head_dim).transpose(
            1, 2
        )
        K = K.view(bsz, 5, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(bsz, 5, self.num_heads, self.head_dim).transpose(1, 2)
        # Q: [bsz, num_heads, global_slots, head_dim]
        # K: [bsz, num_heads, 5, head_dim]
        # V: [bsz, num_heads, 5, head_dim]

        # Scaled Dot-Product Attention
        scale = self.head_dim**-0.5
        # [bsz, num_heads, global_slots, 5]
        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) * scale
        attn_probs = F.softmax(attn_weights, dim=-1)
        attn_probs = self.attn_dropout(attn_probs)

        # [bsz, num_heads, global_slots, head_dim]
        attn_output = torch.matmul(attn_probs, V)

        # Merge heads: [bsz, global_slots, compress_dim]
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, self.global_slots, self.compress_dim)

        # Output projection (merges multi-head information)
        G_compressed = self.o_proj(attn_output)

        # ========== Stage 3: expand to hidden_size ==========
        # expand_scale via softplus is always positive
        G = self.expand(G_compressed) * self.expand_scale

        return G

    def forward_causal(self, local_repr: torch.Tensor) -> torch.Tensor:
        """
        Causal forward pass: compute an independent G_i for each segment.

        For segment i, G_i is computed only from L_1, ..., L_i cumulative statistics,
        guaranteeing no leakage from future segments.

        Args:
            local_repr: [bsz, num_chunks, local_slots, hidden_size]

        Returns:
            G_all: [bsz, num_chunks, global_slots, hidden_size]
                   G_all[:, i, :, :] = Agg(L_1, ..., L_{i+1})
        """
        bsz, num_chunks, local_slots, hidden_size = local_repr.shape

        # ========== Stage 1: cumulative statistics ==========
        # Per-chunk aggregation along the chunk dimension, then cumulate
        # local_repr: [bsz, N, L, H]

        # Per-chunk aggregation
        sum_per_chunk = local_repr.sum(dim=2)           # [bsz, N, H]
        max_per_chunk = local_repr.max(dim=2).values    # [bsz, N, H]
        min_per_chunk = local_repr.min(dim=2).values    # [bsz, N, H]

        # Cumulative statistics along chunk dimension
        cumsum = sum_per_chunk.cumsum(dim=1)                # [bsz, N, H]
        counts = torch.arange(1, num_chunks + 1, device=local_repr.device,
                              dtype=local_repr.dtype).view(1, -1, 1) * local_slots
        cum_mean = cumsum / counts                          # [bsz, N, H]

        cum_max = max_per_chunk.cummax(dim=1).values        # [bsz, N, H]
        cum_min = min_per_chunk.cummin(dim=1).values        # [bsz, N, H]

        # Cumulative std: sqrt(E[X²] - E[X]²)
        with torch.amp.autocast(device_type="cuda", enabled=False):
            local_fp32 = local_repr.float()
            sq_sum_per_chunk = (local_fp32 ** 2).sum(dim=2)  # [bsz, N, H]
            cum_sq_sum = sq_sum_per_chunk.cumsum(dim=1)
            counts_f = counts.float()
            cum_sq_mean = cum_sq_sum / counts_f
            cum_mean_f = cumsum.float() / counts_f
            cum_var = (cum_sq_mean - cum_mean_f ** 2).clamp(min=1e-12)
            cum_std = cum_var.sqrt()
        cum_std = cum_std.to(local_repr.dtype)

        cum_norm_mean = F.normalize(cum_mean, dim=-1, p=2, eps=1e-6)  # [bsz, N, H]

        # Stack: [bsz, N, 5, H]
        cum_stats = torch.stack([cum_mean, cum_max, cum_min, cum_std, cum_norm_mean], dim=2)

        # ========== Stage 1b: compress (batch over N) ==========
        # Reshape to [bsz*N, 5, H] → process as batch
        cum_stats_flat = cum_stats.reshape(bsz * num_chunks, 5, hidden_size)
        compressed_list = [
            self.stat_compressors[i](cum_stats_flat[:, i, :])
            for i in range(5)
        ]
        compressed_stats = torch.stack(compressed_list, dim=1)  # [bsz*N, 5, compress_dim]

        # ========== Stage 2: lightweight MHA (batch over N) ==========
        BN = bsz * num_chunks
        Q = self.global_queries.unsqueeze(0).expand(BN, -1, -1)
        Q = self.q_proj(Q)
        K = self.k_proj(compressed_stats)
        V = self.v_proj(compressed_stats)

        Q = Q.view(BN, self.global_slots, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(BN, 5, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(BN, 5, self.num_heads, self.head_dim).transpose(1, 2)

        scale = self.head_dim ** -0.5
        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) * scale
        attn_probs = F.softmax(attn_weights, dim=-1)
        attn_probs = self.attn_dropout(attn_probs)
        attn_output = torch.matmul(attn_probs, V)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(BN, self.global_slots, self.compress_dim)
        G_compressed = self.o_proj(attn_output)

        # ========== Stage 3: expand to hidden_size ==========
        G = self.expand(G_compressed) * self.expand_scale  # [bsz*N, global_slots, H]

        # Reshape back: [bsz, N, global_slots, H]
        G = G.view(bsz, num_chunks, self.global_slots, hidden_size)
        return G


# ============================================================================
# GlobalIntegratorShared — shared compression layer variant
# ============================================================================
class GlobalIntegratorShared(nn.Module):
    """
    GlobalIntegratorShared - shared compression layer variant

    Key optimizations:
    1. 92% parameter reduction: stat compression from 10.5M to 0.85M
    2. Shared compression backbone: all 5 statistics share one 4096→128 layer
    3. Statistic fusion: 5×128→512 fusion layer integrates all statistics
    4. Retains two-stage design: compression + Lightweight Attention
    5. Stronger inductive bias: shared params force universal feature extractor

    Rationale:
    - Parameter Sharing: analogous to CNN weight sharing
    - Inductive Bias: forces same feature extractor for all 5 statistics
    - Information Bottleneck: small intermediate dim (128) controls capacity

    Parameter count comparison (hidden_size=4096, compress_dim=512):
        Original stat compression: 5 × (4096 × 512) = 10.5M

        Optimized:
        - Shared compressor:   4096 × 128 = 0.524M
        - Stat fusion:         5×128 × 512 = 0.328M
        - Total:               0.852M (saves 92%)

        Other layers unchanged:
        - Q/K/V projection:    0.8M
        - O projection:        0.26M
        - Expansion layer:     2.1M

        Total: 0.852M + 0.8M + 0.26M + 2.1M = 4.0M/layer (vs. original 13.7M)
        Saving: 71%

    Input/Output:
        Input:  local_repr [bsz, num_chunks, local_slots, hidden_size]
        Output: global_context  [bsz, global_slots, hidden_size]
    """

    _init_msg_printed = False

    def __init__(
        self,
        hidden_size: int = 4096,
        global_slots: int = 4,
        compress_dim: int = 512,
        shared_compress_dim: int = 128,  # shared compressor dimension
        num_heads: int = 8,
        dropout: float = 0.0,
        local_slots: int = 16,  # compatibility parameter
        use_bottleneck: bool = False,  # compatibility parameter
        bottleneck_dim: int = 4096,  # compatibility parameter
        init_from_embeddings: Optional[torch.Tensor] = None,
        use_high_norm_init: bool = True,
        output_scale_init: float = 0.1,
    ):
        """
        Args:
            hidden_size: hidden dimension (typically 4096)
            global_slots: number of global context slots (typically 4-16)
            compress_dim: final compression dimension (typically 512)
            shared_compress_dim: intermediate dim of shared compressor (typically 128)
            num_heads: number of attention heads
            dropout: attention dropout probability
            init_from_embeddings: pretrained embeddings for initialization
            use_high_norm_init: whether to use high-norm token selection for initialization
            output_scale_init: initial value for output scale
        """
        super().__init__()

        # ============ Parameter validation ============
        assert compress_dim % num_heads == 0, (
            f"compress_dim ({compress_dim}) must be divisible by num_heads ({num_heads})"
        )
        assert output_scale_init > 0, "output_scale_init must be positive"

        # ============ Save configuration ============
        self.hidden_size = hidden_size
        self.num_global = global_slots  # compatibility alias
        self.global_slots = global_slots
        self.shared_compress_dim = shared_compress_dim
        self.num_heads = num_heads
        self.dropout_p = dropout
        self.use_high_norm_init = use_high_norm_init
        self._output_scale_init = output_scale_init

        # ============ Stage 1: shared compression layer ============
        self.stat_names = ["mean", "max", "min", "std", "norm_mean"]

        # Key optimization: all statistics share one compression layer
        self.shared_compressor = nn.Sequential(
            nn.Linear(hidden_size, shared_compress_dim, bias=False),
            nn.LayerNorm(shared_compress_dim),
        )
        # Params: 4096 × shared_compress_dim

        # Only create expansion layer when dimensions need to grow
        # If shared_compress_dim == compress_dim, no expansion needed
        if shared_compress_dim < compress_dim:
            self.stat_expand = nn.Sequential(
                nn.Linear(shared_compress_dim, compress_dim, bias=False),
                nn.LayerNorm(compress_dim),
            )
            # Params: shared_compress_dim × compress_dim
            self.compress_dim = compress_dim
        else:
            # shared_compress_dim >= compress_dim: no expansion needed
            # Use Identity (0 params)
            self.stat_expand = nn.Identity()
            if shared_compress_dim > compress_dim:
                print(
                    f"Warning: shared_compress_dim ({shared_compress_dim}) > compress_dim ({compress_dim})"
                )
                print(f"   Setting compress_dim = shared_compress_dim for consistency")
            self.compress_dim = shared_compress_dim

        self.head_dim = self.compress_dim // num_heads
        # Total: 524K + 66K = 590K (5.6% of original 10.5M)

        # ============ Stage 2: lightweight multi-head attention ============
        self.global_queries = nn.Parameter(torch.zeros(global_slots, self.compress_dim))

        self.q_proj = nn.Linear(self.compress_dim, self.compress_dim, bias=False)
        self.k_proj = nn.Linear(self.compress_dim, self.compress_dim, bias=False)
        self.v_proj = nn.Linear(self.compress_dim, self.compress_dim, bias=False)
        self.o_proj = nn.Linear(self.compress_dim, self.compress_dim, bias=False)

        self.attn_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # ============ Stage 3: dimension expansion ============
        self.expand = nn.Linear(self.compress_dim, hidden_size, bias=False)
        std_init = 0.02 / math.sqrt(self.compress_dim)
        nn.init.normal_(self.expand.weight, mean=0.0, std=std_init)

        # Output scale parameter
        init_param = math.log(math.exp(output_scale_init) - 1)
        self.expand_scale_param = nn.Parameter(torch.tensor(init_param))

        # ============ Initialize ============
        self._init_weights(init_from_embeddings)
        self._print_init_info()

    @property
    def expand_scale(self) -> torch.Tensor:
        """Scale via softplus to ensure it remains positive."""
        return F.softplus(self.expand_scale_param)

    def _init_weights(self, embed_weight: Optional[torch.Tensor] = None):
        """Initialize weights."""
        if embed_weight is not None:
            with torch.no_grad():
                if self.use_high_norm_init:
                    embed_norms = torch.norm(embed_weight, dim=-1)
                    _, top_indices = torch.topk(embed_norms, k=self.global_slots)
                    init_embeddings = embed_weight[top_indices]
                else:
                    indices = torch.randperm(embed_weight.size(0))[: self.global_slots]
                    init_embeddings = embed_weight[indices]

                # Ensure device and dtype match
                target_device = self.shared_compressor[0].weight.device
                target_dtype = self.shared_compressor[0].weight.dtype
                init_embeddings = init_embeddings.to(
                    device=target_device, dtype=target_dtype
                )

                # Initialize via shared compressor + expansion layer
                init_compressed = self.shared_compressor(
                    init_embeddings
                )  # [global_slots, 128]
                init_expanded = self.stat_expand(init_compressed)  # [global_slots, 512]
                self.global_queries.copy_(init_expanded)
        else:
            nn.init.xavier_uniform_(self.global_queries)

        # Initialize projection layers
        for proj in [self.q_proj, self.k_proj, self.v_proj, self.o_proj]:
            nn.init.xavier_uniform_(proj.weight)
        # for proj in [self.q_proj, self.k_proj, self.v_proj]:
        #     nn.init.xavier_uniform_(proj.weight)

    def _print_init_info(self):
        """Print initialization info."""
        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank == 0 and not GlobalIntegratorShared._init_msg_printed:
            total_params = sum(p.numel() for p in self.parameters())

            # Count stat compression params
            stat_compress_params = sum(
                p.numel() for p in self.shared_compressor.parameters()
            ) + sum(p.numel() for p in self.stat_expand.parameters())

            print(f"   GlobalIntegratorShared initialized (shared compressor)")

            # Show different design descriptions depending on whether expansion is used
            if isinstance(self.stat_expand, nn.Identity):
                design_desc = "Shared Compressor + Lightweight MHA (no expansion)"
            else:
                design_desc = (
                    "Shared Compressor + Statistical Expansion + Lightweight MHA"
                )

            print(f"       - Design: {design_desc}")
            print(f"       - Global slots: {self.global_slots}")
            print(f"       - Shared compress dim: {self.shared_compress_dim}")
            print(f"       - Final compress dim: {self.compress_dim}")
            print(f"       - Num heads: {self.num_heads}")
            print(f"       - Output scale (init): {self._output_scale_init}")
            print(
                f"       - Stat compression params: {stat_compress_params:,} ({stat_compress_params / 1e6:.2f}M)"
            )
            print(
                f"       - Total params/layer: {total_params:,} ({total_params / 1e6:.1f}M)"
            )
            print(
                f"       - 🎯 Saved {(1 - total_params / 13.7e6) * 100:.0f}% compared to original"
            )
            GlobalIntegratorShared._init_msg_printed = True

    def forward(self, local_repr: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            local_repr: [bsz, num_chunks, local_slots, hidden_size]

        Returns:
            G: [bsz, global_slots, hidden_size]

        Data flow:
            local_repr [bsz, C, L, H]
                ↓ reshape
            all_local [bsz, C*L, H]
                ↓ 5 statistics
            stats [bsz, H] × 5
                ↓ shared compress (each stat passes independently)
            compressed_stats_list: 5 × [bsz, 128]
                ↓ expand (each stat passes independently)
            expanded_stats_list: 5 × [bsz, 512]
                ↓ stack (kept separate)
            compressed_stats [bsz, 5, 512]
                ↓ Multi-Head Attention (selects over 5 statistics)
            G_compressed [bsz, G, D]
                ↓ expand + scale
            G [bsz, G, H]
        """
        bsz, num_chunks, local_slots, hidden_size = local_repr.shape

        # ========== Stage 1a: extract statistics ==========
        all_local = local_repr.reshape(bsz, -1, hidden_size)

        # Compute 5 statistics
        mean_pool = all_local.mean(dim=1)
        max_pool, _ = all_local.max(dim=1)
        min_pool, _ = all_local.min(dim=1)

        # Compute std in fp32 for numerical stability
        with torch.amp.autocast(device_type="cuda", enabled=False):
            all_local_fp32 = all_local.float()
            std_pool = all_local_fp32.std(dim=1, unbiased=False).clamp(min=1e-6)
        std_pool = std_pool.to(all_local.dtype)

        # L2-normalized mean
        norm_mean = F.normalize(mean_pool, dim=-1, p=2, eps=1e-6)

        # ========== Stage 1b: shared compress + expand (keep 5 stats separate) ==========
        # Batch optimization: process all 5 stats at once
        # Key: keep stats separate so Attention can learn selective usage
        stats_list = [mean_pool, max_pool, min_pool, std_pool, norm_mean]

        # Stack: [bsz, 5, hidden_size]
        stats_stacked = torch.stack(stats_list, dim=1)
        num_stats = 5

        # Batch compress: view to [bsz*5, hidden_size] → compress → view to [bsz, 5, 128]
        compressed_stats = self.shared_compressor(
            stats_stacked.view(bsz * num_stats, hidden_size)
        ).view(bsz, num_stats, -1)

        # Batch expand: view to [bsz*5, 128] → expand → view to [bsz, 5, 512]
        compressed_stats = self.stat_expand(
            compressed_stats.view(bsz * num_stats, -1)
        ).view(bsz, num_stats, self.compress_dim)
        # compressed_stats: [bsz, 5, 512] (same shape as original)

        # ========== Stage 2: lightweight multi-head attention ==========
        # Q: [bsz, global_slots, compress_dim]
        Q = self.global_queries.unsqueeze(0).expand(bsz, -1, -1)
        Q = self.q_proj(Q)

        # K, V: [bsz, 5, compress_dim]
        K = self.k_proj(compressed_stats)
        V = self.v_proj(compressed_stats)

        # Split heads: [bsz, num_heads, seq_len, head_dim]
        Q = Q.view(bsz, self.global_slots, self.num_heads, self.head_dim).transpose(
            1, 2
        )
        K = K.view(bsz, 5, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(bsz, 5, self.num_heads, self.head_dim).transpose(1, 2)
        # Q: [bsz, num_heads, global_slots, head_dim]
        # K: [bsz, num_heads, 5, head_dim]
        # V: [bsz, num_heads, 5, head_dim]

        # Scaled Dot-Product Attention
        scale = self.head_dim**-0.5
        # attn_weights: [bsz, num_heads, global_slots, 5]
        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) * scale
        attn_probs = F.softmax(attn_weights, dim=-1)
        attn_probs = self.attn_dropout(attn_probs)

        # attn_output: [bsz, num_heads, global_slots, head_dim]
        attn_output = torch.matmul(attn_probs, V)

        # Merge heads
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, self.global_slots, self.compress_dim)

        # Output projection
        G_compressed = self.o_proj(attn_output)

        # ========== Stage 3: expand to hidden_size ==========
        # G = self.expand(attn_output) * self.expand_scale
        G = self.expand(G_compressed) * self.expand_scale

        return G

    def forward_causal(self, local_repr: torch.Tensor) -> torch.Tensor:
        """
        Causal forward pass: compute an independent G_i for each segment (shared compressor).

        For segment i, G_i is computed only from L_1, ..., L_i cumulative statistics.

        Args:
            local_repr: [bsz, num_chunks, local_slots, hidden_size]

        Returns:
            G_all: [bsz, num_chunks, global_slots, hidden_size]
        """
        bsz, num_chunks, local_slots, hidden_size = local_repr.shape

        # ========== Stage 1: cumulative statistics ==========
        sum_per_chunk = local_repr.sum(dim=2)           # [bsz, N, H]
        max_per_chunk = local_repr.max(dim=2).values    # [bsz, N, H]
        min_per_chunk = local_repr.min(dim=2).values    # [bsz, N, H]

        cumsum = sum_per_chunk.cumsum(dim=1)                # [bsz, N, H]
        counts = torch.arange(1, num_chunks + 1, device=local_repr.device,
                              dtype=local_repr.dtype).view(1, -1, 1) * local_slots
        cum_mean = cumsum / counts                          # [bsz, N, H]

        cum_max = max_per_chunk.cummax(dim=1).values        # [bsz, N, H]
        cum_min = min_per_chunk.cummin(dim=1).values        # [bsz, N, H]

        with torch.amp.autocast(device_type="cuda", enabled=False):
            local_fp32 = local_repr.float()
            sq_sum_per_chunk = (local_fp32 ** 2).sum(dim=2)
            cum_sq_sum = sq_sum_per_chunk.cumsum(dim=1)
            counts_f = counts.float()
            cum_sq_mean = cum_sq_sum / counts_f
            cum_mean_f = cumsum.float() / counts_f
            cum_var = (cum_sq_mean - cum_mean_f ** 2).clamp(min=1e-12)
            cum_std = cum_var.sqrt()
        cum_std = cum_std.to(local_repr.dtype)

        cum_norm_mean = F.normalize(cum_mean, dim=-1, p=2, eps=1e-6)

        # Stack: [bsz, N, 5, H]
        cum_stats = torch.stack([cum_mean, cum_max, cum_min, cum_std, cum_norm_mean], dim=2)

        # ========== Stage 1b: shared compress + expand (batch over N) ==========
        BN = bsz * num_chunks
        # [bsz*N, 5, H]
        cum_stats_flat = cum_stats.reshape(BN * 5, hidden_size)
        compressed_stats = self.shared_compressor(cum_stats_flat).view(BN, 5, -1)
        compressed_stats = self.stat_expand(
            compressed_stats.view(BN * 5, -1)
        ).view(BN, 5, self.compress_dim)

        # ========== Stage 2: lightweight MHA (batch over N) ==========
        Q = self.global_queries.unsqueeze(0).expand(BN, -1, -1)
        Q = self.q_proj(Q)
        K = self.k_proj(compressed_stats)
        V = self.v_proj(compressed_stats)

        Q = Q.view(BN, self.global_slots, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(BN, 5, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(BN, 5, self.num_heads, self.head_dim).transpose(1, 2)

        scale = self.head_dim ** -0.5
        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) * scale
        attn_probs = F.softmax(attn_weights, dim=-1)
        attn_probs = self.attn_dropout(attn_probs)
        attn_output = torch.matmul(attn_probs, V)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(BN, self.global_slots, self.compress_dim)
        G_compressed = self.o_proj(attn_output)

        # ========== Stage 3: expand to hidden_size ==========
        G = self.expand(G_compressed) * self.expand_scale  # [BN, global_slots, H]
        G = G.view(bsz, num_chunks, self.global_slots, hidden_size)
        return G


# Training variant: HiCI with Recurrence Cache (integrated)
def forward_flashattn_hierarchical_with_cache(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    padding_mask: Optional[torch.LongTensor] = None,
    # Parameters controlled directly by this function
    use_global_context: bool = True,
    use_local_repr: bool = True,
    use_recurrence_cache: bool = False,  # whether to use recurrence cache (Transformer-XL style)
    recurrence_size: Optional[int] = 128,  # recurrence cache size
    # group_size_ratio: Optional[float] = 0.25,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """
    HiCI hierarchical attention with cache support (optimized).

    Integrates:
    1. Recurrence cache (Transformer-XL style)
    2. HiCI modules (LocalConstructor + GlobalIntegrator)
    3. Ablation modes (use_global_context, use_local_repr)
    4. All logic controlled by parameters

    Key optimization: Q is chunk-only, no HiCI slots prepended (saves compute)
    - K/V: [global_context?, local?, cache?, chunk]
    - Output is chunk tokens directly, no extraction needed

    CRITICAL: cache must immediately precede chunk (positionally contiguous)

    Concatenation order:
    - Position-independent components (global_context, local) go first
    - Cache must be adjacent to chunk (positional continuity)

    Three ablation modes:
    - Mode 1 (recommended): use_global_context=True, use_local_repr=False
      Q:   [chunk]
      K/V: [global_context, cache, chunk]

    - Mode 2: use_global_context=False, use_local_repr=True
      Q:   [chunk]
      K/V: [local_i, cache, chunk]

    - Mode 3: use_global_context=True, use_local_repr=True
      Q:   [chunk]
      K/V: [global_context, local_i, cache, chunk]

    Args:
        use_global_context: whether to use GlobalIntegrator (aggregates all local slots)
        use_local_repr: whether to prepend LocalConstructor slots to each chunk
        use_recurrence_cache: whether to use Transformer-XL style recurrence cache
    """
    if not self.training:
        warnings.warn(
            "This function should be used just for training. For inference, use forward_flashattn_inference."
        )

    if output_attentions:
        warnings.warn(
            "Output attentions is not supported for patched `LlamaAttention`, returning `None` instead."
        )

    bsz, q_len, hidden_size = hidden_states.size()

    # Print ablation config once (rank 0, layer 0 only)
    if not hasattr(self, "_ablation_config_printed"):
        rank = dist.get_rank() if dist.is_initialized() else 0
        layer_idx = getattr(self, "layer_idx", 0)  # get current layer index

        if rank == 0 and layer_idx == 0:  # print only on main process, first layer
            print("\n" + "=" * 80)
            print("📋 HiCI Ablation Configuration")
            print("=" * 80)
            print(f"  ✅ use_global_context    : {use_global_context}  (GlobalIntegrator)")
            print(
                f"  {'✅' if use_local_repr else '❌'} use_local_repr    : {use_local_repr}  (LocalConstructor slots)"
            )
            print(
                f"  ✅ use_recurrence_cache : {use_recurrence_cache}  (Recurrence cache)"
            )
            print()

            # Display current ablation mode (optimized: Q contains chunk only)
            if use_global_context and not use_local_repr and use_recurrence_cache:
                print("Current Mode: Mode 1 (recommended)")
                print("   Q:   [chunk]")
                print("   K/V: [global_context, cache, chunk]")
                print("   Advantage: no redundancy, highly aggregated, shorter Q")
            elif not use_global_context and use_local_repr and use_recurrence_cache:
                print("Current Mode: Mode 2")
                print("   Q:   [chunk]")
                print("   K/V: [local_i, cache, chunk]")
                print("   Advantage: each chunk has its own compressed representation")
            elif use_global_context and use_local_repr and use_recurrence_cache:
                print("Current Mode: Mode 3 (full)")
                print("   Q:   [chunk]")
                print("   K/V: [global_context, local_i, cache, chunk]")
                print("   Advantage: all features included")
            else:
                print("Current Mode: Custom")
                print(
                    f"   Config: global_context={use_global_context}, local={use_local_repr}, cache={use_recurrence_cache}"
                )
                print("   Q:   [chunk]")
                print("   K/V: [hici_slots?, cache?, chunk]")

            print("=" * 80 + "\n", flush=True)

        self._ablation_config_printed = True

    # ========== Step 1: Split into chunks ==========
    group_size = int(q_len * group_size_ratio)
    if q_len % group_size > 0:
        raise ValueError(
            f"q_len {q_len} should be divisible by group size {group_size}."
        )
    if not hasattr(self, "_group_size_printed"):
        layer_idx = getattr(self, "layer_idx", 0)
        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank == 0 and layer_idx == 0:
            print(
                f"[forward_flashattn_hierarchical_with_cache] group_size_ratio={group_size_ratio}, group_size={group_size}"
            )
        self._group_size_printed = True

    num_groups = q_len // group_size

    # Reshape into chunks: [bsz, num_groups, group_size, hidden_size]
    chunks = hidden_states.view(bsz, num_groups, group_size, hidden_size)

    # ========== Step 2: Extract local memories (compress each chunk) ==========
    # ⚠️ CRITICAL: Check if global_context exists before using it!
    if (use_global_context or use_local_repr) and hasattr(self, "local_constructor"):
        # Process all chunks in parallel
        all_chunks = chunks.view(bsz * num_groups, group_size, hidden_size)

        # attention_mask: [bsz, q_len] -> [bsz * num_groups, group_size]
        if attention_mask is not None:
            attention_mask_chunks = attention_mask.view(bsz, num_groups, group_size)
            attention_mask_chunks = attention_mask_chunks.view(
                bsz * num_groups, group_size
            )
        else:
            attention_mask_chunks = None

        # all_local_repr = self.local_constructor(
        #     all_chunks, attention_mask_chunks
        # )  # [bsz * num_groups, num_slots, hidden_size]
        all_local_repr = self.local_constructor(all_chunks)  # zxy

        # Reshape back: [bsz, num_groups, num_slots, hidden_size]
        num_local_slots = all_local_repr.shape[1]
        local_repr_stacked = all_local_repr.view(
            bsz, num_groups, num_local_slots, hidden_size
        )
    else:
        num_local_slots = 0
        local_repr_stacked = None

    # ========== Step 3: Global Integration — aggregate local repr {Li} into global context G ==========
    # ⚠️ CRITICAL: Requires both local_constructor (Local Construction) AND global_integrator (Global Integration)!
    _causal_mode = CAUSAL_CONTEXT_MODE
    _is_causal = _causal_mode in ("causal_gi", "causal_shift", "causal_shift_g", "causal_gi_gonly")

    if (
        use_global_context
        and hasattr(self, "global_integrator")
        and local_repr_stacked is not None
    ):
        if _is_causal and hasattr(self.global_integrator, "forward_causal"):
            global_context_per_group = self.global_integrator.forward_causal(
                local_repr_stacked
            )
            num_global_slots = global_context_per_group.shape[2]

            if _causal_mode in ("causal_shift", "causal_shift_g"):
                zeros_g = torch.zeros(
                    bsz, 1, num_global_slots, hidden_size,
                    device=global_context_per_group.device,
                    dtype=global_context_per_group.dtype,
                )
                global_context_per_group = torch.cat(
                    [zeros_g, global_context_per_group[:, :-1, :, :]], dim=1
                )

            global_context = None
        else:
            global_context = self.global_integrator(local_repr_stacked)
            num_global_slots = global_context.shape[1]
            global_context_per_group = None
    else:
        global_context = None
        global_context_per_group = None
        num_global_slots = 0

    # causal_shift_g / causal_gi_gonly: use G only, skip L
    if _causal_mode in ("causal_shift_g", "causal_gi_gonly"):
        local_repr_stacked = None
        num_local_slots = 0

    if (
        _causal_mode == "causal_shift"
        and use_local_repr
        and local_repr_stacked is not None
    ):
        zeros_l = torch.zeros(
            bsz, 1, num_local_slots, hidden_size,
            device=local_repr_stacked.device,
            dtype=local_repr_stacked.dtype,
        )
        local_repr_stacked = torch.cat(
            [zeros_l, local_repr_stacked[:, :-1, :, :]], dim=1
        )

    # ========== Step 4: Q/K/V projections (fused K/V for global_context + hidden_states) ==========
    # Q: project hidden_states only
    query_states = (
        self.q_proj(hidden_states)
        .view(bsz, q_len, self.num_heads, self.head_dim)
        .transpose(1, 2)
    )  # [bsz, nh, q_len, hd]

    # K/V: fused projection to save memory
    global_context_k_per_group = global_context_v_per_group = None

    if use_global_context and global_context is not None:
        # Non-causal: concatenate then project [global_context, hidden_states]
        combined_input = torch.cat(
            [global_context, hidden_states], dim=1
        )  # [bsz, num_global_slots + q_len, hidden_size]

        combined_k = (
            self.k_proj(combined_input)
            .view(
                bsz,
                num_global_slots + q_len,
                self.num_key_value_heads,
                self.head_dim,
            )
            .transpose(1, 2)
        )

        combined_v = (
            self.v_proj(combined_input)
            .view(
                bsz,
                num_global_slots + q_len,
                self.num_key_value_heads,
                self.head_dim,
            )
            .transpose(1, 2)
        )

        global_context_k = combined_k[:, :, :num_global_slots, :]
        key_states = combined_k[:, :, num_global_slots:, :]
        global_context_v = combined_v[:, :, :num_global_slots, :]
        value_states = combined_v[:, :, num_global_slots:, :]
    elif use_global_context and global_context_per_group is not None:
        # Causal mode: project per-group G separately
        key_states = (
            self.k_proj(hidden_states)
            .view(bsz, q_len, self.num_key_value_heads, self.head_dim)
            .transpose(1, 2)
        )
        value_states = (
            self.v_proj(hidden_states)
            .view(bsz, q_len, self.num_key_value_heads, self.head_dim)
            .transpose(1, 2)
        )
        global_context_k = global_context_v = None

        gc_flat = global_context_per_group.view(
            bsz * num_groups, num_global_slots, hidden_size
        )
        gc_k_flat = (
            self.k_proj(gc_flat)
            .view(bsz * num_groups, num_global_slots, self.num_key_value_heads, self.head_dim)
            .transpose(1, 2)
        )
        gc_v_flat = (
            self.v_proj(gc_flat)
            .view(bsz * num_groups, num_global_slots, self.num_key_value_heads, self.head_dim)
            .transpose(1, 2)
        )
        # repeat_kv after RoPE section below
        global_context_k_per_group_raw = gc_k_flat
        global_context_v_per_group_raw = gc_v_flat
    else:
        key_states = (
            self.k_proj(hidden_states)
            .view(bsz, q_len, self.num_key_value_heads, self.head_dim)
            .transpose(1, 2)
        )
        value_states = (
            self.v_proj(hidden_states)
            .view(bsz, q_len, self.num_key_value_heads, self.head_dim)
            .transpose(1, 2)
        )
        global_context_k = global_context_v = None

    # Apply RoPE to sequence tokens only, not global context G or local repr Li
    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]

    if position_ids is not None:
        max_pos = position_ids.max().item() + 1
        rope_seq_len = max(kv_seq_len, max_pos)
    else:
        rope_seq_len = kv_seq_len

    cos, sin = self.rotary_emb(value_states, seq_len=rope_seq_len)
    query_states, key_states = apply_rotary_pos_emb(
        query_states, key_states, cos, sin, position_ids
    )

    # Past Key value support
    if past_key_value is not None:
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)

    past_key_value = (key_states, value_states) if use_cache else None

    # Repeat k/v heads if n_kv_heads < n_heads
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)
    if global_context_k is not None:
        global_context_k = repeat_kv(global_context_k, self.num_key_value_groups)
        global_context_v = repeat_kv(global_context_v, self.num_key_value_groups)

    # Causal mode: repeat_kv and reshape per-group G
    if global_context_k_per_group is None and global_context_per_group is not None:
        # global_context_k_per_group_raw was set in Step 4 above
        global_context_k_per_group_raw = repeat_kv(global_context_k_per_group_raw, self.num_key_value_groups)
        global_context_v_per_group_raw = repeat_kv(global_context_v_per_group_raw, self.num_key_value_groups)
        global_context_k_per_group = global_context_k_per_group_raw.view(
            bsz, num_groups, self.num_heads, num_global_slots, self.head_dim
        )
        global_context_v_per_group = global_context_v_per_group_raw.view(
            bsz, num_groups, self.num_heads, num_global_slots, self.head_dim
        )

    # ========== Step 6: Reshape into chunks ==========
    query_chunks = query_states.view(
        bsz, self.num_heads, num_groups, group_size, self.head_dim
    )
    key_chunks = key_states.view(
        bsz, self.num_heads, num_groups, group_size, self.head_dim
    )
    value_chunks = value_states.view(
        bsz, self.num_heads, num_groups, group_size, self.head_dim
    )

    # Local memories for each chunk (if needed)
    # Batch-project all local repr {Li} K/V; Q is derived from segment tokens only
    if use_local_repr and local_repr_stacked is not None:
        # Reshape: [bsz, num_groups, num_slots, hidden] -> [bsz*num_groups, num_slots, hidden]
        local_mems_flat = local_repr_stacked.view(
            bsz * num_groups, num_local_slots, hidden_size
        )

        # Project all local memories K/V in one batched call
        local_k_flat = (
            self.k_proj(local_mems_flat)
            .view(
                bsz * num_groups,
                num_local_slots,
                self.num_key_value_heads,
                self.head_dim,
            )
            .transpose(1, 2)
        )

        local_v_flat = (
            self.v_proj(local_mems_flat)
            .view(
                bsz * num_groups,
                num_local_slots,
                self.num_key_value_heads,
                self.head_dim,
            )
            .transpose(1, 2)
        )

        # Repeat k/v heads (batched)
        local_k_flat = repeat_kv(local_k_flat, self.num_key_value_groups)
        local_v_flat = repeat_kv(local_v_flat, self.num_key_value_groups)

        # Reshape back: [bsz, num_groups, nh, num_slots, hd]
        local_k_all = local_k_flat.view(
            bsz, num_groups, self.num_heads, num_local_slots, self.head_dim
        )
        local_v_all = local_v_flat.view(
            bsz, num_groups, self.num_heads, num_local_slots, self.head_dim
        )

    # ========== Step 7: Build Q and K/V using vectorized ops ==========
    # Q: chunk tokens only
    # K/V: [global_context?, local?, cache?, chunk]

    kv_components_k = []
    kv_components_v = []

    # 7.1 Global context G (shared across all chunks)
    if use_global_context and global_context_k is not None:
        # Non-causal: all chunks share the same G
        global_context_k_exp = global_context_k.unsqueeze(2).expand(
            -1, -1, num_groups, -1, -1
        )
        global_context_v_exp = global_context_v.unsqueeze(2).expand(
            -1, -1, num_groups, -1, -1
        )
        kv_components_k.append(global_context_k_exp)
        kv_components_v.append(global_context_v_exp)
    elif use_global_context and global_context_k_per_group is not None:
        # Causal: each chunk has its own G_i
        kv_components_k.append(global_context_k_per_group.permute(0, 2, 1, 3, 4))
        kv_components_v.append(global_context_v_per_group.permute(0, 2, 1, 3, 4))

    # 7.2 Local memories (different per chunk, permute directly)
    if use_local_repr and local_repr_stacked is not None:
        # local_k_all: [bsz, num_groups, nh, num_local_slots, hd]
        # permute to [bsz, nh, num_groups, num_local_slots, hd]
        local_k_exp = local_k_all.permute(0, 2, 1, 3, 4)
        local_v_exp = local_v_all.permute(0, 2, 1, 3, 4)
        kv_components_k.append(local_k_exp)
        kv_components_v.append(local_v_exp)

    # 7.3 Recurrence cache (vectorized)
    if use_recurrence_cache:
        # key_chunks: [bsz, nh, num_groups, group_size, hd]
        # Take the tail of each chunk as the cache for the next chunk
        chunk_tails_k = key_chunks[
            :, :, :, -recurrence_size:, :
        ]  # [bsz, nh, num_groups, recurrence_size, hd]
        chunk_tails_v = value_chunks[:, :, :, -recurrence_size:, :]

        # Build cache: zeros for chunk_0, tail of chunk_{i-1} for chunk_i
        # [zeros, chunk_0_tail, chunk_1_tail, ..., chunk_{n-2}_tail]
        dummy = torch.zeros(
            bsz,
            self.num_heads,
            1,  # placeholder for chunk_0
            recurrence_size,
            self.head_dim,
            device=key_states.device,
            dtype=key_states.dtype,
        )
        # Tails of chunk_0..n-2 serve as cache for chunk_1..n-1
        cache_k = torch.cat(
            [dummy, chunk_tails_k[:, :, :-1, :, :]], dim=2
        )  # [bsz, nh, num_groups, recurrence_size, hd]
        cache_v = torch.cat([dummy, chunk_tails_v[:, :, :-1, :, :]], dim=2)
        kv_components_k.append(cache_k)
        kv_components_v.append(cache_v)

    # 7.4 Chunk tokens (must be last)
    kv_components_k.append(key_chunks)
    kv_components_v.append(value_chunks)

    # Concatenate K/V: [bsz, nh, num_groups, total_kv_len, hd]
    key_with_ctx = torch.cat(kv_components_k, dim=3)
    value_with_ctx = torch.cat(kv_components_v, dim=3)

    # Q: use query_chunks directly [bsz, nh, num_groups, group_size, hd]
    q_len_per_chunk = group_size  # Q contains chunk tokens only
    kv_len_per_chunk = key_with_ctx.shape[3]  # hici_slots + cache + chunk

    # ========== Step 8: Flash Attention with KV packed (following original implementation) ==========
    # CRITICAL: Use flash_attn_varlen_kvpacked_func to support Q and KV with different lengths
    # Reshape for flash attention
    # Q: [bsz, nh, num_groups, group_size, hd] -> [bsz * num_groups, group_size, nh, hd]
    all_chunks_q_flat = query_chunks.permute(0, 2, 3, 1, 4).reshape(
        bsz * num_groups, q_len_per_chunk, self.num_heads, self.head_dim
    )

    # K/V: [bsz, nh, num_groups, kv_len, hd] -> [bsz * num_groups, kv_len, nh, hd]
    key_flat = key_with_ctx.permute(0, 2, 3, 1, 4).reshape(
        bsz * num_groups, kv_len_per_chunk, self.num_heads, self.head_dim
    )
    value_flat = value_with_ctx.permute(0, 2, 3, 1, 4).reshape(
        bsz * num_groups, kv_len_per_chunk, self.num_heads, self.head_dim
    )

    # Pack K/V: [bsz * num_groups, kv_len, 2, nh, hd]
    all_chunks_kv_flat = torch.stack([key_flat, value_flat], dim=2)

    # ========== Step 9: Prepare padding masks (1=real token, 0=padding) =======
    # Reshape chunk masks: [bsz, num_groups, group_size]
    chunk_masks_reshaped = attention_mask.view(bsz, num_groups, group_size)

    # 9.1 Q padding masks
    # CRITICAL: do not transpose — data is batch-first
    all_masks_q_flat = chunk_masks_reshaped.reshape(bsz * num_groups, q_len_per_chunk)

    # 9.2 K/V padding masks (filled in-place)
    all_masks_kv_stacked = torch.empty(
        bsz,
        num_groups,
        kv_len_per_chunk,
        dtype=attention_mask.dtype,
        device=attention_mask.device,
    )

    # In-place fill
    offset = 0
    if use_global_context:
        all_masks_kv_stacked[:, :, offset : offset + num_global_slots] = 1
        offset += num_global_slots
    if use_local_repr:
        all_masks_kv_stacked[:, :, offset : offset + num_local_slots] = 1
        offset += num_local_slots

    # causal_shift/causal_shift_g: segment_0 G (and L) are zero-padded, mask out
    if _causal_mode in ("causal_shift", "causal_shift_g"):
        mem_offset = 0
        if use_global_context:
            all_masks_kv_stacked[:, 0, mem_offset : mem_offset + num_global_slots] = 0
            mem_offset += num_global_slots
        if use_local_repr:
            all_masks_kv_stacked[:, 0, mem_offset : mem_offset + num_local_slots] = 0

    # Cache masks: derived from the last recurrence_size tokens of the previous chunk
    if use_recurrence_cache:
        # Cache masks: [bsz, num_groups, recurrence_size]
        # chunk_0: dummy (all zeros); chunk_i: last recurrence_size tokens of chunk_{i-1}
        cache_masks = torch.zeros(
            bsz,
            num_groups,
            recurrence_size,
            dtype=all_masks_kv_stacked.dtype,
            device=all_masks_kv_stacked.device,
        )

        if num_groups > 1:
            # Vectorized: last recurrence_size token masks from chunks 0..n-2
            prev_chunk_tails = chunk_masks_reshaped[
                :, :-1, -recurrence_size:
            ]  # [bsz, num_groups-1, recurrence_size]
            cache_masks[:, 1:, :] = prev_chunk_tails  # fill for chunk_1, chunk_2, ...

        all_masks_kv_stacked[:, :, offset : offset + recurrence_size] = cache_masks
        offset += recurrence_size

    # Chunk masks
    all_masks_kv_stacked[:, :, offset : offset + group_size] = chunk_masks_reshaped

    # CRITICAL: do not transpose — data is batch-first
    all_masks_kv_flat = all_masks_kv_stacked.reshape(bsz * num_groups, kv_len_per_chunk)

    # Unpad Q
    q_unpad, indices_q, cu_seqlens_q, max_seqlen_q = unpad_input(
        rearrange(all_chunks_q_flat, "b s h d -> b s (h d)"), all_masks_q_flat
    )
    q_unpad = rearrange(q_unpad, "nnz (h d) -> nnz h d", h=self.num_heads)

    # Unpad KV (already packed)
    kv_flat_2d = rearrange(all_chunks_kv_flat, "b s two h d -> b s (two h d)")
    kv_unpad, indices_kv, cu_seqlens_kv, max_seqlen_kv = unpad_input(
        kv_flat_2d, all_masks_kv_flat
    )
    kv_unpad = rearrange(
        kv_unpad, "nnz (two h d) -> nnz two h d", two=2, h=self.num_heads
    )

    # Flash Attention with KV packed (supports different Q and KV lengths!)
    # Each chunk is an independent causal sequence
    output_unpad = flash_attn_varlen_kvpacked_func(
        q_unpad,  # [total_q_tokens, num_heads, head_dim]
        kv_unpad,  # [total_kv_tokens, 2, num_heads, head_dim] - packed K/V
        cu_seqlens_q,  # Q sequence boundaries
        cu_seqlens_kv,  # KV sequence boundaries
        max_seqlen_q,  # Q max sequence length
        max_seqlen_kv,  # KV max sequence length
        dropout_p=0.0,
        softmax_scale=None,
        causal=True,
    )

    # Pad back to original shape (using Q's indices and length)
    output = rearrange(
        pad_input(
            rearrange(output_unpad, "nnz h d -> nnz (h d)"),
            indices_q,
            num_groups * bsz,
            q_len_per_chunk,
        ),
        "b s (h d) -> b s h d",
        h=self.num_heads,
    )  # [bsz*num_groups, q_len_per_chunk, nh, hd]

    # CRITICAL: view(bsz, num_groups, ...) not view(num_groups, bsz, ...) — data is batch-first
    output = output.view(bsz, num_groups, group_size, self.num_heads, self.head_dim)
    output = output.view(bsz, q_len, self.num_heads, self.head_dim)

    # Output projection
    attn_output = self.o_proj(rearrange(output, "b s h d -> b s (h d)"))

    return attn_output, None, past_key_value


# Training: Global Context + Recurrence Cache (simplified, no Global Integration)
# region ===========================================================================
def forward_flashattn_global_with_cache(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    padding_mask: Optional[torch.LongTensor] = None,
    # Parameters controlled directly in this function
    use_recurrence_cache: bool = True,  # whether to use Transformer-XL style recurrence cache
    recurrence_size: Optional[int] = 128,  # size of the recurrence cache
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """
    HiCI Global Context + Recurrence Cache (simplified).

    Simplified from forward_flashattn_hierarchical_with_cache:
    - Removes global_context (hierarchical aggregation)
    - Removes per-chunk Local Construction (no local repr Li)
    - Extracts a single global context from the full input (like forward_flashattn_hybrid)
    - Retains the recurrence cache mechanism for cross-chunk information flow

    Structure:
    - Q:   [chunk]  (no HiCI slots)
    - K/V: [global_context, cache?, chunk]

    Advantages:
    - Simpler than hierarchical version, fewer parameters
    - Global context extracted from full input, maximally informative
    - Cache mechanism supports cross-chunk information propagation

    Args:
        use_recurrence_cache: whether to use Transformer-XL style recurrence cache
        recurrence_size: size of the recurrence cache
    """
    if not self.training:
        warnings.warn(
            "This function should be used just for training. For inference, use forward_flashattn_inference."
        )

    if output_attentions:
        warnings.warn(
            "Output attentions is not supported for patched `LlamaAttention`, returning `None` instead."
        )

    bsz, q_len, hidden_size = hidden_states.size()

    if not hasattr(self, "_global_cache_config_printed"):
        rank = dist.get_rank() if dist.is_initialized() else 0
        layer_idx = getattr(self, "layer_idx", 0)

        if rank == 0 and layer_idx == 0:
            print("\n" + "=" * 80)
            print("📋 HiCI Global Context + Cache Configuration")
            print("=" * 80)
            print(
                f"  ✅ use_recurrence_cache : {use_recurrence_cache}  (Recurrence cache)"
            )
            print(f"  📊 recurrence_size      : {recurrence_size}")
            print()
            print("📌 Current Mode: HiCI Global Context + Cache")
            print("   Q:   [chunk]")
            print("   K/V: [global_context, cache?, chunk]")
            print("   Global context extracted from full input, shared by all chunks")
            print("=" * 80 + "\n", flush=True)

        self._global_cache_config_printed = True

    # ========== Step 1: Extract global context from the full input ==========
    has_hici = hasattr(self, "local_constructor")

    if has_hici:
        global_context = self.local_constructor(
            hidden_states, attention_mask
        )  # [bsz, num_local_slots, hidden_size]
        num_local_slots = global_context.shape[1]
    else:
        global_context = None
        num_local_slots = 0

    # ========== Step 2: Split into chunks ==========
    group_size = int(q_len * group_size_ratio)
    if q_len % group_size > 0:
        raise ValueError(
            f"q_len {q_len} should be divisible by group size {group_size}."
        )
    if not hasattr(self, "_group_size_printed_global_cache"):
        layer_idx = getattr(self, "layer_idx", 0)
        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank == 0 and layer_idx == 0:
            print(
                f"[forward_flashattn_global_with_cache] group_size_ratio={group_size_ratio}, group_size={group_size}"
            )
        self._group_size_printed_global_cache = True

    num_groups = q_len // group_size

    if use_recurrence_cache and recurrence_size > group_size:
        raise ValueError(
            f"recurrence_size ({recurrence_size}) should be <= group_size ({group_size})"
        )

    # ========== Step 3: Q/K/V projections (fused K/V) ==========
    # Q: project hidden_states only
    query_states = (
        self.q_proj(hidden_states)
        .view(bsz, q_len, self.num_heads, self.head_dim)
        .transpose(1, 2)
    )  # [bsz, nh, q_len, hd]

    # K/V: fused projection to save memory
    if has_hici and global_context is not None:
        # Concatenate then project: [global_context, hidden_states]
        combined_input = torch.cat(
            [global_context, hidden_states], dim=1
        )  # [bsz, num_local_slots + q_len, hidden_size]

        # K projection
        combined_k = (
            self.k_proj(combined_input)
            .view(
                bsz,
                num_local_slots + q_len,
                self.num_key_value_heads,
                self.head_dim,
            )
            .transpose(1, 2)
        )  # [bsz, nkv, num_local_slots + q_len, hd]

        # V projection
        combined_v = (
            self.v_proj(combined_input)
            .view(
                bsz,
                num_local_slots + q_len,
                self.num_key_value_heads,
                self.head_dim,
            )
            .transpose(1, 2)
        )

        # Separate global and sequence parts (zero-copy slice)
        global_k = combined_k[:, :, :num_local_slots, :]
        key_states = combined_k[:, :, num_local_slots:, :]
        global_v = combined_v[:, :, :num_local_slots, :]
        value_states = combined_v[:, :, num_local_slots:, :]
    else:
        # No global context G, project hidden_states directly
        key_states = (
            self.k_proj(hidden_states)
            .view(bsz, q_len, self.num_key_value_heads, self.head_dim)
            .transpose(1, 2)
        )
        value_states = (
            self.v_proj(hidden_states)
            .view(bsz, q_len, self.num_key_value_heads, self.head_dim)
            .transpose(1, 2)
        )
        global_k = global_v = None

    # ========== Step 4: RoPE (sequence tokens only, not global context) ==========
    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]

    if position_ids is not None:
        max_pos = position_ids.max().item() + 1
        rope_seq_len = max(kv_seq_len, max_pos)
    else:
        rope_seq_len = kv_seq_len

    cos, sin = self.rotary_emb(value_states, seq_len=rope_seq_len)
    query_states, key_states = apply_rotary_pos_emb(
        query_states, key_states, cos, sin, position_ids
    )
    # Past Key value support
    if past_key_value is not None:
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)

    past_key_value = (key_states, value_states) if use_cache else None

    # Repeat k/v heads if n_kv_heads < n_heads
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)
    if global_k is not None:
        global_k = repeat_kv(global_k, self.num_key_value_groups)
        global_v = repeat_kv(global_v, self.num_key_value_groups)

    # ========== Step 5: Reshape into chunks ==========
    query_chunks = query_states.view(
        bsz, self.num_heads, num_groups, group_size, self.head_dim
    )
    key_chunks = key_states.view(
        bsz, self.num_heads, num_groups, group_size, self.head_dim
    )
    value_chunks = value_states.view(
        bsz, self.num_heads, num_groups, group_size, self.head_dim
    )

    # ========== Step 6: Build Q and K/V using vectorized ops ==========
    # Q: chunk tokens only
    # K/V: [global_context G?, cache?, chunk]

    kv_components_k = []
    kv_components_v = []

    # 6.1 Global context G (same for all chunks, broadcast via expand)
    if has_hici and global_k is not None:
        # global_k: [bsz, nh, num_local_slots, hd]
        # expand to [bsz, nh, num_groups, num_local_slots, hd]
        global_k_exp = global_k.unsqueeze(2).expand(-1, -1, num_groups, -1, -1)
        global_v_exp = global_v.unsqueeze(2).expand(-1, -1, num_groups, -1, -1)
        kv_components_k.append(global_k_exp)
        kv_components_v.append(global_v_exp)

    # 6.2 Recurrence cache (vectorized)
    if use_recurrence_cache:
        # Take the tail of each chunk as the cache for the next chunk
        chunk_tails_k = key_chunks[:, :, :, -recurrence_size:, :]
        chunk_tails_v = value_chunks[:, :, :, -recurrence_size:, :]

        # Build cache: zeros for chunk_0, tail of chunk_{i-1} for chunk_i
        dummy = torch.zeros(
            bsz,
            self.num_heads,
            1,  # placeholder for chunk_0
            recurrence_size,
            self.head_dim,
            device=key_states.device,
            dtype=key_states.dtype,
        )
        cache_k = torch.cat([dummy, chunk_tails_k[:, :, :-1, :, :]], dim=2)
        cache_v = torch.cat([dummy, chunk_tails_v[:, :, :-1, :, :]], dim=2)
        kv_components_k.append(cache_k)
        kv_components_v.append(cache_v)

    # 6.3 Chunk tokens (must be last)
    kv_components_k.append(key_chunks)
    kv_components_v.append(value_chunks)

    # Concatenate K/V: [bsz, nh, num_groups, total_kv_len, hd]
    key_with_ctx = torch.cat(kv_components_k, dim=3)
    value_with_ctx = torch.cat(kv_components_v, dim=3)

    q_len_per_chunk = group_size  # Q contains chunk tokens only
    kv_len_per_chunk = key_with_ctx.shape[3]  # hici_slots + cache + chunk

    # ========== Step 7: Flash Attention with KV packed ==========
    # Reshape for flash attention
    # Q: [bsz, nh, num_groups, group_size, hd] -> [bsz * num_groups, group_size, nh, hd]
    all_chunks_q_flat = query_chunks.permute(0, 2, 3, 1, 4).reshape(
        bsz * num_groups, q_len_per_chunk, self.num_heads, self.head_dim
    )

    # K/V: [bsz, nh, num_groups, kv_len, hd] -> [bsz * num_groups, kv_len, nh, hd]
    key_flat = key_with_ctx.permute(0, 2, 3, 1, 4).reshape(
        bsz * num_groups, kv_len_per_chunk, self.num_heads, self.head_dim
    )
    value_flat = value_with_ctx.permute(0, 2, 3, 1, 4).reshape(
        bsz * num_groups, kv_len_per_chunk, self.num_heads, self.head_dim
    )

    # Pack K/V: [bsz * num_groups, kv_len, 2, nh, hd]
    all_chunks_kv_flat = torch.stack([key_flat, value_flat], dim=2)

    # ========== Step 8: Prepare padding masks ==========
    # Reshape chunk masks: [bsz, num_groups, group_size]
    chunk_masks_reshaped = attention_mask.view(bsz, num_groups, group_size)

    # Q mask: chunk tokens only
    all_masks_q_flat = chunk_masks_reshaped.reshape(bsz * num_groups, q_len_per_chunk)

    # K/V mask: [global_context G?, cache?, chunk]
    all_masks_kv_stacked = torch.empty(
        bsz,
        num_groups,
        kv_len_per_chunk,
        dtype=attention_mask.dtype,
        device=attention_mask.device,
    )

    offset = 0
    # Global context G mask (always visible)
    if has_hici and global_k is not None:
        all_masks_kv_stacked[:, :, offset : offset + num_local_slots] = 1
        offset += num_local_slots

    # Cache mask
    if use_recurrence_cache:
        cache_masks = torch.zeros(
            bsz,
            num_groups,
            recurrence_size,
            dtype=all_masks_kv_stacked.dtype,
            device=all_masks_kv_stacked.device,
        )
        if num_groups > 1:
            prev_chunk_tails = chunk_masks_reshaped[:, :-1, -recurrence_size:]
            cache_masks[:, 1:, :] = prev_chunk_tails
        all_masks_kv_stacked[:, :, offset : offset + recurrence_size] = cache_masks
        offset += recurrence_size

    # Chunk mask
    all_masks_kv_stacked[:, :, offset : offset + group_size] = chunk_masks_reshaped

    all_masks_kv_flat = all_masks_kv_stacked.reshape(bsz * num_groups, kv_len_per_chunk)

    # Unpad Q
    q_unpad, indices_q, cu_seqlens_q, max_seqlen_q = unpad_input(
        rearrange(all_chunks_q_flat, "b s h d -> b s (h d)"), all_masks_q_flat
    )
    q_unpad = rearrange(q_unpad, "nnz (h d) -> nnz h d", h=self.num_heads)

    # Unpad KV
    kv_flat_2d = rearrange(all_chunks_kv_flat, "b s two h d -> b s (two h d)")
    kv_unpad, indices_kv, cu_seqlens_kv, max_seqlen_kv = unpad_input(
        kv_flat_2d, all_masks_kv_flat
    )
    kv_unpad = rearrange(
        kv_unpad, "nnz (two h d) -> nnz two h d", two=2, h=self.num_heads
    )

    # Flash Attention
    output_unpad = flash_attn_varlen_kvpacked_func(
        q_unpad,
        kv_unpad,
        cu_seqlens_q,
        cu_seqlens_kv,
        max_seqlen_q,
        max_seqlen_kv,
        dropout_p=0.0,
        softmax_scale=None,
        causal=True,
    )

    # Pad back
    output = rearrange(
        pad_input(
            rearrange(output_unpad, "nnz h d -> nnz (h d)"),
            indices_q,
            bsz * num_groups,
            q_len_per_chunk,
        ),
        "b s (h d) -> b s h d",
        h=self.num_heads,
    )

    # Reshape to [bsz, q_len, nh, hd]
    output = output.view(bsz, num_groups, group_size, self.num_heads, self.head_dim)
    output = output.view(bsz, q_len, self.num_heads, self.head_dim)

    # Output projection
    attn_output = self.o_proj(rearrange(output, "b s h d -> b s (h d)"))

    return attn_output, None, past_key_value


# endregion ===========================================================================


# Training: Hierarchical Memory without recurrence cache
def forward_flashattn_hierarchical(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    padding_mask: Optional[torch.LongTensor] = None,
    # Parameters controlled directly in this function
    use_global_context: bool = True,
    use_local_repr: bool = True,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """
    HiCI hierarchical attention (simplified, no recurrence cache).

    Integrates:
    1. HiCI modules (LocalConstructor + GlobalIntegrator)
    2. Ablation modes (use_global_context, use_local_repr)

    Optimized: Q contains chunk tokens only; K/V contains [memories, chunk]
    - Memories are not included in Q, saving compute
    - Chunk tokens can attend to memories via K/V
    - Output is directly the chunk tokens, no extra extraction needed

    Layout:
    - Q:   [chunk]
    - K/V: [global_context?, local?, chunk]

    Ablation modes:
    - Mode 1 (recommended): use_global_context=True, use_local_repr=False
      Q: [chunk], K/V: [global_context, chunk]

    - Mode 2: use_global_context=False, use_local_repr=True
      Q: [chunk], K/V: [local_i, chunk]

    - Mode 3: use_global_context=True, use_local_repr=True
      Q: [chunk], K/V: [global_context, local_i, chunk]

    Args:
        use_global_context: whether to use GlobalIntegrator (aggregates all local slots)
        use_local_repr: whether to prepend LocalConstructor slots to each chunk
    """
    if not self.training:
        warnings.warn(
            "This function should be used just for training. For inference, use forward_flashattn_inference."
        )

    if output_attentions:
        warnings.warn(
            "Output attentions is not supported for patched `LlamaAttention`, returning `None` instead."
        )

    bsz, q_len, hidden_size = hidden_states.size()

    if not hasattr(self, "_hierarchical_no_cache_printed"):
        rank = dist.get_rank() if dist.is_initialized() else 0
        layer_idx = getattr(self, "layer_idx", 0)

        if rank == 0 and layer_idx == 0:
            print("\n" + "=" * 80)
            print("HiCI Hierarchical (Optimized: Q=[chunk], K/V=[hici_slots,chunk])")
            print("=" * 80)
            print(f"  use_global_context : {use_global_context}")
            print(f"  use_local_repr  : {use_local_repr}")

            if use_global_context and not use_local_repr:
                print("  Mode 1: Q=[chunk], K/V=[global_context, chunk]")
            elif not use_global_context and use_local_repr:
                print("  Mode 2: Q=[chunk], K/V=[local_i, chunk]")
            elif use_global_context and use_local_repr:
                print("  Mode 3: Q=[chunk], K/V=[global_context, local_i, chunk]")
            else:
                print("  Baseline: Q=K/V=[chunk]")

            if CAUSAL_CONTEXT_MODE != "none":
                print(f"  🔒 CAUSAL_CONTEXT_MODE: {CAUSAL_CONTEXT_MODE}")
                if CAUSAL_CONTEXT_MODE == "causal_gi":
                    print("     segment_i uses G_i=Agg(L_1..L_i) + L_i")
                elif CAUSAL_CONTEXT_MODE == "causal_shift":
                    print("     segment_i uses G_{i-1}=Agg(L_1..L_{i-1}) + L_{i-1}")
                elif CAUSAL_CONTEXT_MODE == "causal_shift_g":
                    print("     segment_i uses G_{i-1}=Agg(L_1..L_{i-1}) only (no L)")
                elif CAUSAL_CONTEXT_MODE == "causal_gi_gonly":
                    print("     segment_i uses G_i=Agg(L_1..L_i) only (no L in KV, double bottleneck)")

            print("=" * 80 + "\n", flush=True)

        self._hierarchical_no_cache_printed = True

    # ========== Step 1: Split into chunks ==========
    # Mixed-group training: randomly choose group count each forward pass
    # All layers must use the same grouping within a single forward pass
    global _mixed_group_current_ratio, _mixed_group_call_count

    layer_idx = getattr(self, "layer_idx", 0)

    if self.training and MIXED_GROUP_TRAINING:
        # Layer 0 selects new grouping; subsequent layers reuse it
        if layer_idx == 0:
            _mixed_group_current_ratio = random.choice(GROUP_SIZE_RATIOS)
            _mixed_group_call_count += 1
            # Print every 100 batches to confirm mixed grouping is active
            if _mixed_group_call_count % 100 == 1:
                local_rank = dist.get_rank() if dist.is_initialized() else 0
                if local_rank == 0:
                    num_groups = int(1 / _mixed_group_current_ratio)
                    print(
                        f"[Batch {_mixed_group_call_count}] Mixed grouping: {num_groups} groups (ratio={_mixed_group_current_ratio})"
                    )
        current_ratio = _mixed_group_current_ratio
        group_size = int(q_len * current_ratio)
    elif USE_FIXED_SEGMENT_SIZE:
        # Eval mode: use fixed segment_size (consistent with training)
        group_size = FIXED_SEGMENT_SIZE
        if q_len < group_size:
            # Input too short, use the full sequence as one group
            group_size = q_len
    else:
        current_ratio = group_size_ratio
        group_size = int(q_len * current_ratio)

    group_size = max(1, group_size)

    # Handle non-divisible case: round down to nearest divisible size
    if q_len % group_size > 0:
        num_complete_groups = q_len // group_size
        if num_complete_groups == 0:
            group_size = q_len  # use the full sequence as one group

    if not hasattr(self, "_hierarchical_group_printed"):
        local_rank = dist.get_rank() if dist.is_initialized() else 0
        if local_rank == 0 and layer_idx == 0:
            if self.training and MIXED_GROUP_TRAINING:
                print(
                    f"[forward_flashattn_hierarchical] 🔥 MIXED_GROUP_TRAINING enabled, ratios={GROUP_SIZE_RATIOS}"
                )
            elif USE_FIXED_SEGMENT_SIZE:
                num_groups_actual = q_len // group_size
                print(
                    f"[forward_flashattn_hierarchical] 🎯 FIXED_SEGMENT_SIZE mode: "
                    f"segment_size={FIXED_SEGMENT_SIZE} tokens, {num_groups_actual} groups for q_len={q_len}"
                )
            else:
                num_groups_actual = q_len // group_size
                print(
                    f"[forward_flashattn_hierarchical] Fixed ratio grouping: {num_groups_actual} groups, "
                    f"segment_size={group_size} tokens (ratio={group_size_ratio})"
                )
        self._hierarchical_group_printed = True

    num_groups = q_len // group_size

    # Reshape into chunks: [bsz, num_groups, group_size, hidden_size]
    chunks = hidden_states.view(bsz, num_groups, group_size, hidden_size)

    # attention_mask: [bsz, q_len] -> chunk_masks_reshaped: [bsz, num_groups, group_size]
    chunk_masks_reshaped = attention_mask.view(bsz, num_groups, group_size)

    # ========== Step 2: Extract local memories (compress each chunk) ==========
    if (use_global_context or use_local_repr) and hasattr(self, "local_constructor"):
        # Process all chunks in parallel
        all_chunks = chunks.view(bsz * num_groups, group_size, hidden_size)

        # Cast to bfloat16 for Local Construction if input is float32
        original_dtype = all_chunks.dtype
        if all_chunks.dtype == torch.float32:
            all_chunks = all_chunks.to(torch.bfloat16)

        # [bsz, num_groups, group_size] -> [bsz * num_groups, group_size]
        attention_mask_chunks = chunk_masks_reshaped.view(bsz * num_groups, group_size)
        all_local_repr = self.local_constructor(
            all_chunks, attention_mask_chunks
        )  # [bsz * num_groups, num_slots, hidden_size]

        # Cast back to original dtype for consistency
        if original_dtype == torch.float32:
            all_local_repr = all_local_repr.to(torch.float32)

        # Reshape back: [bsz, num_groups, num_slots, hidden_size]
        num_local_slots = all_local_repr.shape[1]
        local_repr_stacked = all_local_repr.view(
            bsz, num_groups, num_local_slots, hidden_size
        )
    else:
        num_local_slots = 0
        local_repr_stacked = None

    # ========== Step 3: Global Integration — aggregate local repr {Li} into global context G ==========
    _causal_mode = CAUSAL_CONTEXT_MODE  # "none", "causal_gi", "causal_shift", "causal_shift_g"
    _is_causal = _causal_mode in ("causal_gi", "causal_shift", "causal_shift_g", "causal_gi_gonly")

    if (
        use_global_context
        and hasattr(self, "global_integrator")
        and local_repr_stacked is not None
    ):
        if _is_causal and hasattr(self.global_integrator, "forward_causal"):
            # Causal mode: each segment gets its own G_i
            global_context_per_group = self.global_integrator.forward_causal(
                local_repr_stacked
            )  # [bsz, num_groups, global_slots, hidden_size]
            num_global_slots = global_context_per_group.shape[2]

            if _causal_mode in ("causal_shift", "causal_shift_g"):
                # segment_i uses G_{i-1}; segment_0 gets zeros
                zeros_g = torch.zeros(
                    bsz, 1, num_global_slots, hidden_size,
                    device=global_context_per_group.device,
                    dtype=global_context_per_group.dtype,
                )
                global_context_per_group = torch.cat(
                    [zeros_g, global_context_per_group[:, :-1, :, :]], dim=1
                )

            global_context = None  # use per-group mode
        else:
            # Non-causal mode: all segments share the same G
            global_context = self.global_integrator(local_repr_stacked)
            num_global_slots = global_context.shape[1]
            global_context_per_group = None
    else:
        global_context = None
        global_context_per_group = None
        num_global_slots = 0

    # causal_shift_g / causal_gi_gonly: use G only, skip L
    if _causal_mode in ("causal_shift_g", "causal_gi_gonly"):
        local_repr_stacked = None
        num_local_slots = 0

    # causal_shift: also shift L_i so segment_i uses L_{i-1}
    if (
        _causal_mode == "causal_shift"
        and use_local_repr
        and local_repr_stacked is not None
    ):
        zeros_l = torch.zeros(
            bsz, 1, num_local_slots, hidden_size,
            device=local_repr_stacked.device,
            dtype=local_repr_stacked.dtype,
        )
        local_repr_stacked = torch.cat(
            [zeros_l, local_repr_stacked[:, :-1, :, :]], dim=1
        )

    # ========== Step 4: Standard Q/K/V projections ==========
    query_states = (
        self.q_proj(hidden_states)
        .view(bsz, q_len, self.num_heads, self.head_dim)
        .transpose(1, 2)
    )
    key_states = (
        self.k_proj(hidden_states)
        .view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        .transpose(1, 2)
    )
    value_states = (
        self.v_proj(hidden_states)
        .view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        .transpose(1, 2)
    )
    # [bsz, nh, q_len, hd]

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]

    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(
        query_states, key_states, cos, sin, position_ids
    )

    # Past Key value support
    if past_key_value is not None:
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)

    past_key_value = (key_states, value_states) if use_cache else None

    # Repeat k/v heads if n_kv_heads < n_heads
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    # ========== Step 5: Project memories to Q/K/V ==========
    # Global context G (output of Global Integration)
    global_context_k = global_context_v = None
    global_context_k_per_group = global_context_v_per_group = None

    if use_global_context and global_context is not None:
        # Non-causal mode: one G shared by all chunks
        global_context_k = (
            self.k_proj(global_context)
            .view(bsz, num_global_slots, self.num_key_value_heads, self.head_dim)
            .transpose(1, 2)
        )
        global_context_v = (
            self.v_proj(global_context)
            .view(bsz, num_global_slots, self.num_key_value_heads, self.head_dim)
            .transpose(1, 2)
        )
        global_context_k = repeat_kv(global_context_k, self.num_key_value_groups)
        global_context_v = repeat_kv(global_context_v, self.num_key_value_groups)
    elif use_global_context and global_context_per_group is not None:
        # Causal mode: each chunk has its own G_i
        # global_context_per_group: [bsz, num_groups, global_slots, hidden_size]
        gc_flat = global_context_per_group.view(
            bsz * num_groups, num_global_slots, hidden_size
        )
        gc_k_flat = (
            self.k_proj(gc_flat)
            .view(bsz * num_groups, num_global_slots, self.num_key_value_heads, self.head_dim)
            .transpose(1, 2)
        )
        gc_v_flat = (
            self.v_proj(gc_flat)
            .view(bsz * num_groups, num_global_slots, self.num_key_value_heads, self.head_dim)
            .transpose(1, 2)
        )
        gc_k_flat = repeat_kv(gc_k_flat, self.num_key_value_groups)
        gc_v_flat = repeat_kv(gc_v_flat, self.num_key_value_groups)
        # [bsz*num_groups, nh, global_slots, hd] -> [bsz, num_groups, nh, global_slots, hd]
        global_context_k_per_group = gc_k_flat.view(
            bsz, num_groups, self.num_heads, num_global_slots, self.head_dim
        )
        global_context_v_per_group = gc_v_flat.view(
            bsz, num_groups, self.num_heads, num_global_slots, self.head_dim
        )

    # ========== Step 6: Reshape into chunks ==========
    query_chunks = query_states.view(
        bsz, self.num_heads, num_groups, group_size, self.head_dim
    )
    key_chunks = key_states.view(
        bsz, self.num_heads, num_groups, group_size, self.head_dim
    )
    value_chunks = value_states.view(
        bsz, self.num_heads, num_groups, group_size, self.head_dim
    )

    # Local memories for each chunk (if needed)
    # Project K/V only — Q contains chunk tokens; G and Li are context-only (no Q)
    if use_local_repr and local_repr_stacked is not None:
        # Reshape: [bsz, num_groups, num_slots, hidden] -> [bsz*num_groups, num_slots, hidden]
        local_mems_flat = local_repr_stacked.view(
            bsz * num_groups, num_local_slots, hidden_size
        )

        local_k_flat = (
            self.k_proj(local_mems_flat)
            .view(
                bsz * num_groups,
                num_local_slots,
                self.num_key_value_heads,
                self.head_dim,
            )
            .transpose(1, 2)
        )
        local_v_flat = (
            self.v_proj(local_mems_flat)
            .view(
                bsz * num_groups,
                num_local_slots,
                self.num_key_value_heads,
                self.head_dim,
            )
            .transpose(1, 2)
        )

        # Repeat k/v heads (batched)
        local_k_flat = repeat_kv(local_k_flat, self.num_key_value_groups)
        local_v_flat = repeat_kv(local_v_flat, self.num_key_value_groups)

        # Reshape: [bsz*num_groups, nh, num_slots, hd] -> [bsz, num_groups, nh, num_slots, hd]
        local_k_all = local_k_flat.view(
            bsz, num_groups, self.num_heads, num_local_slots, self.head_dim
        )
        local_v_all = local_v_flat.view(
            bsz, num_groups, self.num_heads, num_local_slots, self.head_dim
        )
    else:
        local_k_all = None
        local_v_all = None

    # ========== Step 7: Process chunks with memories (vectorized) ==========
    # Q: chunk tokens only, K/V: [memories, chunk] — all tensor ops, no Python loops

    # query_chunks: [bsz, nh, num_groups, group_size, hd]
    # target: [bsz * num_groups, group_size, nh, hd]  (batch-first)
    all_chunks_q_flat = query_chunks.permute(0, 2, 3, 1, 4).reshape(
        bsz * num_groups, group_size, self.num_heads, self.head_dim
    )

    # Compute K/V total length (G + Li prefix + chunk tokens)
    prefix_len = 0
    if use_global_context and hasattr(self, "global_integrator"):
        prefix_len += num_global_slots
    if use_local_repr and hasattr(self, "local_constructor"):
        prefix_len += num_local_slots
    kv_len_per_chunk = prefix_len + group_size

    if prefix_len > 0:
        # Use torch.cat for context prefix assembly (better memory efficiency)
        kv_components_k = []
        kv_components_v = []

        # Append global_context
        if use_global_context and global_context_k is not None:
            # Non-causal: all chunks share the same G
            global_context_k_exp = global_context_k.unsqueeze(2).expand(
                -1, -1, num_groups, -1, -1
            )
            global_context_v_exp = global_context_v.unsqueeze(2).expand(
                -1, -1, num_groups, -1, -1
            )
            kv_components_k.append(global_context_k_exp)
            kv_components_v.append(global_context_v_exp)
        elif use_global_context and global_context_k_per_group is not None:
            # Causal: each chunk has its own G_i
            # convert [bsz, num_groups, nh, global_slots, hd] -> [bsz, nh, num_groups, global_slots, hd]
            kv_components_k.append(global_context_k_per_group.permute(0, 2, 1, 3, 4))
            kv_components_v.append(global_context_v_per_group.permute(0, 2, 1, 3, 4))

        # Append local memories (different per chunk)
        if use_local_repr and local_k_all is not None:
            # convert [bsz, num_groups, nh, num_local_slots, hd] -> [bsz, nh, num_groups, num_local_slots, hd]
            local_k_exp = local_k_all.permute(0, 2, 1, 3, 4)
            local_v_exp = local_v_all.permute(0, 2, 1, 3, 4)
            kv_components_k.append(local_k_exp)
            kv_components_v.append(local_v_exp)

        # Append chunk tokens
        kv_components_k.append(key_chunks)
        kv_components_v.append(value_chunks)

        # Concatenate all components at once (dim=3 is the seq_len dimension)
        all_k = torch.cat(
            kv_components_k, dim=3
        )  # [bsz, nh, num_groups, kv_len_per_chunk, hd]
        all_v = torch.cat(kv_components_v, dim=3)
    else:
        # No context prefix (G/Li), use chunk K/V directly
        all_k = key_chunks
        all_v = value_chunks
        kv_len_per_chunk = group_size

    # Convert to flash attention format
    # [bsz, nh, num_groups, kv_len, hd] -> [bsz * num_groups, kv_len, nh, hd]  (batch-first)
    all_k_flat = all_k.permute(0, 2, 3, 1, 4).reshape(
        bsz * num_groups, kv_len_per_chunk, self.num_heads, self.head_dim
    )
    all_v_flat = all_v.permute(0, 2, 3, 1, 4).reshape(
        bsz * num_groups, kv_len_per_chunk, self.num_heads, self.head_dim
    )

    # Pack K and V: [bsz * num_groups, kv_len, 2, nh, hd]  (batch-first)
    all_chunks_kv_flat = torch.stack([all_k_flat, all_v_flat], dim=2)

    q_len_per_chunk = group_size

    # ========== Step 9: Prepare padding masks (1=real token, 0=padding) ==========
    # 9.1 Q padding masks (chunk tokens only, batch-first: [bsz * num_groups, group_size])
    all_masks_q_flat = chunk_masks_reshaped.reshape(bsz * num_groups, q_len_per_chunk)

    # 9.2 K/V padding masks (memories + chunk)
    all_masks_kv_stacked = torch.empty(
        bsz,
        num_groups,
        kv_len_per_chunk,
        dtype=chunk_masks_reshaped.dtype,
        device=chunk_masks_reshaped.device,
    )

    offset = 0
    if use_global_context:
        all_masks_kv_stacked[:, :, offset : offset + num_global_slots] = 1
        offset += num_global_slots
    if use_local_repr:
        all_masks_kv_stacked[:, :, offset : offset + num_local_slots] = 1
        offset += num_local_slots
    all_masks_kv_stacked[:, :, offset : offset + group_size] = chunk_masks_reshaped

    # causal_shift/causal_shift_g: segment_0 G (and L) are zero-padded, mask out
    if _causal_mode in ("causal_shift", "causal_shift_g"):
        mem_offset = 0
        if use_global_context:
            all_masks_kv_stacked[:, 0, mem_offset : mem_offset + num_global_slots] = 0
            mem_offset += num_global_slots
        if use_local_repr:
            all_masks_kv_stacked[:, 0, mem_offset : mem_offset + num_local_slots] = 0

    all_masks_kv_flat = all_masks_kv_stacked.reshape(bsz * num_groups, kv_len_per_chunk)

    # Unpad Q
    q_unpad, indices_q, cu_seqlens_q, max_seqlen_q = unpad_input(
        rearrange(all_chunks_q_flat, "b s h d -> b s (h d)"), all_masks_q_flat
    )
    q_unpad = rearrange(q_unpad, "nnz (h d) -> nnz h d", h=self.num_heads)

    # Unpad KV (already packed)
    kv_flat_2d = rearrange(all_chunks_kv_flat, "b s two h d -> b s (two h d)")
    kv_unpad, indices_kv, cu_seqlens_kv, max_seqlen_kv = unpad_input(
        kv_flat_2d, all_masks_kv_flat
    )
    kv_unpad = rearrange(
        kv_unpad, "nnz (two h d) -> nnz two h d", two=2, h=self.num_heads
    )

    # Flash Attention with KV packed
    # Q: [chunk tokens], K/V: [memories, chunk tokens]
    output_unpad = flash_attn_varlen_kvpacked_func(
        q_unpad,  # [total_q_tokens, num_heads, head_dim]
        kv_unpad,  # [total_kv_tokens, 2, num_heads, head_dim] - packed K/V
        cu_seqlens_q,  # Q sequence boundaries
        cu_seqlens_kv,  # KV sequence boundaries
        max_seqlen_q,  # Q max sequence length
        max_seqlen_kv,  # KV max sequence length
        dropout_p=0.0,
        softmax_scale=None,
        causal=True,
    )

    # Pad back to original shape
    output = rearrange(
        pad_input(
            rearrange(output_unpad, "nnz h d -> nnz (h d)"),
            indices_q,
            num_groups * bsz,
            q_len_per_chunk,
        ),
        "b s (h d) -> b s h d",
        h=self.num_heads,
    )  # [num_groups*bsz, group_size, nh, hd]

    # Reshape: [bsz*num_groups, group_size, nh, hd] -> [bsz, num_groups, group_size, nh, hd]
    output = output.view(bsz, num_groups, group_size, self.num_heads, self.head_dim)

    output = output.view(bsz, q_len, self.num_heads, self.head_dim)

    # Visualization: collect attention stats (inference only)
    if COLLECT_ATTENTION_FOR_VIZ and not self.training and prefix_len > 0:
        with torch.no_grad():
            # Manually compute attention weights for visualization
            # Average over all segments rather than just the first one
            scale = 1.0 / math.sqrt(self.head_dim)
            attn_weights = (
                torch.matmul(
                    all_chunks_q_flat.transpose(
                        1, 2
                    ),  # [bsz*num_groups, nh, group_size, hd]
                    all_k_flat.transpose(1, 2).transpose(
                        -1, -2
                    ),  # [bsz*num_groups, nh, hd, kv_len]
                )
                * scale
            )  # [bsz*num_groups, nh, group_size, kv_len]
            attn_probs = F.softmax(attn_weights, dim=-1)

            # K layout: [global_context, local, chunk]
            # Average over all segments, heads, and query positions
            offset = 0
            attn_to_global = 0.0
            attn_to_local = 0.0

            if use_global_context and num_global_slots > 0:
                attn_to_global = (
                    attn_probs[:, :, :, offset : offset + num_global_slots]
                    .mean()
                    .item()
                )
                offset += num_global_slots
            if use_local_repr and num_local_slots > 0:
                attn_to_local = (
                    attn_probs[:, :, :, offset : offset + num_local_slots].mean().item()
                )
                offset += num_local_slots
            attn_to_tokens = attn_probs[:, :, :, offset:].mean().item()

            attention_visualizer["num_global_slots"] = num_global_slots
            attention_visualizer["num_local_slots"] = num_local_slots
            attention_visualizer["segment_len"] = group_size
            attention_visualizer["layer_attn_to_global"].append(attn_to_global)
            attention_visualizer["layer_attn_to_local"].append(attn_to_local)
            attention_visualizer["layer_attn_to_tokens"].append(attn_to_tokens)

            # Save first sample's attention maps across layers (max 32 to limit file size)
            layer_idx = getattr(self, "layer_idx", 0)
            saved_count = len(attention_visualizer["segment_attention_maps"])
            if saved_count < 32:
                attn_map = attn_probs[0, 0, :, :].cpu().numpy().tolist()
                attention_visualizer["segment_attention_maps"].append(
                    {"layer": layer_idx, "attention_map": attn_map}
                )

    # Output projection
    attn_output = self.o_proj(rearrange(output, "b s h d -> b s (h d)"))

    return attn_output, None, past_key_value


def forward_flashattn_hierarchical_inference(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    padding_mask: Optional[torch.LongTensor] = None,
    use_global_context: bool = True,
    use_local_repr: bool = True,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """
    HiCI hierarchical attention for inference (supports arbitrary-length input with padding).

    Pads input to a multiple of segment_size, runs chunked HiCI attention, then truncates back.
    K/V layout: [global_context?, local?, chunk]. Q contains chunk tokens only.

    Ablation modes:
    - Mode 1: use_global_context=True,  use_local_repr=False  → K/V=[G, chunk]
    - Mode 2: use_global_context=False, use_local_repr=True   → K/V=[L_i, chunk]
    - Mode 3: use_global_context=True,  use_local_repr=True   → K/V=[G, L_i, chunk]

    Args:
        use_global_context: whether to use GlobalIntegrator (aggregates all local slots)
        use_local_repr: whether to prepend LocalConstructor slots to each chunk
    """
    if output_attentions:
        warnings.warn(
            "Output attentions is not supported for patched `LlamaAttention`, returning `None` instead."
        )

    bsz, q_len, hidden_size = hidden_states.size()

    # ========================================================================
    # 🔥 Decode mode: q_len is very small with a KV cache — use full attention.
    # HiCI's value is at prefill; during decode there is only 1 token and chunking
    # is meaningless. KV cache stores raw K/V; G is recomputed at prefill, not cached.
    # ========================================================================
    if q_len <= 32 and past_key_value is not None:
        # Decode mode: full attention (same as LongLoRA inference)
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]

        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids
        )

        # Prepend past KV cache
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        # Repeat k/v heads if n_kv_heads < n_heads (GQA)
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # Flash Attention
        query_states = query_states.transpose(1, 2)  # [bsz, q_len, nh, hd]
        key_states = key_states.transpose(1, 2)  # [bsz, kv_len, nh, hd]
        value_states = value_states.transpose(1, 2)  # [bsz, kv_len, nh, hd]

        attn_output = flash_attn_func(
            query_states,
            key_states,
            value_states,
            dropout_p=0.0,
            softmax_scale=None,
            causal=True,
        )

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value

    # ========================================================================
    # 🔥 Prefill mode: HiCI chunking with Local Construction + Global Integration
    # ========================================================================

    # Save original length for truncation after padding
    original_q_len = q_len

    # Print config once (rank 0, layer 0 only)
    if not hasattr(self, "_hierarchical_inference_printed"):
        rank = dist.get_rank() if dist.is_initialized() else 0
        layer_idx = getattr(self, "layer_idx", 0)

        if rank == 0 and layer_idx == 0:
            print("\n" + "=" * 80)
            print("HiCI Hierarchical INFERENCE (with padding support)")
            print("=" * 80)
            print(f"  use_global_context : {use_global_context}")
            print(f"  use_local_repr  : {use_local_repr}")

            if use_global_context and not use_local_repr:
                print("  Mode 1: Q=[chunk], K/V=[global_context, chunk]")
            elif not use_global_context and use_local_repr:
                print("  Mode 2: Q=[chunk], K/V=[local_i, chunk]")
            elif use_global_context and use_local_repr:
                print("  Mode 3: Q=[chunk], K/V=[global_context, local_i, chunk]")
            else:
                print("  Baseline: Q=K/V=[chunk]")

            print("=" * 80 + "\n", flush=True)

        self._hierarchical_inference_printed = True

    # ========== Step 1: Chunk split (simplified for inference) ==========
    layer_idx = getattr(self, "layer_idx", 0)

    # ========================================================================
    # 🔥 Full Attention + HiCI mode: no chunking, but still applies Local Construction + Global Integration.
    #
    # When USE_FULL_ATTN_WITH_HICI = True:
    # - Treat entire input as a single chunk (num_groups = 1)
    # - Extract local repr Li from the whole input -> aggregate to global context G
    # - All tokens attend to [G, all_tokens]
    # ========================================================================
    if USE_FULL_ATTN_WITH_HICI:
        # Full Attention + HiCI: entire input is a single chunk
        group_size = q_len
        num_groups = 1
        padding_needed = 0

        if not getattr(
            forward_flashattn_hierarchical_inference, "_full_attn_mem_printed", False
        ):
            local_rank = dist.get_rank() if dist.is_initialized() else 0
            if local_rank == 0 and layer_idx == 0:
                print("\n" + "=" * 80)
                print(
                    "🔥 Full Attention + HiCI mode (USE_FULL_ATTN_WITH_HICI=True)"
                )
                print("=" * 80)
                print(f"  input_len: {q_len}")
                print(f"  No chunking, entire input as a single chunk")
                print(
                    f"  HiCI enabled: use_global_context={use_global_context}, use_local_repr={use_local_repr}"
                )
                print(f"  Q: [all_tokens], K/V: [global_context, all_tokens]")
                print("=" * 80 + "\n", flush=True)
                forward_flashattn_hierarchical_inference._full_attn_mem_printed = True
    else:
        # Standard chunking mode
        # 🔥 Inference: always use a fixed segment_size
        group_size = (
            FIXED_SEGMENT_SIZE
            if USE_FIXED_SEGMENT_SIZE
            else int(q_len * group_size_ratio)
        )

        # Handle sequences shorter than group_size
        if q_len < group_size:
            group_size = q_len

        group_size = max(1, group_size)

        # 🔥 Inference: handle non-divisible lengths with padding
        padding_needed = 0
        if q_len % group_size > 0:
            padded_q_len = ((q_len + group_size - 1) // group_size) * group_size
            padding_needed = padded_q_len - q_len

            # Pad hidden_states: [bsz, q_len, hidden_size] -> [bsz, padded_q_len, hidden_size]
            hidden_states = torch.nn.functional.pad(
                hidden_states, (0, 0, 0, padding_needed), mode="constant", value=0
            )

            # Pad attention_mask: [bsz, q_len] -> [bsz, padded_q_len] (padding positions = 0)
            if attention_mask is not None:
                attention_mask = torch.nn.functional.pad(
                    attention_mask, (0, padding_needed), mode="constant", value=0
                )

            # 🔥 Pad position_ids: [bsz, q_len] -> [bsz, padded_q_len] (continue incrementing)
            if position_ids is not None:
                last_pos = position_ids[:, -1:] + 1
                padding_positions = last_pos + torch.arange(
                    padding_needed, device=position_ids.device, dtype=position_ids.dtype
                ).unsqueeze(
                    0
                )  # [1, padding_needed] -> broadcast to [bsz, padding_needed]
                position_ids = torch.cat([position_ids, padding_positions], dim=1)

            q_len = padded_q_len

        num_groups = q_len // group_size

        # ========================================================================
        # 🔥 Single-group: when num_groups == 1 in standard chunking mode, disable G and Li.
        # Set USE_FULL_ATTN_WITH_HICI = True to use HiCI without chunking.
        # ========================================================================
        if num_groups == 1:
            # No HiCI context (G/Li) for a single group — chunking is meaningless
            # Set USE_FULL_ATTN_WITH_HICI = True to enable HiCI with single group
            use_global_context = False
            use_local_repr = False

        if not getattr(
            forward_flashattn_hierarchical_inference, "_prefill_printed", False
        ):
            local_rank = dist.get_rank() if dist.is_initialized() else 0
            if local_rank == 0 and layer_idx == 0:
                print(
                    f"[HiCI Prefill] 🎯 original_len={original_q_len}, padded_len={q_len}, "
                    f"segment_size={group_size}, num_groups={num_groups}, padding={padding_needed}"
                )
                if num_groups == 1:
                    print(
                        f"[HiCI Prefill] ⚠️ Single group detected, HiCI disabled. "
                        f"Set USE_FULL_ATTN_WITH_HICI=True to enable HiCI with single group."
                    )
                forward_flashattn_hierarchical_inference._prefill_printed = True

    # Reshape into chunks: [bsz, num_groups, group_size, hidden_size]
    chunks = hidden_states.view(bsz, num_groups, group_size, hidden_size)

    # attention_mask: [bsz, q_len] -> chunk_masks_reshaped: [bsz, num_groups, group_size]
    chunk_masks_reshaped = attention_mask.view(bsz, num_groups, group_size)

    # ========== Step 2: Extract local memories (compress each chunk) ==========
    if (use_global_context or use_local_repr) and hasattr(self, "local_constructor"):
        # Process all chunks in parallel
        all_chunks = chunks.view(bsz * num_groups, group_size, hidden_size)

        # Cast to bfloat16 for Local Construction if input is float32
        original_dtype = all_chunks.dtype
        if all_chunks.dtype == torch.float32:
            all_chunks = all_chunks.to(torch.bfloat16)

        # [bsz, num_groups, group_size] -> [bsz * num_groups, group_size]
        attention_mask_chunks = chunk_masks_reshaped.view(bsz * num_groups, group_size)
        all_local_repr = self.local_constructor(
            all_chunks, attention_mask_chunks
        )  # [bsz * num_groups, num_slots, hidden_size]

        # Cast back to original dtype for consistency
        if original_dtype == torch.float32:
            all_local_repr = all_local_repr.to(torch.float32)

        # Reshape back: [bsz, num_groups, num_slots, hidden_size]
        num_local_slots = all_local_repr.shape[1]
        local_repr_stacked = all_local_repr.view(
            bsz, num_groups, num_local_slots, hidden_size
        )
    else:
        num_local_slots = 0
        local_repr_stacked = None

    # ========== Step 3: Global Integration — aggregate local repr {Li} into global context G ==========
    _causal_mode = CAUSAL_CONTEXT_MODE  # "none", "causal_gi", "causal_shift", "causal_shift_g"
    _is_causal = _causal_mode in ("causal_gi", "causal_shift", "causal_shift_g", "causal_gi_gonly")

    if (
        use_global_context
        and hasattr(self, "global_integrator")
        and local_repr_stacked is not None
    ):
        if _is_causal and hasattr(self.global_integrator, "forward_causal"):
            # Causal mode: each segment gets its own G_i
            global_context_per_group = self.global_integrator.forward_causal(
                local_repr_stacked
            )  # [bsz, num_groups, global_slots, hidden_size]
            num_global_slots = global_context_per_group.shape[2]

            if _causal_mode in ("causal_shift", "causal_shift_g"):
                # segment_i uses G_{i-1}; segment_0 gets zeros
                zeros_g = torch.zeros(
                    bsz, 1, num_global_slots, hidden_size,
                    device=global_context_per_group.device,
                    dtype=global_context_per_group.dtype,
                )
                global_context_per_group = torch.cat(
                    [zeros_g, global_context_per_group[:, :-1, :, :]], dim=1
                )

            global_context = None  # use per-group mode
        else:
            # Non-causal mode: all segments share the same G
            global_context = self.global_integrator(local_repr_stacked)
            num_global_slots = global_context.shape[1]
            global_context_per_group = None
    else:
        global_context = None
        global_context_per_group = None
        num_global_slots = 0

    # causal_shift_g / causal_gi_gonly: use G only, skip L
    if _causal_mode in ("causal_shift_g", "causal_gi_gonly"):
        local_repr_stacked = None
        num_local_slots = 0

    # causal_shift: also shift L_i so segment_i uses L_{i-1}
    if (
        _causal_mode == "causal_shift"
        and use_local_repr
        and local_repr_stacked is not None
    ):
        zeros_l = torch.zeros(
            bsz, 1, num_local_slots, hidden_size,
            device=local_repr_stacked.device,
            dtype=local_repr_stacked.dtype,
        )
        local_repr_stacked = torch.cat(
            [zeros_l, local_repr_stacked[:, :-1, :, :]], dim=1
        )

    # ========== Step 4: Standard Q/K/V projections ==========
    query_states = (
        self.q_proj(hidden_states)
        .view(bsz, q_len, self.num_heads, self.head_dim)
        .transpose(1, 2)
    )
    key_states = (
        self.k_proj(hidden_states)
        .view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        .transpose(1, 2)
    )
    value_states = (
        self.v_proj(hidden_states)
        .view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        .transpose(1, 2)
    )
    # [bsz, nh, q_len, hd]

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]

    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(
        query_states, key_states, cos, sin, position_ids
    )

    # Past Key value support
    if past_key_value is not None:
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)

    past_key_value = (key_states, value_states) if use_cache else None

    # Repeat k/v heads if n_kv_heads < n_heads
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    # ========== Step 5: Project memories to Q/K/V ==========
    # Global context G (output of Global Integration)
    global_context_k = global_context_v = None
    global_context_k_per_group = global_context_v_per_group = None

    if use_global_context and global_context is not None:
        # Non-causal mode: one G shared by all chunks
        global_context_k = (
            self.k_proj(global_context)
            .view(bsz, num_global_slots, self.num_key_value_heads, self.head_dim)
            .transpose(1, 2)
        )
        global_context_v = (
            self.v_proj(global_context)
            .view(bsz, num_global_slots, self.num_key_value_heads, self.head_dim)
            .transpose(1, 2)
        )
        global_context_k = repeat_kv(global_context_k, self.num_key_value_groups)
        global_context_v = repeat_kv(global_context_v, self.num_key_value_groups)
    elif use_global_context and global_context_per_group is not None:
        # Causal mode: each chunk has its own G_i
        # global_context_per_group: [bsz, num_groups, global_slots, hidden_size]
        gc_flat = global_context_per_group.view(
            bsz * num_groups, num_global_slots, hidden_size
        )
        gc_k_flat = (
            self.k_proj(gc_flat)
            .view(bsz * num_groups, num_global_slots, self.num_key_value_heads, self.head_dim)
            .transpose(1, 2)
        )
        gc_v_flat = (
            self.v_proj(gc_flat)
            .view(bsz * num_groups, num_global_slots, self.num_key_value_heads, self.head_dim)
            .transpose(1, 2)
        )
        gc_k_flat = repeat_kv(gc_k_flat, self.num_key_value_groups)
        gc_v_flat = repeat_kv(gc_v_flat, self.num_key_value_groups)
        # [bsz*num_groups, nh, global_slots, hd] -> [bsz, num_groups, nh, global_slots, hd]
        global_context_k_per_group = gc_k_flat.view(
            bsz, num_groups, self.num_heads, num_global_slots, self.head_dim
        )
        global_context_v_per_group = gc_v_flat.view(
            bsz, num_groups, self.num_heads, num_global_slots, self.head_dim
        )

    # ========== Step 6: Reshape into chunks ==========
    query_chunks = query_states.view(
        bsz, self.num_heads, num_groups, group_size, self.head_dim
    )
    key_chunks = key_states.view(
        bsz, self.num_heads, num_groups, group_size, self.head_dim
    )
    value_chunks = value_states.view(
        bsz, self.num_heads, num_groups, group_size, self.head_dim
    )

    # Local memories for each chunk (if needed)
    # Project K/V only — Q contains chunk tokens; G and Li are context-only (no Q)
    if use_local_repr and local_repr_stacked is not None:
        # Reshape: [bsz, num_groups, num_slots, hidden] -> [bsz*num_groups, num_slots, hidden]
        local_mems_flat = local_repr_stacked.view(
            bsz * num_groups, num_local_slots, hidden_size
        )

        local_k_flat = (
            self.k_proj(local_mems_flat)
            .view(
                bsz * num_groups,
                num_local_slots,
                self.num_key_value_heads,
                self.head_dim,
            )
            .transpose(1, 2)
        )
        local_v_flat = (
            self.v_proj(local_mems_flat)
            .view(
                bsz * num_groups,
                num_local_slots,
                self.num_key_value_heads,
                self.head_dim,
            )
            .transpose(1, 2)
        )

        # Repeat k/v heads (batched)
        local_k_flat = repeat_kv(local_k_flat, self.num_key_value_groups)
        local_v_flat = repeat_kv(local_v_flat, self.num_key_value_groups)

        # Reshape: [bsz*num_groups, nh, num_slots, hd] -> [bsz, num_groups, nh, num_slots, hd]
        local_k_all = local_k_flat.view(
            bsz, num_groups, self.num_heads, num_local_slots, self.head_dim
        )
        local_v_all = local_v_flat.view(
            bsz, num_groups, self.num_heads, num_local_slots, self.head_dim
        )
    else:
        local_k_all = None
        local_v_all = None

    # ========== Step 7: Process chunks with memories (vectorized) ==========
    # Q: chunk tokens only, K/V: [memories, chunk] — all tensor ops, no Python loops

    # query_chunks: [bsz, nh, num_groups, group_size, hd]
    # target: [bsz * num_groups, group_size, nh, hd]  (batch-first)
    all_chunks_q_flat = query_chunks.permute(0, 2, 3, 1, 4).reshape(
        bsz * num_groups, group_size, self.num_heads, self.head_dim
    )

    # Compute K/V total length (G + Li prefix + chunk tokens)
    prefix_len = 0
    if use_global_context and hasattr(self, "global_integrator"):
        prefix_len += num_global_slots
    if use_local_repr and hasattr(self, "local_constructor"):
        prefix_len += num_local_slots
    kv_len_per_chunk = prefix_len + group_size

    if prefix_len > 0:
        # Use torch.cat for context prefix assembly (better memory efficiency)
        kv_components_k = []
        kv_components_v = []

        # Append global_context
        if use_global_context and global_context_k is not None:
            # Non-causal: all chunks share the same G
            global_context_k_exp = global_context_k.unsqueeze(2).expand(
                -1, -1, num_groups, -1, -1
            )
            global_context_v_exp = global_context_v.unsqueeze(2).expand(
                -1, -1, num_groups, -1, -1
            )
            kv_components_k.append(global_context_k_exp)
            kv_components_v.append(global_context_v_exp)
        elif use_global_context and global_context_k_per_group is not None:
            # Causal: each chunk has its own G_i
            # convert [bsz, num_groups, nh, global_slots, hd] -> [bsz, nh, num_groups, global_slots, hd]
            kv_components_k.append(global_context_k_per_group.permute(0, 2, 1, 3, 4))
            kv_components_v.append(global_context_v_per_group.permute(0, 2, 1, 3, 4))

        # Append local memories (different per chunk)
        if use_local_repr and local_k_all is not None:
            # convert [bsz, num_groups, nh, num_local_slots, hd] -> [bsz, nh, num_groups, num_local_slots, hd]
            local_k_exp = local_k_all.permute(0, 2, 1, 3, 4)
            local_v_exp = local_v_all.permute(0, 2, 1, 3, 4)
            kv_components_k.append(local_k_exp)
            kv_components_v.append(local_v_exp)

        # Append chunk tokens
        kv_components_k.append(key_chunks)
        kv_components_v.append(value_chunks)

        # Concatenate all components at once (dim=3 is the seq_len dimension)
        all_k = torch.cat(
            kv_components_k, dim=3
        )  # [bsz, nh, num_groups, kv_len_per_chunk, hd]
        all_v = torch.cat(kv_components_v, dim=3)
    else:
        # No context prefix (G/Li), use chunk K/V directly
        all_k = key_chunks
        all_v = value_chunks
        kv_len_per_chunk = group_size

    # Convert to flash attention format
    # [bsz, nh, num_groups, kv_len, hd] -> [bsz * num_groups, kv_len, nh, hd]  (batch-first)
    all_k_flat = all_k.permute(0, 2, 3, 1, 4).reshape(
        bsz * num_groups, kv_len_per_chunk, self.num_heads, self.head_dim
    )
    all_v_flat = all_v.permute(0, 2, 3, 1, 4).reshape(
        bsz * num_groups, kv_len_per_chunk, self.num_heads, self.head_dim
    )

    # Pack K and V: [bsz * num_groups, kv_len, 2, nh, hd]  (batch-first)
    all_chunks_kv_flat = torch.stack([all_k_flat, all_v_flat], dim=2)

    q_len_per_chunk = group_size

    # ========== Step 9: Prepare padding masks (1=real token, 0=padding) ==========
    # 9.1 Q padding masks (chunk tokens only, batch-first: [bsz * num_groups, group_size])
    all_masks_q_flat = chunk_masks_reshaped.reshape(bsz * num_groups, q_len_per_chunk)

    # 9.2 K/V padding masks (memories + chunk)
    all_masks_kv_stacked = torch.empty(
        bsz,
        num_groups,
        kv_len_per_chunk,
        dtype=chunk_masks_reshaped.dtype,
        device=chunk_masks_reshaped.device,
    )

    offset = 0
    if use_global_context:
        all_masks_kv_stacked[:, :, offset : offset + num_global_slots] = 1
        offset += num_global_slots
    if use_local_repr:
        all_masks_kv_stacked[:, :, offset : offset + num_local_slots] = 1
        offset += num_local_slots
    all_masks_kv_stacked[:, :, offset : offset + group_size] = chunk_masks_reshaped

    # causal_shift/causal_shift_g: segment_0 G (and L) are zero-padded, mask out
    if _causal_mode in ("causal_shift", "causal_shift_g"):
        mem_offset = 0
        if use_global_context:
            all_masks_kv_stacked[:, 0, mem_offset : mem_offset + num_global_slots] = 0
            mem_offset += num_global_slots
        if use_local_repr:
            all_masks_kv_stacked[:, 0, mem_offset : mem_offset + num_local_slots] = 0

    all_masks_kv_flat = all_masks_kv_stacked.reshape(bsz * num_groups, kv_len_per_chunk)

    # Unpad Q
    q_unpad, indices_q, cu_seqlens_q, max_seqlen_q = unpad_input(
        rearrange(all_chunks_q_flat, "b s h d -> b s (h d)"), all_masks_q_flat
    )
    q_unpad = rearrange(q_unpad, "nnz (h d) -> nnz h d", h=self.num_heads)

    # Unpad KV (already packed)
    kv_flat_2d = rearrange(all_chunks_kv_flat, "b s two h d -> b s (two h d)")
    kv_unpad, indices_kv, cu_seqlens_kv, max_seqlen_kv = unpad_input(
        kv_flat_2d, all_masks_kv_flat
    )
    kv_unpad = rearrange(
        kv_unpad, "nnz (two h d) -> nnz two h d", two=2, h=self.num_heads
    )

    # Flash Attention with KV packed
    # Q: [chunk tokens], K/V: [memories, chunk tokens]
    output_unpad = flash_attn_varlen_kvpacked_func(
        q_unpad,  # [total_q_tokens, num_heads, head_dim]
        kv_unpad,  # [total_kv_tokens, 2, num_heads, head_dim] - packed K/V
        cu_seqlens_q,  # Q sequence boundaries
        cu_seqlens_kv,  # KV sequence boundaries
        max_seqlen_q,  # Q max sequence length
        max_seqlen_kv,  # KV max sequence length
        dropout_p=0.0,
        softmax_scale=None,
        causal=True,
    )

    # Pad back to original shape
    output = rearrange(
        pad_input(
            rearrange(output_unpad, "nnz h d -> nnz (h d)"),
            indices_q,
            num_groups * bsz,
            q_len_per_chunk,
        ),
        "b s (h d) -> b s h d",
        h=self.num_heads,
    )  # [num_groups*bsz, group_size, nh, hd]

    # Reshape: [bsz*num_groups, group_size, nh, hd] -> [bsz, num_groups, group_size, nh, hd]
    output = output.view(bsz, num_groups, group_size, self.num_heads, self.head_dim)

    # [bsz*num_groups, group_size, nh, hd] -> [bsz, q_len, nh, hd]
    output = output.view(bsz, q_len, self.num_heads, self.head_dim)

    # 🔥 Inference: truncate back to original length (remove padding)
    if original_q_len < q_len:
        output = output[:, :original_q_len, :, :]

        # 🔥 CRITICAL: also truncate past_key_value; otherwise decode position_ids mismatch.
        # Prefill returns padded KV cache length, but output is already truncated — RoPE
        # position IDs during decode would be misaligned, causing repetition/garbage output.
        if past_key_value is not None:
            past_key_value = (
                past_key_value[0][:, :, :original_q_len, :],  # key: [bsz, nh, seq, hd]
                past_key_value[1][
                    :, :, :original_q_len, :
                ],  # value: [bsz, nh, seq, hd]
            )

    # Output projection
    attn_output = self.o_proj(rearrange(output, "b s h d -> b s (h d)"))

    return attn_output, None, past_key_value


def forward_noflashattn_hierarchical(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    padding_mask: Optional[torch.LongTensor] = None,
    use_global_context: bool = False,
    use_local_repr: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """
    HiCI hierarchical attention without Flash Attention (for ablation studies).

    Same logic as forward_flashattn_hierarchical but uses standard matmul+softmax.
    K/V layout: [global_context?, local?, chunk]. Q contains chunk tokens only.

    Args:
        use_global_context: whether to use GlobalIntegrator (aggregates all local slots)
        use_local_repr: whether to prepend LocalConstructor slots to each chunk
    """
    if not self.training:
        warnings.warn(
            "This function should be used just for training. For inference, use forward_flashattn_inference."
        )

    if output_attentions:
        warnings.warn(
            "Output attentions is not supported for patched `LlamaAttention`, returning `None` instead."
        )

    bsz, q_len, hidden_size = hidden_states.size()

    # Print config once (rank 0, layer 0 only)
    if not hasattr(self, "_hierarchical_noflash_printed"):
        rank = dist.get_rank() if dist.is_initialized() else 0
        layer_idx = getattr(self, "layer_idx", 0)

        if rank == 0 and layer_idx == 0:
            print("\n" + "=" * 80)
            print("HiCI Hierarchical (No Flash Attention - for ablation)")
            print("=" * 80)
            print(f"  use_global_context : {use_global_context}")
            print(f"  use_local_repr  : {use_local_repr}")

            if use_global_context and not use_local_repr:
                print("  Mode 1: Q=[chunk], K/V=[global_context, chunk]")
            elif not use_global_context and use_local_repr:
                print("  Mode 2: Q=[chunk], K/V=[local_i, chunk]")
            elif use_global_context and use_local_repr:
                print("  Mode 3: Q=[chunk], K/V=[global_context, local_i, chunk]")
            else:
                print("  Baseline: Q=K/V=[chunk]")

            print("=" * 80 + "\n", flush=True)

        self._hierarchical_noflash_printed = True

    # ========== Step 1: Chunk split ==========
    group_size = int(q_len * group_size_ratio)
    if q_len % group_size > 0:
        raise ValueError(
            f"q_len {q_len} should be divisible by group size {group_size}."
        )

    if not hasattr(self, "_hierarchical_noflash_group_printed"):
        layer_idx = getattr(self, "layer_idx", 0)
        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank == 0 and layer_idx == 0:
            print(
                f"[forward_noflashattn_hierarchical] group_size_ratio={group_size_ratio}, group_size={group_size}"
            )
        self._hierarchical_noflash_group_printed = True

    num_groups = q_len // group_size

    # Reshape into chunks: [bsz, num_groups, group_size, hidden_size]
    chunks = hidden_states.view(bsz, num_groups, group_size, hidden_size)

    # IMPORTANT: attention_mask format — forward_noflashattn does not replace
    # _prepare_decoder_attention_mask, so it receives HuggingFace's 4D causal mask
    # [bsz, 1, q_len, q_len]. LocalConstructorFlash needs a 2D padding mask [bsz, seq_len].
    # Extract padding mask: a key position j is valid if any query can attend to it (max > -inf).
    if attention_mask is not None and attention_mask.dim() == 4:
        padding_mask_2d = (
            attention_mask[:, 0, :, :].max(dim=-2)[0] > -1e4
        ).long()  # [bsz, q_len], 1=valid, 0=padding
    else:
        # No mask or already 2D — treat all tokens as valid
        padding_mask_2d = torch.ones(
            bsz, q_len, dtype=torch.long, device=hidden_states.device
        )

    # Reshape padding mask for chunks: [bsz, q_len] -> [bsz, num_groups, group_size]
    chunk_masks_reshaped = padding_mask_2d.view(bsz, num_groups, group_size)

    # ========== Step 2: Extract local memories (compress each chunk) ==========
    if (use_global_context or use_local_repr) and hasattr(self, "local_constructor"):
        all_chunks = chunks.view(bsz * num_groups, group_size, hidden_size)
        attention_mask_chunks = chunk_masks_reshaped.view(bsz * num_groups, group_size)
        all_local_repr = self.local_constructor(all_chunks, attention_mask_chunks)

        num_local_slots = all_local_repr.shape[1]
        local_repr_stacked = all_local_repr.view(
            bsz, num_groups, num_local_slots, hidden_size
        )
    else:
        num_local_slots = 0
        local_repr_stacked = None

    # ========== Step 3: Global Integration — aggregate local repr {Li} into global context G ==========
    _causal_mode = CAUSAL_CONTEXT_MODE
    _is_causal = _causal_mode in ("causal_gi", "causal_shift", "causal_shift_g", "causal_gi_gonly")

    if (
        use_global_context
        and hasattr(self, "global_integrator")
        and local_repr_stacked is not None
    ):
        if _is_causal and hasattr(self.global_integrator, "forward_causal"):
            global_context_per_group = self.global_integrator.forward_causal(
                local_repr_stacked
            )
            num_global_slots = global_context_per_group.shape[2]

            if _causal_mode in ("causal_shift", "causal_shift_g"):
                zeros_g = torch.zeros(
                    bsz, 1, num_global_slots, hidden_size,
                    device=global_context_per_group.device,
                    dtype=global_context_per_group.dtype,
                )
                global_context_per_group = torch.cat(
                    [zeros_g, global_context_per_group[:, :-1, :, :]], dim=1
                )

            global_context = None
        else:
            global_context = self.global_integrator(local_repr_stacked)
            num_global_slots = global_context.shape[1]
            global_context_per_group = None
    else:
        global_context = None
        global_context_per_group = None
        num_global_slots = 0

    # causal_shift_g / causal_gi_gonly: use G only, skip L
    if _causal_mode in ("causal_shift_g", "causal_gi_gonly"):
        local_repr_stacked = None
        num_local_slots = 0

    # causal_shift: shift L_i so segment_i uses L_{i-1}
    if (
        _causal_mode == "causal_shift"
        and use_local_repr
        and local_repr_stacked is not None
    ):
        zeros_l = torch.zeros(
            bsz, 1, num_local_slots, hidden_size,
            device=local_repr_stacked.device,
            dtype=local_repr_stacked.dtype,
        )
        local_repr_stacked = torch.cat(
            [zeros_l, local_repr_stacked[:, :-1, :, :]], dim=1
        )

    # ========== Step 4: Standard Q/K/V projections ==========
    if self.config.pretraining_tp > 1:
        key_value_slicing = (
            self.num_key_value_heads * self.head_dim
        ) // self.config.pretraining_tp
        query_slices = self.q_proj.weight.split(
            (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
        )
        key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
        value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

        query_states = [
            F.linear(hidden_states, query_slices[i])
            for i in range(self.config.pretraining_tp)
        ]
        query_states = torch.cat(query_states, dim=-1)

        key_states = [
            F.linear(hidden_states, key_slices[i])
            for i in range(self.config.pretraining_tp)
        ]
        key_states = torch.cat(key_states, dim=-1)

        value_states = [
            F.linear(hidden_states, value_slices[i])
            for i in range(self.config.pretraining_tp)
        ]
        value_states = torch.cat(value_states, dim=-1)
    else:
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

    query_states = query_states.view(
        bsz, q_len, self.num_heads, self.head_dim
    ).transpose(1, 2)
    key_states = key_states.view(
        bsz, q_len, self.num_key_value_heads, self.head_dim
    ).transpose(1, 2)
    value_states = value_states.view(
        bsz, q_len, self.num_key_value_heads, self.head_dim
    ).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]

    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(
        query_states, key_states, cos, sin, position_ids
    )

    # Past Key value support
    if past_key_value is not None:
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)

    past_key_value = (key_states, value_states) if use_cache else None

    # Repeat k/v heads if n_kv_heads < n_heads
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    # ========== Step 5: Project memories to K/V ==========
    global_context_k = global_context_v = None
    global_context_k_per_group = global_context_v_per_group = None

    if use_global_context and global_context is not None:
        global_context_k = (
            self.k_proj(global_context)
            .view(bsz, num_global_slots, self.num_key_value_heads, self.head_dim)
            .transpose(1, 2)
        )
        global_context_v = (
            self.v_proj(global_context)
            .view(bsz, num_global_slots, self.num_key_value_heads, self.head_dim)
            .transpose(1, 2)
        )
        global_context_k = repeat_kv(global_context_k, self.num_key_value_groups)
        global_context_v = repeat_kv(global_context_v, self.num_key_value_groups)
    elif use_global_context and global_context_per_group is not None:
        gc_flat = global_context_per_group.view(
            bsz * num_groups, num_global_slots, hidden_size
        )
        gc_k_flat = (
            self.k_proj(gc_flat)
            .view(bsz * num_groups, num_global_slots, self.num_key_value_heads, self.head_dim)
            .transpose(1, 2)
        )
        gc_v_flat = (
            self.v_proj(gc_flat)
            .view(bsz * num_groups, num_global_slots, self.num_key_value_heads, self.head_dim)
            .transpose(1, 2)
        )
        gc_k_flat = repeat_kv(gc_k_flat, self.num_key_value_groups)
        gc_v_flat = repeat_kv(gc_v_flat, self.num_key_value_groups)
        global_context_k_per_group = gc_k_flat.view(
            bsz, num_groups, self.num_heads, num_global_slots, self.head_dim
        )
        global_context_v_per_group = gc_v_flat.view(
            bsz, num_groups, self.num_heads, num_global_slots, self.head_dim
        )

    # ========== Step 6: Reshape into chunks ==========
    query_chunks = query_states.view(
        bsz, self.num_heads, num_groups, group_size, self.head_dim
    )
    key_chunks = key_states.view(
        bsz, self.num_heads, num_groups, group_size, self.head_dim
    )
    value_chunks = value_states.view(
        bsz, self.num_heads, num_groups, group_size, self.head_dim
    )

    # Local memories K/V projection
    if use_local_repr and local_repr_stacked is not None:
        local_mems_flat = local_repr_stacked.view(
            bsz * num_groups, num_local_slots, hidden_size
        )

        local_k_flat = (
            self.k_proj(local_mems_flat)
            .view(
                bsz * num_groups,
                num_local_slots,
                self.num_key_value_heads,
                self.head_dim,
            )
            .transpose(1, 2)
        )
        local_v_flat = (
            self.v_proj(local_mems_flat)
            .view(
                bsz * num_groups,
                num_local_slots,
                self.num_key_value_heads,
                self.head_dim,
            )
            .transpose(1, 2)
        )

        local_k_flat = repeat_kv(local_k_flat, self.num_key_value_groups)
        local_v_flat = repeat_kv(local_v_flat, self.num_key_value_groups)

        local_k_all = local_k_flat.view(
            bsz, num_groups, self.num_heads, num_local_slots, self.head_dim
        )
        local_v_all = local_v_flat.view(
            bsz, num_groups, self.num_heads, num_local_slots, self.head_dim
        )
    else:
        local_k_all = None
        local_v_all = None

    # ========== Step 7: Build K/V for all chunks (vectorized, mirrors Flash Attention path) ==========
    prefix_len = 0
    if use_global_context and hasattr(self, "global_integrator"):
        prefix_len += num_global_slots
    if use_local_repr and hasattr(self, "local_constructor"):
        prefix_len += num_local_slots
    kv_len_per_chunk = prefix_len + group_size

    if prefix_len > 0:
        # Pre-allocate K/V: [bsz, nh, num_groups, kv_len_per_chunk, hd]
        all_k = torch.empty(
            bsz,
            self.num_heads,
            num_groups,
            kv_len_per_chunk,
            self.head_dim,
            dtype=key_chunks.dtype,
            device=key_chunks.device,
        )
        all_v = torch.empty(
            bsz,
            self.num_heads,
            num_groups,
            kv_len_per_chunk,
            self.head_dim,
            dtype=value_chunks.dtype,
            device=value_chunks.device,
        )

        offset = 0
        # Fill global_context
        if use_global_context and global_context_k is not None:
            # Non-causal: all chunks share the same G
            all_k[:, :, :, offset : offset + num_global_slots, :] = (
                global_context_k.unsqueeze(2)
            )
            all_v[:, :, :, offset : offset + num_global_slots, :] = (
                global_context_v.unsqueeze(2)
            )
            offset += num_global_slots
        elif use_global_context and global_context_k_per_group is not None:
            # Causal: each chunk has its own G_i
            # [bsz, num_groups, nh, global_slots, hd] -> [bsz, nh, num_groups, global_slots, hd]
            all_k[:, :, :, offset : offset + num_global_slots, :] = (
                global_context_k_per_group.permute(0, 2, 1, 3, 4)
            )
            all_v[:, :, :, offset : offset + num_global_slots, :] = (
                global_context_v_per_group.permute(0, 2, 1, 3, 4)
            )
            offset += num_global_slots

        # Fill local memories (different per chunk)
        if use_local_repr and local_k_all is not None:
            # [bsz, num_groups, nh, num_local_slots, hd] -> [bsz, nh, num_groups, num_local_slots, hd]
            all_k[:, :, :, offset : offset + num_local_slots, :] = local_k_all.permute(
                0, 2, 1, 3, 4
            )
            all_v[:, :, :, offset : offset + num_local_slots, :] = local_v_all.permute(
                0, 2, 1, 3, 4
            )
            offset += num_local_slots

        # Fill chunk tokens
        all_k[:, :, :, offset : offset + group_size, :] = key_chunks
        all_v[:, :, :, offset : offset + group_size, :] = value_chunks
    else:
        # No context prefix (G/Li), use chunk K/V directly
        all_k = key_chunks
        all_v = value_chunks
        kv_len_per_chunk = group_size

    # ========== Step 8: Reshape to batched format ==========
    # Q: [bsz, nh, num_groups, group_size, hd] -> [bsz * num_groups, nh, group_size, hd]
    query_states_flat = query_chunks.permute(0, 2, 1, 3, 4).reshape(
        bsz * num_groups, self.num_heads, group_size, self.head_dim
    )

    # K/V: [bsz, nh, num_groups, kv_len, hd] -> [bsz * num_groups, nh, kv_len, hd]
    key_states_flat = all_k.permute(0, 2, 1, 3, 4).reshape(
        bsz * num_groups, self.num_heads, kv_len_per_chunk, self.head_dim
    )
    value_states_flat = all_v.permute(0, 2, 1, 3, 4).reshape(
        bsz * num_groups, self.num_heads, kv_len_per_chunk, self.head_dim
    )

    # ========== Step 9: Manual attention computation ==========
    attn_weights = torch.matmul(
        query_states_flat, key_states_flat.transpose(2, 3)
    ) / math.sqrt(self.head_dim)

    if attn_weights.size() != (
        bsz * num_groups,
        self.num_heads,
        group_size,
        kv_len_per_chunk,
    ):
        raise ValueError(
            f"Attention weights should be of size {(bsz * num_groups, self.num_heads, group_size, kv_len_per_chunk)}, but is"
            f" {attn_weights.size()}"
        )

    # ========== Step 10: Build attention_mask for chunked layout (vectorized, no loops) ==========
    # Input is 4D HuggingFace causal mask [bsz, 1, q_len, q_len].
    # Extract diagonal blocks: [bsz * num_groups, 1, group_size, kv_len_per_chunk]

    # Step 1: Extract diagonal blocks (causal mask for chunk tokens)
    # Reshape: [bsz, 1, q_len, q_len] -> [bsz, 1, num_groups, group_size, num_groups, group_size]
    mask_6d = attention_mask.view(
        bsz, 1, num_groups, group_size, num_groups, group_size
    )

    # torch.diagonal extracts mask_6d[:, :, i, :, i, :] for all i
    # Result: [bsz, 1, group_size, group_size, num_groups]
    diagonal_blocks = torch.diagonal(mask_6d, dim1=2, dim2=4)

    # Permute to [num_groups, bsz, 1, group_size, group_size]
    diagonal_blocks = diagonal_blocks.permute(4, 0, 1, 2, 3)

    # Reshape to [bsz * num_groups, 1, group_size, group_size]
    chunk_masks = diagonal_blocks.reshape(bsz * num_groups, 1, group_size, group_size)

    if prefix_len > 0:
        # Step 2: Create mask for context prefix tokens G/Li (visible to all query tokens)
        # [bsz, 1, q_len, prefix_len]
        hici_mask_cols = torch.zeros(
            bsz,
            1,
            q_len,
            prefix_len,
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )

        # Reshape Q dimension to group: [bsz, 1, num_groups, group_size, prefix_len]
        hici_mask_grouped = hici_mask_cols.view(
            bsz, 1, num_groups, group_size, prefix_len
        )

        # Reshape to [bsz * num_groups, 1, group_size, prefix_len]
        hici_mask_flat = hici_mask_grouped.permute(0, 2, 1, 3, 4).reshape(
            bsz * num_groups, 1, group_size, prefix_len
        )

        # causal_shift/causal_shift_g: segment_0 G (and L) are zero-padded, mask out
        if _causal_mode in ("causal_shift", "causal_shift_g"):
            # Layout: [batch0_group0, batch0_group1, ..., batch1_group0, ...]
            # segment_0 = first group per batch, stride = num_groups
            seg0_indices = torch.arange(0, bsz * num_groups, num_groups,
                                        device=hici_mask_flat.device)
            hici_mask_flat[seg0_indices, :, :, :] = torch.finfo(hici_mask_flat.dtype).min

        # Step 3: Concatenate [memories, chunk]
        # [bsz * num_groups, 1, group_size, prefix_len + group_size]
        attention_mask_expanded = torch.cat([hici_mask_flat, chunk_masks], dim=3)
    else:
        # No context prefix, use chunk masks directly
        attention_mask_expanded = chunk_masks

    # Apply attention mask
    if attention_mask_expanded is not None:
        if attention_mask_expanded.size() != (
            bsz * num_groups,
            1,
            group_size,
            kv_len_per_chunk,
        ):
            raise ValueError(
                f"Attention mask should be of size {(bsz * num_groups, 1, group_size, kv_len_per_chunk)}, but is {attention_mask_expanded.size()}"
            )
        attn_weights = attn_weights + attention_mask_expanded

    # ========== Step 12: Softmax and compute output ==========
    # Upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
        query_states_flat.dtype
    )
    attn_output = torch.matmul(attn_weights, value_states_flat)

    if attn_output.size() != (
        bsz * num_groups,
        self.num_heads,
        group_size,
        self.head_dim,
    ):
        raise ValueError(
            f"`attn_output` should be of size {(bsz * num_groups, self.num_heads, group_size, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    # Transpose: [bsz * num_groups, nh, group_size, hd] -> [bsz * num_groups, group_size, nh, hd]
    attn_output = attn_output.transpose(1, 2).contiguous()

    # Reshape back: [bsz * num_groups, group_size, nh, hd] -> [bsz, q_len, nh, hd]
    attn_output = attn_output.reshape(bsz, q_len, self.num_heads, self.head_dim)

    # Final reshape: [bsz, q_len, nh, hd] -> [bsz, q_len, hidden_size]
    attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim)

    # ========== Step 13: Output projection ==========
    if self.config.pretraining_tp > 1:
        attn_output = attn_output.split(
            self.hidden_size // self.config.pretraining_tp, dim=2
        )
        o_proj_slices = self.o_proj.weight.split(
            self.hidden_size // self.config.pretraining_tp, dim=1
        )
        attn_output = sum(
            [
                F.linear(attn_output[i], o_proj_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
        )
    else:
        attn_output = self.o_proj(attn_output)

    return attn_output, None, past_key_value


# Evaluation helpers — full attention variants (no chunking)
# region
def forward_flashattn_full(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    padding_mask: Optional[torch.LongTensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """Input shape: Batch x Time x Channel

    attention_mask: [bsz, q_len]
    """
    if output_attentions:
        warnings.warn(
            "Output attentions is not supported for patched `LlamaAttention`, returning `None` instead."
        )

    bsz, q_len, _ = hidden_states.size()

    query_states = (
        self.q_proj(hidden_states)
        .view(bsz, q_len, self.num_heads, self.head_dim)
        .transpose(1, 2)
    )
    key_states = (
        self.k_proj(hidden_states)
        .view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        .transpose(1, 2)
    )
    value_states = (
        self.v_proj(hidden_states)
        .view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        .transpose(1, 2)
    )
    # [bsz, q_len, nh, hd]
    # [bsz, nh, q_len, hd]

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]
    # (Unused) Use position_ids max to set RoPE cache length for absolute positions:
    # if position_ids is not None:
    #     max_pos = position_ids.max().item() + 1
    #     rope_seq_len = max(kv_seq_len, max_pos)
    # else:
    #     rope_seq_len = kv_seq_len
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(
        query_states, key_states, cos, sin, position_ids
    )

    # Past Key value support
    if past_key_value is not None:
        # reuse k, v, self_attention
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)

    past_key_value = (key_states, value_states) if use_cache else None

    # repeat k/v heads if n_kv_heads < n_heads
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    # Flash attention codes from
    # https://github.com/HazyResearch/flash-attention/blob/main/flash_attn/flash_attention.py

    # transform the data into the format required by flash attention
    qkv = torch.stack(
        [query_states, key_states, value_states], dim=2
    )  # [bsz, nh, 3, q_len, hd]
    qkv = qkv.transpose(1, 3)  # [bsz, q_len, 3, nh, hd]

    # We have disabled _prepare_decoder_attention_mask in LlamaModel
    # the attention_mask should be the same as the key_padding_mask

    key_padding_mask = attention_mask
    nheads = qkv.shape[-2]
    x = rearrange(qkv, "b s three h d -> b s (three h d)")
    x_unpad, indices, cu_q_lens, max_s = unpad_input(x, key_padding_mask)
    x_unpad = rearrange(x_unpad, "nnz (three h d) -> nnz three h d", three=3, h=nheads)
    output_unpad = flash_attn_varlen_qkvpacked_func(
        x_unpad, cu_q_lens, max_s, 0.0, softmax_scale=None, causal=True
    )
    output = rearrange(
        pad_input(rearrange(output_unpad, "nnz h d -> nnz (h d)"), indices, bsz, q_len),
        "b s (h d) -> b s h d",
        h=nheads,
    )
    output = output.reshape(bsz, q_len, self.num_heads, self.head_dim)

    return self.o_proj(rearrange(output, "b s h d -> b s (h d)")), None, past_key_value


# endregion
def forward_noflashattn(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    padding_mask: Optional[torch.LongTensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()

    group_size = int(q_len * group_size_ratio)

    if q_len % group_size > 0:
        raise ValueError(
            "q_len %d should be divisible by group size %d." % (q_len, group_size)
        )
    num_group = q_len // group_size

    if self.config.pretraining_tp > 1:
        key_value_slicing = (
            self.num_key_value_heads * self.head_dim
        ) // self.config.pretraining_tp
        query_slices = self.q_proj.weight.split(
            (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
        )
        key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
        value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

        query_states = [
            F.linear(hidden_states, query_slices[i])
            for i in range(self.config.pretraining_tp)
        ]
        query_states = torch.cat(query_states, dim=-1)

        key_states = [
            F.linear(hidden_states, key_slices[i])
            for i in range(self.config.pretraining_tp)
        ]
        key_states = torch.cat(key_states, dim=-1)

        value_states = [
            F.linear(hidden_states, value_slices[i])
            for i in range(self.config.pretraining_tp)
        ]
        value_states = torch.cat(value_states, dim=-1)

    else:
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

    query_states = query_states.view(
        bsz, q_len, self.num_heads, self.head_dim
    ).transpose(1, 2)
    key_states = key_states.view(
        bsz, q_len, self.num_key_value_heads, self.head_dim
    ).transpose(1, 2)
    value_states = value_states.view(
        bsz, q_len, self.num_key_value_heads, self.head_dim
    ).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(
        query_states, key_states, cos, sin, position_ids
    )

    if past_key_value is not None:
        # reuse k, v, self_attention
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)

    past_key_value = (key_states, value_states) if use_cache else None

    # repeat k/v heads if n_kv_heads < n_heads
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    # shift
    def shift(qkv, bsz, q_len, group_size, num_heads, head_dim):
        qkv[:, num_heads // 2 :] = qkv[:, num_heads // 2 :].roll(
            -group_size // 2, dims=2
        )
        qkv = (
            qkv.transpose(1, 2)
            .reshape(bsz * (q_len // group_size), group_size, num_heads, head_dim)
            .transpose(1, 2)
        )
        return qkv

    query_states = shift(
        query_states, bsz, q_len, group_size, self.num_heads, self.head_dim
    )
    key_states = shift(
        key_states, bsz, q_len, group_size, self.num_heads, self.head_dim
    )
    value_states = shift(
        value_states, bsz, q_len, group_size, self.num_heads, self.head_dim
    )

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(
        self.head_dim
    )

    if attn_weights.size() != (bsz * num_group, self.num_heads, group_size, group_size):
        raise ValueError(
            f"Attention weights should be of size {(bsz * num_group, self.num_heads, group_size, group_size)}, but is"
            f" {attn_weights.size()}"
        )

    attention_mask = attention_mask[:, :, :group_size, :group_size].repeat(
        num_group, 1, 1, 1
    )
    if attention_mask is not None:
        if attention_mask.size() != (bsz * num_group, 1, group_size, group_size):
            raise ValueError(
                f"Attention mask should be of size {(bsz * num_group, 1, group_size, group_size)}, but is {attention_mask.size()}"
            )
        attn_weights = attn_weights + attention_mask

    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
        query_states.dtype
    )
    attn_output = torch.matmul(attn_weights, value_states)

    if attn_output.size() != (
        bsz * num_group,
        self.num_heads,
        group_size,
        self.head_dim,
    ):
        raise ValueError(
            f"`attn_output` should be of size {(bsz * num_group, self.num_heads, group_size, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )
    attn_output = attn_output.transpose(1, 2).contiguous()

    attn_output = attn_output.reshape(bsz, q_len, self.num_heads, self.head_dim)

    # shift back
    attn_output[:, :, self.num_heads // 2 :] = attn_output[
        :, :, self.num_heads // 2 :
    ].roll(group_size // 2, dims=1)

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    if self.config.pretraining_tp > 1:
        attn_output = attn_output.split(
            self.hidden_size // self.config.pretraining_tp, dim=2
        )
        o_proj_slices = self.o_proj.weight.split(
            self.hidden_size // self.config.pretraining_tp, dim=1
        )
        attn_output = sum(
            [
                F.linear(attn_output[i], o_proj_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
        )
    else:
        attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


# Disable the transformation of the attention mask in LlamaModel as the flash attention
# requires the attention mask to be the same as the key_padding_mask
def _prepare_decoder_attention_mask(
    self, attention_mask, input_shape, inputs_embeds, past_key_values_length
):
    # [bsz, seq_len]
    return attention_mask


def apply_rotary_pos_emb_inference(q, k, cos_sin, position_ids):
    gather_indices = position_ids[:, :, None, None]  # [bsz, seq_len, 1, 1]
    gather_indices = gather_indices.repeat(
        1, 1, cos_sin[0].shape[1], cos_sin[0].shape[3]
    )
    bsz = gather_indices.shape[0]
    cos, sin = (
        torch.gather(x.transpose(1, 2).repeat(bsz, 1, 1, 1), 1, gather_indices)
        for x in cos_sin
    )
    q, k = ((x * cos) + (rotate_half(x) * sin) for x in (q, k))
    return q, k


def forward_flashattn_inference(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    padding_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    if output_attentions:
        warnings.warn(
            "Output attentions is not supported for patched `LlamaAttention`, returning `None` instead."
        )

    bsz, q_len, _ = hidden_states.size()
    kv_heads = getattr(self, "num_key_value_heads", self.num_heads)

    q, k, v = (
        op(hidden_states).view(bsz, q_len, nh, self.head_dim)
        for op, nh in (
            (self.q_proj, self.num_heads),
            (self.k_proj, kv_heads),
            (self.v_proj, kv_heads),
        )
    )
    # shape: (b, s, num_heads, head_dim)

    kv_seq_len = k.shape[1]
    past_kv_len = 0
    if past_key_value is not None:
        past_kv_len = past_key_value[0].shape[2]
        kv_seq_len += past_kv_len

    cos_sin = self.rotary_emb(v, seq_len=kv_seq_len)
    q, k = apply_rotary_pos_emb_inference(q, k, cos_sin, position_ids)

    if past_key_value is not None:
        assert flash_attn_version >= "2.1.0", (
            "past_key_value support requires flash-attn >= 2.1.0"
        )
        # reuse k, v
        k = torch.cat([past_key_value[0].transpose(1, 2), k], dim=1)
        v = torch.cat([past_key_value[1].transpose(1, 2), v], dim=1)

    past_key_value = (k.transpose(1, 2), v.transpose(1, 2)) if use_cache else None

    if attention_mask is None:
        output = flash_attn_func(q, k, v, 0.0, softmax_scale=None, causal=True).view(
            bsz, q_len, -1
        )
    else:
        q, indices, cu_q_lens, max_s = unpad_input(q, attention_mask[:, -q_len:])
        # We can skip concat and call unpad twice but seems better to call unpad only once.
        kv, _, cu_k_lens, max_k = unpad_input(
            torch.stack((k, v), dim=2), attention_mask
        )
        output_unpad = flash_attn_varlen_kvpacked_func(
            q,
            kv,
            cu_q_lens,
            cu_k_lens,
            max_s,
            max_k,
            0.0,
            softmax_scale=None,
            causal=True,
        )
        output_unpad = output_unpad.reshape(-1, self.num_heads * self.head_dim)
        output = pad_input(output_unpad, indices, bsz, q_len)

    return self.o_proj(output), None, past_key_value


def _prepare_decoder_attention_mask_inference(
    self, attention_mask, input_shape, inputs_embeds, past_key_values_length
):
    # [bsz, seq_len]
    if past_key_values_length > 0 and attention_mask is not None:
        attention_mask = torch.cat(
            (
                torch.full(
                    (input_shape[0], past_key_values_length),
                    True,
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                ),
                attention_mask,
            ),
            dim=-1,
        )

    if attention_mask is not None and torch.all(attention_mask):
        return None  # This uses the faster call when training with full samples

    return attention_mask


def replace_llama_attn(
    use_flash_attn=True,
    use_full=False,
    inference=False,
    eval_mode=None,
    use_hierarchical_forward: Optional[bool] = False,
):
    """
    Replace LlamaAttention forward function with HiCI implementations.

    IMPORTANT: This function only patches the forward method.
    To register LocalConstructor parameters, call register_hici_to_model()
    after loading the model and before initializing the optimizer.

    Args:
        use_flash_attn: Whether to use flash attention (default: True)
        use_full: Kept for backward compatibility, has no effect. Use eval_mode instead.
        inference: Whether in inference mode (default: False)
        eval_mode: Evaluation mode selection (default: None)
            - None: Chunked HiCI attention (same as training)
            - "full": Full attention without HiCI
        use_hierarchical_forward: Use forward_flashattn_hierarchical (LocalConstructor + GlobalIntegrator)
    """
    rank = dist.get_rank() if dist.is_initialized() else 0
    if use_flash_attn:
        if rank == 0:
            warnings.warn(
                "Flash attention is only supported on A100 or H100 GPU during training due to head dim > 64 backward."
                "ref: https://github.com/HazyResearch/flash-attention/issues/190#issuecomment-1523359593"
            )
        if inference:
            transformers.models.llama.modeling_llama.LlamaModel._prepare_decoder_attention_mask = _prepare_decoder_attention_mask_inference
            transformers.models.llama.modeling_llama.LlamaAttention.forward = (
                forward_flashattn_inference
            )
        else:
            transformers.models.llama.modeling_llama.LlamaModel._prepare_decoder_attention_mask = _prepare_decoder_attention_mask

            # Select attention function based on eval_mode or use_full
            if eval_mode == "full":
                # eval_mode="full": Full Attention without HiCI
                transformers.models.llama.modeling_llama.LlamaAttention.forward = (
                    forward_flashattn_full
                )
                if rank == 0:
                    print(f"   Attention fn: forward_flashattn_full (full attention eval)")
            else:
                # Default: HiCI chunked attention
                if use_hierarchical_forward:
                    # Select training vs. inference variant based on USE_FIXED_SEGMENT_SIZE
                    if USE_FIXED_SEGMENT_SIZE:
                        # Inference: padding-aware hierarchical variant
                        transformers.models.llama.modeling_llama.LlamaAttention.forward = forward_flashattn_hierarchical_inference
                        if rank == 0:
                            print(
                                f"   🎯 Using forward_flashattn_hierarchical_inference (with padding, segment_size={FIXED_SEGMENT_SIZE})"
                            )
                    else:
                        # Training: standard hierarchical variant
                        transformers.models.llama.modeling_llama.LlamaAttention.forward = forward_flashattn_hierarchical
                        if rank == 0:
                            print(
                                "   🧪 Using forward_flashattn_hierarchical (training mode)"
                            )
                else:
                    transformers.models.llama.modeling_llama.LlamaAttention.forward = (
                        forward_flashattn
                    )
                    if rank == 0:
                        print(f"   Attention fn: forward_flashattn (LongLoRA baseline)")
    else:
        transformers.models.llama.modeling_llama.LlamaAttention.forward = (
            forward_noflashattn
        )


def register_hici_to_model(
    model,
    num_local_slots=8,
    global_slots=4,
    num_heads=8,
    use_bottleneck=True,
    bottleneck_dim=512,
    use_local_constructor=True,
    use_global_integrator=True,
    use_local_constructor_flash: Optional[bool] = False,
    use_llama_init=False,  # Init Q/K/V from LLaMA pretrained weights (option C)
    use_shared_compressor=True,  # Use GlobalIntegratorShared (saves 71% params vs. original)
    compress_dim=512,  # Bottleneck dim for GlobalIntegrator (13B: 640, 10 heads)
    shared_compress_dim=128,  # Shared compressor dim (7B: 128, 13B: 160)
    ds_config_path=None,  # DeepSpeed config path for ZeRO-3 parameter sharding
):
    """
    Register HiCI modules (LocalConstructor, GlobalIntegrator) to each LlamaAttention layer.

    This MUST be called after model loading and before optimizer initialization!

    Args:
        model: LlamaForCausalLM or PeftModelForCausalLM
        num_local_slots: Number of Local Representation Slots (for LocalConstructor, default: 8)
        global_slots: Number of global context vectors (for GlobalIntegrator, default: 4)
        num_heads: Number of attention heads (default: 32)
        bottleneck_dim: Bottleneck dimension for efficiency (default: 2048)
        use_global_integrator: If True, also register GlobalIntegrator (default: False)
        use_llama_init: If True, initialize Q/K/V projections from LLaMA pretrained weights.
            Aligns HiCI projections with the pretrained semantic space for faster convergence.
        use_shared_compressor: If True, use GlobalIntegratorShared (saves 71% params vs. original).
            GlobalIntegrator: 13.7M/layer; GlobalIntegratorShared: 4.0M/layer.
        shared_compress_dim: Shared compressor intermediate dim (only when use_shared_compressor=True, default 128).

    Example usage in fine-tune.py:
        # 1. Load model
        model = transformers.AutoModelForCausalLM.from_pretrained(...)

        # 2. Replace attention mechanism
        replace_llama_attn(use_flash_attn=True)

        # 3. Register HiCI modules (BEFORE optimizer!)
        # Local Construction only:
        register_hici_to_model(model, num_local_slots=8)

        # Full HiCI (Local Construction + Global Integration):
        register_hici_to_model(
            model,
            num_local_slots=8,   # M slot vectors per segment
            global_slots=4,      # K global context vectors
            use_global_integrator=True
        )

        # 4. Setup LoRA (if needed)
        model = get_peft_model(model, lora_config)

        # 5. NOW initialize optimizer (will include HiCI parameters)
        optimizer = torch.optim.AdamW(model.parameters(), lr=...)
    """
    rank = dist.get_rank() if dist.is_initialized() else 0

    # ⚠️ Validate configuration
    if use_global_integrator and not use_local_constructor:
        if rank == 0:
            print("\n" + "=" * 80)
            print("❌ ERROR: Invalid Configuration!")
            print("=" * 80)
            print("use_global_integrator=True requires use_local_constructor=True")
            print(
                "Reason: GlobalIntegrator needs local memories from LocalConstructor"
            )
            print()
            print("Fix: Set use_local_constructor=True, or set use_global_integrator=False")
            print("=" * 80 + "\n")
        raise ValueError(
            "Invalid configuration: use_global_integrator=True requires use_local_constructor=True. "
            "GlobalIntegrator needs local memories from LocalConstructor to aggregate."
        )

    if rank == 0:
        print("\n" + "=" * 80)
        config_str = []
        if use_local_constructor:
            config_str.append("LocalConstructor")
        if use_global_integrator:
            config_str.append("Hierarchical Aggregator")

        if config_str:
            print(f"🔧 Registering: {' + '.join(config_str)}")
        else:
            print("⚠️ No HiCI modules enabled!")

    # Navigate to base LlamaModel
    if hasattr(model, "base_model"):
        # PeftModelForCausalLM
        base_model = model.base_model
        if hasattr(base_model, "model"):
            # PeftModel wraps LlamaForCausalLM
            llama_model = base_model.model.model
        else:
            llama_model = base_model
    else:
        # LlamaForCausalLM
        llama_model = model.model

    # Get hidden_size from first layer
    hidden_size = llama_model.layers[0].self_attn.hidden_size

    model_dtype = llama_model.embed_tokens.weight.dtype
    if rank == 0:
        print(f"   Model dtype: {model_dtype}")

    embed_weight = llama_model.embed_tokens.weight.data  # [vocab_size, hidden_size]

    # Auto-detect ZeRO-3: from env var or explicit ds_config_path
    use_zero3_init = False
    zero3_init_context = None
    detected_ds_config = ds_config_path

    if detected_ds_config is None:
        detected_ds_config = os.environ.get("DEEPSPEED_CONFIG_FILE", None)

    if detected_ds_config is not None and os.path.isfile(detected_ds_config):
        try:
            with open(detected_ds_config, "r") as f:
                ds_config = json.load(f)
            zero_stage = ds_config.get("zero_optimization", {}).get("stage", 0)
            if zero_stage == 3:
                use_zero3_init = True
                import deepspeed

                if rank == 0:
                    print(f"   🔧 ZeRO-3 detected (config: {detected_ds_config})")
                    print(
                        f"   🔧 Using deepspeed.zero.Init() for HiCI module sharding"
                    )
        except Exception as e:
            if rank == 0:
                print(f"   ⚠️ Failed to load DeepSpeed config: {e}")

    # Register modules to each attention layer
    # Use ZeRO-3 Init context manager so new parameters are correctly sharded
    if use_zero3_init:
        import deepspeed
        import copy

        zero3_config = copy.deepcopy(ds_config)

        # Replace "auto" batch size placeholders required by deepspeed.zero.Init
        import torch.distributed as torch_dist

        world_size = torch_dist.get_world_size() if torch_dist.is_initialized() else 1

        if zero3_config.get("train_batch_size") == "auto":
            zero3_config["train_batch_size"] = world_size  # 1 * 1 * world_size
        if zero3_config.get("train_micro_batch_size_per_gpu") == "auto":
            zero3_config["train_micro_batch_size_per_gpu"] = 1
        if zero3_config.get("gradient_accumulation_steps") == "auto":
            zero3_config["gradient_accumulation_steps"] = 1

        zero3_init_context = deepspeed.zero.Init(config_dict_or_path=zero3_config)
        zero3_init_context.__enter__()

    for layer_idx, layer in enumerate(llama_model.layers):
        attn = layer.self_attn
        attn.layer_idx = layer_idx  # Important for layer identification

        # Module 1: LocalConstructor
        if use_local_constructor:
            if use_local_constructor_flash:
                attn.local_constructor = LocalConstructorFlash(
                    hidden_size=hidden_size,
                    num_local_slots=num_local_slots,
                    num_heads=num_heads,
                    init_from_embeddings=embed_weight,
                    init_from_attn=attn if use_llama_init else None,
                    use_bottleneck=use_bottleneck,
                    bottleneck_dim=bottleneck_dim,
                ).to(model_dtype)
            else:
                # Default: LocalConstructorMulti (pure-PyTorch multi-head cross-attention)
                # attn.local_constructor = LocalConstructor(
                #     hidden_size=hidden_size,
                #     num_local_slots=num_local_slots,
                #     num_heads=num_heads,
                #     init_from_embeddings=embed_weight,
                # ).to(model_dtype)
                attn.local_constructor = LocalConstructorMulti(
                    hidden_size=hidden_size,
                    num_local_slots=num_local_slots,
                    num_heads=num_heads,
                    init_from_embeddings=embed_weight,
                    init_from_attn=attn if use_llama_init else None,
                    use_bottleneck=use_bottleneck,
                    bottleneck_dim=bottleneck_dim,
                ).to(model_dtype)

        # Module 2: GlobalIntegrator (optional)
        if use_global_integrator:
            if use_shared_compressor:
                # GlobalIntegratorShared: shared compression layer saves ~71% params (4.0M vs 13.7M/layer)
                attn.global_integrator = GlobalIntegratorShared(
                    hidden_size=hidden_size,
                    global_slots=global_slots,
                    compress_dim=bottleneck_dim,
                    shared_compress_dim=shared_compress_dim,
                    num_heads=num_heads,
                    # num_heads=8,  # Fix to 8 heads to reduce params
                    init_from_embeddings=embed_weight,
                    use_high_norm_init=True,
                ).to(model_dtype)
            else:
                # Original GlobalIntegrator: 13.7M/layer
                attn.global_integrator = GlobalIntegrator(
                    hidden_size=hidden_size,
                    global_slots=global_slots,
                    compress_dim=compress_dim,
                    num_heads=num_heads,
                    init_from_embeddings=embed_weight,
                    use_high_norm_init=True,
                ).to(model_dtype)

    if use_zero3_init and zero3_init_context is not None:
        zero3_init_context.__exit__(None, None, None)
        if rank == 0:
            print(f"   ✅ ZeRO-3 HiCI module sharding complete")

    # Verify registration
    total_params = sum(p.numel() for p in model.parameters())

    local_constructor_params = 0
    aggregator_params = 0

    if use_local_constructor or use_global_integrator:
        for name, param in model.named_parameters():
            if "local_constructor" in name:
                local_constructor_params += param.numel()
            elif "global_integrator" in name:
                aggregator_params += param.numel()

    if rank == 0:
        print()
        print("=" * 80)
        print("✅ HiCI Module Registration Complete")
        print("=" * 80)

        print(f"Model: {total_params:,} params ({total_params / 1e9:.2f}B)")
        print(f"Layers: {len(llama_model.layers)}")

        if use_local_constructor and use_global_integrator:
            total_hici_params = local_constructor_params + aggregator_params
            print(f"\nRegistered Modules:")
            print(f"  ✓ LocalConstructor ({local_constructor_params:,} params)")
            print(f"  ✓ Hierarchical Aggregator ({aggregator_params:,} params)")
            print(
                f"\nTotal HiCI Params: {total_hici_params:,} ({total_hici_params / total_params * 100:.2f}%)"
            )

        elif use_local_constructor and not use_global_integrator:
            print(f"\nRegistered Modules:")
            print(f"  ✓ LocalConstructor ({local_constructor_params:,} params)")
            print(
                f"\nTotal HiCI Params: {local_constructor_params:,} ({local_constructor_params / total_params * 100:.2f}%)"
            )

        elif not use_local_constructor and use_global_integrator:
            print(f"\n⚠️ Warning: GlobalIntegrator registered without LocalConstructor!")
            print(f"  ✓ Hierarchical Aggregator ({aggregator_params:,} params)")
            print(
                f"\nTotal HiCI Params: {aggregator_params:,} ({aggregator_params / total_params * 100:.2f}%)"
            )

        else:
            print(f"\nRegistered Modules: None")

        print("=" * 80 + "\n")


# def register_hici_to_model(
#     model,
#     num_local_slots=8,
#     recurrence_size=256
# ):
#     # region
#     """
#     Register LocalConstructor to each LlamaAttention layer.

#     This MUST be called after model loading and before optimizer initialization!

#     Args:
#         model: LlamaForCausalLM or PeftModelForCausalLM
#         num_local_slots: Number of learnable memory slots per layer (default: 8)
#         recurrence_size: Number of tokens to carry from previous chunk (default: 256)

#     Example usage in fine-tune.py:
#         # 1. Load model
#         model = transformers.AutoModelForCausalLM.from_pretrained(...)

#         # 2. Replace attention mechanism
#         replace_llama_attn(use_flash_attn=True, use_full=False)

#         # 3. Register global memory (BEFORE optimizer!)
#         register_hici_to_model(model, num_local_slots=8, recurrence_size=256)

#         # 4. Setup LoRA (if needed)
#         model = get_peft_model(model, lora_config)

#         # 5. NOW initialize optimizer (will include global memory parameters)
#         optimizer = torch.optim.AdamW(model.parameters(), lr=...)
#     """
#     #endregion
#     # Navigate to base LlamaModel
#     if hasattr(model, 'base_model'):
#         # PeftModelForCausalLM
#         base_model = model.base_model
#         if hasattr(base_model, 'model'):
#             # PeftModel wraps LlamaForCausalLM
#             llama_model = base_model.model.model
#         else:
#             llama_model = base_model
#     else:
#         # LlamaForCausalLM
#         llama_model = model.model

#     # Get hidden_size from first layer
#     hidden_size = llama_model.layers[0].self_attn.hidden_size

#     # Register LocalConstructor to each attention layer
#     for layer_idx, layer in enumerate(llama_model.layers):
#         attn = layer.self_attn

#         # Create and register LocalConstructor as a sub-module
#         attn.local_constructor = LocalConstructor(
#             hidden_size=hidden_size,
#             num_local_slots=num_local_slots
#         )

#         # Store recurrence_size as attribute (used in forward_flashattn_varlen_grouped)
#         attn.recurrence_size = recurrence_size

#         print(f"Registered LocalConstructor to layer {layer_idx} "
#               f"({num_local_slots} slots × {hidden_size}D, recurrence_size={recurrence_size})")

#     # Verify registration
#     total_params = sum(p.numel() for p in model.parameters())
#     memory_params = sum(
#         p.numel() for layer in llama_model.layers
#         for p in layer.self_attn.local_constructor.parameters()
#     )

#     print(f"\n✅ Global memory registration complete!")
#     print(f"   Total model params: {total_params:,}")
#     print(f"   Global memory params: {memory_params:,} "
#           f"({memory_params/total_params*100:.2f}% of total)")
#     print(f"   Per-layer config: {num_local_slots} slots × {hidden_size}D = "
#           f"{num_local_slots * hidden_size * 4:,} params × {len(llama_model.layers)} layers")
#     print(f"   Recurrence size: {recurrence_size} tokens (carried from previous chunk)\n")
