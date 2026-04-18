# Qwen3 HiCI (Hierarchical Construction-Integration) Attention
# Adapted from qwen_attn_hici.py for Qwen3 + transformers >= 4.51
#
# Key differences from Qwen2 (qwen_attn_hici.py) version:
# 1. Imports from transformers.models.qwen3 instead of qwen2
# 2. Qwen3Attention has bias=False (config-driven), Qwen2 has bias=True (hardcoded)
# 3. Qwen3 has QK-Norm (RMSNorm on Q/K after projection, before RoPE)
# 4. New transformers 4.51+ API:
#    - forward signature: (hidden_states, position_embeddings, attention_mask, ...)
#    - RoPE pre-computed by Qwen3Model, passed as position_embeddings=(cos, sin)
#    - self.rotary_emb moved from Attention to Model level
#    - self.num_heads etc. moved to self.config.num_attention_heads
#    - _prepare_decoder_attention_mask → _update_causal_mask
# 5. Requires: transformers >= 4.51, torch >= 2.4, flash-attn >= 2.5

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

from transformers.models.qwen3.modeling_qwen3 import (
    apply_rotary_pos_emb,
    repeat_kv,
    rotate_half,
    Qwen3RMSNorm,
)

import math
import random
import os
import json

# ============================================================================
# Mixed-group training configuration
# ============================================================================
MIXED_GROUP_TRAINING = False
GROUP_SIZE_RATIOS = [1 / 2, 1 / 4, 1 / 8]  # corresponding to 2, 4, 8 groups

group_size_ratio = 1 / 4  # default (used when MIXED_GROUP_TRAINING=False)

# ============================================================================
# Fixed segment-size mode (for evaluation; matches training segment size)
# ============================================================================
USE_FIXED_SEGMENT_SIZE = False
FIXED_SEGMENT_SIZE = 1024  # tokens per segment

# ============================================================================
# Causal context mode
# ============================================================================
# "none"        - default: all segments share one G (non-causal)
# "causal_gi"   - option A: segment_i uses G_i=Agg(L_1..L_i) and L_i
#                 G is causal; L_i has bounded intra-segment leakage (bottleneck compression)
# "causal_shift"- option B: segment_i uses G_{i-1}=Agg(L_1..L_{i-1}) and L_{i-1}
#                 strictly causal, zero leakage; segment_1 has no G or L
CAUSAL_CONTEXT_MODE = "causal_gi"

# ============================================================================
# Full attention + HiCI mode
# ============================================================================
USE_FULL_ATTN_WITH_HICI = True

# Global state: all layers share the same grouping within one forward pass
_mixed_group_current_ratio = None
_mixed_group_call_count = 0
rank = dist.get_rank() if dist.is_initialized() else 0

# ============================================================================
# Cache fill mode for testing
# ============================================================================
CACHE_FILL_MODE = "zeros"

# ============================================================================
# Attention visualization configuration (inference only)
# ============================================================================
COLLECT_ATTENTION_FOR_VIZ = False

attention_visualizer = {
    "enabled": False,
    "layer_attn_to_global": [],
    "layer_attn_to_local": [],
    "layer_attn_to_tokens": [],
    "segment_attention_maps": [],
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
    stats = {
        "num_global_slots": attention_visualizer["num_global_slots"],
        "num_local_slots": attention_visualizer["num_local_slots"],
        "segment_len": attention_visualizer["segment_len"],
        "layer_attn_to_global": attention_visualizer["layer_attn_to_global"],
        "layer_attn_to_local": attention_visualizer["layer_attn_to_local"],
        "layer_attn_to_tokens": attention_visualizer["layer_attn_to_tokens"],
        "segment_attention_maps": attention_visualizer.get("segment_attention_maps", []),
    }
    with open(save_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"✅ Attention stats saved to {save_path}")


# ============================================================================
# LocalConstructorMulti — Local Construction module (multi-head, standard PyTorch)
# ============================================================================
# Architecture-agnostic: depends only on hidden_size and num_heads.
# Consistent with LocalConstructorMulti in llama_attn_hici.py.
class LocalConstructorMulti(nn.Module):
    """
    Learnable query slots for capturing document-level context.

    Multi-head cross-attention implementation (no Flash Attention). Supports:
    1. Multi-head attention — improved expressiveness
    2. Attention mask — correct handling of padding tokens
    3. Bottleneck compression — optional information bottleneck
    4. Weight initialization — warm-start from pretrained weights

    Args:
        hidden_size: Model hidden dimension (e.g., 4096)
        num_local_slots: Number of learnable query slots (default: 8)
        num_heads: Number of attention heads (default: 8)
        init_from_embeddings: Optional pretrained embeddings for memory_slots initialization
        init_from_attn: Optional Attention layer for Q/K/V projection initialization
        use_bottleneck: Whether to use bottleneck compression (default: True)
        bottleneck_dim: Bottleneck dimension (default: 512)
    """

    # Class-level flag: print initialization info only once
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

        # Learnable memory_slots (query): [num_slots, hidden_size]
        std = 1.0 / math.sqrt(hidden_size)
        self.memory_slots = nn.Parameter(
            torch.randn(num_local_slots, hidden_size) * std
        )
        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank == 0 and not LocalConstructorMulti._init_msg_printed:
            print(
                f"⚠️  LocalConstructorMulti Fallback: Initialized memory_slots with std={std}"
            )
            LocalConstructorMulti._init_msg_printed = True

        # Cross-attention projections with optional bottleneck
        if use_bottleneck:
            assert bottleneck_dim % num_heads == 0, (
                f"bottleneck_dim ({bottleneck_dim}) must be divisible by num_heads ({num_heads})"
            )
            rank = dist.get_rank() if dist.is_initialized() else 0
            if rank == 0:
                print(
                    f"✅ LocalConstructorMulti: bottleneck_dim: {bottleneck_dim}, num_heads: {num_heads}"
                )

            self.q_proj = nn.Linear(hidden_size, bottleneck_dim, bias=False)
            self.k_proj = nn.Linear(hidden_size, bottleneck_dim, bias=False)
            self.v_proj = nn.Linear(hidden_size, bottleneck_dim, bias=False)
            self.o_proj = nn.Linear(bottleneck_dim, hidden_size, bias=False)

            self.effective_dim = bottleneck_dim
            self.effective_head_dim = bottleneck_dim // num_heads
        else:
            self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
            self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
            self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
            self.o_proj = None

            self.effective_dim = hidden_size
            self.effective_head_dim = self.head_dim

        # Warm initialization from pretrained attention weights
        if init_from_attn is not None and not use_bottleneck:
            rank = dist.get_rank() if dist.is_initialized() else 0
            with torch.no_grad():
                self.q_proj.weight.copy_(init_from_attn.q_proj.weight)
                self.k_proj.weight.copy_(init_from_attn.k_proj.weight)
                self.v_proj.weight.copy_(init_from_attn.v_proj.weight)
            if rank == 0:
                print(
                    f"✅ [LocalConstructorMulti] Initialized Q/K/V projections from pretrained weights"
                )

    def forward(self, hidden_states, attention_mask=None):
        """
        Compute global context via multi-head cross-attention (standard PyTorch, no Flash Attention).

        Args:
            hidden_states: [bsz, seq_len, hidden_size]
            attention_mask: [bsz, seq_len] - 1 for valid tokens, 0 for padding

        Returns:
            global_context: [bsz, num_slots, hidden_size]
        """
        bsz, seq_len, _ = hidden_states.shape

        slots_expanded = self.memory_slots.unsqueeze(0).expand(bsz, -1, -1)

        Q_slots = self.q_proj(slots_expanded)
        K_seq = self.k_proj(hidden_states)
        V_seq = self.v_proj(hidden_states)

        Q_slots = Q_slots.view(bsz, self.num_local_slots, self.num_heads, self.effective_head_dim)
        K_seq = K_seq.view(bsz, seq_len, self.num_heads, self.effective_head_dim)
        V_seq = V_seq.view(bsz, seq_len, self.num_heads, self.effective_head_dim)

        # Transpose for attention: [bsz, num_heads, seqlen, head_dim]
        Q_slots = Q_slots.transpose(1, 2)
        K_seq = K_seq.transpose(1, 2)
        V_seq = V_seq.transpose(1, 2)

        # Compute attention scores: Q @ K^T
        scores = torch.matmul(Q_slots, K_seq.transpose(-2, -1)) / math.sqrt(
            self.effective_head_dim
        )

        # Apply attention mask if provided
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(1).unsqueeze(2)
            mask_value = torch.finfo(scores.dtype).min
            scores = scores.masked_fill(mask_expanded == 0, mask_value)

        attn_weights = torch.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V_seq)

        attn_output = attn_output.transpose(1, 2).contiguous()
        global_context = attn_output.view(bsz, self.num_local_slots, self.effective_dim)

        if self.o_proj is not None:
            global_context = self.o_proj(global_context)

        return global_context


# ============================================================================
# LocalConstructorFlash — Local Construction module (flash-attn variant)
# ============================================================================
class LocalConstructorFlash(nn.Module):
    """
    Learnable query slots for local context construction using Flash Attention.

    Supports very long sequences (100k+) via O(N) memory complexity and
    correct padding handling via unpad_input. Has its own Q/K/V projections
    (does not reuse the model's projections), compatible with GQA models.
    """

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

        std = 1.0 / math.sqrt(hidden_size)
        self.memory_slots = nn.Parameter(
            torch.randn(num_local_slots, hidden_size) * std
        )
        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank == 0 and not LocalConstructorFlash._init_msg_printed:
            print(f"LocalConstructorFlash: Initialized memory_slots with std={std}")
            LocalConstructorFlash._init_msg_printed = True

        if use_bottleneck:
            assert bottleneck_dim % num_heads == 0, (
                f"bottleneck_dim ({bottleneck_dim}) must be divisible by num_heads ({num_heads})"
            )
            if rank == 0:
                print(f"LocalConstructorFlash: bottleneck_dim={bottleneck_dim}")

            self.q_proj = nn.Linear(hidden_size, bottleneck_dim, bias=False)
            self.k_proj = nn.Linear(hidden_size, bottleneck_dim, bias=False)
            self.v_proj = nn.Linear(hidden_size, bottleneck_dim, bias=False)
            self.o_proj = nn.Linear(bottleneck_dim, hidden_size, bias=False)

            self.effective_dim = bottleneck_dim
            self.effective_head_dim = bottleneck_dim // num_heads
        else:
            self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
            self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
            self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
            self.o_proj = None

            self.effective_dim = hidden_size
            self.effective_head_dim = self.head_dim

        if init_from_attn is not None:
            rank = dist.get_rank() if dist.is_initialized() else 0
            layer_idx = getattr(self, "layer_idx", 0)
            with torch.no_grad():
                self.q_proj.weight.copy_(init_from_attn.q_proj.weight)
                self.k_proj.weight.copy_(init_from_attn.k_proj.weight)
                self.v_proj.weight.copy_(init_from_attn.v_proj.weight)
            if rank == 0 and layer_idx == 0:
                print(f"[LocalConstructorFlash] Initialized Q/K/V projections from pretrained weights")

    def forward(self, hidden_states, attention_mask=None):
        """
        Compute local context via Flash Attention cross-attention.

        Args:
            hidden_states: [bsz, seq_len, hidden_size] - full input sequence
            attention_mask: [bsz, seq_len] - 1 for valid, 0 for padding (optional)

        Returns:
            global_context: [bsz, num_slots, hidden_size]
        """
        bsz, seq_len, _ = hidden_states.shape

        slots_input = self.memory_slots.unsqueeze(0).expand(bsz, -1, -1)

        Q_slots = self.q_proj(slots_input)   # [bsz, num_slots, effective_dim]
        K_seq = self.k_proj(hidden_states)   # [bsz, seq_len, effective_dim]
        V_seq = self.v_proj(hidden_states)   # [bsz, seq_len, effective_dim]

        Q_slots = Q_slots.view(bsz, self.num_local_slots, self.num_heads, self.effective_head_dim)
        K_seq = K_seq.view(bsz, seq_len, self.num_heads, self.effective_head_dim)
        V_seq = V_seq.view(bsz, seq_len, self.num_heads, self.effective_head_dim)

        if attention_mask is not None:
            kv = torch.stack([K_seq, V_seq], dim=2)
            kv_for_unpad = rearrange(kv, "b s two h d -> b s (two h d)")
            kv_unpad, indices_kv, cu_seqlens_kv, max_seqlen_kv = unpad_input(
                kv_for_unpad, attention_mask
            )
            kv_unpad = rearrange(
                kv_unpad, "nnz (two h d) -> nnz two h d", two=2, h=self.num_heads
            )

            q_unpad = rearrange(Q_slots, "b s h d -> (b s) h d")
            cu_seqlens_q = torch.arange(
                0,
                (bsz + 1) * self.num_local_slots,
                self.num_local_slots,
                device=hidden_states.device,
                dtype=torch.int32,
            )

            output_unpad = flash_attn_varlen_kvpacked_func(
                q_unpad,
                kv_unpad,
                cu_seqlens_q,
                cu_seqlens_kv,
                self.num_local_slots,
                max_seqlen_kv,
                dropout_p=0.0,
                softmax_scale=None,
                causal=False,
            )

            global_context = rearrange(
                output_unpad, "(b s) h d -> b s (h d)", b=bsz, s=self.num_local_slots
            )
        else:
            global_context = flash_attn_func(
                Q_slots, K_seq, V_seq,
                dropout_p=0.0,
                softmax_scale=None,
                causal=False,
            )
            global_context = rearrange(global_context, "b s h d -> b s (h d)")

        if self.o_proj is not None:
            global_context = self.o_proj(global_context)

        return global_context


# ============================================================================
# GlobalIntegrator — Global Integration module (independent compressors)
# ============================================================================
# Ported from llama_attn_hici.py - used when use_shared_compressor=False
# Architecture-agnostic: pure nn.Module, no model-specific dependencies
class GlobalIntegrator(nn.Module):
    """
    Global Integration module — independent-compressor variant.

    Five independent Linear compressors (one per statistic).
    Higher parameter count (~13.7M/layer) but more expressive.

    Input/output:
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
        local_slots: int = 16,
        use_bottleneck: bool = False,
        bottleneck_dim: int = 4096,
        init_from_embeddings: Optional[torch.Tensor] = None,
        use_high_norm_init: bool = True,
        output_scale_init: float = 0.1,
    ):
        super().__init__()

        assert compress_dim % num_heads == 0, (
            f"compress_dim ({compress_dim}) must be divisible by num_heads ({num_heads})"
        )
        assert output_scale_init > 0, "output_scale_init must be positive"

        self.hidden_size = hidden_size
        self.num_global = global_slots
        self.global_slots = global_slots
        self.compress_dim = compress_dim
        self.num_heads = num_heads
        self.head_dim = compress_dim // num_heads
        self.dropout_p = dropout
        self.use_high_norm_init = use_high_norm_init
        self._output_scale_init = output_scale_init

        # Stage 1: Independent stat compressors (5 separate Linear layers)
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

        # Stage 2: Lightweight Multi-Head Attention
        self.global_queries = nn.Parameter(torch.zeros(global_slots, compress_dim))
        self.q_proj = nn.Linear(compress_dim, compress_dim, bias=False)
        self.k_proj = nn.Linear(compress_dim, compress_dim, bias=False)
        self.v_proj = nn.Linear(compress_dim, compress_dim, bias=False)
        self.o_proj = nn.Linear(compress_dim, compress_dim, bias=False)
        self.attn_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Stage 3: Dimension expansion
        self.expand = nn.Linear(compress_dim, hidden_size, bias=False)
        std_init = 0.02 / math.sqrt(compress_dim)
        nn.init.normal_(self.expand.weight, mean=0.0, std=std_init)

        init_param = math.log(math.exp(output_scale_init) - 1)
        self.expand_scale_param = nn.Parameter(torch.tensor(init_param))

        self._init_weights(init_from_embeddings)
        self._print_init_info()

    @property
    def expand_scale(self) -> torch.Tensor:
        return F.softplus(self.expand_scale_param)

    def _init_weights(self, embed_weight: Optional[torch.Tensor] = None):
        if embed_weight is not None:
            with torch.no_grad():
                if self.use_high_norm_init:
                    embed_norms = torch.norm(embed_weight, dim=-1)
                    _, top_indices = torch.topk(embed_norms, k=self.global_slots)
                    init_embeddings = embed_weight[top_indices]
                else:
                    indices = torch.randperm(embed_weight.size(0))[:self.global_slots]
                    init_embeddings = embed_weight[indices]

                target_device = self.stat_compressors[0][0].weight.device
                target_dtype = self.stat_compressors[0][0].weight.dtype
                init_embeddings = init_embeddings.to(
                    device=target_device, dtype=target_dtype
                )

                init_compressed = self.stat_compressors[0](init_embeddings)
                self.global_queries.copy_(init_compressed)
        else:
            nn.init.xavier_uniform_(self.global_queries)

        for proj in [self.q_proj, self.k_proj, self.v_proj, self.o_proj]:
            nn.init.xavier_uniform_(proj.weight)

    def _print_init_info(self):
        local_rank = dist.get_rank() if dist.is_initialized() else 0
        if local_rank == 0 and not GlobalIntegrator._init_msg_printed:
            total_params = sum(p.numel() for p in self.parameters())
            print(f"   GlobalIntegrator initialized (independent compressors)")
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
        Args:
            local_repr: [bsz, num_chunks, local_slots, hidden_size]
        Returns:
            G: [bsz, global_slots, hidden_size]
        """
        bsz, num_chunks, local_slots, hidden_size = local_repr.shape

        all_local = local_repr.reshape(bsz, -1, hidden_size)

        mean_pool = all_local.mean(dim=1)
        max_pool, _ = all_local.max(dim=1)
        min_pool, _ = all_local.min(dim=1)

        with torch.amp.autocast(device_type="cuda", enabled=False):
            all_local_fp32 = all_local.float()
            std_pool = all_local_fp32.std(dim=1, unbiased=False).clamp(min=1e-6)
        std_pool = std_pool.to(all_local.dtype)

        norm_mean = F.normalize(mean_pool, dim=-1, p=2, eps=1e-6)

        stats_list = [mean_pool, max_pool, min_pool, std_pool, norm_mean]
        compressed_stats = torch.stack(
            [self.stat_compressors[i](stat) for i, stat in enumerate(stats_list)], dim=1
        )  # [bsz, 5, compress_dim]

        # Lightweight Multi-Head Attention
        Q = self.global_queries.unsqueeze(0).expand(bsz, -1, -1)
        Q = self.q_proj(Q)
        K = self.k_proj(compressed_stats)
        V = self.v_proj(compressed_stats)

        Q = Q.view(bsz, self.global_slots, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(bsz, 5, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(bsz, 5, self.num_heads, self.head_dim).transpose(1, 2)

        scale = self.head_dim ** -0.5
        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) * scale
        attn_probs = F.softmax(attn_weights, dim=-1)
        attn_probs = self.attn_dropout(attn_probs)

        attn_output = torch.matmul(attn_probs, V)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, self.global_slots, self.compress_dim)

        G_compressed = self.o_proj(attn_output)
        G = self.expand(G_compressed) * self.expand_scale

        return G


# ============================================================================
# GlobalIntegratorShared — Global Integration module (shared compressor, parameter-efficient)
# ============================================================================
# Architecture-agnostic: pure nn.Module, no model-specific dependencies
class GlobalIntegratorShared(nn.Module):
    """
    Global Integration module — shared-compressor variant.
    92% parameter reduction via shared compressor across 5 statistics.
    Architecture-agnostic: works with any hidden_size.
    """

    _init_msg_printed = False

    def __init__(
        self,
        hidden_size: int = 4096,
        global_slots: int = 4,
        compress_dim: int = 512,
        shared_compress_dim: int = 128,
        num_heads: int = 8,
        dropout: float = 0.0,
        local_slots: int = 16,
        use_bottleneck: bool = False,
        bottleneck_dim: int = 4096,
        init_from_embeddings: Optional[torch.Tensor] = None,
        use_high_norm_init: bool = True,
        output_scale_init: float = 0.1,
    ):
        super().__init__()

        assert compress_dim % num_heads == 0, (
            f"compress_dim ({compress_dim}) must be divisible by num_heads ({num_heads})"
        )
        assert output_scale_init > 0, "output_scale_init must be positive"

        self.hidden_size = hidden_size
        self.num_global = global_slots
        self.global_slots = global_slots
        self.shared_compress_dim = shared_compress_dim
        self.num_heads = num_heads
        self.dropout_p = dropout
        self.use_high_norm_init = use_high_norm_init
        self._output_scale_init = output_scale_init

        # Stage 1: Shared compressor
        self.stat_names = ["mean", "max", "min", "std", "norm_mean"]
        self.shared_compressor = nn.Sequential(
            nn.Linear(hidden_size, shared_compress_dim, bias=False),
            nn.LayerNorm(shared_compress_dim),
        )

        if shared_compress_dim < compress_dim:
            self.stat_expand = nn.Sequential(
                nn.Linear(shared_compress_dim, compress_dim, bias=False),
                nn.LayerNorm(compress_dim),
            )
            self.compress_dim = compress_dim
        else:
            self.stat_expand = nn.Identity()
            if shared_compress_dim > compress_dim:
                print(f"⚠️  Warning: shared_compress_dim ({shared_compress_dim}) > compress_dim ({compress_dim})")
            self.compress_dim = shared_compress_dim

        self.head_dim = self.compress_dim // num_heads

        # Stage 2: Lightweight Multi-Head Attention
        self.global_queries = nn.Parameter(torch.zeros(global_slots, self.compress_dim))
        self.q_proj = nn.Linear(self.compress_dim, self.compress_dim, bias=False)
        self.k_proj = nn.Linear(self.compress_dim, self.compress_dim, bias=False)
        self.v_proj = nn.Linear(self.compress_dim, self.compress_dim, bias=False)
        self.o_proj = nn.Linear(self.compress_dim, self.compress_dim, bias=False)
        self.attn_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Stage 3: Dimension expansion
        self.expand = nn.Linear(self.compress_dim, hidden_size, bias=False)
        std_init = 0.02 / math.sqrt(self.compress_dim)
        nn.init.normal_(self.expand.weight, mean=0.0, std=std_init)

        init_param = math.log(math.exp(output_scale_init) - 1)
        self.expand_scale_param = nn.Parameter(torch.tensor(init_param))

        self._init_weights(init_from_embeddings)
        self._print_init_info()

    @property
    def expand_scale(self) -> torch.Tensor:
        return F.softplus(self.expand_scale_param)

    def _init_weights(self, embed_weight: Optional[torch.Tensor] = None):
        if embed_weight is not None:
            with torch.no_grad():
                if self.use_high_norm_init:
                    embed_norms = torch.norm(embed_weight, dim=-1)
                    _, top_indices = torch.topk(embed_norms, k=self.global_slots)
                    init_embeddings = embed_weight[top_indices]
                else:
                    indices = torch.randperm(embed_weight.size(0))[:self.global_slots]
                    init_embeddings = embed_weight[indices]

                target_device = self.shared_compressor[0].weight.device
                target_dtype = self.shared_compressor[0].weight.dtype
                init_embeddings = init_embeddings.to(device=target_device, dtype=target_dtype)

                init_compressed = self.shared_compressor(init_embeddings)
                init_expanded = self.stat_expand(init_compressed)
                self.global_queries.copy_(init_expanded)
        else:
            nn.init.xavier_uniform_(self.global_queries)

        for proj in [self.q_proj, self.k_proj, self.v_proj, self.o_proj]:
            nn.init.xavier_uniform_(proj.weight)

    def _print_init_info(self):
        local_rank = dist.get_rank() if dist.is_initialized() else 0
        if local_rank == 0 and not GlobalIntegratorShared._init_msg_printed:
            total_params = sum(p.numel() for p in self.parameters())
            stat_compress_params = sum(
                p.numel() for p in self.shared_compressor.parameters()
            ) + sum(p.numel() for p in self.stat_expand.parameters())

            print(f"   GlobalIntegratorShared initialized (shared compressor)")
            if isinstance(self.stat_expand, nn.Identity):
                design_desc = "Shared Compressor + Lightweight MHA (no expansion)"
            else:
                design_desc = "Shared Compressor + Statistical Expansion + Lightweight MHA"
            print(f"       - Design: {design_desc}")
            print(f"       - Global slots: {self.global_slots}")
            print(f"       - Shared compress dim: {self.shared_compress_dim}")
            print(f"       - Final compress dim: {self.compress_dim}")
            print(f"       - Num heads: {self.num_heads}")
            print(f"       - Output scale (init): {self._output_scale_init}")
            print(f"       - Stat compression params: {stat_compress_params:,} ({stat_compress_params / 1e6:.2f}M)")
            print(f"       - Total params/layer: {total_params:,} ({total_params / 1e6:.1f}M)")
            print(f"       - 🎯 Saved {(1 - total_params / 13.7e6) * 100:.0f}% compared to original")
            GlobalIntegratorShared._init_msg_printed = True

    def forward(self, local_repr: torch.Tensor) -> torch.Tensor:
        """
        Args:
            local_repr: [bsz, num_chunks, local_slots, hidden_size]
        Returns:
            G: [bsz, global_slots, hidden_size]
        """
        bsz, num_chunks, local_slots, hidden_size = local_repr.shape

        all_local = local_repr.reshape(bsz, -1, hidden_size)

        # 5 statistics
        mean_pool = all_local.mean(dim=1)
        max_pool, _ = all_local.max(dim=1)
        min_pool, _ = all_local.min(dim=1)

        with torch.amp.autocast(device_type="cuda", enabled=False):
            all_local_fp32 = all_local.float()
            std_pool = all_local_fp32.std(dim=1, unbiased=False).clamp(min=1e-6)
        std_pool = std_pool.to(all_local.dtype)

        norm_mean = F.normalize(mean_pool, dim=-1, p=2, eps=1e-6)

        stats_list = [mean_pool, max_pool, min_pool, std_pool, norm_mean]
        stats_stacked = torch.stack(stats_list, dim=1)
        num_stats = 5

        compressed_stats = self.shared_compressor(
            stats_stacked.view(bsz * num_stats, hidden_size)
        ).view(bsz, num_stats, -1)

        compressed_stats = self.stat_expand(
            compressed_stats.view(bsz * num_stats, -1)
        ).view(bsz, num_stats, self.compress_dim)

        # Lightweight Multi-Head Attention
        Q = self.global_queries.unsqueeze(0).expand(bsz, -1, -1)
        Q = self.q_proj(Q)
        K = self.k_proj(compressed_stats)
        V = self.v_proj(compressed_stats)

        Q = Q.view(bsz, self.global_slots, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(bsz, 5, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(bsz, 5, self.num_heads, self.head_dim).transpose(1, 2)

        scale = self.head_dim ** -0.5
        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) * scale
        attn_probs = F.softmax(attn_weights, dim=-1)
        attn_probs = self.attn_dropout(attn_probs)

        attn_output = torch.matmul(attn_probs, V)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, self.global_slots, self.compress_dim)

        G_compressed = self.o_proj(attn_output)
        G = self.expand(G_compressed) * self.expand_scale

        return G

    def forward_causal(self, local_repr: torch.Tensor) -> torch.Tensor:
        """
        Causal forward pass: computes an independent G_i per segment (shared-compressor variant).

        For segment i, G_i is computed from the cumulative statistics of L_1, ..., L_i.

        Args:
            local_repr: [bsz, num_chunks, local_slots, hidden_size]

        Returns:
            G_all: [bsz, num_chunks, global_slots, hidden_size]
        """
        bsz, num_chunks, local_slots, hidden_size = local_repr.shape

        # ========== Stage 1: Cumulative statistics extraction ==========
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

        # ========== Stage 1b: Shared compression + expansion (batched over N) ==========
        BN = bsz * num_chunks
        cum_stats_flat = cum_stats.reshape(BN * 5, hidden_size)
        compressed_stats = self.shared_compressor(cum_stats_flat).view(BN, 5, -1)
        compressed_stats = self.stat_expand(
            compressed_stats.view(BN * 5, -1)
        ).view(BN, 5, self.compress_dim)

        # ========== Stage 2: Lightweight Multi-Head Attention (batched over N) ==========
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

        # ========== Stage 3: Dimension expansion ==========
        G = self.expand(G_compressed) * self.expand_scale  # [BN, global_slots, H]
        G = G.view(bsz, num_chunks, self.global_slots, hidden_size)
        return G


# ============================================================================
# forward_flashattn_hierarchical — HiCI training forward (Qwen3, transformers >= 4.51)
# ============================================================================
def forward_flashattn_hierarchical(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    attention_mask: Optional[torch.Tensor] = None,
    past_key_value=None,
    cache_position: Optional[torch.LongTensor] = None,
    use_global_context: bool = True,
    use_local_repr: bool = True,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Hierarchical HiCI forward for Qwen3 attention layers (transformers >= 4.51).

    API changes from Qwen2 version:
    - position_embeddings=(cos, sin) pre-computed by Qwen3Model, replaces position_ids + self.rotary_emb
    - Attributes via self.config (num_attention_heads, num_key_value_heads, hidden_size)
    - QK-Norm: self.q_norm / self.k_norm applied before RoPE
    - Returns (attn_output, attn_weights) instead of (attn_output, None)

    Q=[chunk], K/V=[global_context?, local?, chunk]
    """
    if not self.training:
        warnings.warn(
            "This function should be used just for training. For inference, use forward_flashattn_inference."
        )

    # --- Qwen3 attribute access (moved from self.* to self.config.*) ---
    num_heads = self.config.num_attention_heads
    num_kv_heads = self.config.num_key_value_heads
    num_kv_groups = num_heads // num_kv_heads
    head_dim = self.head_dim
    hidden_size = self.config.hidden_size

    bsz, q_len, _ = hidden_states.size()

    # If no attention_mask provided (e.g., eval without padding), create all-ones mask
    if attention_mask is None:
        attention_mask = torch.ones(bsz, q_len, device=hidden_states.device, dtype=hidden_states.dtype)

    # Print config (once, rank 0, layer 0)
    if not hasattr(self, "_hierarchical_no_cache_printed"):
        local_rank = dist.get_rank() if dist.is_initialized() else 0
        layer_idx = getattr(self, "layer_idx", 0)

        if local_rank == 0 and layer_idx == 0:
            print("\n" + "=" * 80)
            print("Hierarchical HiCI - Qwen3 (Optimized: Q=[chunk], K/V=[hici_prefix,chunk])")
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

            # Causal context mode info
            if CAUSAL_CONTEXT_MODE != "none":
                print(f"  CAUSAL_CONTEXT_MODE: {CAUSAL_CONTEXT_MODE}")
                if CAUSAL_CONTEXT_MODE == "causal_gi":
                    print("     segment_i uses G_i=Agg(L_1..L_i) + L_i")
                elif CAUSAL_CONTEXT_MODE == "causal_shift":
                    print("     segment_i uses G_{i-1}=Agg(L_1..L_{i-1}) + L_{i-1}")

            print("=" * 80 + "\n", flush=True)

        self._hierarchical_no_cache_printed = True

    # ========== Step 1: Chunk splitting ==========
    global _mixed_group_current_ratio, _mixed_group_call_count

    layer_idx = getattr(self, "layer_idx", 0)

    if self.training and MIXED_GROUP_TRAINING:
        if layer_idx == 0:
            _mixed_group_current_ratio = random.choice(GROUP_SIZE_RATIOS)
            _mixed_group_call_count += 1
            if _mixed_group_call_count % 100 == 1:
                local_rank = dist.get_rank() if dist.is_initialized() else 0
                if local_rank == 0:
                    num_groups = int(1 / _mixed_group_current_ratio)
                    print(f"[Batch {_mixed_group_call_count}] Mixed grouping: {num_groups} groups (ratio={_mixed_group_current_ratio})")
        current_ratio = _mixed_group_current_ratio
        group_size = int(q_len * current_ratio)
    elif USE_FIXED_SEGMENT_SIZE:
        group_size = FIXED_SEGMENT_SIZE
        if q_len < group_size:
            group_size = q_len
    else:
        current_ratio = group_size_ratio
        group_size = int(q_len * current_ratio)

    group_size = max(1, group_size)

    if q_len % group_size > 0:
        num_complete_groups = q_len // group_size
        if num_complete_groups == 0:
            group_size = q_len

    if not hasattr(self, "_hierarchical_group_printed"):
        local_rank = dist.get_rank() if dist.is_initialized() else 0
        if local_rank == 0 and layer_idx == 0:
            if self.training and MIXED_GROUP_TRAINING:
                print(f"[forward_flashattn_hierarchical] 🔥 MIXED_GROUP_TRAINING enabled, ratios={GROUP_SIZE_RATIOS}")
            elif USE_FIXED_SEGMENT_SIZE:
                num_groups_actual = q_len // group_size
                print(f"[forward_flashattn_hierarchical] 🎯 FIXED_SEGMENT_SIZE mode: segment_size={FIXED_SEGMENT_SIZE} tokens, {num_groups_actual} groups for q_len={q_len}")
            else:
                num_groups_actual = q_len // group_size
                print(f"[forward_flashattn_hierarchical] Fixed ratio grouping: {num_groups_actual} groups, segment_size={group_size} tokens (ratio={group_size_ratio})")
        self._hierarchical_group_printed = True

    num_groups = q_len // group_size

    chunks = hidden_states.view(bsz, num_groups, group_size, hidden_size)
    chunk_masks_reshaped = attention_mask.view(bsz, num_groups, group_size)

    # ========== Step 2: Extract local memories ==========
    if (use_global_context or use_local_repr) and hasattr(self, "local_constructor"):
        all_chunks = chunks.view(bsz * num_groups, group_size, hidden_size)

        original_dtype = all_chunks.dtype
        if all_chunks.dtype == torch.float32:
            all_chunks = all_chunks.to(torch.bfloat16)

        attention_mask_chunks = chunk_masks_reshaped.view(bsz * num_groups, group_size)
        all_local_repr = self.local_constructor(all_chunks, attention_mask_chunks)

        if original_dtype == torch.float32:
            all_local_repr = all_local_repr.to(torch.float32)

        num_local_slots = all_local_repr.shape[1]
        local_repr_stacked = all_local_repr.view(bsz, num_groups, num_local_slots, hidden_size)
    else:
        num_local_slots = 0
        local_repr_stacked = None

    # ========== Step 3: Aggregate local slots into global context vectors ==========
    _causal_mode = CAUSAL_CONTEXT_MODE  # "none", "causal_gi", "causal_shift"
    _is_causal = _causal_mode in ("causal_gi", "causal_shift")

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

            if _causal_mode == "causal_shift":
                # option B (causal_shift): segment_i uses G_{i-1}; segment_0 gets zeros
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
            # Non-causal mode: all segments share a single G
            global_context = self.global_integrator(local_repr_stacked)
            num_global_slots = global_context.shape[1]
            global_context_per_group = None
    else:
        global_context = None
        global_context_per_group = None
        num_global_slots = 0

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

    # ========== Step 4: Q/K/V projections + QK-Norm + RoPE ==========
    # Qwen3: project → QK-Norm → RoPE (unlike Qwen2 which has no QK-Norm)
    hidden_shape = (bsz, q_len, -1, head_dim)

    query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    # RoPE: use pre-computed position_embeddings from Qwen3Model
    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    # GQA: Q has num_heads, K/V have num_kv_heads. flash_attn handles GQA natively.

    # ========== Step 5: Project memories to K/V ==========
    global_context_k = global_context_v = None
    global_context_k_per_group = global_context_v_per_group = None

    if use_global_context and global_context is not None:
        # Non-causal mode: one G shared across all chunks
        global_context_k = (
            self.k_proj(global_context)
            .view(bsz, num_global_slots, num_kv_heads, head_dim)
            .transpose(1, 2)
        )
        global_context_v = (
            self.v_proj(global_context)
            .view(bsz, num_global_slots, num_kv_heads, head_dim)
            .transpose(1, 2)
        )
    elif use_global_context and global_context_per_group is not None:
        # Causal mode: each chunk has its own G_i
        # global_context_per_group: [bsz, num_groups, global_slots, hidden_size]
        gc_flat = global_context_per_group.view(
            bsz * num_groups, num_global_slots, hidden_size
        )
        gc_k_flat = (
            self.k_proj(gc_flat)
            .view(bsz * num_groups, num_global_slots, num_kv_heads, head_dim)
            .transpose(1, 2)
        )
        gc_v_flat = (
            self.v_proj(gc_flat)
            .view(bsz * num_groups, num_global_slots, num_kv_heads, head_dim)
            .transpose(1, 2)
        )
        # [bsz*num_groups, num_kv_heads, global_slots, hd] -> [bsz, num_groups, num_kv_heads, global_slots, hd]
        global_context_k_per_group = gc_k_flat.view(
            bsz, num_groups, num_kv_heads, num_global_slots, head_dim
        )
        global_context_v_per_group = gc_v_flat.view(
            bsz, num_groups, num_kv_heads, num_global_slots, head_dim
        )

    # ========== Step 6: Reshape into chunks ==========
    query_chunks = query_states.view(bsz, num_heads, num_groups, group_size, head_dim)
    key_chunks = key_states.view(bsz, num_kv_heads, num_groups, group_size, head_dim)
    value_chunks = value_states.view(bsz, num_kv_heads, num_groups, group_size, head_dim)

    if use_local_repr and local_repr_stacked is not None:
        local_mems_flat = local_repr_stacked.view(bsz * num_groups, num_local_slots, hidden_size)
        local_k_flat = (
            self.k_proj(local_mems_flat)
            .view(bsz * num_groups, num_local_slots, num_kv_heads, head_dim)
            .transpose(1, 2)
        )
        local_v_flat = (
            self.v_proj(local_mems_flat)
            .view(bsz * num_groups, num_local_slots, num_kv_heads, head_dim)
            .transpose(1, 2)
        )
        local_k_all = local_k_flat.view(bsz, num_groups, num_kv_heads, num_local_slots, head_dim)
        local_v_all = local_v_flat.view(bsz, num_groups, num_kv_heads, num_local_slots, head_dim)
    else:
        local_k_all = None
        local_v_all = None

    # ========== Step 7: Process chunks with memories (vectorized) ==========
    all_chunks_q_flat = query_chunks.permute(0, 2, 3, 1, 4).reshape(
        bsz * num_groups, group_size, num_heads, head_dim
    )

    prefix_len = 0
    if use_global_context and hasattr(self, "global_integrator"):
        prefix_len += num_global_slots
    if use_local_repr and hasattr(self, "local_constructor"):
        prefix_len += num_local_slots
    kv_len_per_chunk = prefix_len + group_size

    if prefix_len > 0:
        kv_components_k = []
        kv_components_v = []

        if use_global_context and global_context_k is not None:
            # Non-causal mode: all chunks share one G
            global_context_k_exp = global_context_k.unsqueeze(2).expand(-1, -1, num_groups, -1, -1)
            global_context_v_exp = global_context_v.unsqueeze(2).expand(-1, -1, num_groups, -1, -1)
            kv_components_k.append(global_context_k_exp)
            kv_components_v.append(global_context_v_exp)
        elif use_global_context and global_context_k_per_group is not None:
            # Causal mode: each chunk has its own G_i
            # global_context_k_per_group: [bsz, num_groups, nh, global_slots, hd]
            # Transpose to: [bsz, nh, num_groups, global_slots, hd]
            kv_components_k.append(global_context_k_per_group.permute(0, 2, 1, 3, 4))
            kv_components_v.append(global_context_v_per_group.permute(0, 2, 1, 3, 4))

        if use_local_repr and local_k_all is not None:
            local_k_exp = local_k_all.permute(0, 2, 1, 3, 4)
            local_v_exp = local_v_all.permute(0, 2, 1, 3, 4)
            kv_components_k.append(local_k_exp)
            kv_components_v.append(local_v_exp)

        kv_components_k.append(key_chunks)
        kv_components_v.append(value_chunks)

        all_k = torch.cat(kv_components_k, dim=3)
        all_v = torch.cat(kv_components_v, dim=3)
    else:
        all_k = key_chunks
        all_v = value_chunks
        kv_len_per_chunk = group_size

    all_k_flat = all_k.permute(0, 2, 3, 1, 4).reshape(
        bsz * num_groups, kv_len_per_chunk, num_kv_heads, head_dim
    )
    all_v_flat = all_v.permute(0, 2, 3, 1, 4).reshape(
        bsz * num_groups, kv_len_per_chunk, num_kv_heads, head_dim
    )

    all_chunks_kv_flat = torch.stack([all_k_flat, all_v_flat], dim=2)

    q_len_per_chunk = group_size

    # ========== Step 8: Prepare padding masks ==========
    all_masks_q_flat = chunk_masks_reshaped.reshape(bsz * num_groups, q_len_per_chunk)

    all_masks_kv_stacked = torch.empty(
        bsz, num_groups, kv_len_per_chunk,
        dtype=chunk_masks_reshaped.dtype,
        device=chunk_masks_reshaped.device,
    )

    offset = 0
    if use_global_context:
        all_masks_kv_stacked[:, :, offset:offset + num_global_slots] = 1
        offset += num_global_slots
    if use_local_repr:
        all_masks_kv_stacked[:, :, offset:offset + num_local_slots] = 1
        offset += num_local_slots
    all_masks_kv_stacked[:, :, offset:offset + group_size] = chunk_masks_reshaped

    # causal_shift: segment_0 memory is zero-padded; mask it to avoid wasting attention mass
    if _causal_mode == "causal_shift":
        mem_offset = 0
        if use_global_context:
            all_masks_kv_stacked[:, 0, mem_offset : mem_offset + num_global_slots] = 0
            mem_offset += num_global_slots
        if use_local_repr:
            all_masks_kv_stacked[:, 0, mem_offset : mem_offset + num_local_slots] = 0

    all_masks_kv_flat = all_masks_kv_stacked.reshape(bsz * num_groups, kv_len_per_chunk)

    # Unpad Q (num_heads)
    q_unpad, indices_q, cu_seqlens_q, max_seqlen_q = unpad_input(
        rearrange(all_chunks_q_flat, "b s h d -> b s (h d)"), all_masks_q_flat
    )
    q_unpad = rearrange(q_unpad, "nnz (h d) -> nnz h d", h=num_heads)

    # Unpad KV (num_kv_heads — flash_attn handles GQA natively)
    kv_flat_2d = rearrange(all_chunks_kv_flat, "b s two h d -> b s (two h d)")
    kv_unpad, indices_kv, cu_seqlens_kv, max_seqlen_kv = unpad_input(
        kv_flat_2d, all_masks_kv_flat
    )
    kv_unpad = rearrange(
        kv_unpad, "nnz (two h d) -> nnz two h d", two=2, h=num_kv_heads
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
            num_groups * bsz,
            q_len_per_chunk,
        ),
        "b s (h d) -> b s h d",
        h=num_heads,
    )

    output = output.view(bsz, num_groups, group_size, num_heads, head_dim)
    output = output.view(bsz, q_len, num_heads, head_dim)

    # ========== Attention visualization (inference only) ==========
    if COLLECT_ATTENTION_FOR_VIZ and not self.training and prefix_len > 0:
        with torch.no_grad():
            scale = 1.0 / math.sqrt(head_dim)
            # Expand K to num_heads for visualization matmul only
            all_k_flat_viz = repeat_kv(all_k_flat.transpose(1, 2), num_kv_groups).transpose(1, 2)
            attn_weights = (
                torch.matmul(
                    all_chunks_q_flat.transpose(1, 2),
                    all_k_flat_viz.transpose(1, 2).transpose(-1, -2),
                )
                * scale
            )
            attn_probs = F.softmax(attn_weights, dim=-1)

            offset = 0
            attn_to_global = 0.0
            attn_to_local = 0.0
            if use_global_context and num_global_slots > 0:
                attn_to_global = attn_probs[:, :, :, offset:offset + num_global_slots].mean().item()
                offset += num_global_slots
            if use_local_repr and num_local_slots > 0:
                attn_to_local = attn_probs[:, :, :, offset:offset + num_local_slots].mean().item()
                offset += num_local_slots
            attn_to_tokens = attn_probs[:, :, :, offset:].mean().item()

            attention_visualizer["num_global_slots"] = num_global_slots
            attention_visualizer["num_local_slots"] = num_local_slots
            attention_visualizer["segment_len"] = group_size
            attention_visualizer["layer_attn_to_global"].append(attn_to_global)
            attention_visualizer["layer_attn_to_local"].append(attn_to_local)
            attention_visualizer["layer_attn_to_tokens"].append(attn_to_tokens)

            layer_idx = getattr(self, "layer_idx", 0)
            saved_count = len(attention_visualizer["segment_attention_maps"])
            if saved_count < 32:
                attn_map = attn_probs[0, 0, :, :].cpu().numpy().tolist()
                attention_visualizer["segment_attention_maps"].append(
                    {"layer": layer_idx, "attention_map": attn_map}
                )

    # Output projection
    attn_output = self.o_proj(rearrange(output, "b s h d -> b s (h d)"))

    return attn_output, None


# ============================================================================
# forward_flashattn_hierarchical_inference — HiCI inference forward (padding support)
# ============================================================================
def forward_flashattn_hierarchical_inference(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    attention_mask: Optional[torch.Tensor] = None,
    past_key_value=None,
    cache_position: Optional[torch.LongTensor] = None,
    use_global_context: bool = True,
    use_local_repr: bool = True,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Hierarchical HiCI forward for INFERENCE (supports arbitrary-length input via padding).
    Qwen3 API (transformers >= 4.51).

    Key differences from training version:
    1. Supports arbitrary input lengths (pads to segment_size multiples)
    2. Truncates back to original length after processing
    3. Decode mode (q_len <= 32 with past_key_value): uses full attention with KV cache
    """
    # --- Qwen3 attribute access ---
    num_heads = self.config.num_attention_heads
    num_kv_heads = self.config.num_key_value_heads
    num_kv_groups = num_heads // num_kv_heads
    head_dim = self.head_dim
    hidden_size = self.config.hidden_size

    bsz, q_len, _ = hidden_states.size()

    # If no attention_mask provided (e.g., eval without padding), create all-ones mask
    if attention_mask is None:
        attention_mask = torch.ones(bsz, q_len, device=hidden_states.device, dtype=hidden_states.dtype)

    # ========================================================================
    # Decode mode: when q_len is small and past_key_value exists, use full attention
    # ========================================================================
    if q_len <= 32 and past_key_value is not None:
        # Decode mode: QK-Norm + RoPE
        hidden_shape = (bsz, q_len, -1, head_dim)
        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)

        # flash_attn_func handles GQA natively (num_heads Q, num_kv_heads K/V)
        attn_output = flash_attn_func(
            query_states.transpose(1, 2),
            key_states.transpose(1, 2),
            value_states.transpose(1, 2),
            dropout_p=0.0, softmax_scale=None, causal=True,
        )

        attn_output = attn_output.reshape(bsz, q_len, hidden_size)
        attn_output = self.o_proj(attn_output)
        return attn_output, None

    # ========================================================================
    # Prefill mode: HiCI grouping
    # ========================================================================
    original_q_len = q_len

    if not hasattr(self, "_hierarchical_inference_printed"):
        local_rank = dist.get_rank() if dist.is_initialized() else 0
        layer_idx = getattr(self, "layer_idx", 0)
        if local_rank == 0 and layer_idx == 0:
            print("\n" + "=" * 80)
            print("Hierarchical HiCI INFERENCE - Qwen3 (with padding support)")
            print("=" * 80)
            print(f"  use_global_context : {use_global_context}")
            print(f"  use_local_repr  : {use_local_repr}")
            print("=" * 80 + "\n", flush=True)
        self._hierarchical_inference_printed = True

    layer_idx = getattr(self, "layer_idx", 0)

    if USE_FULL_ATTN_WITH_HICI:
        group_size = q_len
        num_groups = 1
        padding_needed = 0
    else:
        group_size = FIXED_SEGMENT_SIZE if USE_FIXED_SEGMENT_SIZE else int(q_len * group_size_ratio)
        if q_len < group_size:
            group_size = q_len
        group_size = max(1, group_size)

        padding_needed = 0
        if q_len % group_size > 0:
            padded_q_len = ((q_len + group_size - 1) // group_size) * group_size
            padding_needed = padded_q_len - q_len

            hidden_states = F.pad(hidden_states, (0, 0, 0, padding_needed), mode="constant", value=0)
            if attention_mask is not None:
                attention_mask = F.pad(attention_mask, (0, padding_needed), mode="constant", value=0)
            if position_ids is not None:
                last_pos = position_ids[:, -1:] + 1
                padding_positions = last_pos + torch.arange(
                    padding_needed, device=position_ids.device, dtype=position_ids.dtype
                ).unsqueeze(0)
                position_ids = torch.cat([position_ids, padding_positions], dim=1)

            q_len = padded_q_len

        num_groups = q_len // group_size

        if num_groups == 1:
            use_global_context = False
            use_local_repr = False

    if not getattr(forward_flashattn_hierarchical_inference, "_prefill_printed", False):
        local_rank = dist.get_rank() if dist.is_initialized() else 0
        if local_rank == 0 and layer_idx == 0:
            print(
                f"[HiCI Prefill Qwen3] original_len={original_q_len}, padded_len={q_len}, "
                f"segment_size={group_size}, num_groups={num_groups}, padding={padding_needed}"
            )
            forward_flashattn_hierarchical_inference._prefill_printed = True

    chunks = hidden_states.view(bsz, num_groups, group_size, hidden_size)
    chunk_masks_reshaped = attention_mask.view(bsz, num_groups, group_size)

    # ========== Step 2: Extract local memories ==========
    if (use_global_context or use_local_repr) and hasattr(self, "local_constructor"):
        all_chunks = chunks.view(bsz * num_groups, group_size, hidden_size)
        original_dtype = all_chunks.dtype
        if all_chunks.dtype == torch.float32:
            all_chunks = all_chunks.to(torch.bfloat16)

        attention_mask_chunks = chunk_masks_reshaped.view(bsz * num_groups, group_size)
        all_local_repr = self.local_constructor(all_chunks, attention_mask_chunks)

        if original_dtype == torch.float32:
            all_local_repr = all_local_repr.to(torch.float32)

        num_local_slots = all_local_repr.shape[1]
        local_repr_stacked = all_local_repr.view(bsz, num_groups, num_local_slots, hidden_size)
    else:
        num_local_slots = 0
        local_repr_stacked = None

    # ========== Step 3: Aggregate to higher-level global ==========
    if use_global_context and hasattr(self, "global_integrator") and local_repr_stacked is not None:
        global_context = self.global_integrator(local_repr_stacked)
        num_global_slots = global_context.shape[1]
    else:
        global_context = None
        num_global_slots = 0

    # ========== Step 4: Q/K/V projections + QK-Norm + RoPE ==========
    hidden_shape = (bsz, q_len, -1, head_dim)
    query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    # GQA: Q has num_heads, K/V have num_kv_heads. flash_attn handles GQA natively.

    # ========== Step 5: Project memories to K/V ==========
    if use_global_context and global_context is not None:
        global_context_k = self.k_proj(global_context).view(bsz, num_global_slots, num_kv_heads, head_dim).transpose(1, 2)
        global_context_v = self.v_proj(global_context).view(bsz, num_global_slots, num_kv_heads, head_dim).transpose(1, 2)
    else:
        global_context_k = global_context_v = None

    # ========== Step 6: Reshape into chunks ==========
    query_chunks = query_states.view(bsz, num_heads, num_groups, group_size, head_dim)
    key_chunks = key_states.view(bsz, num_kv_heads, num_groups, group_size, head_dim)
    value_chunks = value_states.view(bsz, num_kv_heads, num_groups, group_size, head_dim)

    if use_local_repr and local_repr_stacked is not None:
        local_mems_flat = local_repr_stacked.view(bsz * num_groups, num_local_slots, hidden_size)
        local_k_flat = self.k_proj(local_mems_flat).view(bsz * num_groups, num_local_slots, num_kv_heads, head_dim).transpose(1, 2)
        local_v_flat = self.v_proj(local_mems_flat).view(bsz * num_groups, num_local_slots, num_kv_heads, head_dim).transpose(1, 2)
        local_k_all = local_k_flat.view(bsz, num_groups, num_kv_heads, num_local_slots, head_dim)
        local_v_all = local_v_flat.view(bsz, num_groups, num_kv_heads, num_local_slots, head_dim)
    else:
        local_k_all = None
        local_v_all = None

    # ========== Step 7: Process chunks with memories (vectorized) ==========
    all_chunks_q_flat = query_chunks.permute(0, 2, 3, 1, 4).reshape(bsz * num_groups, group_size, num_heads, head_dim)

    prefix_len = 0
    if use_global_context and hasattr(self, "global_integrator"):
        prefix_len += num_global_slots
    if use_local_repr and hasattr(self, "local_constructor"):
        prefix_len += num_local_slots
    kv_len_per_chunk = prefix_len + group_size

    if prefix_len > 0:
        kv_components_k = []
        kv_components_v = []

        if use_global_context and global_context_k is not None:
            kv_components_k.append(global_context_k.unsqueeze(2).expand(-1, -1, num_groups, -1, -1))
            kv_components_v.append(global_context_v.unsqueeze(2).expand(-1, -1, num_groups, -1, -1))

        if use_local_repr and local_k_all is not None:
            kv_components_k.append(local_k_all.permute(0, 2, 1, 3, 4))
            kv_components_v.append(local_v_all.permute(0, 2, 1, 3, 4))

        kv_components_k.append(key_chunks)
        kv_components_v.append(value_chunks)

        all_k = torch.cat(kv_components_k, dim=3)
        all_v = torch.cat(kv_components_v, dim=3)
    else:
        all_k = key_chunks
        all_v = value_chunks
        kv_len_per_chunk = group_size

    all_k_flat = all_k.permute(0, 2, 3, 1, 4).reshape(bsz * num_groups, kv_len_per_chunk, num_kv_heads, head_dim)
    all_v_flat = all_v.permute(0, 2, 3, 1, 4).reshape(bsz * num_groups, kv_len_per_chunk, num_kv_heads, head_dim)
    all_chunks_kv_flat = torch.stack([all_k_flat, all_v_flat], dim=2)

    q_len_per_chunk = group_size

    # ========== Step 8: Prepare padding masks ==========
    all_masks_q_flat = chunk_masks_reshaped.reshape(bsz * num_groups, q_len_per_chunk)

    all_masks_kv_stacked = torch.empty(
        bsz, num_groups, kv_len_per_chunk,
        dtype=chunk_masks_reshaped.dtype, device=chunk_masks_reshaped.device,
    )

    offset = 0
    if use_global_context:
        all_masks_kv_stacked[:, :, offset:offset + num_global_slots] = 1
        offset += num_global_slots
    if use_local_repr:
        all_masks_kv_stacked[:, :, offset:offset + num_local_slots] = 1
        offset += num_local_slots
    all_masks_kv_stacked[:, :, offset:offset + group_size] = chunk_masks_reshaped

    all_masks_kv_flat = all_masks_kv_stacked.reshape(bsz * num_groups, kv_len_per_chunk)

    q_unpad, indices_q, cu_seqlens_q, max_seqlen_q = unpad_input(
        rearrange(all_chunks_q_flat, "b s h d -> b s (h d)"), all_masks_q_flat
    )
    q_unpad = rearrange(q_unpad, "nnz (h d) -> nnz h d", h=num_heads)

    kv_flat_2d = rearrange(all_chunks_kv_flat, "b s two h d -> b s (two h d)")
    kv_unpad, indices_kv, cu_seqlens_kv, max_seqlen_kv = unpad_input(kv_flat_2d, all_masks_kv_flat)
    kv_unpad = rearrange(kv_unpad, "nnz (two h d) -> nnz two h d", two=2, h=num_kv_heads)

    output_unpad = flash_attn_varlen_kvpacked_func(
        q_unpad, kv_unpad,
        cu_seqlens_q, cu_seqlens_kv,
        max_seqlen_q, max_seqlen_kv,
        dropout_p=0.0, softmax_scale=None, causal=True,
    )

    output = rearrange(
        pad_input(rearrange(output_unpad, "nnz h d -> nnz (h d)"), indices_q, num_groups * bsz, q_len_per_chunk),
        "b s (h d) -> b s h d", h=num_heads,
    )

    output = output.view(bsz, num_groups, group_size, num_heads, head_dim)
    output = output.view(bsz, q_len, num_heads, head_dim)

    # Truncate back to original length (remove padding)
    if original_q_len < q_len:
        output = output[:, :original_q_len, :, :]
        if past_key_value is not None:
            past_key_value = (
                past_key_value[0][:, :, :original_q_len, :],
                past_key_value[1][:, :, :original_q_len, :],
            )

    attn_output = self.o_proj(rearrange(output, "b s h d -> b s (h d)"))
    return attn_output, None


# ============================================================================
# forward_flashattn_full - Full attention with proper unpad handling (evaluation)
# ============================================================================
def forward_flashattn_full(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    attention_mask: Optional[torch.Tensor] = None,
    past_key_value=None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Standard full attention for Qwen3 with proper unpad handling (evaluation baseline)."""
    # --- Qwen3 attribute access ---
    num_heads = self.config.num_attention_heads
    num_kv_heads = self.config.num_key_value_heads
    num_kv_groups = num_heads // num_kv_heads
    head_dim = self.head_dim
    hidden_size = self.config.hidden_size

    bsz, q_len, _ = hidden_states.size()

    # Q/K/V + QK-Norm + RoPE
    hidden_shape = (bsz, q_len, -1, head_dim)
    query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    # GQA: Q has num_heads, K/V have num_kv_heads. flash_attn handles GQA natively.
    key_padding_mask = attention_mask

    # Transpose to flash_attn format: [bsz, seq_len, heads, head_dim]
    query_states = query_states.transpose(1, 2)   # [bsz, q_len, num_heads, hd]
    key_states = key_states.transpose(1, 2)       # [bsz, q_len, num_kv_heads, hd]
    value_states = value_states.transpose(1, 2)   # [bsz, q_len, num_kv_heads, hd]

    if key_padding_mask is None or torch.all(key_padding_mask):
        # No padding: use simple flash_attn_func (faster, handles GQA natively)
        output = flash_attn_func(
            query_states, key_states, value_states,
            dropout_p=0.0, softmax_scale=None, causal=True,
        )
    else:
        # Has padding: unpad Q and KV separately (different head counts)
        q_unpad, indices_q, cu_seqlens_q, max_seqlen_q = unpad_input(
            rearrange(query_states, "b s h d -> b s (h d)"), key_padding_mask
        )
        q_unpad = rearrange(q_unpad, "nnz (h d) -> nnz h d", h=num_heads)

        kv = torch.stack([key_states, value_states], dim=2)  # [bsz, q_len, 2, num_kv_heads, hd]
        kv_2d = rearrange(kv, "b s two h d -> b s (two h d)")
        kv_unpad, _, cu_seqlens_kv, max_seqlen_kv = unpad_input(kv_2d, key_padding_mask)
        kv_unpad = rearrange(kv_unpad, "nnz (two h d) -> nnz two h d", two=2, h=num_kv_heads)

        output_unpad = flash_attn_varlen_kvpacked_func(
            q_unpad, kv_unpad,
            cu_seqlens_q, cu_seqlens_kv,
            max_seqlen_q, max_seqlen_kv,
            0.0, softmax_scale=None, causal=True,
        )
        output = rearrange(
            pad_input(rearrange(output_unpad, "nnz h d -> nnz (h d)"), indices_q, bsz, q_len),
            "b s (h d) -> b s h d", h=num_heads,
        )

    return self.o_proj(rearrange(output, "b s h d -> b s (h d)")), None


# ============================================================================
# _update_causal_mask override for Qwen3 (transformers 4.51+)
# ============================================================================
# In transformers 4.51+, _prepare_decoder_attention_mask no longer exists.
# Qwen3Model uses _update_causal_mask instead. For HiCI with flash attention,
# we override it to return the raw 2D padding mask (or None if no padding),
# since flash_attn handles causal masking internally.
# ============================================================================
def _update_causal_mask_for_hici(
    self,
    attention_mask,
    input_tensor,
    cache_position,
    past_key_values=None,
    output_attentions=False,
):
    """
    Override Qwen3Model._update_causal_mask to return 2D padding mask directly.
    Flash Attention handles causal masking internally, so we only need padding info.

    Unlike standard transformers which may return None for the no-padding case,
    HiCI forward functions always need the 2D mask for chunk reshaping.
    Consistent with Llama version's _prepare_decoder_attention_mask.
    """
    if attention_mask is None:
        # No padding mask provided (e.g., eval without DataCollator).
        # Create all-ones mask: all tokens are valid.
        attention_mask = torch.ones(
            input_tensor.shape[:2], device=input_tensor.device, dtype=input_tensor.dtype
        )
    return attention_mask


# ============================================================================
# replace_qwen3_attn — replace Qwen3Attention forward with HiCI implementation
# ============================================================================
def replace_qwen3_attn(
    use_flash_attn=True,
    use_full=False,
    inference=False,
    use_hierarchical_forward: Optional[bool] = False,
):
    """
    Replace Qwen3Attention forward function with HiCI-augmented implementations.
    Requires transformers >= 4.51.

    Args:
        use_flash_attn: Whether to use flash attention (default: True)
        use_full: Whether to use full attention without chunking (default: False)
        inference: Whether in inference mode with KV cache (default: False)
        use_hierarchical_forward: Whether to use hierarchical HiCI forward (default: False)
    """
    local_rank = dist.get_rank() if dist.is_initialized() else 0

    if not use_flash_attn:
        raise NotImplementedError(
            "Non-flash attention is not yet implemented for Qwen3 HiCI. "
            "Please use --use_flash_attn True (requires A100/H100 GPU)."
        )

    if local_rank == 0:
        warnings.warn(
            "Flash attention is only supported on A100 or H100 GPU during training due to head dim > 64 backward."
        )

    # Override causal mask to pass raw padding mask to flash attention
    transformers.models.qwen3.modeling_qwen3.Qwen3Model._update_causal_mask = (
        _update_causal_mask_for_hici
    )

    if inference:
        transformers.models.qwen3.modeling_qwen3.Qwen3Attention.forward = forward_flashattn_hierarchical_inference
        if local_rank == 0:
            print("   🔮 Using forward_flashattn_hierarchical_inference (Qwen3, inference mode)")
    elif use_full:
        transformers.models.qwen3.modeling_qwen3.Qwen3Attention.forward = forward_flashattn_full
        if local_rank == 0:
            print("   ⚡ Using forward_flashattn_full (Qwen3, full attn, no HiCI)")
    elif use_hierarchical_forward:
        if USE_FIXED_SEGMENT_SIZE:
            transformers.models.qwen3.modeling_qwen3.Qwen3Attention.forward = forward_flashattn_hierarchical_inference
            if local_rank == 0:
                print(f"   🎯 Using forward_flashattn_hierarchical_inference (Qwen3, fixed segment_size={FIXED_SEGMENT_SIZE})")
        else:
            transformers.models.qwen3.modeling_qwen3.Qwen3Attention.forward = forward_flashattn_hierarchical
            if local_rank == 0:
                print("   🧪 Using forward_flashattn_hierarchical (Qwen3, training mode)")
    else:
        transformers.models.qwen3.modeling_qwen3.Qwen3Attention.forward = forward_flashattn_hierarchical
        if local_rank == 0:
            print("   🧪 Using forward_flashattn_hierarchical (Qwen3, default)")


# ============================================================================
# register_hici_to_qwen3_model — register HiCI modules to each Qwen3Attention layer
# ============================================================================
def register_hici_to_qwen3_model(
    model,
    num_local_slots=8,
    global_slots=4,
    num_heads=32,
    use_bottleneck=True,
    bottleneck_dim=512,
    use_local_constructor=True,
    use_global_integrator=True,
    use_local_constructor_flash=False,
    use_attn_init=False,
    use_shared_compressor=True,
    compress_dim=512,
    shared_compress_dim=128,
    ds_config_path=None,
):
    """
    Register LocalConstructor and GlobalIntegrator to each Qwen3Attention layer.

    This MUST be called after model loading and before optimizer initialization!

    Key differences from Qwen2 version:
    - Navigates Qwen3ForCausalLM -> Qwen3Model -> layers -> Qwen3DecoderLayer -> self_attn
    - Reads hidden_size from config (self.config.hidden_size in 4.51+)
    - HiCI modules are architecture-agnostic (same classes as Qwen2/LLaMA version)
    """
    local_rank = dist.get_rank() if dist.is_initialized() else 0

    if use_global_integrator and not use_local_constructor:
        if local_rank == 0:
            print("❌ ERROR: use_global_integrator=True requires use_local_constructor=True")
        raise ValueError(
            "Invalid configuration: use_global_integrator=True requires use_local_constructor=True."
        )

    if local_rank == 0:
        print("\n" + "=" * 80)
        config_str = []
        if use_local_constructor:
            config_str.append("LocalConstructor")
        if use_global_integrator:
            config_str.append("Hierarchical Aggregator")
        if config_str:
            print(f"🔧 Registering (Qwen3): {' + '.join(config_str)}")
        else:
            print("⚠️ No HiCI modules enabled!")

    # Navigate to base Qwen3Model
    if hasattr(model, "base_model"):
        # PeftModelForCausalLM
        base_model = model.base_model
        if hasattr(base_model, "model"):
            qwen_model = base_model.model.model  # PeftModel -> Qwen3ForCausalLM -> Qwen3Model
        else:
            qwen_model = base_model
    else:
        # Qwen3ForCausalLM
        qwen_model = model.model

    # Get config from first layer
    hidden_size = qwen_model.layers[0].self_attn.config.hidden_size
    model_dtype = qwen_model.embed_tokens.weight.dtype
    embed_weight = qwen_model.embed_tokens.weight.data

    if local_rank == 0:
        print(f"   Model dtype: {model_dtype}")
        print(f"   Hidden size: {hidden_size}")
        print(f"   Num layers: {len(qwen_model.layers)}")

    # ZeRO-3 detection
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
                if local_rank == 0:
                    print(f"   🔧 ZeRO-3 detected (config: {detected_ds_config})")
        except Exception as e:
            if local_rank == 0:
                print(f"   ⚠️ Failed to load DeepSpeed config: {e}")

    if use_zero3_init:
        import deepspeed
        import copy
        import torch.distributed as torch_dist

        zero3_config = copy.deepcopy(ds_config)
        world_size = torch_dist.get_world_size() if torch_dist.is_initialized() else 1

        if zero3_config.get("train_batch_size") == "auto":
            zero3_config["train_batch_size"] = world_size
        if zero3_config.get("train_micro_batch_size_per_gpu") == "auto":
            zero3_config["train_micro_batch_size_per_gpu"] = 1
        if zero3_config.get("gradient_accumulation_steps") == "auto":
            zero3_config["gradient_accumulation_steps"] = 1

        zero3_init_context = deepspeed.zero.Init(config_dict_or_path=zero3_config)
        zero3_init_context.__enter__()

    # Register modules to each attention layer
    for layer_idx, layer in enumerate(qwen_model.layers):
        attn = layer.self_attn
        attn.layer_idx = layer_idx

        if use_local_constructor:
            if use_local_constructor_flash:
                attn.local_constructor = LocalConstructorFlash(
                    hidden_size=hidden_size,
                    num_local_slots=num_local_slots,
                    num_heads=num_heads,
                    init_from_embeddings=embed_weight,
                    init_from_attn=attn if use_attn_init else None,
                    use_bottleneck=use_bottleneck,
                    bottleneck_dim=bottleneck_dim,
                ).to(model_dtype)
            else:
                # Default: use LocalConstructorMulti (consistent with llama_attn_hici.py)
                attn.local_constructor = LocalConstructorMulti(
                    hidden_size=hidden_size,
                    num_local_slots=num_local_slots,
                    num_heads=num_heads,
                    init_from_embeddings=embed_weight,
                    init_from_attn=attn if use_attn_init else None,
                    use_bottleneck=use_bottleneck,
                    bottleneck_dim=bottleneck_dim,
                ).to(model_dtype)

        if use_global_integrator:
            if use_shared_compressor:
                attn.global_integrator = GlobalIntegratorShared(
                    hidden_size=hidden_size,
                    global_slots=global_slots,
                    compress_dim=bottleneck_dim,
                    shared_compress_dim=shared_compress_dim,
                    num_heads=num_heads,
                    init_from_embeddings=embed_weight,
                    use_high_norm_init=True,
                ).to(model_dtype)
            else:
                # Independent compressors per statistic (5 separate Linear layers)
                # Higher parameter count (~13.7M/layer) but more expressive
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
        if local_rank == 0:
            print(f"   ✅ ZeRO-3 HiCI module sharding complete")

    # Verify registration
    total_params = sum(p.numel() for p in model.parameters())
    local_constructor_params = 0
    global_integrator_params = 0

    if use_local_constructor or use_global_integrator:
        for name, param in model.named_parameters():
            if "local_constructor" in name:
                local_constructor_params += param.numel()
            elif "global_integrator" in name:
                global_integrator_params += param.numel()

    if local_rank == 0:
        print()
        print("=" * 80)
        print("✅ HiCI Module Registration Complete (Qwen3)")
        print("=" * 80)
        print(f"Model: {total_params:,} params ({total_params / 1e9:.2f}B)")
        print(f"Layers: {len(qwen_model.layers)}")

        if use_local_constructor and use_global_integrator:
            total_hici_params = local_constructor_params + global_integrator_params
            print(f"\nRegistered Modules:")
            print(f"  ✓ LocalConstructor ({local_constructor_params:,} params)")
            print(f"  ✓ GlobalIntegrator ({global_integrator_params:,} params)")
            print(f"\nTotal HiCI Params: {total_hici_params:,} ({total_hici_params / total_params * 100:.2f}%)")
        elif use_local_constructor:
            print(f"\nRegistered Modules:")
            print(f"  ✓ LocalConstructor ({local_constructor_params:,} params)")
            print(f"\nTotal HiCI Params: {local_constructor_params:,} ({local_constructor_params / total_params * 100:.2f}%)")

        print("=" * 80 + "\n")