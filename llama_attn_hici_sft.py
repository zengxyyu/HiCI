# HiCI attention module — SFT variant.
# Modified based on https://github.com/lm-sys/FastChat
#
# SFT-specific handling: SFT sequences have irregular lengths (e.g., 3000, 5000, 12000 tokens).
# When q_len is not divisible by 4096, the nearest valid divisor of q_len is chosen as group_size
# rather than using group_size_ratio directly. sft_group_size = 8192 is used as a fallback
# for the non-flash (forward_noflashattn) path.
#
# Corresponding training script: fine-tune_hici_sft.py

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

group_size_ratio = 1   # Fraction of tokens per chunk for LocalConstructor
sft_group_size = 8192  # Fixed group size for SFT (handles irregular sequence lengths)


# ============================================================================
# HiCI Inference KV-Cache Configuration
# ============================================================================
# Controls whether the KV cache includes HiCI memory slots during decoding.
#
# True:  KV cache = [global_context, local_slots, tokens]
#        - HiCI slots remain accessible during token-by-token decoding.
#        - attention_mask must be extended to cover the prefixed slot positions.
#
# False: KV cache = [tokens]
#        - HiCI slots are not stored in the KV cache after prefill.
INCLUDE_HICI_IN_KV_CACHE = True

# Debug switch: disable HiCI during prefill and fall back to standard Flash Attention.
# Useful for isolating whether issues originate from the inference routing logic.
DISABLE_HICI_IN_PREFILL = False

# One-shot print guards (print configuration only on the first forward call)
_HICI_INFERENCE_PRINTED = False
_HICI_GROUP_PRINTED = False
_HICI_CACHE_PRINTED = False



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

        # Learnable query slot embeddings: [num_slots, hidden_size]
        # Embedding-based initialization is disabled; use standard normal with 1/sqrt(H) std.
        if False:
            indices = torch.randperm(init_from_embeddings.size(0))[:num_local_slots]
            self.memory_slots = nn.Parameter(init_from_embeddings[indices].clone())
        else:
            std = 1.0 / math.sqrt(hidden_size)
            self.memory_slots = nn.Parameter(
                torch.randn(num_local_slots, hidden_size) * std
            )

        # Cross-attention projections for local context summarization
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

        # Expand memory for batch
        slots_input = self.memory_slots.unsqueeze(0).expand(
            bsz, -1, -1
        )  # [bsz, num_slots, hidden_size]

        # Cross-attention: memory attends to full sequence
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


class LocalConstructorMulti(nn.Module):
    """
    Multi-head Local Construction module (standard PyTorch implementation, no Flash Attention).

    Extracts M learnable query slot representations from each input segment via
    multi-head cross-attention. Supports optional bottleneck compression to reduce
    parameter count while preserving expressiveness.

    This module is registered as a sub-module of LlamaAttention, ensuring:
    1. Parameters are properly registered in model.parameters()
    2. Optimizer tracks and updates these parameters
    3. Saved/loaded with model checkpoints

    Args:
        hidden_size: Model hidden dimension (e.g., 4096 for LLaMA-2-7B)
        num_local_slots: Number of learnable query slots M (default: 8)
        num_heads: Number of attention heads (default: 32)
        init_from_embeddings: Optional pretrained embeddings for slot initialization
        init_from_attn: Optional LlamaAttention layer for warm-starting Q/K/V projections
        use_bottleneck: Whether to apply bottleneck compression (default: True)
        bottleneck_dim: Bottleneck intermediate dimension (default: 512)
    """

    # Class-level flag: print initialization info only once
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

        # Learnable query slot embeddings: [num_slots, hidden_size]
        # Embedding-based initialization is disabled; use standard normal with 1/sqrt(H) std.
        if False:
            indices = torch.randperm(init_from_embeddings.size(0))[:num_local_slots]
            self.memory_slots = nn.Parameter(init_from_embeddings[indices].clone())
            rank = dist.get_rank() if dist.is_initialized() else 0
            if rank == 0:
                print(
                    f"    LocalConstructorMulti: initialized memory_slots from pretrained embeddings "
                    f"(sampled {num_local_slots} tokens)"
                )
        else:
            std = 1.0 / math.sqrt(hidden_size)
            self.memory_slots = nn.Parameter(
                torch.randn(num_local_slots, hidden_size) * std
            )
            rank = dist.get_rank() if dist.is_initialized() else 0
            if rank == 0 and not LocalConstructorMulti._init_msg_printed:
                print(
                    f"LocalConstructorMulti: initialized memory_slots with std={std:.4f}"
                )
                LocalConstructorMulti._init_msg_printed = True

        # Cross-attention projections with optional bottleneck compression
        if use_bottleneck:
            assert bottleneck_dim % num_heads == 0, (
                f"bottleneck_dim ({bottleneck_dim}) must be divisible by num_heads ({num_heads})"
            )
            rank = dist.get_rank() if dist.is_initialized() else 0
            if rank == 0:
                print(
                    f"LocalConstructorMulti: bottleneck_dim={bottleneck_dim}, num_heads={num_heads}"
                )

            # Input projection: hidden_size -> bottleneck_dim
            self.q_proj = nn.Linear(hidden_size, bottleneck_dim, bias=False)
            self.k_proj = nn.Linear(hidden_size, bottleneck_dim, bias=False)
            self.v_proj = nn.Linear(hidden_size, bottleneck_dim, bias=False)

            # Output projection: bottleneck_dim -> hidden_size
            self.o_proj = nn.Linear(bottleneck_dim, hidden_size, bias=False)

            # Effective dimensions for attention computation
            self.effective_dim = bottleneck_dim
            self.effective_head_dim = bottleneck_dim // num_heads
        else:
            # Standard full-size projections
            self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
            self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
            self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
            self.o_proj = None  # No output projection needed at full size

            # Use original dimensions
            self.effective_dim = hidden_size
            self.effective_head_dim = self.head_dim

        # Optional warm-start: copy Q/K/V projections from pretrained LLaMA attention weights.
        # Only applicable when not using bottleneck (dimensions must match).
        if init_from_attn is not None and not use_bottleneck:
            rank = dist.get_rank() if dist.is_initialized() else 0
            layer_idx = getattr(self, "layer_idx", 0)
            with torch.no_grad():
                self.q_proj.weight.copy_(init_from_attn.q_proj.weight)
                self.k_proj.weight.copy_(init_from_attn.k_proj.weight)
                self.v_proj.weight.copy_(init_from_attn.v_proj.weight)
            if rank == 0 and layer_idx == 0:
                print(
                    f"LocalConstructorMulti: Q/K/V projections initialized from LLaMA pretrained weights"
                )

    def forward(self, hidden_states, attention_mask=None):
        """
        Extract M local slot representations via multi-head cross-attention.

        Q: learnable slot embeddings (no padding), length = num_local_slots.
        K/V: input segment tokens, with optional attention_mask for padding.

        Args:
            hidden_states: [bsz, seq_len, hidden_size] — input segment
            attention_mask: [bsz, seq_len] — 1 for valid tokens, 0 for padding (optional)

        Returns:
            global_context: [bsz, num_local_slots, hidden_size] — local slot representations
        """
        bsz, seq_len, _ = hidden_states.shape

        # Expand memory for batch
        slots_input = self.memory_slots.unsqueeze(0).expand(
            bsz, -1, -1
        )  # [bsz, num_slots, hidden_size]

        # Cross-attention projections to effective dimension (bottleneck or full)
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

        # Compute attention scores: Q @ K^T
        # [bsz, num_heads, num_slots, effective_head_dim] @ [bsz, num_heads, effective_head_dim, seq_len]
        # -> [bsz, num_heads, num_slots, seq_len]
        scores = torch.matmul(Q_slots, K_seq.transpose(-2, -1)) / math.sqrt(
            self.effective_head_dim
        )

        # Apply attention mask if provided
        if attention_mask is not None:
            # attention_mask: [bsz, seq_len] - 1 for valid, 0 for padding
            # Expand for multi-head and num_slots: [bsz, 1, 1, seq_len]
            mask_expanded = attention_mask.unsqueeze(1).unsqueeze(
                2
            )  # [bsz, 1, 1, seq_len]

            # Convert to additive mask: 0 for valid, -inf for padding
            # This will zero out attention to padding tokens after softmax
            mask_value = torch.finfo(scores.dtype).min  # -inf for the dtype
            scores = scores.masked_fill(mask_expanded == 0, mask_value)

        # Apply softmax: [bsz, num_heads, num_slots, seq_len]
        attn_weights = torch.softmax(scores, dim=-1)

        # Apply attention to values: attn_weights @ V
        # [bsz, num_heads, num_slots, seq_len] @ [bsz, num_heads, seq_len, effective_head_dim]
        # -> [bsz, num_heads, num_slots, effective_head_dim]
        attn_output = torch.matmul(attn_weights, V_seq)

        # Transpose back and reshape: [bsz, num_heads, num_slots, effective_head_dim]
        # -> [bsz, num_slots, num_heads, effective_head_dim]
        # -> [bsz, num_slots, effective_dim]
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


# region ===========================================================================
class LocalConstructorFlash(nn.Module):
    """
    Flash Attention-based Local Construction module.

    Uses flash_attn_varlen_kvpacked_func for memory-efficient cross-attention,
    supporting sequences of 100k+ tokens with O(N) memory complexity.
    Padding tokens are removed via unpad_input before the attention kernel.

    This module is registered as a sub-module of LlamaAttention, ensuring:
    1. Parameters are properly registered in model.parameters()
    2. Optimizer tracks and updates these parameters
    3. Saved/loaded with model checkpoints

    Args:
        hidden_size: Model hidden dimension (e.g., 4096 for LLaMA-2-7B)
        num_local_slots: Number of learnable query slots M (default: 8)
        num_heads: Number of attention heads (default: 32)
    """

    def __init__(
        self, hidden_size, num_local_slots=8, num_heads=32, init_from_embeddings=None
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_local_slots = num_local_slots
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        assert hidden_size % num_heads == 0, (
            f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})"
        )

        # Learnable query slot embeddings: [num_slots, hidden_size]
        # Embedding-based initialization is disabled; use standard normal with 1/sqrt(H) std.
        if False:
            indices = torch.randperm(init_from_embeddings.size(0))[:num_local_slots]
            self.memory_slots = nn.Parameter(init_from_embeddings[indices].clone())
        else:
            std = 1.0 / math.sqrt(hidden_size)
            self.memory_slots = nn.Parameter(
                torch.randn(num_local_slots, hidden_size) * std
            )

        # Cross-attention projections for local context summarization
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, hidden_states, attention_mask=None):
        """
        Extract M local slot representations via Flash Attention cross-attention.

        Q: learnable slot embeddings, length = num_local_slots (no padding).
        K/V: input segment tokens; padding is removed via unpad_input before the kernel.

        Args:
            hidden_states: [bsz, seq_len, hidden_size] — input segment
            attention_mask: [bsz, seq_len] — 1 for valid tokens, 0 for padding (optional)

        Returns:
            global_context: [bsz, num_local_slots, hidden_size] — local slot representations
        """
        bsz, seq_len, _ = hidden_states.shape

        # Expand memory for batch
        slots_input = self.memory_slots.unsqueeze(0).expand(
            bsz, -1, -1
        )  # [bsz, num_slots, hidden_size]

        # Cross-attention projections
        Q_slots = self.q_proj(slots_input)  # [bsz, num_slots, hidden_size]
        K_seq = self.k_proj(hidden_states)  # [bsz, seq_len, hidden_size]
        V_seq = self.v_proj(hidden_states)  # [bsz, seq_len, hidden_size]

        # Reshape for multi-head attention: [bsz, seqlen, num_heads, head_dim]
        Q_slots = Q_slots.view(bsz, self.num_local_slots, self.num_heads, self.head_dim)
        K_seq = K_seq.view(bsz, seq_len, self.num_heads, self.head_dim)
        V_seq = V_seq.view(bsz, seq_len, self.num_heads, self.head_dim)

        if attention_mask is not None:
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

            # Q has no padding; flatten to [bsz * num_slots, num_heads, head_dim]
            q_unpad = rearrange(Q_slots, "b s h d -> (b s) h d")

            # cu_seqlens_q: cumulative slot counts per sample, e.g., [0, 16, 32] for bsz=2, num_slots=16
            cu_seqlens_q = torch.arange(
                0,
                (bsz + 1) * self.num_local_slots,
                self.num_local_slots,
                device=hidden_states.device,
                dtype=torch.int32,
            )
            # Q:  [bsz * num_slots, num_heads, head_dim]
            # KV: [total_valid_kv_tokens, 2, num_heads, head_dim]
            output_unpad = flash_attn_varlen_kvpacked_func(
                q_unpad,
                kv_unpad,
                cu_seqlens_q,
                cu_seqlens_kv,
                self.num_local_slots,  # max_seqlen_q (fixed)
                max_seqlen_kv,
                dropout_p=0.0,
                softmax_scale=None,
                causal=False,  # cross-attention does not require a causal mask
            )
            # output_unpad: [bsz * num_slots, num_heads, head_dim]

            # Reshape back: [bsz, num_slots, hidden_size]
            global_context = rearrange(
                output_unpad, "(b s) h d -> b s (h d)", b=bsz, s=self.num_local_slots
            )
        else:
            # No padding: use the simpler flash_attn_func directly
            global_context = flash_attn_func(
                Q_slots,  # [bsz, num_slots, num_heads, head_dim]
                K_seq,  # [bsz, seq_len, num_heads, head_dim]
                V_seq,  # [bsz, seq_len, num_heads, head_dim]
                dropout_p=0.0,
                softmax_scale=None,
                causal=False,
            )
            # global_context: [bsz, num_slots, num_heads, head_dim]

            # Reshape: [bsz, num_slots, hidden_size]
            global_context = rearrange(global_context, "b s h d -> b s (h d)")

        return global_context


# endregion ===========================================================================
class LocalConstructorFlashPlus(nn.Module):
    """
    Flash Attention-based Local Construction module (K/V-reuse variant).

    Reuses the pre-projected K/V states from the host LlamaAttention layer.
    Only an independent Q projection for the slot embeddings is required,
    eliminating redundant K/V projections.

    NOTE: This variant expects pre-projected K/V as input (not raw hidden states).
    It is therefore incompatible with forward_flashattn_hierarchical, which passes
    raw hidden states to local_constructor. Use LocalConstructorMulti or
    LocalConstructorFlash with that training path.

    Args:
        hidden_size: Model hidden dimension (e.g., 4096 for LLaMA-2-7B)
        num_local_slots: Number of learnable query slots M (default: 8)
        num_heads: Number of attention heads (default: 32)
    """

    def __init__(
        self, hidden_size, num_local_slots=8, num_heads=32, init_from_embeddings=None
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_local_slots = num_local_slots
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        assert hidden_size % num_heads == 0, (
            f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})"
        )

        # Learnable memory slots: [num_slots, hidden_size]
        std = 1.0 / math.sqrt(hidden_size)
        self.memory_slots = nn.Parameter(
            torch.randn(num_local_slots, hidden_size) * std
        )
        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank == 0:
            print(
                f"    LocalConstructorFlashPlus: initialized memory_slots with std={std:.4f}"
            )
            print(
                f"    LocalConstructorFlashPlus: reuses host K/V projections; only Q is independently projected"
            )

        # Only a Q projection is needed; K/V come from the host attention layer
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, key_states, value_states, attention_mask=None):
        """
        Extract M local slot representations via Flash Attention cross-attention.

        Accepts pre-projected (and RoPE-applied) K/V from the host LlamaAttention layer.
        Only the query slot embeddings are independently projected.

        Args:
            key_states:   [bsz, seq_len, num_heads, head_dim] — pre-projected K
            value_states: [bsz, seq_len, num_heads, head_dim] — pre-projected V
            attention_mask: [bsz, seq_len] — 1 for valid tokens, 0 for padding (optional)

        Returns:
            global_context: [bsz, num_local_slots, hidden_size] — local slot representations
        """
        bsz = key_states.shape[0]

        # Expand memory for batch and project to Q
        slots_input = self.memory_slots.unsqueeze(0).expand(
            bsz, -1, -1
        )  # [bsz, num_slots, hidden_size]

        # Project slot embeddings to Q; K/V are passed in directly
        Q_slots = self.q_proj(slots_input)  # [bsz, num_slots, hidden_size]

        # Reshape Q for multi-head attention: [bsz, num_slots, num_heads, head_dim]
        Q_slots = Q_slots.view(bsz, self.num_local_slots, self.num_heads, self.head_dim)

        # K/V are already in the correct shape: [bsz, seq_len, num_heads, head_dim]
        K_seq = key_states
        V_seq = value_states

        if attention_mask is not None:
            # Flash Attention + unpad: remove padding tokens from K/V before the kernel
            # Pack K and V together: [bsz, seq_len, 2, num_heads, head_dim]
            kv = torch.stack([K_seq, V_seq], dim=2)

            # Reshape for unpad_input: [bsz, seq_len, 2 * num_heads * head_dim]
            kv_for_unpad = rearrange(kv, "b s two h d -> b s (two h d)")

            # Remove padding from K/V
            kv_unpad, indices_kv, cu_seqlens_kv, max_seqlen_kv = unpad_input(
                kv_for_unpad, attention_mask
            )

            # Reshape back: [total_valid_kv_tokens, 2, num_heads, head_dim]
            kv_unpad = rearrange(
                kv_unpad, "nnz (two h d) -> nnz two h d", two=2, h=self.num_heads
            )

            # Q has no padding, flatten: [bsz, num_slots, h, d] -> [bsz * num_slots, h, d]
            q_unpad = rearrange(Q_slots, "b s h d -> (b s) h d")

            # cu_seqlens_q: cumulative slot counts per sample
            cu_seqlens_q = torch.arange(
                0,
                (bsz + 1) * self.num_local_slots,
                self.num_local_slots,
                device=key_states.device,
                dtype=torch.int32,
            )

            output_unpad = flash_attn_varlen_kvpacked_func(
                q_unpad,
                kv_unpad,
                cu_seqlens_q,
                cu_seqlens_kv,
                self.num_local_slots,  # max_seqlen_q (fixed)
                max_seqlen_kv,
                dropout_p=0.0,
                softmax_scale=None,
                causal=False,  # cross-attention does not require a causal mask
            )

            # Reshape back: [bsz, num_slots, hidden_size]
            global_context = rearrange(
                output_unpad, "(b s) h d -> b s (h d)", b=bsz, s=self.num_local_slots
            )
        else:
            # No padding: use the simpler flash_attn_func directly
            global_context = flash_attn_func(
                Q_slots,  # [bsz, num_slots, num_heads, head_dim]
                K_seq,  # [bsz, seq_len, num_heads, head_dim]
                V_seq,  # [bsz, seq_len, num_heads, head_dim]
                dropout_p=0.0,
                softmax_scale=None,
                causal=False,
            )
            # Reshape: [bsz, num_slots, hidden_size]
            global_context = rearrange(global_context, "b s h d -> b s (h d)")

        return global_context


class GlobalIntegrator(nn.Module):
    """
    Global Integration module — two-stage aggregation of local slot representations.

    Aggregates the M × num_chunks local slot outputs from LocalConstructor into K
    global context vectors via statistical pooling followed by lightweight multi-head
    attention:

      Stage 1 — Statistical compression:
        Five statistics (mean, max, min, std, L2-normalized mean) are computed over
        all local slots and individually compressed to compress_dim via learned linear
        projections with LayerNorm.

      Stage 2 — Lightweight multi-head attention:
        K learned global query vectors attend over the 5 compressed statistics,
        producing the K global context vectors in compress_dim space.

      Stage 3 — Dimension expansion:
        The compressed global vectors are projected back to hidden_size with a learned
        scaling factor (via softplus to ensure positivity).

    Input:  local_repr [bsz, num_chunks, local_slots, hidden_size]
    Output: global_context  [bsz, global_slots, hidden_size]

    Parameter count (hidden_size=4096, compress_dim=512, global_slots=4):
        Statistical compressors: 5 × (4096 × 512) ≈ 10.5M
        Q/K/V projections:       3 × (512 × 512)  = 0.8M
        Output projection:       512 × 512         = 0.26M
        Expansion layer:         512 × 4096        = 2.1M
        Total:                   ~13.7M / layer
    """

    _init_msg_printed = False

    def __init__(
        self,
        hidden_size: int = 4096,
        global_slots: int = 4,
        compress_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.0,
        local_slots: int = 16,  # unused; kept for backward compatibility
        use_bottleneck: bool = False,  # unused; kept for backward compatibility
        bottleneck_dim: int = 4096,  # unused; kept for backward compatibility
        init_from_embeddings: Optional[torch.Tensor] = None,
        use_high_norm_init: bool = True,
        output_scale_init: float = 0.1,
    ):
        """
        Args:
            hidden_size: Model hidden dimension (default: 4096)
            global_slots: Number of global context vectors K (default: 4)
            compress_dim: Intermediate bottleneck dimension for statistics and attention (default: 512)
            num_heads: Number of attention heads; compress_dim must be divisible by num_heads
            dropout: Attention dropout probability
            init_from_embeddings: Optional pretrained embedding matrix for global_queries initialization
            use_high_norm_init: If True, initialize global_queries from the top-K highest-norm embeddings
            output_scale_init: Initial value for the softplus output scaling factor
        """
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

        # Stage 1: per-statistic compression layers (hidden_size -> compress_dim)
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

        # Stage 2: lightweight multi-head attention over 5 compressed statistics
        self.global_queries = nn.Parameter(torch.zeros(global_slots, compress_dim))

        self.q_proj = nn.Linear(compress_dim, compress_dim, bias=False)
        self.k_proj = nn.Linear(compress_dim, compress_dim, bias=False)
        self.v_proj = nn.Linear(compress_dim, compress_dim, bias=False)
        self.o_proj = nn.Linear(compress_dim, compress_dim, bias=False)

        self.attn_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Stage 3: dimension expansion back to hidden_size
        self.expand = nn.Linear(compress_dim, hidden_size, bias=False)
        std_init = 0.02 / math.sqrt(compress_dim)
        nn.init.normal_(self.expand.weight, mean=0.0, std=std_init)

        # Learnable output scaling factor; stored in pre-softplus space to ensure positivity.
        # init_param is the inverse softplus of output_scale_init.
        init_param = math.log(math.exp(output_scale_init) - 1)
        self.expand_scale_param = nn.Parameter(torch.tensor(init_param))

        self._init_weights(init_from_embeddings)
        self._print_init_info()

    @property
    def expand_scale(self) -> torch.Tensor:
        """Output scale (always positive via softplus)."""
        return F.softplus(self.expand_scale_param)

    def _init_weights(self, embed_weight: Optional[torch.Tensor] = None):
        """Initialize global_queries from embeddings (if provided) and Q/K/V/O via Xavier."""
        if embed_weight is not None:
            with torch.no_grad():
                if self.use_high_norm_init:
                    embed_norms = torch.norm(embed_weight, dim=-1)
                    _, top_indices = torch.topk(embed_norms, k=self.global_slots)
                    init_embeddings = embed_weight[top_indices]
                else:
                    indices = torch.randperm(embed_weight.size(0))[: self.global_slots]
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
        """Print initialization summary (rank 0, once per process)."""
        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank == 0 and not GlobalIntegrator._init_msg_printed:
            total_params = sum(p.numel() for p in self.parameters())
            print(f"   GlobalIntegrator initialized")
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
        Aggregate local slot representations into K global context vectors.

        Args:
            local_repr: [bsz, num_chunks, local_slots, hidden_size]

        Returns:
            G: [bsz, global_slots, hidden_size]

        Data flow:
            local_repr [bsz, C, L, H]
                → flatten to all_local [bsz, C*L, H]
                → 5 statistics [bsz, H] each
                → compress each to [bsz, D]
                → stack to compressed_stats [bsz, 5, D]
                → lightweight MHA → G_compressed [bsz, K, D]
                → expand + scale → G [bsz, K, H]
        """
        bsz, num_chunks, local_slots, hidden_size = local_repr.shape

        # ========== Stage 1: statistical extraction and compression ==========
        # Flatten local slots across all chunks: [bsz, num_chunks * local_slots, hidden_size]
        all_local = local_repr.reshape(bsz, -1, hidden_size)

        # Five statistics, each [bsz, hidden_size]
        mean_pool = all_local.mean(dim=1)
        max_pool, _ = all_local.max(dim=1)
        min_pool, _ = all_local.min(dim=1)

        # Compute std in fp32 for numerical stability
        with torch.amp.autocast(device_type="cuda", enabled=False):
            all_local_fp32 = all_local.float()
            std_pool = all_local_fp32.std(dim=1, unbiased=False).clamp(min=1e-6)
        std_pool = std_pool.to(all_local.dtype)

        # L2-normalized mean (direction vector)
        norm_mean = F.normalize(mean_pool, dim=-1, p=2, eps=1e-6)

        # Compress each statistic independently: [bsz, hidden_size] -> [bsz, compress_dim]
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

        # Multi-head split: [bsz, num_heads, seq_len, head_dim]
        Q = Q.view(bsz, self.global_slots, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(bsz, 5, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(bsz, 5, self.num_heads, self.head_dim).transpose(1, 2)
        # Q: [bsz, num_heads, global_slots, head_dim]
        # K: [bsz, num_heads, 5, head_dim]
        # V: [bsz, num_heads, 5, head_dim]

        # Scaled dot-product attention: [bsz, num_heads, global_slots, 5]
        scale = self.head_dim**-0.5
        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) * scale
        attn_probs = F.softmax(attn_weights, dim=-1)
        attn_probs = self.attn_dropout(attn_probs)

        # [bsz, num_heads, global_slots, head_dim]
        attn_output = torch.matmul(attn_probs, V)

        # Merge heads: [bsz, global_slots, compress_dim]
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, self.global_slots, self.compress_dim)

        # Output projection
        G_compressed = self.o_proj(attn_output)

        # ========== Stage 3: dimension expansion ==========
        G = self.expand(G_compressed) * self.expand_scale

        return G


class GlobalIntegratorShared(nn.Module):
    """
    Global Integration module — shared-compressor variant (parameter-efficient).

    Identical to GlobalIntegrator but replaces the 5 independent per-statistic
    compressors with a single shared compressor followed by a dimension-expansion
    layer, reducing the statistical compression parameter count by ~92%.

    Parameter breakdown (hidden_size=4096, compress_dim=512, shared_compress_dim=128):
        Shared compressor:    4096 × 128          = 0.524M
        Statistical expansion: 128 × 512          = 0.066M
        Q/K/V projections:    3 × (512 × 512)     = 0.786M
        Output projection:    512 × 512            = 0.262M
        Expansion layer:      512 × 4096           = 2.097M
        Total:                ~4.0M / layer  (vs. ~13.7M for GlobalIntegrator, –71%)

    Input:  local_repr [bsz, num_chunks, local_slots, hidden_size]
    Output: global_context  [bsz, global_slots, hidden_size]
    """

    _init_msg_printed = False

    def __init__(
        self,
        hidden_size: int = 4096,
        global_slots: int = 4,
        compress_dim: int = 512,
        shared_compress_dim: int = 128,  # intermediate dimension of the shared compressor
        num_heads: int = 8,
        dropout: float = 0.0,
        local_slots: int = 16,  # unused; kept for backward compatibility
        use_bottleneck: bool = False,  # unused; kept for backward compatibility
        bottleneck_dim: int = 4096,  # unused; kept for backward compatibility
        init_from_embeddings: Optional[torch.Tensor] = None,
        use_high_norm_init: bool = True,
        output_scale_init: float = 0.1,
    ):
        """
        Args:
            hidden_size: Model hidden dimension (default: 4096)
            global_slots: Number of global context vectors K (default: 4)
            compress_dim: Final compressed dimension for attention (default: 512)
            shared_compress_dim: Intermediate dimension of the shared compressor (default: 128)
            num_heads: Number of attention heads
            dropout: Attention dropout probability
            init_from_embeddings: Optional pretrained embedding matrix for global_queries initialization
            use_high_norm_init: If True, initialize global_queries from the top-K highest-norm embeddings
            output_scale_init: Initial value for the softplus output scaling factor
        """
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

        # Stage 1: shared compressor — all 5 statistics pass through the same projection
        self.stat_names = ["mean", "max", "min", "std", "norm_mean"]
        self.shared_compressor = nn.Sequential(
            nn.Linear(hidden_size, shared_compress_dim, bias=False),
            nn.LayerNorm(shared_compress_dim),
        )

        # Optional expansion layer: shared_compress_dim -> compress_dim
        # Not needed when shared_compress_dim == compress_dim.
        if shared_compress_dim < compress_dim:
            self.stat_expand = nn.Sequential(
                nn.Linear(shared_compress_dim, compress_dim, bias=False),
                nn.LayerNorm(compress_dim),
            )
            self.compress_dim = compress_dim
        else:
            self.stat_expand = nn.Identity()
            if shared_compress_dim > compress_dim:
                print(
                    f"Warning: shared_compress_dim ({shared_compress_dim}) > compress_dim ({compress_dim}); "
                    f"setting compress_dim = shared_compress_dim"
                )
            self.compress_dim = shared_compress_dim

        self.head_dim = self.compress_dim // num_heads

        # Stage 2: lightweight multi-head attention over 5 compressed statistics
        self.global_queries = nn.Parameter(torch.zeros(global_slots, self.compress_dim))

        self.q_proj = nn.Linear(self.compress_dim, self.compress_dim, bias=False)
        self.k_proj = nn.Linear(self.compress_dim, self.compress_dim, bias=False)
        self.v_proj = nn.Linear(self.compress_dim, self.compress_dim, bias=False)
        self.o_proj = nn.Linear(self.compress_dim, self.compress_dim, bias=False)

        self.attn_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Stage 3: dimension expansion back to hidden_size
        self.expand = nn.Linear(self.compress_dim, hidden_size, bias=False)
        std_init = 0.02 / math.sqrt(self.compress_dim)
        nn.init.normal_(self.expand.weight, mean=0.0, std=std_init)

        # Learnable output scaling factor (stored in pre-softplus space)
        init_param = math.log(math.exp(output_scale_init) - 1)
        self.expand_scale_param = nn.Parameter(torch.tensor(init_param))

        self._init_weights(init_from_embeddings)
        self._print_init_info()

    @property
    def expand_scale(self) -> torch.Tensor:
        """Output scale (always positive via softplus)."""
        return F.softplus(self.expand_scale_param)

    def _init_weights(self, embed_weight: Optional[torch.Tensor] = None):
        """Initialize global_queries from embeddings (if provided) and Q/K/V/O via Xavier."""
        if embed_weight is not None:
            with torch.no_grad():
                if self.use_high_norm_init:
                    embed_norms = torch.norm(embed_weight, dim=-1)
                    _, top_indices = torch.topk(embed_norms, k=self.global_slots)
                    init_embeddings = embed_weight[top_indices]
                else:
                    indices = torch.randperm(embed_weight.size(0))[: self.global_slots]
                    init_embeddings = embed_weight[indices]

                target_device = self.shared_compressor[0].weight.device
                target_dtype = self.shared_compressor[0].weight.dtype
                init_embeddings = init_embeddings.to(
                    device=target_device, dtype=target_dtype
                )

                # Initialize global_queries via the shared compressor + expansion path
                init_compressed = self.shared_compressor(init_embeddings)  # [K, shared_compress_dim]
                init_expanded = self.stat_expand(init_compressed)           # [K, compress_dim]
                self.global_queries.copy_(init_expanded)
        else:
            nn.init.xavier_uniform_(self.global_queries)

        for proj in [self.q_proj, self.k_proj, self.v_proj, self.o_proj]:
            nn.init.xavier_uniform_(proj.weight)

    def _print_init_info(self):
        """Print initialization summary (rank 0, once per process)."""
        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank == 0 and not GlobalIntegratorShared._init_msg_printed:
            total_params = sum(p.numel() for p in self.parameters())
            stat_compress_params = sum(
                p.numel() for p in self.shared_compressor.parameters()
            ) + sum(p.numel() for p in self.stat_expand.parameters())

            if isinstance(self.stat_expand, nn.Identity):
                design_desc = "Shared Compressor + Lightweight MHA (no expansion)"
            else:
                design_desc = "Shared Compressor + Statistical Expansion + Lightweight MHA"

            print(f"   GlobalIntegratorShared initialized")
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
                f"       - Saved {(1 - total_params / 13.7e6) * 100:.0f}% compared to GlobalIntegrator"
            )
            GlobalIntegratorShared._init_msg_printed = True

    def forward(self, local_repr: torch.Tensor) -> torch.Tensor:
        """
        Aggregate local slot representations into K global context vectors.

        Args:
            local_repr: [bsz, num_chunks, local_slots, hidden_size]

        Returns:
            G: [bsz, global_slots, hidden_size]

        Data flow:
            local_repr [bsz, C, L, H]
                → flatten to all_local [bsz, C*L, H]
                → 5 statistics [bsz, H] each
                → shared_compressor → [bsz, 5, shared_compress_dim]
                → stat_expand      → compressed_stats [bsz, 5, compress_dim]
                → lightweight MHA  → G_compressed [bsz, K, compress_dim]
                → expand + scale   → G [bsz, K, H]
        """
        bsz, num_chunks, local_slots, hidden_size = local_repr.shape

        # ========== Stage 1a: statistical extraction ==========
        all_local = local_repr.reshape(bsz, -1, hidden_size)

        # Five statistics, each [bsz, hidden_size]
        mean_pool = all_local.mean(dim=1)
        max_pool, _ = all_local.max(dim=1)
        min_pool, _ = all_local.min(dim=1)

        # Compute std in fp32 for numerical stability
        with torch.amp.autocast(device_type="cuda", enabled=False):
            all_local_fp32 = all_local.float()
            std_pool = all_local_fp32.std(dim=1, unbiased=False).clamp(min=1e-6)
        std_pool = std_pool.to(all_local.dtype)

        # L2-normalized mean (direction vector)
        norm_mean = F.normalize(mean_pool, dim=-1, p=2, eps=1e-6)

        # ========== Stage 1b: shared compression + expansion ==========
        # All 5 statistics share the same compressor; keep them separate so the
        # attention in Stage 2 can learn to selectively weight them.
        stats_list = [mean_pool, max_pool, min_pool, std_pool, norm_mean]

        # Batch process: [bsz, 5, H] -> compress -> [bsz, 5, shared_compress_dim]
        stats_stacked = torch.stack(stats_list, dim=1)
        num_stats = 5

        compressed_stats = self.shared_compressor(
            stats_stacked.view(bsz * num_stats, hidden_size)
        ).view(bsz, num_stats, -1)

        # Expand: [bsz, 5, shared_compress_dim] -> [bsz, 5, compress_dim]
        compressed_stats = self.stat_expand(
            compressed_stats.view(bsz * num_stats, -1)
        ).view(bsz, num_stats, self.compress_dim)

        # ========== Stage 2: lightweight multi-head attention ==========
        # Q: [bsz, global_slots, compress_dim]
        Q = self.global_queries.unsqueeze(0).expand(bsz, -1, -1)
        Q = self.q_proj(Q)

        # K, V: [bsz, 5, compress_dim]
        K = self.k_proj(compressed_stats)
        V = self.v_proj(compressed_stats)

        # Multi-head split: [bsz, num_heads, seq_len, head_dim]
        Q = Q.view(bsz, self.global_slots, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(bsz, 5, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(bsz, 5, self.num_heads, self.head_dim).transpose(1, 2)
        # Q: [bsz, num_heads, global_slots, head_dim]
        # K: [bsz, num_heads, 5, head_dim]
        # V: [bsz, num_heads, 5, head_dim]

        # Scaled dot-product attention: [bsz, num_heads, global_slots, 5]
        scale = self.head_dim**-0.5
        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) * scale
        attn_probs = F.softmax(attn_weights, dim=-1)
        attn_probs = self.attn_dropout(attn_probs)

        # [bsz, num_heads, global_slots, head_dim]
        attn_output = torch.matmul(attn_probs, V)

        # Merge heads: [bsz, global_slots, compress_dim]
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, self.global_slots, self.compress_dim)

        # Output projection
        G_compressed = self.o_proj(attn_output)

        # ========== Stage 3: dimension expansion ==========
        G = self.expand(G_compressed) * self.expand_scale

        return G


def forward_flashattn_hierarchical(
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
    HiCI hierarchical attention — SFT training forward pass (no KV-cache recurrence).

    Implements the three-stage HiCI pipeline per transformer layer:
      1. Local Construction  — LocalConstructor extracts M local slot representations
                               per segment via cross-attention.
      2. Global Integration  — GlobalIntegrator aggregates all segments' slot outputs
                               into K global context vectors (optional).
      3. Top-down Broadcast  — Each segment's tokens attend to:
                               Q = [chunk tokens]
                               K/V = [global_context?, local_slots?, chunk tokens]

    Attention layout:
        Mode 1 (global only): Q=[chunk], K/V=[global_context, chunk]
        Mode 2 (local only):  Q=[chunk], K/V=[local_i, chunk]
        Mode 3 (full HiCI):   Q=[chunk], K/V=[global_context, local_i, chunk]
        Baseline:             Q=K/V=[chunk]

    Args:
        use_global_context: include GlobalIntegrator output in K/V (default: True)
        use_local_repr: prepend per-segment LocalConstructor slots to K/V (default: True)
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

    # Print configuration once (rank 0, layer 0 only)
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

            print("=" * 80 + "\n", flush=True)

        self._hierarchical_no_cache_printed = True

    # ========== Step 1: segment the sequence into chunks ==========
    # SFT sequences may have irregular lengths (not divisible by 4096).
    # When the length is irregular, find the divisor of q_len that is closest
    # to the target group size (q_len * group_size_ratio), requiring no padding.
    if q_len % 4096 == 0:
        group_size = int(q_len * group_size_ratio)
    else:
        target_group_size = int(q_len * group_size_ratio)

        # Enumerate all divisors of q_len
        divisors = []
        for i in range(1, int(q_len**0.5) + 1):
            if q_len % i == 0:
                divisors.append(i)
                if i != q_len // i:
                    divisors.append(q_len // i)

        # Filter out divisors that are too small (< 10) or would leave only 1 group
        min_size = 10 if target_group_size >= 10 else 1
        max_size = q_len // 2 if q_len >= 20 else q_len

        valid_divisors = [d for d in divisors if min_size <= d <= max_size]

        if valid_divisors:
            # Choose the divisor closest to the target group size
            group_size = min(valid_divisors, key=lambda x: abs(x - target_group_size))
        else:
            # No suitable divisor; treat the entire sequence as one segment
            group_size = q_len

    if q_len % group_size > 0:
        raise ValueError(
            f"q_len {q_len} should be divisible by group size {group_size}."
        )
    if not hasattr(self, "_hierarchical_group_printed"):
        layer_idx = getattr(self, "layer_idx", 0)
        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank == 0 and layer_idx == 0:
            print(
                f"[forward_flashattn_hierarchical] group_size_ratio={group_size_ratio}, group_size={group_size}"
            )
        self._hierarchical_group_printed = True

    num_groups = q_len // group_size

    # Reshape into chunks: [bsz, num_groups, group_size, hidden_size]
    chunks = hidden_states.view(bsz, num_groups, group_size, hidden_size)

    # attention_mask: [bsz, q_len] -> chunk_masks_reshaped: [bsz, num_groups, group_size]
    # attention_mask may be None during eval (all tokens valid)
    if attention_mask is None:
        attention_mask = torch.ones(
            bsz, q_len, dtype=torch.bool, device=hidden_states.device
        )
    chunk_masks_reshaped = attention_mask.view(bsz, num_groups, group_size)

    # ========== Step 2: Local Construction — extract M local slot representations per chunk ==========
    if (use_global_context or use_local_repr) and hasattr(self, "local_constructor"):
        # Batch all chunks together for parallel processing
        all_chunks = chunks.view(bsz * num_groups, group_size, hidden_size)

        # [bsz, num_groups, group_size] -> [bsz * num_groups, group_size]
        attention_mask_chunks = chunk_masks_reshaped.view(bsz * num_groups, group_size)
        all_local_repr = self.local_constructor(
            all_chunks, attention_mask_chunks
        )  # [bsz * num_groups, num_slots, hidden_size]

        # Reshape back: [bsz, num_groups, num_slots, hidden_size]
        num_local_slots = all_local_repr.shape[1]
        local_repr_stacked = all_local_repr.view(
            bsz, num_groups, num_local_slots, hidden_size
        )
    else:
        num_local_slots = 0
        local_repr_stacked = None

    # ========== Step 3: Global Integration — aggregate local slots into K global context vectors ==========
    if (
        use_global_context
        and hasattr(self, "global_integrator")
        and local_repr_stacked is not None
    ):
        global_context = self.global_integrator(local_repr_stacked)
        # [bsz, global_slots, hidden_size]
        num_global_slots = global_context.shape[1]
    else:
        global_context = None
        num_global_slots = 0

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

    # ========== Step 5: Project G and {Li} to K/V ==========
    # Global context G (output of Global Integration)
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
        # Repeat k/v heads
        global_context_k = repeat_kv(global_context_k, self.num_key_value_groups)
        global_context_v = repeat_kv(global_context_v, self.num_key_value_groups)
    else:
        global_context_k = global_context_v = None

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
    # Project local repr {Li} to K/V only; Q is derived from segment tokens
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

        # Repeat k/v heads for GQA
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

    # ========== Step 7: Top-down Broadcast — attend with Q=[chunk], K/V=[memory, chunk] ==========
    # Flatten query chunks: [bsz, nh, num_groups, group_size, hd]
    #                     -> [bsz * num_groups, group_size, nh, hd]
    all_chunks_q_flat = query_chunks.permute(0, 2, 3, 1, 4).reshape(
        bsz * num_groups, group_size, self.num_heads, self.head_dim
    )

    # Compute total K/V length: prefix (memory slots) + chunk tokens
    prefix_len = 0
    if use_global_context:
        prefix_len += num_global_slots
    if use_local_repr:
        prefix_len += num_local_slots
    kv_len_per_chunk = prefix_len + group_size

    if prefix_len > 0:
        # Pre-allocate K/V tensors: [bsz, nh, num_groups, kv_len_per_chunk, hd]
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
        # Fill global context vectors (shared across all chunks)
        if use_global_context and global_context_k is not None:
            # global_context_k: [bsz, nh, num_global_slots, hd]
            # Broadcast to all chunks: [bsz, nh, num_groups, num_global_slots, hd]
            all_k[:, :, :, offset : offset + num_global_slots, :] = (
                global_context_k.unsqueeze(2)
            )
            all_v[:, :, :, offset : offset + num_global_slots, :] = (
                global_context_v.unsqueeze(2)
            )
            offset += num_global_slots

        # Fill local slot memories (per-chunk)
        if use_local_repr and local_k_all is not None:
            # local_k_all: [bsz, num_groups, nh, num_local_slots, hd]
            # Transpose to: [bsz, nh, num_groups, num_local_slots, hd]
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
        # No memory prefix; use chunk tokens directly
        all_k = key_chunks
        all_v = value_chunks
        kv_len_per_chunk = group_size

    # Reshape for Flash Attention:
    # [bsz, nh, num_groups, kv_len, hd] -> [bsz * num_groups, kv_len, nh, hd]
    all_k_flat = all_k.permute(0, 2, 3, 1, 4).reshape(
        bsz * num_groups, kv_len_per_chunk, self.num_heads, self.head_dim
    )
    all_v_flat = all_v.permute(0, 2, 3, 1, 4).reshape(
        bsz * num_groups, kv_len_per_chunk, self.num_heads, self.head_dim
    )

    # Pack K and V: [bsz * num_groups, kv_len, 2, nh, hd]
    all_chunks_kv_flat = torch.stack([all_k_flat, all_v_flat], dim=2)

    q_len_per_chunk = group_size

    # ========== Step 9: Prepare padding masks (1=valid, 0=padding) ==========
    # Q mask: chunk tokens only, [bsz * num_groups, group_size]
    all_masks_q_flat = chunk_masks_reshaped.reshape(bsz * num_groups, q_len_per_chunk)

    # K/V mask: memory prefix (always valid) + chunk tokens
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

    # Flash Attention: Q=[chunk tokens], K/V=[memory prefix + chunk tokens]
    # causal=True: token i attends to all memory slots and chunk tokens 0..i
    output_unpad = flash_attn_varlen_kvpacked_func(
        q_unpad,      # [total_q_tokens, num_heads, head_dim]
        kv_unpad,     # [total_kv_tokens, 2, num_heads, head_dim]
        cu_seqlens_q,
        cu_seqlens_kv,
        max_seqlen_q,
        max_seqlen_kv,
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

    # Merge chunks: [bsz, num_groups, group_size, nh, hd] -> [bsz, q_len, nh, hd]
    output = output.view(bsz, q_len, self.num_heads, self.head_dim)

    # Output projection
    attn_output = self.o_proj(rearrange(output, "b s h d -> b s (h d)"))

    return attn_output, None, past_key_value

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
    """
    Standard full-attention forward pass (no HiCI, no chunking).

    Used as the baseline evaluation path when eval_mode="full" is specified.
    hidden_states: [bsz, seq_len, hidden_size]
    attention_mask: [bsz, seq_len]
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

    # SFT: use sft_group_size for irregular sequence lengths not divisible by 4096
    if q_len % 4096 == 0:
        group_size = int(q_len * group_size_ratio)
    else:
        group_size = sft_group_size
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


# Inference-specific attention mask preparation: extends the mask to cover cached tokens.
def _prepare_decoder_attention_mask_inference(
    self, attention_mask, input_shape, inputs_embeds, past_key_values_length
):
    """
    Prepare the attention mask for inference with KV cache.

    Unlike the training version, when past_key_values_length > 0 the mask must be
    extended to cover the cached tokens (all of which are valid).

    Args:
        attention_mask: [bsz, seq_len] — mask for the current tokens
        input_shape: (bsz, seq_len)
        inputs_embeds: input embeddings
        past_key_values_length: number of tokens already in the KV cache

    Returns:
        attention_mask: [bsz, past_len + seq_len], or None if all tokens are valid
    """
    if past_key_values_length > 0 and attention_mask is not None:
        # Prepend True mask entries for the cached tokens (they are always valid)
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

    # Return None when all tokens are valid — enables the faster Flash Attention code path
    if attention_mask is not None and torch.all(attention_mask):
        return None

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


# ============================================================================
# HiCI SFT Inference Forward (used for LongBench and similar evaluations)
# ============================================================================
def forward_hici_sft_inference(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    padding_mask: Optional[torch.LongTensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """
    HiCI SFT inference forward — used for LongBench and similar evaluation tasks.

    Inference flow:
    1. Prefill (q_len > 1, past_key_value is None):
       - HiCI hierarchical attention consistent with training.
       - The entire sequence is treated as one chunk (group_size = q_len).
       - Mode 3: Q=[chunk], K/V=[global_context, local_slots, chunk]

    2. Decode (q_len == 1 or past_key_value is not None):
       - Standard Flash Attention with KV cache.
       - HiCI information was fused into token representations during prefill.
    """
    bsz, q_len, hidden_size = hidden_states.size()

    # ========== Decode stage: standard Flash Attention with KV cache ==========
    if q_len == 1 or past_key_value is not None:
        kv_heads = getattr(self, "num_key_value_heads", self.num_heads)

        # Flash Attention expects [bsz, seq_len, num_heads, head_dim]
        q = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(bsz, q_len, kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(bsz, q_len, kv_heads, self.head_dim)

        kv_seq_len = k.shape[1]
        past_kv_len = 0
        if past_key_value is not None:
            past_kv_len = past_key_value[0].shape[2]
            kv_seq_len += past_kv_len

        cos_sin = self.rotary_emb(v, seq_len=kv_seq_len)
        q, k = apply_rotary_pos_emb_inference(q, k, cos_sin, position_ids)

        # Concatenate with KV cache
        if past_key_value is not None:
            # KV cache format: [bsz, num_heads, seq_len, head_dim]
            # Flash Attention format: [bsz, seq_len, num_heads, head_dim]
            k = torch.cat([past_key_value[0].transpose(1, 2), k], dim=1)
            v = torch.cat([past_key_value[1].transpose(1, 2), v], dim=1)

        # Update KV cache (store in [bsz, num_heads, seq_len, head_dim] format)
        past_key_value = (k.transpose(1, 2), v.transpose(1, 2)) if use_cache else None

        # When INCLUDE_HICI_IN_KV_CACHE=True, the KV cache includes HiCI slot prefixes.
        # Extend the attention_mask to cover the prefix (all slots are always valid).
        if INCLUDE_HICI_IN_KV_CACHE and hasattr(self, "_hici_cache_prefix_len"):
            prefix_len = getattr(self, "_hici_cache_prefix_len", 0)
            if prefix_len > 0 and attention_mask is not None:
                hici_mask = torch.ones(
                    bsz,
                    prefix_len,
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                attention_mask = torch.cat([hici_mask, attention_mask], dim=1)

                if not hasattr(self, "_hici_mask_fix_printed"):
                    layer_idx = getattr(self, "layer_idx", 0)
                    if layer_idx == 0:
                        print(
                            f"[HiCI Decode] Extended attention_mask: added {prefix_len} HiCI slots, "
                            f"new shape={attention_mask.shape}"
                        )
                    self._hici_mask_fix_printed = True

        if attention_mask is None:
            output = flash_attn_func(
                q, k, v, 0.0, softmax_scale=None, causal=True
            ).view(bsz, q_len, -1)
        else:
            q_unpad, indices, cu_q_lens, max_s = unpad_input(
                q, attention_mask[:, -q_len:]
            )
            kv, _, cu_k_lens, max_k = unpad_input(
                torch.stack((k, v), dim=2), attention_mask
            )
            output_unpad = flash_attn_varlen_kvpacked_func(
                q_unpad,
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

        attn_output = self.o_proj(output)

        return attn_output, None, past_key_value

    # ========== Prefill stage ==========
    global _HICI_INFERENCE_PRINTED, _HICI_GROUP_PRINTED, _HICI_CACHE_PRINTED

    if DISABLE_HICI_IN_PREFILL:
        if not _HICI_INFERENCE_PRINTED:
            layer_idx = getattr(self, "layer_idx", 0)
            if layer_idx == 0:
                print("\n" + "=" * 80)
                print("HiCI SFT Inference - DISABLE_HICI_IN_PREFILL=True")
                print("Using standard Flash Attention (no HiCI)")
                print("=" * 80 + "\n", flush=True)
                _HICI_INFERENCE_PRINTED = True

        # Standard Flash Attention — no HiCI (debug/ablation path)
        kv_heads = getattr(self, "num_key_value_heads", self.num_heads)

        q = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(bsz, q_len, kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(bsz, q_len, kv_heads, self.head_dim)

        cos_sin = self.rotary_emb(v, seq_len=q_len)
        q, k = apply_rotary_pos_emb_inference(q, k, cos_sin, position_ids)

        past_key_value = (k.transpose(1, 2), v.transpose(1, 2)) if use_cache else None

        if attention_mask is None:
            output = flash_attn_func(
                q, k, v, 0.0, softmax_scale=None, causal=True
            ).view(bsz, q_len, -1)
        else:
            q_unpad, indices, cu_q_lens, max_s = unpad_input(q, attention_mask)
            kv, _, cu_k_lens, max_k = unpad_input(
                torch.stack((k, v), dim=2), attention_mask
            )
            output_unpad = flash_attn_varlen_kvpacked_func(
                q_unpad,
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

        attn_output = self.o_proj(output)
        return attn_output, None, past_key_value

    # ========== Prefill stage: HiCI hierarchical attention ==========
    if not _HICI_INFERENCE_PRINTED:
        layer_idx = getattr(self, "layer_idx", 0)
        if layer_idx == 0:
            print("\n" + "=" * 80)
            print("HiCI SFT Inference Mode (LongBench)")
            print("=" * 80)
            print(f"  group_size_ratio  : {group_size_ratio}")
            print("  Prefill: HiCI hierarchical (global_context + local_slots)")
            print("  Decode:  standard attention + KV cache")
            print("=" * 80 + "\n", flush=True)
            _HICI_INFERENCE_PRINTED = True

    # Treat the entire prefill sequence as a single chunk (no segmentation at inference).
    # Alternative grouping strategies are commented out below for reference.
    #
    # Option A (active): no segmentation — entire sequence is one chunk
    group_size = q_len
    num_groups = 1
    #
    # Option B: segment by group_size_ratio (falls back to 1 group if not divisible)
    # group_size = int(q_len * group_size_ratio) if q_len * group_size_ratio >= 1 else q_len
    # if q_len % group_size != 0:
    #     group_size = q_len
    # num_groups = q_len // group_size
    #
    # Option C: training-consistent segmentation (find nearest valid divisor)
    # if q_len % 1024 == 0:
    #     group_size = int(q_len * group_size_ratio)
    # else:
    #     target_group_size = int(q_len * group_size_ratio)
    #     divisors = [i for i in range(1, int(q_len**0.5) + 1) if q_len % i == 0]
    #     divisors += [q_len // i for i in divisors if i != q_len // i]
    #     valid_divisors = [d for d in divisors if 10 <= d <= q_len // 2]
    #     group_size = min(valid_divisors, key=lambda x: abs(x - target_group_size)) if valid_divisors else q_len
    # if q_len % group_size != 0:
    #     group_size = q_len
    # num_groups = q_len // group_size

    if not _HICI_GROUP_PRINTED:
        layer_idx = getattr(self, "layer_idx", 0)
        if layer_idx == 0:
            print(
                f"[HiCI Inference] q_len={q_len}, group_size={group_size}, num_groups={num_groups}"
            )
            _HICI_GROUP_PRINTED = True

    # Reshape into chunks: [bsz, num_groups, group_size, hidden_size]
    chunks = hidden_states.view(bsz, num_groups, group_size, hidden_size)

    # Build per-chunk attention masks
    if attention_mask is not None and attention_mask.dim() == 2:
        chunk_masks = attention_mask.view(bsz, num_groups, group_size)
    else:
        chunk_masks = torch.ones(
            bsz, num_groups, group_size, dtype=torch.bool, device=hidden_states.device
        )

    # ========== Local Construction ==========
    use_global_context = True
    use_local_repr = True

    if hasattr(self, "local_constructor"):
        all_chunks = chunks.view(bsz * num_groups, group_size, hidden_size)
        mask_chunks = chunk_masks.view(bsz * num_groups, group_size)
        all_local_repr = self.local_constructor(all_chunks, mask_chunks)
        num_local_slots = all_local_repr.shape[1]
        local_repr_stacked = all_local_repr.view(
            bsz, num_groups, num_local_slots, hidden_size
        )
    else:
        num_local_slots = 0
        local_repr_stacked = None

    # ========== Global Integration ==========
    if hasattr(self, "global_integrator") and local_repr_stacked is not None:
        global_context = self.global_integrator(local_repr_stacked)
        num_global_slots = global_context.shape[1]
    else:
        global_context = None
        num_global_slots = 0

    # Q/K/V projections
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

    # Apply RoPE to token Q/K (memory slots do not carry positional encoding)
    cos, sin = self.rotary_emb(value_states, seq_len=q_len)
    query_states, key_states = apply_rotary_pos_emb(
        query_states, key_states, cos, sin, position_ids
    )

    # Keep pre-repeat versions for KV cache storage (num_key_value_heads)
    key_states_for_cache = key_states
    value_states_for_cache = value_states

    # Repeat k/v heads for GQA (num_heads)
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    # Project memory slots to K/V
    if global_context is not None:
        global_context_k_cache = (
            self.k_proj(global_context)
            .view(bsz, num_global_slots, self.num_key_value_heads, self.head_dim)
            .transpose(1, 2)
        )
        global_context_v_cache = (
            self.v_proj(global_context)
            .view(bsz, num_global_slots, self.num_key_value_heads, self.head_dim)
            .transpose(1, 2)
        )
        global_context_k = repeat_kv(global_context_k_cache, self.num_key_value_groups)
        global_context_v = repeat_kv(global_context_v_cache, self.num_key_value_groups)
    else:
        global_context_k = global_context_v = None
        global_context_k_cache = global_context_v_cache = None

    # Reshape into chunks
    q_chunks = query_states.view(
        bsz, self.num_heads, num_groups, group_size, self.head_dim
    )
    k_chunks = key_states.view(
        bsz, self.num_heads, num_groups, group_size, self.head_dim
    )
    v_chunks = value_states.view(
        bsz, self.num_heads, num_groups, group_size, self.head_dim
    )

    # Project local repr {Li} to K/V
    if local_repr_stacked is not None:
        local_mems_flat = local_repr_stacked.view(bsz * num_groups, num_local_slots, hidden_size)
        lm_k_cache = (
            self.k_proj(local_mems_flat)
            .view(
                bsz * num_groups,
                num_local_slots,
                self.num_key_value_heads,
                self.head_dim,
            )
            .transpose(1, 2)
        )
        lm_v_cache = (
            self.v_proj(local_mems_flat)
            .view(
                bsz * num_groups,
                num_local_slots,
                self.num_key_value_heads,
                self.head_dim,
            )
            .transpose(1, 2)
        )
        lm_k = repeat_kv(lm_k_cache, self.num_key_value_groups)
        lm_v = repeat_kv(lm_v_cache, self.num_key_value_groups)
        # Use reshape instead of view to handle non-contiguous tensors from repeat_kv
        lm_k = lm_k.reshape(
            bsz, num_groups, self.num_heads, num_local_slots, self.head_dim
        )
        lm_v = lm_v.reshape(
            bsz, num_groups, self.num_heads, num_local_slots, self.head_dim
        )
        # Reshape for KV cache (num_groups=1 when using full sequence as one chunk)
        lm_k_cache = lm_k_cache.reshape(
            bsz, num_groups * num_local_slots, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        lm_v_cache = lm_v_cache.reshape(
            bsz, num_groups * num_local_slots, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
    else:
        lm_k = lm_v = None
        lm_k_cache = lm_v_cache = None

    # ========== Build KV cache ==========
    if use_cache:
        if INCLUDE_HICI_IN_KV_CACHE and (
            global_context_k_cache is not None or lm_k_cache is not None
        ):
            # KV cache = [global_context, local_slots, tokens]
            cache_components_k = []
            cache_components_v = []
            cache_prefix_len = 0

            if global_context_k_cache is not None:
                cache_components_k.append(global_context_k_cache)
                cache_components_v.append(global_context_v_cache)
                cache_prefix_len += num_global_slots
            if lm_k_cache is not None:
                cache_components_k.append(lm_k_cache)
                cache_components_v.append(lm_v_cache)
                cache_prefix_len += num_groups * num_local_slots

            cache_components_k.append(key_states_for_cache)
            cache_components_v.append(value_states_for_cache)

            full_key_cache = torch.cat(cache_components_k, dim=2)
            full_value_cache = torch.cat(cache_components_v, dim=2)
            past_key_value = (full_key_cache, full_value_cache)

            # Record the HiCI prefix length for the decode stage attention_mask extension
            self._hici_cache_prefix_len = cache_prefix_len

            if not _HICI_CACHE_PRINTED:
                layer_idx = getattr(self, "layer_idx", 0)
                if layer_idx == 0:
                    print(f"\n{'=' * 60}")
                    print(f"[HiCI Prefill] INCLUDE_HICI_IN_KV_CACHE=True")
                    print(f"  global_context slots : {num_global_slots}")
                    print(f"  local_constructor slots : {num_groups * num_local_slots}")
                    print(f"  prefix_len total    : {cache_prefix_len}")
                    print(f"  token_len           : {q_len}")
                    print(f"  KV cache length     : {full_key_cache.shape[2]}")
                    print(f"{'=' * 60}\n")
                    _HICI_CACHE_PRINTED = True
        else:
            # KV cache = [tokens] only
            past_key_value = (key_states_for_cache, value_states_for_cache)
            self._hici_cache_prefix_len = 0

            if not _HICI_CACHE_PRINTED:
                layer_idx = getattr(self, "layer_idx", 0)
                if layer_idx == 0:
                    print(
                        f"[HiCI KV Cache] INCLUDE_HICI_IN_KV_CACHE=False: storing token KV only"
                    )
                    _HICI_CACHE_PRINTED = True
    else:
        past_key_value = None

    # ========== Top-down Broadcast: build K/V = [global_context, local_slots, chunk] ==========
    prefix_len = num_global_slots + num_local_slots
    kv_len = prefix_len + group_size

    if prefix_len > 0:
        all_k = torch.empty(
            bsz,
            self.num_heads,
            num_groups,
            kv_len,
            self.head_dim,
            dtype=k_chunks.dtype,
            device=k_chunks.device,
        )
        all_v = torch.empty(
            bsz,
            self.num_heads,
            num_groups,
            kv_len,
            self.head_dim,
            dtype=v_chunks.dtype,
            device=v_chunks.device,
        )

        offset = 0
        if global_context_k is not None:
            all_k[:, :, :, offset : offset + num_global_slots, :] = global_context_k.unsqueeze(2)
            all_v[:, :, :, offset : offset + num_global_slots, :] = global_context_v.unsqueeze(2)
            offset += num_global_slots
        if lm_k is not None:
            all_k[:, :, :, offset : offset + num_local_slots, :] = lm_k.permute(
                0, 2, 1, 3, 4
            )
            all_v[:, :, :, offset : offset + num_local_slots, :] = lm_v.permute(
                0, 2, 1, 3, 4
            )
            offset += num_local_slots
        all_k[:, :, :, offset:, :] = k_chunks
        all_v[:, :, :, offset:, :] = v_chunks
    else:
        all_k = k_chunks
        all_v = v_chunks
        kv_len = group_size

    # Flash Attention
    q_flat = q_chunks.permute(0, 2, 3, 1, 4).reshape(
        bsz * num_groups, group_size, self.num_heads, self.head_dim
    )
    k_flat = all_k.permute(0, 2, 3, 1, 4).reshape(
        bsz * num_groups, kv_len, self.num_heads, self.head_dim
    )
    v_flat = all_v.permute(0, 2, 3, 1, 4).reshape(
        bsz * num_groups, kv_len, self.num_heads, self.head_dim
    )
    kv_flat = torch.stack([k_flat, v_flat], dim=2)

    # Masks
    q_mask = chunk_masks.reshape(bsz * num_groups, group_size)
    kv_mask = torch.ones(
        bsz * num_groups, kv_len, dtype=q_mask.dtype, device=q_mask.device
    )
    kv_mask[:, prefix_len:] = q_mask

    # Unpad
    q_unpad, idx_q, cu_q, max_q = unpad_input(
        rearrange(q_flat, "b s h d -> b s (h d)"), q_mask
    )
    q_unpad = rearrange(q_unpad, "nnz (h d) -> nnz h d", h=self.num_heads)

    kv_2d = rearrange(kv_flat, "b s two h d -> b s (two h d)")
    kv_unpad, idx_kv, cu_kv, max_kv = unpad_input(kv_2d, kv_mask)
    kv_unpad = rearrange(
        kv_unpad, "nnz (two h d) -> nnz two h d", two=2, h=self.num_heads
    )

    # Flash attention
    out_unpad = flash_attn_varlen_kvpacked_func(
        q_unpad,
        kv_unpad,
        cu_q,
        cu_kv,
        max_q,
        max_kv,
        dropout_p=0.0,
        softmax_scale=None,
        causal=True,
    )

    # Pad back
    output = rearrange(
        pad_input(
            rearrange(out_unpad, "nnz h d -> nnz (h d)"),
            idx_q,
            bsz * num_groups,
            group_size,
        ),
        "b s (h d) -> b s h d",
        h=self.num_heads,
    )
    output = output.view(bsz, q_len, self.num_heads, self.head_dim)
    attn_output = self.o_proj(rearrange(output, "b s h d -> b s (h d)"))

    return attn_output, None, past_key_value


def replace_llama_attn_hici_inference():
    """
    Replace LlamaAttention.forward with the HiCI SFT inference function.
    Used for LongBench and similar evaluation tasks.

    Prefill: forward_hici_sft_inference (full sequence as one chunk, two-level HiCI)
    Decode:  standard KV-cache attention
    """
    print("=" * 80)
    print("Replacing LlamaAttention.forward with HiCI SFT Inference")
    print(f"  group_size_ratio : {group_size_ratio}")
    print("=" * 80)

    transformers.models.llama.modeling_llama.LlamaModel._prepare_decoder_attention_mask = _prepare_decoder_attention_mask_inference
    transformers.models.llama.modeling_llama.LlamaAttention.forward = (
        forward_hici_sft_inference
    )


def replace_llama_attn(
    use_flash_attn=True,
    use_full=False,
    inference=False,
    eval_mode=None,
    use_hierarchical_forward: bool = True,
):
    """
    Replace LlamaAttention forward function with HiCI implementations.

    IMPORTANT: This function only patches the forward method.
    To register HiCI parameters, call register_hici_to_model() after loading
    the model and before initializing the optimizer.

    Args:
        use_flash_attn: Whether to use flash attention (default: True)
        use_full: Kept for backward compatibility. Use eval_mode="full" instead.
        inference: Whether in inference mode — uses forward_flashattn_inference
        eval_mode: Evaluation mode (default: None = HiCI hierarchical)
            - None: HiCI hierarchical attention (same as training)
            - "full": Full attention without HiCI
        use_hierarchical_forward: Use forward_flashattn_hierarchical (default: True)
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

            if eval_mode == "full" or (eval_mode is None and use_full):
                # Full attention without HiCI (baseline evaluation)
                transformers.models.llama.modeling_llama.LlamaAttention.forward = (
                    forward_flashattn_full
                )
                if rank == 0:
                    print("   forward: forward_flashattn_full (full attention, no HiCI)")
            else:
                # HiCI hierarchical attention (training path and default eval)
                transformers.models.llama.modeling_llama.LlamaAttention.forward = (
                    forward_flashattn_hierarchical
                )
                if rank == 0:
                    print("   forward: forward_flashattn_hierarchical (LocalConstructor + GlobalIntegrator)")
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
    use_llama_init=False,  # warm-start Q/K/V projections from LLaMA pretrained weights
    use_shared_compressor=True,  # use GlobalIntegratorShared (–71% parameters vs. GlobalIntegrator)
    shared_compress_dim=128,  # intermediate dimension of the shared compressor
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

    Example usage in fine-tune.py:
        # 1. Load model
        model = transformers.AutoModelForCausalLM.from_pretrained(...)

        # 2. Replace attention mechanism
        replace_llama_attn(use_flash_attn=True)

        # 3. Register global memory (BEFORE optimizer!)
        # For simple global memory:
        register_hici_to_model(model, num_local_slots=8)

        # For hierarchical memory:
        register_hici_to_model(
            model,
            num_local_slots=8,   # local slots
            global_slots=4,        # higher-level global slots
            use_global_integrator=True
        )

        # 4. Setup LoRA (if needed)
        model = get_peft_model(model, lora_config)

        # 5. NOW initialize optimizer (will include global memory parameters)
        optimizer = torch.optim.AdamW(model.parameters(), lr=...)
    """
    rank = dist.get_rank() if dist.is_initialized() else 0

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
            config_str.append("GlobalIntegrator")

        if config_str:
            print(f"Registering: {' + '.join(config_str)}")
        else:
            print("Warning: no HiCI modules enabled")
        print("=" * 80)

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

    # Ensure HiCI modules are created in the same dtype as the base model
    model_dtype = llama_model.embed_tokens.weight.dtype
    rank = dist.get_rank() if dist.is_initialized() else 0
    if rank == 0:
        print(f"   Model dtype: {model_dtype}")

    # Embedding weights used for optional slot initialization
    embed_weight = llama_model.embed_tokens.weight.data  # [vocab_size, hidden_size]

    # Register modules to each attention layer
    for layer_idx, layer in enumerate(llama_model.layers):
        attn = layer.self_attn
        attn.layer_idx = layer_idx  # Important for layer identification

        # Module 1: LocalConstructor
        if use_local_constructor:
            if use_local_constructor_flash:
                # LocalConstructorFlash: independent Q/K/V projections, Flash Attention
                attn.local_constructor = LocalConstructorFlash(
                    hidden_size=hidden_size,
                    num_local_slots=num_local_slots,
                    num_heads=num_heads,
                    init_from_embeddings=embed_weight,
                ).to(model_dtype)
            else:
                # LocalConstructorMulti: standard PyTorch multi-head cross-attention
                # Compatible with forward_flashattn_hierarchical
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
                # GlobalIntegratorShared: shared compressor, ~4.0M params/layer
                attn.global_integrator = GlobalIntegratorShared(
                    hidden_size=hidden_size,
                    global_slots=global_slots,
                    compress_dim=512,
                    shared_compress_dim=shared_compress_dim,
                    init_from_embeddings=embed_weight,
                    use_high_norm_init=True,
                ).to(model_dtype)
            else:
                # GlobalIntegrator: independent per-statistic compressors, ~13.7M params/layer
                attn.global_integrator = GlobalIntegrator(
                    hidden_size=hidden_size,
                    global_slots=global_slots,
                    compress_dim=512,
                    num_heads=num_heads,
                    init_from_embeddings=embed_weight,
                    use_high_norm_init=True,
                ).to(model_dtype)

    # Verify registration and report parameter counts
    total_params = sum(p.numel() for p in model.parameters())

    # Use named_parameters() to avoid double-counting shared parameters
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
        print("HiCI Module Registration Complete")
        print("=" * 80)

        print(f"Model: {total_params:,} params ({total_params / 1e9:.2f}B)")
        print(f"Layers: {len(llama_model.layers)}")

        if use_local_constructor and use_global_integrator:
            total_hici_params = local_constructor_params + aggregator_params
            print(f"\nRegistered Modules:")
            print(f"  LocalConstructor  ({local_constructor_params:,} params)")
            print(f"  GlobalIntegrator  ({aggregator_params:,} params)")
            print(
                f"\nTotal HiCI Params: {total_hici_params:,} ({total_hici_params / total_params * 100:.2f}%)"
            )

        elif use_local_constructor and not use_global_integrator:
            print(f"\nRegistered Modules:")
            print(f"  LocalConstructor  ({local_constructor_params:,} params)")
            print(
                f"\nTotal HiCI Params: {local_constructor_params:,} ({local_constructor_params / total_params * 100:.2f}%)"
            )

        elif not use_local_constructor and use_global_integrator:
            print(f"\nWarning: GlobalIntegrator registered without LocalConstructor")
            print(f"  GlobalIntegrator  ({aggregator_params:,} params)")
            print(
                f"\nTotal HiCI Params: {aggregator_params:,} ({aggregator_params / total_params * 100:.2f}%)"
            )

        else:
            print(f"\nRegistered Modules: None")

        print("=" * 80 + "\n")
