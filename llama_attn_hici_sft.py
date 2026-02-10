# Modified based on https://github.com/lm-sys/FastChat
#
#
#

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

group_size_ratio = 1  # Fraction of tokens in each group for local representation
sft_group_size = 8192  # Fixed group size for SFT (handles irregular sequence lengths)

# ============================================================================
# ============================================================================
# True:  KV cache = [higher_global, local_repr, tokens]
# False: KV cache = [tokens]
INCLUDE_REPR_IN_KV_CACHE = True

DISABLE_HICI_IN_PREFILL = False

_HICI_INFERENCE_PRINTED = False
_HICI_GROUP_PRINTED = False
_HICI_CACHE_PRINTED = False


class LocalConstructor(nn.Module):
    """
    Learnable HiCI for capturing document-level context.

    This module is registered as a sub-module of LlamaAttention, ensuring:
    1. Parameters are properly registered in model.parameters()
    2. Optimizer tracks and updates these parameters
    3. Saved/loaded with model checkpoints

    Args:
        hidden_size: Model hidden dimension (e.g., 4096 for Llama-2-7B)
        num_local_slots: Number of learnable local query slots (default: 16)
    """

    def __init__(
        self,
        hidden_size,
        num_local_slots=16,
        num_heads: Optional[int] = None,
        init_from_embeddings=None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_local_slots = num_local_slots

        # Learnable local query slots: [num_slots, hidden_size]
        # if init_from_embeddings is not None:
        if False:
            indices = torch.randperm(init_from_embeddings.size(0))[:num_local_slots]
            self.local_queries = nn.Parameter(init_from_embeddings[indices].clone())
            rank = dist.get_rank() if dist.is_initialized() else 0
            if rank == 0:
                print(
                    f"     Initialized local_queries from pretrained embeddings (sampled {num_local_slots} tokens)"
                )
        else:
            std = 1.0 / math.sqrt(hidden_size)
            self.local_queries = nn.Parameter(
                torch.randn(num_local_slots, hidden_size) * std
            )
            rank = dist.get_rank() if dist.is_initialized() else 0
            if rank == 0:
                print(f"      Fallback: Initialized local_queries with std={std}")

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
            global_repr: [bsz, num_slots, hidden_size] - global summary
        """
        bsz, seq_len, _ = hidden_states.shape

        # Expand queries for batch
        queries = self.local_queries.unsqueeze(0).expand(
            bsz, -1, -1
        )  # [bsz, num_slots, hidden_size]

        # Cross-attention: queries attend to full sequence
        Q_mem = self.q_proj(queries)  # [bsz, num_slots, hidden_size]
        K_seq = self.k_proj(hidden_states)  # [bsz, seq_len, hidden_size]
        V_seq = self.v_proj(hidden_states)  # [bsz, seq_len, hidden_size]

        # Compute attention scores
        scores = torch.matmul(Q_mem, K_seq.transpose(-2, -1)) / math.sqrt(
            self.hidden_size
        )
        attn_weights = torch.softmax(scores, dim=-1)  # [bsz, num_slots, seq_len]

        # Apply attention to get global context
        global_repr = torch.matmul(
            attn_weights, V_seq
        )  # [bsz, num_slots, hidden_size]

        return global_repr

class LocalConstructorMulti(nn.Module):
    """
    Learnable HiCI for capturing document-level context.


    This module is registered as a sub-module of LlamaAttention, ensuring:
    1. Parameters are properly registered in model.parameters()
    2. Optimizer tracks and updates these parameters
    3. Saved/loaded with model checkpoints

    Args:
        hidden_size: Model hidden dimension (e.g., 4096 for Llama-2-7B)
        num_local_slots: Number of learnable local query slots (default: 16)
        num_heads: Number of attention heads (default: 32)
        init_from_embeddings: Optional pretrained embeddings for local_queries initialization
        init_from_llama_attn: Optional LlamaAttention layer for Q/K/V projection initialization
        use_bottleneck: Whether to use bottleneck compression (default: True)
        bottleneck_dim: Bottleneck dimension (default: 2048)
    """

    _init_msg_printed = False

    def __init__(
        self,
        hidden_size,
        num_local_slots=16,
        num_heads=32,
        init_from_embeddings=None,
        init_from_llama_attn=None,
        use_bottleneck: Optional[bool] = True,
        bottleneck_dim: Optional[int] = 2048,
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

        # Learnable local query slots: [num_slots, hidden_size]
        if False:
            indices = torch.randperm(init_from_embeddings.size(0))[:num_local_slots]
            self.local_queries = nn.Parameter(init_from_embeddings[indices].clone())
            rank = dist.get_rank() if dist.is_initialized() else 0
            if rank == 0:
                print(
                    f"     Initialized local_queries from pretrained embeddings (sampled {num_local_slots} tokens)"
                )
        else:
            std = 1.0 / math.sqrt(hidden_size)  # ≈ 0.0156
            self.local_queries = nn.Parameter(
                torch.randn(num_local_slots, hidden_size) * std
            )
            rank = dist.get_rank() if dist.is_initialized() else 0
            if rank == 0 and not LocalConstructorMulti._init_msg_printed:
                print(
                    f"  LocalConstructorMulti Fallback: Initialized local_queries with std={std}"
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
                    f" LocalConstructorMulti: bottleneck_dim: {bottleneck_dim}, num_heads: {num_heads}"
                )

            self.q_proj = nn.Linear(hidden_size, bottleneck_dim, bias=False)
            self.k_proj = nn.Linear(hidden_size, bottleneck_dim, bias=False)
            self.v_proj = nn.Linear(hidden_size, bottleneck_dim, bias=False)

            self.o_proj = nn.Linear(bottleneck_dim, hidden_size, bias=False)

            # Effective dimensions for attention computation
            self.effective_dim = bottleneck_dim
            self.effective_head_dim = bottleneck_dim // num_heads
        else:
            # Standard full-size projections: 4096 -> 4096
            self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
            self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
            self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
            self.o_proj = None

            # Use original dimensions
            self.effective_dim = hidden_size
            self.effective_head_dim = self.head_dim

        if init_from_llama_attn is not None and not use_bottleneck:
            rank = dist.get_rank() if dist.is_initialized() else 0
            layer_idx = getattr(self, "layer_idx", 0)
            with torch.no_grad():
                self.q_proj.weight.copy_(init_from_llama_attn.q_proj.weight)
                self.k_proj.weight.copy_(init_from_llama_attn.k_proj.weight)
                self.v_proj.weight.copy_(init_from_llama_attn.v_proj.weight)
            if rank == 0 and layer_idx == 0:
                print(
                f" [LocalConstructorMulti C] Initialized Q/K/V projections from LLaMA pretrained weights"
                )

    def forward(self, hidden_states, attention_mask=None):
        """
        Compute global context via multi-head cross-attention (standard PyTorch, no Flash Attention).


        Args:
            hidden_states: [bsz, seq_len, hidden_size] - full input sequence
            attention_mask: [bsz, seq_len] - 1 for valid tokens, 0 for padding (optional)

        Returns:
            global_repr: [bsz, num_slots, hidden_size] - global summary
        """
        bsz, seq_len, _ = hidden_states.shape

        # Expand queries for batch
        queries = self.local_queries.unsqueeze(0).expand(
            bsz, -1, -1
        )  # [bsz, num_slots, hidden_size]

        Q_mem = self.q_proj(queries)  # [bsz, num_slots, effective_dim]
        K_seq = self.k_proj(hidden_states)  # [bsz, seq_len, effective_dim]
        V_seq = self.v_proj(hidden_states)  # [bsz, seq_len, effective_dim]

        # Reshape for multi-head attention: [bsz, seqlen, num_heads, effective_head_dim]
        Q_mem = Q_mem.view(
            bsz, self.num_local_slots, self.num_heads, self.effective_head_dim
        )
        K_seq = K_seq.view(bsz, seq_len, self.num_heads, self.effective_head_dim)
        V_seq = V_seq.view(bsz, seq_len, self.num_heads, self.effective_head_dim)

        # Transpose for attention: [bsz, num_heads, seqlen, head_dim]
        Q_mem = Q_mem.transpose(1, 2)  # [bsz, num_heads, num_slots, effective_head_dim]
        K_seq = K_seq.transpose(1, 2)  # [bsz, num_heads, seq_len, effective_head_dim]
        V_seq = V_seq.transpose(1, 2)  # [bsz, num_heads, seq_len, effective_head_dim]

        # Compute attention scores: Q @ K^T
        # [bsz, num_heads, num_slots, effective_head_dim] @ [bsz, num_heads, effective_head_dim, seq_len]
        # -> [bsz, num_heads, num_slots, seq_len]
        scores = torch.matmul(Q_mem, K_seq.transpose(-2, -1)) / math.sqrt(
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
        global_repr = attn_output.view(
            bsz, self.num_local_slots, self.effective_dim
        )

        # Apply output projection if using bottleneck: effective_dim -> hidden_size
        if self.o_proj is not None:
            global_repr = self.o_proj(
                global_repr
            )  # [bsz, num_slots, hidden_size]

        return global_repr

class LocalConstructorFlash(nn.Module):
    """
    Learnable HiCI for capturing document-level context.


    This module is registered as a sub-module of LlamaAttention, ensuring:
    1. Parameters are properly registered in model.parameters()
    2. Optimizer tracks and updates these parameters
    3. Saved/loaded with model checkpoints

    Args:
        hidden_size: Model hidden dimension (e.g., 4096 for Llama-2-7B)
        num_local_slots: Number of learnable local query slots (default: 16)
        num_heads: Number of attention heads (default: 32, for Flash Attention)
    """

    def __init__(
        self, hidden_size, num_local_slots=16, num_heads=32, init_from_embeddings=None
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_local_slots = num_local_slots
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        assert hidden_size % num_heads == 0, (
            f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})"
        )

        # Learnable local query slots: [num_slots, hidden_size]
        # if init_from_embeddings is not None:
        if False:
            indices = torch.randperm(init_from_embeddings.size(0))[:num_local_slots]
            self.local_queries = nn.Parameter(init_from_embeddings[indices].clone())
            rank = dist.get_rank() if dist.is_initialized() else 0
            if rank == 0:
                print(
                    f"     Initialized local_queries from pretrained embeddings (sampled {num_local_slots} tokens)"
                )
        else:
            std = 1.0 / math.sqrt(hidden_size)
            self.local_queries = nn.Parameter(
                torch.randn(num_local_slots, hidden_size) * std
            )
            rank = dist.get_rank() if dist.is_initialized() else 0
            if rank == 0:
                print(f"      Fallback: Initialized local_queries with std={std}")

        # Cross-attention projections for summarization
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, hidden_states, attention_mask=None):
        """
        Compute global context via Flash Attention cross-attention.


        Args:
            hidden_states: [bsz, seq_len, hidden_size] - full input sequence
            attention_mask: [bsz, seq_len] - 1 for valid, 0 for padding (optional)

        Returns:
            global_repr: [bsz, num_slots, hidden_size] - global summary
        """
        bsz, seq_len, _ = hidden_states.shape

        # Expand queries for batch
        queries = self.local_queries.unsqueeze(0).expand(
            bsz, -1, -1
        )  # [bsz, num_slots, hidden_size]

        # Cross-attention projections
        Q_mem = self.q_proj(queries)  # [bsz, num_slots, hidden_size]
        K_seq = self.k_proj(hidden_states)  # [bsz, seq_len, hidden_size]
        V_seq = self.v_proj(hidden_states)  # [bsz, seq_len, hidden_size]

        # Reshape for multi-head attention: [bsz, seqlen, num_heads, head_dim]
        Q_mem = Q_mem.view(bsz, self.num_local_slots, self.num_heads, self.head_dim)
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

            # Q has no padding, flatten: [bsz, num_slots, h, d] -> [bsz * num_slots, h, d]
            q_unpad = rearrange(Q_mem, "b s h d -> (b s) h d")

            # e.g., bsz=2, num_slots=16 -> cu_seqlens_q = [0, 16, 32]
            cu_seqlens_q = torch.arange(
                0,
                (bsz + 1) * self.num_local_slots,
                self.num_local_slots,
                device=hidden_states.device,
                dtype=torch.int32,
            )
            # Q: [total_q, num_heads, head_dim] where total_q = bsz * num_slots
            # KV: [total_kv, 2, num_heads, head_dim] where total_kv = sum of valid lengths
            output_unpad = flash_attn_varlen_kvpacked_func(
                q_unpad,  # [bsz * num_slots, num_heads, head_dim]
                kv_unpad,  # [total_valid_kv, 2, num_heads, head_dim]
                cu_seqlens_q,  # [bsz + 1]
                cu_seqlens_kv,  # [bsz + 1]
                self.num_local_slots,
                max_seqlen_kv,
                dropout_p=0.0,
                softmax_scale=None,
                causal=False,
            )
            # output_unpad: [bsz * num_slots, num_heads, head_dim]

            # Reshape back: [bsz, num_slots, hidden_size]
            global_repr = rearrange(
                output_unpad, "(b s) h d -> b s (h d)", b=bsz, s=self.num_local_slots
            )
        else:
            global_repr = flash_attn_func(
                Q_mem,  # [bsz, num_slots, num_heads, head_dim]
                K_seq,  # [bsz, seq_len, num_heads, head_dim]
                V_seq,  # [bsz, seq_len, num_heads, head_dim]
                dropout_p=0.0,
                softmax_scale=None,
                causal=False,
            )
            # global_repr: [bsz, num_slots, num_heads, head_dim]

            # Reshape: [bsz, num_slots, hidden_size]
            global_repr = rearrange(global_repr, "b s h d -> b s (h d)")

        return global_repr

class LocalConstructorFlashPlus(nn.Module):
    """
    Learnable HiCI for capturing document-level context.


    Args:
        hidden_size: Model hidden dimension (e.g., 4096 for Llama-2-7B)
        num_local_slots: Number of learnable local query slots (default: 16)
        num_heads: Number of attention heads (default: 32, for Flash Attention)
    """

    def __init__(
        self, hidden_size, num_local_slots=16, num_heads=32, init_from_embeddings=None
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_local_slots = num_local_slots
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        assert hidden_size % num_heads == 0, (
            f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})"
        )

        # Learnable local query slots: [num_slots, hidden_size]
        std = 1.0 / math.sqrt(hidden_size)
        self.local_queries = nn.Parameter(
            torch.randn(num_local_slots, hidden_size) * std
        )
        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank == 0:
            print(
                f"    [LocalConstructorFlashPlus] Initialized local_queries with std={std}"
            )
            print(
            f" [LocalConstructorFlashPlus] LLaMA K/V ， Q "
            )

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, key_states, value_states, attention_mask=None):
        """
        Compute global context via Flash Attention cross-attention.


        Args:
            attention_mask: [bsz, seq_len] - 1 for valid, 0 for padding (optional)

        Returns:
            global_repr: [bsz, num_slots, hidden_size] - global summary
        """
        bsz = key_states.shape[0]

        # Expand queries for batch and project to Q
        queries = self.local_queries.unsqueeze(0).expand(
            bsz, -1, -1
        )  # [bsz, num_slots, hidden_size]

        Q_mem = self.q_proj(queries)  # [bsz, num_slots, hidden_size]

        # Reshape Q for multi-head attention: [bsz, num_slots, num_heads, head_dim]
        Q_mem = Q_mem.view(bsz, self.num_local_slots, self.num_heads, self.head_dim)

        K_seq = key_states
        V_seq = value_states

        if attention_mask is not None:
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
            q_unpad = rearrange(Q_mem, "b s h d -> (b s) h d")

            cu_seqlens_q = torch.arange(
                0,
                (bsz + 1) * self.num_local_slots,
                self.num_local_slots,
                device=key_states.device,
                dtype=torch.int32,
            )

            output_unpad = flash_attn_varlen_kvpacked_func(
                q_unpad,  # [bsz * num_slots, num_heads, head_dim]
                kv_unpad,  # [total_valid_kv, 2, num_heads, head_dim]
                cu_seqlens_q,  # [bsz + 1]
                cu_seqlens_kv,  # [bsz + 1]
                self.num_local_slots,
                max_seqlen_kv,  # max_seqlen_kv
                dropout_p=0.0,
                softmax_scale=None,
                causal=False,
            )

            # Reshape back: [bsz, num_slots, hidden_size]
            global_repr = rearrange(
                output_unpad, "(b s) h d -> b s (h d)", b=bsz, s=self.num_local_slots
            )
        else:
            global_repr = flash_attn_func(
                Q_mem,  # [bsz, num_slots, num_heads, head_dim]
                K_seq,  # [bsz, seq_len, num_heads, head_dim]
                V_seq,  # [bsz, seq_len, num_heads, head_dim]
                dropout_p=0.0,
                softmax_scale=None,
                causal=False,
            )
            # Reshape: [bsz, num_slots, hidden_size]
            global_repr = rearrange(global_repr, "b s h d -> b s (h d)")

        return global_repr

class GlobalIntegrator(nn.Module):
    """


    "We propose a hybrid approach combining deterministic statistical
     aggregation with learned attention-based refinement. First, we extract
     five statistical features from local memories and compress them to a
     low-dimensional bottleneck space. Then, global learned queries attend
     over these compressed statistics to extract document-level context."

        Input:  local_reprs [bsz, num_chunks, local_slots, hidden_size]
        Output: global_repr  [bsz, global_slots, hidden_size]

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
        """
        Args:
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

        self.global_queries = nn.Parameter(torch.zeros(global_slots, compress_dim))

        self.q_proj = nn.Linear(compress_dim, compress_dim, bias=False)
        self.k_proj = nn.Linear(compress_dim, compress_dim, bias=False)
        self.v_proj = nn.Linear(compress_dim, compress_dim, bias=False)
        self.o_proj = nn.Linear(compress_dim, compress_dim, bias=False)

        self.attn_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.expand = nn.Linear(compress_dim, hidden_size, bias=False)
        std_init = 0.02 / math.sqrt(compress_dim)
        nn.init.normal_(self.expand.weight, mean=0.0, std=std_init)

        # softplus(x) ≈ x when x > 0, softplus(0) ≈ 0.693
        init_param = math.log(math.exp(output_scale_init) - 1)  # inverse softplus
        self.expand_scale_param = nn.Parameter(torch.tensor(init_param))

        self._init_weights(init_from_embeddings)
        self._print_init_info()

    @property
    def expand_scale(self) -> torch.Tensor:
        """"""
        return F.softplus(self.expand_scale_param)

    def _init_weights(self, embed_weight: Optional[torch.Tensor] = None):
        """"""
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
        """"""
        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank == 0 and not GlobalIntegrator._init_msg_printed:
            total_params = sum(p.numel() for p in self.parameters())
            print(f" GlobalIntegratorClean initialized (EMA)")
            print(f"       - Design: Statistical Aggregation + Lightweight MHA")
            print(f"       - Global slots: {self.global_slots}")
            print(f"       - Compress dim: {self.compress_dim}")
            print(f"       - Num heads: {self.num_heads}")
            print(f"       - Output scale (init): {self._output_scale_init}")
            print(
                f"       - Params/layer: {total_params:,} ({total_params / 1e6:.1f}M)"
            )
            GlobalIntegrator._init_msg_printed = True

    def forward(self, local_reprs: torch.Tensor) -> torch.Tensor:
        """

        Args:
            local_reprs: [bsz, num_chunks, local_slots, hidden_size]

        Returns:
            G: [bsz, global_slots, hidden_size]

            local_reprs [bsz, C, L, H]
                ↓ reshape
            all_local [bsz, C*L, H]
            stats [bsz, H] × 5
            compressed_stats [bsz, 5, D]
                ↓ Multi-Head Attention
            G_compressed [bsz, G, D]
                ↓ expand + scale
            G [bsz, G, H]
        """
        bsz, num_chunks, local_slots, hidden_size = local_reprs.shape

        # Flatten: [bsz, num_chunks * local_slots, hidden_size]
        all_local = local_reprs.reshape(bsz, -1, hidden_size)

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

        # Q: [bsz, global_slots, compress_dim]
        Q = self.global_queries.unsqueeze(0).expand(bsz, -1, -1)
        Q = self.q_proj(Q)

        # K, V: [bsz, 5, compress_dim]
        K = self.k_proj(compressed_stats)
        V = self.v_proj(compressed_stats)

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

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, self.global_slots, self.compress_dim)

        G_compressed = self.o_proj(attn_output)

        G = self.expand(G_compressed) * self.expand_scale

        return G

class GlobalIntegratorShared(nn.Module):
    """


        Input:  local_reprs [bsz, num_chunks, local_slots, hidden_size]
        Output: global_repr  [bsz, global_slots, hidden_size]
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
        """
        Args:
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
                print(
                    f"  Warning: shared_compress_dim ({shared_compress_dim}) > compress_dim ({compress_dim})"
                )
                print(f"   Setting compress_dim = shared_compress_dim for consistency")
            self.compress_dim = shared_compress_dim

        self.head_dim = self.compress_dim // num_heads

        self.global_queries = nn.Parameter(torch.zeros(global_slots, self.compress_dim))

        self.q_proj = nn.Linear(self.compress_dim, self.compress_dim, bias=False)
        self.k_proj = nn.Linear(self.compress_dim, self.compress_dim, bias=False)
        self.v_proj = nn.Linear(self.compress_dim, self.compress_dim, bias=False)
        self.o_proj = nn.Linear(self.compress_dim, self.compress_dim, bias=False)

        self.attn_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.expand = nn.Linear(self.compress_dim, hidden_size, bias=False)
        std_init = 0.02 / math.sqrt(self.compress_dim)
        nn.init.normal_(self.expand.weight, mean=0.0, std=std_init)

        init_param = math.log(math.exp(output_scale_init) - 1)
        self.expand_scale_param = nn.Parameter(torch.tensor(init_param))

        self._init_weights(init_from_embeddings)
        self._print_init_info()

    @property
    def expand_scale(self) -> torch.Tensor:
        """"""
        return F.softplus(self.expand_scale_param)

    def _init_weights(self, embed_weight: Optional[torch.Tensor] = None):
        """"""
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

                init_compressed = self.shared_compressor(
                    init_embeddings
                )  # [global_slots, 128]
                init_expanded = self.stat_expand(init_compressed)  # [global_slots, 512]
                self.global_queries.copy_(init_expanded)
        else:
            nn.init.xavier_uniform_(self.global_queries)

        for proj in [self.q_proj, self.k_proj, self.v_proj, self.o_proj]:
            nn.init.xavier_uniform_(proj.weight)
        # for proj in [self.q_proj, self.k_proj, self.v_proj]:
        #     nn.init.xavier_uniform_(proj.weight)

    def _print_init_info(self):
        """"""
        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank == 0 and not GlobalIntegratorShared._init_msg_printed:
            total_params = sum(p.numel() for p in self.parameters())

            stat_compress_params = sum(
                p.numel() for p in self.shared_compressor.parameters()
            ) + sum(p.numel() for p in self.stat_expand.parameters())

            print(f" GlobalIntegratorShared initialized ()")

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
                f"       -  Saved {(1 - total_params / 13.7e6) * 100:.0f}% compared to original"
            )
            GlobalIntegratorShared._init_msg_printed = True

    def forward(self, local_reprs: torch.Tensor) -> torch.Tensor:
        """

        Args:
            local_reprs: [bsz, num_chunks, local_slots, hidden_size]

        Returns:
            G: [bsz, global_slots, hidden_size]

            local_reprs [bsz, C, L, H]
                ↓ reshape
            all_local [bsz, C*L, H]
            stats [bsz, H] × 5
            compressed_stats_list: 5 × [bsz, 128]
            expanded_stats_list: 5 × [bsz, 512]
            compressed_stats [bsz, 5, 512]
            G_compressed [bsz, G, D]
                ↓ expand + scale
            G [bsz, G, H]
        """
        bsz, num_chunks, local_slots, hidden_size = local_reprs.shape

        all_local = local_reprs.reshape(bsz, -1, hidden_size)

        mean_pool = all_local.mean(dim=1)
        max_pool, _ = all_local.max(dim=1)
        min_pool, _ = all_local.min(dim=1)

        with torch.amp.autocast(device_type="cuda", enabled=False):
            all_local_fp32 = all_local.float()
            std_pool = all_local_fp32.std(dim=1, unbiased=False).clamp(min=1e-6)
        std_pool = std_pool.to(all_local.dtype)

        norm_mean = F.normalize(mean_pool, dim=-1, p=2, eps=1e-6)

        stats_list = [mean_pool, max_pool, min_pool, std_pool, norm_mean]

        # Stack: [bsz, 5, hidden_size]
        stats_stacked = torch.stack(stats_list, dim=1)
        num_stats = 5

        compressed_stats = self.shared_compressor(
            stats_stacked.view(bsz * num_stats, hidden_size)
        ).view(bsz, num_stats, -1)

        compressed_stats = self.stat_expand(
            compressed_stats.view(bsz * num_stats, -1)
        ).view(bsz, num_stats, self.compress_dim)

        # Q: [bsz, global_slots, compress_dim]
        Q = self.global_queries.unsqueeze(0).expand(bsz, -1, -1)
        Q = self.q_proj(Q)

        K = self.k_proj(compressed_stats)
        V = self.v_proj(compressed_stats)

        Q = Q.view(bsz, self.global_slots, self.num_heads, self.head_dim).transpose(
            1, 2
        )
        K = K.view(bsz, 5, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(bsz, 5, self.num_heads, self.head_dim).transpose(1, 2)
        #  Q: [bsz, num_heads, global_slots, head_dim]

        # Scaled Dot-Product Attention
        scale = self.head_dim**-0.5
        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) * scale
        attn_probs = F.softmax(attn_weights, dim=-1)
        attn_probs = self.attn_dropout(attn_probs)

        #  attn_output: [bsz, num_heads, global_slots, head_dim]
        attn_output = torch.matmul(attn_probs, V)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, self.global_slots, self.compress_dim)

        # Output projection
        G_compressed = self.o_proj(attn_output)

        G = self.expand(G_compressed) * self.expand_scale

        return G

def forward_flashattn(
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

    NEW: Uses HiCI + cross-attention instead of shift operation.
    Benefits:
    - No data duplication (1x computation vs 2x in original)
    - Direct global context injection before each chunk
    - O(M×N + N) complexity where M=num_local_slots (16), N=seq_len

    attention_mask: [bsz, q_len]
    """
    if not self.training:
        warnings.warn(
            "This function should be used just for training as it may exhibit reduced inference performance. For inference, please use forward_flashattn_inference."
        )

    if output_attentions:
        warnings.warn(
            "Output attentions is not supported for patched `LlamaAttention`, returning `None` instead."
        )

    bsz, q_len, _ = hidden_states.size()

    # ========== Step 1: Compute global context (NEW!) ==========
    # This captures document-level information before chunking
    # Note: self.local_constructor is registered via replace_llama_attn()
    global_repr = self.local_constructor(
        hidden_states, attention_mask
    )  # [bsz, num_slots, hidden_size]

    num_local_slots = global_repr.shape[1]

    # ========== Step 2: Standard Q/K/V projections (unchanged) ==========
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
        # reuse k, v, self_attention
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)

    past_key_value = (key_states, value_states) if use_cache else None

    # repeat k/v heads if n_kv_heads < n_heads
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    # ========== Step 3: Inject global context into Q/K/V (NEW!) ==========
    # Convert global_repr to Q/K/V format and prepend to each chunk

    # Project global context through Q/K/V projections
    global_q = (
        self.q_proj(global_repr)
        .view(bsz, num_local_slots, self.num_heads, self.head_dim)
        .transpose(1, 2)
    )  # [bsz, nh, num_slots, hd]
    global_k = (
        self.k_proj(global_repr)
        .view(bsz, num_local_slots, self.num_key_value_heads, self.head_dim)
        .transpose(1, 2)
    )
    global_v = (
        self.v_proj(global_repr)
        .view(bsz, num_local_slots, self.num_key_value_heads, self.head_dim)
        .transpose(1, 2)
    )

    # Repeat k/v heads for global context
    global_k = repeat_kv(global_k, self.num_key_value_groups)
    global_v = repeat_kv(global_v, self.num_key_value_groups)

    # ========== Step 4: Prepare chunked attention with global prefix (NEW!) ==========
    if q_len % 4096 == 0:
        group_size = int(q_len * group_size_ratio)
    else:
        group_size = sft_group_size

    num_groups = q_len // group_size

    # For each chunk, prepend global context: [global_ctx, chunk]
    # This gives each chunk access to document-level information

    # Reshape query/key/value into chunks
    query_chunks = query_states.view(
        bsz, self.num_heads, num_groups, group_size, self.head_dim
    )
    key_chunks = key_states.view(
        bsz, self.num_heads, num_groups, group_size, self.head_dim
    )
    value_chunks = value_states.view(
        bsz, self.num_heads, num_groups, group_size, self.head_dim
    )

    # Expand global context for each chunk
    global_q_expanded = global_q.unsqueeze(2).expand(-1, -1, num_groups, -1, -1)
    global_k_expanded = global_k.unsqueeze(2).expand(-1, -1, num_groups, -1, -1)
    global_v_expanded = global_v.unsqueeze(2).expand(-1, -1, num_groups, -1, -1)

    # Concatenate: [global_ctx, chunk] for each chunk
    query_with_ctx = torch.cat(
        [global_q_expanded, query_chunks], dim=3
    )  # [bsz, nh, num_groups, num_slots+group_size, hd]
    key_with_ctx = torch.cat([global_k_expanded, key_chunks], dim=3)
    value_with_ctx = torch.cat([global_v_expanded, value_chunks], dim=3)

    # Reshape for flash attention: treat chunks as separate sequences
    chunk_len = num_local_slots + group_size
    query_with_ctx = query_with_ctx.permute(0, 2, 3, 1, 4).reshape(
        bsz * num_groups, chunk_len, self.num_heads, self.head_dim
    )
    key_with_ctx = key_with_ctx.permute(0, 2, 3, 1, 4).reshape(
        bsz * num_groups, chunk_len, self.num_heads, self.head_dim
    )
    value_with_ctx = value_with_ctx.permute(0, 2, 3, 1, 4).reshape(
        bsz * num_groups, chunk_len, self.num_heads, self.head_dim
    )

    # ========== Step 5: Flash attention with global+chunk sequences ==========
    # Stack Q/K/V for flash attention
    qkv = torch.stack(
        [query_with_ctx, key_with_ctx, value_with_ctx], dim=2
    )  # [bsz*num_groups, chunk_len, 3, nh, hd]

    # Prepare attention mask for chunks with global context
    # Each chunk: [1,1,...,1 (num_slots), mask[i*group_size:(i+1)*group_size]]
    chunk_masks = []
    for i in range(num_groups):
        # Global context is always valid (all 1s)
        global_mask = torch.ones(
            bsz,
            num_local_slots,
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        # Chunk-specific mask
        chunk_mask = attention_mask[:, i * group_size : (i + 1) * group_size]
        # Concatenate
        chunk_masks.append(torch.cat([global_mask, chunk_mask], dim=1))

    key_padding_mask = torch.stack(chunk_masks, dim=1).view(bsz * num_groups, chunk_len)

    nheads = qkv.shape[-2]
    x = rearrange(qkv, "b s three h d -> b s (three h d)")
    x_unpad, indices, cu_q_lens, max_s = unpad_input(x, key_padding_mask)

    x_unpad = rearrange(x_unpad, "nnz (three h d) -> nnz three h d", three=3, h=nheads)
    output_unpad = flash_attn_varlen_qkvpacked_func(
        x_unpad, cu_q_lens, max_s, 0.0, softmax_scale=None, causal=True
    )
    output = rearrange(
        pad_input(
            rearrange(output_unpad, "nnz h d -> nnz (h d)"),
            indices,
            bsz * num_groups,
            chunk_len,
        ),
        "b s (h d) -> b s h d",
        h=nheads,
    )
    # [bsz * num_groups, chunk_len, nh, hd]

    # ========== Step 6: Extract chunk outputs (discard global context outputs) ==========
    # Reshape back and remove global context portion
    output = output.view(bsz, num_groups, chunk_len, self.num_heads, self.head_dim)
    output = output[
        :, :, num_local_slots:, :, :
    ]  # Keep only chunk outputs, discard global ctx outputs
    output = output.reshape(bsz, q_len, self.num_heads, self.head_dim)

    return self.o_proj(rearrange(output, "b s h d -> b s (h d)")), None, past_key_value


def forward_flashattn_optimized(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    padding_mask: Optional[torch.LongTensor] = None,
    # use_local_repr: bool = True,
    # group_size_ratio: float = 0.25,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """
    Optimized version of forward_flashattn with the following improvements:
    1. Combined QKV projections: 6 calls → 3 calls (reduced kernel launch overhead)
    2. Vectorized mask construction: Python for-loop → single tensor operation
    3. Pre-allocated tensors: avoid multiple torch.cat allocations

    Mathematically equivalent to forward_flashattn.
    """
    if output_attentions:
        warnings.warn(
            "Output attentions is not supported for patched `LlamaAttention`, returning `None` instead."
        )

    bsz, q_len, hidden_size = hidden_states.size()

    use_local_constructor = hasattr(self, "local_constructor")

    if use_local_constructor:
        global_repr = self.local_constructor(
            hidden_states, attention_mask
        )  # [bsz, num_slots, hidden_size]
        num_local_slots = global_repr.shape[1]
    else:
        num_local_slots = 0

    # ========== Step 2: Compute group parameters ==========
    if q_len % 4096 == 0:
        group_size = int(q_len * group_size_ratio)
    else:
        group_size = sft_group_size
    num_groups = q_len // group_size
    chunk_len = num_local_slots + group_size

    # ========== Step 3: QKV projections ==========
    if use_local_constructor:
        combined_input = torch.cat(
            [global_repr, hidden_states], dim=1
        )  # [bsz, num_slots + q_len, hidden_size]

        combined_q = (
            self.q_proj(combined_input)
            .view(bsz, num_local_slots + q_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )  # [bsz, nh, num_slots + q_len, hd]

        combined_k = (
            self.k_proj(combined_input)
            .view(
                bsz, num_local_slots + q_len, self.num_key_value_heads, self.head_dim
            )
            .transpose(1, 2)
        )

        combined_v = (
            self.v_proj(combined_input)
            .view(
                bsz, num_local_slots + q_len, self.num_key_value_heads, self.head_dim
            )
            .transpose(1, 2)
        )

        global_q = combined_q[:, :, :num_local_slots, :]  # [bsz, nh, num_slots, hd]
        query_states = combined_q[:, :, num_local_slots:, :]  # [bsz, nh, q_len, hd]

        global_k = combined_k[:, :, :num_local_slots, :]
        key_states = combined_k[:, :, num_local_slots:, :]

        global_v = combined_v[:, :, :num_local_slots, :]
        value_states = combined_v[:, :, num_local_slots:, :]
    else:
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
    if use_local_constructor:
        global_k = repeat_kv(global_k, self.num_key_value_groups)
        global_v = repeat_kv(global_v, self.num_key_value_groups)

    # ========== Step 5: Chunk concatenation ==========
    # Reshape sequence to chunks: [bsz, nh, num_groups, group_size, hd]
    query_chunks = query_states.view(
        bsz, self.num_heads, num_groups, group_size, self.head_dim
    )
    key_chunks = key_states.view(
        bsz, self.num_heads, num_groups, group_size, self.head_dim
    )
    value_chunks = value_states.view(
        bsz, self.num_heads, num_groups, group_size, self.head_dim
    )

    if use_local_constructor:
        query_with_ctx = torch.empty(
            bsz,
            self.num_heads,
            num_groups,
            chunk_len,
            self.head_dim,
            dtype=query_states.dtype,
            device=query_states.device,
        )
        key_with_ctx = torch.empty_like(query_with_ctx)
        value_with_ctx = torch.empty_like(query_with_ctx)

        query_with_ctx[:, :, :, :num_local_slots, :] = global_q.unsqueeze(
            2
        )  # broadcast across num_groups
        query_with_ctx[:, :, :, num_local_slots:, :] = query_chunks

        key_with_ctx[:, :, :, :num_local_slots, :] = global_k.unsqueeze(2)
        key_with_ctx[:, :, :, num_local_slots:, :] = key_chunks

        value_with_ctx[:, :, :, :num_local_slots, :] = global_v.unsqueeze(2)
        value_with_ctx[:, :, :, num_local_slots:, :] = value_chunks

        # Reshape for flash attention: [bsz * num_groups, chunk_len, nh, hd]
        query_with_ctx = query_with_ctx.permute(0, 2, 3, 1, 4).reshape(
            bsz * num_groups, chunk_len, self.num_heads, self.head_dim
        )
        key_with_ctx = key_with_ctx.permute(0, 2, 3, 1, 4).reshape(
            bsz * num_groups, chunk_len, self.num_heads, self.head_dim
        )
        value_with_ctx = value_with_ctx.permute(0, 2, 3, 1, 4).reshape(
            bsz * num_groups, chunk_len, self.num_heads, self.head_dim
        )
    else:
        # Reshape for flash attention: [bsz * num_groups, group_size, nh, hd]
        query_with_ctx = query_chunks.permute(0, 2, 3, 1, 4).reshape(
            bsz * num_groups, group_size, self.num_heads, self.head_dim
        )
        key_with_ctx = key_chunks.permute(0, 2, 3, 1, 4).reshape(
            bsz * num_groups, group_size, self.num_heads, self.head_dim
        )
        value_with_ctx = value_chunks.permute(0, 2, 3, 1, 4).reshape(
            bsz * num_groups, group_size, self.num_heads, self.head_dim
        )

    # ========== Step 6: Mask construction ==========
    # attention_mask: [bsz, q_len] -> [bsz, num_groups, group_size]
    attention_mask_chunks = attention_mask.view(bsz, num_groups, group_size)

    if use_local_constructor:
        global_mask = attention_mask.new_ones(bsz, num_groups, num_local_slots)
        key_padding_mask = torch.cat([global_mask, attention_mask_chunks], dim=2)
    else:
        key_padding_mask = attention_mask_chunks

    # reshape: [bsz * num_groups, chunk_len]
    key_padding_mask = key_padding_mask.view(bsz * num_groups, chunk_len)

    # ========== Step 7: Flash Attention ==========
    qkv = torch.stack([query_with_ctx, key_with_ctx, value_with_ctx], dim=2)
    # [bsz * num_groups, chunk_len, 3, nh, hd]

    nheads = qkv.shape[-2]
    x = rearrange(qkv, "b s three h d -> b s (three h d)")
    x_unpad, indices, cu_q_lens, max_s = unpad_input(x, key_padding_mask)

    x_unpad = rearrange(x_unpad, "nnz (three h d) -> nnz three h d", three=3, h=nheads)
    output_unpad = flash_attn_varlen_qkvpacked_func(
        x_unpad, cu_q_lens, max_s, 0.0, softmax_scale=None, causal=True
    )

    output = rearrange(
        pad_input(
            rearrange(output_unpad, "nnz h d -> nnz (h d)"),
            indices,
            bsz * num_groups,
            chunk_len,
        ),
        "b s (h d) -> b s h d",
        h=nheads,
    )
    # [bsz * num_groups, chunk_len, nh, hd]

    output = output.view(bsz, num_groups, chunk_len, self.num_heads, self.head_dim)
    output = output[
        :, :, num_local_slots:, :, :
    ]
    output = output.reshape(bsz, q_len, self.num_heads, self.head_dim)

    return self.o_proj(rearrange(output, "b s h d -> b s (h d)")), None, past_key_value


def forward_flashattn_optimized_plus(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    padding_mask: Optional[torch.LongTensor] = None,
    # use_local_repr: bool = True,
    # group_size_ratio: float = 0.25,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """

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
    )  # [bsz, num_heads, q_len, head_dim]

    key_states = (
        self.k_proj(hidden_states)
        .view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        .transpose(1, 2)
    )  # [bsz, num_kv_heads, q_len, head_dim]

    value_states = (
        self.v_proj(hidden_states)
        .view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        .transpose(1, 2)
    )  # [bsz, num_kv_heads, q_len, head_dim]

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

    use_local_constructor = hasattr(self, "local_constructor")

    if use_local_constructor:
        key_for_repr = key_states.transpose(
            1, 2
        )  # [bsz, seq_len, num_heads, head_dim]
        value_for_repr = value_states.transpose(
            1, 2
        )  # [bsz, seq_len, num_heads, head_dim]

        global_repr = self.local_constructor(
            key_for_repr, value_for_repr, attention_mask
        )  # [bsz, num_slots, hidden_size]

        num_local_slots = global_repr.shape[1]

        global_q = (
            self.q_proj(global_repr)
            .view(bsz, num_local_slots, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )  # [bsz, num_heads, num_slots, head_dim]

        global_k = (
            self.k_proj(global_repr)
            .view(bsz, num_local_slots, self.num_key_value_heads, self.head_dim)
            .transpose(1, 2)
        )
        global_v = (
            self.v_proj(global_repr)
            .view(bsz, num_local_slots, self.num_key_value_heads, self.head_dim)
            .transpose(1, 2)
        )

        # Repeat k/v heads for global
        global_k = repeat_kv(global_k, self.num_key_value_groups)
        global_v = repeat_kv(global_v, self.num_key_value_groups)
    else:
        num_local_slots = 0

    # ========== Step 4: Compute group parameters ==========
    if q_len % 4096 == 0:
        group_size = int(q_len * group_size_ratio)
    else:
        group_size = sft_group_size
    num_groups = q_len // group_size
    chunk_len = num_local_slots + group_size

    # ========== Step 5: Chunk concatenation ==========
    # Reshape sequence to chunks: [bsz, nh, num_groups, group_size, hd]
    query_chunks = query_states.view(
        bsz, self.num_heads, num_groups, group_size, self.head_dim
    )
    key_chunks = key_states.view(
        bsz, self.num_heads, num_groups, group_size, self.head_dim
    )
    value_chunks = value_states.view(
        bsz, self.num_heads, num_groups, group_size, self.head_dim
    )

    if use_local_constructor:
        query_with_ctx = torch.empty(
            bsz,
            self.num_heads,
            num_groups,
            chunk_len,
            self.head_dim,
            dtype=query_states.dtype,
            device=query_states.device,
        )
        key_with_ctx = torch.empty_like(query_with_ctx)
        value_with_ctx = torch.empty_like(query_with_ctx)

        query_with_ctx[:, :, :, :num_local_slots, :] = global_q.unsqueeze(
            2
        )  # broadcast across num_groups
        query_with_ctx[:, :, :, num_local_slots:, :] = query_chunks

        key_with_ctx[:, :, :, :num_local_slots, :] = global_k.unsqueeze(2)
        key_with_ctx[:, :, :, num_local_slots:, :] = key_chunks

        value_with_ctx[:, :, :, :num_local_slots, :] = global_v.unsqueeze(2)
        value_with_ctx[:, :, :, num_local_slots:, :] = value_chunks

        # Reshape for flash attention: [bsz * num_groups, chunk_len, nh, hd]
        query_with_ctx = query_with_ctx.permute(0, 2, 3, 1, 4).reshape(
            bsz * num_groups, chunk_len, self.num_heads, self.head_dim
        )
        key_with_ctx = key_with_ctx.permute(0, 2, 3, 1, 4).reshape(
            bsz * num_groups, chunk_len, self.num_heads, self.head_dim
        )
        value_with_ctx = value_with_ctx.permute(0, 2, 3, 1, 4).reshape(
            bsz * num_groups, chunk_len, self.num_heads, self.head_dim
        )
    else:
        # Reshape for flash attention: [bsz * num_groups, group_size, nh, hd]
        query_with_ctx = query_chunks.permute(0, 2, 3, 1, 4).reshape(
            bsz * num_groups, group_size, self.num_heads, self.head_dim
        )
        key_with_ctx = key_chunks.permute(0, 2, 3, 1, 4).reshape(
            bsz * num_groups, group_size, self.num_heads, self.head_dim
        )
        value_with_ctx = value_chunks.permute(0, 2, 3, 1, 4).reshape(
            bsz * num_groups, group_size, self.num_heads, self.head_dim
        )

    # ========== Step 6: Mask construction ==========
    # attention_mask: [bsz, q_len] -> [bsz, num_groups, group_size]
    attention_mask_chunks = attention_mask.view(bsz, num_groups, group_size)

    if use_local_constructor:
        global_mask = attention_mask.new_ones(bsz, num_groups, num_local_slots)
        key_padding_mask = torch.cat([global_mask, attention_mask_chunks], dim=2)
    else:
        key_padding_mask = attention_mask_chunks

    # reshape: [bsz * num_groups, chunk_len]
    key_padding_mask = key_padding_mask.view(bsz * num_groups, chunk_len)

    # ========== Step 7: Flash Attention ==========
    qkv = torch.stack([query_with_ctx, key_with_ctx, value_with_ctx], dim=2)
    # [bsz * num_groups, chunk_len, 3, nh, hd]

    nheads = qkv.shape[-2]
    x = rearrange(qkv, "b s three h d -> b s (three h d)")
    x_unpad, indices, cu_q_lens, max_s = unpad_input(x, key_padding_mask)

    x_unpad = rearrange(x_unpad, "nnz (three h d) -> nnz three h d", three=3, h=nheads)
    output_unpad = flash_attn_varlen_qkvpacked_func(
        x_unpad, cu_q_lens, max_s, 0.0, softmax_scale=None, causal=True
    )

    output = rearrange(
        pad_input(
            rearrange(output_unpad, "nnz h d -> nnz (h d)"),
            indices,
            bsz * num_groups,
            chunk_len,
        ),
        "b s (h d) -> b s h d",
        h=nheads,
    )
    # [bsz * num_groups, chunk_len, nh, hd]

    output = output.view(bsz, num_groups, chunk_len, self.num_heads, self.head_dim)
    output = output[
        :, :, num_local_slots:, :, :
    ]
    output = output.reshape(bsz, q_len, self.num_heads, self.head_dim)

    return self.o_proj(rearrange(output, "b s h d -> b s (h d)")), None, past_key_value

def forward_flashattn_hierarchical_with_cache(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    padding_mask: Optional[torch.LongTensor] = None,
    use_higher_global: bool = True,
    use_local_repr: bool = True,
    use_recurrence_cache: bool = False,
    recurrence_size: Optional[int] = 128,
    group_size_ratio: Optional[float] = 0.25,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """

    1.  Recurrence cache (Transformer-XL style)
    2.  HiCI (LocalConstructor + GlobalIntegrator)
    3.  Ablation modes (use_higher_global, use_local_repr)


      Q:   [higher_global, chunk]
      K/V: [higher_global, cache, chunk]

    - Mode 2: use_higher_global=False, use_local_repr=True
      Q:   [local_i, chunk]
      K/V: [local_i, cache, chunk]

    - Mode 3: use_higher_global=True, use_local_repr=True
      Q:   [higher_global, local_i, chunk]
      K/V: [higher_global, local_i, cache, chunk]

    Args:
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

    if not hasattr(self, "_ablation_config_printed"):
        rank = dist.get_rank() if dist.is_initialized() else 0
        layer_idx = getattr(self, "layer_idx", 0)

        if rank == 0 and layer_idx == 0:
            print("\n" + "=" * 80)
            print(" HiCI Ablation Configuration")
            print("=" * 80)
            print(f" use_higher_global : {use_higher_global} ()")
            print(
            f" {'' if use_local_repr else ''} use_local_repr : {use_local_repr} ()"
            )
            print(
                f"   use_recurrence_cache : {use_recurrence_cache}  (Recurrence cache)"
            )
            print()

            if use_higher_global and not use_local_repr and use_recurrence_cache:
                print(" Current Mode: Mode 1 ()")
                print("   Q:   [higher_global, chunk]")
                print("   K/V: [higher_global, cache, chunk]")
                print(" : ，")
            elif not use_higher_global and use_local_repr and use_recurrence_cache:
                print(" Current Mode: Mode 2")
                print("   Q:   [local_i, chunk]")
                print("   K/V: [local_i, cache, chunk]")
                print(" : chunk ")
            elif use_higher_global and use_local_repr and use_recurrence_cache:
                print(" Current Mode: Mode 3 ()")
                print("   Q:   [higher_global, local_i, chunk]")
                print("   K/V: [higher_global, local_i, cache, chunk]")
                print(" : ，")
            else:
                print(" Current Mode: Custom")
                print(
                f" : higher_global={use_higher_global}, local={use_local_repr}, cache={use_recurrence_cache}"
                )

            print("=" * 80 + "\n", flush=True)

        self._ablation_config_printed = True

    if q_len % 4096 == 0:
        group_size = int(q_len * group_size_ratio)
    else:
        group_size = sft_group_size

    num_groups = q_len // group_size

    # Reshape into chunks: [bsz, num_groups, group_size, hidden_size]
    chunks = hidden_states.view(bsz, num_groups, group_size, hidden_size)

    #  CRITICAL: Check if local_constructor exists before using it!
    if (use_higher_global or use_local_repr) and hasattr(self, "local_constructor"):
        all_chunks = chunks.view(bsz * num_groups, group_size, hidden_size)

        all_local_mems = self.local_constructor(
            all_chunks
        )  # [bsz * num_groups, num_slots, hidden_size]

        # Reshape back: [bsz, num_groups, num_slots, hidden_size]
        num_local_slots = all_local_mems.shape[1]
        local_reprs_stacked = all_local_mems.view(
            bsz, num_groups, num_local_slots, hidden_size
        )
    else:
        num_local_slots = 0
        local_reprs_stacked = None

    #  CRITICAL: Requires BOTH local_constructor (for local extraction) AND global_integrator!
    if (
        use_higher_global
        and hasattr(self, "global_integrator")
        and local_reprs_stacked is not None
    ):
        higher_global = self.global_integrator(local_reprs_stacked)
        # [bsz, global_slots, hidden_size]
        num_global_slots = higher_global.shape[1]
    else:
        higher_global = None
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

    # RoPE
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

    # ========== Step 5: Project memories to Q/K/V ==========
    # Higher-level HiCI
    if use_higher_global and higher_global is not None:
        higher_global_q = (
            self.q_proj(higher_global)
            .view(bsz, num_global_slots, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        higher_global_k = (
            self.k_proj(higher_global)
            .view(bsz, num_global_slots, self.num_key_value_heads, self.head_dim)
            .transpose(1, 2)
        )
        higher_global_v = (
            self.v_proj(higher_global)
            .view(bsz, num_global_slots, self.num_key_value_heads, self.head_dim)
            .transpose(1, 2)
        )
        # Repeat k/v heads
        higher_global_k = repeat_kv(higher_global_k, self.num_key_value_groups)
        higher_global_v = repeat_kv(higher_global_v, self.num_key_value_groups)
    else:
        higher_global_q = higher_global_k = higher_global_v = None

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
    if use_local_repr and local_reprs_stacked is not None:
        # Reshape: [bsz, num_groups, num_slots, hidden] -> [bsz*num_groups, num_slots, hidden]
        local_mems_flat = local_reprs_stacked.view(
            bsz * num_groups, num_local_slots, hidden_size
        )

        local_q_flat = (
            self.q_proj(local_mems_flat)
            .view(bsz * num_groups, num_local_slots, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )  # [bsz*num_groups, nh, num_slots, hd]

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

        # Reshape back: [bsz, num_groups, nh, num_slots, hd]
        local_q_all = local_q_flat.view(
            bsz, num_groups, self.num_heads, num_local_slots, self.head_dim
        )
        local_k_all = local_k_flat.view(
            bsz, num_groups, self.num_heads, num_local_slots, self.head_dim
        )
        local_v_all = local_v_flat.view(
            bsz, num_groups, self.num_heads, num_local_slots, self.head_dim
        )

        local_reprs_q = [local_q_all[:, i, :, :, :] for i in range(num_groups)]
        local_reprs_k = [local_k_all[:, i, :, :, :] for i in range(num_groups)]
        local_reprs_v = [local_v_all[:, i, :, :, :] for i in range(num_groups)]

    # ========== Step 7: Process chunks with cache and memories ==========
    all_chunks_q = []
    all_chunks_k = []
    all_chunks_v = []

    prefix_q, prefix_k, prefix_v = [], [], []
    if use_higher_global and higher_global_q is not None:
        prefix_q.append(higher_global_q)
        prefix_k.append(higher_global_k)
        prefix_v.append(higher_global_v)

    dummy_cache_k, dummy_cache_v = None, None
    if use_recurrence_cache:
        dummy_cache_k = torch.zeros(
            bsz,
            self.num_heads,
            recurrence_size,
            self.head_dim,
            device=key_states.device,
            dtype=key_states.dtype,
        )
        dummy_cache_v = torch.zeros(
            bsz,
            self.num_heads,
            recurrence_size,
            self.head_dim,
            device=value_states.device,
            dtype=value_states.dtype,
        )

    for chunk_idx in range(num_groups):
        chunk_q = query_chunks[:, :, chunk_idx, :, :]  # [bsz, nh, group_size, hd]
        chunk_k = key_chunks[:, :, chunk_idx, :, :]
        chunk_v = value_chunks[:, :, chunk_idx, :, :]

        seq_q = prefix_q.copy()
        if use_local_repr and local_reprs_stacked is not None:
            seq_q.append(local_reprs_q[chunk_idx])
        seq_q.append(chunk_q)

        seq_k = prefix_k.copy()
        seq_v = prefix_v.copy()
        if use_local_repr and local_reprs_stacked is not None:
            seq_k.append(local_reprs_k[chunk_idx])
            seq_v.append(local_reprs_v[chunk_idx])

        if use_recurrence_cache:
            if chunk_idx == 0:
                seq_k.append(dummy_cache_k)
                seq_v.append(dummy_cache_v)
            else:
                seq_k.append(all_chunks_k[chunk_idx - 1][:, :, -recurrence_size:, :])
                seq_v.append(all_chunks_v[chunk_idx - 1][:, :, -recurrence_size:, :])

        seq_k.append(chunk_k)
        seq_v.append(chunk_v)

        all_chunks_q.append(torch.cat(seq_q, dim=2))
        all_chunks_k.append(torch.cat(seq_k, dim=2))
        all_chunks_v.append(torch.cat(seq_v, dim=2))

    q_len_per_chunk = all_chunks_q[0].shape[2]  # [bsz, nh, q_len, hd] -> q_len
    kv_len_per_chunk = all_chunks_k[0].shape[2]  # [bsz, nh, kv_len, hd] -> kv_len

    # ========== Step 8: Flash Attention with KV packed (following original implementation) ==========
    # CRITICAL: Use flash_attn_varlen_kvpacked_func to support Q and KV with different lengths

    # Stack all chunks and permute
    # Q: [bsz, nh, q_len_per_chunk, hd] -> [bsz, q_len_per_chunk, nh, hd]
    all_chunks_q_permuted = [c.permute(0, 2, 1, 3) for c in all_chunks_q]
    all_chunks_q_stacked = torch.stack(
        all_chunks_q_permuted, dim=0
    )  # [num_groups, bsz, q_len_per_chunk, nh, hd]
    all_chunks_q_flat = all_chunks_q_stacked.reshape(
        num_groups * bsz, q_len_per_chunk, self.num_heads, self.head_dim
    )

    # K/V: Stack K and V together for each chunk
    all_chunks_kv_permuted = []
    for k_chunk, v_chunk in zip(all_chunks_k, all_chunks_v):
        # k_chunk, v_chunk: [bsz, nh, kv_len_per_chunk, hd]
        k_perm = k_chunk.permute(0, 2, 1, 3)  # [bsz, kv_len_per_chunk, nh, hd]
        v_perm = v_chunk.permute(0, 2, 1, 3)  # [bsz, kv_len_per_chunk, nh, hd]
        kv_perm = torch.stack(
            [k_perm, v_perm], dim=2
        )  # [bsz, kv_len_per_chunk, 2, nh, hd]
        all_chunks_kv_permuted.append(kv_perm)

    all_chunks_kv_stacked = torch.stack(
        all_chunks_kv_permuted, dim=0
    )  # [num_groups, bsz, kv_len_per_chunk, 2, nh, hd]
    all_chunks_kv_flat = all_chunks_kv_stacked.reshape(
        num_groups * bsz, kv_len_per_chunk, 2, self.num_heads, self.head_dim
    )

    # ========== Step 9: Prepare padding masks (1=real token, 0=padding) =======

    # Reshape chunk masks: [bsz, num_groups, group_size]
    chunk_masks_reshaped = attention_mask.view(bsz, num_groups, group_size)

    all_masks_q_stacked = torch.empty(
        bsz,
        num_groups,
        q_len_per_chunk,
        dtype=attention_mask.dtype,
        device=attention_mask.device,
    )

    # In-place fill
    offset = 0
    if use_higher_global:
        all_masks_q_stacked[:, :, offset : offset + num_global_slots] = 1
        offset += num_global_slots
    if use_local_repr:
        all_masks_q_stacked[:, :, offset : offset + num_local_slots] = 1
        offset += num_local_slots

    all_masks_q_stacked[:, :, offset : offset + group_size] = chunk_masks_reshaped

    all_masks_q_stacked = all_masks_q_stacked.transpose(0, 1).contiguous()
    all_masks_q_flat = all_masks_q_stacked.reshape(num_groups * bsz, q_len_per_chunk)

    all_masks_kv_stacked = torch.empty(
        bsz,
        num_groups,
        kv_len_per_chunk,
        dtype=attention_mask.dtype,
        device=attention_mask.device,
    )

    # In-place fill
    offset = 0
    if use_higher_global:
        all_masks_kv_stacked[:, :, offset : offset + num_global_slots] = 1
        offset += num_global_slots
    if use_local_repr:
        all_masks_kv_stacked[:, :, offset : offset + num_local_slots] = 1
        offset += num_local_slots

    if use_recurrence_cache:
        cache_masks = torch.zeros(
            bsz,
            num_groups,
            recurrence_size,
            dtype=all_masks_kv_stacked.dtype,
            device=all_masks_kv_stacked.device,
        )

        if num_groups > 1:
            prev_chunk_tails = chunk_masks_reshaped[
                :, :-1, -recurrence_size:
            ]  # [bsz, num_groups-1, recurrence_size]
            cache_masks[:, 1:, :] = prev_chunk_tails

        all_masks_kv_stacked[:, :, offset : offset + recurrence_size] = cache_masks
        offset += recurrence_size

    # Chunk masks
    all_masks_kv_stacked[:, :, offset : offset + group_size] = chunk_masks_reshaped

    all_masks_kv_stacked = all_masks_kv_stacked.transpose(0, 1).contiguous()
    all_masks_kv_flat = all_masks_kv_stacked.reshape(num_groups * bsz, kv_len_per_chunk)

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
    )  # [num_groups*bsz, q_len_per_chunk, nh, hd]

    # Reshape from [num_groups*bsz, q_len_per_chunk, nh, hd] to [num_groups, bsz, q_len_per_chunk, nh, hd]
    output = output.reshape(
        num_groups, bsz, q_len_per_chunk, self.num_heads, self.head_dim
    )

    # Extract actual token outputs (skip memories, keep only chunk tokens)
    chunk_outputs = []
    skip_len = q_len_per_chunk - group_size

    for chunk_idx in range(num_groups):
        chunk_output = output[chunk_idx]  # [bsz, q_len_per_chunk, nh, hd]
        # Extract chunk tokens: skip memories, keep next group_size tokens
        tokens_output = chunk_output[
            :, skip_len : skip_len + group_size, :, :
        ]  # [bsz, group_size, nh, hd]
        chunk_outputs.append(tokens_output)

    # Concatenate all chunk outputs: [bsz, num_groups*group_size, nh, hd]
    output = torch.cat(chunk_outputs, dim=1)  # [bsz, q_len, nh, hd]

    # Output projection
    attn_output = self.o_proj(rearrange(output, "b s h d -> b s (h d)"))

    return attn_output, None, past_key_value

def forward_flashattn_hierarchical(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    padding_mask: Optional[torch.LongTensor] = None,
    use_higher_global: bool = True,
    use_local_repr: bool = True,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """

    1. HiCI (LocalConstructor + GlobalIntegrator)
    2. Ablation modes (use_higher_global, use_local_repr)


    - Q:   [chunk]
    - K/V: [higher_global?, local?, chunk]

      Q: [chunk], K/V: [higher_global, chunk]

    - Mode 2: use_higher_global=False, use_local_repr=True
      Q: [chunk], K/V: [local_i, chunk]

    - Mode 3: use_higher_global=True, use_local_repr=True
      Q: [chunk], K/V: [higher_global, local_i, chunk]

    Args:
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
            print("HiCI (Optimized: Q=[chunk], K/V=[memories,chunk])")
            print("=" * 80)
            print(f"  use_higher_global : {use_higher_global}")
            print(f"  use_local_repr  : {use_local_repr}")

            if use_higher_global and not use_local_repr:
                print("  Mode 1: Q=[chunk], K/V=[higher_global, chunk]")
            elif not use_higher_global and use_local_repr:
                print("  Mode 2: Q=[chunk], K/V=[local_i, chunk]")
            elif use_higher_global and use_local_repr:
                print("  Mode 3: Q=[chunk], K/V=[higher_global, local_i, chunk]")
            else:
                print("  Baseline: Q=K/V=[chunk]")

            print("=" * 80 + "\n", flush=True)

        self._hierarchical_no_cache_printed = True

    if q_len % 4096 == 0:
        group_size = int(q_len * group_size_ratio)
    else:
        target_group_size = int(q_len * group_size_ratio)

        divisors = []
        for i in range(1, int(q_len**0.5) + 1):
            if q_len % i == 0:
                divisors.append(i)
                if i != q_len // i:
                    divisors.append(q_len // i)

        min_size = 10 if target_group_size >= 10 else 1
        max_size = q_len // 2 if q_len >= 20 else q_len

        valid_divisors = [d for d in divisors if min_size <= d <= max_size]

        if valid_divisors:
            group_size = min(valid_divisors, key=lambda x: abs(x - target_group_size))
        else:
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
    if attention_mask is None:
        attention_mask = torch.ones(
            bsz, q_len, dtype=torch.bool, device=hidden_states.device
        )
    chunk_masks_reshaped = attention_mask.view(bsz, num_groups, group_size)

    if (use_higher_global or use_local_repr) and hasattr(self, "local_constructor"):
        all_chunks = chunks.view(bsz * num_groups, group_size, hidden_size)

        # [bsz, num_groups, group_size] -> [bsz * num_groups, group_size]
        attention_mask_chunks = chunk_masks_reshaped.view(bsz * num_groups, group_size)
        all_local_mems = self.local_constructor(
            all_chunks, attention_mask_chunks
        )  # [bsz * num_groups, num_slots, hidden_size]

        # Reshape back: [bsz, num_groups, num_slots, hidden_size]
        num_local_slots = all_local_mems.shape[1]
        local_reprs_stacked = all_local_mems.view(
            bsz, num_groups, num_local_slots, hidden_size
        )
    else:
        num_local_slots = 0
        local_reprs_stacked = None

    if (
        use_higher_global
        and hasattr(self, "global_integrator")
        and local_reprs_stacked is not None
    ):
        higher_global = self.global_integrator(local_reprs_stacked)
        # [bsz, global_slots, hidden_size]
        num_global_slots = higher_global.shape[1]
    else:
        higher_global = None
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

    # ========== Step 5: Project memories to Q/K/V ==========
    # Higher-level HiCI
    if use_higher_global and higher_global is not None:
        higher_global_k = (
            self.k_proj(higher_global)
            .view(bsz, num_global_slots, self.num_key_value_heads, self.head_dim)
            .transpose(1, 2)
        )
        higher_global_v = (
            self.v_proj(higher_global)
            .view(bsz, num_global_slots, self.num_key_value_heads, self.head_dim)
            .transpose(1, 2)
        )
        # Repeat k/v heads
        higher_global_k = repeat_kv(higher_global_k, self.num_key_value_groups)
        higher_global_v = repeat_kv(higher_global_v, self.num_key_value_groups)
    else:
        higher_global_k = higher_global_v = None

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
    if use_local_repr and local_reprs_stacked is not None:
        # Reshape: [bsz, num_groups, num_slots, hidden] -> [bsz*num_groups, num_slots, hidden]
        local_mems_flat = local_reprs_stacked.view(
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

    # query_chunks: [bsz, nh, num_groups, group_size, hd]
    all_chunks_q_flat = query_chunks.permute(0, 2, 3, 1, 4).reshape(
        bsz * num_groups, group_size, self.num_heads, self.head_dim
    )

    # key_chunks/value_chunks: [bsz, nh, num_groups, group_size, hd]

    repr_len = 0
    if use_higher_global:
        repr_len += num_global_slots
    if use_local_repr:
        repr_len += num_local_slots
    kv_len_per_chunk = repr_len + group_size

    if repr_len > 0:
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
        if use_higher_global and higher_global_k is not None:
            # higher_global_k: [bsz, nh, num_global_slots, hd]
            all_k[:, :, :, offset : offset + num_global_slots, :] = (
                higher_global_k.unsqueeze(2)
            )
            all_v[:, :, :, offset : offset + num_global_slots, :] = (
                higher_global_v.unsqueeze(2)
            )
            offset += num_global_slots

        if use_local_repr and local_k_all is not None:
            # local_k_all: [bsz, num_groups, nh, num_local_slots, hd]
            all_k[:, :, :, offset : offset + num_local_slots, :] = local_k_all.permute(
                0, 2, 1, 3, 4
            )
            all_v[:, :, :, offset : offset + num_local_slots, :] = local_v_all.permute(
                0, 2, 1, 3, 4
            )
            offset += num_local_slots

        all_k[:, :, :, offset : offset + group_size, :] = key_chunks
        all_v[:, :, :, offset : offset + group_size, :] = value_chunks
    else:
        all_k = key_chunks
        all_v = value_chunks
        kv_len_per_chunk = group_size

    all_k_flat = all_k.permute(0, 2, 3, 1, 4).reshape(
        bsz * num_groups, kv_len_per_chunk, self.num_heads, self.head_dim
    )
    all_v_flat = all_v.permute(0, 2, 3, 1, 4).reshape(
        bsz * num_groups, kv_len_per_chunk, self.num_heads, self.head_dim
    )

    all_chunks_kv_flat = torch.stack([all_k_flat, all_v_flat], dim=2)

    q_len_per_chunk = group_size

    # ========== Step 9: Prepare padding masks (1=real token, 0=padding) ==========
    all_masks_q_flat = chunk_masks_reshaped.reshape(bsz * num_groups, q_len_per_chunk)

    # 9.2 K/V padding masks（memories + chunk）
    all_masks_kv_stacked = torch.empty(
        bsz,
        num_groups,
        kv_len_per_chunk,
        dtype=chunk_masks_reshaped.dtype,
        device=chunk_masks_reshaped.device,
    )

    offset = 0
    if use_higher_global:
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

    # Output projection
    attn_output = self.o_proj(rearrange(output, "b s h d -> b s (h d)"))

    return attn_output, None, past_key_value

def forward_flashattn_hierarchical_sft_inference(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    padding_mask: Optional[torch.LongTensor] = None,
    use_higher_global: bool = True,
    use_local_repr: bool = True,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """

       - Q=[tokens], K/V=[higher_global, local, tokens]


    - use_higher_global=True
    - use_local_repr=True
    - Mode 3: Q=[chunk], K/V=[higher_global, local, chunk]
    """

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
            print("HiCI (Optimized: Q=[chunk], K/V=[memories,chunk])")
            print("=" * 80)
            print(f"  use_higher_global : {use_higher_global}")
            print(f"  use_local_repr  : {use_local_repr}")

            if use_higher_global and not use_local_repr:
                print("  Mode 1: Q=[chunk], K/V=[higher_global, chunk]")
            elif not use_higher_global and use_local_repr:
                print("  Mode 2: Q=[chunk], K/V=[local_i, chunk]")
            elif use_higher_global and use_local_repr:
                print("  Mode 3: Q=[chunk], K/V=[higher_global, local_i, chunk]")
            else:
                print("  Baseline: Q=K/V=[chunk]")

            print("=" * 80 + "\n", flush=True)

        self._hierarchical_no_cache_printed = True

    if q_len % 1024 == 0:
        group_size = int(q_len * group_size_ratio)
    else:
        target_group_size = int(q_len * group_size_ratio)

        divisors = []
        for i in range(1, int(q_len**0.5) + 1):
            if q_len % i == 0:
                divisors.append(i)
                if i != q_len // i:
                    divisors.append(q_len // i)

        min_size = 10 if target_group_size >= 10 else 1
        max_size = q_len // 2 if q_len >= 20 else q_len

        valid_divisors = [d for d in divisors if min_size <= d <= max_size]

        if valid_divisors:
            group_size = min(valid_divisors, key=lambda x: abs(x - target_group_size))
        else:
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
    if attention_mask is None:
        attention_mask = torch.ones(
            bsz, q_len, dtype=torch.bool, device=hidden_states.device
        )
    chunk_masks_reshaped = attention_mask.view(bsz, num_groups, group_size)

    if (use_higher_global or use_local_repr) and hasattr(self, "local_constructor"):
        all_chunks = chunks.view(bsz * num_groups, group_size, hidden_size)

        # [bsz, num_groups, group_size] -> [bsz * num_groups, group_size]
        attention_mask_chunks = chunk_masks_reshaped.view(bsz * num_groups, group_size)
        all_local_mems = self.local_constructor(
            all_chunks, attention_mask_chunks
        )  # [bsz * num_groups, num_slots, hidden_size]

        # Reshape back: [bsz, num_groups, num_slots, hidden_size]
        num_local_slots = all_local_mems.shape[1]
        local_reprs_stacked = all_local_mems.view(
            bsz, num_groups, num_local_slots, hidden_size
        )
    else:
        num_local_slots = 0
        local_reprs_stacked = None

    if (
        use_higher_global
        and hasattr(self, "global_integrator")
        and local_reprs_stacked is not None
    ):
        higher_global = self.global_integrator(local_reprs_stacked)
        # [bsz, global_slots, hidden_size]
        num_global_slots = higher_global.shape[1]
    else:
        higher_global = None
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

    # ========== Step 5: Project memories to Q/K/V ==========
    # Higher-level HiCI
    if use_higher_global and higher_global is not None:
        higher_global_k = (
            self.k_proj(higher_global)
            .view(bsz, num_global_slots, self.num_key_value_heads, self.head_dim)
            .transpose(1, 2)
        )
        higher_global_v = (
            self.v_proj(higher_global)
            .view(bsz, num_global_slots, self.num_key_value_heads, self.head_dim)
            .transpose(1, 2)
        )
        # Repeat k/v heads
        higher_global_k = repeat_kv(higher_global_k, self.num_key_value_groups)
        higher_global_v = repeat_kv(higher_global_v, self.num_key_value_groups)
    else:
        higher_global_k = higher_global_v = None

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
    if use_local_repr and local_reprs_stacked is not None:
        # Reshape: [bsz, num_groups, num_slots, hidden] -> [bsz*num_groups, num_slots, hidden]
        local_mems_flat = local_reprs_stacked.view(
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

    # query_chunks: [bsz, nh, num_groups, group_size, hd]
    all_chunks_q_flat = query_chunks.permute(0, 2, 3, 1, 4).reshape(
        bsz * num_groups, group_size, self.num_heads, self.head_dim
    )

    # key_chunks/value_chunks: [bsz, nh, num_groups, group_size, hd]

    repr_len = 0
    if use_higher_global:
        repr_len += num_global_slots
    if use_local_repr:
        repr_len += num_local_slots
    kv_len_per_chunk = repr_len + group_size

    if repr_len > 0:
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
        if use_higher_global and higher_global_k is not None:
            # higher_global_k: [bsz, nh, num_global_slots, hd]
            all_k[:, :, :, offset : offset + num_global_slots, :] = (
                higher_global_k.unsqueeze(2)
            )
            all_v[:, :, :, offset : offset + num_global_slots, :] = (
                higher_global_v.unsqueeze(2)
            )
            offset += num_global_slots

        if use_local_repr and local_k_all is not None:
            # local_k_all: [bsz, num_groups, nh, num_local_slots, hd]
            all_k[:, :, :, offset : offset + num_local_slots, :] = local_k_all.permute(
                0, 2, 1, 3, 4
            )
            all_v[:, :, :, offset : offset + num_local_slots, :] = local_v_all.permute(
                0, 2, 1, 3, 4
            )
            offset += num_local_slots

        all_k[:, :, :, offset : offset + group_size, :] = key_chunks
        all_v[:, :, :, offset : offset + group_size, :] = value_chunks
    else:
        all_k = key_chunks
        all_v = value_chunks
        kv_len_per_chunk = group_size

    all_k_flat = all_k.permute(0, 2, 3, 1, 4).reshape(
        bsz * num_groups, kv_len_per_chunk, self.num_heads, self.head_dim
    )
    all_v_flat = all_v.permute(0, 2, 3, 1, 4).reshape(
        bsz * num_groups, kv_len_per_chunk, self.num_heads, self.head_dim
    )

    all_chunks_kv_flat = torch.stack([all_k_flat, all_v_flat], dim=2)

    q_len_per_chunk = group_size

    # ========== Step 9: Prepare padding masks (1=real token, 0=padding) ==========
    all_masks_q_flat = chunk_masks_reshaped.reshape(bsz * num_groups, q_len_per_chunk)

    # 9.2 K/V padding masks（memories + chunk）
    all_masks_kv_stacked = torch.empty(
        bsz,
        num_groups,
        kv_len_per_chunk,
        dtype=chunk_masks_reshaped.dtype,
        device=chunk_masks_reshaped.device,
    )

    offset = 0
    if use_higher_global:
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

    # Output projection
    attn_output = self.o_proj(rearrange(output, "b s h d -> b s (h d)"))

    return attn_output, None, past_key_value

def forward_flashattn_hierarchical_Seg(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    padding_mask: Optional[torch.LongTensor] = None,
    use_higher_global: bool = True,
    use_local_repr: bool = True,
    target_segment_size: int = 1024,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """


    1. HiCI (LocalConstructor + GlobalIntegrator)
    2. Ablation modes (use_higher_global, use_local_repr)


    - Q:   [chunk]
    - K/V: [higher_global?, local?, chunk]

      Q: [chunk], K/V: [higher_global, chunk]

    - Mode 2: use_higher_global=False, use_local_repr=True
      Q: [chunk], K/V: [local_i, chunk]

    - Mode 3: use_higher_global=True, use_local_repr=True
      Q: [chunk], K/V: [higher_global, local_i, chunk]

    Args:
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
            print("HiCI (Optimized: Q=[chunk], K/V=[memories,chunk])")
            print("=" * 80)
            print(f"  use_higher_global : {use_higher_global}")
            print(f"  use_local_repr  : {use_local_repr}")

            if use_higher_global and not use_local_repr:
                print("  Mode 1: Q=[chunk], K/V=[higher_global, chunk]")
            elif not use_higher_global and use_local_repr:
                print("  Mode 2: Q=[chunk], K/V=[local_i, chunk]")
            elif use_higher_global and use_local_repr:
                print("  Mode 3: Q=[chunk], K/V=[higher_global, local_i, chunk]")
            else:
                print("  Baseline: Q=K/V=[chunk]")

            print("=" * 80 + "\n", flush=True)

        self._hierarchical_no_cache_printed = True

    if q_len % target_segment_size == 0:
        group_size = target_segment_size
    else:
        divisors = []
        for i in range(1, int(q_len**0.5) + 1):
            if q_len % i == 0:
                divisors.append(i)
                if i != q_len // i:
                    divisors.append(q_len // i)

        min_size = 512
        max_size = min(2048, q_len // 2) if q_len >= 1024 else q_len

        valid_divisors = [d for d in divisors if min_size <= d <= max_size]

        if valid_divisors:
            group_size = min(valid_divisors, key=lambda x: abs(x - target_segment_size))
        else:
            valid_divisors = [d for d in divisors if 256 <= d <= min(4096, q_len)]
            if valid_divisors:
                group_size = min(
                    valid_divisors, key=lambda x: abs(x - target_segment_size)
                )
            else:
                group_size = q_len

    if q_len % group_size > 0:
        raise ValueError(
            f"q_len {q_len} should be divisible by group size {group_size}."
        )
    if not hasattr(self, "_hierarchical_seg_printed"):
        layer_idx = getattr(self, "layer_idx", 0)
        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank == 0 and layer_idx == 0:
            num_groups_tmp = q_len // group_size
            print(
                f"[forward_flashattn_hierarchical_Seg] target_segment_size={target_segment_size}, actual group_size={group_size}, num_groups={num_groups_tmp}"
            )
        self._hierarchical_seg_printed = True

    num_groups = q_len // group_size

    # Reshape into chunks: [bsz, num_groups, group_size, hidden_size]
    chunks = hidden_states.view(bsz, num_groups, group_size, hidden_size)

    # attention_mask: [bsz, q_len] -> chunk_masks_reshaped: [bsz, num_groups, group_size]
    if attention_mask is None:
        attention_mask = torch.ones(
            bsz, q_len, dtype=torch.bool, device=hidden_states.device
        )
    chunk_masks_reshaped = attention_mask.view(bsz, num_groups, group_size)

    if (use_higher_global or use_local_repr) and hasattr(self, "local_constructor"):
        all_chunks = chunks.view(bsz * num_groups, group_size, hidden_size)

        # [bsz, num_groups, group_size] -> [bsz * num_groups, group_size]
        attention_mask_chunks = chunk_masks_reshaped.view(bsz * num_groups, group_size)
        all_local_mems = self.local_constructor(
            all_chunks, attention_mask_chunks
        )  # [bsz * num_groups, num_slots, hidden_size]

        # Reshape back: [bsz, num_groups, num_slots, hidden_size]
        num_local_slots = all_local_mems.shape[1]
        local_reprs_stacked = all_local_mems.view(
            bsz, num_groups, num_local_slots, hidden_size
        )
    else:
        num_local_slots = 0
        local_reprs_stacked = None

    if (
        use_higher_global
        and hasattr(self, "global_integrator")
        and local_reprs_stacked is not None
    ):
        higher_global = self.global_integrator(local_reprs_stacked)
        # [bsz, global_slots, hidden_size]
        num_global_slots = higher_global.shape[1]
    else:
        higher_global = None
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

    # ========== Step 5: Project memories to Q/K/V ==========
    # Higher-level HiCI
    if use_higher_global and higher_global is not None:
        higher_global_k = (
            self.k_proj(higher_global)
            .view(bsz, num_global_slots, self.num_key_value_heads, self.head_dim)
            .transpose(1, 2)
        )
        higher_global_v = (
            self.v_proj(higher_global)
            .view(bsz, num_global_slots, self.num_key_value_heads, self.head_dim)
            .transpose(1, 2)
        )
        # Repeat k/v heads
        higher_global_k = repeat_kv(higher_global_k, self.num_key_value_groups)
        higher_global_v = repeat_kv(higher_global_v, self.num_key_value_groups)
    else:
        higher_global_k = higher_global_v = None

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
    if use_local_repr and local_reprs_stacked is not None:
        # Reshape: [bsz, num_groups, num_slots, hidden] -> [bsz*num_groups, num_slots, hidden]
        local_mems_flat = local_reprs_stacked.view(
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

    # query_chunks: [bsz, nh, num_groups, group_size, hd]
    all_chunks_q_flat = query_chunks.permute(0, 2, 3, 1, 4).reshape(
        bsz * num_groups, group_size, self.num_heads, self.head_dim
    )

    # key_chunks/value_chunks: [bsz, nh, num_groups, group_size, hd]

    repr_len = 0
    if use_higher_global:
        repr_len += num_global_slots
    if use_local_repr:
        repr_len += num_local_slots
    kv_len_per_chunk = repr_len + group_size

    if repr_len > 0:
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
        if use_higher_global and higher_global_k is not None:
            # higher_global_k: [bsz, nh, num_global_slots, hd]
            all_k[:, :, :, offset : offset + num_global_slots, :] = (
                higher_global_k.unsqueeze(2)
            )
            all_v[:, :, :, offset : offset + num_global_slots, :] = (
                higher_global_v.unsqueeze(2)
            )
            offset += num_global_slots

        if use_local_repr and local_k_all is not None:
            # local_k_all: [bsz, num_groups, nh, num_local_slots, hd]
            all_k[:, :, :, offset : offset + num_local_slots, :] = local_k_all.permute(
                0, 2, 1, 3, 4
            )
            all_v[:, :, :, offset : offset + num_local_slots, :] = local_v_all.permute(
                0, 2, 1, 3, 4
            )
            offset += num_local_slots

        all_k[:, :, :, offset : offset + group_size, :] = key_chunks
        all_v[:, :, :, offset : offset + group_size, :] = value_chunks
    else:
        all_k = key_chunks
        all_v = value_chunks
        kv_len_per_chunk = group_size

    all_k_flat = all_k.permute(0, 2, 3, 1, 4).reshape(
        bsz * num_groups, kv_len_per_chunk, self.num_heads, self.head_dim
    )
    all_v_flat = all_v.permute(0, 2, 3, 1, 4).reshape(
        bsz * num_groups, kv_len_per_chunk, self.num_heads, self.head_dim
    )

    all_chunks_kv_flat = torch.stack([all_k_flat, all_v_flat], dim=2)

    q_len_per_chunk = group_size

    # ========== Step 9: Prepare padding masks (1=real token, 0=padding) ==========
    all_masks_q_flat = chunk_masks_reshaped.reshape(bsz * num_groups, q_len_per_chunk)

    # 9.2 K/V padding masks（memories + chunk）
    all_masks_kv_stacked = torch.empty(
        bsz,
        num_groups,
        kv_len_per_chunk,
        dtype=chunk_masks_reshaped.dtype,
        device=chunk_masks_reshaped.device,
    )

    offset = 0
    if use_higher_global:
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

    # Output projection
    attn_output = self.o_proj(rearrange(output, "b s h d -> b s (h d)"))

    return attn_output, None, past_key_value

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

def forward_flashattn_full_hierarchical(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    padding_mask: Optional[torch.LongTensor] = None,
    use_higher_global: bool = True,
    use_local_repr: bool = True,
    group_size_ratio: Optional[float] = 1,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """


    Q/K/V: [higher_global, local_repr, full_sequence]

    """
    if output_attentions:
        warnings.warn(
            "Output attentions is not supported for patched `LlamaAttention`, returning `None` instead."
        )

    bsz, q_len, hidden_size = hidden_states.size()

    if not hasattr(self, "_eval_config_printed_way3"):
        rank = dist.get_rank() if dist.is_initialized() else 0
        layer_idx = getattr(self, "layer_idx", 0)

        if rank == 0 and layer_idx == 0:
            print("\n" + "=" * 80)
            print(" Evaluation Mode: Full Attention + HiCI")
            print("=" * 80)
            print(f"   use_higher_global : {use_higher_global}")
            print(f"   use_local_repr  : {use_local_repr}")
            print(
                f"   group_size_ratio  : {group_size_ratio} (for representation extraction)"
            )
            print(f"   recurrence_cache  : disabled")
            print()
            print(" Attention Structure:")
            print("   Q/K/V: [local_constructor, local_repr, full_sequence]")
            print("   → Full Attention (no chunking)")
            print("=" * 80 + "\n", flush=True)

        self._eval_config_printed_way3 = True

    if q_len % 4096 == 0:
        group_size = int(q_len * group_size_ratio)
    else:
        group_size = sft_group_size
    num_groups = q_len // group_size

    # Reshape into chunks for representation extraction
    chunks = hidden_states.view(bsz, num_groups, group_size, hidden_size)

    if (use_higher_global or use_local_repr) and hasattr(self, "local_constructor"):
        all_chunks = chunks.view(bsz * num_groups, group_size, hidden_size)

        # attention_mask: [bsz, q_len] -> [bsz * num_groups, group_size]
        if attention_mask is not None:
            attention_mask_chunks = attention_mask.view(bsz, num_groups, group_size)
            attention_mask_chunks = attention_mask_chunks.view(
                bsz * num_groups, group_size
            )
        else:
            attention_mask_chunks = None

        all_local_mems = self.local_constructor(all_chunks, attention_mask_chunks)
        num_local_slots = all_local_mems.shape[1]
        local_reprs_stacked = all_local_mems.view(
            bsz, num_groups, num_local_slots, hidden_size
        )
        local_reprs_flat = local_reprs_stacked.view(
            bsz, num_groups * num_local_slots, hidden_size
        )
    else:
        num_local_slots = 0
        local_reprs_stacked = None
        local_reprs_flat = None

    if (
        use_higher_global
        and hasattr(self, "global_integrator")
        and local_reprs_stacked is not None
    ):
        higher_global = self.global_integrator(local_reprs_stacked)
        num_global_slots = higher_global.shape[1]
    else:
        higher_global = None
        num_global_slots = 0

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

    # RoPE
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

    # Repeat k/v heads
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    if use_higher_global and higher_global is not None:
        higher_global_q = (
            self.q_proj(higher_global)
            .view(bsz, num_global_slots, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        higher_global_k = (
            self.k_proj(higher_global)
            .view(bsz, num_global_slots, self.num_key_value_heads, self.head_dim)
            .transpose(1, 2)
        )
        higher_global_v = (
            self.v_proj(higher_global)
            .view(bsz, num_global_slots, self.num_key_value_heads, self.head_dim)
            .transpose(1, 2)
        )
        higher_global_k = repeat_kv(higher_global_k, self.num_key_value_groups)
        higher_global_v = repeat_kv(higher_global_v, self.num_key_value_groups)
    else:
        higher_global_q = higher_global_k = higher_global_v = None

    if use_local_repr and local_reprs_flat is not None:
        total_local_slots = num_groups * num_local_slots
        local_q = (
            self.q_proj(local_reprs_flat)
            .view(bsz, total_local_slots, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        local_k = (
            self.k_proj(local_reprs_flat)
            .view(bsz, total_local_slots, self.num_key_value_heads, self.head_dim)
            .transpose(1, 2)
        )
        local_v = (
            self.v_proj(local_reprs_flat)
            .view(bsz, total_local_slots, self.num_key_value_heads, self.head_dim)
            .transpose(1, 2)
        )
        local_k = repeat_kv(local_k, self.num_key_value_groups)
        local_v = repeat_kv(local_v, self.num_key_value_groups)
    else:
        local_q = local_k = local_v = None
        total_local_slots = 0

    q_parts = []
    if higher_global_q is not None:
        q_parts.append(higher_global_q)
    if local_q is not None:
        q_parts.append(local_q)
    q_parts.append(query_states)
    full_q = torch.cat(q_parts, dim=2)  # [bsz, nh, total_len, hd]

    k_parts = []
    v_parts = []
    if higher_global_k is not None:
        k_parts.append(higher_global_k)
        v_parts.append(higher_global_v)
    if local_k is not None:
        k_parts.append(local_k)
        v_parts.append(local_v)
    k_parts.append(key_states)
    v_parts.append(value_states)
    full_k = torch.cat(k_parts, dim=2)  # [bsz, nh, total_len, hd]
    full_v = torch.cat(v_parts, dim=2)  # [bsz, nh, total_len, hd]

    repr_len = num_global_slots + total_local_slots
    total_len = repr_len + q_len

    # ========== Step 7: Flash Attention (Full Attention) ==========
    full_q = full_q.transpose(1, 2)
    full_k = full_k.transpose(1, 2)
    full_v = full_v.transpose(1, 2)

    # Stack Q/K/V: [bsz, seq, 3, nh, hd]
    qkv = torch.stack([full_q, full_k, full_v], dim=2)

    # [bsz, total_len]
    repr_mask = torch.ones(
        bsz, repr_len, dtype=attention_mask.dtype, device=attention_mask.device
    )
    full_mask = torch.cat([repr_mask, attention_mask], dim=1)

    # Flash Attention
    key_padding_mask = full_mask
    nheads = qkv.shape[-2]
    x = rearrange(qkv, "b s three h d -> b s (three h d)")
    x_unpad, indices, cu_q_lens, max_s = unpad_input(x, key_padding_mask)
    x_unpad = rearrange(x_unpad, "nnz (three h d) -> nnz three h d", three=3, h=nheads)

    output_unpad = flash_attn_varlen_qkvpacked_func(
        x_unpad, cu_q_lens, max_s, 0.0, softmax_scale=None, causal=True
    )

    output = rearrange(
        pad_input(
            rearrange(output_unpad, "nnz h d -> nnz (h d)"), indices, bsz, total_len
        ),
        "b s (h d) -> b s h d",
        h=nheads,
    )

    # output: [bsz, total_len, nh, hd]
    seq_output = output[:, repr_len:, :, :]  # [bsz, q_len, nh, hd]

    # Output projection
    attn_output = self.o_proj(rearrange(seq_output, "b s h d -> b s (h d)"))

    return attn_output, None, past_key_value

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

def _prepare_decoder_attention_mask_inference(
    self, attention_mask, input_shape, inputs_embeds, past_key_values_length
):
    """


    Args:
        input_shape: (bsz, seq_len)

    Returns:
    """
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

# ============================================================================
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

       - Mode 3: Q=[chunk], K/V=[higher_global, local, chunk]

    """
    bsz, q_len, hidden_size = hidden_states.size()

    if q_len == 1 or past_key_value is not None:
        kv_heads = getattr(self, "num_key_value_heads", self.num_heads)

        q = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(bsz, q_len, kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(bsz, q_len, kv_heads, self.head_dim)

        kv_seq_len = k.shape[1]
        past_kv_len = 0
        if past_key_value is not None:
            past_kv_len = past_key_value[0].shape[2]
            kv_seq_len += past_kv_len

        if INCLUDE_REPR_IN_KV_CACHE and not hasattr(self, "_hici_decode_printed"):
            layer_idx = getattr(self, "layer_idx", 0)
            if layer_idx == 0:
                repr_len_in_cache = getattr(self, "_hici_cache_repr_len", 0)
                print(f"\n{'=' * 60}")
                print(f"[HiCI Decode ] Layer 0, Decode")
                print(f" HF position_ids : {position_ids.item()}")
                print(f"  repr_len_in_cache     : {repr_len_in_cache}")
                print(f"  past_kv_len (KV cache)  : {past_kv_len}")
                print(f" kv_seq_len (token) : {kv_seq_len}")
                print(f" : HF position_ids token ，")
                print(f"{'=' * 60}\n")
            self._hici_decode_printed = True

        cos_sin = self.rotary_emb(v, seq_len=kv_seq_len)
        q, k = apply_rotary_pos_emb_inference(q, k, cos_sin, position_ids)

        if past_key_value is not None:
            k = torch.cat([past_key_value[0].transpose(1, 2), k], dim=1)
            v = torch.cat([past_key_value[1].transpose(1, 2), v], dim=1)

        past_key_value = (k.transpose(1, 2), v.transpose(1, 2)) if use_cache else None

        if INCLUDE_REPR_IN_KV_CACHE and hasattr(self, "_hici_cache_repr_len"):
            repr_len = getattr(self, "_hici_cache_repr_len", 0)
            if repr_len > 0 and attention_mask is not None:
                repr_mask = torch.ones(
                    bsz,
                    repr_len,
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                attention_mask = torch.cat([repr_mask, attention_mask], dim=1)

                if not hasattr(self, "_hici_mask_fix_printed"):
                    layer_idx = getattr(self, "layer_idx", 0)
                    if layer_idx == 0:
                        print(
                            f"[HiCI Decode] Extended attention_mask: added {repr_len} local query slots, new shape={attention_mask.shape}"
                        )
                    self._hici_mask_fix_printed = True

        if not hasattr(self, "_hici_attn_branch_printed"):
            layer_idx = getattr(self, "layer_idx", 0)
            if layer_idx == 0:
                print(
                f"\n[HiCI Decode ] attention_mask is None: {attention_mask is None}"
                )
                print(f"  q.shape: {q.shape}, k.shape: {k.shape}, v.shape: {v.shape}")
                if attention_mask is not None:
                    print(f"  attention_mask.shape: {attention_mask.shape}")
            self._hici_attn_branch_printed = True

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

        kv_heads = getattr(self, "num_key_value_heads", self.num_heads)

        q = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(bsz, q_len, kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(bsz, q_len, kv_heads, self.head_dim)

        cos_sin = self.rotary_emb(v, seq_len=q_len)
        q, k = apply_rotary_pos_emb_inference(q, k, cos_sin, position_ids)

        past_key_value = (k.transpose(1, 2), v.transpose(1, 2)) if use_cache else None

        # Flash Attention
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

    if not _HICI_INFERENCE_PRINTED:
        layer_idx = getattr(self, "layer_idx", 0)
        if layer_idx == 0:
            print("\n" + "=" * 80)
            print("HiCI SFT Inference Mode (LongBench)")
            print("=" * 80)
            print(f"  group_size_ratio  : {group_size_ratio}")
            print("  Prefill: HiCI (higher_global + local)")
            print("  Decode:  standard attention + KV cache")
            print("=" * 80 + "\n", flush=True)
            _HICI_INFERENCE_PRINTED = True

    # -----------------------------------------------------
    group_size = q_len
    num_groups = 1

    if not _HICI_GROUP_PRINTED:
        layer_idx = getattr(self, "layer_idx", 0)
        if layer_idx == 0:
            print(
                f"[HiCI Inference] q_len={q_len}, group_size={group_size}, num_groups={num_groups}"
            )
            _HICI_GROUP_PRINTED = True

    # Reshape into chunks
    chunks = hidden_states.view(bsz, num_groups, group_size, hidden_size)

    if attention_mask is not None and attention_mask.dim() == 2:
        chunk_masks = attention_mask.view(bsz, num_groups, group_size)
    else:
        chunk_masks = torch.ones(
            bsz, num_groups, group_size, dtype=torch.bool, device=hidden_states.device
        )

    use_higher_global = True
    use_local_repr = True

    if hasattr(self, "local_constructor"):
        all_chunks = chunks.view(bsz * num_groups, group_size, hidden_size)
        mask_chunks = chunk_masks.view(bsz * num_groups, group_size)
        all_local_mems = self.local_constructor(all_chunks, mask_chunks)
        num_local_slots = all_local_mems.shape[1]
        local_reprs = all_local_mems.view(
            bsz, num_groups, num_local_slots, hidden_size
        )
    else:
        num_local_slots = 0
        local_reprs = None

    if hasattr(self, "global_integrator") and local_reprs is not None:
        higher_global = self.global_integrator(local_reprs)
        num_global_slots = higher_global.shape[1]
    else:
        higher_global = None
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

    cos, sin = self.rotary_emb(value_states, seq_len=q_len)
    query_states, key_states = apply_rotary_pos_emb(
        query_states, key_states, cos, sin, position_ids
    )

    key_states_for_cache = key_states
    value_states_for_cache = value_states

    # Repeat k/v heads for attention (num_heads)
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    if higher_global is not None:
        hg_k_cache = (
            self.k_proj(higher_global)
            .view(bsz, num_global_slots, self.num_key_value_heads, self.head_dim)
            .transpose(1, 2)
        )
        hg_v_cache = (
            self.v_proj(higher_global)
            .view(bsz, num_global_slots, self.num_key_value_heads, self.head_dim)
            .transpose(1, 2)
        )
        hg_k = repeat_kv(hg_k_cache, self.num_key_value_groups)
        hg_v = repeat_kv(hg_v_cache, self.num_key_value_groups)
    else:
        hg_k = hg_v = None
        hg_k_cache = hg_v_cache = None

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

    if local_reprs is not None:
        lm_flat = local_reprs.view(bsz * num_groups, num_local_slots, hidden_size)
        lm_k_cache = (
            self.k_proj(lm_flat)
            .view(
                bsz * num_groups,
                num_local_slots,
                self.num_key_value_heads,
                self.head_dim,
            )
            .transpose(1, 2)
        )
        lm_v_cache = (
            self.v_proj(lm_flat)
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
        # Use reshape instead of view to handle non-contiguous tensors
        lm_k_cache = lm_k_cache.reshape(
            bsz, num_groups * num_local_slots, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        lm_v_cache = lm_v_cache.reshape(
            bsz, num_groups * num_local_slots, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
    else:
        lm_k = lm_v = None
        lm_k_cache = lm_v_cache = None

    if use_cache:
        if INCLUDE_REPR_IN_KV_CACHE and (
            hg_k_cache is not None or lm_k_cache is not None
        ):
            cache_components_k = []
            cache_components_v = []
            cache_repr_len = 0

            if hg_k_cache is not None:
                cache_components_k.append(hg_k_cache)
                cache_components_v.append(hg_v_cache)
                cache_repr_len += num_global_slots
            if lm_k_cache is not None:
                cache_components_k.append(lm_k_cache)
                cache_components_v.append(lm_v_cache)
                cache_repr_len += num_groups * num_local_slots

            cache_components_k.append(key_states_for_cache)
            cache_components_v.append(value_states_for_cache)

            full_key_cache = torch.cat(cache_components_k, dim=2)
            full_value_cache = torch.cat(cache_components_v, dim=2)
            past_key_value = (full_key_cache, full_value_cache)

            self._hici_cache_repr_len = cache_repr_len

            if not _HICI_CACHE_PRINTED:
                layer_idx = getattr(self, "layer_idx", 0)
                if layer_idx == 0:
                    print(f"\n{'=' * 60}")
                    print(f"[HiCI Prefill ] INCLUDE_REPR_IN_KV_CACHE=True")
                    print(f"  higher_global slots     : {num_global_slots}")
                    print(f"  local_repr slots      : {num_groups * num_local_slots}")
                    print(f" repr_len () : {cache_repr_len}")
                    print(f"  token_len               : {q_len}")
                    print(f" KV cache : {full_key_cache.shape[2]}")
                    print(f"  _hici_cache_repr_len  : {self._hici_cache_repr_len}")
                    print(
                    f" : {cache_repr_len} + {q_len} = {cache_repr_len + q_len} ( {full_key_cache.shape[2]})"
                    )
                    print(f"{'=' * 60}\n")
                    _HICI_CACHE_PRINTED = True
        else:
            past_key_value = (key_states_for_cache, value_states_for_cache)
            self._hici_cache_repr_len = 0

            if not _HICI_CACHE_PRINTED:
                layer_idx = getattr(self, "layer_idx", 0)
                if layer_idx == 0:
                    print(
                        f"[HiCI KV Cache] INCLUDE_REPR_IN_KV_CACHE=False, only tokens in cache"
                    )
                    _HICI_CACHE_PRINTED = True
    else:
        past_key_value = None

    repr_len = num_global_slots + num_local_slots
    kv_len = repr_len + group_size

    if repr_len > 0:
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
        if hg_k is not None:
            all_k[:, :, :, offset : offset + num_global_slots, :] = hg_k.unsqueeze(2)
            all_v[:, :, :, offset : offset + num_global_slots, :] = hg_v.unsqueeze(2)
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
    kv_mask[:, repr_len:] = q_mask

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

USE_TRAINING_FUNCTION_FOR_INFERENCE = False

def replace_llama_attn_hici_inference(
    use_training_function: bool = None,
    include_repr_in_kv_cache: bool = None,
    disable_hici_in_prefill: bool = None,
):
    """


        replace_llama_attn_hici_inference(
            use_training_function=False,
            include_repr_in_kv_cache=True,
            disable_hici_in_prefill=False,
        )

        replace_llama_attn_hici_inference(use_training_function=True)
    """
    global \
        USE_TRAINING_FUNCTION_FOR_INFERENCE, \
        INCLUDE_REPR_IN_KV_CACHE, \
        DISABLE_HICI_IN_PREFILL

    if use_training_function is not None:
        USE_TRAINING_FUNCTION_FOR_INFERENCE = use_training_function
    if include_repr_in_kv_cache is not None:
        INCLUDE_REPR_IN_KV_CACHE = include_repr_in_kv_cache
    if disable_hici_in_prefill is not None:
        DISABLE_HICI_IN_PREFILL = disable_hici_in_prefill

    print("=" * 80)
    print("Replacing LlamaAttention.forward with HiCI SFT Inference")
    print("=" * 80)
    print(f"  group_size_ratio                    : {group_size_ratio}")
    print(
        f"  USE_TRAINING_FUNCTION_FOR_INFERENCE : {USE_TRAINING_FUNCTION_FOR_INFERENCE}"
    )
    print(f"  INCLUDE_REPR_IN_KV_CACHE          : {INCLUDE_REPR_IN_KV_CACHE}")
    print(f"  DISABLE_HICI_IN_PREFILL             : {DISABLE_HICI_IN_PREFILL}")
    print("=" * 80)

    if USE_TRAINING_FUNCTION_FOR_INFERENCE:
        print("\n*** forward_flashattn_hierarchical ***")
        print("*** KV cache，- ***\n")

        transformers.models.llama.modeling_llama.LlamaModel._prepare_decoder_attention_mask = _prepare_decoder_attention_mask_inference
        transformers.models.llama.modeling_llama.LlamaAttention.forward = (
            forward_flashattn_hierarchical
        )
    else:
        print("\n*** forward_hici_sft_inference ***")
        print("*** KV cache，- ***\n")

        transformers.models.llama.modeling_llama.LlamaModel._prepare_decoder_attention_mask = _prepare_decoder_attention_mask_inference
        transformers.models.llama.modeling_llama.LlamaAttention.forward = (
            forward_hici_sft_inference
        )

def replace_llama_attn(
    use_flash_attn=True,
    use_full=False,
    inference=False,
    eval_mode=None,
    use_optimized=True,
    use_optimized_plus=True,
    use_optimized_plus_norope=False,
    use_hierarchical_forward=False,
):
    """
    Replace LlamaAttention forward function with optimized implementations.

    IMPORTANT: This function only patches the forward method.
    To register LocalConstructor parameters, call register_hici_to_model()
    after loading the model and before initializing the optimizer.

    Args:
        use_flash_attn: Whether to use flash attention (default: True)
        use_full: Whether to use full attention without chunking (default: False)
            - True: forward_flashattn_full (full attention, no HiCI)
            - False: forward_flashattn (chunked attention with HiCI, same as training)
        inference: Whether in inference mode (default: False)
        eval_mode: Evaluation mode selection (default: None)
            - None: Use use_full parameter for backward compatibility
            - "chunked": Chunked attention with HiCI (same as training)
            - "full": Full attention without HiCI
            - "full_hierarchical": Full attention + HiCI (evaluation way3)
        use_optimized: Whether to use optimized forward function (default: True)
        use_optimized_plus: Whether to use the improved version (default: False)
        use_optimized_plus_norope: Whether to use the experimental no-RoPE version (default: False)
        use_hierarchical_forward: Whether to use hierarchical forward (default: False)
            - False: Use other forward functions based on other parameters
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
            if eval_mode == "full_hierarchical":
                transformers.models.llama.modeling_llama.LlamaAttention.forward = (
                    forward_flashattn_full_hierarchical
                )
                if rank == 0:
                    print(f"   eval_mode: {eval_mode}")
            elif eval_mode == "full" or (eval_mode is None and use_full):
                transformers.models.llama.modeling_llama.LlamaAttention.forward = (
                    forward_flashattn_full
                )
                if rank == 0:
                    print("  Using forward_flashattn_full")
            else:
                if use_hierarchical_forward:
                    transformers.models.llama.modeling_llama.LlamaAttention.forward = (
                        forward_flashattn_hierarchical
                    )
                    if rank == 0:
                        print(
                        " Using forward_flashattn_hierarchical (+)"
                        )
                elif use_optimized_plus_norope:
                    if rank == 0:
                        warnings.warn(
                            "  use_optimized_plus_norope is not supported in SFT version, "
                            "falling back to use_optimized_plus"
                        )
                    transformers.models.llama.modeling_llama.LlamaAttention.forward = (
                        forward_flashattn_optimized_plus
                    )
                    if rank == 0:
                        print(
                            "    Using forward_flashattn_optimized_plus (reuse K/V projections)"
                        )
                elif use_optimized_plus:
                    transformers.models.llama.modeling_llama.LlamaAttention.forward = (
                        forward_flashattn_optimized_plus
                    )
                    if rank == 0:
                        print(
                            "    Using forward_flashattn_optimized_plus (reuse K/V projections)"
                        )
                elif use_optimized:
                    transformers.models.llama.modeling_llama.LlamaAttention.forward = (
                        forward_flashattn_optimized
                    )
                    if rank == 0:
                        print(
                            "    Using forward_flashattn_optimized (merged projections + vectorized mask)"
                        )
                else:
                    transformers.models.llama.modeling_llama.LlamaAttention.forward = (
                        forward_flashattn
                    )
                    if rank == 0:
                        print("  Using forward_flashattn")
    else:
        transformers.models.llama.modeling_llama.LlamaAttention.forward = (
            forward_noflashattn
        )

def register_hici_to_model(
    model,
    num_local_slots=16,
    global_slots=2,
    num_chunks: Optional[int] = 4,
    num_heads=32,
    use_bottleneck=True,
    bottleneck_dim=4096,  
    use_local_summary=True,
    use_hierarchical=True,
    use_flash_plus=True,
    use_flash: Optional[bool] = True,
    use_llama_init=False,
    use_shared_compressor=True,
    shared_compress_dim=128,
):
    """
    Register LocalConstructor (and optionally GlobalIntegrator) to each LlamaAttention layer.

    This MUST be called after model loading and before optimizer initialization!

    Args:
        model: LlamaForCausalLM or PeftModelForCausalLM
        num_local_slots: Number of local local query slots (for LocalConstructor, default: 16)
        global_slots: Number of higher-level global slots (for GlobalIntegrator, default: 16)
        num_heads: Number of attention heads (default: 32)
        bottleneck_dim: Bottleneck dimension for efficiency (default: 2048)
        use_hierarchical: If True, also register GlobalIntegrator (default: False)

    Example usage in fine-tune.py:
        # 1. Load model
        model = transformers.AutoModelForCausalLM.from_pretrained(...)

        # 2. Replace attention mechanism
        replace_llama_attn(use_flash_attn=True)

        # 3. Register HiCI (BEFORE optimizer!)
        # For simple HiCI:
        register_hici_to_model(model, num_local_slots=16)

        # For HiCI:
        register_hici_to_model(
            model,
            num_local_slots=16,  # local slots
            global_slots=16,       # higher-level global slots
            use_hierarchical=True
        )

        # 4. Setup LoRA (if needed)
        model = get_peft_model(model, lora_config)

        # 5. NOW initialize optimizer (will include HiCI parameters)
        optimizer = torch.optim.AdamW(model.parameters(), lr=...)
    """
    rank = dist.get_rank() if dist.is_initialized() else 0

    if use_hierarchical and not use_local_summary:
        if rank == 0:
            print("\n" + "=" * 80)
            print(" ERROR: Invalid Configuration!")
            print("=" * 80)
            print("use_hierarchical=True requires use_local_summary=True")
            print(
                "Reason: HierarchicalAggregator needs local memories from LocalConstructor"
            )
            print()
            print("Fix: Set use_local_summary=True, or set use_hierarchical=False")
            print("=" * 80 + "\n")
        raise ValueError(
            "Invalid configuration: use_hierarchical=True requires use_local_summary=True. "
            "HierarchicalAggregator needs local memories from LocalConstructor to aggregate."
        )

    if rank == 0:
        print("\n" + "=" * 80)
        config_str = []
        if use_local_summary:
            config_str.append("Local Constructor")
        if use_hierarchical:
            config_str.append("Global Integrator")

        if config_str:
            print(f" Registering: {' + '.join(config_str)}")
        else:
            print(" No HiCI modules enabled!")
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

    model_dtype = llama_model.embed_tokens.weight.dtype
    rank = dist.get_rank() if dist.is_initialized() else 0
    if rank == 0:
        print(f"   Model dtype: {model_dtype}")

    embed_weight = llama_model.embed_tokens.weight.data  # [vocab_size, hidden_size]

    # Register modules to each attention layer
    for layer_idx, layer in enumerate(llama_model.layers):
        attn = layer.self_attn
        attn.layer_idx = layer_idx  # Important for layer identification

        # Module 1: LocalConstructor (always register)
        if use_local_summary:
            if use_flash_plus:
                attn.local_constructor = LocalConstructorFlashPlus(
                    hidden_size=hidden_size,
                    num_local_slots=num_local_slots,
                    num_heads=num_heads,
                    init_from_embeddings=embed_weight,
                ).to(model_dtype)
            elif use_flash:
                attn.local_constructor = LocalConstructorFlash(
                    hidden_size=hidden_size,
                    num_local_slots=num_local_slots,
                    num_heads=num_heads,
                    init_from_embeddings=embed_weight,
                ).to(model_dtype)
            else:
                attn.local_constructor = LocalConstructorMulti(
                    hidden_size=hidden_size,
                    num_local_slots=num_local_slots,
                    num_heads=num_heads,
                    init_from_embeddings=embed_weight,
                    init_from_llama_attn=attn if use_llama_init else None,
                    use_bottleneck=use_bottleneck,
                    bottleneck_dim=bottleneck_dim,
                ).to(model_dtype)
        # Module 2: Global Integrator (optional)
        if use_hierarchical:
            if use_shared_compressor:
                attn.global_integrator = GlobalIntegratorShared(
                    hidden_size=hidden_size,
                    global_slots=global_slots,
                    compress_dim=512,
                    shared_compress_dim=shared_compress_dim,
                    init_from_embeddings=embed_weight,
                    use_high_norm_init=True,
                ).to(model_dtype)
            else:
                attn.global_integrator = GlobalIntegrator(
                    hidden_size=hidden_size,
                    global_slots=global_slots,
                    compress_dim=512,
                    init_from_embeddings=embed_weight,
                    use_high_norm_init=True,
                ).to(model_dtype)

    total_params = sum(p.numel() for p in model.parameters())

    local_constructor_params = 0
    aggregator_params = 0

    if use_local_summary or use_hierarchical:
        for name, param in model.named_parameters():
            if "local_constructor" in name:
                local_constructor_params += param.numel()
            elif "global_integrator" in name:
                aggregator_params += param.numel()

    if rank == 0:
        print()
        print("=" * 80)
        print(" HiCI Module Registration Complete")
        print("=" * 80)

        print(f"Model: {total_params:,} params ({total_params / 1e9:.2f}B)")
        print(f"Layers: {len(llama_model.layers)}")

        if use_local_summary and use_hierarchical:
            total_hici_params = local_constructor_params + aggregator_params
            print(f"\nRegistered Modules:")
            print(f"   Local Constructor ({local_constructor_params:,} params)")
            print(f"   Global Integrator ({aggregator_params:,} params)")
            print(
                f"\nTotal HiCI Params: {total_hici_params:,} ({total_hici_params / total_params * 100:.2f}%)"
            )

        elif use_local_summary and not use_hierarchical:
            print(f"\nRegistered Modules:")
            print(f"   Local Constructor ({local_constructor_params:,} params)")
            print(
                f"\nTotal HiCI Params: {local_constructor_params:,} ({local_constructor_params / total_params * 100:.2f}%)"
            )

        elif not use_local_summary and use_hierarchical:
            print(f"\n Warning: Hierarchical registered without Local Constructor!")
            print(f"   Global Integrator ({aggregator_params:,} params)")
            print(
                f"\nTotal HiCI Params: {aggregator_params:,} ({aggregator_params / total_params * 100:.2f}%)"
            )

        else:
            print(f"\nRegistered Modules: None")

        print("=" * 80 + "\n")

