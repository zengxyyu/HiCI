# Modified based on https://github.com/lm-sys/FastChat
#
# ✅ SFT版本 (Supervised Fine-Tuning Version)
#
# 主要区别：
# 1. 添加 sft_group_size = 8192（固定group size）
# 2. group_size计算逻辑：
#    - 如果 q_len % 4096 == 0：使用动态计算 int(q_len * group_size_ratio)
#    - 否则：使用固定值 sft_group_size = 8192
# 3. 用于处理SFT数据中的不规则序列长度（如3000, 5000, 12000等）
#
# 对应训练脚本: supervised-fine-tune.py

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

group_size_ratio = 1 / 8  # Fraction of tokens per chunk for LocalConstructor
sft_group_size = 8192  # Fixed group size for SFT (handles irregular sequence lengths)

# ============================================================================
# HiCI 推理 KV Cache 配置
# ============================================================================
# Controls whether KV cache includes HiCI slots during decode
#
# True:  KV cache = [higher_global, local_slots, tokens]
#        - HiCI slots are accessible during decode
#        - position_ids adjustment needed (slots don't consume positions)
#        - attention_mask must include slot positions
#
# False: KV cache = [tokens]
#        - HiCI slots not in KV cache during decode
#
# 注意：当 True 时，会自动：
#       1. Correct position_ids during decode (subtract slot length)
#       2. Extend attention_mask (prepend slot portion)
INCLUDE_HICI_IN_KV_CACHE = True

# 测试开关：关闭 Prefill 阶段的 HiCI，使用标准 Flash Attention
# 用于测试推理函数本身是否正确（排除 HiCI 的影响）
DISABLE_HICI_IN_PREFILL = False

# 全局打印控制标志（避免重复打印）
_HICI_INFERENCE_PRINTED = False
_HICI_GROUP_PRINTED = False
_HICI_CACHE_PRINTED = False


# recurrence_size = 128  # Number of tokens to carry from previous chunk (Transformer-XL style)


# 版本1 没有多头的最初版本
class LocalConstructor(nn.Module):
    """
    Learnable query slots for local context construction (LocalConstructor).

    This module is registered as a sub-module of LlamaAttention, ensuring:
    1. Parameters are properly registered in model.parameters()
    2. Optimizer tracks and updates these parameters
    3. Saved/loaded with model checkpoints

    Args:
        hidden_size: Model hidden dimension (e.g., 4096 for Llama-2-7B)
        num_local_slots: Number of learnable query slots (default: 16)
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

        # Learnable memory slots: [num_slots, hidden_size]
        # ✅ 方案 2: 从预训练嵌入初始化（最优策略）
        # if init_from_embeddings is not None:
        if False:
            # 从预训练嵌入中随机采样 num_local_slots 个作为初始值
            # 理论依据: 预训练嵌入已经包含丰富的语义信息，比随机噪声好得多
            indices = torch.randperm(init_from_embeddings.size(0))[:num_local_slots]
            self.memory_slots = nn.Parameter(init_from_embeddings[indices].clone())
            # ✅ 只在 rank 0 打印
            rank = dist.get_rank() if dist.is_initialized() else 0
            if rank == 0:
                print(
                    f"    ✅ Initialized memory_slots from pretrained embeddings (sampled {num_local_slots} tokens)"
                )
        else:
            # Fallback: 使用 LLaMA 标准初始化 (std=0.02)
            std = 1.0 / math.sqrt(hidden_size)  # ≈ 0.0156 (太小！)
            # std = 0.02  # LLaMA/GPT 标准
            self.memory_slots = nn.Parameter(
                torch.randn(num_local_slots, hidden_size) * std
            )
            rank = dist.get_rank() if dist.is_initialized() else 0
            if rank == 0:
                print(f"    ⚠️  Fallback: Initialized memory_slots with std={std}")

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

        # Expand memory for batch
        slots_input = self.memory_slots.unsqueeze(0).expand(
            bsz, -1, -1
        )  # [bsz, num_slots, hidden_size]

        # Cross-attention: memory attends to full sequence
        Q_mem = self.q_proj(slots_input)  # [bsz, num_slots, hidden_size]
        K_seq = self.k_proj(hidden_states)  # [bsz, seq_len, hidden_size]
        V_seq = self.v_proj(hidden_states)  # [bsz, seq_len, hidden_size]

        # Compute attention scores
        scores = torch.matmul(Q_mem, K_seq.transpose(-2, -1)) / math.sqrt(
            self.hidden_size
        )
        attn_weights = torch.softmax(scores, dim=-1)  # [bsz, num_slots, seq_len]

        # Apply attention to get global context
        global_context = torch.matmul(
            attn_weights, V_seq
        )  # [bsz, num_slots, hidden_size]

        return global_context


# 版本1的基础上 加了多头注意力和mask的实现 已检查
class LocalConstructorMulti(nn.Module):
    """
    Learnable query slots for local context construction (LocalConstructor).

    多头注意力版本（不使用 Flash Attention），使用标准 PyTorch 实现，支持：
    1. 多头注意力 - 更好的表达能力
    2. Attention mask - 正确处理 padding tokens
    3. Bottleneck 压缩 - 可选的信息瓶颈机制
    4. LLaMA 权重初始化 - 从预训练权重 warm start

    This module is registered as a sub-module of LlamaAttention, ensuring:
    1. Parameters are properly registered in model.parameters()
    2. Optimizer tracks and updates these parameters
    3. Saved/loaded with model checkpoints

    Args:
        hidden_size: Model hidden dimension (e.g., 4096 for Llama-2-7B)
        num_local_slots: Number of learnable query slots (default: 16)
        num_heads: Number of attention heads (default: 32)
        init_from_embeddings: Optional pretrained embeddings for memory_slots initialization
        init_from_llama_attn: Optional LlamaAttention layer for Q/K/V projection initialization
        use_bottleneck: Whether to use bottleneck compression (default: True)
        bottleneck_dim: Bottleneck dimension (default: 2048)
    """

    # 类变量：控制只打印一次初始化信息
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

        # Learnable memory slots: [num_slots, hidden_size]
        # ✅ 方案 2: 从预训练嵌入初始化（最优策略）
        if False:
            # 从预训练嵌入中随机采样 num_local_slots 个作为初始值
            # 理论依据: 预训练嵌入已经包含丰富的语义信息，比随机噪声好得多
            indices = torch.randperm(init_from_embeddings.size(0))[:num_local_slots]
            self.memory_slots = nn.Parameter(init_from_embeddings[indices].clone())
            rank = dist.get_rank() if dist.is_initialized() else 0
            if rank == 0:
                print(
                    f"    ✅ Initialized memory_slots from pretrained embeddings (sampled {num_local_slots} tokens)"
                )
        else:
            # Fallback: 使用 LLaMA 标准初始化 (std=0.02)
            std = 1.0 / math.sqrt(hidden_size)  # ≈ 0.0156
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
            # Validate bottleneck_dim divisibility
            assert bottleneck_dim % num_heads == 0, (
                f"bottleneck_dim ({bottleneck_dim}) must be divisible by num_heads ({num_heads})"
            )
            rank = dist.get_rank() if dist.is_initialized() else 0
            if rank == 0:
                print(
                    f"✅ LocalConstructorMulti: bottleneck_dim: {bottleneck_dim}, num_heads: {num_heads}"
                )

            # 直接投影: hidden_size -> bottleneck_dim
            self.q_proj = nn.Linear(hidden_size, bottleneck_dim, bias=False)
            self.k_proj = nn.Linear(hidden_size, bottleneck_dim, bias=False)
            self.v_proj = nn.Linear(hidden_size, bottleneck_dim, bias=False)

            # 输出投影: bottleneck_dim -> hidden_size
            self.o_proj = nn.Linear(bottleneck_dim, hidden_size, bias=False)

            # Effective dimensions for attention computation
            self.effective_dim = bottleneck_dim
            self.effective_head_dim = bottleneck_dim // num_heads
        else:
            # Standard full-size projections: 4096 -> 4096
            self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
            self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
            self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
            self.o_proj = None  # 不需要额外的输出投影

            # Use original dimensions
            self.effective_dim = hidden_size
            self.effective_head_dim = self.head_dim

        # ========== 方案C：从 LLaMA Attention 层初始化 Q/K/V 投影 ==========
        # 理论依据：
        # 1. LLaMA 的 q_proj/k_proj/v_proj 是一起预训练的，有内在的"配对关系"
        # 2. 从预训练权重初始化，Q 和 K 在同一个语义空间，初始 Q×K^T 有意义
        # 3. 投影仍然是独立的参数，可以继续微调（不是冻结的！）
        # 4. 这是 Warm Initialization，比随机初始化收敛更快
        if init_from_llama_attn is not None and not use_bottleneck:
            # 只有在不使用 bottleneck 时才能从 LLaMA 初始化（维度需要匹配）
            rank = dist.get_rank() if dist.is_initialized() else 0
            layer_idx = getattr(self, "layer_idx", 0)  # 获取当前层索引
            with torch.no_grad():
                self.q_proj.weight.copy_(init_from_llama_attn.q_proj.weight)
                self.k_proj.weight.copy_(init_from_llama_attn.k_proj.weight)
                self.v_proj.weight.copy_(init_from_llama_attn.v_proj.weight)
            if rank == 0 and layer_idx == 0:
                print(
                    f"✅ [LocalConstructorMulti 方案C] Initialized Q/K/V projections from LLaMA pretrained weights"
                )

    def forward(self, hidden_states, attention_mask=None):
        """
        Compute global context via multi-head cross-attention (standard PyTorch, no Flash Attention).

        使用标准多头注意力实现：
        - Q: query slots (no padding), fixed length = num_slots
        - K/V: input sequence (可能有 padding), 使用 attention_mask 处理

        数值稳定性优势：
        - 标准 PyTorch attention: 精确计算，无分块近似误差
        - 适合需要高数值精度的场景

        内存占用：
        - 中间张量: O(num_slots × seq_len) = O(M × N)
        - 对于 8 slots × 100k tokens: ~0.8M 元素 ≈ 6 MB (bfloat16)

        Args:
            hidden_states: [bsz, seq_len, hidden_size] - full input sequence
            attention_mask: [bsz, seq_len] - 1 for valid tokens, 0 for padding (optional)

        Returns:
            global_context: [bsz, num_slots, hidden_size] - global summary
        """
        bsz, seq_len, _ = hidden_states.shape

        # Expand memory for batch
        slots_input = self.memory_slots.unsqueeze(0).expand(
            bsz, -1, -1
        )  # [bsz, num_slots, hidden_size]

        # Cross-attention projections: 直接投影到目标维度 (bottleneck or full)
        Q_mem = self.q_proj(slots_input)  # [bsz, num_slots, effective_dim]
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
        global_context = attn_output.view(
            bsz, self.num_local_slots, self.effective_dim
        )

        # Apply output projection if using bottleneck: effective_dim -> hidden_size
        if self.o_proj is not None:
            global_context = self.o_proj(
                global_context
            )  # [bsz, num_slots, hidden_size]

        return global_context


# 版本2 加了flash attn和padding的实现 多头
# region ===========================================================================
class LocalConstructorFlash(nn.Module):
    """
    Learnable query slots for local context construction (LocalConstructor).

    使用 Flash Attention 实现高效的 cross-attention，支持：
    1. 超长序列（100k+）- 内存复杂度 O(N) 而不是 O(N²)
    2. 正确处理 padding - 使用 unpad_input 移除 padding tokens

    This module is registered as a sub-module of LlamaAttention, ensuring:
    1. Parameters are properly registered in model.parameters()
    2. Optimizer tracks and updates these parameters
    3. Saved/loaded with model checkpoints

    Args:
        hidden_size: Model hidden dimension (e.g., 4096 for Llama-2-7B)
        num_local_slots: Number of learnable query slots (default: 16)
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

        # Learnable memory slots: [num_slots, hidden_size]
        # ✅ 方案 2: 从预训练嵌入初始化（最优策略）
        # if init_from_embeddings is not None:
        if False:
            # 从预训练嵌入中随机采样 num_local_slots 个作为初始值
            # 理论依据: 预训练嵌入已经包含丰富的语义信息，比随机噪声好得多
            indices = torch.randperm(init_from_embeddings.size(0))[:num_local_slots]
            self.memory_slots = nn.Parameter(init_from_embeddings[indices].clone())
            # ✅ 只在 rank 0 打印
            rank = dist.get_rank() if dist.is_initialized() else 0
            if rank == 0:
                print(
                    f"    ✅ Initialized memory_slots from pretrained embeddings (sampled {num_local_slots} tokens)"
                )
        else:
            # Fallback: 使用 LLaMA 标准初始化 (std=0.02)
            std = 1.0 / math.sqrt(hidden_size)  # ≈ 0.0156 (太小！)
            # std = 0.02  # LLaMA/GPT 标准
            self.memory_slots = nn.Parameter(
                torch.randn(num_local_slots, hidden_size) * std
            )
            rank = dist.get_rank() if dist.is_initialized() else 0
            if rank == 0:
                print(f"    ⚠️  Fallback: Initialized memory_slots with std={std}")

        # Cross-attention projections for summarization
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, hidden_states, attention_mask=None):
        """
        Compute global context via Flash Attention cross-attention.

        使用 flash_attn_varlen_kvpacked_func 实现：
        - Q: query slots (no padding), fixed length = num_slots
        - K/V: input sequence (可能有 padding), 使用 unpad_input 移除 padding

        内存优势：
        - 标准 matmul: O(num_slots × seq_len) 中间张量
        - Flash Attention: O(1) 中间张量（分块计算）

        Args:
            hidden_states: [bsz, seq_len, hidden_size] - full input sequence
            attention_mask: [bsz, seq_len] - 1 for valid, 0 for padding (optional)

        Returns:
            global_context: [bsz, num_slots, hidden_size] - global summary
        """
        bsz, seq_len, _ = hidden_states.shape

        # Expand memory for batch
        slots_input = self.memory_slots.unsqueeze(0).expand(
            bsz, -1, -1
        )  # [bsz, num_slots, hidden_size]

        # Cross-attention projections
        Q_mem = self.q_proj(slots_input)  # [bsz, num_slots, hidden_size]
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

            # cu_seqlens_q: 每个 batch 样本的 Q 长度都是 num_slots
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
                self.num_local_slots,  # max_seqlen_q (固定)
                max_seqlen_kv,  # max_seqlen_kv (batch 中最长的有效长度)
                dropout_p=0.0,
                softmax_scale=None,  # 默认 1/sqrt(head_dim)
                causal=False,  # Cross-attention 不需要 causal mask
            )
            # output_unpad: [bsz * num_slots, num_heads, head_dim]

            # Reshape back: [bsz, num_slots, hidden_size]
            global_context = rearrange(
                output_unpad, "(b s) h d -> b s (h d)", b=bsz, s=self.num_local_slots
            )
        else:
            # ========== 无 padding，使用简单的 flash_attn_func ==========
            # 这是最高效的情况，直接使用 Flash Attention
            global_context = flash_attn_func(
                Q_mem,  # [bsz, num_slots, num_heads, head_dim]
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
# LocalConstructor version 3 (LocalConstructorFlashPlus)
class LocalConstructorFlashPlus(nn.Module):
    """
    Learnable query slots for local context construction (LocalConstructor).

    改进版：复用 LLaMA 的 K/V 投影，只需要自己的 Q 投影。

    优势：
    1. 高效：不需要重复的 K/V 投影
    2. K/V 已经有 RoPE 位置编码
    3. 与主注意力共享 K/V 表示

    使用 Flash Attention 实现高效的 cross-attention，支持：
    1. 超长序列（100k+）- 内存复杂度 O(N) 而不是 O(N²)
    2. 正确处理 padding - 使用 unpad_input 移除 padding tokens

    Args:
        hidden_size: Model hidden dimension (e.g., 4096 for Llama-2-7B)
        num_local_slots: Number of learnable query slots (default: 16)
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

        # Learnable memory slots: [num_slots, hidden_size]
        std = 1.0 / math.sqrt(hidden_size)
        self.memory_slots = nn.Parameter(
            torch.randn(num_local_slots, hidden_size) * std
        )
        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank == 0:
            print(
                f"    [LocalConstructorFlashPlus] Initialized memory_slots with std={std}"
            )
            print(
                f"    [LocalConstructorFlashPlus] 复用 LLaMA K/V 投影，只有自己的 Q 投影"
            )

        # 只需要 Q 投影！K/V 复用 LLaMA 的投影结果
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        # 不需要 k_proj 和 v_proj 了！

    def forward(self, key_states, value_states, attention_mask=None):
        """
        Compute global context via Flash Attention cross-attention.

        Improved: accepts pre-projected+RoPE K/V, only projects its own query slots.

        Args:
            key_states: [bsz, seq_len, num_heads, head_dim] - 已投影 + RoPE
            value_states: [bsz, seq_len, num_heads, head_dim] - 已投影
            attention_mask: [bsz, seq_len] - 1 for valid, 0 for padding (optional)

        Returns:
            global_context: [bsz, num_slots, hidden_size] - global summary
        """
        bsz = key_states.shape[0]

        # Expand memory for batch and project to Q
        slots_input = self.memory_slots.unsqueeze(0).expand(
            bsz, -1, -1
        )  # [bsz, num_slots, hidden_size]

        # Memory queries (只需要投影 memory，不需要投影输入！)
        Q_mem = self.q_proj(slots_input)  # [bsz, num_slots, hidden_size]

        # Reshape Q for multi-head attention: [bsz, num_slots, num_heads, head_dim]
        Q_mem = Q_mem.view(bsz, self.num_local_slots, self.num_heads, self.head_dim)

        # K/V 已经是正确的形状了: [bsz, seq_len, num_heads, head_dim]
        K_seq = key_states
        V_seq = value_states

        if attention_mask is not None:
            # ========== Flash Attention + unpad (正确处理 padding) ==========
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

            # cu_seqlens_q: 每个 batch 样本的 Q 长度都是 num_slots
            cu_seqlens_q = torch.arange(
                0,
                (bsz + 1) * self.num_local_slots,
                self.num_local_slots,
                device=key_states.device,
                dtype=torch.int32,
            )

            # Flash Attention 变长 cross-attention
            output_unpad = flash_attn_varlen_kvpacked_func(
                q_unpad,  # [bsz * num_slots, num_heads, head_dim]
                kv_unpad,  # [total_valid_kv, 2, num_heads, head_dim]
                cu_seqlens_q,  # [bsz + 1]
                cu_seqlens_kv,  # [bsz + 1]
                self.num_local_slots,  # max_seqlen_q (固定)
                max_seqlen_kv,  # max_seqlen_kv
                dropout_p=0.0,
                softmax_scale=None,
                causal=False,  # Cross-attention 不需要 causal mask
            )

            # Reshape back: [bsz, num_slots, hidden_size]
            global_context = rearrange(
                output_unpad, "(b s) h d -> b s (h d)", b=bsz, s=self.num_local_slots
            )
        else:
            # ========== 无 padding，使用简单的 flash_attn_func ==========
            global_context = flash_attn_func(
                Q_mem,  # [bsz, num_slots, num_heads, head_dim]
                K_seq,  # [bsz, seq_len, num_heads, head_dim]
                V_seq,  # [bsz, seq_len, num_heads, head_dim]
                dropout_p=0.0,
                softmax_scale=None,
                causal=False,
            )
            # Reshape: [bsz, num_slots, hidden_size]
            global_context = rearrange(global_context, "b s h d -> b s (h d)")

        return global_context


# region ===========================================================================
# ============================================================================
# [NEW] Competitive Write Memory - Top-k Token Selection
# Theory: Winner-Take-All, Competitive Learning
# ============================================================================
# class CompetitiveWriteMemory(nn.Module):
#     """
#     Competitive memory writing: Only important tokens write to global memory.

#     Key innovations:
#     1. Importance scoring: Learn which tokens are important
#     2. Top-k selection: Only top 25% tokens participate in memory update
#     3. Winner-take-all: Competition for limited memory slots

#     Theory:
#     - Inspired by competitive learning in neural networks
#     - Information bottleneck: Compress only essential information
#     - Cognitive science: Working memory has limited capacity

#     Args:
#         hidden_size: Model hidden dimension (e.g., 4096)
#         num_local_slots: Number of Global Representation Slots (default: 16)
#         num_heads: Number of attention heads (default: 32)
#         bottleneck_dim: Bottleneck dimension for efficiency (default: 2048)
#         write_ratio: Fraction of tokens that can write to memory (default: 0.25)
#     """

#     def __init__(
#         self,
#         hidden_size: int = 4096,
#         num_local_slots: int = 16,
#         num_heads: int = 32,
#         bottleneck_dim: int = 2048,
#         write_ratio: float = 0.25,
#     ):
#         super().__init__()

#         self.hidden_size = hidden_size
#         self.num_local_slots = num_local_slots
#         self.num_heads = num_heads
#         self.bottleneck_dim = bottleneck_dim
#         self.head_dim = bottleneck_dim // num_heads
#         self.write_ratio = write_ratio

#         assert bottleneck_dim % num_heads == 0, \
#             f"bottleneck_dim ({bottleneck_dim}) must be divisible by num_heads ({num_heads})"

#         # ====================================================================
#         # Memory Slots: Learnable global context
#         # ====================================================================
#         self.memory_slots = nn.Parameter(
#             torch.empty(num_local_slots, hidden_size)
#         )
#         nn.init.xavier_uniform_(self.memory_slots)  # Better initialization

#         # ====================================================================
#         # Importance Scorer: Learns which tokens are important
#         # ====================================================================
#         self.importance_scorer = nn.Sequential(
#             nn.Linear(hidden_size, hidden_size // 4),
#             nn.GELU(),
#             nn.Linear(hidden_size // 4, 1),
#         )

#         # ====================================================================
#         # Attention Projections: Multi-head cross-attention
#         # ====================================================================
#         self.q_proj = nn.Linear(hidden_size, bottleneck_dim, bias=False)
#         self.k_proj = nn.Linear(hidden_size, bottleneck_dim, bias=False)
#         self.v_proj = nn.Linear(hidden_size, bottleneck_dim, bias=False)
#         self.o_proj = nn.Linear(bottleneck_dim, hidden_size, bias=False)

#         # Initialize projections with smaller std (Llama-style)
#         for proj in [self.q_proj, self.k_proj, self.v_proj, self.o_proj]:
#             nn.init.normal_(proj.weight, mean=0.0, std=0.02)

#     def forward(self, hidden_states, return_debug_info=False):
#         """
#         Competitive memory update with top-k token selection.

#         Args:
#             hidden_states: [bsz, seq_len, hidden_size] - Input sequence
#             return_debug_info: If True, return debug statistics

#         Returns:
#             global_context: [bsz, num_slots, hidden_size] - Updated memory
#             debug_info: (Optional) Dictionary with selection statistics
#         """
#         bsz, seq_len, _ = hidden_states.shape

#         # ====================================================================
#         # Step 1: Importance Scoring (Competitive Selection)
#         # ====================================================================
#         # Compute importance score for each token
#         importance_scores = self.importance_scorer(hidden_states).squeeze(-1)
#         # [bsz, seq_len]

#         # Top-k selection: Only winners can write to memory
#         k = max(int(seq_len * self.write_ratio), self.num_local_slots)
#         top_k_scores, top_k_indices = torch.topk(importance_scores, k, dim=-1)
#         # top_k_indices: [bsz, k]

#         # Extract winner tokens
#         winner_tokens = torch.gather(
#             hidden_states,
#             dim=1,
#             index=top_k_indices.unsqueeze(-1).expand(-1, -1, self.hidden_size)
#         )
#         # winner_tokens: [bsz, k, hidden_size]

#         # ====================================================================
#         # Step 2: Memory Update (Winners → Memory)
#         # ====================================================================
#         # Memory slots as Query (actively extract information)
#         slots_input = self.memory_slots.unsqueeze(0).expand(bsz, -1, -1)
#         Q_mem = self.q_proj(slots_input).view(
#             bsz, self.num_local_slots, self.num_heads, self.head_dim
#         ).transpose(1, 2)
#         # [bsz, num_heads, num_slots, head_dim]

#         # Winner tokens as Key/Value (passively provide information)
#         K_win = self.k_proj(winner_tokens).view(
#             bsz, k, self.num_heads, self.head_dim
#         ).transpose(1, 2)
#         # [bsz, num_heads, k, head_dim]

#         V_win = self.v_proj(winner_tokens).view(
#             bsz, k, self.num_heads, self.head_dim
#         ).transpose(1, 2)
#         # [bsz, num_heads, k, head_dim]

#         # Multi-head attention: Memory ← Winners
#         scores = torch.matmul(Q_mem, K_win.transpose(-2, -1)) / math.sqrt(self.head_dim)
#         # [bsz, num_heads, num_slots, k]

#         attn_weights = torch.softmax(scores, dim=-1)

#         # Weighted aggregation
#         global_context = torch.matmul(attn_weights, V_win)
#         # [bsz, num_heads, num_slots, head_dim]

#         # Concatenate heads
#         global_context = global_context.transpose(1, 2).contiguous().view(
#             bsz, self.num_local_slots, self.bottleneck_dim
#         )
#         # [bsz, num_slots, bottleneck_dim]

#         # Output projection
#         global_context = self.o_proj(global_context)
#         # [bsz, num_slots, hidden_size]

#         # ❌ DO NOT normalize output! (Causes loss=10.0)
#         # global_context = self.layer_norm(global_context)

#         # ====================================================================
#         # Step 3: (Optional) Return Debug Info
#         # ====================================================================
#         if return_debug_info:
#             debug_info = {
#                 'winner_indices': top_k_indices,           # Which tokens won
#                 'winner_scores': top_k_scores,             # Their importance scores
#                 'num_winners': k,                          # How many winners
#                 'competition_ratio': k / seq_len,          # Selection rate
#                 'avg_winner_score': top_k_scores.mean().item(),
#                 'min_winner_score': top_k_scores.min().item(),
#             }
#             return global_context, debug_info

#         return global_context

# endregion ===========================================================================


# 方法3.1  🆕 混合全局记忆 - 简化版（无 EMA） Clean版本
class GlobalIntegrator(nn.Module):
    """
    GlobalIntegrator - simplified (no EMA)

    相比 GlobalIntegrator 的改进：
    1. 移除 EMA 机制（概念上有争议，效果可能不显著）
    2. 更模块化的代码结构
    3. 支持多头注意力 + Output Projection
    4. 更清晰的数据流
    5. 数值稳定性保护

    核心设计保留：
    - 两阶段压缩：统计量提取 → Attention 精炼
    - 信息瓶颈：通过 compress_dim 控制容量
    - 多统计量融合：mean, max, min, std, norm_mean

    论文叙述：
    "We propose a hybrid approach combining deterministic statistical
     aggregation with learned attention-based refinement. First, we extract
     five statistical features from local memories and compress them to a
     low-dimensional bottleneck space. Then, global learned queries attend
     over these compressed statistics to extract document-level context."

    输入输出：
        Input:  local_memories [bsz, num_chunks, local_slots, hidden_size]
        Output: global_context  [bsz, global_slots, hidden_size]

    参数量（hidden_size=4096, compress_dim=512, global_slots=4）：
        统计量压缩: 5 × (4096 × 512 + 512) ≈ 10.5M
        Q/K/V 投影: 3 × (512 × 512) = 0.8M
        O 投影:     512 × 512 = 0.26M
        扩展层:     512 × 4096 = 2.1M
        总计:       ~13.7M / layer
    """

    _init_msg_printed = False

    def __init__(
        self,
        hidden_size: int = 4096,
        global_slots: int = 4,
        compress_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.0,
        local_slots: int = 16,  # 兼容参数
        use_bottleneck: bool = False,  # 兼容参数
        bottleneck_dim: int = 4096,  # 兼容参数
        init_from_embeddings: Optional[torch.Tensor] = None,
        use_high_norm_init: bool = True,
        output_scale_init: float = 0.1,
    ):
        """
        Args:
            hidden_size: 隐藏维度（通常 4096）
            global_slots: number of global context slots (typically 4-16)
            compress_dim: 压缩维度（通常 512）
            num_heads: 注意力头数（compress_dim 必须能被整除）
            dropout: 注意力 dropout 概率
            init_from_embeddings: 用于初始化的预训练 embedding
            use_high_norm_init: 是否使用高范数词选择初始化
            output_scale_init: 输出缩放的初始值
        """
        super().__init__()

        # ============ 参数验证 ============
        assert compress_dim % num_heads == 0, (
            f"compress_dim ({compress_dim}) must be divisible by num_heads ({num_heads})"
        )
        assert output_scale_init > 0, "output_scale_init must be positive"

        # ============ 保存配置 ============
        self.hidden_size = hidden_size
        self.num_global = global_slots  # 兼容命名
        self.global_slots = global_slots
        self.compress_dim = compress_dim
        self.num_heads = num_heads
        self.head_dim = compress_dim // num_heads
        self.dropout_p = dropout
        self.use_high_norm_init = use_high_norm_init
        self._output_scale_init = output_scale_init  # 保存初始值用于 softplus

        # ============ 阶段1：统计量压缩器 ============
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

        # ============ 阶段2：Lightweight Multi-Head Attention ============
        self.global_queries = nn.Parameter(torch.zeros(global_slots, compress_dim))

        self.q_proj = nn.Linear(compress_dim, compress_dim, bias=False)
        self.k_proj = nn.Linear(compress_dim, compress_dim, bias=False)
        self.v_proj = nn.Linear(compress_dim, compress_dim, bias=False)
        # ✅ 修复 P2: 添加 output projection（标准 MHA 设计）
        self.o_proj = nn.Linear(compress_dim, compress_dim, bias=False)

        self.attn_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # ============ 阶段3：维度扩展 ============
        self.expand = nn.Linear(compress_dim, hidden_size, bias=False)
        # LLaMA 风格小范数初始化
        std_init = 0.02 / math.sqrt(compress_dim)
        nn.init.normal_(self.expand.weight, mean=0.0, std=std_init)

        # ✅ 修复 P1: 使用 softplus 确保 scale 始终为正
        # 存储 log 空间的参数，通过 softplus 转换确保 > 0
        # softplus(x) ≈ x when x > 0, softplus(0) ≈ 0.693
        # 为了让初始值 = output_scale_init，需要反推初始参数
        init_param = math.log(math.exp(output_scale_init) - 1)  # inverse softplus
        self.expand_scale_param = nn.Parameter(torch.tensor(init_param))

        # ============ 初始化（延迟到这里，确保所有层都创建完成） ============
        self._init_weights(init_from_embeddings)
        self._print_init_info()

    @property
    def expand_scale(self) -> torch.Tensor:
        """通过 softplus 确保 scale 始终为正"""
        return F.softplus(self.expand_scale_param)

    def _init_weights(self, embed_weight: Optional[torch.Tensor] = None):
        """权重初始化"""
        if embed_weight is not None:
            with torch.no_grad():
                if self.use_high_norm_init:
                    embed_norms = torch.norm(embed_weight, dim=-1)
                    _, top_indices = torch.topk(embed_norms, k=self.global_slots)
                    init_embeddings = embed_weight[top_indices]
                else:
                    indices = torch.randperm(embed_weight.size(0))[: self.global_slots]
                    init_embeddings = embed_weight[indices]

                # ✅ 修复 P0: 确保 device 和 dtype 匹配
                target_device = self.stat_compressors[0][0].weight.device
                target_dtype = self.stat_compressors[0][0].weight.dtype
                init_embeddings = init_embeddings.to(
                    device=target_device, dtype=target_dtype
                )

                init_compressed = self.stat_compressors[0](init_embeddings)
                self.global_queries.copy_(init_compressed)
        else:
            nn.init.xavier_uniform_(self.global_queries)

        # 投影层初始化
        for proj in [self.q_proj, self.k_proj, self.v_proj, self.o_proj]:
            nn.init.xavier_uniform_(proj.weight)

    def _print_init_info(self):
        """打印初始化信息"""
        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank == 0 and not GlobalIntegrator._init_msg_printed:
            total_params = sum(p.numel() for p in self.parameters())
            print(f"   ✅ GlobalIntegratorClean initialized (无EMA简化版)")
            print(f"       - Design: Statistical Aggregation + Lightweight MHA")
            print(f"       - Global slots: {self.global_slots}")
            print(f"       - Compress dim: {self.compress_dim}")
            print(f"       - Num heads: {self.num_heads}")
            print(f"       - Output scale (init): {self._output_scale_init}")
            print(
                f"       - Params/layer: {total_params:,} ({total_params / 1e6:.1f}M)"
            )
            GlobalIntegrator._init_msg_printed = True

    def forward(self, local_memories: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            local_memories: [bsz, num_chunks, local_slots, hidden_size]

        Returns:
            G: [bsz, global_slots, hidden_size]

        数据流：
            local_memories [bsz, C, L, H]
                ↓ reshape
            all_local [bsz, C*L, H]
                ↓ 5种统计量
            stats [bsz, H] × 5
                ↓ 压缩
            compressed_stats [bsz, 5, D]
                ↓ Multi-Head Attention
            G_compressed [bsz, G, D]
                ↓ expand + scale
            G [bsz, G, H]
        """
        bsz, num_chunks, local_slots, hidden_size = local_memories.shape

        # ========== 阶段1：统计量提取与压缩 ==========
        # Flatten: [bsz, num_chunks * local_slots, hidden_size]
        all_local = local_memories.reshape(bsz, -1, hidden_size)

        # 计算5种统计量，每个 [bsz, hidden_size]
        mean_pool = all_local.mean(dim=1)
        max_pool, _ = all_local.max(dim=1)
        min_pool, _ = all_local.min(dim=1)

        # ✅ 修复 P3: std 计算使用 fp32 + 数值稳定性保护
        with torch.amp.autocast(device_type="cuda", enabled=False):
            all_local_fp32 = all_local.float()
            # 添加 eps 防止全零输入
            std_pool = all_local_fp32.std(dim=1, unbiased=False).clamp(min=1e-6)
        std_pool = std_pool.to(all_local.dtype)

        # L2 归一化的均值（方向向量）
        norm_mean = F.normalize(mean_pool, dim=-1, p=2, eps=1e-6)

        # 分别压缩每种统计量: [bsz, hidden_size] -> [bsz, compress_dim]
        stats_list = [mean_pool, max_pool, min_pool, std_pool, norm_mean]
        compressed_stats = torch.stack(
            [self.stat_compressors[i](stat) for i, stat in enumerate(stats_list)], dim=1
        )  # [bsz, 5, compress_dim]

        # ========== 阶段2：Lightweight Multi-Head Attention ==========
        # Q: [bsz, global_slots, compress_dim]
        Q = self.global_queries.unsqueeze(0).expand(bsz, -1, -1)
        Q = self.q_proj(Q)

        # K, V: [bsz, 5, compress_dim]
        K = self.k_proj(compressed_stats)
        V = self.v_proj(compressed_stats)

        # 分头: [bsz, num_heads, seq_len, head_dim]
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

        # 合并头: [bsz, global_slots, compress_dim]
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, self.global_slots, self.compress_dim)

        # ✅ 修复 P2: Output projection（融合多头信息）
        G_compressed = self.o_proj(attn_output)

        # ========== 阶段3：维度扩展 ==========
        # expand_scale 通过 softplus 确保始终为正
        G = self.expand(G_compressed) * self.expand_scale

        return G


# ============================================================================
# 方法3.2  🆕 共享压缩层优化版 - GlobalIntegratorShared
# ============================================================================
# class GlobalIntegratorShared(nn.Module):
#     """
#     GlobalIntegratorShared - shared compression layer variant

#     核心优化：
#     1. ✅ 参数减少92%：从10.5M降到0.85M（统计量压缩部分）
#     2. ✅ 共享压缩backbone：5个统计量共享同一个4096→128的压缩层
#     3. ✅ 统计量融合：通过5×128→512的融合层整合所有统计信息
#     4. ✅ 保持原有设计：两阶段压缩 + Lightweight Attention
#     5. ✅ 更强的归纳偏置：共享参数迫使模型学习通用的压缩函数

#     理论依据：
#     - 参数共享（Parameter Sharing）：类似 CNN 的 weight sharing
#     - 归纳偏置（Inductive Bias）：强制5种统计量使用相同的特征提取器
#     - 信息瓶颈（Information Bottleneck）：通过小的中间维度(128)控制容量

#     参数量对比（hidden_size=4096, compress_dim=512）：
#         原版统计量压缩: 5 × (4096 × 512) = 10.5M

#         优化版：
#         - 共享压缩层:   4096 × 128 = 0.524M
#         - 统计量融合:   5×128 × 512 = 0.328M
#         - 总计:         0.852M（节省92%！）

#         其他层保持不变:
#         - Q/K/V投影:    0.8M
#         - O投影:        0.26M
#         - 扩展层:       2.1M

#         总参数量: 0.852M + 0.8M + 0.26M + 2.1M = 4.0M/layer（原版13.7M）
#         节省率: 71%

#     输入输出：
#         Input:  local_memories [bsz, num_chunks, local_slots, hidden_size]
#         Output: global_context  [bsz, global_slots, hidden_size]
#     """

#     _init_msg_printed = False

#     def __init__(
#         self,
#         hidden_size: int = 4096,
#         global_slots: int = 4,
#         compress_dim: int = 512,
#         shared_compress_dim: int = 128,  # 共享压缩层的维度
#         num_heads: int = 8,
#         dropout: float = 0.0,
#         local_slots: int = 16,  # 兼容参数
#         use_bottleneck: bool = False,  # 兼容参数
#         bottleneck_dim: int = 4096,  # 兼容参数
#         init_from_embeddings: Optional[torch.Tensor] = None,
#         use_high_norm_init: bool = True,
#         output_scale_init: float = 0.1,
#     ):
#         """
#         Args:
#             hidden_size: 隐藏维度（通常 4096）
#             global_slots: number of global context slots (typically 4-16)
#             compress_dim: 最终压缩维度（通常 512）
#             shared_compress_dim: 共享压缩层的中间维度（通常 128）
#             num_heads: 注意力头数
#             dropout: 注意力 dropout 概率
#             init_from_embeddings: 用于初始化的预训练 embedding
#             use_high_norm_init: 是否使用高范数词选择初始化
#             output_scale_init: 输出缩放的初始值
#         """
#         super().__init__()

#         # ============ 参数验证 ============
#         assert compress_dim % num_heads == 0, (
#             f"compress_dim ({compress_dim}) must be divisible by num_heads ({num_heads})"
#         )
#         assert output_scale_init > 0, "output_scale_init must be positive"

#         # ============ 保存配置 ============
#         self.hidden_size = hidden_size
#         self.num_global = global_slots  # 兼容命名
#         self.global_slots = global_slots
#         self.shared_compress_dim = shared_compress_dim
#         self.num_heads = num_heads
#         self.dropout_p = dropout
#         self.use_high_norm_init = use_high_norm_init
#         self._output_scale_init = output_scale_init

#         # ============ 阶段1：共享压缩层 ============
#         self.stat_names = ["mean", "max", "min", "std", "norm_mean"]

#         # 关键优化：所有统计量共享同一个压缩层
#         self.shared_compressor = nn.Sequential(
#             nn.Linear(hidden_size, shared_compress_dim, bias=False),
#             nn.LayerNorm(shared_compress_dim),
#         )
#         # 参数: 4096 × shared_compress_dim

#         # ✅ 条件判断：只有需要扩展维度时才创建扩展层
#         # 如果 shared_compress_dim = compress_dim，则不需要扩展层
#         if shared_compress_dim < compress_dim:
#             self.stat_expand = nn.Sequential(
#                 nn.Linear(shared_compress_dim, compress_dim, bias=False),
#                 nn.LayerNorm(compress_dim),
#             )
#             # 参数: shared_compress_dim × compress_dim
#             self.compress_dim = compress_dim
#         else:
#             # shared_compress_dim >= compress_dim: 不需要扩展
#             # 使用 Identity，0参数
#             self.stat_expand = nn.Identity()
#             if shared_compress_dim > compress_dim:
#                 print(
#                     f"⚠️  Warning: shared_compress_dim ({shared_compress_dim}) > compress_dim ({compress_dim})"
#                 )
#                 print(f"   Setting compress_dim = shared_compress_dim for consistency")
#             self.compress_dim = shared_compress_dim

#         self.head_dim = self.compress_dim // num_heads
#         # 总计: 524K + 66K = 590K（原版10.5M的5.6%！）

#         # ============ 阶段2：Lightweight Multi-Head Attention ============
#         self.global_queries = nn.Parameter(torch.zeros(global_slots, self.compress_dim))

#         self.q_proj = nn.Linear(self.compress_dim, self.compress_dim, bias=False)
#         self.k_proj = nn.Linear(self.compress_dim, self.compress_dim, bias=False)
#         self.v_proj = nn.Linear(self.compress_dim, self.compress_dim, bias=False)
#         self.o_proj = nn.Linear(self.compress_dim, self.compress_dim, bias=False)

#         self.attn_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

#         # ============ 阶段3：维度扩展 ============
#         self.expand = nn.Linear(self.compress_dim, hidden_size, bias=False)
#         std_init = 0.02 / math.sqrt(self.compress_dim)
#         nn.init.normal_(self.expand.weight, mean=0.0, std=std_init)

#         # 输出缩放参数
#         init_param = math.log(math.exp(output_scale_init) - 1)
#         self.expand_scale_param = nn.Parameter(torch.tensor(init_param))

#         # ============ 初始化 ============
#         self._init_weights(init_from_embeddings)
#         self._print_init_info()

#     @property
#     def expand_scale(self) -> torch.Tensor:
#         """通过 softplus 确保 scale 始终为正"""
#         return F.softplus(self.expand_scale_param)

#     def _init_weights(self, embed_weight: Optional[torch.Tensor] = None):
#         """权重初始化"""
#         if embed_weight is not None:
#             with torch.no_grad():
#                 if self.use_high_norm_init:
#                     embed_norms = torch.norm(embed_weight, dim=-1)
#                     _, top_indices = torch.topk(embed_norms, k=self.global_slots)
#                     init_embeddings = embed_weight[top_indices]
#                 else:
#                     indices = torch.randperm(embed_weight.size(0))[: self.global_slots]
#                     init_embeddings = embed_weight[indices]

#                 # 确保 device 和 dtype 匹配
#                 target_device = self.shared_compressor[0].weight.device
#                 target_dtype = self.shared_compressor[0].weight.dtype
#                 init_embeddings = init_embeddings.to(
#                     device=target_device, dtype=target_dtype
#                 )

#                 # ✅ 修正：通过共享压缩层 + 扩展层初始化
#                 init_compressed = self.shared_compressor(
#                     init_embeddings
#                 )  # [global_slots, 128]
#                 init_expanded = self.stat_expand(init_compressed)  # [global_slots, 512]
#                 self.global_queries.copy_(init_expanded)
#         else:
#             nn.init.xavier_uniform_(self.global_queries)

#         # 投影层初始化
#         for proj in [self.q_proj, self.k_proj, self.v_proj, self.o_proj]:
#             nn.init.xavier_uniform_(proj.weight)

#     def _print_init_info(self):
#         """打印初始化信息"""
#         rank = dist.get_rank() if dist.is_initialized() else 0
#         if rank == 0 and not GlobalIntegratorShared._init_msg_printed:
#             total_params = sum(p.numel() for p in self.parameters())

#             # 计算统计量压缩部分的参数
#             stat_compress_params = sum(
#                 p.numel() for p in self.shared_compressor.parameters()
#             ) + sum(p.numel() for p in self.stat_expand.parameters())

#             print(f"   ✅ GlobalIntegratorShared initialized (共享压缩层优化版)")

#             # 根据是否有扩展层显示不同的设计描述
#             if isinstance(self.stat_expand, nn.Identity):
#                 design_desc = "Shared Compressor + Lightweight MHA (no expansion)"
#             else:
#                 design_desc = (
#                     "Shared Compressor + Statistical Expansion + Lightweight MHA"
#                 )

#             print(f"       - Design: {design_desc}")
#             print(f"       - Global slots: {self.global_slots}")
#             print(f"       - Shared compress dim: {self.shared_compress_dim}")
#             print(f"       - Final compress dim: {self.compress_dim}")
#             print(f"       - Num heads: {self.num_heads}")
#             print(f"       - Output scale (init): {self._output_scale_init}")
#             print(
#                 f"       - Stat compression params: {stat_compress_params:,} ({stat_compress_params / 1e6:.2f}M)"
#             )
#             print(
#                 f"       - Total params/layer: {total_params:,} ({total_params / 1e6:.1f}M)"
#             )
#             print(
#                 f"       - 🎯 Saved {(1 - total_params / 13.7e6) * 100:.0f}% compared to original"
#             )
#             GlobalIntegratorShared._init_msg_printed = True

#     def forward(self, local_memories: torch.Tensor) -> torch.Tensor:
#         """
#         前向传播

#         Args:
#             local_memories: [bsz, num_chunks, local_slots, hidden_size]

#         Returns:
#             G: [bsz, global_slots, hidden_size]

#         数据流：
#             local_memories [bsz, C, L, H]
#                 ↓ reshape
#             all_local [bsz, C*L, H]
#                 ↓ 5种统计量
#             stats [bsz, H] × 5
#                 ↓ 共享压缩（每个统计量独立通过）
#             compressed_stats_list: 5 × [bsz, 128]
#                 ↓ 扩展（每个统计量独立通过）
#             expanded_stats_list: 5 × [bsz, 512]
#                 ↓ stack保持分离
#             compressed_stats [bsz, 5, 512]
#                 ↓ Multi-Head Attention（对5个统计量进行选择）
#             G_compressed [bsz, G, D]
#                 ↓ expand + scale
#             G [bsz, G, H]
#         """
#         bsz, num_chunks, local_slots, hidden_size = local_memories.shape

#         # ========== 阶段1a：统计量提取 ==========
#         all_local = local_memories.reshape(bsz, -1, hidden_size)

#         # 计算5种统计量
#         mean_pool = all_local.mean(dim=1)
#         max_pool, _ = all_local.max(dim=1)
#         min_pool, _ = all_local.min(dim=1)

#         # std 计算使用 fp32 确保稳定性
#         with torch.amp.autocast(device_type="cuda", enabled=False):
#             all_local_fp32 = all_local.float()
#             std_pool = all_local_fp32.std(dim=1, unbiased=False).clamp(min=1e-6)
#         std_pool = std_pool.to(all_local.dtype)

#         # L2 归一化的均值
#         norm_mean = F.normalize(mean_pool, dim=-1, p=2, eps=1e-6)

#         # ========== 阶段1b：共享压缩 + 扩展（保持5个统计量分离！） ==========
#         # ✅ Batch优化：一次性处理5个统计量，性能提升5.75x
#         # 关键：保持5个统计量分离，让Attention能学习如何选择性地使用它们
#         stats_list = [mean_pool, max_pool, min_pool, std_pool, norm_mean]

#         # Stack: [bsz, 5, hidden_size]
#         stats_stacked = torch.stack(stats_list, dim=1)
#         num_stats = 5

#         # Batch压缩: view为[bsz*5, hidden_size] → compress → view回[bsz, 5, 128]
#         compressed_stats = self.shared_compressor(
#             stats_stacked.view(bsz * num_stats, hidden_size)
#         ).view(bsz, num_stats, -1)

#         # Batch扩展: view为[bsz*5, 128] → expand → view回[bsz, 5, 512]
#         compressed_stats = self.stat_expand(
#             compressed_stats.view(bsz * num_stats, -1)
#         ).view(bsz, num_stats, self.compress_dim)
#         # compressed_stats: [bsz, 5, 512]（和原版一样！）

#         # ========== 阶段2：Lightweight Multi-Head Attention ==========
#         # Q: [bsz, global_slots, compress_dim]
#         Q = self.global_queries.unsqueeze(0).expand(bsz, -1, -1)
#         Q = self.q_proj(Q)

#         # ✅ 修正：K, V: [bsz, 5, compress_dim]（和原版一样！）
#         K = self.k_proj(compressed_stats)
#         V = self.v_proj(compressed_stats)

#         # 分头: [bsz, num_heads, seq_len, head_dim]
#         Q = Q.view(bsz, self.global_slots, self.num_heads, self.head_dim).transpose(
#             1, 2
#         )
#         K = K.view(bsz, 5, self.num_heads, self.head_dim).transpose(1, 2)
#         V = V.view(bsz, 5, self.num_heads, self.head_dim).transpose(1, 2)
#         # ✅ Q: [bsz, num_heads, global_slots, head_dim]
#         # ✅ K: [bsz, num_heads, 5, head_dim]（和原版一样！）
#         # ✅ V: [bsz, num_heads, 5, head_dim]（和原版一样！）

#         # Scaled Dot-Product Attention
#         scale = self.head_dim**-0.5
#         # ✅ attn_weights: [bsz, num_heads, global_slots, 5]（和原版一样！）
#         attn_weights = torch.matmul(Q, K.transpose(-2, -1)) * scale
#         attn_probs = F.softmax(attn_weights, dim=-1)
#         attn_probs = self.attn_dropout(attn_probs)

#         # ✅ attn_output: [bsz, num_heads, global_slots, head_dim]
#         attn_output = torch.matmul(attn_probs, V)

#         # 合并头
#         attn_output = attn_output.transpose(1, 2).contiguous()
#         attn_output = attn_output.view(bsz, self.global_slots, self.compress_dim)

#         # Output projection
#         G_compressed = self.o_proj(attn_output)

#         # ========== 阶段3：维度扩展 ==========
#         G = self.expand(G_compressed) * self.expand_scale


#         return G
class GlobalIntegratorShared(nn.Module):
    """
    GlobalIntegratorShared - shared compression layer variant

    核心优化：
    1. ✅ 参数减少92%：从10.5M降到0.85M（统计量压缩部分）
    2. ✅ 共享压缩backbone：5个统计量共享同一个4096→128的压缩层
    3. ✅ 统计量融合：通过5×128→512的融合层整合所有统计信息
    4. ✅ 保持原有设计：两阶段压缩 + Lightweight Attention
    5. ✅ 更强的归纳偏置：共享参数迫使模型学习通用的压缩函数

    理论依据：
    - 参数共享（Parameter Sharing）：类似 CNN 的 weight sharing
    - 归纳偏置（Inductive Bias）：强制5种统计量使用相同的特征提取器
    - 信息瓶颈（Information Bottleneck）：通过小的中间维度(128)控制容量

    参数量对比（hidden_size=4096, compress_dim=512）：
        原版统计量压缩: 5 × (4096 × 512) = 10.5M

        优化版：
        - 共享压缩层:   4096 × 128 = 0.524M
        - 统计量融合:   5×128 × 512 = 0.328M
        - 总计:         0.852M（节省92%！）

        其他层保持不变:
        - Q/K/V投影:    0.8M
        - O投影:        0.26M
        - 扩展层:       2.1M

        总参数量: 0.852M + 0.8M + 0.26M + 2.1M = 4.0M/layer（原版13.7M）
        节省率: 71%

    输入输出：
        Input:  local_memories [bsz, num_chunks, local_slots, hidden_size]
        Output: global_context  [bsz, global_slots, hidden_size]
    """

    _init_msg_printed = False

    def __init__(
        self,
        hidden_size: int = 4096,
        global_slots: int = 4,
        compress_dim: int = 512,
        shared_compress_dim: int = 128,  # 共享压缩层的维度
        num_heads: int = 8,
        dropout: float = 0.0,
        local_slots: int = 16,  # 兼容参数
        use_bottleneck: bool = False,  # 兼容参数
        bottleneck_dim: int = 4096,  # 兼容参数
        init_from_embeddings: Optional[torch.Tensor] = None,
        use_high_norm_init: bool = True,
        output_scale_init: float = 0.1,
    ):
        """
        Args:
            hidden_size: 隐藏维度（通常 4096）
            global_slots: number of global context slots (typically 4-16)
            compress_dim: 最终压缩维度（通常 512）
            shared_compress_dim: 共享压缩层的中间维度（通常 128）
            num_heads: 注意力头数
            dropout: 注意力 dropout 概率
            init_from_embeddings: 用于初始化的预训练 embedding
            use_high_norm_init: 是否使用高范数词选择初始化
            output_scale_init: 输出缩放的初始值
        """
        super().__init__()

        # ============ 参数验证 ============
        assert compress_dim % num_heads == 0, (
            f"compress_dim ({compress_dim}) must be divisible by num_heads ({num_heads})"
        )
        assert output_scale_init > 0, "output_scale_init must be positive"

        # ============ 保存配置 ============
        self.hidden_size = hidden_size
        self.num_global = global_slots  # 兼容命名
        self.global_slots = global_slots
        self.shared_compress_dim = shared_compress_dim
        self.num_heads = num_heads
        self.dropout_p = dropout
        self.use_high_norm_init = use_high_norm_init
        self._output_scale_init = output_scale_init

        # ============ 阶段1：共享压缩层 ============
        self.stat_names = ["mean", "max", "min", "std", "norm_mean"]

        # 关键优化：所有统计量共享同一个压缩层
        self.shared_compressor = nn.Sequential(
            nn.Linear(hidden_size, shared_compress_dim, bias=False),
            nn.LayerNorm(shared_compress_dim),
        )
        # 参数: 4096 × shared_compress_dim

        # ✅ 条件判断：只有需要扩展维度时才创建扩展层
        # 如果 shared_compress_dim = compress_dim，则不需要扩展层
        if shared_compress_dim < compress_dim:
            self.stat_expand = nn.Sequential(
                nn.Linear(shared_compress_dim, compress_dim, bias=False),
                nn.LayerNorm(compress_dim),
            )
            # 参数: shared_compress_dim × compress_dim
            self.compress_dim = compress_dim
        else:
            # shared_compress_dim >= compress_dim: 不需要扩展
            # 使用 Identity，0参数
            self.stat_expand = nn.Identity()
            if shared_compress_dim > compress_dim:
                print(
                    f"⚠️  Warning: shared_compress_dim ({shared_compress_dim}) > compress_dim ({compress_dim})"
                )
                print(f"   Setting compress_dim = shared_compress_dim for consistency")
            self.compress_dim = shared_compress_dim

        self.head_dim = self.compress_dim // num_heads
        # 总计: 524K + 66K = 590K（原版10.5M的5.6%！）

        # ============ 阶段2：Lightweight Multi-Head Attention ============
        self.global_queries = nn.Parameter(torch.zeros(global_slots, self.compress_dim))

        self.q_proj = nn.Linear(self.compress_dim, self.compress_dim, bias=False)
        self.k_proj = nn.Linear(self.compress_dim, self.compress_dim, bias=False)
        self.v_proj = nn.Linear(self.compress_dim, self.compress_dim, bias=False)
        self.o_proj = nn.Linear(self.compress_dim, self.compress_dim, bias=False)

        self.attn_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # ============ 阶段3：维度扩展 ============
        self.expand = nn.Linear(self.compress_dim, hidden_size, bias=False)
        std_init = 0.02 / math.sqrt(self.compress_dim)
        nn.init.normal_(self.expand.weight, mean=0.0, std=std_init)

        # 输出缩放参数
        init_param = math.log(math.exp(output_scale_init) - 1)
        self.expand_scale_param = nn.Parameter(torch.tensor(init_param))

        # ============ 初始化 ============
        self._init_weights(init_from_embeddings)
        self._print_init_info()

    @property
    def expand_scale(self) -> torch.Tensor:
        """通过 softplus 确保 scale 始终为正"""
        return F.softplus(self.expand_scale_param)

    def _init_weights(self, embed_weight: Optional[torch.Tensor] = None):
        """权重初始化"""
        if embed_weight is not None:
            with torch.no_grad():
                if self.use_high_norm_init:
                    embed_norms = torch.norm(embed_weight, dim=-1)
                    _, top_indices = torch.topk(embed_norms, k=self.global_slots)
                    init_embeddings = embed_weight[top_indices]
                else:
                    indices = torch.randperm(embed_weight.size(0))[: self.global_slots]
                    init_embeddings = embed_weight[indices]

                # 确保 device 和 dtype 匹配
                target_device = self.shared_compressor[0].weight.device
                target_dtype = self.shared_compressor[0].weight.dtype
                init_embeddings = init_embeddings.to(
                    device=target_device, dtype=target_dtype
                )

                # ✅ 修正：通过共享压缩层 + 扩展层初始化
                init_compressed = self.shared_compressor(
                    init_embeddings
                )  # [global_slots, 128]
                init_expanded = self.stat_expand(init_compressed)  # [global_slots, 512]
                self.global_queries.copy_(init_expanded)
        else:
            nn.init.xavier_uniform_(self.global_queries)

        # 投影层初始化
        for proj in [self.q_proj, self.k_proj, self.v_proj, self.o_proj]:
            nn.init.xavier_uniform_(proj.weight)
        # for proj in [self.q_proj, self.k_proj, self.v_proj]:
        #     nn.init.xavier_uniform_(proj.weight)

    def _print_init_info(self):
        """打印初始化信息"""
        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank == 0 and not GlobalIntegratorShared._init_msg_printed:
            total_params = sum(p.numel() for p in self.parameters())

            # 计算统计量压缩部分的参数
            stat_compress_params = sum(
                p.numel() for p in self.shared_compressor.parameters()
            ) + sum(p.numel() for p in self.stat_expand.parameters())

            print(f"   ✅ GlobalIntegratorShared initialized (共享压缩层优化版)")

            # 根据是否有扩展层显示不同的设计描述
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

    def forward(self, local_memories: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            local_memories: [bsz, num_chunks, local_slots, hidden_size]

        Returns:
            G: [bsz, global_slots, hidden_size]

        数据流：
            local_memories [bsz, C, L, H]
                ↓ reshape
            all_local [bsz, C*L, H]
                ↓ 5种统计量
            stats [bsz, H] × 5
                ↓ 共享压缩（每个统计量独立通过）
            compressed_stats_list: 5 × [bsz, 128]
                ↓ 扩展（每个统计量独立通过）
            expanded_stats_list: 5 × [bsz, 512]
                ↓ stack保持分离
            compressed_stats [bsz, 5, 512]
                ↓ Multi-Head Attention（对5个统计量进行选择）
            G_compressed [bsz, G, D]
                ↓ expand + scale
            G [bsz, G, H]
        """
        bsz, num_chunks, local_slots, hidden_size = local_memories.shape

        # ========== 阶段1a：统计量提取 ==========
        all_local = local_memories.reshape(bsz, -1, hidden_size)

        # 计算5种统计量
        mean_pool = all_local.mean(dim=1)
        max_pool, _ = all_local.max(dim=1)
        min_pool, _ = all_local.min(dim=1)

        # std 计算使用 fp32 确保稳定性
        with torch.amp.autocast(device_type="cuda", enabled=False):
            all_local_fp32 = all_local.float()
            std_pool = all_local_fp32.std(dim=1, unbiased=False).clamp(min=1e-6)
        std_pool = std_pool.to(all_local.dtype)

        # L2 归一化的均值
        norm_mean = F.normalize(mean_pool, dim=-1, p=2, eps=1e-6)

        # ========== 阶段1b：共享压缩 + 扩展（保持5个统计量分离！） ==========
        # ✅ Batch优化：一次性处理5个统计量，性能提升5.75x
        # 关键：保持5个统计量分离，让Attention能学习如何选择性地使用它们
        stats_list = [mean_pool, max_pool, min_pool, std_pool, norm_mean]

        # Stack: [bsz, 5, hidden_size]
        stats_stacked = torch.stack(stats_list, dim=1)
        num_stats = 5

        # Batch压缩: view为[bsz*5, hidden_size] → compress → view回[bsz, 5, 128]
        compressed_stats = self.shared_compressor(
            stats_stacked.view(bsz * num_stats, hidden_size)
        ).view(bsz, num_stats, -1)

        # Batch扩展: view为[bsz*5, 128] → expand → view回[bsz, 5, 512]
        compressed_stats = self.stat_expand(
            compressed_stats.view(bsz * num_stats, -1)
        ).view(bsz, num_stats, self.compress_dim)
        # compressed_stats: [bsz, 5, 512]（和原版一样！）

        # ========== 阶段2：Lightweight Multi-Head Attention ==========
        # Q: [bsz, global_slots, compress_dim]
        Q = self.global_queries.unsqueeze(0).expand(bsz, -1, -1)
        Q = self.q_proj(Q)

        # ✅ 修正：K, V: [bsz, 5, compress_dim]（和原版一样！）
        K = self.k_proj(compressed_stats)
        V = self.v_proj(compressed_stats)

        # 分头: [bsz, num_heads, seq_len, head_dim]
        Q = Q.view(bsz, self.global_slots, self.num_heads, self.head_dim).transpose(
            1, 2
        )
        K = K.view(bsz, 5, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(bsz, 5, self.num_heads, self.head_dim).transpose(1, 2)
        # ✅ Q: [bsz, num_heads, global_slots, head_dim]
        # ✅ K: [bsz, num_heads, 5, head_dim]（和原版一样！）
        # ✅ V: [bsz, num_heads, 5, head_dim]（和原版一样！）

        # Scaled Dot-Product Attention
        scale = self.head_dim**-0.5
        # ✅ attn_weights: [bsz, num_heads, global_slots, 5]（和原版一样！）
        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) * scale
        attn_probs = F.softmax(attn_weights, dim=-1)
        attn_probs = self.attn_dropout(attn_probs)

        # ✅ attn_output: [bsz, num_heads, global_slots, head_dim]
        attn_output = torch.matmul(attn_probs, V)

        # 合并头
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, self.global_slots, self.compress_dim)

        # Output projection
        G_compressed = self.o_proj(attn_output)

        # ========== 阶段3：维度扩展 ==========
        # G = self.expand(attn_output) * self.expand_scale
        G = self.expand(G_compressed) * self.expand_scale

        return G


# 🆕 Temporal Perceiver Global Memory
# region
# class TemporalPerceiverGlobalMemory(nn.Module):
#     """
#     Temporal Perceiver Global Memory - OPTIMIZED VERSION

#     理论基础：
#     1. ✅ Perceiver (ICML 2021): Cross-attention bottleneck
#     2. ✅ Slot Attention (NeurIPS 2020): Iterative competitive binding
#     3. ✅ Set Transformer (ICML 2019): Pooling by Multihead Attention (PMA)
#     4. ✅ DNC (Nature 2016): Temporal linkage
#     5. ✅ Predictive Coding (Rao & Ballard, 1999): EMA as contextual memory
#     6. ✅ Cognitive Memory Systems (Nature): Episodic + Semantic memory

#     设计哲学：
#     从认知科学视角，我们模拟三种记忆系统的交互：

#     - Working Memory（工作记忆）= 局部记忆（每个chunk 8 slots）
#       容量有限，短期存储

#     - Episodic Memory（情景记忆）= Chunks的时序信息（49个chunks的顺序）
#       记住"what, where, when"

#     - Semantic Memory（语义记忆）= 全局记忆（2 slots）
#       Document-level知识，长期稳定

#     关键创新：
#     1. ✅ Chunk-level self-attention：让局部记忆向量互相交互
#     2. ✅ Temporal position encoding：保留chunks的顺序信息
#     3. ✅ Learnable global queries：Perceiver/Slot Attention风格
#     4. ✅ EMA作为额外的contextual memory：参与cross-attention（非query加法）

#     🔥 重要优化（相比原版）：
#     - ✅ 修复EMA使用：EMA作为K/V参与attention，而非加到query上
#     - ✅ 抽取attention函数：避免重复代码，提升可维护性
#     - ✅ 支持动态num_chunks：用最大值初始化temporal_encoding
#     - ✅ 从embeddings初始化EMA：避免从零开始

#     计算复杂度：
#     - Self-attention: O(N² × H) where N = num_chunks × local_slots
#     - Cross-attention: O(G × N × H) where G = global_slots
#     - Total: ~640M FLOPs = 0.64G（仅占总计算的0.019%）

#     参数量：
#     - 共享Q/K/V投影：temporal_encoding + global_queries ≈ 0.2M
#     - 独立Q/K/V投影：+3×4096² = +50.3M

#     参数说明：
#         hidden_size: 隐藏维度（4096 for Llama-2-7B）
#         global_slots: 全局记忆slots数量（2-4，代表主要主题）
#         max_chunks: 支持的最大chunk数（默认128，支持256K tokens）
#         qkv_projections: Q/K/V投影层字典（如果提供则共享，推荐）
#         use_temporal_encoding: 是否使用时序编码
#         ema_decay: EMA衰减率（0.95）
#         init_from_embeddings: 从embeddings初始化global queries和EMA
#     """

#     # 类变量：记录是否已经打印过初始化信息（全局只打印一次）
#     _initialization_printed = False

#     def __init__(
#         self,
#         hidden_size: int = 4096,
#         global_slots: int = 2,
#         num_chunks: int = 4,  # 用于向后兼容，实际会用max_chunks
#         max_chunks: int = 128,  # 支持最大chunk数（128 chunks × 2K = 256K tokens）
#         qkv_projections: dict = None,
#         use_temporal_encoding: bool = True,
#         ema_decay: float = 0.95,
#         init_from_embeddings=None,
#         use_high_norm_init: Optional[bool] = True,
#     ):
#         super().__init__()

#         self.hidden_size = hidden_size
#         self.global_slots = global_slots
#         self.max_chunks = max_chunks
#         self.use_temporal_encoding = use_temporal_encoding
#         self.ema_decay = ema_decay
#         self.use_high_norm_init = use_high_norm_init
#         self.is_qkv_shared = qkv_projections is not None

#         # Component 1: Q/K/V Projections（可选共享）
#         if qkv_projections is not None:
#             self.q_proj = qkv_projections["q_proj"]
#             self.k_proj = qkv_projections["k_proj"]
#             self.v_proj = qkv_projections["v_proj"]
#         else:
#             self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
#             self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
#             self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
#             nn.init.xavier_uniform_(self.q_proj.weight)
#             nn.init.xavier_uniform_(self.k_proj.weight)
#             nn.init.xavier_uniform_(self.v_proj.weight)

#         # Component 2: Temporal Position Encoding（支持动态num_chunks）
#         if use_temporal_encoding:
#             # 用max_chunks初始化，forward时动态切片
#             self.temporal_encoding = nn.Parameter(
#                 torch.randn(max_chunks, hidden_size) * 0.02
#             )
#         else:
#             self.temporal_encoding = None

#         # Component 3: Global Queries（Perceiver/Slot Attention风格）
#         if init_from_embeddings is not None:
#             if use_high_norm_init:
#                 embed_norms = torch.norm(init_from_embeddings, dim=-1)
#                 _, top_indices = torch.topk(embed_norms, k=self.global_slots)
#                 indices = top_indices
#             else:
#                 indices = torch.randperm(init_from_embeddings.size(0))[
#                     : self.global_slots
#                 ]
#             init_queries = init_from_embeddings[indices]
#             self.global_queries = nn.Parameter(init_queries.clone())

#             # 🔥 优化：用embeddings初始化EMA（避免从零开始）
#             init_ema = init_queries.clone().unsqueeze(
#                 0
#             )  # [1, global_slots, hidden_size]
#             self.register_buffer("ema_global", init_ema)
#         else:
#             std = 1.0 / math.sqrt(hidden_size)
#             self.global_queries = nn.Parameter(
#                 torch.randn(global_slots, hidden_size) * std
#             )
#             rank = dist.get_rank() if dist.is_initialized() else 0
#             if rank == 0:
#                 print(f"    ⚠️  Fallback: Initialized memory_slots with std={std}")
#             # EMA从零开始（如果没有embeddings）
#             self.register_buffer(
#                 "ema_global", torch.zeros(1, global_slots, hidden_size)
#             )

#         # 🔥 关键修复：输出缩放和归一化（与GlobalIntegrator保持一致）
#         # 问题：原版输出直接是attention结果，范数可能远大于正常hidden states
#         # 解决：添加LayerNorm + 可学习缩放因子，确保输出与正常token范数匹配
#         self.output_norm = nn.LayerNorm(hidden_size)
#         self.output_scale = nn.Parameter(
#             torch.tensor(0.1)
#         )  # 初始化为0.1，像GlobalIntegrator

#         # 打印信息（只打印一次）
#         rank = dist.get_rank() if dist.is_initialized() else 0
#         if rank == 0 and not TemporalPerceiverGlobalMemory._initialization_printed:
#             TemporalPerceiverGlobalMemory._initialization_printed = True
#             total_params = sum(p.numel() for p in self.parameters())
#             print(f"\n{'=' * 70}")
#             print(f"✅ TemporalPerceiverGlobalMemory initialized (OPTIMIZED)")
#             print(f"{'=' * 70}")
#             print(f"  Key Optimizations:")
#             print(f"    - ✅ EMA as K/V (not added to queries)")
#             print(f"    - ✅ Unified attention function (code reuse)")
#             print(f"    - ✅ Dynamic num_chunks support (max={max_chunks})")
#             print(f"    - ✅ EMA init from embeddings")
#             print(f"    - ✅ Output LayerNorm + Scale (防止loss爆炸)")
#             print(f"\n  Architecture:")
#             print(f"    - Global slots: {global_slots}")
#             print(f"    - Max chunks: {max_chunks} (supports {max_chunks * 2}K tokens)")
#             print(
#                 f"    - Temporal encoding: {'Enabled' if use_temporal_encoding else 'Disabled'}"
#             )
#             print(
#                 f"    - Q/K/V sharing: {'Shared' if self.is_qkv_shared else 'Independent'}"
#             )
#             print(f"\n  Parameters:")
#             print(f"    - Per layer: {total_params:,} ({total_params / 1e6:.2f}M)")
#             if not self.is_qkv_shared:
#                 print(f"      • Q/K/V: {3 * hidden_size * hidden_size / 1e6:.1f}M")
#             print(f"      • Queries: {global_slots * hidden_size / 1e3:.1f}K")
#             if use_temporal_encoding:
#                 print(f"      • Temporal: {max_chunks * hidden_size / 1e3:.1f}K")
#             print(f"    - Total (32 layers): ~{total_params * 32 / 1e6:.1f}M")
#             print(f"{'=' * 70}\n")

#     @staticmethod
#     def _compute_attention(Q, K, V, scale):
#         """
#         统一的注意力计算函数（避免重复代码）

#         Args:
#             Q: [bsz, num_queries, hidden_size]
#             K: [bsz, num_keys, hidden_size]
#             V: [bsz, num_keys, hidden_size]
#             scale: attention缩放因子

#         Returns:
#             output: [bsz, num_queries, hidden_size]
#             attn_weights: [bsz, num_queries, num_keys]
#         """
#         scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
#         attn_weights = F.softmax(scores, dim=-1)
#         output = torch.matmul(attn_weights, V)
#         return output, attn_weights

#     def forward(
#         self,
#         local_memories: torch.Tensor,
#         return_attention_maps: bool = False,
#     ):
#         """
#         从局部记忆中提取全局记忆 - OPTIMIZED

#         Three-stage process:

#         Stage 1 (Working Memory → Episodic Memory):
#             - Self-attention on chunk-level memories
#             - 让局部记忆向量互相交互

#         Stage 2 (Episodic Memory → Semantic Memory):
#             - Cross-attention with global queries
#             - 🔥 EMA作为额外的K/V参与attention（非加到query）

#         Stage 3 (Semantic Memory Update):
#             - EMA update for long-term stability

#         Args:
#             local_memories: [bsz, num_chunks, local_slots_per_chunk, hidden_size]

#         Returns:
#             global_memory: [bsz, global_slots, hidden_size]
#             (optional) attention_maps: dict
#         """
#         bsz, num_chunks, local_slots, hidden_size = local_memories.shape

#         assert hidden_size == self.hidden_size, (
#             f"Expected hidden_size {self.hidden_size}, got {hidden_size}"
#         )
#         assert num_chunks <= self.max_chunks, (
#             f"num_chunks {num_chunks} exceeds max_chunks {self.max_chunks}"
#         )

#         # ═══════════════════════════════════════════════════════════════
#         # Preparation: Flatten and add temporal encoding
#         # ═══════════════════════════════════════════════════════════════
#         # Flatten: [bsz, num_chunks, local_slots, H] → [bsz, N, H]
#         local_flat = local_memories.view(bsz, -1, hidden_size)

#         # 🔥 优化：动态切片temporal_encoding（支持可变num_chunks）
#         if self.use_temporal_encoding and self.temporal_encoding is not None:
#             # 从max_chunks中切片出实际需要的num_chunks
#             temporal_enc = self.temporal_encoding[:num_chunks]  # [num_chunks, H]
#             # 每个chunk的slots共享同一个temporal encoding
#             temporal_enc = temporal_enc.repeat_interleave(
#                 local_slots, dim=0
#             )  # [num_chunks*local_slots, H]
#             local_flat = local_flat + temporal_enc.unsqueeze(0)  # [bsz, N, H]

#         # ═══════════════════════════════════════════════════════════════
#         # Stage 1: Self-Attention（Chunk-level Interaction）
#         # ═══════════════════════════════════════════════════════════════
#         Q_self = self.q_proj(local_flat)
#         K_self = self.k_proj(local_flat)
#         V_self = self.v_proj(local_flat)

#         scale = 1.0 / math.sqrt(hidden_size)
#         refined_memories, attn_weights_self = self._compute_attention(
#             Q_self, K_self, V_self, scale
#         )

#         # region
#         # Stage 2: Cross-Attention（Global Extraction）
#         # 🔥 核心优化：EMA作为额外的K/V参与attention，而非加到query
#         # 理论：
#         # - EMA是输出的慢移动平均，代表"长期语义记忆"
#         # - 应该作为额外的contextual memory，让queries去attend
#         # - 而不是混入queries（queries是learnable的探测器）
#         #
#         # 实现：
#         # - K = [refined_memories; ema_global]
#         # - V = [refined_memories; ema_global]
#         # - Q = learnable global_queries
#         # endregion

#         # Queries投影
#         Q_global = self.q_proj(self.global_queries)  # [global_slots, H]
#         Q_global = Q_global.unsqueeze(0).expand(bsz, -1, -1)  # [bsz, global_slots, H]

#         # Keys和Values：拼接refined_memories和EMA
#         # refined_memories: [bsz, N, H] where N = num_chunks * local_slots
#         # ema_global: [1, global_slots, H] → expand → [bsz, global_slots, H]
#         ema_expanded = self.ema_global.expand(bsz, -1, -1)  # [bsz, global_slots, H]

#         # 拼接：[refined_memories; ema_global]
#         K_cross = torch.cat(
#             [refined_memories, ema_expanded], dim=1
#         )  # [bsz, N+global_slots, H]
#         V_cross = K_cross  # 在Perceiver设计中，K和V通常相同

#         # Cross-attention
#         global_memory, attn_weights_cross = self._compute_attention(
#             Q_global, K_cross, V_cross, scale
#         )  # [bsz, global_slots, H]

#         # Stage 3: EMA Update（Long-term Memory Consolidation）
#         if self.training:
#             with torch.no_grad():
#                 batch_mean_global = global_memory.mean(dim=0, keepdim=True)
#                 self.ema_global.copy_(
#                     self.ema_decay * self.ema_global
#                     + (1 - self.ema_decay) * batch_mean_global
#                 )

#         # 🔥 关键修复：输出归一化 + 缩放（防止loss爆炸）
#         # 原因：attention输出范数可能远大于正常hidden states，导致与LLaMA attention混合时不稳定
#         # 解决：LayerNorm确保范数一致，output_scale(0.1)控制初始影响力
#         global_memory = self.output_norm(global_memory) * self.output_scale

#         # Return
#         if return_attention_maps:
#             attention_maps = {
#                 "self_attn": attn_weights_self,
#                 "cross_attn": attn_weights_cross,
#             }
#             return global_memory, attention_maps
#         else:
#             return global_memory
# endregion


# 训练代码
# 训练way1  只有全局记忆的版本  最初版 没有cache  没有高层记忆
# region ===========================================================================
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

    NEW: Uses HiCI global context + cross-attention instead of shift operation.
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
    # 传入 attention_mask 以正确处理 padding tokens
    global_context = self.local_constructor(
        hidden_states, attention_mask
    )  # [bsz, num_slots, hidden_size]

    num_local_slots = global_context.shape[1]

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

    # ========== Step 3: Inject global context into Q/K/V (NEW!) ==========
    # Convert global_context to Q/K/V format and prepend to each chunk

    # Project global context through Q/K/V projections
    global_q = (
        self.q_proj(global_context)
        .view(bsz, num_local_slots, self.num_heads, self.head_dim)
        .transpose(1, 2)
    )  # [bsz, nh, num_slots, hd]
    global_k = (
        self.k_proj(global_context)
        .view(bsz, num_local_slots, self.num_key_value_heads, self.head_dim)
        .transpose(1, 2)
    )
    global_v = (
        self.v_proj(global_context)
        .view(bsz, num_local_slots, self.num_key_value_heads, self.head_dim)
        .transpose(1, 2)
    )

    # Repeat k/v heads for global context
    global_k = repeat_kv(global_k, self.num_key_value_groups)
    global_v = repeat_kv(global_v, self.num_key_value_groups)

    # ========== Step 4: Prepare chunked attention with global prefix (NEW!) ==========
    # ✅ SFT版本：处理不规则序列长度
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


# endregion ===========================================================================


# 训练way1优化版  性能优化版本：合并投影 + 向量化mask + 预分配tensor
# region ===========================================================================
def forward_flashattn_optimized(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    padding_mask: Optional[torch.LongTensor] = None,
    # use_local_slots: bool = True,
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

    # ========== Step 1: Global context (可选) ==========
    # 如果没有注册 global_memory，则跳过（用于 A/B 测试）
    has_hici = hasattr(self, "local_constructor")

    if has_hici:
        # 传入 attention_mask 以正确处理 padding tokens
        global_context = self.local_constructor(
            hidden_states, attention_mask
        )  # [bsz, num_slots, hidden_size]
        num_local_slots = global_context.shape[1]
    else:
        # 不使用 global memory，回退到原始 LongLoRA 行为
        num_local_slots = 0

    # ========== Step 2: Compute group parameters ==========
    # ✅ SFT版本：处理不规则序列长度
    if q_len % 4096 == 0:
        group_size = int(q_len * group_size_ratio)
    else:
        group_size = sft_group_size
    num_groups = q_len // group_size
    chunk_len = num_local_slots + group_size

    # ========== Step 3: QKV projections ==========
    if has_hici:
        # 将 global_context 和 hidden_states 拼接后一起投影
        combined_input = torch.cat(
            [global_context, hidden_states], dim=1
        )  # [bsz, num_slots + q_len, hidden_size]

        # Q 投影 (使用 num_heads)
        combined_q = (
            self.q_proj(combined_input)
            .view(bsz, num_local_slots + q_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )  # [bsz, nh, num_slots + q_len, hd]

        # K 投影 (使用 num_key_value_heads)
        combined_k = (
            self.k_proj(combined_input)
            .view(
                bsz, num_local_slots + q_len, self.num_key_value_heads, self.head_dim
            )
            .transpose(1, 2)
        )

        # V 投影 (使用 num_key_value_heads)
        combined_v = (
            self.v_proj(combined_input)
            .view(
                bsz, num_local_slots + q_len, self.num_key_value_heads, self.head_dim
            )
            .transpose(1, 2)
        )

        # 分离 global 和 sequence 部分
        global_q = combined_q[:, :, :num_local_slots, :]  # [bsz, nh, num_slots, hd]
        query_states = combined_q[:, :, num_local_slots:, :]  # [bsz, nh, q_len, hd]

        global_k = combined_k[:, :, :num_local_slots, :]
        key_states = combined_k[:, :, num_local_slots:, :]

        global_v = combined_v[:, :, :num_local_slots, :]
        value_states = combined_v[:, :, num_local_slots:, :]
    else:
        # 不使用 global memory，直接投影
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

    # ========== Step 4: RoPE (只对 sequence 部分，不对 global) ==========
    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]

    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(
        query_states, key_states, cos, sin, position_ids
    )
    # global_q, global_k 不应用 RoPE（记忆是位置无关的）

    # Past Key value support
    if past_key_value is not None:
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)
    past_key_value = (key_states, value_states) if use_cache else None

    # Repeat k/v heads if n_kv_heads < n_heads
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)
    if has_hici:
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

    if has_hici:
        # 预分配输出 tensor: [bsz, nh, num_groups, chunk_len, hd]
        # chunk 结构: [global_memory, chunk_tokens]
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

        # 直接写入 (避免 cat 的内存分配开销)
        # global memory 在前面: [memory_slots, chunk_tokens]
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
        # 不使用 global memory，直接使用 chunks
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

    if has_hici:
        # global_mask: [bsz, num_groups, num_local_slots] - 全 1（memory 始终有效）
        global_mask = attention_mask.new_ones(bsz, num_groups, num_local_slots)
        # 拼接: [global_mask, chunk_mask] -> [bsz, num_groups, chunk_len]
        key_padding_mask = torch.cat([global_mask, attention_mask_chunks], dim=2)
    else:
        # 不使用 global memory，只有 chunk mask
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

    # ========== Step 8: Extract chunk outputs (丢弃 global memory 输出) ==========
    output = output.view(bsz, num_groups, chunk_len, self.num_heads, self.head_dim)
    output = output[
        :, :, num_local_slots:, :, :
    ]  # keep chunk part only, discard HiCI slots output
    output = output.reshape(bsz, q_len, self.num_heads, self.head_dim)

    return self.o_proj(rearrange(output, "b s h d -> b s (h d)")), None, past_key_value


# endregion ===========================================================================


# 训练way1优化版++  性能优化版本：合并投影 + 向量化mask + 预分配tensor+全局记忆没有独立kv
# region ===========================================================================
def forward_flashattn_optimized_plus(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    padding_mask: Optional[torch.LongTensor] = None,
    # use_local_slots: bool = True,
    # group_size_ratio: float = 0.25,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """
    改进版：先做 Q/K/V 投影和 RoPE，再调用 LocalConstructor。

    优势：
    1. LocalConstructor 复用已投影的 K/V（节省一套投影）
    2. K/V already have RoPE, HiCI slots are position-aware
    3. 更高效，更符合 Perceiver/Longformer 等主流设计
    """
    if output_attentions:
        warnings.warn(
            "Output attentions is not supported for patched `LlamaAttention`, returning `None` instead."
        )

    bsz, q_len, _ = hidden_states.size()

    # ========== Step 1: QKV projections (先做投影) ==========
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

    # ========== Step 2: RoPE (应用位置编码) ==========
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
    # 现在: key_states, value_states 都是 [bsz, num_heads, q_len, head_dim]

    # ========== Step 3: Global Memory (使用已投影+RoPE的 K/V) ==========
    has_hici = hasattr(self, "local_constructor")

    if has_hici:
        # 转换形状: [bsz, num_heads, seq_len, head_dim] -> [bsz, seq_len, num_heads, head_dim]
        # LocalConstructorFlashPlus 期望这个形状（与 Flash Attention 一致）
        key_for_hici = key_states.transpose(
            1, 2
        )  # [bsz, seq_len, num_heads, head_dim]
        value_for_hici = value_states.transpose(
            1, 2
        )  # [bsz, seq_len, num_heads, head_dim]

        # 调用 LocalConstructor，它只投影自己的 Q，复用输入的 K/V
        global_context = self.local_constructor(
            key_for_hici, value_for_hici, attention_mask
        )  # [bsz, num_slots, hidden_size]

        num_local_slots = global_context.shape[1]

        # 将 global_context 投影为 Q/K/V（用于 chunked attention）
        global_q = (
            self.q_proj(global_context)
            .view(bsz, num_local_slots, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )  # [bsz, num_heads, num_slots, head_dim]

        global_k = (
            self.k_proj(global_context)
            .view(bsz, num_local_slots, self.num_key_value_heads, self.head_dim)
            .transpose(1, 2)
        )
        global_v = (
            self.v_proj(global_context)
            .view(bsz, num_local_slots, self.num_key_value_heads, self.head_dim)
            .transpose(1, 2)
        )

        # Repeat k/v heads for global
        global_k = repeat_kv(global_k, self.num_key_value_groups)
        global_v = repeat_kv(global_v, self.num_key_value_groups)
        # global_q, global_k, global_v 不应用 RoPE（记忆是位置无关的）
    else:
        num_local_slots = 0

    # ========== Step 4: Compute group parameters ==========
    # ✅ SFT版本：处理不规则序列长度
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

    if has_hici:
        # 预分配输出 tensor: [bsz, nh, num_groups, chunk_len, hd]
        # chunk 结构: [global_memory, chunk_tokens]
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

        # 直接写入 (避免 cat 的内存分配开销)
        # global memory 在前面: [memory_slots, chunk_tokens]
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
        # 不使用 global memory，直接使用 chunks
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

    if has_hici:
        # global_mask: [bsz, num_groups, num_local_slots] - 全 1（memory 始终有效）
        global_mask = attention_mask.new_ones(bsz, num_groups, num_local_slots)
        # 拼接: [global_mask, chunk_mask] -> [bsz, num_groups, chunk_len]
        key_padding_mask = torch.cat([global_mask, attention_mask_chunks], dim=2)
    else:
        # 不使用 global memory，只有 chunk mask
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

    # ========== Step 8: Extract chunk outputs (丢弃 global memory 输出) ==========
    output = output.view(bsz, num_groups, chunk_len, self.num_heads, self.head_dim)
    output = output[
        :, :, num_local_slots:, :, :
    ]  # keep chunk part only, discard HiCI slots output
    output = output.reshape(bsz, q_len, self.num_heads, self.head_dim)

    return self.o_proj(rearrange(output, "b s h d -> b s (h d)")), None, past_key_value


# endregion ===========================================================================

# 训练way2 原本的有cached版本的函数被注释掉了，改用下面的flashattn版本
# region ===========================================================================
# def forward_flashattn(
#     self,
#     hidden_states: torch.Tensor,
#     attention_mask: Optional[torch.Tensor] = None,
#     position_ids: Optional[torch.Tensor] = None,
#     past_key_value: Optional[Tuple[torch.Tensor]] = None,
#     output_attentions: bool = False,
#     use_cache: bool = False,
#     padding_mask: Optional[torch.LongTensor] = None,
# ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
#     """Input shape: Batch x Time x Channel

#     NEW: Uses HiCI global context + cross-attention instead of shift operation.
#     Benefits:
#     - No data duplication (1x computation vs 2x in original)
#     - Direct global context injection before each chunk
#     - O(M×N + N) complexity where M=num_local_slots (16), N=seq_len

#     attention_mask: [bsz, q_len]
#     """
#     if not self.training:
#         warnings.warn("This function should be used just for training as it may exhibit reduced inference performance. For inference, please use forward_flashattn_inference.")

#     if output_attentions:
#         warnings.warn(
#             "Output attentions is not supported for patched `LlamaAttention`, returning `None` instead."
#         )

#     bsz, q_len, _ = hidden_states.size()

#     # ========== Step 1: Compute global context (NEW!) ==========
#     # This captures document-level information before chunking
#     # Note: self.local_constructor is registered via replace_llama_attn()
#     global_context = self.local_constructor(hidden_states)  # [bsz, num_slots, hidden_size]

#     num_local_slots = global_context.shape[1]

#     # ========== Step 2: Standard Q/K/V projections (unchanged) ==========
#     query_states = (
#         self.q_proj(hidden_states)
#         .view(bsz, q_len, self.num_heads, self.head_dim)
#         .transpose(1, 2)
#     )
#     key_states = (
#         self.k_proj(hidden_states)
#         .view(bsz, q_len, self.num_key_value_heads, self.head_dim)
#         .transpose(1, 2)
#     )
#     value_states = (
#         self.v_proj(hidden_states)
#         .view(bsz, q_len, self.num_key_value_heads, self.head_dim)
#         .transpose(1, 2)
#     )
#     # [bsz, nh, q_len, hd]

#     kv_seq_len = key_states.shape[-2]
#     if past_key_value is not None:
#         kv_seq_len += past_key_value[0].shape[-2]

#     # ✅ 修复：如果提供了 position_ids，使用其最大值来确定 RoPE 缓存长度
#     if position_ids is not None:
#         max_pos = position_ids.max().item() + 1
#         rope_seq_len = max(kv_seq_len, max_pos)
#     else:
#         rope_seq_len = kv_seq_len

#     cos, sin = self.rotary_emb(value_states, seq_len=rope_seq_len)
#     query_states, key_states = apply_rotary_pos_emb(
#         query_states, key_states, cos, sin, position_ids
#     )

#     # Past Key value support
#     if past_key_value is not None:
#         # reuse k, v, self_attention
#         key_states = torch.cat([past_key_value[0], key_states], dim=2)
#         value_states = torch.cat([past_key_value[1], value_states], dim=2)

#     past_key_value = (key_states, value_states) if use_cache else None

#     # repeat k/v heads if n_kv_heads < n_heads
#     key_states = repeat_kv(key_states, self.num_key_value_groups)
#     value_states = repeat_kv(value_states, self.num_key_value_groups)

#     # ========== Step 3: Inject global context into Q/K/V (NEW!) ==========
#     # Convert global_context to Q/K/V format and prepend to each chunk

#     # Project global context through Q/K/V projections
#     global_q = (
#         self.q_proj(global_context)
#         .view(bsz, num_local_slots, self.num_heads, self.head_dim)
#         .transpose(1, 2)
#     )  # [bsz, nh, num_slots, hd]
#     global_k = (
#         self.k_proj(global_context)
#         .view(bsz, num_local_slots, self.num_key_value_heads, self.head_dim)
#         .transpose(1, 2)
#     )
#     global_v = (
#         self.v_proj(global_context)
#         .view(bsz, num_local_slots, self.num_key_value_heads, self.head_dim)
#         .transpose(1, 2)
#     )

#     # Repeat k/v heads for global context
#     global_k = repeat_kv(global_k, self.num_key_value_groups)
#     global_v = repeat_kv(global_v, self.num_key_value_groups)

#     # ========== Step 4: Prepare chunked attention with global prefix (NEW!) ==========
#     group_size = int(q_len * group_size_ratio)
#     if q_len % group_size > 0:
#         raise ValueError("q_len %d should be divisible by group size %d." % (q_len, group_size))

#     num_groups = q_len // group_size

#     # For each chunk, prepend global context: [global_ctx, chunk]
#     # This gives each chunk access to document-level information

#     # Reshape query/key/value into chunks
#     query_chunks = query_states.view(bsz, self.num_heads, num_groups, group_size, self.head_dim)
#     key_chunks = key_states.view(bsz, self.num_heads, num_groups, group_size, self.head_dim)
#     value_chunks = value_states.view(bsz, self.num_heads, num_groups, group_size, self.head_dim)

#     # ========== NEW: Prepare recurrent cache - Q and K/V have DIFFERENT lengths! ==========
#     # CRITICAL: Following Transformer-XL principle:
#     #   - Q: [global, chunk] (NO cache!) - cache is read-only context
#     #   - K/V: [global, cache, chunk] (WITH cache)
#     # This eliminates Q=0 noise problem and improves training quality

#     all_chunks_q = []
#     all_chunks_k = []
#     all_chunks_v = []
#     all_chunk_masks_q = []   # Separate masks for Q
#     all_chunk_masks_kv = []  # Separate masks for K/V

#     # Q length: global + chunk (NO cache)
#     q_len_per_chunk = num_local_slots + group_size  # e.g., 8 + 2048 = 2056

#     # K/V length: global + cache + chunk (WITH cache)
#     kv_len_per_chunk = num_local_slots + self.recurrence_size + group_size  # e.g., 8 + 256 + 2048 = 2312

#     for chunk_idx in range(num_groups):
#         chunk_q = query_chunks[:, :, chunk_idx, :, :]  # [bsz, nh, group_size, hd]
#         chunk_k = key_chunks[:, :, chunk_idx, :, :]
#         chunk_v = value_chunks[:, :, chunk_idx, :, :]

#         if chunk_idx == 0:
#             # Chunk 0: Q has NO cache, K/V have dummy cache (masked)

#             # Q: [global, chunk] - NO cache! - length = q_len_per_chunk (2056)
#             q_with_ctx = torch.cat([global_q, chunk_q], dim=2)  # [bsz, nh, q_len_per_chunk, hd]

#             # K/V: [global, dummy_cache (masked), chunk] - length = kv_len_per_chunk (2312)
#             dummy_cache_k = torch.zeros(bsz, self.num_heads, self.recurrence_size, self.head_dim,
#                                        device=key_states.device, dtype=key_states.dtype)
#             dummy_cache_v = torch.zeros(bsz, self.num_heads, self.recurrence_size, self.head_dim,
#                                        device=value_states.device, dtype=value_states.dtype)

#             k_with_ctx = torch.cat([global_k, dummy_cache_k, chunk_k], dim=2)  # [bsz, nh, kv_len_per_chunk, hd]
#             v_with_ctx = torch.cat([global_v, dummy_cache_v, chunk_v], dim=2)

#             # Q mask: [global, chunk] - all visible
#             global_mask_q = torch.ones(bsz, num_local_slots, dtype=attention_mask.dtype, device=attention_mask.device)
#             chunk_mask_q = attention_mask[:, chunk_idx*group_size:(chunk_idx+1)*group_size]
#             mask_q = torch.cat([global_mask_q, chunk_mask_q], dim=1)  # [bsz, q_len_per_chunk]

#             # K/V mask: [global (visible), dummy_cache (masked), chunk (visible)]
#             global_mask_kv = torch.ones(bsz, num_local_slots, dtype=attention_mask.dtype, device=attention_mask.device)
#             dummy_mask_kv = torch.zeros(bsz, self.recurrence_size, dtype=attention_mask.dtype, device=attention_mask.device)  # padding=0
#             chunk_mask_kv = attention_mask[:, chunk_idx*group_size:(chunk_idx+1)*group_size]
#             mask_kv = torch.cat([global_mask_kv, dummy_mask_kv, chunk_mask_kv], dim=1)  # [bsz, kv_len_per_chunk]
#         else:
#             # Chunks 1-N: Q has NO cache, K/V have real cache
#             # CRITICAL: Following Transformer-XL principle - cache is read-only context

#             # Q: [global, chunk] - NO cache! - length = q_len_per_chunk (2056)
#             q_with_ctx = torch.cat([global_q, chunk_q], dim=2)  # [bsz, nh, q_len_per_chunk, hd]

#             # K/V: [global, real_cache, chunk] - length = kv_len_per_chunk (2312)
#             # Extract recurrent cache K/V from previous chunk
#             cache_k = key_chunks[:, :, chunk_idx-1, -self.recurrence_size:, :]  # [bsz, nh, recurrence_size, hd]
#             cache_v = value_chunks[:, :, chunk_idx-1, -self.recurrence_size:, :]

#             k_with_ctx = torch.cat([global_k, cache_k, chunk_k], dim=2)  # [bsz, nh, kv_len_per_chunk, hd]
#             v_with_ctx = torch.cat([global_v, cache_v, chunk_v], dim=2)

#             # Q mask: [global, chunk] - all visible
#             global_mask_q = torch.ones(bsz, num_local_slots, dtype=attention_mask.dtype, device=attention_mask.device)
#             chunk_mask_q = attention_mask[:, chunk_idx*group_size:(chunk_idx+1)*group_size]
#             mask_q = torch.cat([global_mask_q, chunk_mask_q], dim=1)  # [bsz, q_len_per_chunk]

#             # K/V mask: [global, recurrent_cache, chunk] - all visible
#             global_mask_kv = torch.ones(bsz, num_local_slots, dtype=attention_mask.dtype, device=attention_mask.device)
#             recurrent_mask_kv = torch.ones(bsz, self.recurrence_size, dtype=attention_mask.dtype, device=attention_mask.device)  # cache visible!
#             chunk_mask_kv = attention_mask[:, chunk_idx*group_size:(chunk_idx+1)*group_size]
#             mask_kv = torch.cat([global_mask_kv, recurrent_mask_kv, chunk_mask_kv], dim=1)  # [bsz, kv_len_per_chunk]

#         all_chunks_q.append(q_with_ctx)
#         all_chunks_k.append(k_with_ctx)
#         all_chunks_v.append(v_with_ctx)
#         all_chunk_masks_q.append(mask_q)
#         all_chunk_masks_kv.append(mask_kv)

#     # ========== Step 5: Stack all chunks (Q and K/V have DIFFERENT lengths) ==========
#     # Process Q separately
#     all_chunks_q_permuted = [c.permute(0, 2, 1, 3) for c in all_chunks_q]  # List of [bsz, q_len_per_chunk, nh, hd]
#     all_chunks_q_stacked = torch.stack(all_chunks_q_permuted, dim=0)  # [num_groups, bsz, q_len_per_chunk, nh, hd]
#     all_chunks_q_flat = all_chunks_q_stacked.reshape(num_groups * bsz, q_len_per_chunk, self.num_heads, self.head_dim)
#     all_masks_q_stacked = torch.stack(all_chunk_masks_q, dim=0)
#     all_masks_q_flat = all_masks_q_stacked.reshape(num_groups * bsz, q_len_per_chunk)

#     # Process K/V together (more efficient - avoid redundant operations)
#     # First, stack K and V together for each chunk: [bsz, nh, kv_len, hd] -> [bsz, kv_len, 2, nh, hd]
#     all_chunks_kv_permuted = []
#     for k_chunk, v_chunk in zip(all_chunks_k, all_chunks_v):
#         # k_chunk, v_chunk: [bsz, nh, kv_len_per_chunk, hd]
#         k_perm = k_chunk.permute(0, 2, 1, 3)  # [bsz, kv_len, nh, hd]
#         v_perm = v_chunk.permute(0, 2, 1, 3)  # [bsz, kv_len, nh, hd]
#         kv_perm = torch.stack([k_perm, v_perm], dim=2)  # [bsz, kv_len, 2, nh, hd]
#         all_chunks_kv_permuted.append(kv_perm)

#     # Stack and reshape KV together
#     all_chunks_kv_stacked = torch.stack(all_chunks_kv_permuted, dim=0)  # [num_groups, bsz, kv_len, 2, nh, hd]
#     all_chunks_kv_flat = all_chunks_kv_stacked.reshape(num_groups * bsz, kv_len_per_chunk, 2, self.num_heads, self.head_dim)
#     all_masks_kv_stacked = torch.stack(all_chunk_masks_kv, dim=0)
#     all_masks_kv_flat = all_masks_kv_stacked.reshape(num_groups * bsz, kv_len_per_chunk)

#     # ========== Step 6: Flash Attention with KV packed (support different Q and KV lengths) ==========
#     # CRITICAL: Use flash_attn_varlen_kvpacked_func to support Q and KV with different lengths

#     # Unpad Q (using Q's mask)
#     q_unpad, indices_q, cu_seqlens_q, max_seqlen_q = unpad_input(
#         rearrange(all_chunks_q_flat, "b s h d -> b s (h d)"),
#         all_masks_q_flat
#     )
#     q_unpad = rearrange(q_unpad, "nnz (h d) -> nnz h d", h=self.num_heads)

#     # Unpad KV (already packed from Step 5)
#     # all_chunks_kv_flat: [num_groups*bsz, kv_len_per_chunk, 2, nh, hd]
#     kv_flat_2d = rearrange(all_chunks_kv_flat, "b s two h d -> b s (two h d)")

#     # Unpad KV together (using KV's mask)
#     kv_unpad, indices_kv, cu_seqlens_kv, max_seqlen_kv = unpad_input(kv_flat_2d, all_masks_kv_flat)
#     kv_unpad = rearrange(kv_unpad, "nnz (two h d) -> nnz two h d", two=2, h=self.num_heads)
#     # kv_unpad shape: [total_kv_tokens, 2, num_heads, head_dim] - packed K/V format

#     # Flash Attention with KV packed (supports different Q and KV lengths!)
#     # CRITICAL: Each chunk is processed as an INDEPENDENT causal sequence
#     # Q length: num_groups*bsz sequences of q_len_per_chunk tokens each
#     # KV length: num_groups*bsz sequences of kv_len_per_chunk tokens each
#     output_unpad = flash_attn_varlen_kvpacked_func(
#         q_unpad,              # [total_q_tokens, num_heads, head_dim]
#         kv_unpad,             # [total_kv_tokens, 2, num_heads, head_dim] - packed K/V
#         cu_seqlens_q,         # Q sequence boundaries
#         cu_seqlens_kv,        # KV sequence boundaries
#         max_seqlen_q,         # Q max sequence length
#         max_seqlen_kv,        # KV max sequence length
#         dropout_p=0.0,
#         softmax_scale=None,
#         causal=True
#     )  # Output length = Q length (total_q_tokens)

#     # Pad back to original shape (using Q's indices and length)
#     output = rearrange(
#         pad_input(rearrange(output_unpad, "nnz h d -> nnz (h d)"), indices_q, num_groups * bsz, q_len_per_chunk),
#         "b s (h d) -> b s h d",
#         h=self.num_heads,
#     )  # [num_groups*bsz, q_len_per_chunk, nh, hd] - Note: Q's length, not K/V's!

#     # ========== Step 7: Extract outputs and reshape ==========
#     # Reshape from [num_groups*bsz, q_len_per_chunk, nh, hd] to [num_groups, bsz, q_len_per_chunk, nh, hd]
#     output = output.reshape(num_groups, bsz, q_len_per_chunk, self.num_heads, self.head_dim)

#     # Extract actual token outputs (skip global, NO need to skip cache since Q doesn't have cache!)
#     # Structure for ALL chunks: [global(num_local_slots), tokens(group_size)]
#     # Note: Q has NO cache positions, so skip_len is ONLY global
#     chunk_outputs = []
#     skip_len = num_local_slots  # Only skip global (8), NO cache in Q!

#     for chunk_idx in range(num_groups):
#         chunk_output = output[chunk_idx]  # [bsz, q_len_per_chunk, nh, hd]

#         # All chunks have same structure: [global, tokens]
#         # Extract tokens: skip global (8), keep next group_size (2048) tokens
#         tokens_output = chunk_output[:, skip_len:skip_len+group_size, :, :]  # [bsz, group_size, nh, hd]

#         chunk_outputs.append(tokens_output)

#     # Concatenate all chunk outputs: [bsz, num_groups*group_size, nh, hd]
#     output = torch.cat(chunk_outputs, dim=1)  # [bsz, q_len, nh, hd]

#     return self.o_proj(rearrange(output, "b s h d -> b s (h d)")), None, past_key_value

# endregion


# 训练way3 NEW: Hierarchical Memory with Cache (整合版本)
def forward_flashattn_hierarchical_with_cache(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    padding_mask: Optional[torch.LongTensor] = None,
    # 直接在这个函数中控制的参数
    use_higher_global: bool = True,
    use_local_slots: bool = True,
    use_recurrence_cache: bool = False,  # 是否使用 recurrence cache（Transformer-XL style）
    recurrence_size: Optional[int] = 128,  # recurrence cache 大小
    group_size_ratio: Optional[float] = 0.25,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """
    HiCI hierarchical attention with cache support (full integrated version).

    整合了以下功能：
    1. ✅ Recurrence cache (Transformer-XL style)
    2. ✅ HiCI modules (LocalConstructor + GlobalIntegrator)
    3. ✅ Ablation modes (use_higher_global, use_local_slots)
    4. ✅ 所有逻辑通过参数控制

    ⚠️ CRITICAL: cache 必须紧挨着 chunk，因为它们在位置上是连续的！

    拼接顺序规则：
    - 位置无关的组件（higher_global, local）放在前面
    - cache 必须紧挨着 chunk（位置连续）

    三种 Ablation 模式：
    - Mode 1 (推荐): use_higher_global=True, use_local_slots=False
      Q:   [higher_global, chunk]
      K/V: [higher_global, cache, chunk]
      优势: 无冗余，higher_global 是跨 chunk 信息，cache 是位置延续，chunk 是当前内容

    - Mode 2: use_higher_global=False, use_local_slots=True
      Q:   [local_i, chunk]
      K/V: [local_i, cache, chunk]
      优势: 每个 chunk 有专属的压缩表示

    - Mode 3: use_higher_global=True, use_local_slots=True
      Q:   [higher_global, local_i, chunk]
      K/V: [higher_global, local_i, cache, chunk]
      劣势: 可能冗余（local_i 信息已包含在 higher_global 和 chunk 中）

    Args:
        use_higher_global: whether to use GlobalIntegrator (aggregates all local slots)
        use_local_slots: whether to prepend LocalConstructor slots to each chunk
        use_recurrence_cache: 是否使用 Transformer-XL style 的 recurrence cache
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

    # ✅ 打印 Ablation 配置（只在第一次调用时打印，且只在 rank 0 和 layer 0 打印）
    if not hasattr(self, "_ablation_config_printed"):
        rank = dist.get_rank() if dist.is_initialized() else 0
        layer_idx = getattr(self, "layer_idx", 0)  # 获取当前层索引

        if rank == 0 and layer_idx == 0:  # 只在主进程且第一层打印
            print("\n" + "=" * 80)
            print("📋 HiCI Ablation Configuration")
            print("=" * 80)
            print(f"  ✅ use_higher_global    : {use_higher_global}  (GlobalIntegrator)")
            print(
                f"  {'✅' if use_local_slots else '❌'} use_local_slots    : {use_local_slots}  (LocalConstructor slots)"
            )
            print(
                f"  ✅ use_recurrence_cache : {use_recurrence_cache}  (Recurrence cache)"
            )
            print()

            # 显示当前 Ablation Mode
            if use_higher_global and not use_local_slots and use_recurrence_cache:
                print("📌 Current Mode: Mode 1 (推荐)")
                print("   Q:   [higher_global, chunk]")
                print("   K/V: [higher_global, cache, chunk]")
                print("   优势: 无冗余，信息高度聚合")
            elif not use_higher_global and use_local_slots and use_recurrence_cache:
                print("📌 Current Mode: Mode 2")
                print("   Q:   [local_i, chunk]")
                print("   K/V: [local_i, cache, chunk]")
                print("   优势: 每个 chunk 有专属压缩表示")
            elif use_higher_global and use_local_slots and use_recurrence_cache:
                print("📌 Current Mode: Mode 3 (完整模式)")
                print("   Q:   [higher_global, local_i, chunk]")
                print("   K/V: [higher_global, local_i, cache, chunk]")
                print("   优势: 全部特征，但有冗余")
            else:
                print("📌 Current Mode: Custom")
                print(
                    f"   配置: higher_global={use_higher_global}, local={use_local_slots}, cache={use_recurrence_cache}"
                )

            print("=" * 80 + "\n", flush=True)

        self._ablation_config_printed = True

    # ========== Step 1: 分 chunk ==========
    # ✅ SFT版本：处理不规则序列长度
    if q_len % 4096 == 0:
        group_size = int(q_len * group_size_ratio)
    else:
        group_size = sft_group_size

    num_groups = q_len // group_size

    # Reshape into chunks: [bsz, num_groups, group_size, hidden_size]
    chunks = hidden_states.view(bsz, num_groups, group_size, hidden_size)

    # ========== Step 2: 提取局部记忆（对每个 chunk 提取压缩表示）==========
    # 使用 LocalConstructor 对每个 chunk 单独提取局部全局表示
    # ⚠️ CRITICAL: Check if global_memory exists before using it!
    if (use_higher_global or use_local_slots) and hasattr(self, "local_constructor"):
        # 批处理所有 chunks（并行）
        all_chunks = chunks.view(bsz * num_groups, group_size, hidden_size)

        # 将 attention_mask 也 reshape 成对应的 chunks
        # attention_mask: [bsz, q_len] -> [bsz * num_groups, group_size]
        # zxy: 这里暂时注释掉 attention_mask 的处理，假设 global_memory 不需要 mask
        # if attention_mask is not None:
        #     attention_mask_chunks = attention_mask.view(bsz, num_groups, group_size)
        #     attention_mask_chunks = attention_mask_chunks.view(
        #         bsz * num_groups, group_size
        #     )
        # else:
        #     attention_mask_chunks = None

        # all_local_mems = self.local_constructor(
        #     all_chunks, attention_mask_chunks
        # )  # [bsz * num_groups, num_slots, hidden_size]
        all_local_mems = self.local_constructor(
            all_chunks
        )  # [bsz * num_groups, num_slots, hidden_size]

        # Reshape back: [bsz, num_groups, num_slots, hidden_size]
        num_local_slots = all_local_mems.shape[1]
        local_memories_stacked = all_local_mems.view(
            bsz, num_groups, num_local_slots, hidden_size
        )
    else:
        num_local_slots = 0
        local_memories_stacked = None

    # ========== Step 3: 聚合到高层全局记忆（可选）==========
    # ⚠️ CRITICAL: Requires BOTH global_memory (for local extraction) AND hierarchical_aggregator!
    if (
        use_higher_global
        and hasattr(self, "global_integrator")
        and local_memories_stacked is not None
    ):
        higher_global = self.global_integrator(local_memories_stacked)
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
    # Higher-level global memory
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
    # 优化: 批处理所有 local memories，避免循环调用 projection
    if use_local_slots and local_memories_stacked is not None:
        # Reshape: [bsz, num_groups, num_slots, hidden] -> [bsz*num_groups, num_slots, hidden]
        local_mems_flat = local_memories_stacked.view(
            bsz * num_groups, num_local_slots, hidden_size
        )

        # 一次性投影所有 local memories (批处理，1 次调用 vs num_groups 次!)
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

        # Repeat k/v heads (批处理)
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

        # 转换为列表格式 (与后续代码兼容)
        local_memories_q = [local_q_all[:, i, :, :, :] for i in range(num_groups)]
        local_memories_k = [local_k_all[:, i, :, :, :] for i in range(num_groups)]
        local_memories_v = [local_v_all[:, i, :, :, :] for i in range(num_groups)]

    # ========== Step 7: Process chunks with cache and memories ==========
    # 优化: 预创建固定组件，避免循环中重复操作
    all_chunks_q = []
    all_chunks_k = []
    all_chunks_v = []

    # 预创建固定前缀 (在所有 chunks 中都相同)
    prefix_q, prefix_k, prefix_v = [], [], []
    if use_higher_global and higher_global_q is not None:
        prefix_q.append(higher_global_q)
        prefix_k.append(higher_global_k)
        prefix_v.append(higher_global_v)

    # 预创建 dummy cache (仅 chunk_0 使用)
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

        # 构建 Q (固定前缀 + 可选 local + chunk)
        seq_q = prefix_q.copy()
        if use_local_slots and local_memories_stacked is not None:
            seq_q.append(local_memories_q[chunk_idx])
        seq_q.append(chunk_q)

        # 构建 K/V (固定前缀 + 可选 local + cache + chunk)
        seq_k = prefix_k.copy()
        seq_v = prefix_v.copy()
        if use_local_slots and local_memories_stacked is not None:
            seq_k.append(local_memories_k[chunk_idx])
            seq_v.append(local_memories_v[chunk_idx])

        # Cache (仅 K/V): chunk_0 用 dummy，其他用前一个 chunk 的尾部
        if use_recurrence_cache:
            if chunk_idx == 0:
                seq_k.append(dummy_cache_k)
                seq_v.append(dummy_cache_v)
            else:
                seq_k.append(all_chunks_k[chunk_idx - 1][:, :, -recurrence_size:, :])
                seq_v.append(all_chunks_v[chunk_idx - 1][:, :, -recurrence_size:, :])

        seq_k.append(chunk_k)
        seq_v.append(chunk_v)

        # 拼接
        all_chunks_q.append(torch.cat(seq_q, dim=2))
        all_chunks_k.append(torch.cat(seq_k, dim=2))
        all_chunks_v.append(torch.cat(seq_v, dim=2))

    # 获取每个 chunk 的 Q 和 K/V 长度 (所有 chunks 应该长度相同)
    q_len_per_chunk = all_chunks_q[0].shape[2]  # [bsz, nh, q_len, hd] -> q_len
    kv_len_per_chunk = all_chunks_k[0].shape[2]  # [bsz, nh, kv_len, hd] -> kv_len

    # ========== Step 8: Flash Attention with KV packed (following original implementation) ==========
    # CRITICAL: Use flash_attn_varlen_kvpacked_func to support Q and KV with different lengths
    # 参考原始实现 (lines 605-615)，使用简单的 causal=True 而非复杂自定义 mask

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
    # 优化: 使用 in-place 操作，减少内存分配

    # Reshape chunk masks: [bsz, num_groups, group_size]
    chunk_masks_reshaped = attention_mask.view(bsz, num_groups, group_size)

    # 9.1 构建 Q padding masks (in-place)
    # 预分配完整 tensor，避免多次 concat/expand
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
    if use_local_slots:
        all_masks_q_stacked[:, :, offset : offset + num_local_slots] = 1
        offset += num_local_slots

    # Chunk masks (直接 copy，避免额外分配)
    all_masks_q_stacked[:, :, offset : offset + group_size] = chunk_masks_reshaped

    # 转置: [num_groups, bsz, q_len_per_chunk]
    all_masks_q_stacked = all_masks_q_stacked.transpose(0, 1).contiguous()
    all_masks_q_flat = all_masks_q_stacked.reshape(num_groups * bsz, q_len_per_chunk)

    # 9.2 构建 K/V padding masks (in-place)
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
    if use_local_slots:
        all_masks_kv_stacked[:, :, offset : offset + num_local_slots] = 1
        offset += num_local_slots

    # Cache masks: 从前一个 chunk 的最后 recurrence_size 个 tokens 的 mask 提取
    if use_recurrence_cache:
        # 构建 cache masks: [bsz, num_groups, recurrence_size]
        # chunk_0: dummy cache (全 0)
        # chunk_i (i>0): 从 chunk_{i-1} 的最后 recurrence_size 个 tokens 提取 mask
        cache_masks = torch.zeros(
            bsz,
            num_groups,
            recurrence_size,
            dtype=all_masks_kv_stacked.dtype,
            device=all_masks_kv_stacked.device,
        )

        if num_groups > 1:
            # 向量化提取：前 num_groups-1 个 chunks 的最后 recurrence_size 个 tokens 的 mask
            prev_chunk_tails = chunk_masks_reshaped[
                :, :-1, -recurrence_size:
            ]  # [bsz, num_groups-1, recurrence_size]
            cache_masks[:, 1:, :] = prev_chunk_tails  # 填充到 chunk_1, chunk_2, ...

        # 填充到 all_masks_kv_stacked
        all_masks_kv_stacked[:, :, offset : offset + recurrence_size] = cache_masks
        offset += recurrence_size

    # Chunk masks
    all_masks_kv_stacked[:, :, offset : offset + group_size] = chunk_masks_reshaped

    # 转置: [num_groups, bsz, kv_len_per_chunk]
    all_masks_kv_stacked = all_masks_kv_stacked.transpose(0, 1).contiguous()
    all_masks_kv_flat = all_masks_kv_stacked.reshape(num_groups * bsz, kv_len_per_chunk)

    # Unpad Q unpad_input开始
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
    # CRITICAL: 使用 causal=True 实现标准的 causal attention
    # 每个 chunk 是独立的 causal 序列
    output_unpad = flash_attn_varlen_kvpacked_func(
        q_unpad,  # [total_q_tokens, num_heads, head_dim]
        kv_unpad,  # [total_kv_tokens, 2, num_heads, head_dim] - packed K/V
        cu_seqlens_q,  # Q sequence boundaries
        cu_seqlens_kv,  # KV sequence boundaries
        max_seqlen_q,  # Q max sequence length
        max_seqlen_kv,  # KV max sequence length
        dropout_p=0.0,
        softmax_scale=None,
        causal=True,  # ← 简单的 causal mask，与原始实现一致！
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
    # Q 结构: [higher_global?, local?, chunk]
    # 我们只需要 chunk 部分的输出
    chunk_outputs = []
    skip_len = q_len_per_chunk - group_size  # memories 的总长度 (higher_global + local)

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


# 训练SFTway4 NEW: Hierarchical Memory without Cache
def forward_flashattn_hierarchical(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    padding_mask: Optional[torch.LongTensor] = None,
    # 直接在这个函数中控制的参数
    use_higher_global: bool = True,
    use_local_slots: bool = True,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """
    HiCI hierarchical attention (simplified, no recurrence cache).

    整合了以下功能：
    1. HiCI modules (LocalConstructor + GlobalIntegrator)
    2. Ablation modes (use_higher_global, use_local_slots)

    优化：Q 只包含 chunk tokens，K/V 包含 [memories, chunk]
    - 节省计算：memories 不参与 Q 计算
    - chunk tokens 可以 attend 到 memories（通过 K/V）
    - 输出直接就是 chunk tokens，无需额外提取

    拼接顺序：
    - Q:   [chunk]
    - K/V: [higher_global?, local?, chunk]

    三种 Ablation 模式：
    - Mode 1 (推荐): use_higher_global=True, use_local_slots=False
      Q: [chunk], K/V: [higher_global, chunk]

    - Mode 2: use_higher_global=False, use_local_slots=True
      Q: [chunk], K/V: [local_i, chunk]

    - Mode 3: use_higher_global=True, use_local_slots=True
      Q: [chunk], K/V: [higher_global, local_i, chunk]

    Args:
        use_higher_global: whether to use GlobalIntegrator (aggregates all local slots)
        use_local_slots: whether to prepend LocalConstructor slots to each chunk
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

    # 打印配置（只在第一次调用时打印，且只在 rank 0 和 layer 0 打印）
    if not hasattr(self, "_hierarchical_no_cache_printed"):
        rank = dist.get_rank() if dist.is_initialized() else 0
        layer_idx = getattr(self, "layer_idx", 0)

        if rank == 0 and layer_idx == 0:
            print("\n" + "=" * 80)
            print("HiCI Hierarchical (Optimized: Q=[chunk], K/V=[hici_slots,chunk])")
            print("=" * 80)
            print(f"  use_higher_global : {use_higher_global}")
            print(f"  use_local_slots  : {use_local_slots}")

            if use_higher_global and not use_local_slots:
                print("  Mode 1: Q=[chunk], K/V=[higher_global, chunk]")
            elif not use_higher_global and use_local_slots:
                print("  Mode 2: Q=[chunk], K/V=[local_i, chunk]")
            elif use_higher_global and use_local_slots:
                print("  Mode 3: Q=[chunk], K/V=[higher_global, local_i, chunk]")
            else:
                print("  Baseline: Q=K/V=[chunk]")

            print("=" * 80 + "\n", flush=True)

        self._hierarchical_no_cache_printed = True

    # ========== Step 1: 分 chunk ==========
    # ✅ SFT版本：动态计算能整除 q_len 的 group_size（无 padding，最高效） 原来是4096
    if q_len % 4096 == 0:
        # 标准长度：使用 group_size_ratio
        group_size = int(q_len * group_size_ratio)
    else:
        # 不规则长度：找最接近 target 的因数，完全不需要 padding
        target_group_size = int(q_len * group_size_ratio)

        # 找到 q_len 的所有因数
        divisors = []
        for i in range(1, int(q_len**0.5) + 1):
            if q_len % i == 0:
                divisors.append(i)
                if i != q_len // i:
                    divisors.append(q_len // i)

        # 过滤太小或太大的因数
        # 最小：10（避免太多组）
        # 最大：q_len // 2（至少 2 组，除非 q_len 本身很小）
        min_size = 10 if target_group_size >= 10 else 1
        max_size = q_len // 2 if q_len >= 20 else q_len

        valid_divisors = [d for d in divisors if min_size <= d <= max_size]

        if valid_divisors:
            # 选择最接近 target_group_size 的因数
            group_size = min(valid_divisors, key=lambda x: abs(x - target_group_size))
        else:
            # 没有合适的因数，使用整个序列作为一组
            group_size = q_len

    # 最终检查
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
    # 推理时 attention_mask 可能为 None（所有 token 都有效）
    if attention_mask is None:
        attention_mask = torch.ones(
            bsz, q_len, dtype=torch.bool, device=hidden_states.device
        )
    chunk_masks_reshaped = attention_mask.view(bsz, num_groups, group_size)

    # ========== Step 2: 提取局部记忆（对每个 chunk 提取压缩表示）==========
    # 使用 LocalConstructorFlash 对每个 chunk 单独提取局部全局表示
    if (use_higher_global or use_local_slots) and hasattr(self, "local_constructor"):
        # 批处理所有 chunks（并行）
        all_chunks = chunks.view(bsz * num_groups, group_size, hidden_size)

        # [bsz, num_groups, group_size] -> [bsz * num_groups, group_size]
        attention_mask_chunks = chunk_masks_reshaped.view(bsz * num_groups, group_size)
        all_local_mems = self.local_constructor(
            all_chunks, attention_mask_chunks
        )  # [bsz * num_groups, num_slots, hidden_size]

        # Reshape back: [bsz, num_groups, num_slots, hidden_size]
        num_local_slots = all_local_mems.shape[1]
        local_memories_stacked = all_local_mems.view(
            bsz, num_groups, num_local_slots, hidden_size
        )
    else:
        num_local_slots = 0
        local_memories_stacked = None

    # ========== Step 3: 聚合到高层全局记忆（可选）==========
    if (
        use_higher_global
        and hasattr(self, "global_integrator")
        and local_memories_stacked is not None
    ):
        higher_global = self.global_integrator(local_memories_stacked)
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
    # Higher-level global memory
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
    # 优化：只计算 K/V 投影，Q 不需要（Q 只包含 chunk tokens）
    if use_local_slots and local_memories_stacked is not None:
        # Reshape: [bsz, num_groups, num_slots, hidden] -> [bsz*num_groups, num_slots, hidden]
        local_mems_flat = local_memories_stacked.view(
            bsz * num_groups, num_local_slots, hidden_size
        )

        # 只投影 K/V（Q 不需要，节省计算）
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

        # Repeat k/v heads (批处理)
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
    # 优化：Q 只包含 chunk tokens，K/V 包含 [memories, chunk]
    # 全部使用张量操作，避免 Python 循环

    # query_chunks: [bsz, nh, num_groups, group_size, hd]
    # 目标: [bsz * num_groups, group_size, nh, hd]  (batch 优先)
    all_chunks_q_flat = query_chunks.permute(0, 2, 3, 1, 4).reshape(
        bsz * num_groups, group_size, self.num_heads, self.head_dim
    )

    # K/V: 需要拼接 [higher_global?, local?, chunk]
    # key_chunks/value_chunks: [bsz, nh, num_groups, group_size, hd]

    # 计算 K/V 总长度（只包含实际会放入K/V的memory）
    # 注意：num_local_slots可能非零（用于聚合higher_global），但不一定会放入K/V
    prefix_len = 0
    if use_higher_global:
        prefix_len += num_global_slots
    if use_local_slots:
        prefix_len += num_local_slots
    kv_len_per_chunk = prefix_len + group_size

    if prefix_len > 0:
        # 预分配 K/V tensors: [bsz, nh, num_groups, kv_len_per_chunk, hd]
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
        # 填充 higher_global（所有 chunks 共享）
        if use_higher_global and higher_global_k is not None:
            # higher_global_k: [bsz, nh, num_global_slots, hd]
            # 扩展到所有 chunks: [bsz, nh, num_groups, num_global_slots, hd]
            all_k[:, :, :, offset : offset + num_global_slots, :] = (
                higher_global_k.unsqueeze(2)
            )
            all_v[:, :, :, offset : offset + num_global_slots, :] = (
                higher_global_v.unsqueeze(2)
            )
            offset += num_global_slots

        # 填充 local memories（每个 chunk 不同）
        if use_local_slots and local_k_all is not None:
            # local_k_all: [bsz, num_groups, nh, num_local_slots, hd]
            # 需要转换为: [bsz, nh, num_groups, num_local_slots, hd]
            all_k[:, :, :, offset : offset + num_local_slots, :] = local_k_all.permute(
                0, 2, 1, 3, 4
            )
            all_v[:, :, :, offset : offset + num_local_slots, :] = local_v_all.permute(
                0, 2, 1, 3, 4
            )
            offset += num_local_slots

        # 填充 chunk tokens
        all_k[:, :, :, offset : offset + group_size, :] = key_chunks
        all_v[:, :, :, offset : offset + group_size, :] = value_chunks
    else:
        # 没有 memory，直接使用 chunk
        all_k = key_chunks
        all_v = value_chunks
        kv_len_per_chunk = group_size

    # 转换为 flash attention 需要的格式
    # [bsz, nh, num_groups, kv_len, hd] -> [bsz * num_groups, kv_len, nh, hd]  (batch 优先)
    all_k_flat = all_k.permute(0, 2, 3, 1, 4).reshape(
        bsz * num_groups, kv_len_per_chunk, self.num_heads, self.head_dim
    )
    all_v_flat = all_v.permute(0, 2, 3, 1, 4).reshape(
        bsz * num_groups, kv_len_per_chunk, self.num_heads, self.head_dim
    )

    # Pack K and V: [bsz * num_groups, kv_len, 2, nh, hd]  (batch 优先)
    all_chunks_kv_flat = torch.stack([all_k_flat, all_v_flat], dim=2)

    q_len_per_chunk = group_size

    # ========== Step 9: Prepare padding masks (1=real token, 0=padding) ==========
    # 9.1 Q padding masks（只有 chunk tokens）
    # 数据是按 batch 优先排列的：[bsz * num_groups, group_size]
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
    if use_local_slots:
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
    # causal=True: chunk tokens 只能 attend 到 memories + 自己之前的 chunk tokens
    output_unpad = flash_attn_varlen_kvpacked_func(
        q_unpad,  # [total_q_tokens, num_heads, head_dim]
        kv_unpad,  # [total_kv_tokens, 2, num_heads, head_dim] - packed K/V
        cu_seqlens_q,  # Q sequence boundaries
        cu_seqlens_kv,  # KV sequence boundaries
        max_seqlen_q,  # Q max sequence length
        max_seqlen_kv,  # KV max sequence length
        dropout_p=0.0,
        softmax_scale=None,
        causal=True,  # chunk token i 可以 attend 到所有 memories + chunk tokens 0..i
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

    # 转换为 [bsz, q_len, nh, hd]
    output = output.view(bsz, q_len, self.num_heads, self.head_dim)

    # Output projection
    attn_output = self.o_proj(rearrange(output, "b s h d -> b s (h d)"))

    return attn_output, None, past_key_value


# 推理生成函数way4 NEW: Hierarchical Memory without Cache
def forward_flashattn_hierarchical_sft_inference(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    padding_mask: Optional[torch.LongTensor] = None,
    # 直接在这个函数中控制的参数
    use_higher_global: bool = True,
    use_local_slots: bool = True,
    **kwargs,  # 兼容新版 transformers
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """
    HiCI SFT 推理函数 - 用于 LongBench 等评估任务.

    推理流程：
    1. Prefill phase (q_len > 1): uses HiCI hierarchical attention (same as training)
       - 提取 local memories
       - Aggregates into higher_global context via GlobalIntegrator
       - Q=[tokens], K/V=[higher_global, local, tokens]

    2. Decode 阶段 (q_len == 1)：使用标准 attention + KV cache
       - 新 token attend 到所有 past tokens
       - HiCI information is already fused into token representations via prefill

    与训练保持一致：
    - group_size_ratio=1 (整个序列作为一个 chunk)
    - use_higher_global=True
    - use_local_slots=True
    - Mode 3: Q=[chunk], K/V=[higher_global, local, chunk]
    """

    if output_attentions:
        warnings.warn(
            "Output attentions is not supported for patched `LlamaAttention`, returning `None` instead."
        )

    bsz, q_len, hidden_size = hidden_states.size()

    # 打印配置（只在第一次调用时打印，且只在 rank 0 和 layer 0 打印）
    if not hasattr(self, "_hierarchical_no_cache_printed"):
        rank = dist.get_rank() if dist.is_initialized() else 0
        layer_idx = getattr(self, "layer_idx", 0)

        if rank == 0 and layer_idx == 0:
            print("\n" + "=" * 80)
            print("HiCI Hierarchical (Optimized: Q=[chunk], K/V=[hici_slots,chunk])")
            print("=" * 80)
            print(f"  use_higher_global : {use_higher_global}")
            print(f"  use_local_slots  : {use_local_slots}")

            if use_higher_global and not use_local_slots:
                print("  Mode 1: Q=[chunk], K/V=[higher_global, chunk]")
            elif not use_higher_global and use_local_slots:
                print("  Mode 2: Q=[chunk], K/V=[local_i, chunk]")
            elif use_higher_global and use_local_slots:
                print("  Mode 3: Q=[chunk], K/V=[higher_global, local_i, chunk]")
            else:
                print("  Baseline: Q=K/V=[chunk]")

            print("=" * 80 + "\n", flush=True)

        self._hierarchical_no_cache_printed = True

    # ========== Step 1: 分 chunk ==========
    # ✅ SFT版本：动态计算能整除 q_len 的 group_size（无 padding，最高效） 原来是4096
    if q_len % 1024 == 0:
        # 标准长度：使用 group_size_ratio
        group_size = int(q_len * group_size_ratio)
    else:
        # 不规则长度：找最接近 target 的因数，完全不需要 padding
        target_group_size = int(q_len * group_size_ratio)

        # 找到 q_len 的所有因数
        divisors = []
        for i in range(1, int(q_len**0.5) + 1):
            if q_len % i == 0:
                divisors.append(i)
                if i != q_len // i:
                    divisors.append(q_len // i)

        # 过滤太小或太大的因数
        # 最小：10（避免太多组）
        # 最大：q_len // 2（至少 2 组，除非 q_len 本身很小）
        min_size = 10 if target_group_size >= 10 else 1
        max_size = q_len // 2 if q_len >= 20 else q_len

        valid_divisors = [d for d in divisors if min_size <= d <= max_size]

        if valid_divisors:
            # 选择最接近 target_group_size 的因数
            group_size = min(valid_divisors, key=lambda x: abs(x - target_group_size))
        else:
            # 没有合适的因数，使用整个序列作为一组
            group_size = q_len

    # 最终检查
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
    # 推理时 attention_mask 可能为 None（所有 token 都有效）
    if attention_mask is None:
        attention_mask = torch.ones(
            bsz, q_len, dtype=torch.bool, device=hidden_states.device
        )
    chunk_masks_reshaped = attention_mask.view(bsz, num_groups, group_size)

    # ========== Step 2: 提取局部记忆（对每个 chunk 提取压缩表示）==========
    # 使用 LocalConstructorFlash 对每个 chunk 单独提取局部全局表示
    if (use_higher_global or use_local_slots) and hasattr(self, "local_constructor"):
        # 批处理所有 chunks（并行）
        all_chunks = chunks.view(bsz * num_groups, group_size, hidden_size)

        # [bsz, num_groups, group_size] -> [bsz * num_groups, group_size]
        attention_mask_chunks = chunk_masks_reshaped.view(bsz * num_groups, group_size)
        all_local_mems = self.local_constructor(
            all_chunks, attention_mask_chunks
        )  # [bsz * num_groups, num_slots, hidden_size]

        # Reshape back: [bsz, num_groups, num_slots, hidden_size]
        num_local_slots = all_local_mems.shape[1]
        local_memories_stacked = all_local_mems.view(
            bsz, num_groups, num_local_slots, hidden_size
        )
    else:
        num_local_slots = 0
        local_memories_stacked = None

    # ========== Step 3: 聚合到高层全局记忆（可选）==========
    if (
        use_higher_global
        and hasattr(self, "global_integrator")
        and local_memories_stacked is not None
    ):
        higher_global = self.global_integrator(local_memories_stacked)
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
    # Higher-level global memory
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
    # 优化：只计算 K/V 投影，Q 不需要（Q 只包含 chunk tokens）
    if use_local_slots and local_memories_stacked is not None:
        # Reshape: [bsz, num_groups, num_slots, hidden] -> [bsz*num_groups, num_slots, hidden]
        local_mems_flat = local_memories_stacked.view(
            bsz * num_groups, num_local_slots, hidden_size
        )

        # 只投影 K/V（Q 不需要，节省计算）
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

        # Repeat k/v heads (批处理)
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
    # 优化：Q 只包含 chunk tokens，K/V 包含 [memories, chunk]
    # 全部使用张量操作，避免 Python 循环

    # query_chunks: [bsz, nh, num_groups, group_size, hd]
    # 目标: [bsz * num_groups, group_size, nh, hd]  (batch 优先)
    all_chunks_q_flat = query_chunks.permute(0, 2, 3, 1, 4).reshape(
        bsz * num_groups, group_size, self.num_heads, self.head_dim
    )

    # K/V: 需要拼接 [higher_global?, local?, chunk]
    # key_chunks/value_chunks: [bsz, nh, num_groups, group_size, hd]

    # 计算 K/V 总长度（只包含实际会放入K/V的memory）
    # 注意：num_local_slots可能非零（用于聚合higher_global），但不一定会放入K/V
    prefix_len = 0
    if use_higher_global:
        prefix_len += num_global_slots
    if use_local_slots:
        prefix_len += num_local_slots
    kv_len_per_chunk = prefix_len + group_size

    if prefix_len > 0:
        # 预分配 K/V tensors: [bsz, nh, num_groups, kv_len_per_chunk, hd]
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
        # 填充 higher_global（所有 chunks 共享）
        if use_higher_global and higher_global_k is not None:
            # higher_global_k: [bsz, nh, num_global_slots, hd]
            # 扩展到所有 chunks: [bsz, nh, num_groups, num_global_slots, hd]
            all_k[:, :, :, offset : offset + num_global_slots, :] = (
                higher_global_k.unsqueeze(2)
            )
            all_v[:, :, :, offset : offset + num_global_slots, :] = (
                higher_global_v.unsqueeze(2)
            )
            offset += num_global_slots

        # 填充 local memories（每个 chunk 不同）
        if use_local_slots and local_k_all is not None:
            # local_k_all: [bsz, num_groups, nh, num_local_slots, hd]
            # 需要转换为: [bsz, nh, num_groups, num_local_slots, hd]
            all_k[:, :, :, offset : offset + num_local_slots, :] = local_k_all.permute(
                0, 2, 1, 3, 4
            )
            all_v[:, :, :, offset : offset + num_local_slots, :] = local_v_all.permute(
                0, 2, 1, 3, 4
            )
            offset += num_local_slots

        # 填充 chunk tokens
        all_k[:, :, :, offset : offset + group_size, :] = key_chunks
        all_v[:, :, :, offset : offset + group_size, :] = value_chunks
    else:
        # 没有 memory，直接使用 chunk
        all_k = key_chunks
        all_v = value_chunks
        kv_len_per_chunk = group_size

    # 转换为 flash attention 需要的格式
    # [bsz, nh, num_groups, kv_len, hd] -> [bsz * num_groups, kv_len, nh, hd]  (batch 优先)
    all_k_flat = all_k.permute(0, 2, 3, 1, 4).reshape(
        bsz * num_groups, kv_len_per_chunk, self.num_heads, self.head_dim
    )
    all_v_flat = all_v.permute(0, 2, 3, 1, 4).reshape(
        bsz * num_groups, kv_len_per_chunk, self.num_heads, self.head_dim
    )

    # Pack K and V: [bsz * num_groups, kv_len, 2, nh, hd]  (batch 优先)
    all_chunks_kv_flat = torch.stack([all_k_flat, all_v_flat], dim=2)

    q_len_per_chunk = group_size

    # ========== Step 9: Prepare padding masks (1=real token, 0=padding) ==========
    # 9.1 Q padding masks（只有 chunk tokens）
    # 数据是按 batch 优先排列的：[bsz * num_groups, group_size]
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
    if use_local_slots:
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
    # causal=True: chunk tokens 只能 attend 到 memories + 自己之前的 chunk tokens
    output_unpad = flash_attn_varlen_kvpacked_func(
        q_unpad,  # [total_q_tokens, num_heads, head_dim]
        kv_unpad,  # [total_kv_tokens, 2, num_heads, head_dim] - packed K/V
        cu_seqlens_q,  # Q sequence boundaries
        cu_seqlens_kv,  # KV sequence boundaries
        max_seqlen_q,  # Q max sequence length
        max_seqlen_kv,  # KV max sequence length
        dropout_p=0.0,
        softmax_scale=None,
        causal=True,  # chunk token i 可以 attend 到所有 memories + chunk tokens 0..i
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

    # 转换为 [bsz, q_len, nh, hd]
    output = output.view(bsz, q_len, self.num_heads, self.head_dim)

    # Output projection
    attn_output = self.o_proj(rearrange(output, "b s h d -> b s (h d)"))

    return attn_output, None, past_key_value


# 训练way4 NEW: Hierarchical Memory without Cache 尽量让段大小接近训练大小(1024)
def forward_flashattn_hierarchical_Seg(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    padding_mask: Optional[torch.LongTensor] = None,
    # 直接在这个函数中控制的参数
    use_higher_global: bool = True,
    use_local_slots: bool = True,
    target_segment_size: int = 1024,  # ✅ 新增：目标段大小，与预训练保持一致
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """
    HiCI hierarchical attention (simplified, no recurrence cache).

    ✅ 修改：固定段大小≈1024，与预训练保持一致（组数会变化）

    分组逻辑：
    - 目标段大小 = 1024（可通过 target_segment_size 参数调整）
    - 如果 q_len 能被 1024 整除，直接使用 1024
    - 否则找 q_len 的因数中最接近 1024 的
    - 范围限制：512 ~ 2048（保证段大小不会偏离太多）

    整合了以下功能：
    1. HiCI modules (LocalConstructor + GlobalIntegrator)
    2. Ablation modes (use_higher_global, use_local_slots)

    优化：Q 只包含 chunk tokens，K/V 包含 [memories, chunk]
    - 节省计算：memories 不参与 Q 计算
    - chunk tokens 可以 attend 到 memories（通过 K/V）
    - 输出直接就是 chunk tokens，无需额外提取

    拼接顺序：
    - Q:   [chunk]
    - K/V: [higher_global?, local?, chunk]

    三种 Ablation 模式：
    - Mode 1 (推荐): use_higher_global=True, use_local_slots=False
      Q: [chunk], K/V: [higher_global, chunk]

    - Mode 2: use_higher_global=False, use_local_slots=True
      Q: [chunk], K/V: [local_i, chunk]

    - Mode 3: use_higher_global=True, use_local_slots=True
      Q: [chunk], K/V: [higher_global, local_i, chunk]

    Args:
        use_higher_global: whether to use GlobalIntegrator (aggregates all local slots)
        use_local_slots: whether to prepend LocalConstructor slots to each chunk
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

    # 打印配置（只在第一次调用时打印，且只在 rank 0 和 layer 0 打印）
    if not hasattr(self, "_hierarchical_no_cache_printed"):
        rank = dist.get_rank() if dist.is_initialized() else 0
        layer_idx = getattr(self, "layer_idx", 0)

        if rank == 0 and layer_idx == 0:
            print("\n" + "=" * 80)
            print("HiCI Hierarchical (Optimized: Q=[chunk], K/V=[hici_slots,chunk])")
            print("=" * 80)
            print(f"  use_higher_global : {use_higher_global}")
            print(f"  use_local_slots  : {use_local_slots}")

            if use_higher_global and not use_local_slots:
                print("  Mode 1: Q=[chunk], K/V=[higher_global, chunk]")
            elif not use_higher_global and use_local_slots:
                print("  Mode 2: Q=[chunk], K/V=[local_i, chunk]")
            elif use_higher_global and use_local_slots:
                print("  Mode 3: Q=[chunk], K/V=[higher_global, local_i, chunk]")
            else:
                print("  Baseline: Q=K/V=[chunk]")

            print("=" * 80 + "\n", flush=True)

        self._hierarchical_no_cache_printed = True

    # ========== Step 1: 分 chunk ==========
    # ✅ SFT版本（Seg）：固定段大小≈1024，与预训练保持一致
    if q_len % target_segment_size == 0:
        # 能整除：直接使用目标段大小
        group_size = target_segment_size
    else:
        # 不能整除：找最接近 target_segment_size 的因数
        # 找到 q_len 的所有因数
        divisors = []
        for i in range(1, int(q_len**0.5) + 1):
            if q_len % i == 0:
                divisors.append(i)
                if i != q_len // i:
                    divisors.append(q_len // i)

        # 过滤合理范围的段大小 (512 ~ 2048)
        min_size = 512
        max_size = min(2048, q_len // 2) if q_len >= 1024 else q_len

        valid_divisors = [d for d in divisors if min_size <= d <= max_size]

        if valid_divisors:
            # 选择最接近 target_segment_size 的因数
            group_size = min(valid_divisors, key=lambda x: abs(x - target_segment_size))
        else:
            # 没有合适的因数，尝试放宽范围 (256 ~ 4096)
            valid_divisors = [d for d in divisors if 256 <= d <= min(4096, q_len)]
            if valid_divisors:
                group_size = min(
                    valid_divisors, key=lambda x: abs(x - target_segment_size)
                )
            else:
                # 仍然没有，使用整个序列作为一组
                group_size = q_len

    # 最终检查
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
    # 推理时 attention_mask 可能为 None（所有 token 都有效）
    if attention_mask is None:
        attention_mask = torch.ones(
            bsz, q_len, dtype=torch.bool, device=hidden_states.device
        )
    chunk_masks_reshaped = attention_mask.view(bsz, num_groups, group_size)

    # ========== Step 2: 提取局部记忆（对每个 chunk 提取压缩表示）==========
    # 使用 LocalConstructorFlash 对每个 chunk 单独提取局部全局表示
    if (use_higher_global or use_local_slots) and hasattr(self, "local_constructor"):
        # 批处理所有 chunks（并行）
        all_chunks = chunks.view(bsz * num_groups, group_size, hidden_size)

        # [bsz, num_groups, group_size] -> [bsz * num_groups, group_size]
        attention_mask_chunks = chunk_masks_reshaped.view(bsz * num_groups, group_size)
        all_local_mems = self.local_constructor(
            all_chunks, attention_mask_chunks
        )  # [bsz * num_groups, num_slots, hidden_size]

        # Reshape back: [bsz, num_groups, num_slots, hidden_size]
        num_local_slots = all_local_mems.shape[1]
        local_memories_stacked = all_local_mems.view(
            bsz, num_groups, num_local_slots, hidden_size
        )
    else:
        num_local_slots = 0
        local_memories_stacked = None

    # ========== Step 3: 聚合到高层全局记忆（可选）==========
    if (
        use_higher_global
        and hasattr(self, "global_integrator")
        and local_memories_stacked is not None
    ):
        higher_global = self.global_integrator(local_memories_stacked)
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
    # Higher-level global memory
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
    # 优化：只计算 K/V 投影，Q 不需要（Q 只包含 chunk tokens）
    if use_local_slots and local_memories_stacked is not None:
        # Reshape: [bsz, num_groups, num_slots, hidden] -> [bsz*num_groups, num_slots, hidden]
        local_mems_flat = local_memories_stacked.view(
            bsz * num_groups, num_local_slots, hidden_size
        )

        # 只投影 K/V（Q 不需要，节省计算）
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

        # Repeat k/v heads (批处理)
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
    # 优化：Q 只包含 chunk tokens，K/V 包含 [memories, chunk]
    # 全部使用张量操作，避免 Python 循环

    # query_chunks: [bsz, nh, num_groups, group_size, hd]
    # 目标: [bsz * num_groups, group_size, nh, hd]  (batch 优先)
    all_chunks_q_flat = query_chunks.permute(0, 2, 3, 1, 4).reshape(
        bsz * num_groups, group_size, self.num_heads, self.head_dim
    )

    # K/V: 需要拼接 [higher_global?, local?, chunk]
    # key_chunks/value_chunks: [bsz, nh, num_groups, group_size, hd]

    # 计算 K/V 总长度（只包含实际会放入K/V的memory）
    # 注意：num_local_slots可能非零（用于聚合higher_global），但不一定会放入K/V
    prefix_len = 0
    if use_higher_global:
        prefix_len += num_global_slots
    if use_local_slots:
        prefix_len += num_local_slots
    kv_len_per_chunk = prefix_len + group_size

    if prefix_len > 0:
        # 预分配 K/V tensors: [bsz, nh, num_groups, kv_len_per_chunk, hd]
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
        # 填充 higher_global（所有 chunks 共享）
        if use_higher_global and higher_global_k is not None:
            # higher_global_k: [bsz, nh, num_global_slots, hd]
            # 扩展到所有 chunks: [bsz, nh, num_groups, num_global_slots, hd]
            all_k[:, :, :, offset : offset + num_global_slots, :] = (
                higher_global_k.unsqueeze(2)
            )
            all_v[:, :, :, offset : offset + num_global_slots, :] = (
                higher_global_v.unsqueeze(2)
            )
            offset += num_global_slots

        # 填充 local memories（每个 chunk 不同）
        if use_local_slots and local_k_all is not None:
            # local_k_all: [bsz, num_groups, nh, num_local_slots, hd]
            # 需要转换为: [bsz, nh, num_groups, num_local_slots, hd]
            all_k[:, :, :, offset : offset + num_local_slots, :] = local_k_all.permute(
                0, 2, 1, 3, 4
            )
            all_v[:, :, :, offset : offset + num_local_slots, :] = local_v_all.permute(
                0, 2, 1, 3, 4
            )
            offset += num_local_slots

        # 填充 chunk tokens
        all_k[:, :, :, offset : offset + group_size, :] = key_chunks
        all_v[:, :, :, offset : offset + group_size, :] = value_chunks
    else:
        # 没有 memory，直接使用 chunk
        all_k = key_chunks
        all_v = value_chunks
        kv_len_per_chunk = group_size

    # 转换为 flash attention 需要的格式
    # [bsz, nh, num_groups, kv_len, hd] -> [bsz * num_groups, kv_len, nh, hd]  (batch 优先)
    all_k_flat = all_k.permute(0, 2, 3, 1, 4).reshape(
        bsz * num_groups, kv_len_per_chunk, self.num_heads, self.head_dim
    )
    all_v_flat = all_v.permute(0, 2, 3, 1, 4).reshape(
        bsz * num_groups, kv_len_per_chunk, self.num_heads, self.head_dim
    )

    # Pack K and V: [bsz * num_groups, kv_len, 2, nh, hd]  (batch 优先)
    all_chunks_kv_flat = torch.stack([all_k_flat, all_v_flat], dim=2)

    q_len_per_chunk = group_size

    # ========== Step 9: Prepare padding masks (1=real token, 0=padding) ==========
    # 9.1 Q padding masks（只有 chunk tokens）
    # 数据是按 batch 优先排列的：[bsz * num_groups, group_size]
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
    if use_local_slots:
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
    # causal=True: chunk tokens 只能 attend 到 memories + 自己之前的 chunk tokens
    output_unpad = flash_attn_varlen_kvpacked_func(
        q_unpad,  # [total_q_tokens, num_heads, head_dim]
        kv_unpad,  # [total_kv_tokens, 2, num_heads, head_dim] - packed K/V
        cu_seqlens_q,  # Q sequence boundaries
        cu_seqlens_kv,  # KV sequence boundaries
        max_seqlen_q,  # Q max sequence length
        max_seqlen_kv,  # KV max sequence length
        dropout_p=0.0,
        softmax_scale=None,
        causal=True,  # chunk token i 可以 attend 到所有 memories + chunk tokens 0..i
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

    # 转换为 [bsz, q_len, nh, hd]
    output = output.view(bsz, q_len, self.num_heads, self.head_dim)

    # Output projection
    attn_output = self.o_proj(rearrange(output, "b s h d -> b s (h d)"))

    return attn_output, None, past_key_value


# zxy
# forward_flashattn = forward_flashattn_hierarchical_with_cache
# forward_flashattn_optimized = forward_flashattn_hierarchical_with_cache


#  评估代码
# 评估way1 longlora原本的full attn的函数
# 对较短序列或评估的情况，直接用 flash attention 计算 没有分组 (group_size) 的复杂操作
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
    # ✅ 修复：如果提供了 position_ids，使用其最大值来确定 RoPE 缓存长度
    # 这样可以支持绝对位置编码（如 chunk 使用 start_idx 到 end_idx 的绝对位置）
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

# 评估way2 在原本的评估的基础上引入了一点记忆机制 这是之前没有cache和没有双层记忆机制的版本
# region
# def forward_flashattn_full(
#     self,
#     hidden_states: torch.Tensor,
#     attention_mask: Optional[torch.Tensor] = None,
#     position_ids: Optional[torch.Tensor] = None,
#     past_key_value: Optional[Tuple[torch.Tensor]] = None,
#     output_attentions: bool = False,
#     use_cache: bool = False,
#     padding_mask: Optional[torch.LongTensor] = None,
# ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
#     """Input shape: Batch x Time x Channel

#     attention_mask: [bsz, q_len]
#     """
#     if output_attentions:
#         warnings.warn(
#             "Output attentions is not supported for patched `LlamaAttention`, returning `None` instead."
#         )

#     bsz, q_len, _ = hidden_states.size()

#     # ========== Step 1: Compute global context (CRITICAL for evaluation!) ==========
#     # Use trained global memory to capture document-level information
#     if hasattr(self, 'global_memory'):
#         global_context = self.local_constructor(hidden_states)  # [bsz, num_slots, hidden_size]
#         num_local_slots = global_context.shape[1]

#         # ========== Step 2: Project global context to Q/K/V ==========
#         # Use main attention projections (same as training!)
#         # NOTE: global_context is already processed by global_memory's internal cross-attention,
#         # now we project it to Q/K/V space using the main attention projections
#         global_q = (
#             self.q_proj(global_context)
#             .view(bsz, num_local_slots, self.num_heads, self.head_dim)
#             .transpose(1, 2)
#         )
#         global_k = (
#             self.k_proj(global_context)
#             .view(bsz, num_local_slots, self.num_key_value_heads, self.head_dim)
#             .transpose(1, 2)
#         )
#         global_v = (
#             self.v_proj(global_context)
#             .view(bsz, num_local_slots, self.num_key_value_heads, self.head_dim)
#             .transpose(1, 2)
#         )
#         # [bsz, nh, num_slots, hd]
#     else:
#         # Fallback: if no global memory (shouldn't happen if model trained with it)
#         global_q = None
#         global_k = None
#         global_v = None
#         num_local_slots = 0

#     # ========== Step 3: Standard Q/K/V projections for sequence ==========
#     query_states = (
#         self.q_proj(hidden_states)
#         .view(bsz, q_len, self.num_heads, self.head_dim)
#         .transpose(1, 2)
#     )
#     key_states = (
#         self.k_proj(hidden_states)
#         .view(bsz, q_len, self.num_key_value_heads, self.head_dim)
#         .transpose(1, 2)
#     )
#     value_states = (
#         self.v_proj(hidden_states)
#         .view(bsz, q_len, self.num_key_value_heads, self.head_dim)
#         .transpose(1, 2)
#     )
#     # [bsz, nh, q_len, hd]

#     kv_seq_len = key_states.shape[-2]
#     if past_key_value is not None:
#         kv_seq_len += past_key_value[0].shape[-2]

#     # ========== Step 4: Apply RoPE to sequence Q/K (not to global memory!) ==========
#     # Global Representation Slots are position-independent, so they don't get RoPE
#     if position_ids is not None:
#         max_pos = position_ids.max().item() + 1
#         rope_seq_len = max(kv_seq_len, max_pos)
#     else:
#         rope_seq_len = kv_seq_len

#     cos, sin = self.rotary_emb(value_states, seq_len=rope_seq_len)
#     query_states, key_states = apply_rotary_pos_emb(
#         query_states, key_states, cos, sin, position_ids
#     )

#     # Past Key value support
#     if past_key_value is not None:
#         # reuse k, v, self_attention
#         key_states = torch.cat([past_key_value[0], key_states], dim=2)
#         value_states = torch.cat([past_key_value[1], value_states], dim=2)

#     past_key_value = (key_states, value_states) if use_cache else None

#     # repeat k/v heads if n_kv_heads < n_heads
#     key_states = repeat_kv(key_states, self.num_key_value_groups)
#     value_states = repeat_kv(value_states, self.num_key_value_groups)

#     # ========== Step 5: Concatenate global memory with sequence ==========
#     if global_q is not None:
#         # Repeat global K/V heads to match sequence head count
#         global_k = repeat_kv(global_k, self.num_key_value_groups)
#         global_v = repeat_kv(global_v, self.num_key_value_groups)

#         # Concatenate: [global, sequence]
#         query_states = torch.cat([global_q, query_states], dim=2)  # [bsz, nh, num_slots+q_len, hd]
#         key_states = torch.cat([global_k, key_states], dim=2)
#         value_states = torch.cat([global_v, value_states], dim=2)

#         # Update total sequence length
#         total_seq_len = num_local_slots + q_len
#     else:
#         total_seq_len = q_len

#     # Flash attention codes from
#     # https://github.com/HazyResearch/flash-attention/blob/main/flash_attn/flash_attention.py

#     # ========== Step 6: Prepare attention mask ==========
#     # Global memory positions are always visible (all 1s)
#     if global_q is not None and attention_mask is not None:
#         # Create mask for global memory (all 1s)
#         global_mask = torch.ones(
#             bsz, num_local_slots,
#             dtype=attention_mask.dtype,
#             device=attention_mask.device
#         )
#         # Concatenate: [global_mask, sequence_mask]
#         key_padding_mask = torch.cat([global_mask, attention_mask], dim=1)  # [bsz, num_slots+q_len]
#     else:
#         key_padding_mask = attention_mask

#     # ========== Step 7: Flash Attention ==========
#     # transform the data into the format required by flash attention
#     qkv = torch.stack(
#         [query_states, key_states, value_states], dim=2
#     )  # [bsz, nh, 3, total_seq_len, hd]
#     qkv = qkv.transpose(1, 3)  # [bsz, total_seq_len, 3, nh, hd]

#     nheads = qkv.shape[-2]
#     x = rearrange(qkv, "b s three h d -> b s (three h d)")
#     x_unpad, indices, cu_q_lens, max_s = unpad_input(x, key_padding_mask)
#     x_unpad = rearrange(
#         x_unpad, "nnz (three h d) -> nnz three h d", three=3, h=nheads
#     )
#     output_unpad = flash_attn_varlen_qkvpacked_func(
#         x_unpad, cu_q_lens, max_s, 0.0, softmax_scale=None, causal=True
#     )
#     output = rearrange(
#         pad_input(
#             rearrange(output_unpad, "nnz h d -> nnz (h d)"), indices, bsz, total_seq_len
#         ),
#         "b s (h d) -> b s h d",
#         h=nheads,
#     )
#     # [bsz, total_seq_len, nh, hd]

#     # ========== Step 8: Extract sequence outputs (skip global memory positions) ==========
#     if global_q is not None:
#         # Skip the first num_local_slots positions (global memory outputs)
#         # Keep only the sequence token outputs
#         output = output[:, num_local_slots:, :, :]  # [bsz, q_len, nh, hd]

#     output = output.reshape(bsz, q_len, self.num_heads, self.head_dim)

#     return self.o_proj(rearrange(output, "b s h d -> b s (h d)")), None, past_key_value
# endregion


# 评估way3 longlora评估时的full attn的函数（加入了双层记忆机制但没有cache机制）
# region
def forward_flashattn_full_hierarchical(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    padding_mask: Optional[torch.LongTensor] = None,
    # 记忆提取参数
    use_higher_global: bool = True,
    use_local_slots: bool = True,
    group_size_ratio: Optional[float] = 1,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """
    Eval mode 3: Full Attention + HiCI (no chunked attention, no cache)

    特点：
    1. HiCI extraction: chunks → local slots → global context
    2. Attention: Full Attention (no chunking), HiCI slots prepended to sequence
    3. 无 recurrence cache

    拼接结构：
    Q/K/V: [higher_global, local_slots, full_sequence]

    与训练版本的区别：
    - 训练：分组attention，每个chunk独立计算
    - 评估：Full attention，所有token一起计算
    """
    if output_attentions:
        warnings.warn(
            "Output attentions is not supported for patched `LlamaAttention`, returning `None` instead."
        )

    bsz, q_len, hidden_size = hidden_states.size()

    # ✅ 打印配置（只在第一次调用时打印）
    if not hasattr(self, "_eval_config_printed_way3"):
        rank = dist.get_rank() if dist.is_initialized() else 0
        layer_idx = getattr(self, "layer_idx", 0)

        if rank == 0 and layer_idx == 0:
            print("\n" + "=" * 80)
            print("📋 Evaluation Mode: Full Attention + HiCI")
            print("=" * 80)
            print(f"  ✅ use_higher_global : {use_higher_global}")
            print(f"  ✅ use_local_slots  : {use_local_slots}")
            print(
                f"  ✅ group_size_ratio  : {group_size_ratio} (for HiCI extraction)"
            )
            print(f"  ❌ recurrence_cache  : disabled")
            print()
            print("📌 Attention Structure:")
            print("   Q/K/V: [global_context, local_slots, full_sequence]")
            print("   → Full Attention (no chunking)")
            print("=" * 80 + "\n", flush=True)

        self._eval_config_printed_way3 = True

    # ========== Step 1: 分组提取局部记忆 ==========
    # ✅ SFT版本：处理不规则序列长度
    if q_len % 4096 == 0:
        group_size = int(q_len * group_size_ratio)
    else:
        group_size = sft_group_size
    num_groups = q_len // group_size

    # Reshape into chunks for memory extraction
    chunks = hidden_states.view(bsz, num_groups, group_size, hidden_size)

    # ========== Step 2: 提取局部记忆 ==========
    if (use_higher_global or use_local_slots) and hasattr(self, "local_constructor"):
        all_chunks = chunks.view(bsz * num_groups, group_size, hidden_size)

        # 将 attention_mask 也 reshape 成对应的 chunks
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
        local_memories_stacked = all_local_mems.view(
            bsz, num_groups, num_local_slots, hidden_size
        )
        # 将所有局部记忆展平为 [bsz, num_groups * num_local_slots, hidden_size]
        local_memories_flat = local_memories_stacked.view(
            bsz, num_groups * num_local_slots, hidden_size
        )
    else:
        num_local_slots = 0
        local_memories_stacked = None
        local_memories_flat = None

    # ========== Step 3: 聚合到全局记忆 ==========
    if (
        use_higher_global
        and hasattr(self, "global_integrator")
        and local_memories_stacked is not None
    ):
        higher_global = self.global_integrator(local_memories_stacked)
        num_global_slots = higher_global.shape[1]
    else:
        higher_global = None
        num_global_slots = 0

    # ========== Step 4: 标准 Q/K/V 投影（全序列）==========
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

    # ========== Step 5: 投影记忆到 Q/K/V ==========
    # 全局记忆
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

    # 局部记忆（展平后的所有局部记忆）
    if use_local_slots and local_memories_flat is not None:
        total_local_slots = num_groups * num_local_slots
        local_q = (
            self.q_proj(local_memories_flat)
            .view(bsz, total_local_slots, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        local_k = (
            self.k_proj(local_memories_flat)
            .view(bsz, total_local_slots, self.num_key_value_heads, self.head_dim)
            .transpose(1, 2)
        )
        local_v = (
            self.v_proj(local_memories_flat)
            .view(bsz, total_local_slots, self.num_key_value_heads, self.head_dim)
            .transpose(1, 2)
        )
        local_k = repeat_kv(local_k, self.num_key_value_groups)
        local_v = repeat_kv(local_v, self.num_key_value_groups)
    else:
        local_q = local_k = local_v = None
        total_local_slots = 0

    # ========== Step 6: 拼接 [global, local, sequence] ==========
    # 构建完整的 Q
    q_parts = []
    if higher_global_q is not None:
        q_parts.append(higher_global_q)
    if local_q is not None:
        q_parts.append(local_q)
    q_parts.append(query_states)
    full_q = torch.cat(q_parts, dim=2)  # [bsz, nh, total_len, hd]

    # 构建完整的 K/V
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

    # 计算总长度
    prefix_len = num_global_slots + total_local_slots
    total_len = prefix_len + q_len

    # ========== Step 7: Flash Attention (Full Attention) ==========
    # 转换格式: [bsz, nh, seq, hd] -> [bsz, seq, nh, hd]
    full_q = full_q.transpose(1, 2)
    full_k = full_k.transpose(1, 2)
    full_v = full_v.transpose(1, 2)

    # Stack Q/K/V: [bsz, seq, 3, nh, hd]
    qkv = torch.stack([full_q, full_k, full_v], dim=2)

    # 构建 attention mask: 记忆部分始终可见，序列部分用原始 mask
    # [bsz, total_len]
    hici_mask = torch.ones(
        bsz, prefix_len, dtype=attention_mask.dtype, device=attention_mask.device
    )
    full_mask = torch.cat([hici_mask, attention_mask], dim=1)

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

    # ========== Step 8: 提取序列部分的输出（跳过记忆部分）==========
    # output: [bsz, total_len, nh, hd]
    # 我们只需要后面 q_len 个 token 的输出
    seq_output = output[:, prefix_len:, :, :]  # [bsz, q_len, nh, hd]

    # Output projection
    attn_output = self.o_proj(rearrange(seq_output, "b s h d -> b s (h d)"))

    return attn_output, None, past_key_value


# endregion
# 评估way4 longlora评估时的full attn的函数（加入了双层记忆机制和cache机制） 有错误
# region
# def forward_flashattn_full_way3(
#     self,
#     hidden_states: torch.Tensor,
#     attention_mask: Optional[torch.Tensor] = None,
#     position_ids: Optional[torch.Tensor] = None,
#     past_key_value: Optional[Tuple[torch.Tensor]] = None,
#     output_attentions: bool = False,
#     use_cache: bool = False,
#     padding_mask: Optional[torch.LongTensor] = None,
# ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
#     """
#     方案 1: 混合评估策略（推荐用于评估）

#     两阶段处理：
#     1. 记忆提取阶段：分 chunk 应用记忆机制（与训练一致）
#        - 对每个 chunk 应用 LocalConstructor
#        - 用 HierarchicalAggregator 聚合成全局表示

#     2. Attention 阶段：Full attention（不分 chunk，更准确）
#        - 将全局表示拼接到完整序列前面
#        - 一次性做 full flash attention

#     优点：
#     - ✅ 记忆机制与训练一致（避免 train/test mismatch）
#     - ✅ 使用所有训练的参数（GlobalMemory + HierarchicalAgg）
#     - ✅ Full attention 更准确（像 LongLoRA 的评估策略）

#     Args:
#         hidden_states: [batch, seq_len, hidden_size]
#         attention_mask: [batch, seq_len]
#         position_ids: [batch, seq_len]

#     Returns:
#         attn_output: [batch, seq_len, hidden_size]
#         None (no attention weights)
#         past_key_value (if use_cache)
#     """
#     if output_attentions:
#         warnings.warn(
#             "Output attentions is not supported for patched `LlamaAttention`, returning `None` instead."
#         )

#     bsz, q_len, hidden_size = hidden_states.size()

#     # ========================================================================
#     # Part 1: 记忆提取阶段（分 chunk，与训练一致）
#     # ========================================================================
#     higher_global = None
#     num_local_slots = 0

#     if hasattr(self, "local_constructor") and hasattr(self, "global_integrator"):
#         # 分 chunk 配置（必须与训练时一致）
#         chunk_size = 2048
#         num_chunks = (q_len + chunk_size - 1) // chunk_size

#         # Step 1.1: 对每个 chunk 提取局部记忆
#         local_memories = []

#         for i in range(num_chunks):
#             start_idx = i * chunk_size
#             end_idx = min(start_idx + chunk_size, q_len)
#             chunk = hidden_states[
#                 :, start_idx:end_idx, :
#             ]  # [batch, chunk_len, hidden_size]

#             # 如果 chunk 不足 chunk_size，需要 padding
#             chunk_len = chunk.shape[1]
#             if chunk_len < chunk_size:
#                 padding = torch.zeros(
#                     bsz,
#                     chunk_size - chunk_len,
#                     hidden_size,
#                     device=hidden_states.device,
#                     dtype=hidden_states.dtype,
#                 )
#                 chunk = torch.cat(
#                     [chunk, padding], dim=1
#                 )  # [batch, chunk_size, hidden_size]

#             # 应用 LocalConstructor（与训练时相同）
#             local_mem = self.local_constructor(
#                 chunk
#             )  # [batch, num_local_slots, hidden_size]
#             local_memories.append(local_mem)

#         # Step 1.2: 聚合成全局记忆
#         all_local_memories = torch.cat(
#             local_memories, dim=1
#         )  # [batch, num_chunks*num_local_slots, hidden_size]
#         higher_global = self.global_integrator(
#             all_local_memories
#         )  # [batch, global_slots, hidden_size]
#         num_local_slots = higher_global.shape[1]

#     # ========================================================================
#     # Part 2: Full Attention 阶段（不分 chunk，一次性处理）
#     # ========================================================================

#     # Step 2.1: 标准 Q/K/V 投影（整个序列）
#     query_states = (
#         self.q_proj(hidden_states)
#         .view(bsz, q_len, self.num_heads, self.head_dim)
#         .transpose(1, 2)
#     )  # [batch, num_heads, seq_len, head_dim]

#     key_states = (
#         self.k_proj(hidden_states)
#         .view(bsz, q_len, self.num_key_value_heads, self.head_dim)
#         .transpose(1, 2)
#     )  # [batch, num_kv_heads, seq_len, head_dim]

#     value_states = (
#         self.v_proj(hidden_states)
#         .view(bsz, q_len, self.num_key_value_heads, self.head_dim)
#         .transpose(1, 2)
#     )  # [batch, num_kv_heads, seq_len, head_dim]

#     # Step 2.2: 处理 past_key_value（KV cache for generation）
#     kv_seq_len = key_states.shape[-2]
#     if past_key_value is not None:
#         kv_seq_len += past_key_value[0].shape[-2]

#     # Step 2.3: 应用 RoPE（旋转位置编码）到序列 Q/K
#     # 注意：全局记忆不需要 RoPE（位置无关）
#     if position_ids is not None:
#         max_pos = position_ids.max().item() + 1
#         rope_seq_len = max(kv_seq_len, max_pos)
#     else:
#         rope_seq_len = kv_seq_len

#     cos, sin = self.rotary_emb(value_states, seq_len=rope_seq_len)
#     query_states, key_states = apply_rotary_pos_emb(
#         query_states, key_states, cos, sin, position_ids
#     )

#     # Step 2.4: 处理 past_key_value（用于生成）
#     if past_key_value is not None:
#         key_states = torch.cat([past_key_value[0], key_states], dim=2)
#         value_states = torch.cat([past_key_value[1], value_states], dim=2)

#     past_key_value = (key_states, value_states) if use_cache else None

#     # Step 2.5: Repeat K/V heads if needed (GQA)
#     key_states = repeat_kv(key_states, self.num_key_value_groups)
#     value_states = repeat_kv(value_states, self.num_key_value_groups)

#     # Step 2.6: 如果有全局记忆，拼接到序列前面
#     if higher_global is not None:
#         # 全局记忆的 Q/K/V 投影
#         global_q = (
#             self.q_proj(higher_global)
#             .view(bsz, num_local_slots, self.num_heads, self.head_dim)
#             .transpose(1, 2)
#         )  # [batch, num_heads, num_local_slots, head_dim]

#         global_k = (
#             self.k_proj(higher_global)
#             .view(bsz, num_local_slots, self.num_key_value_heads, self.head_dim)
#             .transpose(1, 2)
#         )  # [batch, num_kv_heads, num_local_slots, head_dim]

#         global_v = (
#             self.v_proj(higher_global)
#             .view(bsz, num_local_slots, self.num_key_value_heads, self.head_dim)
#             .transpose(1, 2)
#         )  # [batch, num_kv_heads, num_local_slots, head_dim]

#         # Repeat K/V heads for global memory
#         global_k = repeat_kv(global_k, self.num_key_value_groups)
#         global_v = repeat_kv(global_v, self.num_key_value_groups)

#         # 拼接：[global_memory, sequence]
#         query_states = torch.cat(
#             [global_q, query_states], dim=2
#         )  # [batch, nh, num_slots+seq_len, hd]
#         key_states = torch.cat([global_k, key_states], dim=2)
#         value_states = torch.cat([global_v, value_states], dim=2)

#     # ========================================================================
#     # Part 3: Flash Attention（一次性，不分 chunk）
#     # ========================================================================

#     # 准备 attention mask
#     if attention_mask is not None:
#         if higher_global is not None:
#             # 为全局记忆添加 mask（全 1，始终可见）
#             global_mask = torch.ones(
#                 bsz,
#                 num_local_slots,
#                 dtype=attention_mask.dtype,
#                 device=attention_mask.device,
#             )
#             key_padding_mask = torch.cat([global_mask, attention_mask], dim=1)
#         else:
#             key_padding_mask = attention_mask
#     else:
#         key_padding_mask = None

#     # 转换为 flash attention 格式
#     query_states = query_states.transpose(1, 2)  # [batch, seq, num_heads, head_dim]
#     key_states = key_states.transpose(1, 2)
#     value_states = value_states.transpose(1, 2)

#     # 调用 Flash Attention
#     if key_padding_mask is not None:
#         # Unpad input for flash attention
#         query_states = rearrange(query_states, "b s h d -> b s (h d)")
#         key_states = rearrange(key_states, "b s h d -> b s (h d)")
#         value_states = rearrange(value_states, "b s h d -> b s (h d)")

#         qkv = torch.stack(
#             [query_states, key_states, value_states], dim=2
#         )  # [b, s, 3, (h d)]
#         qkv = rearrange(qkv, "b s three (h d) -> b s three h d", h=self.num_heads)

#         qkv_unpad, indices, cu_q_lens, max_s = unpad_input(qkv, key_padding_mask)
#         qkv_unpad = rearrange(qkv_unpad, "nnz three h d -> nnz three h d")

#         output_unpad = flash_attn_varlen_qkvpacked_func(
#             qkv_unpad, cu_q_lens, max_s, 0.0, softmax_scale=None, causal=True
#         )

#         output = rearrange(
#             pad_input(
#                 rearrange(output_unpad, "nnz h d -> nnz (h d)"),
#                 indices,
#                 bsz,
#                 query_states.shape[1],
#             ),
#             "b s (h d) -> b s h d",
#             h=self.num_heads,
#         )
#     else:
#         # No padding, use regular flash attention
#         output = flash_attn_func(
#             query_states, key_states, value_states, causal=True, softmax_scale=None
#         )  # [batch, seq, num_heads, head_dim]

#     # ========================================================================
#     # Part 4: 输出处理
#     # ========================================================================

#     # 如果有全局记忆，移除输出中的全局记忆部分（只保留序列输出）
#     if higher_global is not None:
#         output = output[
#             :, num_local_slots:, :, :
#         ]  # [batch, seq_len, num_heads, head_dim]

#     # 重塑并投影
#     output = output.reshape(bsz, q_len, self.num_heads * self.head_dim)
#     output = self.o_proj(output)

#     return output, None, past_key_value


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

    # ✅ SFT版本：处理不规则序列长度
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


# 推理专用版本：正确处理 KV cache 中的 past tokens
def _prepare_decoder_attention_mask_inference(
    self, attention_mask, input_shape, inputs_embeds, past_key_values_length
):
    """
    推理时的 attention mask 准备函数。

    与训练版本的关键区别：
    - 当有 KV cache 时（past_key_values_length > 0），需要为 cached tokens 添加 True mask
    - 这样 Flash Attention 才能正确 attend 到所有 cached tokens

    Args:
        attention_mask: [bsz, seq_len] 当前 token 的 mask
        input_shape: (bsz, seq_len)
        inputs_embeds: 输入 embeddings
        past_key_values_length: KV cache 中已有的 token 数量

    Returns:
        attention_mask: [bsz, past_len + seq_len] 扩展后的 mask
    """
    # [bsz, seq_len]
    if past_key_values_length > 0 and attention_mask is not None:
        # 为 KV cache 中的 tokens 添加 True mask（它们都是有效的）
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

    # 如果 mask 全为 True，返回 None 以使用更快的 Flash Attention 路径
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


# Flash attention 推理版（不拆分 group，支持 past KV）
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
# HiCI SFT 推理专用函数 (用于 LongBench 等评估)
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
    HiCI SFT 推理函数 - 用于 LongBench 等评估任务.

    推理流程：
    1. Prefill 阶段 (q_len > 1, past_key_value is None)：
       - Uses HiCI hierarchical attention consistent with training
       - group_size_ratio=1 (整个序列作为一个 chunk)
       - Mode 3: Q=[chunk], K/V=[higher_global, local, chunk]

    2. Decode 阶段 (q_len == 1 或 past_key_value is not None)：
       - 使用标准 attention + KV cache
       - HiCI information is already fused into token representations via prefill
    """
    bsz, q_len, hidden_size = hidden_states.size()

    # ========== Decode 阶段：使用 Flash Attention（与原始 LongLoRA 推理一致）==========
    if q_len == 1 or past_key_value is not None:
        kv_heads = getattr(self, "num_key_value_heads", self.num_heads)

        # Flash Attention 需要 [bsz, seq_len, num_heads, head_dim] 格式
        q = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(bsz, q_len, kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(bsz, q_len, kv_heads, self.head_dim)

        kv_seq_len = k.shape[1]
        past_kv_len = 0
        if past_key_value is not None:
            past_kv_len = past_key_value[0].shape[2]
            kv_seq_len += past_kv_len

        # 注意：HuggingFace 传入的 position_ids 已经是基于 token 数量的正确位置
        # 不需要调整！（之前错误地假设 HF 用 KV cache 长度）
        if INCLUDE_HICI_IN_KV_CACHE and not hasattr(self, "_hici_decode_printed"):
            layer_idx = getattr(self, "layer_idx", 0)
            if layer_idx == 0:
                prefix_len_in_cache = getattr(self, "_hici_cache_prefix_len", 0)
                print(f"\n{'=' * 60}")
                print(f"[HiCI Decode 诊断] Layer 0, 第一次 Decode")
                print(f"  HF 传入 position_ids    : {position_ids.item()}")
                print(f"  prefix_len_in_cache     : {prefix_len_in_cache}")
                print(f"  past_kv_len (KV cache)  : {past_kv_len}")
                print(f"  kv_seq_len (含新token)  : {kv_seq_len}")
                print(f"  说明: HF position_ids 已经是正确的 token 位置，不需要调整")
                print(f"{'=' * 60}\n")
            self._hici_decode_printed = True

        # 使用与 forward_flashattn_inference 相同的 RoPE 应用方式
        cos_sin = self.rotary_emb(v, seq_len=kv_seq_len)
        q, k = apply_rotary_pos_emb_inference(q, k, cos_sin, position_ids)

        # 拼接 KV cache
        if past_key_value is not None:
            # past_key_value 存储格式: [bsz, num_heads, seq_len, head_dim]
            # Flash Attention 需要: [bsz, seq_len, num_heads, head_dim]
            k = torch.cat([past_key_value[0].transpose(1, 2), k], dim=1)
            v = torch.cat([past_key_value[1].transpose(1, 2), v], dim=1)

        # 更新 KV cache（存回 [bsz, num_heads, seq_len, head_dim] 格式）
        past_key_value = (k.transpose(1, 2), v.transpose(1, 2)) if use_cache else None

        # 关键修复：当 KV cache 包含 memory 时，需要扩展 attention_mask
        if INCLUDE_HICI_IN_KV_CACHE and hasattr(self, "_hici_cache_prefix_len"):
            prefix_len = getattr(self, "_hici_cache_prefix_len", 0)
            if prefix_len > 0 and attention_mask is not None:
                # attention_mask 的长度不包括 memory，但 k/v 的长度包括
                # 需要在前面添加 memory 部分的 mask（全为 1，因为 memory 始终有效）
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
                            f"[HiCI Decode] Extended attention_mask: added {prefix_len} HiCI slots, new shape={attention_mask.shape}"
                        )
                    self._hici_mask_fix_printed = True

        # 使用 Flash Attention（与原始 LongLoRA 推理一致）
        # 诊断：检查 attention_mask 状态
        if not hasattr(self, "_hici_attn_branch_printed"):
            layer_idx = getattr(self, "layer_idx", 0)
            if layer_idx == 0:
                print(
                    f"\n[HiCI Decode 诊断] attention_mask is None: {attention_mask is None}"
                )
                print(f"  q.shape: {q.shape}, k.shape: {k.shape}, v.shape: {v.shape}")
                if attention_mask is not None:
                    print(f"  attention_mask.shape: {attention_mask.shape}")
            self._hici_attn_branch_printed = True

        if attention_mask is None:
            # 无 padding，直接使用 flash_attn_func
            output = flash_attn_func(
                q, k, v, 0.0, softmax_scale=None, causal=True
            ).view(bsz, q_len, -1)
        else:
            # 有 padding mask，使用 flash_attn_varlen_kvpacked_func
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

    # ========== Prefill 阶段 ==========
    global _HICI_INFERENCE_PRINTED, _HICI_GROUP_PRINTED, _HICI_CACHE_PRINTED

    # 测试模式：关闭 HiCI，使用标准 Flash Attention
    if DISABLE_HICI_IN_PREFILL:
        if not _HICI_INFERENCE_PRINTED:
            layer_idx = getattr(self, "layer_idx", 0)
            if layer_idx == 0:
                print("\n" + "=" * 80)
                print("HiCI SFT Inference - DISABLE_HICI_IN_PREFILL=True")
                print("Using standard Flash Attention (no HiCI)")
                print("=" * 80 + "\n", flush=True)
                _HICI_INFERENCE_PRINTED = True

        # 使用标准 Flash Attention（与 LongLoRA forward_flashattn_inference 一致）
        kv_heads = getattr(self, "num_key_value_heads", self.num_heads)

        q = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(bsz, q_len, kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(bsz, q_len, kv_heads, self.head_dim)

        cos_sin = self.rotary_emb(v, seq_len=q_len)
        q, k = apply_rotary_pos_emb_inference(q, k, cos_sin, position_ids)

        # 保存 KV cache
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

    # ========== Prefill 阶段：Hierarchical Memory ==========
    # 打印配置（使用全局变量，只打印一次）
    if not _HICI_INFERENCE_PRINTED:
        layer_idx = getattr(self, "layer_idx", 0)
        if layer_idx == 0:
            print("\n" + "=" * 80)
            print("HiCI SFT Inference Mode (LongBench)")
            print("=" * 80)
            print(f"  group_size_ratio  : {group_size_ratio}")
            print("  Prefill: HiCI hierarchical (higher_global + local_slots)")
            print("  Decode:  standard attention + KV cache")
            print("=" * 80 + "\n", flush=True)
            _HICI_INFERENCE_PRINTED = True

    # ==================== 分组逻辑选择 ====================
    # 取消注释你想要使用的方案，注释掉其他方案
    #
    # 方案A: 完全不分组（整个序列作为一组，num_groups=1）
    # 适用于：测试不分组的效果
    # -----------------------------------------------------
    group_size = q_len
    num_groups = 1
    # -----------------------------------------------------
    #
    # 方案B: 简单分组（使用 group_size_ratio，不能整除时 fallback 到 1 组）
    # 适用于：group_size_ratio = 1/4 且序列长度规整
    # -----------------------------------------------------
    # group_size = (
    #     int(q_len * group_size_ratio) if q_len * group_size_ratio >= 1 else q_len
    # )
    # if q_len % group_size != 0:
    #     group_size = q_len  # fallback: 整个序列为一组
    # num_groups = q_len // group_size
    # -----------------------------------------------------
    #
    # 方案C: 与训练一致的分组逻辑（找最接近的因数，确保能整除）
    # 适用于：需要严格匹配训练时的分组行为
    # -----------------------------------------------------
    # if q_len % 1024 == 0:
    #     group_size = int(q_len * group_size_ratio)
    # else:
    #     target_group_size = int(q_len * group_size_ratio)
    #     divisors = []
    #     for i in range(1, int(q_len**0.5) + 1):
    #         if q_len % i == 0:
    #             divisors.append(i)
    #             if i != q_len // i:
    #                 divisors.append(q_len // i)
    #     min_size = 10 if target_group_size >= 10 else 1
    #     max_size = q_len // 2 if q_len >= 20 else q_len
    #     valid_divisors = [d for d in divisors if min_size <= d <= max_size]
    #     if valid_divisors:
    #         group_size = min(valid_divisors, key=lambda x: abs(x - target_group_size))
    #     else:
    #         group_size = q_len
    # if q_len % group_size != 0:
    #     group_size = q_len
    # num_groups = q_len // group_size
    # -----------------------------------------------------

    if not _HICI_GROUP_PRINTED:
        layer_idx = getattr(self, "layer_idx", 0)
        if layer_idx == 0:
            print(
                f"[HiCI Inference] q_len={q_len}, group_size={group_size}, num_groups={num_groups}"
            )
            _HICI_GROUP_PRINTED = True

    # Reshape into chunks
    chunks = hidden_states.view(bsz, num_groups, group_size, hidden_size)

    # attention_mask 处理
    if attention_mask is not None and attention_mask.dim() == 2:
        chunk_masks = attention_mask.view(bsz, num_groups, group_size)
    else:
        chunk_masks = torch.ones(
            bsz, num_groups, group_size, dtype=torch.bool, device=hidden_states.device
        )

    # 提取局部记忆
    use_higher_global = True
    use_local_slots = True

    if hasattr(self, "local_constructor"):
        all_chunks = chunks.view(bsz * num_groups, group_size, hidden_size)
        mask_chunks = chunk_masks.view(bsz * num_groups, group_size)
        all_local_mems = self.local_constructor(all_chunks, mask_chunks)
        num_local_slots = all_local_mems.shape[1]
        local_memories = all_local_mems.view(
            bsz, num_groups, num_local_slots, hidden_size
        )
    else:
        num_local_slots = 0
        local_memories = None

    # 聚合到高层全局记忆
    if hasattr(self, "global_integrator") and local_memories is not None:
        higher_global = self.global_integrator(local_memories)
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

    # RoPE (只对 token 的 K 应用，memory 不需要位置编码)
    cos, sin = self.rotary_emb(value_states, seq_len=q_len)
    query_states, key_states = apply_rotary_pos_emb(
        query_states, key_states, cos, sin, position_ids
    )

    # 保留 repeat 之前的版本用于 KV cache (num_key_value_heads)
    key_states_for_cache = key_states
    value_states_for_cache = value_states

    # Repeat k/v heads for attention (num_heads)
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    # Project memories (保留 repeat 前后两个版本)
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

    # Local memories K/V (保留 repeat 前后两个版本)
    if local_memories is not None:
        lm_flat = local_memories.view(bsz * num_groups, num_local_slots, hidden_size)
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
        # 为 KV cache 重新整形 (group_size_ratio=1 时 num_groups=1)
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

    # ========== 构建 KV cache ==========
    if use_cache:
        if INCLUDE_HICI_IN_KV_CACHE and (
            hg_k_cache is not None or lm_k_cache is not None
        ):
            # 拼接 [higher_global, local_memory, tokens] 到 KV cache
            cache_components_k = []
            cache_components_v = []
            cache_prefix_len = 0

            if hg_k_cache is not None:
                cache_components_k.append(hg_k_cache)
                cache_components_v.append(hg_v_cache)
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

            # 记录 memory 长度，供 Decode 阶段调整 position_ids
            self._hici_cache_prefix_len = cache_prefix_len

            # 打印配置（使用全局变量，只打印一次）
            if not _HICI_CACHE_PRINTED:
                layer_idx = getattr(self, "layer_idx", 0)
                if layer_idx == 0:
                    print(f"\n{'=' * 60}")
                    print(f"[HiCI Prefill 诊断] INCLUDE_HICI_IN_KV_CACHE=True")
                    print(f"  higher_global slots     : {num_global_slots}")
                    print(f"  LocalConstructor slots  : {num_groups * num_local_slots}")
                    print(f"  prefix_len (总计)       : {cache_prefix_len}")
                    print(f"  token_len               : {q_len}")
                    print(f"  KV cache 总长度         : {full_key_cache.shape[2]}")
                    print(f"  _hici_cache_prefix_len  : {self._hici_cache_prefix_len}")
                    print(
                        f"  验证: {cache_prefix_len} + {q_len} = {cache_prefix_len + q_len} (应等于 {full_key_cache.shape[2]})"
                    )
                    print(f"{'=' * 60}\n")
                    _HICI_CACHE_PRINTED = True
        else:
            # 只保存 token 的 KV
            past_key_value = (key_states_for_cache, value_states_for_cache)
            self._hici_cache_prefix_len = 0

            if not _HICI_CACHE_PRINTED:
                layer_idx = getattr(self, "layer_idx", 0)
                if layer_idx == 0:
                    print(
                        f"[HiCI KV Cache] INCLUDE_HICI_IN_KV_CACHE=False, only tokens in cache"
                    )
                    _HICI_CACHE_PRINTED = True
    else:
        past_key_value = None

    # 构建 K/V: [higher_global, local, chunk]
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


# 全局开关：是否使用训练函数做推理（无 KV cache，但保证训练-推理一致性）
# False - 使用推理函数 + KV cache（高效，已修复 position_ids 和 attention_mask 问题）
# True  - 使用训练函数（慢，但保证一致性，用于调试）
USE_TRAINING_FUNCTION_FOR_INFERENCE = False


def replace_llama_attn_hici_inference(
    use_training_function: bool = None,
    include_hici_in_kv_cache: bool = None,
    disable_hici_in_prefill: bool = None,
):
    """
    替换 LlamaAttention.forward 为 HiCI SFT 推理函数.
    用于 LongBench 等评估任务.

    参数:
        use_training_function: 是否使用训练函数做推理（无 KV cache）
            True  - 直接使用训练函数（慢，但保证一致性）
            False - 使用推理函数 + KV cache（快）
            None  - 使用全局变量 USE_TRAINING_FUNCTION_FOR_INFERENCE

        include_hici_in_kv_cache: whether KV cache includes HiCI slots
            True  - KV cache = [hici_slots, tokens], Decode can access HiCI
            False - KV cache = [tokens] only, HiCI not accessible during Decode
            None  - 使用全局变量 INCLUDE_HICI_IN_KV_CACHE

        disable_hici_in_prefill: 是否在 Prefill 阶段禁用 HiCI
            True  - Prefill 使用标准 attention（用于调试）
            False - Prefill uses HiCI modules only (slots not in KV cache)
            None  - 使用全局变量 DISABLE_HICI_IN_PREFILL

    使用方法:
        # 方式1：使用 KV cache + memory（推荐，高效）
        replace_llama_attn_hici_inference(
            use_training_function=False,
            include_hici_in_kv_cache=True,
            disable_hici_in_prefill=False,
        )

        # 方式2：使用训练函数（慢，但保证一致性）
        replace_llama_attn_hici_inference(use_training_function=True)
    """
    global \
        USE_TRAINING_FUNCTION_FOR_INFERENCE, \
        INCLUDE_HICI_IN_KV_CACHE, \
        DISABLE_HICI_IN_PREFILL

    # 如果传入了参数，更新全局变量
    if use_training_function is not None:
        USE_TRAINING_FUNCTION_FOR_INFERENCE = use_training_function
    if include_hici_in_kv_cache is not None:
        INCLUDE_HICI_IN_KV_CACHE = include_hici_in_kv_cache
    if disable_hici_in_prefill is not None:
        DISABLE_HICI_IN_PREFILL = disable_hici_in_prefill

    print("=" * 80)
    print("Replacing LlamaAttention.forward with HiCI SFT Inference")
    print("=" * 80)
    print(f"  group_size_ratio                    : {group_size_ratio}")
    print(
        f"  USE_TRAINING_FUNCTION_FOR_INFERENCE : {USE_TRAINING_FUNCTION_FOR_INFERENCE}"
    )
    print(f"  INCLUDE_HICI_IN_KV_CACHE          : {INCLUDE_HICI_IN_KV_CACHE}")
    print(f"  DISABLE_HICI_IN_PREFILL             : {DISABLE_HICI_IN_PREFILL}")
    print("=" * 80)

    if USE_TRAINING_FUNCTION_FOR_INFERENCE:
        # 方案：直接使用训练函数做推理
        # 优点：保证训练-推理完全一致
        # 缺点：无 KV cache，每次 generate 都重新计算整个序列
        print("\n*** 使用训练函数 forward_flashattn_hierarchical 做推理 ***")
        print("*** 无 KV cache，保证训练-推理一致性 ***\n")

        # 使用推理版本的 mask 函数
        transformers.models.llama.modeling_llama.LlamaModel._prepare_decoder_attention_mask = _prepare_decoder_attention_mask_inference
        # 直接使用训练函数！
        transformers.models.llama.modeling_llama.LlamaAttention.forward = (
            forward_flashattn_hierarchical
        )
    else:
        # 原方案：使用专门的推理函数（有 KV cache）
        print("\n*** 使用推理函数 forward_hici_sft_inference ***")
        print("*** 有 KV cache，但可能有训练-推理不一致问题 ***\n")

        # 关键修复：使用推理版本的 mask 函数，正确处理 KV cache
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
    use_optimized_plus=True,  # 新增：使用改进版（复用 K/V 投影）
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
            - False: forward_flashattn (HiCI chunked attention, same as training)
        inference: Whether in inference mode (default: False)
        eval_mode: Evaluation mode selection (default: None)
            - None: Use use_full parameter for backward compatibility
            - "chunked": HiCI chunked attention (same as training)
            - "full": Full attention without HiCI
            - "full_hierarchical": Full attention + HiCI (eval mode 3)
        use_optimized: Whether to use optimized forward function (default: True)
            - True: Use forward_flashattn_optimized (合并投影 + 向量化mask)
            - False: Use forward_flashattn (原始版本)
        use_optimized_plus: Whether to use the improved version (default: False)
            - True: Use forward_flashattn_optimized_plus (复用 LLaMA K/V 投影)
            - 需要配合 register_hici_to_model(use_flash_plus=True) 使用
        use_optimized_plus_norope: Whether to use the experimental no-RoPE version (default: False)
            - 注意：SFT版本暂不支持此功能，如设置为True会回退到use_optimized_plus
        use_hierarchical_forward: Whether to use hierarchical forward (default: False)
            - True: Use forward_flashattn_hierarchical (LocalConstructor + GlobalIntegrator)
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
                # 评估方式3: Full Attention + Hierarchical Memory
                transformers.models.llama.modeling_llama.LlamaAttention.forward = (
                    forward_flashattn_full_hierarchical
                )
                if rank == 0:
                    print(f"   eval_mode: {eval_mode}")
            elif eval_mode == "full" or (eval_mode is None and use_full):
                # 评估方式1: Full Attention without memory
                transformers.models.llama.modeling_llama.LlamaAttention.forward = (
                    forward_flashattn_full
                )
                if rank == 0:
                    print(f"   调用函数: forward_flashattn_full  原本的评估方式")
            else:
                # 训练/评估方式2: Chunked attention with memory
                if use_hierarchical_forward:
                    # 分层记忆版本：局部摘要记忆 + 全局高层记忆
                    transformers.models.llama.modeling_llama.LlamaAttention.forward = (
                        forward_flashattn_hierarchical
                    )
                    if rank == 0:
                        print(
                            "   🧪 Using forward_flashattn_hierarchical (LocalConstructor + GlobalIntegrator)"
                        )
                elif use_optimized_plus_norope:
                    # 实验版：SFT版本暂不支持，回退到use_optimized_plus
                    if rank == 0:
                        warnings.warn(
                            "⚠️  use_optimized_plus_norope is not supported in SFT version, "
                            "falling back to use_optimized_plus"
                        )
                    transformers.models.llama.modeling_llama.LlamaAttention.forward = (
                        forward_flashattn_optimized_plus
                    )
                    if rank == 0:
                        print(
                            "   ⚡ Using forward_flashattn_optimized_plus (reuse K/V projections)"
                        )
                elif use_optimized_plus:
                    # 改进版：复用 LLaMA K/V 投影，先投影再调用 LocalConstructor
                    # 需要配合 LocalConstructorFlashPlus 使用
                    transformers.models.llama.modeling_llama.LlamaAttention.forward = (
                        forward_flashattn_optimized_plus
                    )
                    if rank == 0:
                        print(
                            "   ⚡ Using forward_flashattn_optimized_plus (reuse K/V projections)"
                        )
                elif use_optimized:
                    # 优化版本：合并投影 + 向量化mask + 预分配tensor
                    transformers.models.llama.modeling_llama.LlamaAttention.forward = (
                        forward_flashattn_optimized
                    )
                    if rank == 0:
                        print(
                            "   ⚡ Using forward_flashattn_optimized (merged projections + vectorized mask)"
                        )
                else:
                    # 原始版本
                    transformers.models.llama.modeling_llama.LlamaAttention.forward = (
                        forward_flashattn
                    )
                    if rank == 0:
                        print(f"   调用函数: forward_flashattn  原本的训练方式")
    else:
        transformers.models.llama.modeling_llama.LlamaAttention.forward = (
            forward_noflashattn
        )


# 控制参数的地方
def register_hici_to_model(
    model,
    num_local_slots=16,
    global_slots=2,
    num_heads=32,
    use_bottleneck=True,
    bottleneck_dim=4096,  # zxy
    use_local_constructor=True,
    use_global_integrator=True,
    use_flash_plus=True,  # 新增：是否使用 LocalConstructorFlashPlus sh中控制
    use_local_constructor_flash: Optional[bool] = True,  # 是否使用 LocalConstructorFlash
    use_llama_init=False,  # 新增：方案C - 从 LLaMA 预训练权重初始化 Q/K/V 投影
    use_shared_compressor=True,  # 🆕 是否使用共享压缩层优化版（节省71%参数） 在此处修改
    shared_compress_dim=128,  # 🆕 共享压缩层的中间维度
):
    """
    Register HiCI modules (LocalConstructor, GlobalIntegrator) to each LlamaAttention layer.

    This MUST be called after model loading and before optimizer initialization!

    Args:
        model: LlamaForCausalLM or PeftModelForCausalLM
        num_local_slots: Number of Local Representation Slots (for LocalConstructor, default: 16)
        global_slots: Number of global context slots (for GlobalIntegrator, default: 16)
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
        register_hici_to_model(model, num_local_slots=16)

        # For hierarchical memory:
        register_hici_to_model(
            model,
            num_local_slots=16,  # local slots
            global_slots=16,       # higher-level global slots
            use_global_integrator=True
        )

        # 4. Setup LoRA (if needed)
        model = get_peft_model(model, lora_config)

        # 5. NOW initialize optimizer (will include global memory parameters)
        optimizer = torch.optim.AdamW(model.parameters(), lr=...)
    """
    # ✅ 只在 rank 0 打印，避免8个GPU同时打印导致混乱
    rank = dist.get_rank() if dist.is_initialized() else 0

    # ⚠️ 验证配置合法性
    if use_global_integrator and not use_local_constructor:
        if rank == 0:
            print("\n" + "=" * 80)
            print("❌ ERROR: Invalid Configuration!")
            print("=" * 80)
            print("use_global_integrator=True requires use_local_constructor=True")
            print(
                "Reason: HierarchicalAggregator needs local memories from LocalConstructor"
            )
            print()
            print("Fix: Set use_local_constructor=True, or set use_global_integrator=False")
            print("=" * 80 + "\n")
        raise ValueError(
            "Invalid configuration: use_global_integrator=True requires use_local_constructor=True. "
            "HierarchicalAggregator needs local memories from LocalConstructor to aggregate."
        )

    # 注册开始（简化打印，详细信息在结束时显示）
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

    # ✅ 获取模型的 dtype，确保记忆模块与模型精度一致
    model_dtype = llama_model.embed_tokens.weight.dtype
    rank = dist.get_rank() if dist.is_initialized() else 0
    if rank == 0:
        print(f"   Model dtype: {model_dtype}")

    # ✅ 获取预训练嵌入权重用于初始化记忆模块（方案 2）
    embed_weight = llama_model.embed_tokens.weight.data  # [vocab_size, hidden_size]

    # Register modules to each attention layer
    for layer_idx, layer in enumerate(llama_model.layers):
        attn = layer.self_attn
        attn.layer_idx = layer_idx  # Important for layer identification

        # Module 1: LocalConstructor (always register)
        # ✅ 使用预训练嵌入初始化（方案 2）
        # ✅ 使用 Flash Attention 实现高效的 cross-attention（支持 100k+ 序列）
        if use_local_constructor:
            if use_flash_plus:
                # 使用改进版：复用 LLaMA 的 K/V 投影，只有自己的 Q 投影
                # 配合 forward_flashattn_optimized_plus 使用
                # 注意：LocalConstructorFlashPlus 只接受 4 个参数
                attn.local_constructor = LocalConstructorFlashPlus(
                    hidden_size=hidden_size,
                    num_local_slots=num_local_slots,
                    num_heads=num_heads,
                    init_from_embeddings=embed_weight,
                ).to(model_dtype)
            elif use_local_constructor_flash:
                # 原版：有独立的 Q/K/V 投影
                # 配合 forward_flashattn_optimized 使用
                # 注意：LocalConstructorFlash 只接受 4 个参数，不支持 bottleneck
                attn.local_constructor = LocalConstructorFlash(
                    hidden_size=hidden_size,
                    num_local_slots=num_local_slots,
                    num_heads=num_heads,
                    init_from_embeddings=embed_weight,
                ).to(model_dtype)
            else:
                # 基础版：无优化的 cross-attention 单头 无flash attn
                # 配合 forward_flashattn 使用 LocalConstructorMulti
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
                    init_from_llama_attn=attn if use_llama_init else None,
                    use_bottleneck=use_bottleneck,
                    bottleneck_dim=bottleneck_dim,
                ).to(model_dtype)
        # Module 2: Hierarchical Aggregator (optional)
        if use_global_integrator:
            # 🔽 选择使用原版还是共享压缩层优化版
            if use_shared_compressor:
                # 🆕 共享压缩层优化版（方案B+ - 参数节省71%）
                # 参数量：4.0M/layer（原版13.7M的29%）
                # 关键优化：5个统计量共享同一个4096→128的压缩层，再融合到512维
                # 理论依据：参数共享 + 归纳偏置（类似CNN的weight sharing）
                attn.global_integrator = GlobalIntegratorShared(
                    hidden_size=hidden_size,
                    global_slots=global_slots,
                    compress_dim=512,  # 最终压缩维度
                    shared_compress_dim=shared_compress_dim,  # 共享压缩层的中间维度（默认128）
                    init_from_embeddings=embed_weight,
                    use_high_norm_init=True,
                ).to(model_dtype)
            else:
                # 原版：GlobalIntegrator（方案B - ICML推荐：统计量 + Lightweight Attention）
                # 基于Information Bottleneck + Predictive Coding + RG理论
                # 参数量：13.7M/layer
                # 稳定性：✅✅✅ 极高（attention 在 5×512 低维空间）
                attn.global_integrator = GlobalIntegrator(
                    hidden_size=hidden_size,
                    global_slots=global_slots,
                    compress_dim=512,  # ← 统计量压缩维度  此处控制
                    num_heads=num_heads,
                    init_from_embeddings=embed_weight,
                    use_high_norm_init=True,  # ← 使用高范数初始化提升稳定性 此处控制
                ).to(model_dtype)  # ✅ 转换为模型的 dtype

            # 🆕 新方案：TemporalPerceiverGlobalMemory（Chunk-level Self-Attention）
            # 基于 Perceiver (ICML 2021) + Slot Attention (NeurIPS 2020) + Set Transformer (ICML 2019)
            # 让局部记忆之间直接进行 self-attention，然后用 learnable queries 提取全局表示
            # 参数量：~0.02M/layer (共享 Q/K/V) 或 ~50.4M/layer (独立 Q/K/V)
            # 创建共享投影字典（与局部记忆模块共享，节省参数）
            # qkv_projections = (
            #     {
            #         "q_proj": attn.global_memory.q_proj,
            #         "k_proj": attn.global_memory.k_proj,
            #         "v_proj": attn.global_memory.v_proj,
            #     }
            #     if use_local_constructor
            #     else None
            # )

            # attn.hierarchical_aggregator = TemporalPerceiverGlobalMemory(
            #     hidden_size=hidden_size,
            #     global_slots=2,
            #     num_chunks=num_chunks,  # 向后兼容（实际不使用）
            #     max_chunks=num_chunks,
            #     qkv_projections=qkv_projections,  # ← 共享投影（省参数）
            #     use_temporal_encoding=True,  # ← 保留 chunks 的时序信息
            #     ema_decay=0.5,  # ← Predictive Coding 风格的长期记忆
            #     init_from_embeddings=embed_weight,  # ← 从预训练嵌入初始化queries和EMA
            #     use_high_norm_init=True,  # ← 使用高范数初始化提升稳定性
            # )

            # # 🔽 旧代码：StableStatisticalAggregator（已弃用，参数量过大）
            # attn.hierarchical_aggregator = StableStatisticalAggregator(
            #     hidden_size=hidden_size,
            #     local_slots=num_local_slots,
            #     global_slots=global_slots,
            #     use_bottleneck=use_bottleneck,
            #     bottleneck_dim=bottleneck_dim,
            #     init_from_embeddings=embed_weight
            # )

            # 🔽 旧代码：基于Attention的聚合器（已注释，训练不稳定）
            # attn.hierarchical_aggregator = HierarchicalMemoryAggregatorSingleHead(
            #     hidden_size=hidden_size,
            #     local_slots=num_local_slots,
            #     global_slots=global_slots,
            #     use_bottleneck=use_bottleneck,
            #     bottleneck_dim=bottleneck_dim,
            #     init_from_embeddings=embed_weight
            # )

        # 移除冗余的循环打印，在函数结束时统一显示注册结果

    # Verify registration
    total_params = sum(p.numel() for p in model.parameters())

    # 🔥 修复参数计数：使用 named_parameters() 自动去重（避免重复计数共享参数）
    # 之前使用 module.parameters() 会重复计数 hierarchical_aggregator 引用的 q_proj/k_proj/v_proj
    # 现在使用 named_parameters() 遍历整个模型，同一个参数对象只计数一次
    local_constructor_params = 0
    aggregator_params = 0

    if use_local_constructor or use_global_integrator:
        # 遍历整个模型的参数（自动去重）
        for name, param in model.named_parameters():
            if "local_constructor" in name:
                # 属于 GlobalMemory 的参数（包括 q_proj/k_proj/v_proj）
                local_constructor_params += param.numel()
            elif "global_integrator" in name:
                # 只属于 Hierarchical 的参数（不包括引用的 q_proj/k_proj/v_proj）
                # 只有 global_queries 和 temporal_encoding
                aggregator_params += param.numel()

    # 统一的注册完成总结（替代之前的三处重复打印）
    if rank == 0:
        print()
        print("=" * 80)
        print("✅ HiCI Module Registration Complete")
        print("=" * 80)

        # 模型总参数
        print(f"Model: {total_params:,} params ({total_params / 1e9:.2f}B)")
        print(f"Layers: {len(llama_model.layers)}")

        # 注册的模块和参数统计
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
#     num_local_slots=16,
#     recurrence_size=256
# ):
#     # region
#     """
#     Register LocalConstructor to each LlamaAttention layer.

#     This MUST be called after model loading and before optimizer initialization!

#     Args:
#         model: LlamaForCausalLM or PeftModelForCausalLM
#         num_local_slots: Number of learnable memory slots per layer (default: 16)
#         recurrence_size: Number of tokens to carry from previous chunk (default: 256)

#     Example usage in fine-tune.py:
#         # 1. Load model
#         model = transformers.AutoModelForCausalLM.from_pretrained(...)

#         # 2. Replace attention mechanism
#         replace_llama_attn(use_flash_attn=True, use_full=False)

#         # 3. Register global memory (BEFORE optimizer!)
#         register_hici_to_model(model, num_local_slots=16, recurrence_size=256)

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
#         attn.global_memory = LocalConstructor(
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
#         for p in layer.self_attn.global_memory.parameters()
#     )

#     print(f"\n✅ Global memory registration complete!")
#     print(f"   Total model params: {total_params:,}")
#     print(f"   Global memory params: {memory_params:,} "
#           f"({memory_params/total_params*100:.2f}% of total)")
#     print(f"   Per-layer config: {num_local_slots} slots × {hidden_size}D = "
#           f"{num_local_slots * hidden_size * 4:,} params × {len(llama_model.layers)} layers")
#     print(f"   Recurrence size: {recurrence_size} tokens (carried from previous chunk)\n")
