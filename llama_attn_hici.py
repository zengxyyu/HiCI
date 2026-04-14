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
# 🔥 混合分组训练配置
# ============================================================================
# MIXED_GROUP_TRAINING = True: 每个 batch 随机选择 2/4/8 组
# MIXED_GROUP_TRAINING = False: 使用固定的 group_size_ratio
MIXED_GROUP_TRAINING = False
GROUP_SIZE_RATIOS = [1 / 2, 1 / 4, 1 / 8]  # 对应 2组, 4组, 8组

group_size_ratio = 1 / 4  # 默认值（MIXED_GROUP_TRAINING=False 时使用）

# ============================================================================
# 🎯 固定 segment_size 模式（用于评估，匹配训练时的分组大小）
# ============================================================================
# USE_FIXED_SEGMENT_SIZE = True: 使用固定的 segment_size（评估时推荐）
# USE_FIXED_SEGMENT_SIZE = False: 使用 group_size_ratio（训练时使用）
USE_FIXED_SEGMENT_SIZE = False
FIXED_SEGMENT_SIZE = 1024  # 固定每组大小（tokens）

# ============================================================================
# 🔥 Full Attention + Memory 模式（用于验证 memory 模块是否正常工作）
# ============================================================================
# USE_FULL_ATTN_WITH_MEMORY = True: 不分组，但仍然使用 memory
#   - 对整个输入提取 local memory -> 聚合成 global memory
#   - 所有 tokens attend 到 [global_memory, all_tokens]
#   - 效果应该和原始 LLaMA 类似，只是多了 memory context
# USE_FULL_ATTN_WITH_MEMORY = False: 使用分组 attention（原始行为）
USE_FULL_ATTN_WITH_MEMORY = True

# 全局变量：确保同一个 forward pass 中所有层使用相同的分组
_mixed_group_current_ratio = None
_mixed_group_call_count = 0  # 用于检测新的 forward pass
rank = dist.get_rank() if dist.is_initialized() else 0

# ============================================================================
# 🎨 Attention可视化配置 (推理时使用)
# ============================================================================
# COLLECT_ATTENTION_FOR_VIZ = True: 收集attention weights用于可视化
# 注意: 仅在推理时开启，会略微增加计算量
# ============================================================================
# 🔒 因果记忆模式 (Causal Memory Mode)
# ============================================================================
# "none"           - 原始行为：所有 segment 共享同一个 G（非因果，R1 质疑的问题）
# "causal_gi"      - 方案一：segment_i 使用 G_i=Agg(L_1..L_i) 和 L_i
#                    G 因果，L_i 有 bounded intra-segment leakage（bottleneck 压缩）
# "causal_shift"   - 方案二：segment_i 使用 G_{i-1}=Agg(L_1..L_{i-1}) 和 L_{i-1}
#                    完全因果，零泄露；segment_0 没有 G 和 L
# "causal_shift_g" - 方案三：segment_i 仅使用 G_{i-1}=Agg(L_1..L_{i-1})，不拼接 L
#                    完全因果，零泄露；避免 L_{i-1} 语义不对等
# "causal_gi_gonly" - 方案五：segment_i 使用 G_i=Agg(L_1..L_i)，不拼接 L_i
#                    G 因果（含当前段），L 不直接暴露；双重瓶颈抑制泄露 不shift
CAUSAL_MEMORY_MODE = "none"

COLLECT_ATTENTION_FOR_VIZ = False

# 全局收集器 - 存储各层的attention统计
attention_visualizer = {
    "enabled": False,
    "layer_attn_to_global": [],  # 各层对global的平均attention比例
    "layer_attn_to_local": [],  # 各层对local的平均attention比例
    "layer_attn_to_tokens": [],  # 各层对tokens的平均attention比例
    "segment_attention_maps": [],  # 段内attention热力图 (可选，仅保存少量)
    "num_global_slots": 0,
    "num_local_slots": 0,
    "segment_len": 0,
}


def reset_attention_visualizer():
    """重置收集器，每次推理前调用"""
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
    """保存收集的attention统计到文件"""
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
    print(f"✅ Attention stats saved to {save_path}")
    if stats["segment_attention_maps"]:
        print(f"   包含 {len(stats['segment_attention_maps'])} 个attention heatmaps")


# 版本1 没有多头的最初版本
class LocalConstructor(nn.Module):
    """
    Learnable global memory for capturing document-level context.

    This module is registered as a sub-module of LlamaAttention, ensuring:
    1. Parameters are properly registered in model.parameters()
    2. Optimizer tracks and updates these parameters
    3. Saved/loaded with model checkpoints

    Args:
        hidden_size: Model hidden dimension (e.g., 4096 for Llama-2-7B)
        num_local_slots: Number of learnable memory slots (default: 16)
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
            layer_idx = getattr(self, "layer_idx", 0)  # 获取当前层索引
            if rank == 0 and layer_idx == 0:
                print(
                    f"⚠️  版本1 LocalConstructor Fallback: Initialized memory_slots with std={std}"
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

        # Expand memory for batch
        memory = self.memory_slots.unsqueeze(0).expand(
            bsz, -1, -1
        )  # [bsz, num_slots, hidden_size]

        # Cross-attention: memory attends to full sequence
        Q_mem = self.q_proj(memory)  # [bsz, num_slots, hidden_size]
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
    Learnable global memory for capturing document-level context.

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
        num_local_slots: Number of learnable memory slots (default: 16)
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
        num_local_slots=8,
        num_heads=8,
        init_from_embeddings=None,
        init_from_llama_attn=None,
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

        # Learnable memory slots: [num_slots, hidden_size]
        # ✅ 方案 2: 从预训练嵌入初始化（最优策略）
        # if init_from_embeddings is not None:
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
        - Q: memory slots (无 padding), 长度固定为 num_slots
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
        memory = self.memory_slots.unsqueeze(0).expand(
            bsz, -1, -1
        )  # [bsz, num_slots, hidden_size]

        # Cross-attention projections: 直接投影到目标维度 (bottleneck or full)
        Q_mem = self.q_proj(memory)  # [bsz, num_slots, effective_dim]
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


# 版本2 加了flash attn和padding的实现 多头 独立的q kv投影
class LocalConstructorFlashOri(nn.Module):
    """
    Learnable global memory for capturing document-level context.

    使用 Flash Attention 实现高效的 cross-attention，支持：
    1. 超长序列（100k+）- 内存复杂度 O(N) 而不是 O(N²)
    2. 正确处理 padding - 使用 unpad_input 移除 padding tokens

    This module is registered as a sub-module of LlamaAttention, ensuring:
    1. Parameters are properly registered in model.parameters()
    2. Optimizer tracks and updates these parameters
    3. Saved/loaded with model checkpoints

    Args:
        hidden_size: Model hidden dimension (e.g., 4096 for Llama-2-7B)
        num_local_slots: Number of learnable memory slots (default: 16)
        num_heads: Number of attention heads (default: 32, for Flash Attention)
        init_from_embeddings: Optional pretrained embeddings for memory_slots initialization
        init_from_llama_attn: Optional LlamaAttention layer for Q/K/V projection initialization (方案C)
    """

    # 类变量：控制只打印一次初始化信息
    _init_msg_printed = False

    def __init__(
        self,
        hidden_size,
        num_local_slots=16,
        num_heads=32,
        init_from_embeddings=None,
        init_from_llama_attn=None,  # 新增：从 LLaMA Attention 层初始化 Q/K/V 投影
        use_bottleneck: Optional[bool] = True,  # 未使用
        bottleneck_dim: Optional[int] = 2048,  # 未使用
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
            if rank == 0 and not LocalConstructorFlashOri._init_msg_printed:
                print(
                    f"⚠️  LocalConstructorFlash Fallback: Initialized memory_slots with std={std}"
                )
                LocalConstructorFlashOri._init_msg_printed = True

        # Cross-attention projections for summarization
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        # ========== 方案C：从 LLaMA Attention 层初始化 Q/K/V 投影 ==========
        # 理论依据：
        # 1. LLaMA 的 q_proj/k_proj/v_proj 是一起预训练的，有内在的"配对关系"
        # 2. 从预训练权重初始化，Q 和 K 在同一个语义空间，初始 Q×K^T 有意义
        # 3. 投影仍然是独立的参数，可以继续微调（不是冻结的！）
        # 4. 这是 Warm Initialization，比随机初始化收敛更快
        if init_from_llama_attn is not None:
            rank = dist.get_rank() if dist.is_initialized() else 0
            layer_idx = getattr(self, "layer_idx", 0)  # 获取当前层索引
            with torch.no_grad():
                self.q_proj.weight.copy_(init_from_llama_attn.q_proj.weight)
                self.k_proj.weight.copy_(init_from_llama_attn.k_proj.weight)
                self.v_proj.weight.copy_(init_from_llama_attn.v_proj.weight)
            if rank == 0 and layer_idx == 0:
                print(
                    f"✅ [方案C] Initialized Q/K/V projections from LLaMA pretrained weights"
                )

    def forward(self, hidden_states, attention_mask=None):
        """
        Compute global context via Flash Attention cross-attention.

        使用 flash_attn_varlen_kvpacked_func 实现：
        - Q: memory slots (无 padding), 长度固定为 num_slots
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
        memory = self.memory_slots.unsqueeze(0).expand(
            bsz, -1, -1
        )  # [bsz, num_slots, hidden_size]

        # Cross-attention projections
        Q_mem = self.q_proj(memory)  # [bsz, num_slots, hidden_size]
        K_seq = self.k_proj(hidden_states)  # [bsz, seq_len, hidden_size]
        V_seq = self.v_proj(hidden_states)  # [bsz, seq_len, hidden_size]

        # Reshape for multi-head attention: [bsz, seqlen, num_heads, head_dim]
        Q_mem = Q_mem.view(bsz, self.num_local_slots, self.num_heads, self.head_dim)
        K_seq = K_seq.view(bsz, seq_len, self.num_heads, self.head_dim)
        V_seq = V_seq.view(bsz, seq_len, self.num_heads, self.head_dim)

        if attention_mask is not None:
            # region
            # ========== 方案 B: Flash Attention + unpad (正确处理 padding) ==========
            #
            # 原理：
            # 1. 用 unpad_input 移除 K/V 中的 padding tokens
            # 2. Q (memory slots) 没有 padding，每个 batch 样本都是 num_slots 个
            # 3. 使用 flash_attn_varlen_kvpacked_func 做变长 cross-attention
            #
            # 示例：
            #   batch 0: 有效长度 500, padding 524
            #   batch 1: 有效长度 800, padding 224
            #
            #   unpad 后 K/V: [500 + 800, num_heads, head_dim] = [1300, ...]
            #   cu_seqlens_kv = [0, 500, 1300]
            #
            #   Q: [bsz * num_slots, num_heads, head_dim] = [32, ...] (假设 num_slots=16, bsz=2)
            #   cu_seqlens_q = [0, 16, 32]

            # Pack K and V together: [bsz, seq_len, 2, num_heads, head_dim]
            # endregion
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

            # Flash Attention 变长 cross-attention
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


# 版本2的瓶颈版本 加了flash attn和padding的实现 多头 独立的q kv投影 + kqv瓶颈投影 已检查
# region ===========================================================================
class LocalConstructorFlash(nn.Module):
    """
    Learnable global memory for capturing document-level context.

    使用 Flash Attention 实现高效的 cross-attention，支持：
    1. 超长序列（100k+）- 内存复杂度 O(N) 而不是 O(N²)
    2. 正确处理 padding - 使用 unpad_input 移除 padding tokens

    This module is registered as a sub-module of LlamaAttention, ensuring:
    1. Parameters are properly registered in model.parameters()
    2. Optimizer tracks and updates these parameters
    3. Saved/loaded with model checkpoints

    Args:
        hidden_size: Model hidden dimension (e.g., 4096 for Llama-2-7B)
        num_local_slots: Number of learnable memory slots (default: 16)
        num_heads: Number of attention heads (default: 32, for Flash Attention)
        init_from_embeddings: Optional pretrained embeddings for memory_slots initialization
        init_from_llama_attn: Optional LlamaAttention layer for Q/K/V projection initialization (方案C)
    """

    # 类变量：控制只打印一次初始化信息
    _init_msg_printed = False

    def __init__(
        self,
        hidden_size,
        num_local_slots=16,
        num_heads=32,
        init_from_embeddings=None,
        init_from_llama_attn=None,  # 新增：从 LLaMA Attention 层初始化 Q/K/V 投影
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
            if rank == 0 and not LocalConstructorFlash._init_msg_printed:
                print(
                    f"⚠️  LocalConstructorFlash_bot Fallback: Initialized memory_slots with std={std}"
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
                print(f"✅ bottleneck_dim: {bottleneck_dim}")

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
        if init_from_llama_attn is not None:
            rank = dist.get_rank() if dist.is_initialized() else 0
            layer_idx = getattr(self, "layer_idx", 0)  # 获取当前层索引
            with torch.no_grad():
                self.q_proj.weight.copy_(init_from_llama_attn.q_proj.weight)
                self.k_proj.weight.copy_(init_from_llama_attn.k_proj.weight)
                self.v_proj.weight.copy_(init_from_llama_attn.v_proj.weight)
            if rank == 0 and layer_idx == 0:
                print(
                    f"✅ [方案C] Initialized Q/K/V projections from LLaMA pretrained weights"
                )

    def forward(self, hidden_states, attention_mask=None):
        """
        Compute global context via Flash Attention cross-attention.

        使用 flash_attn_varlen_kvpacked_func 实现：
        - Q: memory slots (无 padding), 长度固定为 num_slots
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
        memory = self.memory_slots.unsqueeze(0).expand(
            bsz, -1, -1
        )  # [bsz, num_slots, hidden_size]

        # Cross-attention projections: 直接投影到目标维度 (bottleneck or full)
        Q_mem = self.q_proj(memory)  # [bsz, num_slots, effective_dim]
        K_seq = self.k_proj(hidden_states)  # [bsz, seq_len, effective_dim]
        V_seq = self.v_proj(hidden_states)  # [bsz, seq_len, effective_dim]

        # Reshape for multi-head attention: [bsz, seqlen, num_heads, effective_head_dim]
        Q_mem = Q_mem.view(
            bsz, self.num_local_slots, self.num_heads, self.effective_head_dim
        )
        K_seq = K_seq.view(bsz, seq_len, self.num_heads, self.effective_head_dim)
        V_seq = V_seq.view(bsz, seq_len, self.num_heads, self.effective_head_dim)

        if attention_mask is not None:
            # region
            # ========== 方案 B: Flash Attention + unpad (正确处理 padding) ==========
            #
            # 原理：
            # 1. 用 unpad_input 移除 K/V 中的 padding tokens
            # 2. Q (memory slots) 没有 padding，每个 batch 样本都是 num_slots 个
            # 3. 使用 flash_attn_varlen_kvpacked_func 做变长 cross-attention
            #
            # 示例：
            #   batch 0: 有效长度 500, padding 524
            #   batch 1: 有效长度 800, padding 224
            #
            #   unpad 后 K/V: [500 + 800, num_heads, head_dim] = [1300, ...]
            #   cu_seqlens_kv = [0, 500, 1300]
            #
            #   Q: [bsz * num_slots, num_heads, head_dim] = [32, ...] (假设 num_slots=16, bsz=2)
            #   cu_seqlens_q = [0, 16, 32]

            # Pack K and V together: [bsz, seq_len, 2, num_heads, head_dim]
            # endregion
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

            # Flash Attention 变长 cross-attention
            # Q: [total_q, num_heads, effective_head_dim] where total_q = bsz * num_slots
            # KV: [total_kv, 2, num_heads, effective_head_dim] where total_kv = sum of valid lengths
            output_unpad = flash_attn_varlen_kvpacked_func(
                q_unpad,  # [bsz * num_slots, num_heads, effective_head_dim]
                kv_unpad,  # [total_valid_kv, 2, num_heads, effective_head_dim]
                cu_seqlens_q,  # [bsz + 1]
                cu_seqlens_kv,  # [bsz + 1]
                self.num_local_slots,  # max_seqlen_q (固定)
                max_seqlen_kv,  # max_seqlen_kv (batch 中最长的有效长度)
                dropout_p=0.0,
                softmax_scale=None,  # 默认 1/sqrt(head_dim)
                causal=False,  # Cross-attention 不需要 causal mask
            )
            # output_unpad: [bsz * num_slots, num_heads, effective_head_dim]

            # Reshape back: [bsz, num_slots, effective_dim]
            global_context = rearrange(
                output_unpad, "(b s) h d -> b s (h d)", b=bsz, s=self.num_local_slots
            )
        else:
            # ========== 无 padding，使用简单的 flash_attn_func ==========
            # 这是最高效的情况，直接使用 Flash Attention
            global_context = flash_attn_func(
                Q_mem,  # [bsz, num_slots, num_heads, effective_head_dim]
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


# endregion ===========================================================================


# 方法2   🆕 混合全局记忆 - ICML最佳方案（方案B：统计量 + Lightweight Attention）
class GlobalIntegrator_ori(nn.Module):
    """
    混合全局记忆 - 统计量 + Lightweight Attention（ICML推荐）

    设计哲学：
    1. ✅ 稳定性：统计量提供稳定的"锚点"（避免 attention 炸掉）
    2. ✅ 理论性：Attention 学习如何 refine 统计特征（非启发式）
    3. ✅ 高效性：Attention 在低维空间操作（5 × 512，极小！）
    4. ✅ 参数量：13.4M/layer × 32 = 0.43B（比纯统计量版本少 34%）

    理论对齐：
    - Information Bottleneck:
      • 统计量 = 粗粒化压缩（coarse-grained summary）
      • Attention = 学习的 sufficient statistics refinement
      • Capacity constraint via global_slots

    - Predictive Coding:
      • Global = high-level slow prior（EMA 慢更新）
      • Attention = adaptive routing of information

    - Renormalization Group:
      • Step 1: Statistical aggregation（粗粒化）
      • Step 2: Learned blocking transformation（精细化）

    两阶段设计：
        阶段1（稳定基础）：统计量压缩
            local_memories: [bsz, N, 4096]
            → 5种统计量: [mean, max, min, std, norm]
            → 分离压缩: 每个 4096 → 512
            → compressed_stats: [bsz, 5, 512]

        阶段2（理论提升）：Lightweight Attention
            global_queries: [global_slots, 512]  ← parameter
            compressed_stats: [bsz, 5, 512]
            → cross_attn in 512-dim space（极小！）
            → G_compressed: [bsz, global_slots, 512]
            → expand to 4096
            → G: [bsz, global_slots, 4096]

    关键优势：
    1. Attention 只在 5 个统计量 × 512 维上操作（极小规模，风险低）
    2. 统计量提供稳定的语义"锚点"
    3. Attention 学习如何组合这些统计量（非启发式）
    4. 即使 attention 出问题，统计量仍然能保底

    参数量分析：
        统计量压缩: 5 × (4096 × 512) = 10.5M
        Q/K/V 投影: 3 × (512 × 512) = 0.8M
        Expand: 512 × 4096 = 2.1M
        Total: ~13.4M/layer × 32 = 0.43B

    风险评估：
        方案A（纯统计量）: 风险 0%  ✅✅✅
        方案B（混合）:     风险 10% ✅✅  ← 推荐用于ICML
        方案C（pure attn）:风险 40% ⚠️

    论文叙述建议：
    "We propose a hybrid approach that combines deterministic statistical
     aggregation with learned attention-based refinement. First, we extract
     five statistical features (mean, max, min, std, normalized mean) from
     local memories and compress them to a 512-dim bottleneck space. Then,
     global learned queries attend over these compressed statistics to
     extract minimal sufficient statistics for document-level context.
     This design provides both stability (via deterministic statistics)
     and adaptivity (via learned attention)."
    """

    def __init__(
        self,
        hidden_size: int = 4096,
        global_slots: int = 4,  # .sh脚本控制
        compress_dim: int = 512,  # 统计量压缩维度 # 调用处控制
        local_slots: int = 16,  # 兼容参数
        use_bottleneck: bool = False,  # 兼容参数
        bottleneck_dim: int = 4096,  # 兼容参数
        init_from_embeddings=None,  # 调用处控制
        use_high_norm_init: Optional[bool] = True,  # 调用处控制
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_global = global_slots
        self.compress_dim = compress_dim
        self.use_high_norm_init = use_high_norm_init  # 保存配置

        # 阶段1：统计量分离压缩（和之前一样，稳定）
        # 有归一化 LayerNorm
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

        # 阶段2：Lightweight Attention（理论提升）
        # Global queries in compressed space  局部记忆使用torch.randn初始化
        self.global_queries = nn.Parameter(torch.zeros(global_slots, compress_dim))

        # Q/K/V 投影（都在 512 维空间）参数量：3 × (512 × 512) = 0.8M（很小！）
        self.q_proj = nn.Linear(compress_dim, compress_dim, bias=False)
        self.k_proj = nn.Linear(compress_dim, compress_dim, bias=False)
        self.v_proj = nn.Linear(compress_dim, compress_dim, bias=False)

        # 阶段3：扩展到 hidden_size
        # self.expand = nn.Linear(compress_dim, hidden_size)
        self.expand = nn.Linear(compress_dim, hidden_size, bias=False)
        # LLaMA 风格初始化（关键！）
        std_init = 0.02 / math.sqrt(compress_dim)  # ≈ 0.00088
        nn.init.normal_(self.expand.weight, mean=0.0, std=std_init)
        self.expand_scale = nn.Parameter(torch.tensor(0.1))  # 初始化为 0.1 新增

        # ✅ EMA Buffer（Predictive Coding - 作为长期先验）
        # 用途：为 global queries 提供稳定的初始化偏置
        self.register_buffer("ema_global", torch.zeros(1, global_slots, hidden_size))
        self.ema_decay = 0.95
        self.ema_weight = 0.1  # EMA 对 queries 的影响权重

        # 🚀 性能优化：缓存压缩后的 EMA（避免每次 forward 都重新计算）
        self.register_buffer(
            "ema_compressed_cache", torch.zeros(global_slots, compress_dim)
        )
        self._ema_cache_valid = False  # 标记缓存是否有效

        # 🚀 性能优化：缓存 compressor 引用（避免运行时类型检查）
        self._first_compressor = None  # 将在 _init_weights 中设置

        # 初始化
        self._init_weights(
            init_from_embeddings, use_high_norm_init=self.use_high_norm_init
        )

        # 打印信息
        layer_idx = getattr(self, "layer_idx", 0)  # 获取当前层索引
        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank == 0 and layer_idx == 0:
            total_params = sum(p.numel() for p in self.parameters())
            print(f"    ✅ GlobalIntegrator initialized (方案B - ICML推荐)")
            print(f"       - Design: Statistical Aggregation + Lightweight Attention")
            print(f"       - Global slots: {global_slots} (IB capacity)")
            print(f"       - Compress dim: {compress_dim}")
            print(f"       - Attention space: 5 × {compress_dim} (极小！)")
            print(
                f"       - Params/layer: {total_params:,} ({total_params / 1e6:.1f}M)"
            )
            print(
                f"       - 32 layers: {total_params * 32:,} ({total_params * 32 / 1e9:.2f}B)"
            )
            print(f"       - Theory: IB + Predictive Coding + RG")
            print(f"       - Stability: ✅✅✅ 极高（attention 在低维空间）")

    def _init_weights(self, embed_weight, use_high_norm_init=False):
        """初始化 global queries

        Args:
            embed_weight: 预训练的 embedding 权重 [vocab_size, hidden_size]
            use_high_norm_init: 是否使用高范数初始化（实验性功能）
        """
        # 🚀 性能优化：缓存第一个 compressor（避免运行时类型检查）
        if isinstance(self.stat_compressors[0], nn.Sequential):
            self._first_compressor = self.stat_compressors[0][0]
        else:
            self._first_compressor = self.stat_compressors[0]

        if embed_weight is not None:
            with torch.no_grad():
                # 从 embeddings 采样并压缩到 compress_dim
                if use_high_norm_init:
                    # ✅ 实验：选择高范数嵌入
                    # 理由：高频词 norm 大，语义覆盖广，初始梯度可能更健康
                    embed_norms = torch.norm(embed_weight, dim=-1)  # [vocab_size]
                    _, top_indices = torch.topk(embed_norms, k=self.num_global)
                    indices = top_indices
                else:
                    # ❌ 原版：随机采样 # 优点：无偏，探索性强
                    indices = torch.randperm(embed_weight.size(0))[: self.num_global]

                init_embeddings = embed_weight[indices]  # [global_slots, 4096]

                init_embeddings = init_embeddings.to(
                    self._first_compressor.weight.dtype
                )

                # 用第一个 stat compressor 压缩
                # （这样 queries 和统计量在同一空间）
                init_compressed = self._first_compressor(
                    init_embeddings
                )  # [global_slots, 512]
                self.global_queries.copy_(init_compressed)

                # EMA 用完整的 embedding
                self.ema_global.copy_(init_embeddings.unsqueeze(0))

                # 🚀 性能优化：初始化 EMA 压缩缓存
                self.ema_compressed_cache.copy_(init_compressed)
                self._ema_cache_valid = True
        else:
            nn.init.xavier_uniform_(self.global_queries)

    def forward(self, local_memories):
        """
        两阶段前向传播

        Args:
            local_memories: [bsz, num_chunks, local_slots, hidden_size]

        Returns:
            G: [bsz, global_slots, hidden_size]
        """
        bsz, num_chunks, local_slots, hidden_size = local_memories.shape

        # 阶段1：统计量压缩（稳定基础）
        # Flatten: [bsz, num_chunks * local_slots, hidden_size]
        all_local_flat = local_memories.view(bsz, -1, hidden_size)

        # 计算 5 种统计量
        mean_pool = all_local_flat.mean(dim=1)  # [bsz, 4096]
        max_pool, _ = all_local_flat.max(dim=1)  # [bsz, 4096]
        min_pool, _ = all_local_flat.min(dim=1)  # [bsz, 4096]
        std_pool = all_local_flat.std(dim=1)  # [bsz, 4096]
        norm_mean = F.normalize(mean_pool, dim=-1, p=2)  # [bsz, 4096]

        # 分离压缩：每个统计量 4096 → 512
        stats_list = [mean_pool, max_pool, min_pool, std_pool, norm_mean]
        compressed_stats = [
            self.stat_compressors[i](stat) for i, stat in enumerate(stats_list)
        ]

        # Stack: [bsz, 5, 512]
        stats_stacked = torch.stack(compressed_stats, dim=1)

        # 阶段2：Lightweight Attention（理论提升）
        # Q: [bsz, global_slots, 512]
        # ✅ 融合 EMA 长期先验（Predictive Coding）
        # global_queries: 学习的、动态的查询
        # ema_global: 慢变的、稳定的先验
        Q = self.global_queries.unsqueeze(0).expand(bsz, -1, -1)

        # 🚀 性能优化3：使用缓存的 EMA 压缩结果（避免每次 forward 都重新压缩）
        if hasattr(self, "ema_global") and hasattr(self, "ema_weight"):
            # 使用缓存的压缩结果（在 EMA 更新时才重新压缩）
            if self._ema_cache_valid:
                ema_compressed = self.ema_compressed_cache.unsqueeze(0).expand(
                    bsz, -1, -1
                )
            else:
                # 缓存失效，重新压缩（只在首次或缓存失效时执行）
                ema_compressed = self._first_compressor(self.ema_global.squeeze(0))
                self.ema_compressed_cache.copy_(ema_compressed)
                self._ema_cache_valid = True
                ema_compressed = ema_compressed.unsqueeze(0).expand(bsz, -1, -1)

            # 加权融合：Q = learned + α * prior
            Q = (
                Q + self.ema_weight * ema_compressed.detach()
            )  # detach 避免影响 EMA 的梯度

        # 投影到 attention 空间
        Q = self.q_proj(Q)

        # K, V: [bsz, 5, 512]
        K = self.k_proj(stats_stacked)
        V = self.v_proj(stats_stacked)

        # Scaled dot-product attention
        scale = self.compress_dim**-0.5
        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) * scale
        # [bsz, global_slots, 5]

        attn_probs = F.softmax(attn_weights, dim=-1)

        # Weighted sum
        G_compressed = torch.matmul(attn_probs, V)
        # [bsz, global_slots, 512]

        # 阶段3：扩展到 hidden_size zxy
        # G = self.expand(G_compressed)  # [bsz, global_slots, 4096]
        G = self.expand(G_compressed) * self.expand_scale

        # 🚀 性能优化4：EMA 更新时使缓存失效
        if self.training:
            with torch.no_grad():
                batch_mean_G = G.mean(dim=0, keepdim=True)
                self.ema_global.copy_(
                    self.ema_decay * self.ema_global
                    + (1 - self.ema_decay) * batch_mean_G
                )
                # 使 EMA 压缩缓存失效（下次 forward 时重新压缩）
                self._ema_cache_valid = False

        return G


# 方法3   🆕 混合全局记忆 - ICML最佳方案（方案B：统计量 + Lightweight Attention）
class GlobalIntegrator_new(nn.Module):
    """
    混合全局记忆 - 统计量 + Lightweight Attention（ICML推荐）

    设计哲学：
    1. ✅ 稳定性：统计量提供稳定的"锚点"（避免 attention 炸掉）
    2. ✅ 理论性：Attention 学习如何 refine 统计特征（非启发式）
    3. ✅ 高效性：Attention 在低维空间操作（5 × 512，极小！）
    4. ✅ 参数量：13.4M/layer × 32 = 0.43B（比纯统计量版本少 34%）

    理论对齐：
    - Information Bottleneck:
      • 统计量 = 粗粒化压缩（coarse-grained summary）
      • Attention = 学习的 sufficient statistics refinement
      • Capacity constraint via global_slots

    - Predictive Coding:
      • Global = high-level slow prior（EMA 慢更新）
      • Attention = adaptive routing of information

    - Renormalization Group:
      • Step 1: Statistical aggregation（粗粒化）
      • Step 2: Learned blocking transformation（精细化）

    两阶段设计：
        阶段1（稳定基础）：统计量压缩
            local_memories: [bsz, N, 4096]
            → 5种统计量: [mean, max, min, std, norm]
            → 分离压缩: 每个 4096 → 512
            → compressed_stats: [bsz, 5, 512]

        阶段2（理论提升）：Lightweight Attention
            global_queries: [global_slots, 512]  ← parameter
            compressed_stats: [bsz, 5, 512]
            → cross_attn in 512-dim space（极小！）
            → G_compressed: [bsz, global_slots, 512]
            → expand to 4096
            → G: [bsz, global_slots, 4096]

    关键优势：
    1. Attention 只在 5 个统计量 × 512 维上操作（极小规模，风险低）
    2. 统计量提供稳定的语义"锚点"
    3. Attention 学习如何组合这些统计量（非启发式）
    4. 即使 attention 出问题，统计量仍然能保底

    参数量分析：
        统计量压缩: 5 × (4096 × 512) = 10.5M
        Q/K/V 投影: 3 × (512 × 512) = 0.8M
        Expand: 512 × 4096 = 2.1M
        Total: ~13.4M/layer × 32 = 0.43B

    风险评估：
        方案A（纯统计量）: 风险 0%  ✅✅✅
        方案B（混合）:     风险 10% ✅✅  ← 推荐用于ICML
        方案C（pure attn）:风险 40% ⚠️

    论文叙述建议：
    "We propose a hybrid approach that combines deterministic statistical
     aggregation with learned attention-based refinement. First, we extract
     five statistical features (mean, max, min, std, normalized mean) from
     local memories and compress them to a 512-dim bottleneck space. Then,
     global learned queries attend over these compressed statistics to
     extract minimal sufficient statistics for document-level context.
     This design provides both stability (via deterministic statistics)
     and adaptivity (via learned attention)."
    """

    _init_msg_printed = False

    def __init__(
        self,
        hidden_size: int = 4096,
        global_slots: int = 4,  # .sh脚本控制
        compress_dim: int = 512,  # 统计量压缩维度 # 调用处控制
        local_slots: int = 16,  # 兼容参数
        use_bottleneck: bool = False,  # 兼容参数
        bottleneck_dim: int = 4096,  # 兼容参数
        init_from_embeddings=None,  # 调用处控制
        use_high_norm_init: Optional[bool] = True,  # 调用处控制
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_global = global_slots
        self.compress_dim = compress_dim
        self.use_high_norm_init = use_high_norm_init  # 保存配置

        # 阶段1：统计量分离压缩（和之前一样，稳定）
        # 有归一化 LayerNorm
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

        # 阶段2：Lightweight Attention（理论提升）
        # Global queries in compressed space  局部记忆使用torch.randn初始化
        self.global_queries = nn.Parameter(torch.zeros(global_slots, compress_dim))

        # Q/K/V 投影（都在 512 维空间）参数量：3 × (512 × 512) = 0.8M（很小！）
        self.q_proj = nn.Linear(compress_dim, compress_dim, bias=False)
        self.k_proj = nn.Linear(compress_dim, compress_dim, bias=False)
        self.v_proj = nn.Linear(compress_dim, compress_dim, bias=False)

        # 阶段3：扩展到 hidden_size
        # self.expand = nn.Linear(compress_dim, hidden_size)
        self.expand = nn.Linear(compress_dim, hidden_size, bias=False)
        # LLaMA 风格初始化（关键！）
        std_init = 0.02 / math.sqrt(compress_dim)  # ≈ 0.00088
        nn.init.normal_(self.expand.weight, mean=0.0, std=std_init)
        self.expand_scale = nn.Parameter(torch.tensor(0.1))  # 初始化为 0.1 新增

        # ✅ EMA Buffer（Predictive Coding - 作为长期先验）
        # 用途：为 global queries 提供稳定的初始化偏置
        self.register_buffer("ema_global", torch.zeros(1, global_slots, hidden_size))
        self.ema_decay = 0.95
        self.ema_weight = 0.1  # EMA 对 queries 的影响权重

        # 🚀 性能优化：缓存压缩后的 EMA（避免每次 forward 都重新计算）
        self.register_buffer(
            "ema_compressed_cache", torch.zeros(global_slots, compress_dim)
        )
        self._ema_cache_valid = False  # 标记缓存是否有效

        # 🚀 性能优化：缓存 compressor 引用（避免运行时类型检查）
        self._first_compressor = None  # 将在 _init_weights 中设置

        # 初始化
        self._init_weights(
            init_from_embeddings, use_high_norm_init=self.use_high_norm_init
        )

        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank == 0 and not GlobalIntegrator_new._init_msg_printed:
            total_params = sum(p.numel() for p in self.parameters())
            print(f"   ✅ GlobalIntegrator_NEW initialized (方案B - ICML推荐)")
            print(f"       - Design: Statistical Aggregation + Lightweight Attention")
            print(f"       - Global slots: {global_slots} (IB capacity)")
            print(f"       - Compress dim: {compress_dim}")
            print(f"       - Attention space: 5 × {compress_dim} (极小！)")
            print(
                f"       - Params/layer: {total_params:,} ({total_params / 1e6:.1f}M)"
            )
            print(
                f"       - 32 layers: {total_params * 32:,} ({total_params * 32 / 1e9:.2f}B)"
            )
            print(f"       - Theory: IB + Predictive Coding + RG")
            print(f"       - Stability: ✅极高（attention 在低维空间）")
            GlobalIntegrator_new._init_msg_printed = True

    def _init_weights(self, embed_weight, use_high_norm_init=False):
        """初始化 global queries

        Args:
            embed_weight: 预训练的 embedding 权重 [vocab_size, hidden_size]
            use_high_norm_init: 是否使用高范数初始化（实验性功能）
        """
        # 🚀 性能优化：缓存第一个 compressor（避免运行时类型检查）
        # ✅ 修复 Issue A (P0): 缓存完整序列（Linear+LayerNorm），确保 EMA 和统计量使用相同的归一化流程
        # 原问题：只缓存 Linear 导致 ema_compressed 无归一化，尺度可能是 stats 的 5-10 倍，污染 Q → Softmax 崩溃
        self._first_compressor = self.stat_compressors[0]

        if embed_weight is not None:
            with torch.no_grad():
                # 从 embeddings 采样并压缩到 compress_dim
                if use_high_norm_init:
                    # ✅ 实验：选择高范数嵌入
                    # 理由：高频词 norm 大，语义覆盖广，初始梯度可能更健康
                    embed_norms = torch.norm(embed_weight, dim=-1)  # [vocab_size]
                    _, top_indices = torch.topk(embed_norms, k=self.num_global)
                    indices = top_indices
                else:
                    # ❌ 原版：随机采样 # 优点：无偏，探索性强
                    indices = torch.randperm(embed_weight.size(0))[: self.num_global]

                init_embeddings = embed_weight[indices]  # [global_slots, 4096]

                init_embeddings = init_embeddings.to(
                    self._first_compressor[0].weight.dtype
                )

                # 用第一个 stat compressor 压缩
                # （这样 queries 和统计量在同一空间）
                init_compressed = self._first_compressor(
                    init_embeddings
                )  # [global_slots, 512]
                self.global_queries.copy_(init_compressed)

                # EMA 用完整的 embedding
                self.ema_global.copy_(init_embeddings.unsqueeze(0))

                # 🚀 性能优化：初始化 EMA 压缩缓存
                self.ema_compressed_cache.copy_(init_compressed)
                self._ema_cache_valid = True
        else:
            nn.init.xavier_uniform_(self.global_queries)

    def forward(self, local_memories):
        """
        两阶段前向传播

        Args:
            local_memories: [bsz, num_chunks, local_slots, hidden_size]

        Returns:
            G: [bsz, global_slots, hidden_size]
        """
        bsz, num_chunks, local_slots, hidden_size = local_memories.shape

        # 阶段1：统计量压缩（稳定基础）
        # Flatten: [bsz, num_chunks * local_slots, hidden_size]
        all_local_flat = local_memories.reshape(bsz, -1, hidden_size)

        # 计算 5 种统计量
        mean_pool = all_local_flat.mean(dim=1)  # [bsz, 4096]
        max_pool, _ = all_local_flat.max(dim=1)  # [bsz, 4096]
        min_pool, _ = all_local_flat.min(dim=1)  # [bsz, 4096]

        # ✅ 修复 Issue B (P1): std 使用 fp32 精度和 unbiased=False，避免 bf16 数值不稳定
        # 原问题：bf16 下 std 计算有舍入误差，unbiased=True 除以 n-1 可能导致不稳定
        with torch.cuda.amp.autocast(enabled=False):
            std_pool = all_local_flat.float().std(dim=1, unbiased=False)
        std_pool = std_pool.to(all_local_flat.dtype)  # [bsz, 4096]

        norm_mean = F.normalize(mean_pool, dim=-1, p=2)  # [bsz, 4096]

        # 分离压缩：每个统计量 4096 → 512
        stats_list = [mean_pool, max_pool, min_pool, std_pool, norm_mean]
        compressed_stats = [
            self.stat_compressors[i](stat) for i, stat in enumerate(stats_list)
        ]

        # Stack: [bsz, 5, 512]
        stats_stacked = torch.stack(compressed_stats, dim=1)

        # 阶段2：Lightweight Attention（理论提升）
        # Q: [bsz, global_slots, 512]
        # ✅ 融合 EMA 长期先验（Predictive Coding）
        # global_queries: 学习的、动态的查询
        # ema_global: 慢变的、稳定的先验
        Q = self.global_queries.unsqueeze(0).expand(bsz, -1, -1)

        # 🚀 性能优化3：使用缓存的 EMA 压缩结果（避免每次 forward 都重新压缩）
        if hasattr(self, "ema_global") and hasattr(self, "ema_weight"):
            # 使用缓存的压缩结果（在 EMA 更新时才重新压缩）
            if self._ema_cache_valid:
                ema_compressed = self.ema_compressed_cache.unsqueeze(0).expand(
                    bsz, -1, -1
                )
            else:
                # 缓存失效，重新压缩（只在首次或缓存失效时执行）
                ema_compressed = self._first_compressor(self.ema_global.squeeze(0))
                self.ema_compressed_cache.copy_(ema_compressed)
                self._ema_cache_valid = True
                ema_compressed = ema_compressed.unsqueeze(0).expand(bsz, -1, -1)

            # 加权融合：Q = learned + α * prior
            Q = (
                Q + self.ema_weight * ema_compressed.detach()
            )  # detach 避免影响 EMA 的梯度

        # 投影到 attention 空间
        Q = self.q_proj(Q)

        # K, V: [bsz, 5, 512]
        K = self.k_proj(stats_stacked)
        V = self.v_proj(stats_stacked)

        # Scaled dot-product attention
        scale = self.compress_dim**-0.5
        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) * scale
        # [bsz, global_slots, 5]

        attn_probs = F.softmax(attn_weights, dim=-1)

        # Weighted sum
        G_compressed = torch.matmul(attn_probs, V)
        # [bsz, global_slots, 512]

        # 阶段3：扩展到 hidden_size zxy
        # G = self.expand(G_compressed)  # [bsz, global_slots, 4096]
        G_unscaled = self.expand(G_compressed)  # [bsz, global_slots, 4096]
        G = G_unscaled * self.expand_scale  # 输出用于后续计算

        # 🚀 性能优化4：EMA 更新时使缓存失效
        if self.training:
            with torch.no_grad():
                # ✅ 修复 Issue E (P0): EMA 更新使用未缩放的 G，避免 expand_scale 动态变化污染 EMA
                # 原问题：expand_scale 从 0.1→0.15 时，EMA 累积不同尺度的值 → ema_compressed 失控 → Q 被劫持 → 正反馈崩溃
                batch_mean_G = G_unscaled.mean(dim=0, keepdim=True)
                self.ema_global.copy_(
                    self.ema_decay * self.ema_global
                    + (1 - self.ema_decay) * batch_mean_G
                )
                # 使 EMA 压缩缓存失效（下次 forward 时重新压缩）
                self._ema_cache_valid = False

        return G


# 方法3.1  🆕 混合全局记忆 - 简化版（无 EMA） Clean版本
class GlobalIntegrator(nn.Module):
    """
    混合全局记忆 - 简化版（无 EMA）

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
        Output: global_memory  [bsz, global_slots, hidden_size]

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
            global_slots: 全局记忆槽数量（通常 4-16）
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

    def forward_causal(self, local_memories: torch.Tensor) -> torch.Tensor:
        """
        因果版前向传播：为每个 segment 计算独立的 G_i

        对于 segment i，G_i 仅由 L_1, ..., L_i 的累积统计量计算，
        保证不包含未来 segment 的信息。

        Args:
            local_memories: [bsz, num_chunks, local_slots, hidden_size]

        Returns:
            G_all: [bsz, num_chunks, global_slots, hidden_size]
                   G_all[:, i, :, :] = Agg(L_1, ..., L_{i+1})
        """
        bsz, num_chunks, local_slots, hidden_size = local_memories.shape

        # ========== 阶段1：累积统计量提取 ==========
        # 按 chunk 维度计算 per-chunk 聚合，再做 cumulative 操作
        # local_memories: [bsz, N, L, H]

        # Per-chunk aggregation
        sum_per_chunk = local_memories.sum(dim=2)           # [bsz, N, H]
        max_per_chunk = local_memories.max(dim=2).values    # [bsz, N, H]
        min_per_chunk = local_memories.min(dim=2).values    # [bsz, N, H]

        # Cumulative statistics along chunk dimension
        cumsum = sum_per_chunk.cumsum(dim=1)                # [bsz, N, H]
        counts = torch.arange(1, num_chunks + 1, device=local_memories.device,
                              dtype=local_memories.dtype).view(1, -1, 1) * local_slots
        cum_mean = cumsum / counts                          # [bsz, N, H]

        cum_max = max_per_chunk.cummax(dim=1).values        # [bsz, N, H]
        cum_min = min_per_chunk.cummin(dim=1).values        # [bsz, N, H]

        # Cumulative std: sqrt(E[X²] - E[X]²)
        with torch.amp.autocast(device_type="cuda", enabled=False):
            local_fp32 = local_memories.float()
            sq_sum_per_chunk = (local_fp32 ** 2).sum(dim=2)  # [bsz, N, H]
            cum_sq_sum = sq_sum_per_chunk.cumsum(dim=1)
            counts_f = counts.float()
            cum_sq_mean = cum_sq_sum / counts_f
            cum_mean_f = cumsum.float() / counts_f
            cum_var = (cum_sq_mean - cum_mean_f ** 2).clamp(min=1e-12)
            cum_std = cum_var.sqrt()
        cum_std = cum_std.to(local_memories.dtype)

        cum_norm_mean = F.normalize(cum_mean, dim=-1, p=2, eps=1e-6)  # [bsz, N, H]

        # Stack: [bsz, N, 5, H]
        cum_stats = torch.stack([cum_mean, cum_max, cum_min, cum_std, cum_norm_mean], dim=2)

        # ========== 阶段1b：压缩（batch over N）==========
        # Reshape to [bsz*N, 5, H] → process as batch
        cum_stats_flat = cum_stats.reshape(bsz * num_chunks, 5, hidden_size)
        compressed_list = [
            self.stat_compressors[i](cum_stats_flat[:, i, :])
            for i in range(5)
        ]
        compressed_stats = torch.stack(compressed_list, dim=1)  # [bsz*N, 5, compress_dim]

        # ========== 阶段2：Lightweight MHA（batch over N）==========
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

        # ========== 阶段3：维度扩展 ==========
        G = self.expand(G_compressed) * self.expand_scale  # [bsz*N, global_slots, H]

        # Reshape back: [bsz, N, global_slots, H]
        G = G.view(bsz, num_chunks, self.global_slots, hidden_size)
        return G


# ============================================================================
# 方法3.2  🆕 共享压缩层优化版 - GlobalIntegratorShared
# ============================================================================
class GlobalIntegratorShared(nn.Module):
    """
    混合全局记忆 - 共享压缩层优化版

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
        Output: global_memory  [bsz, global_slots, hidden_size]
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
            global_slots: 全局记忆槽数量（通常 4-16）
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

    def forward_causal(self, local_memories: torch.Tensor) -> torch.Tensor:
        """
        因果版前向传播：为每个 segment 计算独立的 G_i（共享压缩层版本）

        对于 segment i，G_i 仅由 L_1, ..., L_i 的累积统计量计算。

        Args:
            local_memories: [bsz, num_chunks, local_slots, hidden_size]

        Returns:
            G_all: [bsz, num_chunks, global_slots, hidden_size]
        """
        bsz, num_chunks, local_slots, hidden_size = local_memories.shape

        # ========== 阶段1：累积统计量提取 ==========
        sum_per_chunk = local_memories.sum(dim=2)           # [bsz, N, H]
        max_per_chunk = local_memories.max(dim=2).values    # [bsz, N, H]
        min_per_chunk = local_memories.min(dim=2).values    # [bsz, N, H]

        cumsum = sum_per_chunk.cumsum(dim=1)                # [bsz, N, H]
        counts = torch.arange(1, num_chunks + 1, device=local_memories.device,
                              dtype=local_memories.dtype).view(1, -1, 1) * local_slots
        cum_mean = cumsum / counts                          # [bsz, N, H]

        cum_max = max_per_chunk.cummax(dim=1).values        # [bsz, N, H]
        cum_min = min_per_chunk.cummin(dim=1).values        # [bsz, N, H]

        with torch.amp.autocast(device_type="cuda", enabled=False):
            local_fp32 = local_memories.float()
            sq_sum_per_chunk = (local_fp32 ** 2).sum(dim=2)
            cum_sq_sum = sq_sum_per_chunk.cumsum(dim=1)
            counts_f = counts.float()
            cum_sq_mean = cum_sq_sum / counts_f
            cum_mean_f = cumsum.float() / counts_f
            cum_var = (cum_sq_mean - cum_mean_f ** 2).clamp(min=1e-12)
            cum_std = cum_var.sqrt()
        cum_std = cum_std.to(local_memories.dtype)

        cum_norm_mean = F.normalize(cum_mean, dim=-1, p=2, eps=1e-6)

        # Stack: [bsz, N, 5, H]
        cum_stats = torch.stack([cum_mean, cum_max, cum_min, cum_std, cum_norm_mean], dim=2)

        # ========== 阶段1b：共享压缩 + 扩展（batch over N）==========
        BN = bsz * num_chunks
        # [bsz*N, 5, H]
        cum_stats_flat = cum_stats.reshape(BN * 5, hidden_size)
        compressed_stats = self.shared_compressor(cum_stats_flat).view(BN, 5, -1)
        compressed_stats = self.stat_expand(
            compressed_stats.view(BN * 5, -1)
        ).view(BN, 5, self.compress_dim)

        # ========== 阶段2：Lightweight MHA（batch over N）==========
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

        # ========== 阶段3：维度扩展 ==========
        G = self.expand(G_compressed) * self.expand_scale  # [BN, global_slots, H]
        G = G.view(bsz, num_chunks, self.global_slots, hidden_size)
        return G


# 方法3   🆕 Temporal Perceiver Global Memory  loss会炸掉
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


# 方法4 🔽 旧实现：LightweightGlobalMemory（理论性不强 已弃用，仅供参考 已升级为GlobalIntegrator）
#
# 问题：
# - 使用启发式统计量（mean/max/min/std），非学习的
# - 理论一致性中等
# - 不如混合版本（方案B）的理论性强
# 保留原因：
# - 如果混合版本训练失败，可以回退到这个版本
# - 风险最低（0%），100%稳定
# region
# class LightweightGlobalMemory(nn.Module):
#     """
#     轻量级全局记忆 - 基于Information Bottleneck理论
#
#     理论基础：
#     1. Information Bottleneck (Tishby & Zaslavsky, Nature MI 2019)
#     2. Predictive Coding (Rao & Ballard, Nature Neuroscience 1999)
#     3. Free Energy Principle (Friston, Nature Reviews Neuroscience 2010)
#
#     关键创新：
#     1. 极小global slots（4个，符合IB + 工程平衡）
#     2. 分离降维（避免参数爆炸：32M vs 302M，减少90%）
#     3. 隐式优化（通过LM loss，无需auxiliary loss）
#     4. EMA慢更新（Predictive Coding高层先验）
#
#     参数量：
#     - 每层：32M（相比之前302M减少89%）
#     - 32层：1.0B（相比之前9.7B减少90%）
#     - 占模型：~10%（相比之前62.6%改善6倍）
#
#     理论解释：
#     从信息论视角，global memory G可形式化为：
#         G = argmin_G Σ_g ℓ(S_g, φ(G)) + λ||G||_capacity
#
#     实践中，此目标隐式优化：
#     - 保真项：通过LM loss（G影响预测→影响loss）
#     - 容量约束：通过architecture（global_slots=4, EMA）
#     """
#
#     def __init__(
#         self,
#         hidden_size: int = 4096,
#         global_slots: int = 4,  # ← 4个slot（方案B：理论与工程平衡）
#         local_slots: int = 16,  # 兼容参数
#         use_bottleneck: bool = False,  # 兼容参数
#         bottleneck_dim: int = 4096,  # 兼容参数
#         init_from_embeddings=None,
#     ):
#         super().__init__()
#
#         self.hidden_size = hidden_size
#         self.num_global = global_slots
#         self.compress_dim = 512  # 中间压缩维度
#
#         # ═══════════════════════════════════════════════════════════
#         # 阶段1：分离降维（关键！避免参数爆炸）
#         # ═══════════════════════════════════════════════════════════
#         # 每个统计量单独降维：4096 → 512
#         # 参数量：5 * (4096 * 512 + 512) = 10.5M
#         # vs 旧方案：(5*4096) * 8192 = 167M（减少94%）
#
#         self.stat_compressors = nn.ModuleList([
#             nn.Sequential(
#                 nn.Linear(hidden_size, self.compress_dim, bias=False),
#                 nn.LayerNorm(self.compress_dim)
#             )
#             for _ in range(5)  # mean, max, min, std, norm_mean
#         ])
#
#         # ═══════════════════════════════════════════════════════════
#         # 阶段2：编码器（2560 → num_global*4096）
#         # ═══════════════════════════════════════════════════════════
#         combined_dim = 5 * self.compress_dim  # 2560
#         output_dim = self.num_global * hidden_size  # 16384 (4 slots)
#
#         # 参数量：(2560*512 + 512) + (512*16384) = 1.3M + 8.4M = 9.7M
#         self.encoder = nn.Sequential(
#             nn.Linear(combined_dim, self.compress_dim),  # 2560 → 512
#             nn.GELU(),
#             nn.Linear(self.compress_dim, output_dim)     # 512 → 16384
#         )
#
#         # ═══════════════════════════════════════════════════════════
#         # 阶段3：EMA慢变量（Predictive Coding高层先验）
#         # ═══════════════════════════════════════════════════════════
#         # 这是architecture约束，不涉及loss
#
#         self.register_buffer(
#             'ema_global',
#             torch.zeros(1, self.num_global, hidden_size)
#         )
#         self.ema_decay = 0.95  # 慢更新系数
#
#         # 初始化
#         self._init_weights(init_from_embeddings)
#
#         # 打印信息
#         rank = dist.get_rank() if dist.is_initialized() else 0
#         if rank == 0:
#             total_params = sum(p.numel() for p in self.parameters())
#             print(f"    ✅ LightweightGlobalMemory initialized")
#             print(f"       - Global slots: {self.num_global} (IB-inspired, 方案B)")
#             print(f"       - Compress dim: {self.compress_dim}")
#             print(f"       - Params/layer: {total_params:,} ({total_params/1e6:.1f}M)")
#             print(f"       - Optimization: Implicit via LM loss (no auxiliary loss!)")
#             print(f"       - Capacity constraint: Structural (architecture-based)")
#             print(f"       - Theory: Information Bottleneck + Predictive Coding")
#
#     def _init_weights(self, embed_weight):
#         """从embeddings初始化EMA（避免random noise导致loss=9.0）"""
#         if embed_weight is not None:
#             with torch.no_grad():
#                 # 从embeddings采样初始化EMA
#                 indices = torch.randperm(embed_weight.size(0))[:self.num_global]
#                 init_global = embed_weight[indices].unsqueeze(0)
#                 self.ema_global.copy_(init_global)
#
#     def forward(self, local_memories):
#         """
#         轻量级全局记忆前向传播 - 隐式优化版本
#
#         Args:
#             local_memories: [bsz, num_chunks, local_slots, hidden_size]
#                 例如 [2, 4, 8, 4096] = 32个local slots
#
#         Returns:
#             higher_global: [bsz, num_global, hidden_size]
#                 例如 [2, 4, 4096] = 4个global slots
#
#         注意：
#             - 无需返回loss！
#             - IB目标通过LM loss隐式优化
#             - 容量约束通过architecture实现
#         """
#         bsz, num_chunks, local_slots, hidden_size = local_memories.shape
#
#         # ═══════════════════════════════════════════════════════════
#         # 步骤1：计算5种统计量（O(n·d)复杂度）
#         # ═══════════════════════════════════════════════════════════
#
#         # 展平所有local slots: [bsz, num_chunks * local_slots, hidden_size]
#         all_local_flat = local_memories.view(bsz, -1, hidden_size)
#
#         # 计算统计量
#         mean_pool = all_local_flat.mean(dim=1)              # [bsz, 4096]
#         max_pool, _ = all_local_flat.max(dim=1)             # [bsz, 4096]
#         min_pool, _ = all_local_flat.min(dim=1)             # [bsz, 4096]
#         std_pool = all_local_flat.std(dim=1)                # [bsz, 4096]
#         norm_mean = F.normalize(mean_pool, dim=-1, p=2)     # [bsz, 4096]
#
#         stats_list = [mean_pool, max_pool, min_pool, std_pool, norm_mean]
#
#         # ═══════════════════════════════════════════════════════════
#         # 步骤2：分离降维（关键！避免参数爆炸）
#         # ═══════════════════════════════════════════════════════════
#         # 每个统计量单独降维：[bsz, 4096] → [bsz, 512]
#         # 这样参数量只有 5 * (4096 * 512) = 10.5M
#         # 而不是 (5*4096) * 8192 = 167M（减少94%）
#
#         compressed_stats = []
#         for i, stat in enumerate(stats_list):
#             compressed = self.stat_compressors[i](stat)  # [bsz, 512]
#             compressed_stats.append(compressed)
#
#         # 组合压缩后的统计量
#         combined_stats = torch.cat(compressed_stats, dim=-1)  # [bsz, 2560]
#
#         # ═══════════════════════════════════════════════════════════
#         # 步骤3：编码为全局表示
#         # ═══════════════════════════════════════════════════════════
#
#         G_flat = self.encoder(combined_stats)  # [bsz, num_global*4096]
#         G = G_flat.view(bsz, self.num_global, -1)  # [bsz, num_global, 4096]
#
#         # ═══════════════════════════════════════════════════════════
#         # 步骤4：EMA更新（Predictive Coding高层先验）
#         # ═══════════════════════════════════════════════════════════
#         # 这是architecture约束，不涉及loss
#
#         if self.training:
#             with torch.no_grad():
#                 # 批次平均作为新观测
#                 batch_mean_G = G.mean(dim=0, keepdim=True)
#
#                 # EMA更新：θ_new = α*θ_old + (1-α)*observation
#                 # 这体现了Predictive Coding的"高层先验慢变"
#                 self.ema_global.copy_(
#                     self.ema_decay * self.ema_global +
#                     (1 - self.ema_decay) * batch_mean_G
#                 )
#
#         return G

# endregion ===========================================================================

# 方法5 🔽 旧实现（已弃用，参数量过大导致训练不稳定）
#
# 问题诊断：
# - 参数量：302M/layer × 32 = 9.7B（占模型62.6%）
# - 设计错误：concatenate-then-project导致参数二次方增长
# - 违背理论：global_slots=4仍然偏大（IB理论建议2-4个）
#
# 保留此代码仅供参考，请勿使用！
# region ===========================================================================
# class StableStatisticalAggregator(nn.Module):
#     """
#     🆕 方案A：基于统计池化的聚合器（极其稳定，无Attention）
#
#     数学原理：
#     通过计算多种统计量来压缩局部记忆，然后用MLP投影到全局表示。
#
#     优势：
#     1. ✅ 极其稳定：无softmax饱和问题，无attention训练不稳定
#     2. ✅ 计算高效：O(n·d)复杂度，无需O(n·m·d)的attention
#     3. ✅ 可解释性强：每个统计量有明确含义
#     4. ✅ 参数少：只有投影层，无Q/K/V投影
#     5. ✅ 可以安全训练：适合加入trainable_params
#
#     统计量说明：
#     - mean: 中心趋势，捕捉平均语义
#     - max: 显著特征，捕捉最强激活
#     - min: 边界信息，捕捉最弱激活
#     - std: 变化程度，捕捉分布离散度
#     - norm_mean: L2归一化均值，鲁棒的中心
#
#     Args:
#         hidden_size: 模型隐藏维度 (e.g., 4096)
#         num_global: 全局记忆slots数量 (default: 16)
#     """
#
#     def __init__(
#         self,
#         hidden_size: int = 4096,
#         # num_global: int = 4,
#         global_slots: int = 4,
#         local_slots: int = 16,  # 兼容参数，实际不使用
#         use_bottleneck: bool = False,  # 兼容参数
#         bottleneck_dim: int = 4096,  # 兼容参数
#         init_from_embeddings=None,  # 兼容参数
#     ):
#         super().__init__()
#
#         self.hidden_size = hidden_size
#         self.num_global = global_slots
#
#         # 5种统计量，每个是hidden_size维
#         stats_dim = 5 * hidden_size
#
#         # 投影层：统计特征 → 全局表示
#         # 🆕 优化版本：8192 bottleneck + GELU activation (2024-12)
#         # 理论依据：
#         #   - 8192 bottleneck: 减少压缩比从10:1到2.5:1，保留更多信息（effective rank ~10,240）
#         #   - GELU: 相比ReLU更平滑，在深度网络中梯度流动更好
#         self.projection = nn.Sequential(
#             nn.Linear(stats_dim, 8192),          # 20480 → 8192 (2.5:1压缩)
#             nn.LayerNorm(8192),
#             nn.GELU(),                           # GELU替代ReLU，更平滑
#             nn.Dropout(0.1),
#             nn.Linear(8192, self.num_global * hidden_size),  # 8192 → 65536
#             nn.LayerNorm(self.num_global * hidden_size)
#         )
#
#         # # 🔽 旧版本：2048 bottleneck + ReLU (已弃用，压缩过于激进)
#         # self.projection = nn.Sequential(
#         #     nn.Linear(stats_dim, 2048),          # 20480 → 2048 (10:1压缩)
#         #     nn.LayerNorm(2048),
#         #     nn.ReLU(),
#         #     nn.Dropout(0.1),
#         #     nn.Linear(2048, self.num_global * hidden_size),
#         #     nn.LayerNorm(self.num_global * hidden_size)
#         # )
#
#         # 用于打印信息
#         rank = dist.get_rank() if dist.is_initialized() else 0
#         if rank == 0:
#             print(f"    ✅ StableStatisticalAggregator initialized (5 statistics → {self.num_global} global slots)")
#             print(f"       No attention mechanism - extremely stable for training!")
#
#     def forward(self, local_memories):
#         """
#         通过统计池化聚合局部记忆。
#
#         Args:
#             local_memories: [bsz, num_chunks, local_slots, hidden_size]
#
#         Returns:
#             higher_global: [bsz, num_global, hidden_size]
#         """
#         bsz, num_chunks, local_slots, hidden_size = local_memories.shape
#
#         # Flatten所有局部记忆: [bsz, num_chunks * local_slots, hidden_size]
#         all_local_flat = local_memories.view(bsz, num_chunks * local_slots, hidden_size)
#
#         # 计算5种统计量（每个都是[bsz, hidden_size]）
#
#         # 1. 均值：中心趋势
#         mean_pool = all_local_flat.mean(dim=1)
#
#         # 2. 最大值：显著特征
#         max_pool, _ = all_local_flat.max(dim=1)
#
#         # 3. 最小值：边界信息
#         min_pool, _ = all_local_flat.min(dim=1)
#
#         # 4. 标准差：分布离散度
#         std_pool = all_local_flat.std(dim=1)
#
#         # 5. L2归一化的均值：鲁棒中心（不受幅度影响）
#         norm_mean = F.normalize(mean_pool, dim=-1, p=2)
#
#         # Concat所有统计量: [bsz, 5 * hidden_size]
#         stats = torch.cat([mean_pool, max_pool, min_pool, std_pool, norm_mean], dim=-1)
#
#         # 投影到全局表示: [bsz, num_global * hidden_size]
#         global_flat = self.projection(stats)
#
#         # Reshape: [bsz, num_global, hidden_size]
#         higher_global = global_flat.view(bsz, self.num_global, hidden_size)
#
#         return higher_global
# endregion

# 方法6  🔽 旧代码：基于Attention的聚合器（loss训练不稳定会炸掉，已注释）
# region
# class HierarchicalMemoryAggregatorSingleHead(nn.Module):

#     '''
#     单头版本的 HierarchicalMemoryAggregator（推荐！）

#     基于观察：LocalConstructor 在单头时效果更好，说明全局一致性比多视角更重要。

#     优势：
#     1. ✅ 全局一致性：避免不同 head 学到冲突的表示
#     2. ✅ 参数更少或相同：无需 head splitting 和 concatenation
#     3. ✅ 计算更快：减少 reshape 和 transpose 操作
#     4. ✅ 更简洁：代码更直观易懂

#     Args:
#         hidden_size: 模型隐藏维度 (e.g., 4096)
#         local_slots: 每个 chunk 的局部记忆 slots 数量 (default: 16)
#         global_slots: 高层全局记忆 slots 数量 (default: 16)
#         use_bottleneck: 是否使用瓶颈维度（False 更简单，参数相同但计算更快）
#         bottleneck_dim: 瓶颈维度（仅当 use_bottleneck=True 时使用）
#     '''
#     def __init__(
#         self,
#         hidden_size: int = 4096,
#         local_slots: int = 16,
#         global_slots: int = 16,
#         use_bottleneck: bool = False,
#         bottleneck_dim: int = 4096,
#         init_from_embeddings=None,
#     ):
#         super().__init__()

#         self.hidden_size = hidden_size
#         self.local_slots = local_slots
#         self.global_slots = global_slots
#         self.use_bottleneck = use_bottleneck

#         self.output_norm = nn.LayerNorm(hidden_size)

#         # 高层全局记忆 slots（长期记忆）
#         # ✅ 方案 2: 从预训练嵌入初始化（最优策略）
#         # if init_from_embeddings is not None:
#         if False:
#             # 从预训练嵌入中随机采样 global_slots 个作为初始值
#             indices = torch.randperm(init_from_embeddings.size(0))[:global_slots]
#             self.global_memory_slots = nn.Parameter(
#                 init_from_embeddings[indices].clone()
#             )
#             # ✅ 只在 rank 0 打印
#             rank = dist.get_rank() if dist.is_initialized() else 0
#             if rank == 0:
#                 print(f"    ✅ Initialized global_memory_slots from pretrained embeddings (sampled {global_slots} tokens)")
#         else:
#             # Fallback: 使用 LLaMA 标准初始化 (std=0.02)
#             std = 1.0 / math.sqrt(hidden_size)  # ≈ 0.0156 (太小！)
#             # std = 0.02  # LLaMA/GPT 标准
#             self.global_memory_slots = nn.Parameter(
#                 torch.randn(global_slots, hidden_size) * std
#             )
#             # ✅ 只在 rank 0 打印
#             rank = dist.get_rank() if dist.is_initialized() else 0
#             if rank == 0:
#                 print(f"    ⚠️  Fallback: Initialized global_memory_slots with std={std}")

#         if use_bottleneck:
#             # 使用瓶颈维度（节省计算）
#             self.dim = bottleneck_dim
#             self.agg_q_proj = nn.Linear(hidden_size, bottleneck_dim, bias=False)
#             self.agg_k_proj = nn.Linear(hidden_size, bottleneck_dim, bias=False)
#             self.agg_v_proj = nn.Linear(hidden_size, bottleneck_dim, bias=False)
#             self.agg_o_proj = nn.Linear(bottleneck_dim, hidden_size, bias=False)
#         else:
#             # 直接使用 hidden_size（更简单，无降维）
#             self.dim = hidden_size
#             self.agg_q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
#             self.agg_k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
#             self.agg_v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
#             # 无需 o_proj（直接输出 hidden_size 维度）
#             self.agg_o_proj = None

#     def forward(self, local_memories):
#         """
#         聚合局部记忆到高层全局记忆（单头 attention）。

#         Args:
#             local_memories: [bsz, num_chunks, local_slots, hidden_size]

#         Returns:
#             higher_global: [bsz, global_slots, hidden_size]
#         """
#         bsz, num_chunks, local_slots, hidden_size = local_memories.shape

#         # Flatten: [bsz, num_chunks * local_slots, hidden_size]
#         all_local_flat = local_memories.view(bsz, num_chunks * local_slots, hidden_size)

#         # 高层全局记忆 slots 作为 Query
#         global_mem = self.global_memory_slots.unsqueeze(0).expand(bsz, -1, -1)
#         # [bsz, global_slots, hidden_size]

#         # Single-head attention（无需 split heads）
#         Q_global = self.agg_q_proj(global_mem)
#         # [bsz, global_slots, dim]

#         K_local = self.agg_k_proj(all_local_flat)
#         # [bsz, num_chunks * local_slots, dim]

#         V_local = self.agg_v_proj(all_local_flat)
#         # [bsz, num_chunks * local_slots, dim]

#         # Attention: Higher Global ← All Local Memories
#         scores = torch.matmul(Q_global, K_local.transpose(-2, -1)) / math.sqrt(self.dim)
#         # [bsz, global_slots, num_chunks * local_slots]

#         attn_weights = torch.softmax(scores, dim=-1)

#         higher_global = torch.matmul(attn_weights, V_local)
#         # [bsz, global_slots, dim]

#         # Output projection (if using bottleneck)
#         if self.agg_o_proj is not None:
#             higher_global = self.agg_o_proj(higher_global)
#         # [bsz, global_slots, hidden_size]
#         higher_global = self.output_norm(higher_global) #加上归一化看是否会崩溃 没有测试


#         return higher_global
# endregion ===========================================================================


# 训练代码===========================================================================
# 原本longlora代码
def forward_flashattn_ori(
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
    if not self.training:
        warnings.warn(
            "This function should be used just for training as it may exhibit reduced inference performance. For inference, please use forward_flashattn_inference."
        )

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

    key_padding_mask = attention_mask.repeat(2, 1)
    nheads = qkv.shape[-2]
    # shift

    group_size = int(q_len * group_size_ratio)
    if q_len % group_size > 0:
        raise ValueError(
            "q_len %d should be divisible by group size %d." % (q_len, group_size)
        )

    qkv = (
        qkv.reshape(bsz, q_len, 3, 2, self.num_heads // 2, self.head_dim)
        .permute(0, 3, 1, 2, 4, 5)
        .reshape(bsz * 2, q_len, 3, self.num_heads // 2, self.head_dim)
    )
    x = rearrange(qkv, "b s three h d -> b s (three h d)")
    x_unpad, indices, cu_q_lens, max_s = unpad_input(x, key_padding_mask)
    cu_q_len_tmp = torch.arange(
        0, max_s, group_size, device=key_padding_mask.device, dtype=cu_q_lens.dtype
    )
    cu_q_len_tmp = torch.stack([cu_q_len_tmp, cu_q_len_tmp + group_size // 2]).repeat(
        bsz, 1
    ) + cu_q_lens[:-1].unsqueeze(-1)
    cu_q_lens = torch.cat([cu_q_len_tmp, cu_q_lens[1:].unsqueeze(-1)], dim=-1).view(-1)

    x_unpad = rearrange(
        x_unpad, "nnz (three h d) -> nnz three h d", three=3, h=nheads // 2
    )
    output_unpad = flash_attn_varlen_qkvpacked_func(
        x_unpad, cu_q_lens, group_size, 0.0, softmax_scale=None, causal=True
    )
    output = rearrange(
        pad_input(
            rearrange(output_unpad, "nnz h d -> nnz (h d)"), indices, bsz * 2, q_len
        ),
        "b s (h d) -> b s h d",
        h=nheads // 2,
    )
    output = (
        output.reshape(bsz, 2, q_len, nheads // 2, self.head_dim)
        .transpose(1, 2)
        .reshape(bsz, q_len, nheads, self.head_dim)
    )

    return self.o_proj(rearrange(output, "b s h d -> b s (h d)")), None, past_key_value


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

    NEW: Uses global memory + cross-attention instead of shift operation.
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
    # 传入 attention_mask 以正确处理 padding tokens  flash版本才传入attention_mask
    global_context = self.local_constructor(
        hidden_states, attention_mask
    )  # [bsz, num_slots, hidden_size]
    # global_context = self.local_constructor(hidden_states)  # [bsz, num_slots, hidden_size]

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
    group_size = int(q_len * group_size_ratio)
    if q_len % group_size > 0:
        raise ValueError(
            "q_len %d should be divisible by group size %d." % (q_len, group_size)
        )

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


# 训练way1混合版  混合优化：合并投影 + 向量化mask + torch.cat拼接（兼顾显存和速度） 已修正
# region ===========================================================================
def forward_flashattn_hybrid(
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
    混合优化版本 - 结合多种优化技巧：
    1. K/V 合并投影（省显存的关键！from optimized）
    2. 向量化 mask（省时间，from optimized）
    3. 使用 torch.cat 拼接（可能更快，from flashattn）
    4. Q 只包含 chunk，K/V 包含 [global_memory, chunk]（省计算！from hierarchical）

    核心优化：
    - Q: 只投影 hidden_states，输出 [chunk]
    - K/V: 拼接 [global_context, hidden_states] 后一起投影（省显存）
    - Flash Attention: Q=[chunk], K/V=[global_memory, chunk]
    - 输出直接是 chunk tokens，无需提取
    """
    if output_attentions:
        warnings.warn(
            "Output attentions is not supported for patched `LlamaAttention`, returning `None` instead."
        )

    bsz, q_len, hidden_size = hidden_states.size()

    # ========== Step 1: Global context ==========
    use_global_memory = hasattr(self, "local_constructor")

    if use_global_memory:
        global_context = self.local_constructor(
            hidden_states, attention_mask
        )  # [bsz, num_slots, hidden_size]
        num_local_slots = global_context.shape[1]
    else:
        num_local_slots = 0

    # ========== Step 2: Group parameters ==========
    group_size = int(q_len * group_size_ratio)
    if not hasattr(self, "_group_size_printed_hybrid"):
        layer_idx = getattr(self, "layer_idx", 0)
        if rank == 0 and layer_idx == 0:
            print(
                f"[forward_flashattn_hybrid] Q=[chunk], K/V=[global_mem,chunk] (merged proj), group_size={group_size}"
            )
        self._group_size_printed_hybrid = True
    if q_len % group_size > 0:
        raise ValueError(
            f"q_len {q_len} should be divisible by group size {group_size}."
        )
    num_groups = q_len // group_size

    # ========== Step 3: Q/K/V 投影 (关键优化！) ==========
    # Q: 只投影 hidden_states（不包含 global_context）
    query_states = (
        self.q_proj(hidden_states)
        .view(bsz, q_len, self.num_heads, self.head_dim)
        .transpose(1, 2)
    )  # [bsz, nh, q_len, hd]

    if use_global_memory:
        # K/V: 拼接后一起投影（省显存的关键！）
        combined_input = torch.cat(
            [global_context, hidden_states], dim=1
        )  # [bsz, num_slots + q_len, hidden_size]

        # K 投影
        combined_k = (
            self.k_proj(combined_input)
            .view(
                bsz, num_local_slots + q_len, self.num_key_value_heads, self.head_dim
            )
            .transpose(1, 2)
        )  # [bsz, nkv, num_slots + q_len, hd]

        # V 投影
        combined_v = (
            self.v_proj(combined_input)
            .view(
                bsz, num_local_slots + q_len, self.num_key_value_heads, self.head_dim
            )
            .transpose(1, 2)
        )

        # 分离 global 和 sequence 部分（view 操作，0 额外显存）
        global_k = combined_k[:, :, :num_local_slots, :]  # [bsz, nkv, num_slots, hd]
        key_states = combined_k[:, :, num_local_slots:, :]  # [bsz, nkv, q_len, hd]

        global_v = combined_v[:, :, :num_local_slots, :]
        value_states = combined_v[:, :, num_local_slots:, :]
    else:
        # 不使用 global memory，直接投影
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

    # ========== Step 4: RoPE (只对 sequence，不对 global) ==========
    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]

    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(
        query_states, key_states, cos, sin, position_ids
    )
    # global_k, global_v 不应用 RoPE（记忆是位置无关的）

    # Past Key value support
    if past_key_value is not None:
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)
    past_key_value = (key_states, value_states) if use_cache else None

    # Repeat k/v heads if n_kv_heads < n_heads
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)
    if use_global_memory:
        global_k = repeat_kv(global_k, self.num_key_value_groups)
        global_v = repeat_kv(global_v, self.num_key_value_groups)

    # ========== Step 5: Chunk reshaping ==========
    # Q: 只包含 chunk tokens
    query_chunks = query_states.view(
        bsz, self.num_heads, num_groups, group_size, self.head_dim
    )
    key_chunks = key_states.view(
        bsz, self.num_heads, num_groups, group_size, self.head_dim
    )
    value_chunks = value_states.view(
        bsz, self.num_heads, num_groups, group_size, self.head_dim
    )

    # ========== Step 6: 拼接 K/V (使用 torch.cat，高效) ==========
    if use_global_memory:
        # 使用 expand + cat（单次 kernel 调用，高度优化）
        global_k_expanded = global_k.unsqueeze(2).expand(
            -1, -1, num_groups, -1, -1
        )  # [bsz, nh, num_groups, num_slots, hd]
        global_v_expanded = global_v.unsqueeze(2).expand(-1, -1, num_groups, -1, -1)

        # K/V: [global_memory, chunk]
        key_with_ctx = torch.cat(
            [global_k_expanded, key_chunks], dim=3
        )  # [bsz, nh, num_groups, num_slots+group_size, hd]
        value_with_ctx = torch.cat([global_v_expanded, value_chunks], dim=3)

        # Reshape for flash attention
        kv_len = num_local_slots + group_size
        key_with_ctx = key_with_ctx.permute(0, 2, 3, 1, 4).reshape(
            bsz * num_groups, kv_len, self.num_heads, self.head_dim
        )
        value_with_ctx = value_with_ctx.permute(0, 2, 3, 1, 4).reshape(
            bsz * num_groups, kv_len, self.num_heads, self.head_dim
        )

        # Q: 只包含 chunk tokens (不包含 global_memory)
        query_with_ctx = query_chunks.permute(0, 2, 3, 1, 4).reshape(
            bsz * num_groups, group_size, self.num_heads, self.head_dim
        )
    else:
        # 不使用 global memory
        query_with_ctx = query_chunks.permute(0, 2, 3, 1, 4).reshape(
            bsz * num_groups, group_size, self.num_heads, self.head_dim
        )
        key_with_ctx = key_chunks.permute(0, 2, 3, 1, 4).reshape(
            bsz * num_groups, group_size, self.num_heads, self.head_dim
        )
        value_with_ctx = value_chunks.permute(0, 2, 3, 1, 4).reshape(
            bsz * num_groups, group_size, self.num_heads, self.head_dim
        )
        kv_len = group_size

    # ========== Step 7: 向量化 mask 构造 (省时间) ==========
    attention_mask_chunks = attention_mask.view(bsz, num_groups, group_size)

    if use_global_memory:
        # 向量化操作，无 Python loop
        global_mask = attention_mask.new_ones(bsz, num_groups, num_local_slots)
        # K/V mask: [global_mask, chunk_mask]
        kv_padding_mask = torch.cat([global_mask, attention_mask_chunks], dim=2)
        kv_padding_mask = kv_padding_mask.view(bsz * num_groups, kv_len)

        # Q mask: 只有 chunk_mask
        q_padding_mask = attention_mask_chunks.view(bsz * num_groups, group_size)
    else:
        q_padding_mask = attention_mask_chunks.view(bsz * num_groups, group_size)
        kv_padding_mask = q_padding_mask

    # ========== Step 8: Flash Attention (使用 kvpacked 格式) ==========
    # Pack K and V: [bsz * num_groups, kv_len, 2, nh, hd]
    kv = torch.stack([key_with_ctx, value_with_ctx], dim=2)

    nheads = query_with_ctx.shape[-2]

    # Unpad Q
    q_rearranged = rearrange(query_with_ctx, "b s h d -> b s (h d)")
    q_unpad, q_indices, cu_q_lens, max_q_len = unpad_input(q_rearranged, q_padding_mask)
    q_unpad = rearrange(q_unpad, "nnz (h d) -> nnz h d", h=nheads)

    # Unpad KV
    kv_rearranged = rearrange(kv, "b s two h d -> b s (two h d)")
    kv_unpad, kv_indices, cu_kv_lens, max_kv_len = unpad_input(
        kv_rearranged, kv_padding_mask
    )
    kv_unpad = rearrange(kv_unpad, "nnz (two h d) -> nnz two h d", two=2, h=nheads)

    # Flash attention with separate Q and KV
    output_unpad = flash_attn_varlen_kvpacked_func(
        q_unpad,
        kv_unpad,
        cu_q_lens,
        cu_kv_lens,
        max_q_len,
        max_kv_len,
        0.0,
        softmax_scale=None,
        causal=True,
    )

    # Pad output
    output = rearrange(
        pad_input(
            rearrange(output_unpad, "nnz h d -> nnz (h d)"),
            q_indices,
            bsz * num_groups,
            group_size,
        ),
        "b s (h d) -> b s h d",
        h=nheads,
    )
    # [bsz * num_groups, group_size, nh, hd]

    # ========== Step 9: Reshape output (直接就是 chunk tokens，无需提取) ==========
    output = output.view(bsz, num_groups, group_size, self.num_heads, self.head_dim)
    output = output.reshape(bsz, q_len, self.num_heads, self.head_dim)

    return self.o_proj(rearrange(output, "b s h d -> b s (h d)")), None, past_key_value


# endregion ===========================================================================


# ================================================================================
# 结合 LongLoRA S²-Attn + Hierarchical Memory (完整版本)
# 在每个 window 的 K/V 前直接拼接 global memory，实现真正的融合
# ================================================================================
# region ===========================================================================
def forward_flashattn_shifted_memory_v1(
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
    S²-Attn + Global Memory 的完整融合版本。

    与 v1 的区别：
    - v1: 先做 window attention，再加 global memory 修正
    - v2: 在每个 window 的 K/V 中直接拼接 global memory (真正融合)

    K/V 结构: [global_memory | window_tokens]
    Q 结构:   [window_tokens]

    每个 window 可以同时:
    1. 通过 window_tokens 与相邻 window 交换信息 (shift)
    2. 通过 global_memory 获取文档级上下文 (memory)
    """
    if not self.training:
        warnings.warn(
            "forward_flashattn_shifted_memory_v2 is for training only. "
            "For inference, use forward_flashattn_inference."
        )

    if output_attentions:
        warnings.warn("Output attentions is not supported, returning `None` instead.")

    bsz, q_len, hidden_size = hidden_states.size()

    # ========== Step 1: 提取 Global Memory ==========
    use_global_memory = hasattr(self, "local_constructor")
    use_global_integrator = hasattr(self, "global_integrator")

    group_size = int(q_len * group_size_ratio)
    if q_len % group_size > 0:
        raise ValueError(
            f"q_len {q_len} should be divisible by group size {group_size}."
        )
    num_groups = q_len // group_size

    if use_global_memory:
        if use_global_integrator:
            # 层级记忆
            chunks = hidden_states.view(bsz, num_groups, group_size, hidden_size)
            all_chunks = chunks.view(bsz * num_groups, group_size, hidden_size)

            if attention_mask is not None:
                attention_mask_chunks = attention_mask.view(bsz, num_groups, group_size)
                attention_mask_chunks_flat = attention_mask_chunks.view(
                    bsz * num_groups, group_size
                )
            else:
                attention_mask_chunks_flat = None

            local_memories = self.local_constructor(all_chunks, attention_mask_chunks_flat)
            num_local_slots = local_memories.shape[1]
            local_memories_stacked = local_memories.view(
                bsz, num_groups, num_local_slots, hidden_size
            )

            global_context = self.global_integrator(local_memories_stacked)
            num_local_slots = global_context.shape[1]
        else:
            global_context = self.local_constructor(hidden_states, attention_mask)
            num_local_slots = global_context.shape[1]
    else:
        global_context = None
        num_local_slots = 0

    # 打印配置
    if not hasattr(self, "_shifted_memory_v2_printed"):
        layer_idx = getattr(self, "layer_idx", 0)
        if rank == 0 and layer_idx == 0:
            print("\n" + "=" * 80)
            print(
                "🚀 forward_flashattn_shifted_memory_v2: True S²-Attn + Memory Fusion"
            )
            print("=" * 80)
            print(
                f"  Config: {num_groups} groups × {group_size} tokens, {self.num_heads} heads"
            )
            print(f"  Memory: {num_local_slots} global slots")
            print(f"  K/V = [global_memory({num_local_slots}) | window({group_size})]")
            print("=" * 80 + "\n")
        self._shifted_memory_v2_printed = True

    # ========== Step 2: Q/K/V 投影 ==========
    query_states = self.q_proj(hidden_states).view(
        bsz, q_len, self.num_heads, self.head_dim
    )
    key_states = self.k_proj(hidden_states).view(
        bsz, q_len, self.num_key_value_heads, self.head_dim
    )
    value_states = self.v_proj(hidden_states).view(
        bsz, q_len, self.num_key_value_heads, self.head_dim
    )

    # Global memory K/V (不应用 RoPE)
    if use_global_memory and global_context is not None:
        global_k = self.k_proj(global_context).view(
            bsz, num_local_slots, self.num_key_value_heads, self.head_dim
        )
        global_v = self.v_proj(global_context).view(
            bsz, num_local_slots, self.num_key_value_heads, self.head_dim
        )
    else:
        global_k = None
        global_v = None

    # ========== Step 3: RoPE ==========
    query_states = query_states.transpose(1, 2)  # [bsz, nh, q_len, hd]
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]

    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(
        query_states, key_states, cos, sin, position_ids
    )

    if past_key_value is not None:
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)
    past_key_value = (key_states, value_states) if use_cache else None

    # Repeat k/v heads
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)
    if global_k is not None:
        global_k = global_k.transpose(1, 2)  # [bsz, nkv, num_slots, hd]
        global_v = global_v.transpose(1, 2)
        global_k = repeat_kv(global_k, self.num_key_value_groups)
        global_v = repeat_kv(global_v, self.num_key_value_groups)

    # ========== Step 4: 分组处理 (S²-Attn 头分组 + Window 分组) ==========
    half_heads = self.num_heads // 2

    # 将序列分成 windows
    # Q: [bsz, nh, q_len, hd] -> [bsz, nh, num_groups, group_size, hd]
    query_chunks = query_states.view(
        bsz, self.num_heads, num_groups, group_size, self.head_dim
    )
    key_chunks = key_states.view(
        bsz, self.num_heads, num_groups, group_size, self.head_dim
    )
    value_chunks = value_states.view(
        bsz, self.num_heads, num_groups, group_size, self.head_dim
    )

    # 分成两组头
    # [bsz, nh, num_groups, group_size, hd] -> [bsz, 2, nh//2, num_groups, group_size, hd]
    query_chunks = query_chunks.view(
        bsz, 2, half_heads, num_groups, group_size, self.head_dim
    )
    key_chunks = key_chunks.view(
        bsz, 2, half_heads, num_groups, group_size, self.head_dim
    )
    value_chunks = value_chunks.view(
        bsz, 2, half_heads, num_groups, group_size, self.head_dim
    )

    if global_k is not None:
        # [bsz, nh, num_slots, hd] -> [bsz, 2, nh//2, num_slots, hd]
        global_k = global_k.view(bsz, 2, half_heads, num_local_slots, self.head_dim)
        global_v = global_v.view(bsz, 2, half_heads, num_local_slots, self.head_dim)

    # ========== Step 5: 处理 Group 2 的 shift ==========
    # Group 2 需要将序列偏移 group_size // 2
    shift_size = group_size // 2

    # Group 1: 正常顺序
    q_g1 = query_chunks[:, 0]  # [bsz, nh//2, num_groups, group_size, hd]
    k_g1 = key_chunks[:, 0]
    v_g1 = value_chunks[:, 0]

    # Group 2: 偏移处理
    # 需要将序列 roll 半个 window
    q_g2 = query_chunks[:, 1]
    k_g2 = key_chunks[:, 1]
    v_g2 = value_chunks[:, 1]

    # Roll 操作：将 Group 2 的序列偏移
    # 这实现了 shifted window 的效果
    # [bsz, nh//2, num_groups, group_size, hd] -> [bsz, nh//2, q_len, hd] -> roll -> reshape back
    q_g2_flat = q_g2.reshape(bsz, half_heads, q_len, self.head_dim)
    k_g2_flat = k_g2.reshape(bsz, half_heads, q_len, self.head_dim)
    v_g2_flat = v_g2.reshape(bsz, half_heads, q_len, self.head_dim)

    # Roll by shift_size (负数表示向左移动)
    q_g2_flat = torch.roll(q_g2_flat, shifts=-shift_size, dims=2)
    k_g2_flat = torch.roll(k_g2_flat, shifts=-shift_size, dims=2)
    v_g2_flat = torch.roll(v_g2_flat, shifts=-shift_size, dims=2)

    # Reshape back to groups
    q_g2 = q_g2_flat.view(bsz, half_heads, num_groups, group_size, self.head_dim)
    k_g2 = k_g2_flat.view(bsz, half_heads, num_groups, group_size, self.head_dim)
    v_g2 = v_g2_flat.view(bsz, half_heads, num_groups, group_size, self.head_dim)

    # ========== Step 6: 在每个 window 的 K/V 前拼接 global memory ==========
    if global_k is not None:
        # global_k: [bsz, 2, nh//2, num_slots, hd]
        global_k_g1 = global_k[:, 0]  # [bsz, nh//2, num_slots, hd]
        global_v_g1 = global_v[:, 0]
        global_k_g2 = global_k[:, 1]
        global_v_g2 = global_v[:, 1]

        # 扩展到每个 group
        # [bsz, nh//2, num_slots, hd] -> [bsz, nh//2, num_groups, num_slots, hd]
        global_k_g1 = global_k_g1.unsqueeze(2).expand(-1, -1, num_groups, -1, -1)
        global_v_g1 = global_v_g1.unsqueeze(2).expand(-1, -1, num_groups, -1, -1)
        global_k_g2 = global_k_g2.unsqueeze(2).expand(-1, -1, num_groups, -1, -1)
        global_v_g2 = global_v_g2.unsqueeze(2).expand(-1, -1, num_groups, -1, -1)

        # 拼接: K/V = [global_memory, window_tokens]
        k_g1 = torch.cat(
            [global_k_g1, k_g1], dim=3
        )  # [bsz, nh//2, num_groups, mem+grp, hd]
        v_g1 = torch.cat([global_v_g1, v_g1], dim=3)
        k_g2 = torch.cat([global_k_g2, k_g2], dim=3)
        v_g2 = torch.cat([global_v_g2, v_g2], dim=3)

        kv_len = num_local_slots + group_size
    else:
        kv_len = group_size

    # ========== Step 7: 准备 Flash Attention 输入 ==========
    # Reshape for flash_attn_func: (batch, seqlen, nheads, headdim)
    # 合并 batch 和 groups 维度

    # Q: [bsz, nh//2, num_groups, group_size, hd] -> [bsz*num_groups, group_size, nh//2, hd]
    q_g1 = q_g1.permute(0, 2, 3, 1, 4).reshape(
        bsz * num_groups, group_size, half_heads, self.head_dim
    )
    q_g2 = q_g2.permute(0, 2, 3, 1, 4).reshape(
        bsz * num_groups, group_size, half_heads, self.head_dim
    )

    # K/V: [bsz, nh//2, num_groups, kv_len, hd] -> [bsz*num_groups, kv_len, nh//2, hd]
    k_g1 = k_g1.permute(0, 2, 3, 1, 4).reshape(
        bsz * num_groups, kv_len, half_heads, self.head_dim
    )
    v_g1 = v_g1.permute(0, 2, 3, 1, 4).reshape(
        bsz * num_groups, kv_len, half_heads, self.head_dim
    )
    k_g2 = k_g2.permute(0, 2, 3, 1, 4).reshape(
        bsz * num_groups, kv_len, half_heads, self.head_dim
    )
    v_g2 = v_g2.permute(0, 2, 3, 1, 4).reshape(
        bsz * num_groups, kv_len, half_heads, self.head_dim
    )

    # ========== Step 8: Flash Attention ==========
    # 使用 flash_attn_func (更简单，不需要处理 padding)
    # 注意：这里假设没有 padding（训练时通常如此）
    # 如果有 padding，需要使用 varlen 版本

    # Group 1
    out_g1 = flash_attn_func(
        q_g1,
        k_g1,
        v_g1,
        dropout_p=0.0,
        softmax_scale=None,
        causal=True,  # Q tokens 只能看到 K/V 中 <= 自己位置的 tokens
    )  # [bsz*num_groups, group_size, nh//2, hd]

    # Group 2
    out_g2 = flash_attn_func(
        q_g2, k_g2, v_g2, dropout_p=0.0, softmax_scale=None, causal=True
    )

    # ========== Step 9: 恢复 Group 2 的 shift ==========
    # 将 Group 2 的输出 roll 回来
    out_g2 = out_g2.view(bsz, num_groups, group_size, half_heads, self.head_dim)
    out_g2 = out_g2.permute(0, 3, 1, 2, 4)  # [bsz, nh//2, num_groups, group_size, hd]
    out_g2_flat = out_g2.reshape(bsz, half_heads, q_len, self.head_dim)
    out_g2_flat = torch.roll(out_g2_flat, shifts=shift_size, dims=2)  # Roll back
    out_g2 = out_g2_flat.view(bsz, half_heads, num_groups, group_size, self.head_dim)

    # Group 1 reshape
    out_g1 = out_g1.view(bsz, num_groups, group_size, half_heads, self.head_dim)
    out_g1 = out_g1.permute(0, 3, 1, 2, 4)  # [bsz, nh//2, num_groups, group_size, hd]

    # ========== Step 10: 合并两组输出 ==========
    # out_g1, out_g2: [bsz, nh//2, num_groups, group_size, hd]
    # Stack: [bsz, 2, nh//2, num_groups, group_size, hd]
    output = torch.stack([out_g1, out_g2], dim=1)
    # Reshape: [bsz, nh, num_groups, group_size, hd] -> [bsz, q_len, nh, hd]
    output = output.view(bsz, self.num_heads, num_groups, group_size, self.head_dim)
    output = output.permute(0, 2, 3, 1, 4)  # [bsz, num_groups, group_size, nh, hd]
    output = output.reshape(bsz, q_len, self.num_heads, self.head_dim)

    return self.o_proj(rearrange(output, "b s h d -> b s (h d)")), None, past_key_value


# endregion ===========================================================================


# v2: S²-Attn + Global Memory (优化版，更早合并 batch 避免重复处理)
# ================================================================================
# region ===========================================================================
def forward_flashattn_hybrid_shift_v2(
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
    S²-Attn + Global Memory (优化版)，更早合并 batch 避免重复处理。

    核心优化：头分组后直接合并到 batch 维度 [bsz, 2, nh//2, L, hd] → [bsz*2, nh//2, L, hd]
    然后对后半部分 (Group 2) 做 shift，统一处理分 chunk、memory 拼接等。

    Q: [chunk_tokens]
    K/V: [memory_slots | chunk_tokens]
    """
    if not self.training:
        warnings.warn("Use forward_flashattn_inference for inference.")

    bsz, q_len, _ = hidden_states.size()
    group_size = int(q_len * group_size_ratio)
    num_groups = q_len // group_size
    half_heads = self.num_heads // 2
    shift_size = group_size // 2

    # ===== 1. Global Memory =====
    use_mem = hasattr(self, "local_constructor")
    if use_mem:
        global_ctx = self.local_constructor(hidden_states, attention_mask)
        M = global_ctx.shape[1]
    else:
        global_ctx, M = None, 0

    # 首次打印
    if not hasattr(self, "_hsv4_printed"):
        if rank == 0 and getattr(self, "layer_idx", 0) == 0:
            print(
                f"\n[Hybrid+Shift v4] groups={num_groups}, group_size={group_size}, "
                f"memory={M}, heads={self.num_heads}→2×{half_heads}, shift={shift_size}\n"
            )
        self._hsv4_printed = True

    # ===== 2. Q/K/V 投影 + RoPE =====
    Q = (
        self.q_proj(hidden_states)
        .view(bsz, q_len, self.num_heads, self.head_dim)
        .transpose(1, 2)
    )
    K = (
        self.k_proj(hidden_states)
        .view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        .transpose(1, 2)
    )
    V = (
        self.v_proj(hidden_states)
        .view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        .transpose(1, 2)
    )

    cos, sin = self.rotary_emb(V, seq_len=q_len)
    Q, K = apply_rotary_pos_emb(Q, K, cos, sin, position_ids)

    if past_key_value is not None:
        K = torch.cat([past_key_value[0], K], dim=2)
        V = torch.cat([past_key_value[1], V], dim=2)
    past_key_value = (K, V) if use_cache else None

    K = repeat_kv(K, self.num_key_value_groups)
    V = repeat_kv(V, self.num_key_value_groups)

    # Memory K/V (无 RoPE)
    if use_mem:
        Km = (
            self.k_proj(global_ctx)
            .view(bsz, M, self.num_key_value_heads, self.head_dim)
            .transpose(1, 2)
        )
        Vm = (
            self.v_proj(global_ctx)
            .view(bsz, M, self.num_key_value_heads, self.head_dim)
            .transpose(1, 2)
        )
        Km = repeat_kv(Km, self.num_key_value_groups)
        Vm = repeat_kv(Vm, self.num_key_value_groups)

    # ===== 3. 头分组 + batch 翻倍 (LongLoRA 核心) =====
    # [bsz, nh, L, hd] → [bsz, 2, nh//2, L, hd] → [bsz*2, nh//2, L, hd]
    Q = Q.view(bsz, 2, half_heads, q_len, self.head_dim).reshape(
        bsz * 2, half_heads, q_len, self.head_dim
    )
    K = K.view(bsz, 2, half_heads, q_len, self.head_dim).reshape(
        bsz * 2, half_heads, q_len, self.head_dim
    )
    V = V.view(bsz, 2, half_heads, q_len, self.head_dim).reshape(
        bsz * 2, half_heads, q_len, self.head_dim
    )

    if use_mem:
        Km = Km.view(bsz, 2, half_heads, M, self.head_dim).reshape(
            bsz * 2, half_heads, M, self.head_dim
        )
        Vm = Vm.view(bsz, 2, half_heads, M, self.head_dim).reshape(
            bsz * 2, half_heads, M, self.head_dim
        )

    # mask 也翻倍: [bsz, q_len] → [bsz*2, q_len]
    mask = attention_mask.repeat(2, 1)

    # ===== 4. Group 2 shift (只对后半部分 batch) =====
    Q[bsz:] = torch.roll(Q[bsz:], shifts=-shift_size, dims=2)
    K[bsz:] = torch.roll(K[bsz:], shifts=-shift_size, dims=2)
    V[bsz:] = torch.roll(V[bsz:], shifts=-shift_size, dims=2)
    mask[bsz:] = torch.roll(mask[bsz:], shifts=-shift_size, dims=1)

    # ===== 5. 分 chunk =====
    # [bsz*2, nh//2, L, hd] → [bsz*2, nh//2, num_groups, group_size, hd]
    Q = Q.view(bsz * 2, half_heads, num_groups, group_size, self.head_dim)
    K = K.view(bsz * 2, half_heads, num_groups, group_size, self.head_dim)
    V = V.view(bsz * 2, half_heads, num_groups, group_size, self.head_dim)

    # ===== 6. 每个 chunk 的 K/V = [memory | chunk] =====
    if use_mem:
        # 扩展到每个 chunk: [bsz*2, nh//2, M, hd] → [bsz*2, nh//2, num_groups, M, hd]
        Km = Km.unsqueeze(2).expand(-1, -1, num_groups, -1, -1)
        Vm = Vm.unsqueeze(2).expand(-1, -1, num_groups, -1, -1)
        # 拼接: K/V = [memory | chunk]
        K = torch.cat([Km, K], dim=3)  # [bsz*2, nh//2, num_groups, M+group_size, hd]
        V = torch.cat([Vm, V], dim=3)
        kv_len = M + group_size
    else:
        kv_len = group_size

    # ===== 7. Reshape for Flash Attention =====
    # [bsz*2, nh//2, num_groups, L, hd] → [bsz*2*num_groups, L, nh//2, hd]
    batch_all = bsz * 2 * num_groups
    Q = Q.permute(0, 2, 3, 1, 4).reshape(
        batch_all, group_size, half_heads, self.head_dim
    )
    K = K.permute(0, 2, 3, 1, 4).reshape(batch_all, kv_len, half_heads, self.head_dim)
    V = V.permute(0, 2, 3, 1, 4).reshape(batch_all, kv_len, half_heads, self.head_dim)

    # ===== 8. 构建 Padding Mask =====
    # Q mask: [bsz*2, q_len] → [bsz*2, num_groups, group_size] → [bsz*2*num_groups, group_size]
    q_mask = mask.view(bsz * 2, num_groups, group_size).reshape(batch_all, group_size)

    # KV mask: [global_mask(全1), chunk_mask]
    if use_mem:
        global_mask = attention_mask.new_ones(bsz * 2, num_groups, M)
        chunk_mask = mask.view(bsz * 2, num_groups, group_size)
        kv_mask = torch.cat([global_mask, chunk_mask], dim=2).reshape(batch_all, kv_len)
    else:
        kv_mask = q_mask

    # ===== 9. Flash Attention =====
    kv = torch.stack([K, V], dim=2)  # [batch_all, kv_len, 2, nh//2, hd]

    q_flat = rearrange(Q, "b s h d -> b s (h d)")
    q_unpad, q_indices, cu_q_lens, max_q_len = unpad_input(q_flat, q_mask)
    q_unpad = rearrange(q_unpad, "nnz (h d) -> nnz h d", h=half_heads)

    kv_flat = rearrange(kv, "b s two h d -> b s (two h d)")
    kv_unpad, _, cu_kv_lens, max_kv_len = unpad_input(kv_flat, kv_mask)
    kv_unpad = rearrange(kv_unpad, "nnz (two h d) -> nnz two h d", two=2, h=half_heads)

    out_unpad = flash_attn_varlen_kvpacked_func(
        q_unpad,
        kv_unpad,
        cu_q_lens,
        cu_kv_lens,
        max_q_len,
        max_kv_len,
        dropout_p=0.0,
        softmax_scale=None,
        causal=True,
    )

    out = pad_input(
        rearrange(out_unpad, "nnz h d -> nnz (h d)"), q_indices, batch_all, group_size
    )
    out = rearrange(out, "b s (h d) -> b s h d", h=half_heads)
    # [bsz*2*num_groups, group_size, nh//2, hd]

    # ===== 10. Reshape back + Group 2 roll 回来 =====
    # [bsz*2*num_groups, group_size, nh//2, hd] → [bsz*2, nh//2, L, hd]
    out = out.view(bsz * 2, num_groups, group_size, half_heads, self.head_dim)
    out = out.permute(0, 3, 1, 2, 4).reshape(bsz * 2, half_heads, q_len, self.head_dim)

    # Group 2 roll 回来 (只对后半部分 batch)
    out[bsz:] = torch.roll(out[bsz:], shifts=shift_size, dims=2)

    # ===== 11. 合并两组头 =====
    # [bsz*2, nh//2, L, hd] → [bsz, 2, nh//2, L, hd] → [bsz, L, nh, hd]
    out = out.view(bsz, 2, half_heads, q_len, self.head_dim)
    out = out.permute(0, 3, 1, 2, 4).reshape(bsz, q_len, self.num_heads, self.head_dim)

    return self.o_proj(rearrange(out, "b s h d -> b s (h d)")), None, past_key_value


# endregion ===========================================================================


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
    use_higher_global: bool = True,  # 是否使用高层全局记忆
    use_local_memory: bool = True,  # 是否使用局部记忆（默认 False 避免冗余）
    use_recurrence_cache: bool = False,  # 是否使用 recurrence cache（Transformer-XL style）
    recurrence_size: Optional[int] = 128,  # recurrence cache 大小
    # group_size_ratio: Optional[float] = 0.25,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """
    Hierarchical memory with cache support (优化版本).

    整合了以下功能：
    1. ✅ Recurrence cache (Transformer-XL style)
    2. ✅ Hierarchical memory (LocalConstructor + HierarchicalMemoryAggregator)
    3. ✅ Ablation modes (use_higher_global, use_local_memory)
    4. ✅ 所有逻辑通过参数控制

    ⚠️ 关键优化 (参考 forward_flashattn_hybrid):
    - Q: 只包含 [chunk]，不拼接 memory（省计算！）
    - K/V: [higher_global?, local?, cache?, chunk]
    - 输出直接就是 chunk tokens，无需提取

    ⚠️ CRITICAL: cache 必须紧挨着 chunk，因为它们在位置上是连续的！

    拼接顺序规则：
    - 位置无关的组件（higher_global, local）放在前面
    - cache 必须紧挨着 chunk（位置连续）

    三种 Ablation 模式：
    - Mode 1 (推荐): use_higher_global=True, use_local_memory=False
      Q:   [chunk]
      K/V: [higher_global, cache, chunk]

    - Mode 2: use_higher_global=False, use_local_memory=True
      Q:   [chunk]
      K/V: [local_i, cache, chunk]

    - Mode 3: use_higher_global=True, use_local_memory=True
      Q:   [chunk]
      K/V: [higher_global, local_i, cache, chunk]

    Args:
        use_higher_global: 是否使用高层全局记忆（聚合所有 local memories）
        use_local_memory: 是否使用局部记忆（每个 chunk 的压缩表示）
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
            print("📋 Hierarchical Memory Ablation Configuration")
            print("=" * 80)
            print(f"  ✅ use_higher_global    : {use_higher_global}  (高层全局记忆)")
            print(
                f"  {'✅' if use_local_memory else '❌'} use_local_memory    : {use_local_memory}  (局部记忆)"
            )
            print(
                f"  ✅ use_recurrence_cache : {use_recurrence_cache}  (Recurrence cache)"
            )
            print()

            # 显示当前 Ablation Mode (优化版: Q 只包含 chunk)
            if use_higher_global and not use_local_memory and use_recurrence_cache:
                print("📌 Current Mode: Mode 1 (推荐)")
                print("   Q:   [chunk]")
                print("   K/V: [higher_global, cache, chunk]")
                print("   优势: 无冗余，信息高度聚合，Q 更短")
            elif not use_higher_global and use_local_memory and use_recurrence_cache:
                print("📌 Current Mode: Mode 2")
                print("   Q:   [chunk]")
                print("   K/V: [local_i, cache, chunk]")
                print("   优势: 每个 chunk 有专属压缩表示")
            elif use_higher_global and use_local_memory and use_recurrence_cache:
                print("📌 Current Mode: Mode 3 (完整模式)")
                print("   Q:   [chunk]")
                print("   K/V: [higher_global, local_i, cache, chunk]")
                print("   优势: 全部特征")
            else:
                print("📌 Current Mode: Custom")
                print(
                    f"   配置: higher_global={use_higher_global}, local={use_local_memory}, cache={use_recurrence_cache}"
                )
                print("   Q:   [chunk]")
                print("   K/V: [memory?, cache?, chunk]")

            print("=" * 80 + "\n", flush=True)

        self._ablation_config_printed = True

    # ========== Step 1: 分 chunk ==========
    group_size = int(q_len * group_size_ratio)
    if q_len % group_size > 0:
        raise ValueError(
            f"q_len {q_len} should be divisible by group size {group_size}."
        )
    if not hasattr(self, "_group_size_printed"):
        layer_idx = getattr(self, "layer_idx", 0)  # 获取当前层索引
        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank == 0 and layer_idx == 0:
            print(
                f"[forward_flashattn_hierarchical_with_cache] group_size_ratio={group_size_ratio}, group_size={group_size}"
            )
        self._group_size_printed = True

    num_groups = q_len // group_size

    # Reshape into chunks: [bsz, num_groups, group_size, hidden_size]
    chunks = hidden_states.view(bsz, num_groups, group_size, hidden_size)

    # ========== Step 2: 提取局部记忆（对每个 chunk 提取压缩表示）==========
    # 使用 LocalConstructorFlash 对每个 chunk 单独提取局部全局表示
    # ⚠️ CRITICAL: Check if global_memory exists before using it!
    if (use_higher_global or use_local_memory) and hasattr(self, "local_constructor"):
        # 批处理所有 chunks（并行）
        all_chunks = chunks.view(bsz * num_groups, group_size, hidden_size)

        # attention_mask: [bsz, q_len] -> [bsz * num_groups, group_size]
        if attention_mask is not None:
            attention_mask_chunks = attention_mask.view(bsz, num_groups, group_size)
            attention_mask_chunks = attention_mask_chunks.view(
                bsz * num_groups, group_size
            )
        else:
            attention_mask_chunks = None

        # all_local_mems = self.local_constructor(
        #     all_chunks, attention_mask_chunks
        # )  # [bsz * num_groups, num_slots, hidden_size]
        all_local_mems = self.local_constructor(all_chunks)  # zxy

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
    _causal_mode = CAUSAL_MEMORY_MODE
    _is_causal = _causal_mode in ("causal_gi", "causal_shift", "causal_shift_g", "causal_gi_gonly")

    if (
        use_higher_global
        and hasattr(self, "global_integrator")
        and local_memories_stacked is not None
    ):
        if _is_causal and hasattr(self.global_integrator, "forward_causal"):
            higher_global_per_group = self.global_integrator.forward_causal(
                local_memories_stacked
            )
            num_global_slots = higher_global_per_group.shape[2]

            if _causal_mode in ("causal_shift", "causal_shift_g"):
                zeros_g = torch.zeros(
                    bsz, 1, num_global_slots, hidden_size,
                    device=higher_global_per_group.device,
                    dtype=higher_global_per_group.dtype,
                )
                higher_global_per_group = torch.cat(
                    [zeros_g, higher_global_per_group[:, :-1, :, :]], dim=1
                )

            higher_global = None
        else:
            higher_global = self.global_integrator(local_memories_stacked)
            num_global_slots = higher_global.shape[1]
            higher_global_per_group = None
    else:
        higher_global = None
        higher_global_per_group = None
        num_global_slots = 0

    # causal_shift_g / causal_gi_gonly: 不拼接 L，仅用 G
    if _causal_mode in ("causal_shift_g", "causal_gi_gonly"):
        local_memories_stacked = None
        num_local_slots = 0

    if (
        _causal_mode == "causal_shift"
        and use_local_memory
        and local_memories_stacked is not None
    ):
        zeros_l = torch.zeros(
            bsz, 1, num_local_slots, hidden_size,
            device=local_memories_stacked.device,
            dtype=local_memories_stacked.dtype,
        )
        local_memories_stacked = torch.cat(
            [zeros_l, local_memories_stacked[:, :-1, :, :]], dim=1
        )

    # ========== Step 4: Q/K/V 投影 (优化：合并 higher_global 和 hidden_states 的 K/V 投影) ==========
    # 参考 forward_flashattn_hybrid 的优化技巧
    # Q: 只投影 hidden_states
    query_states = (
        self.q_proj(hidden_states)
        .view(bsz, q_len, self.num_heads, self.head_dim)
        .transpose(1, 2)
    )  # [bsz, nh, q_len, hd]

    # K/V: 合并投影 (省显存的关键！)
    higher_global_k_per_group = higher_global_v_per_group = None

    if use_higher_global and higher_global is not None:
        # 非因果模式：拼接后一起投影 [higher_global, hidden_states]
        combined_input = torch.cat(
            [higher_global, hidden_states], dim=1
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

        higher_global_k = combined_k[:, :, :num_global_slots, :]
        key_states = combined_k[:, :, num_global_slots:, :]
        higher_global_v = combined_v[:, :, :num_global_slots, :]
        value_states = combined_v[:, :, num_global_slots:, :]
    elif use_higher_global and higher_global_per_group is not None:
        # 因果模式：per-group G 单独投影
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
        higher_global_k = higher_global_v = None

        hg_flat = higher_global_per_group.view(
            bsz * num_groups, num_global_slots, hidden_size
        )
        hg_k_flat = (
            self.k_proj(hg_flat)
            .view(bsz * num_groups, num_global_slots, self.num_key_value_heads, self.head_dim)
            .transpose(1, 2)
        )
        hg_v_flat = (
            self.v_proj(hg_flat)
            .view(bsz * num_groups, num_global_slots, self.num_key_value_heads, self.head_dim)
            .transpose(1, 2)
        )
        # repeat_kv after RoPE section below
        higher_global_k_per_group_raw = hg_k_flat
        higher_global_v_per_group_raw = hg_v_flat
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
        higher_global_k = higher_global_v = None

    # RoPE (只对 sequence 部分，不对 higher_global)
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
    # higher_global_k, higher_global_v 不应用 RoPE（记忆是位置无关的）

    # Past Key value support
    if past_key_value is not None:
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)

    past_key_value = (key_states, value_states) if use_cache else None

    # Repeat k/v heads if n_kv_heads < n_heads
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)
    if higher_global_k is not None:
        higher_global_k = repeat_kv(higher_global_k, self.num_key_value_groups)
        higher_global_v = repeat_kv(higher_global_v, self.num_key_value_groups)

    # 因果模式下的 per-group G: repeat_kv + reshape
    if higher_global_k_per_group is None and higher_global_per_group is not None:
        # higher_global_k_per_group_raw was set in Step 4 above
        higher_global_k_per_group_raw = repeat_kv(higher_global_k_per_group_raw, self.num_key_value_groups)
        higher_global_v_per_group_raw = repeat_kv(higher_global_v_per_group_raw, self.num_key_value_groups)
        higher_global_k_per_group = higher_global_k_per_group_raw.view(
            bsz, num_groups, self.num_heads, num_global_slots, self.head_dim
        )
        higher_global_v_per_group = higher_global_v_per_group_raw.view(
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
    # 优化: 批处理所有 local memories，避免循环调用 projection
    # 只投影 K/V，不投影 Q
    if use_local_memory and local_memories_stacked is not None:
        # Reshape: [bsz, num_groups, num_slots, hidden] -> [bsz*num_groups, num_slots, hidden]
        local_mems_flat = local_memories_stacked.view(
            bsz * num_groups, num_local_slots, hidden_size
        )

        # 一次性投影所有 local memories 的 K/V (批处理，1 次调用 vs num_groups 次!)
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
        local_k_all = local_k_flat.view(
            bsz, num_groups, self.num_heads, num_local_slots, self.head_dim
        )
        local_v_all = local_v_flat.view(
            bsz, num_groups, self.num_heads, num_local_slots, self.head_dim
        )

    # ========== Step 7: 向量化构建 Q 和 K/V (参考 forward_flashattn_hybrid) ==========
    # 优化: 使用 expand + cat 替代循环，避免 Python 循环开销
    # Q: 只包含 chunk tokens
    # K/V: [higher_global?, local?, cache?, chunk]

    kv_components_k = []
    kv_components_v = []

    # 7.1 Higher global memory
    if use_higher_global and higher_global_k is not None:
        # 非因果模式：所有 chunks 共享同一个 G
        higher_global_k_exp = higher_global_k.unsqueeze(2).expand(
            -1, -1, num_groups, -1, -1
        )
        higher_global_v_exp = higher_global_v.unsqueeze(2).expand(
            -1, -1, num_groups, -1, -1
        )
        kv_components_k.append(higher_global_k_exp)
        kv_components_v.append(higher_global_v_exp)
    elif use_higher_global and higher_global_k_per_group is not None:
        # 因果模式：每个 chunk 有独立的 G_i
        kv_components_k.append(higher_global_k_per_group.permute(0, 2, 1, 3, 4))
        kv_components_v.append(higher_global_v_per_group.permute(0, 2, 1, 3, 4))

    # 7.2 Local memories (每个 chunk 不同，直接 permute)
    if use_local_memory and local_memories_stacked is not None:
        # local_k_all: [bsz, num_groups, nh, num_local_slots, hd]
        # permute to [bsz, nh, num_groups, num_local_slots, hd]
        local_k_exp = local_k_all.permute(0, 2, 1, 3, 4)
        local_v_exp = local_v_all.permute(0, 2, 1, 3, 4)
        kv_components_k.append(local_k_exp)
        kv_components_v.append(local_v_exp)

    # 7.3 Recurrence cache (向量化构建)
    if use_recurrence_cache:
        # key_chunks: [bsz, nh, num_groups, group_size, hd]
        # 取每个 chunk 的尾部作为下一个 chunk 的 cache
        chunk_tails_k = key_chunks[
            :, :, :, -recurrence_size:, :
        ]  # [bsz, nh, num_groups, recurrence_size, hd]
        chunk_tails_v = value_chunks[:, :, :, -recurrence_size:, :]

        # 构建 cache: chunk_0 用 zeros，chunk_i (i>0) 用 chunk_{i-1} 的尾部
        # [zeros, chunk_0_tail, chunk_1_tail, ..., chunk_{n-2}_tail]
        dummy = torch.zeros(
            bsz,
            self.num_heads,
            1,  # 只为 chunk_0
            recurrence_size,
            self.head_dim,
            device=key_states.device,
            dtype=key_states.dtype,
        )
        # chunk_tails[:, :, :-1, :, :] 是 chunk_0 到 chunk_{n-2} 的尾部，作为 chunk_1 到 chunk_{n-1} 的 cache
        cache_k = torch.cat(
            [dummy, chunk_tails_k[:, :, :-1, :, :]], dim=2
        )  # [bsz, nh, num_groups, recurrence_size, hd]
        cache_v = torch.cat([dummy, chunk_tails_v[:, :, :-1, :, :]], dim=2)
        kv_components_k.append(cache_k)
        kv_components_v.append(cache_v)

    # 7.4 Chunk tokens (必须在最后)
    kv_components_k.append(key_chunks)
    kv_components_v.append(value_chunks)

    # 拼接 K/V: [bsz, nh, num_groups, total_kv_len, hd]
    key_with_ctx = torch.cat(kv_components_k, dim=3)
    value_with_ctx = torch.cat(kv_components_v, dim=3)

    # Q: 直接使用 query_chunks [bsz, nh, num_groups, group_size, hd]

    # 计算长度
    q_len_per_chunk = group_size  # Q 只包含 chunk
    kv_len_per_chunk = key_with_ctx.shape[3]  # memory + cache + chunk

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
    # 优化: Q 只包含 chunk，mask 构建大幅简化

    # Reshape chunk masks: [bsz, num_groups, group_size]
    chunk_masks_reshaped = attention_mask.view(bsz, num_groups, group_size)

    # 9.1 构建 Q padding masks
    # 优化: Q 只包含 chunk tokens，所以 Q mask 直接就是 chunk mask
    # ⚠️ 关键：不要 transpose！数据是按 batch 优先排列的
    all_masks_q_flat = chunk_masks_reshaped.reshape(bsz * num_groups, q_len_per_chunk)

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
    if use_local_memory:
        all_masks_kv_stacked[:, :, offset : offset + num_local_slots] = 1
        offset += num_local_slots

    # causal_shift/causal_shift_g: segment_0 的 G（和 L）是零填充，mask 掉
    if _causal_mode in ("causal_shift", "causal_shift_g"):
        mem_offset = 0
        if use_higher_global:
            all_masks_kv_stacked[:, 0, mem_offset : mem_offset + num_global_slots] = 0
            mem_offset += num_global_slots
        if use_local_memory:
            all_masks_kv_stacked[:, 0, mem_offset : mem_offset + num_local_slots] = 0

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

    # ⚠️ 关键：不要 transpose！数据是按 batch 优先排列的
    all_masks_kv_flat = all_masks_kv_stacked.reshape(bsz * num_groups, kv_len_per_chunk)

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
    )  # [bsz*num_groups, q_len_per_chunk, nh, hd]

    # Reshape from [bsz*num_groups, q_len_per_chunk, nh, hd] to [bsz, q_len, nh, hd]
    # ⚠️ 关键：必须是 view(bsz, num_groups, ...) 而不是 view(num_groups, bsz, ...)
    # 因为输入是按 batch 优先排列的: [batch0_group0, batch0_group1, ..., batch1_group0, ...]
    output = output.view(bsz, num_groups, group_size, self.num_heads, self.head_dim)
    output = output.view(bsz, q_len, self.num_heads, self.head_dim)

    # Output projection
    attn_output = self.o_proj(rearrange(output, "b s h d -> b s (h d)"))

    return attn_output, None, past_key_value


# 训练way3.5 NEW: Global Memory + Recurrence Cache (简化版，无层级聚合) 已修正
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
    # 直接在这个函数中控制的参数
    use_recurrence_cache: bool = True,  # 是否使用 recurrence cache（Transformer-XL style）
    recurrence_size: Optional[int] = 128,  # recurrence cache 大小
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """
    Global Memory + Recurrence Cache (简化版本).

    基于 forward_flashattn_hierarchical_with_cache 简化而来：
    - 移除了 higher_global（层级聚合）
    - 移除了分 chunk 提取 local_memories
    - 改为对整个输入提取一个全局记忆（像 forward_flashattn_hybrid 一样）
    - 保留了 recurrence cache 机制

    结构:
    - Q:   [chunk]  (不包含 memory)
    - K/V: [global_memory, cache?, chunk]

    优势:
    - 比 hierarchical 版本更简单，参数更少
    - 全局记忆对整个输入提取，信息更全面
    - 保留 cache 机制支持跨 chunk 信息传递

    Args:
        use_recurrence_cache: 是否使用 Transformer-XL style 的 recurrence cache
        recurrence_size: recurrence cache 大小
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

    # ✅ 打印配置（只在第一次调用时打印，且只在 rank 0 和 layer 0 打印）
    if not hasattr(self, "_global_cache_config_printed"):
        rank = dist.get_rank() if dist.is_initialized() else 0
        layer_idx = getattr(self, "layer_idx", 0)

        if rank == 0 and layer_idx == 0:
            print("\n" + "=" * 80)
            print("📋 Global Memory + Cache Configuration")
            print("=" * 80)
            print(
                f"  ✅ use_recurrence_cache : {use_recurrence_cache}  (Recurrence cache)"
            )
            print(f"  📊 recurrence_size      : {recurrence_size}")
            print()
            print("📌 Current Mode: Global Memory + Cache")
            print("   Q:   [chunk]")
            print("   K/V: [global_memory, cache?, chunk]")
            print("   全局记忆对整个输入提取，所有 chunks 共享")
            print("=" * 80 + "\n", flush=True)

        self._global_cache_config_printed = True

    # ========== Step 1: 提取全局记忆（对整个输入） ==========
    use_global_memory = hasattr(self, "local_constructor")

    if use_global_memory:
        global_context = self.local_constructor(
            hidden_states, attention_mask
        )  # [bsz, num_local_slots, hidden_size]
        num_local_slots = global_context.shape[1]
    else:
        global_context = None
        num_local_slots = 0

    # ========== Step 2: 分 chunk ==========
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

    # 检查 recurrence_size 是否合理
    if use_recurrence_cache and recurrence_size > group_size:
        raise ValueError(
            f"recurrence_size ({recurrence_size}) should be <= group_size ({group_size})"
        )

    # ========== Step 3: Q/K/V 投影 (优化：合并投影) ==========
    # Q: 只投影 hidden_states
    query_states = (
        self.q_proj(hidden_states)
        .view(bsz, q_len, self.num_heads, self.head_dim)
        .transpose(1, 2)
    )  # [bsz, nh, q_len, hd]

    # K/V: 合并投影 (省显存的关键！)
    if use_global_memory and global_context is not None:
        # 拼接后一起投影: [global_context, hidden_states]
        combined_input = torch.cat(
            [global_context, hidden_states], dim=1
        )  # [bsz, num_local_slots + q_len, hidden_size]

        # K 投影
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

        # V 投影
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

        # 分离 global 和 sequence 部分 (slice 操作，0 额外显存)
        global_k = combined_k[:, :, :num_local_slots, :]
        key_states = combined_k[:, :, num_local_slots:, :]
        global_v = combined_v[:, :, :num_local_slots, :]
        value_states = combined_v[:, :, num_local_slots:, :]
    else:
        # 不使用 global memory，直接投影 hidden_states
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

    # ========== Step 4: RoPE (只对 sequence 部分，不对 global) ==========
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
    # global_k, global_v 不应用 RoPE（记忆是位置无关的）

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

    # ========== Step 6: 向量化构建 Q 和 K/V ==========
    # Q: 只包含 chunk tokens
    # K/V: [global_memory?, cache?, chunk]

    kv_components_k = []
    kv_components_v = []

    # 6.1 Global memory (对所有 chunks 相同，使用 expand 广播)
    if use_global_memory and global_k is not None:
        # global_k: [bsz, nh, num_local_slots, hd]
        # expand to [bsz, nh, num_groups, num_local_slots, hd]
        global_k_exp = global_k.unsqueeze(2).expand(-1, -1, num_groups, -1, -1)
        global_v_exp = global_v.unsqueeze(2).expand(-1, -1, num_groups, -1, -1)
        kv_components_k.append(global_k_exp)
        kv_components_v.append(global_v_exp)

    # 6.2 Recurrence cache (向量化构建)
    if use_recurrence_cache:
        # key_chunks: [bsz, nh, num_groups, group_size, hd]
        # 取每个 chunk 的尾部作为下一个 chunk 的 cache
        chunk_tails_k = key_chunks[:, :, :, -recurrence_size:, :]
        chunk_tails_v = value_chunks[:, :, :, -recurrence_size:, :]

        # 构建 cache: chunk_0 用 zeros，chunk_i (i>0) 用 chunk_{i-1} 的尾部
        dummy = torch.zeros(
            bsz,
            self.num_heads,
            1,  # 只为 chunk_0
            recurrence_size,
            self.head_dim,
            device=key_states.device,
            dtype=key_states.dtype,
        )
        cache_k = torch.cat([dummy, chunk_tails_k[:, :, :-1, :, :]], dim=2)
        cache_v = torch.cat([dummy, chunk_tails_v[:, :, :-1, :, :]], dim=2)
        kv_components_k.append(cache_k)
        kv_components_v.append(cache_v)

    # 6.3 Chunk tokens (必须在最后)
    kv_components_k.append(key_chunks)
    kv_components_v.append(value_chunks)

    # 拼接 K/V: [bsz, nh, num_groups, total_kv_len, hd]
    key_with_ctx = torch.cat(kv_components_k, dim=3)
    value_with_ctx = torch.cat(kv_components_v, dim=3)

    # 计算长度
    q_len_per_chunk = group_size  # Q 只包含 chunk
    kv_len_per_chunk = key_with_ctx.shape[3]  # memory + cache + chunk

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

    # Q mask: 只包含 chunk mask
    all_masks_q_flat = chunk_masks_reshaped.reshape(bsz * num_groups, q_len_per_chunk)

    # K/V mask: [global_memory?, cache?, chunk]
    all_masks_kv_stacked = torch.empty(
        bsz,
        num_groups,
        kv_len_per_chunk,
        dtype=attention_mask.dtype,
        device=attention_mask.device,
    )

    offset = 0
    # Global memory mask (全 1)
    if use_global_memory and global_k is not None:
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


# 训练way4 NEW: Hierarchical Memory without Cache
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
    use_higher_global: bool = True,  # 是否使用高层全局记忆
    use_local_memory: bool = True,  # 是否使用局部记忆（默认 False 避免冗余）
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """
    Hierarchical memory (简化版本，无 recurrence cache).

    整合了以下功能：
    1. Hierarchical memory (LocalConstructor + HierarchicalMemoryAggregator)
    2. Ablation modes (use_higher_global, use_local_memory)

    优化：Q 只包含 chunk tokens，K/V 包含 [memories, chunk]
    - 节省计算：memories 不参与 Q 计算
    - chunk tokens 可以 attend 到 memories（通过 K/V）
    - 输出直接就是 chunk tokens，无需额外提取

    拼接顺序：
    - Q:   [chunk]
    - K/V: [higher_global?, local?, chunk]

    三种 Ablation 模式：
    - Mode 1 (推荐): use_higher_global=True, use_local_memory=False
      Q: [chunk], K/V: [higher_global, chunk]

    - Mode 2: use_higher_global=False, use_local_memory=True
      Q: [chunk], K/V: [local_i, chunk]

    - Mode 3: use_higher_global=True, use_local_memory=True
      Q: [chunk], K/V: [higher_global, local_i, chunk]

    Args:
        use_higher_global: 是否使用高层全局记忆（聚合所有 local memories）
        use_local_memory: 是否使用局部记忆（每个 chunk 的压缩表示）
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
            print("Hierarchical Memory (Optimized: Q=[chunk], K/V=[memories,chunk])")
            print("=" * 80)
            print(f"  use_higher_global : {use_higher_global}")
            print(f"  use_local_memory  : {use_local_memory}")

            if use_higher_global and not use_local_memory:
                print("  Mode 1: Q=[chunk], K/V=[higher_global, chunk]")
            elif not use_higher_global and use_local_memory:
                print("  Mode 2: Q=[chunk], K/V=[local_i, chunk]")
            elif use_higher_global and use_local_memory:
                print("  Mode 3: Q=[chunk], K/V=[higher_global, local_i, chunk]")
            else:
                print("  Baseline: Q=K/V=[chunk]")

            # 因果模式提示
            if CAUSAL_MEMORY_MODE != "none":
                print(f"  🔒 CAUSAL_MEMORY_MODE: {CAUSAL_MEMORY_MODE}")
                if CAUSAL_MEMORY_MODE == "causal_gi":
                    print("     segment_i uses G_i=Agg(L_1..L_i) + L_i")
                elif CAUSAL_MEMORY_MODE == "causal_shift":
                    print("     segment_i uses G_{i-1}=Agg(L_1..L_{i-1}) + L_{i-1}")
                elif CAUSAL_MEMORY_MODE == "causal_shift_g":
                    print("     segment_i uses G_{i-1}=Agg(L_1..L_{i-1}) only (no L)")
                elif CAUSAL_MEMORY_MODE == "causal_gi_gonly":
                    print("     segment_i uses G_i=Agg(L_1..L_i) only (no L in KV, double bottleneck)")

            print("=" * 80 + "\n", flush=True)

        self._hierarchical_no_cache_printed = True

    # ========== Step 1: 分 chunk ==========
    # 🔥 混合分组训练：随机选择分组数
    # 重要：确保同一个 forward pass 中所有层使用相同的分组！
    global _mixed_group_current_ratio, _mixed_group_call_count

    layer_idx = getattr(self, "layer_idx", 0)

    if self.training and MIXED_GROUP_TRAINING:
        # Layer 0 时选择新的分组，后续层复用
        if layer_idx == 0:
            _mixed_group_current_ratio = random.choice(GROUP_SIZE_RATIOS)
            _mixed_group_call_count += 1
            # 每 100 个 batch 打印一次，确认混合分组在工作
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
        # 评估模式：使用固定的 segment_size（与训练一致）
        group_size = FIXED_SEGMENT_SIZE
        # 处理 q_len 不能被 segment_size 整除的情况
        if q_len < group_size:
            # 输入太短，使用整个序列作为一个组
            group_size = q_len
    else:
        current_ratio = group_size_ratio
        group_size = int(q_len * current_ratio)

    # 确保 group_size 至少为 1
    group_size = max(1, group_size)

    # 处理不能整除的情况：调整 group_size 或截断
    if q_len % group_size > 0:
        # 向下取整到最近的可整除大小
        num_complete_groups = q_len // group_size
        if num_complete_groups == 0:
            group_size = q_len  # 使用整个序列
        # 注意：多余的 tokens 会在后面被处理

    if not hasattr(self, "_hierarchical_group_printed"):
        local_rank = dist.get_rank() if dist.is_initialized() else 0
        if local_rank == 0 and layer_idx == 0:
            # 根据实际使用的逻辑打印
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

    # ========== Step 2: 提取局部记忆（对每个 chunk 提取压缩表示）==========
    # 使用 LocalConstructorFlash 对每个 chunk 单独提取局部全局表示
    if (use_higher_global or use_local_memory) and hasattr(self, "local_constructor"):
        # 批处理所有 chunks（并行）
        all_chunks = chunks.view(bsz * num_groups, group_size, hidden_size)

        # 🔥 方案3: 确保使用 bfloat16 进行 memory extraction（节省显存）
        # 原始代码：直接使用 all_chunks，可能是 float32
        # if all_chunks.dtype == torch.float32:
        #     all_chunks = all_chunks.to(torch.bfloat16)
        # 优化：强制转换为 bfloat16（如果使用 mixed precision training）
        original_dtype = all_chunks.dtype
        if all_chunks.dtype == torch.float32:
            all_chunks = all_chunks.to(torch.bfloat16)

        # [bsz, num_groups, group_size] -> [bsz * num_groups, group_size]
        attention_mask_chunks = chunk_masks_reshaped.view(bsz * num_groups, group_size)
        all_local_mems = self.local_constructor(
            all_chunks, attention_mask_chunks
        )  # [bsz * num_groups, num_slots, hidden_size]

        # 🔥 如果原始是 float32，转换回来保持一致性
        if original_dtype == torch.float32:
            all_local_mems = all_local_mems.to(torch.float32)

        # Reshape back: [bsz, num_groups, num_slots, hidden_size]
        num_local_slots = all_local_mems.shape[1]
        local_memories_stacked = all_local_mems.view(
            bsz, num_groups, num_local_slots, hidden_size
        )
    else:
        num_local_slots = 0
        local_memories_stacked = None

    # ========== Step 3: 聚合到高层全局记忆（可选）==========
    # 根据 CAUSAL_MEMORY_MODE 选择因果模式
    _causal_mode = CAUSAL_MEMORY_MODE  # "none", "causal_gi", "causal_shift", "causal_shift_g"
    _is_causal = _causal_mode in ("causal_gi", "causal_shift", "causal_shift_g", "causal_gi_gonly")

    if (
        use_higher_global
        and hasattr(self, "global_integrator")
        and local_memories_stacked is not None
    ):
        if _is_causal and hasattr(self.global_integrator, "forward_causal"):
            # 因果模式：每个 segment 得到独立的 G_i
            higher_global_per_group = self.global_integrator.forward_causal(
                local_memories_stacked
            )  # [bsz, num_groups, global_slots, hidden_size]
            num_global_slots = higher_global_per_group.shape[2]

            if _causal_mode in ("causal_shift", "causal_shift_g"):
                # segment_i 使用 G_{i-1}，segment_0 用零
                zeros_g = torch.zeros(
                    bsz, 1, num_global_slots, hidden_size,
                    device=higher_global_per_group.device,
                    dtype=higher_global_per_group.dtype,
                )
                higher_global_per_group = torch.cat(
                    [zeros_g, higher_global_per_group[:, :-1, :, :]], dim=1
                )

            higher_global = None  # 使用 per-group 模式
        else:
            # 原始非因果模式：所有 segment 共享同一个 G
            higher_global = self.global_integrator(local_memories_stacked)
            num_global_slots = higher_global.shape[1]
            higher_global_per_group = None
    else:
        higher_global = None
        higher_global_per_group = None
        num_global_slots = 0

    # causal_shift_g / causal_gi_gonly: 不拼接 L，仅用 G
    if _causal_mode in ("causal_shift_g", "causal_gi_gonly"):
        local_memories_stacked = None
        num_local_slots = 0

    # causal_shift 模式下同时 shift L_i → segment_i 用 L_{i-1}
    if (
        _causal_mode == "causal_shift"
        and use_local_memory
        and local_memories_stacked is not None
    ):
        zeros_l = torch.zeros(
            bsz, 1, num_local_slots, hidden_size,
            device=local_memories_stacked.device,
            dtype=local_memories_stacked.dtype,
        )
        local_memories_stacked = torch.cat(
            [zeros_l, local_memories_stacked[:, :-1, :, :]], dim=1
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
    # Higher-level global memory
    higher_global_k = higher_global_v = None
    higher_global_k_per_group = higher_global_v_per_group = None

    if use_higher_global and higher_global is not None:
        # 原始非因果模式：一个 G 共享给所有 chunks
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
    elif use_higher_global and higher_global_per_group is not None:
        # 因果模式：每个 chunk 有独立的 G_i
        # higher_global_per_group: [bsz, num_groups, global_slots, hidden_size]
        hg_flat = higher_global_per_group.view(
            bsz * num_groups, num_global_slots, hidden_size
        )
        hg_k_flat = (
            self.k_proj(hg_flat)
            .view(bsz * num_groups, num_global_slots, self.num_key_value_heads, self.head_dim)
            .transpose(1, 2)
        )
        hg_v_flat = (
            self.v_proj(hg_flat)
            .view(bsz * num_groups, num_global_slots, self.num_key_value_heads, self.head_dim)
            .transpose(1, 2)
        )
        hg_k_flat = repeat_kv(hg_k_flat, self.num_key_value_groups)
        hg_v_flat = repeat_kv(hg_v_flat, self.num_key_value_groups)
        # [bsz*num_groups, nh, global_slots, hd] -> [bsz, num_groups, nh, global_slots, hd]
        higher_global_k_per_group = hg_k_flat.view(
            bsz, num_groups, self.num_heads, num_global_slots, self.head_dim
        )
        higher_global_v_per_group = hg_v_flat.view(
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
    # 优化：只计算 K/V 投影，Q 不需要（Q 只包含 chunk tokens）
    if use_local_memory and local_memories_stacked is not None:
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
    memory_len = 0
    if use_higher_global and hasattr(self, "global_integrator"):
        memory_len += num_global_slots
    if use_local_memory and hasattr(self, "local_constructor"):
        memory_len += num_local_slots
    kv_len_per_chunk = memory_len + group_size

    if memory_len > 0:
        # ========== 🔥 方案4: 使用 torch.cat 替代预分配（优化显存和速度）==========
        # 原始代码：预分配 + offset filling
        # all_k = torch.empty(bsz, self.num_heads, num_groups, kv_len_per_chunk, self.head_dim, ...)
        # all_v = torch.empty(...)
        # offset = 0
        # all_k[:, :, :, offset:offset+num_global_slots, :] = higher_global_k.unsqueeze(2)
        # ...

        # 优化：使用 cat 操作，内存分配更高效
        kv_components_k = []
        kv_components_v = []

        # 填充 higher_global
        if use_higher_global and higher_global_k is not None:
            # 非因果模式：所有 chunks 共享同一个 G
            # higher_global_k: [bsz, nh, num_global_slots, hd]
            higher_global_k_exp = higher_global_k.unsqueeze(2).expand(
                -1, -1, num_groups, -1, -1
            )
            higher_global_v_exp = higher_global_v.unsqueeze(2).expand(
                -1, -1, num_groups, -1, -1
            )
            kv_components_k.append(higher_global_k_exp)
            kv_components_v.append(higher_global_v_exp)
        elif use_higher_global and higher_global_k_per_group is not None:
            # 因果模式：每个 chunk 有独立的 G_i
            # higher_global_k_per_group: [bsz, num_groups, nh, global_slots, hd]
            # 转换为: [bsz, nh, num_groups, global_slots, hd]
            kv_components_k.append(higher_global_k_per_group.permute(0, 2, 1, 3, 4))
            kv_components_v.append(higher_global_v_per_group.permute(0, 2, 1, 3, 4))

        # 填充 local memories（每个 chunk 不同）
        if use_local_memory and local_k_all is not None:
            # local_k_all: [bsz, num_groups, nh, num_local_slots, hd]
            # 需要转换为: [bsz, nh, num_groups, num_local_slots, hd]
            local_k_exp = local_k_all.permute(0, 2, 1, 3, 4)
            local_v_exp = local_v_all.permute(0, 2, 1, 3, 4)
            kv_components_k.append(local_k_exp)
            kv_components_v.append(local_v_exp)

        # 填充 chunk tokens
        kv_components_k.append(key_chunks)
        kv_components_v.append(value_chunks)

        # 一次性 concat（内存分配更高效，dim=3 是 seq_len 维度）
        all_k = torch.cat(
            kv_components_k, dim=3
        )  # [bsz, nh, num_groups, kv_len_per_chunk, hd]
        all_v = torch.cat(kv_components_v, dim=3)
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
    if use_local_memory:
        all_masks_kv_stacked[:, :, offset : offset + num_local_slots] = 1
        offset += num_local_slots
    all_masks_kv_stacked[:, :, offset : offset + group_size] = chunk_masks_reshaped

    # causal_shift/causal_shift_g: segment_0 的 G（和 L）是零填充，mask 掉
    if _causal_mode in ("causal_shift", "causal_shift_g"):
        mem_offset = 0
        if use_higher_global:
            all_masks_kv_stacked[:, 0, mem_offset : mem_offset + num_global_slots] = 0
            mem_offset += num_global_slots
        if use_local_memory:
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

    # ========== 🎨 可视化: 收集attention统计 (仅推理时) ==========
    if COLLECT_ATTENTION_FOR_VIZ and not self.training and memory_len > 0:
        with torch.no_grad():
            # 手动计算attention weights用于可视化
            # all_chunks_q_flat: [bsz * num_groups, group_size, nh, hd]
            # all_chunks_q_flat: [bsz * num_groups, group_size, nh, hd]
            # all_k_flat: [bsz * num_groups, kv_len, nh, hd]
            # 对所有segments取平均，而不是只取第一个
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

            # K结构: [higher_global, local, chunk]
            # 对所有segments、所有heads、所有query positions取平均
            offset = 0
            attn_to_global = 0.0
            attn_to_local = 0.0

            if use_higher_global and num_global_slots > 0:
                attn_to_global = (
                    attn_probs[:, :, :, offset : offset + num_global_slots]
                    .mean()
                    .item()
                )
                offset += num_global_slots
            if use_local_memory and num_local_slots > 0:
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

            # 保存第一个样本的所有层 (只保存一个样本避免文件过大)
            layer_idx = getattr(self, "layer_idx", 0)
            saved_count = len(attention_visualizer["segment_attention_maps"])
            if saved_count < 32:  # 只保存第一个样本的32层
                attn_map = attn_probs[0, 0, :, :].cpu().numpy().tolist()
                attention_visualizer["segment_attention_maps"].append(
                    {"layer": layer_idx, "attention_map": attn_map}
                )

    # Output projection
    attn_output = self.o_proj(rearrange(output, "b s h d -> b s (h d)"))

    return attn_output, None, past_key_value


# 推理生成时候的函数 NEW: Hierarchical Memory without Cache
def forward_flashattn_hierarchical_inference(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    padding_mask: Optional[torch.LongTensor] = None,
    # 直接在这个函数中控制的参数
    use_higher_global: bool = True,  # 是否使用高层全局记忆
    use_local_memory: bool = True,  # 是否使用局部记忆（默认 False 避免冗余）
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """
    Hierarchical memory for INFERENCE (支持任意长度输入的 padding).

    与训练版本的关键区别：
    1. 支持任意长度输入（通过 padding 到 segment_size 的倍数）
    2. 处理完成后截断回原始长度
    3. 无训练相关的警告

    整合了以下功能：
    1. Hierarchical memory (LocalConstructor + HierarchicalMemoryAggregator)
    2. Ablation modes (use_higher_global, use_local_memory)

    优化：Q 只包含 chunk tokens，K/V 包含 [memories, chunk]
    - 节省计算：memories 不参与 Q 计算
    - chunk tokens 可以 attend 到 memories（通过 K/V）
    - 输出直接就是 chunk tokens，无需额外提取

    拼接顺序：
    - Q:   [chunk]
    - K/V: [higher_global?, local?, chunk]

    三种 Ablation 模式：
    - Mode 1 (推荐): use_higher_global=True, use_local_memory=False
      Q: [chunk], K/V: [higher_global, chunk]

    - Mode 2: use_higher_global=False, use_local_memory=True
      Q: [chunk], K/V: [local_i, chunk]

    - Mode 3: use_higher_global=True, use_local_memory=True
      Q: [chunk], K/V: [higher_global, local_i, chunk]

    Args:
        use_higher_global: 是否使用高层全局记忆（聚合所有 local memories）
        use_local_memory: 是否使用局部记忆（每个 chunk 的压缩表示）
    """
    if output_attentions:
        warnings.warn(
            "Output attentions is not supported for patched `LlamaAttention`, returning `None` instead."
        )

    bsz, q_len, hidden_size = hidden_states.size()

    # ========================================================================
    # 🔥 Decode 模式检测：当 q_len 很小且有 past_key_value 时，使用 Full Attention
    #
    # 设计理念（和 LongLoRA 一致）：
    # - HiCI 的价值在于 训练/Prefill 时让模型学会利用全局信息
    # - Decode 时只有 1 个 token，分组没意义，使用 Full Attention
    # - KV Cache 存储原始 K/V，不含 Global Memory（G 是动态计算的）
    #
    # 参考：llama_attn_replace_ori.py 的 forward_flashattn_inference
    # ========================================================================
    if q_len <= 32 and past_key_value is not None:
        # Decode 模式：使用 Full Attention（和 LongLoRA 推理函数一致）
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

        # 拼接 past KV cache
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
    # 🔥 Prefill 模式：使用 HiCI 分组 + Memory
    # ========================================================================

    # 保存原始长度，用于后续截断
    original_q_len = q_len

    # 打印配置（只在第一次调用时打印，且只在 rank 0 和 layer 0 打印）
    if not hasattr(self, "_hierarchical_inference_printed"):
        rank = dist.get_rank() if dist.is_initialized() else 0
        layer_idx = getattr(self, "layer_idx", 0)

        if rank == 0 and layer_idx == 0:
            print("\n" + "=" * 80)
            print("Hierarchical Memory INFERENCE (with padding support)")
            print("=" * 80)
            print(f"  use_higher_global : {use_higher_global}")
            print(f"  use_local_memory  : {use_local_memory}")

            if use_higher_global and not use_local_memory:
                print("  Mode 1: Q=[chunk], K/V=[higher_global, chunk]")
            elif not use_higher_global and use_local_memory:
                print("  Mode 2: Q=[chunk], K/V=[local_i, chunk]")
            elif use_higher_global and use_local_memory:
                print("  Mode 3: Q=[chunk], K/V=[higher_global, local_i, chunk]")
            else:
                print("  Baseline: Q=K/V=[chunk]")

            print("=" * 80 + "\n", flush=True)

        self._hierarchical_inference_printed = True

    # ========== Step 1: 分 chunk (推理模式简化) ==========
    layer_idx = getattr(self, "layer_idx", 0)

    # ========================================================================
    # 🔥 Full Attention + Memory 模式：不分组，但仍然使用 memory
    #
    # 当 USE_FULL_ATTN_WITH_MEMORY = True 时：
    # - 整个输入作为一个 chunk（num_groups = 1）
    # - 对整个输入提取 local memory -> 聚合成 global memory
    # - 所有 tokens attend 到 [global_memory, all_tokens]
    # - 效果应该和原始 LLaMA 类似，只是多了 memory context
    # ========================================================================
    if USE_FULL_ATTN_WITH_MEMORY:
        # Full Attention + Memory 模式：整个输入作为一个 chunk
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
                    "🔥 Full Attention + Memory 模式 (USE_FULL_ATTN_WITH_MEMORY=True)"
                )
                print("=" * 80)
                print(f"  输入长度: {q_len}")
                print(f"  不分组，整个输入作为一个 chunk")
                print(
                    f"  仍然使用 memory: use_higher_global={use_higher_global}, use_local_memory={use_local_memory}"
                )
                print(f"  Q: [all_tokens], K/V: [global_memory, all_tokens]")
                print("=" * 80 + "\n", flush=True)
                forward_flashattn_hierarchical_inference._full_attn_mem_printed = True
    else:
        # 原始分组模式
        # 🔥 推理模式：始终使用固定的 segment_size
        group_size = (
            FIXED_SEGMENT_SIZE
            if USE_FIXED_SEGMENT_SIZE
            else int(q_len * group_size_ratio)
        )

        # 处理 q_len 太短的情况
        if q_len < group_size:
            group_size = q_len

        # 确保 group_size 至少为 1
        group_size = max(1, group_size)

        # 🔥 推理模式核心：处理不能整除的情况 - 使用 padding
        padding_needed = 0
        if q_len % group_size > 0:
            # 计算需要 pad 的长度
            padded_q_len = ((q_len + group_size - 1) // group_size) * group_size
            padding_needed = padded_q_len - q_len

            # Pad hidden_states: [bsz, q_len, hidden_size] -> [bsz, padded_q_len, hidden_size]
            hidden_states = torch.nn.functional.pad(
                hidden_states, (0, 0, 0, padding_needed), mode="constant", value=0
            )

            # Pad attention_mask: [bsz, q_len] -> [bsz, padded_q_len]
            # 注意：padding 位置的 mask 应该是 0（被 mask 掉）
            if attention_mask is not None:
                attention_mask = torch.nn.functional.pad(
                    attention_mask, (0, padding_needed), mode="constant", value=0
                )

            # 🔥 Pad position_ids: [bsz, q_len] -> [bsz, padded_q_len]
            # 延续原来的位置编码（3404, 3405, 3406, ...）
            if position_ids is not None:
                # 创建新的 position_ids，延续原来的位置
                last_pos = position_ids[:, -1:] + 1  # 最后一个位置 + 1
                padding_positions = last_pos + torch.arange(
                    padding_needed, device=position_ids.device, dtype=position_ids.dtype
                ).unsqueeze(
                    0
                )  # [1, padding_needed] -> broadcast to [bsz, padding_needed]
                position_ids = torch.cat([position_ids, padding_positions], dim=1)

            # 更新 q_len 为 padded 后的长度
            q_len = padded_q_len

        # 使用类变量确保全局只打印一次
        num_groups = q_len // group_size

        # ========================================================================
        # 🔥 单 Group 处理：当 num_groups == 1 且不是 Full Attention + Memory 模式时
        #
        # 注意：只有在原始分组模式下才禁用 memory
        # 如果想要测试 "不分组但使用 memory"，请设置 USE_FULL_ATTN_WITH_MEMORY = True
        # ========================================================================
        if num_groups == 1:
            # 单 group 时禁用 memory（原始行为）
            # 如果想要单 group 也使用 memory，请设置 USE_FULL_ATTN_WITH_MEMORY = True
            use_higher_global = False
            use_local_memory = False

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
                        f"[HiCI Prefill] ⚠️ Single group detected, memory disabled. "
                        f"Set USE_FULL_ATTN_WITH_MEMORY=True to enable memory with single group."
                    )
                forward_flashattn_hierarchical_inference._prefill_printed = True

    # Reshape into chunks: [bsz, num_groups, group_size, hidden_size]
    chunks = hidden_states.view(bsz, num_groups, group_size, hidden_size)

    # attention_mask: [bsz, q_len] -> chunk_masks_reshaped: [bsz, num_groups, group_size]
    chunk_masks_reshaped = attention_mask.view(bsz, num_groups, group_size)

    # ========== Step 2: 提取局部记忆（对每个 chunk 提取压缩表示）==========
    # 使用 LocalConstructorFlash 对每个 chunk 单独提取局部全局表示
    if (use_higher_global or use_local_memory) and hasattr(self, "local_constructor"):
        # 批处理所有 chunks（并行）
        all_chunks = chunks.view(bsz * num_groups, group_size, hidden_size)

        # 🔥 方案3: 确保使用 bfloat16 进行 memory extraction（节省显存）
        # 原始代码：直接使用 all_chunks，可能是 float32
        # if all_chunks.dtype == torch.float32:
        #     all_chunks = all_chunks.to(torch.bfloat16)
        # 优化：强制转换为 bfloat16（如果使用 mixed precision training）
        original_dtype = all_chunks.dtype
        if all_chunks.dtype == torch.float32:
            all_chunks = all_chunks.to(torch.bfloat16)

        # [bsz, num_groups, group_size] -> [bsz * num_groups, group_size]
        attention_mask_chunks = chunk_masks_reshaped.view(bsz * num_groups, group_size)
        all_local_mems = self.local_constructor(
            all_chunks, attention_mask_chunks
        )  # [bsz * num_groups, num_slots, hidden_size]

        # 🔥 如果原始是 float32，转换回来保持一致性
        if original_dtype == torch.float32:
            all_local_mems = all_local_mems.to(torch.float32)

        # Reshape back: [bsz, num_groups, num_slots, hidden_size]
        num_local_slots = all_local_mems.shape[1]
        local_memories_stacked = all_local_mems.view(
            bsz, num_groups, num_local_slots, hidden_size
        )
    else:
        num_local_slots = 0
        local_memories_stacked = None

    # ========== Step 3: 聚合到高层全局记忆（可选）==========
    # 根据 CAUSAL_MEMORY_MODE 选择因果模式
    _causal_mode = CAUSAL_MEMORY_MODE  # "none", "causal_gi", "causal_shift", "causal_shift_g"
    _is_causal = _causal_mode in ("causal_gi", "causal_shift", "causal_shift_g", "causal_gi_gonly")

    if (
        use_higher_global
        and hasattr(self, "global_integrator")
        and local_memories_stacked is not None
    ):
        if _is_causal and hasattr(self.global_integrator, "forward_causal"):
            # 因果模式：每个 segment 得到独立的 G_i
            higher_global_per_group = self.global_integrator.forward_causal(
                local_memories_stacked
            )  # [bsz, num_groups, global_slots, hidden_size]
            num_global_slots = higher_global_per_group.shape[2]

            if _causal_mode in ("causal_shift", "causal_shift_g"):
                # segment_i 使用 G_{i-1}，segment_0 用零
                zeros_g = torch.zeros(
                    bsz, 1, num_global_slots, hidden_size,
                    device=higher_global_per_group.device,
                    dtype=higher_global_per_group.dtype,
                )
                higher_global_per_group = torch.cat(
                    [zeros_g, higher_global_per_group[:, :-1, :, :]], dim=1
                )

            higher_global = None  # 使用 per-group 模式
        else:
            # 原始非因果模式：所有 segment 共享同一个 G
            higher_global = self.global_integrator(local_memories_stacked)
            num_global_slots = higher_global.shape[1]
            higher_global_per_group = None
    else:
        higher_global = None
        higher_global_per_group = None
        num_global_slots = 0

    # causal_shift_g / causal_gi_gonly: 不拼接 L，仅用 G
    if _causal_mode in ("causal_shift_g", "causal_gi_gonly"):
        local_memories_stacked = None
        num_local_slots = 0

    # causal_shift 模式下同时 shift L_i → segment_i 用 L_{i-1}
    if (
        _causal_mode == "causal_shift"
        and use_local_memory
        and local_memories_stacked is not None
    ):
        zeros_l = torch.zeros(
            bsz, 1, num_local_slots, hidden_size,
            device=local_memories_stacked.device,
            dtype=local_memories_stacked.dtype,
        )
        local_memories_stacked = torch.cat(
            [zeros_l, local_memories_stacked[:, :-1, :, :]], dim=1
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
    # Higher-level global memory
    higher_global_k = higher_global_v = None
    higher_global_k_per_group = higher_global_v_per_group = None

    if use_higher_global and higher_global is not None:
        # 原始非因果模式：一个 G 共享给所有 chunks
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
    elif use_higher_global and higher_global_per_group is not None:
        # 因果模式：每个 chunk 有独立的 G_i
        # higher_global_per_group: [bsz, num_groups, global_slots, hidden_size]
        hg_flat = higher_global_per_group.view(
            bsz * num_groups, num_global_slots, hidden_size
        )
        hg_k_flat = (
            self.k_proj(hg_flat)
            .view(bsz * num_groups, num_global_slots, self.num_key_value_heads, self.head_dim)
            .transpose(1, 2)
        )
        hg_v_flat = (
            self.v_proj(hg_flat)
            .view(bsz * num_groups, num_global_slots, self.num_key_value_heads, self.head_dim)
            .transpose(1, 2)
        )
        hg_k_flat = repeat_kv(hg_k_flat, self.num_key_value_groups)
        hg_v_flat = repeat_kv(hg_v_flat, self.num_key_value_groups)
        # [bsz*num_groups, nh, global_slots, hd] -> [bsz, num_groups, nh, global_slots, hd]
        higher_global_k_per_group = hg_k_flat.view(
            bsz, num_groups, self.num_heads, num_global_slots, self.head_dim
        )
        higher_global_v_per_group = hg_v_flat.view(
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
    # 优化：只计算 K/V 投影，Q 不需要（Q 只包含 chunk tokens）
    if use_local_memory and local_memories_stacked is not None:
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
    memory_len = 0
    if use_higher_global and hasattr(self, "global_integrator"):
        memory_len += num_global_slots
    if use_local_memory and hasattr(self, "local_constructor"):
        memory_len += num_local_slots
    kv_len_per_chunk = memory_len + group_size

    if memory_len > 0:
        # ========== 🔥 方案4: 使用 torch.cat 替代预分配（优化显存和速度）==========
        # 原始代码：预分配 + offset filling
        # all_k = torch.empty(bsz, self.num_heads, num_groups, kv_len_per_chunk, self.head_dim, ...)
        # all_v = torch.empty(...)
        # offset = 0
        # all_k[:, :, :, offset:offset+num_global_slots, :] = higher_global_k.unsqueeze(2)
        # ...

        # 优化：使用 cat 操作，内存分配更高效
        kv_components_k = []
        kv_components_v = []

        # 填充 higher_global
        if use_higher_global and higher_global_k is not None:
            # 非因果模式：所有 chunks 共享同一个 G
            # higher_global_k: [bsz, nh, num_global_slots, hd]
            higher_global_k_exp = higher_global_k.unsqueeze(2).expand(
                -1, -1, num_groups, -1, -1
            )
            higher_global_v_exp = higher_global_v.unsqueeze(2).expand(
                -1, -1, num_groups, -1, -1
            )
            kv_components_k.append(higher_global_k_exp)
            kv_components_v.append(higher_global_v_exp)
        elif use_higher_global and higher_global_k_per_group is not None:
            # 因果模式：每个 chunk 有独立的 G_i
            # higher_global_k_per_group: [bsz, num_groups, nh, global_slots, hd]
            # 转换为: [bsz, nh, num_groups, global_slots, hd]
            kv_components_k.append(higher_global_k_per_group.permute(0, 2, 1, 3, 4))
            kv_components_v.append(higher_global_v_per_group.permute(0, 2, 1, 3, 4))

        # 填充 local memories（每个 chunk 不同）
        if use_local_memory and local_k_all is not None:
            # local_k_all: [bsz, num_groups, nh, num_local_slots, hd]
            # 需要转换为: [bsz, nh, num_groups, num_local_slots, hd]
            local_k_exp = local_k_all.permute(0, 2, 1, 3, 4)
            local_v_exp = local_v_all.permute(0, 2, 1, 3, 4)
            kv_components_k.append(local_k_exp)
            kv_components_v.append(local_v_exp)

        # 填充 chunk tokens
        kv_components_k.append(key_chunks)
        kv_components_v.append(value_chunks)

        # 一次性 concat（内存分配更高效，dim=3 是 seq_len 维度）
        all_k = torch.cat(
            kv_components_k, dim=3
        )  # [bsz, nh, num_groups, kv_len_per_chunk, hd]
        all_v = torch.cat(kv_components_v, dim=3)
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
    if use_local_memory:
        all_masks_kv_stacked[:, :, offset : offset + num_local_slots] = 1
        offset += num_local_slots
    all_masks_kv_stacked[:, :, offset : offset + group_size] = chunk_masks_reshaped

    # causal_shift/causal_shift_g: segment_0 的 G（和 L）是零填充，mask 掉
    if _causal_mode in ("causal_shift", "causal_shift_g"):
        mem_offset = 0
        if use_higher_global:
            all_masks_kv_stacked[:, 0, mem_offset : mem_offset + num_global_slots] = 0
            mem_offset += num_global_slots
        if use_local_memory:
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

    # 🔥 推理模式：截断回原始长度（移除 padding）
    if original_q_len < q_len:
        output = output[:, :original_q_len, :, :]

        # 🔥 关键修复：也需要截断 past_key_value，否则 decode 时 position_ids 会错乱！
        # Bug: prefill 返回 padded 长度的 KV cache (4096)，但 output 已截断 (3404)
        # 导致 decode 时新 token 的 position_id (3404) 与 KV cache 长度 (4096) 不匹配
        # RoPE 位置编码错乱，生成结果变成重复词或乱码
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
    use_higher_global: bool = False,  # 是否使用高层全局记忆
    use_local_memory: bool = False,  # 是否使用局部记忆
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """
    Hierarchical memory without Flash Attention (用于消融实验).

    与 forward_flashattn_hierarchical 相同的逻辑，但使用标准 attention 计算：
    1. Hierarchical memory extraction (LocalConstructor + HierarchicalMemoryAggregator)
    2. Standard Q/K/V projections
    3. Manual attention computation (matmul + softmax) instead of Flash Attention

    用途：比较显存占用和训练时间（论文消融实验）

    优化：Q 只包含 chunk tokens，K/V 包含 [memories, chunk]

    拼接顺序：
    - Q:   [chunk]
    - K/V: [higher_global?, local?, chunk]

    Args:
        use_higher_global: 是否使用高层全局记忆（聚合所有 local memories）
        use_local_memory: 是否使用局部记忆（每个 chunk 的压缩表示）
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

    # 打印配置（只在第一次调用时打印）
    if not hasattr(self, "_hierarchical_noflash_printed"):
        rank = dist.get_rank() if dist.is_initialized() else 0
        layer_idx = getattr(self, "layer_idx", 0)

        if rank == 0 and layer_idx == 0:
            print("\n" + "=" * 80)
            print("Hierarchical Memory (No Flash Attention - for ablation)")
            print("=" * 80)
            print(f"  use_higher_global : {use_higher_global}")
            print(f"  use_local_memory  : {use_local_memory}")

            if use_higher_global and not use_local_memory:
                print("  Mode 1: Q=[chunk], K/V=[higher_global, chunk]")
            elif not use_higher_global and use_local_memory:
                print("  Mode 2: Q=[chunk], K/V=[local_i, chunk]")
            elif use_higher_global and use_local_memory:
                print("  Mode 3: Q=[chunk], K/V=[higher_global, local_i, chunk]")
            else:
                print("  Baseline: Q=K/V=[chunk]")

            print("=" * 80 + "\n", flush=True)

        self._hierarchical_noflash_printed = True

    # ========== Step 1: 分 chunk ==========
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

    # ========== 重要：attention_mask 格式处理 ==========
    # forward_noflashattn 不替换 _prepare_decoder_attention_mask
    # 所以接收 HuggingFace 默认的 4D causal mask: [bsz, 1, q_len, q_len]
    #
    # 但 LocalConstructorFlash 需要 2D padding mask: [bsz, seq_len]
    # 我们从 4D mask 的对角线提取 padding mask（0=padding, 非负无穷=valid）
    if attention_mask is not None and attention_mask.dim() == 4:
        # attention_mask: [bsz, 1, q_len, q_len]
        # mask[b, 0, i, j]: query i 对 key j 的可见性
        # - 如果 j 是 padding：整列 [:, j] 都是 -inf（没有任何 query 能看到它）
        # - 如果 j 是 valid：至少有一个 i >= j 使得 mask[i, j] = 0
        #
        # 正确提取方法：检查每一列是否有非 -inf 的值
        # dim=-2 是 query 维度，对每一列（key 维度）取最大值
        padding_mask_2d = (
            attention_mask[:, 0, :, :].max(dim=-2)[0] > -1e4
        ).long()  # [bsz, q_len], 1=valid, 0=padding
    else:
        # 如果没有 mask 或已经是 2D，假设所有 tokens 都是 valid
        padding_mask_2d = torch.ones(
            bsz, q_len, dtype=torch.long, device=hidden_states.device
        )

    # Reshape padding mask for chunks: [bsz, q_len] -> [bsz, num_groups, group_size]
    chunk_masks_reshaped = padding_mask_2d.view(bsz, num_groups, group_size)

    # ========== Step 2: 提取局部记忆 ==========
    if (use_higher_global or use_local_memory) and hasattr(self, "local_constructor"):
        all_chunks = chunks.view(bsz * num_groups, group_size, hidden_size)
        attention_mask_chunks = chunk_masks_reshaped.view(bsz * num_groups, group_size)
        all_local_mems = self.local_constructor(all_chunks, attention_mask_chunks)

        num_local_slots = all_local_mems.shape[1]
        local_memories_stacked = all_local_mems.view(
            bsz, num_groups, num_local_slots, hidden_size
        )
    else:
        num_local_slots = 0
        local_memories_stacked = None

    # ========== Step 3: 聚合到高层全局记忆 ==========
    _causal_mode = CAUSAL_MEMORY_MODE
    _is_causal = _causal_mode in ("causal_gi", "causal_shift", "causal_shift_g", "causal_gi_gonly")

    if (
        use_higher_global
        and hasattr(self, "global_integrator")
        and local_memories_stacked is not None
    ):
        if _is_causal and hasattr(self.global_integrator, "forward_causal"):
            higher_global_per_group = self.global_integrator.forward_causal(
                local_memories_stacked
            )
            num_global_slots = higher_global_per_group.shape[2]

            if _causal_mode in ("causal_shift", "causal_shift_g"):
                zeros_g = torch.zeros(
                    bsz, 1, num_global_slots, hidden_size,
                    device=higher_global_per_group.device,
                    dtype=higher_global_per_group.dtype,
                )
                higher_global_per_group = torch.cat(
                    [zeros_g, higher_global_per_group[:, :-1, :, :]], dim=1
                )

            higher_global = None
        else:
            higher_global = self.global_integrator(local_memories_stacked)
            num_global_slots = higher_global.shape[1]
            higher_global_per_group = None
    else:
        higher_global = None
        higher_global_per_group = None
        num_global_slots = 0

    # causal_shift_g / causal_gi_gonly: 不拼接 L，仅用 G
    if _causal_mode in ("causal_shift_g", "causal_gi_gonly"):
        local_memories_stacked = None
        num_local_slots = 0

    if (
        _causal_mode == "causal_shift"
        and use_local_memory
        and local_memories_stacked is not None
    ):
        zeros_l = torch.zeros(
            bsz, 1, num_local_slots, hidden_size,
            device=local_memories_stacked.device,
            dtype=local_memories_stacked.dtype,
        )
        local_memories_stacked = torch.cat(
            [zeros_l, local_memories_stacked[:, :-1, :, :]], dim=1
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
    higher_global_k = higher_global_v = None
    higher_global_k_per_group = higher_global_v_per_group = None

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
        higher_global_k = repeat_kv(higher_global_k, self.num_key_value_groups)
        higher_global_v = repeat_kv(higher_global_v, self.num_key_value_groups)
    elif use_higher_global and higher_global_per_group is not None:
        hg_flat = higher_global_per_group.view(
            bsz * num_groups, num_global_slots, hidden_size
        )
        hg_k_flat = (
            self.k_proj(hg_flat)
            .view(bsz * num_groups, num_global_slots, self.num_key_value_heads, self.head_dim)
            .transpose(1, 2)
        )
        hg_v_flat = (
            self.v_proj(hg_flat)
            .view(bsz * num_groups, num_global_slots, self.num_key_value_heads, self.head_dim)
            .transpose(1, 2)
        )
        hg_k_flat = repeat_kv(hg_k_flat, self.num_key_value_groups)
        hg_v_flat = repeat_kv(hg_v_flat, self.num_key_value_groups)
        higher_global_k_per_group = hg_k_flat.view(
            bsz, num_groups, self.num_heads, num_global_slots, self.head_dim
        )
        higher_global_v_per_group = hg_v_flat.view(
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
    if use_local_memory and local_memories_stacked is not None:
        local_mems_flat = local_memories_stacked.view(
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

    # ========== Step 7: 构建所有 chunks 的 K/V（批处理，和 Flash Attention 版本一致）==========
    # 计算 K/V 总长度
    memory_len = 0
    if use_higher_global and hasattr(self, "global_integrator"):
        memory_len += num_global_slots
    if use_local_memory and hasattr(self, "local_constructor"):
        memory_len += num_local_slots
    kv_len_per_chunk = memory_len + group_size

    if memory_len > 0:
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
        # 填充 higher_global
        if use_higher_global and higher_global_k is not None:
            # 非因果模式：所有 chunks 共享
            all_k[:, :, :, offset : offset + num_global_slots, :] = (
                higher_global_k.unsqueeze(2)
            )
            all_v[:, :, :, offset : offset + num_global_slots, :] = (
                higher_global_v.unsqueeze(2)
            )
            offset += num_global_slots
        elif use_higher_global and higher_global_k_per_group is not None:
            # 因果模式：每个 chunk 不同的 G_i
            # higher_global_k_per_group: [bsz, num_groups, nh, global_slots, hd]
            # -> [bsz, nh, num_groups, global_slots, hd]
            all_k[:, :, :, offset : offset + num_global_slots, :] = (
                higher_global_k_per_group.permute(0, 2, 1, 3, 4)
            )
            all_v[:, :, :, offset : offset + num_global_slots, :] = (
                higher_global_v_per_group.permute(0, 2, 1, 3, 4)
            )
            offset += num_global_slots

        # 填充 local memories（每个 chunk 不同）
        if use_local_memory and local_k_all is not None:
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

    # ========== Step 8: Reshape 成批处理格式（参考 LongLoRA forward_noflashattn）==========
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

    # ========== Step 9: Manual attention computation（参考 LongLoRA）==========
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

    # ========== Step 10: 处理 attention_mask（无循环批处理版本）==========
    # attention_mask 是 4D [bsz, 1, q_len, q_len]（Hugging Face 原始的 causal mask）
    # 需要为每个 group 提取 diagonal blocks: [bsz * num_groups, 1, group_size, kv_len_per_chunk]

    # Step 1: 提取 diagonal blocks（chunk tokens 的 causal mask）
    # Reshape: [bsz, 1, q_len, q_len] -> [bsz, 1, num_groups, group_size, num_groups, group_size]
    mask_6d = attention_mask.view(
        bsz, 1, num_groups, group_size, num_groups, group_size
    )

    # 提取对角线：取 mask_6d[:, :, i, :, i, :] for all i
    # torch.diagonal(input, dim1, dim2) 提取 dim1 和 dim2 的对角线
    # 结果: [bsz, 1, group_size, group_size, num_groups]
    diagonal_blocks = torch.diagonal(mask_6d, dim1=2, dim2=4)

    # Permute to [num_groups, bsz, 1, group_size, group_size]
    diagonal_blocks = diagonal_blocks.permute(4, 0, 1, 2, 3)

    # Reshape to [bsz * num_groups, 1, group_size, group_size]
    chunk_masks = diagonal_blocks.reshape(bsz * num_groups, 1, group_size, group_size)

    if memory_len > 0:
        # Step 2: 为 memory tokens 创建 mask（对所有 tokens 可见）
        # [bsz, 1, q_len, memory_len]
        memory_mask_cols = torch.zeros(
            bsz,
            1,
            q_len,
            memory_len,
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )

        # Reshape Q dimension to group: [bsz, 1, num_groups, group_size, memory_len]
        memory_mask_grouped = memory_mask_cols.view(
            bsz, 1, num_groups, group_size, memory_len
        )

        # Reshape to [bsz * num_groups, 1, group_size, memory_len]
        memory_mask_flat = memory_mask_grouped.permute(0, 2, 1, 3, 4).reshape(
            bsz * num_groups, 1, group_size, memory_len
        )

        # causal_shift/causal_shift_g: segment_0 的 G（和 L）是零填充，mask 掉
        if _causal_mode in ("causal_shift", "causal_shift_g"):
            # memory_mask_flat: [bsz*num_groups, 1, group_size, memory_len]
            # 排列: [batch0_group0, batch0_group1, ..., batch1_group0, ...]
            # segment_0 = 每个 batch 的第一个 group，stride = num_groups
            seg0_indices = torch.arange(0, bsz * num_groups, num_groups,
                                        device=memory_mask_flat.device)
            memory_mask_flat[seg0_indices, :, :, :] = torch.finfo(memory_mask_flat.dtype).min

        # Step 3: 拼接 [memories, chunk]
        # [bsz * num_groups, 1, group_size, memory_len + group_size]
        attention_mask_expanded = torch.cat([memory_mask_flat, chunk_masks], dim=3)
    else:
        # 没有 memory，直接使用 chunk masks
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

    # ========== Step 12: Softmax and compute output（参考 LongLoRA）==========
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

    # ========== Step 13: Output projection（参考 LongLoRA）==========
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
# 原本的longlora的noflashattn函数
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
            - None: Chunked attention with HiCI memory (same as training)
            - "full": Full attention without memory
        use_hierarchical_forward: Use forward_flashattn_hierarchical (local + global memory)
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
                # 评估方式1: Full Attention without memory
                transformers.models.llama.modeling_llama.LlamaAttention.forward = (
                    forward_flashattn_full
                )
                if rank == 0:
                    print(f"   调用函数: forward_flashattn_full  原本的评估方式")
            else:
                # 训练/评估方式2: Chunked attention with memory
                if use_hierarchical_forward:
                    # 🔥 根据是否固定 segment_size 选择训练或推理版本
                    if USE_FIXED_SEGMENT_SIZE:
                        # 推理模式：使用支持 padding 的推理版本
                        transformers.models.llama.modeling_llama.LlamaAttention.forward = forward_flashattn_hierarchical_inference
                        if rank == 0:
                            print(
                                f"   🎯 Using forward_flashattn_hierarchical_inference (with padding, segment_size={FIXED_SEGMENT_SIZE})"
                            )
                    else:
                        # 训练模式：使用原始版本
                        transformers.models.llama.modeling_llama.LlamaAttention.forward = forward_flashattn_hierarchical
                        if rank == 0:
                            print(
                                "   🧪 Using forward_flashattn_hierarchical (training mode)"
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
    num_chunks: Optional[int] = 4,  # 新增：指定 chunk 数量，None 则根据序列长度动态计算
    num_heads=32,
    use_bottleneck=True,
    bottleneck_dim=4096,  # zxy
    use_local_constructor=True,
    use_global_integrator=True,
    use_local_constructor_flash: Optional[bool] = False,  # 是否使用 LocalConstructorFlash
    use_llama_init=False,  # 新增：方案C - 从 LLaMA 预训练权重初始化 Q/K/V 投影
    use_shared_compressor=True,  # 🆕 是否使用共享压缩层优化版（节省71%参数） 在此处修改
    compress_dim=512,  # 压缩层的中间维度 在此处修改  13b 640dim 10头
    shared_compress_dim=128,  # 🆕 共享压缩层的中间维度 在此处修改 7b128 13b 160
    ds_config_path=None,  # 🆕 DeepSpeed 配置路径，用于 ZeRO-3 参数分片
):
    """
    Register LocalConstructor (and optionally HierarchicalMemoryAggregator) to each LlamaAttention layer.

    This MUST be called after model loading and before optimizer initialization!

    Args:
        model: LlamaForCausalLM or PeftModelForCausalLM
        num_local_slots: Number of Local Representation Slots (for LocalConstructor, default: 16)
        global_slots: Number of higher-level global slots (for HierarchicalMemoryAggregator, default: 16)
        num_heads: Number of attention heads (default: 32)
        bottleneck_dim: Bottleneck dimension for efficiency (default: 2048)
        use_global_integrator: If True, also register HierarchicalMemoryAggregator (default: False)
        use_llama_init: If True, initialize Q/K/V projections from LLaMA pretrained weights (方案C)
            - 理论依据：让 Memory 的 Q/K/V 投影与 LLaMA 在同一个预训练语义空间
            - 初始时 Q×K^T 计算有意义，收敛更快
            - 投影仍然是独立参数，可以继续微调
        use_shared_compressor: 🆕 If True, use GlobalIntegratorShared (saves 71% params)
            - 原版 GlobalIntegrator: 13.7M/layer
            - 优化版 GlobalIntegratorShared: 4.0M/layer
            - 关键优化：5个统计量共享同一个4096→128的压缩层
            - 理论依据：参数共享 + 更强的归纳偏置（类似CNN的weight sharing）
        shared_compress_dim: 共享压缩层的中间维度（仅当 use_shared_compressor=True 时生效，默认: 128)

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
            config_str.append("Local Memory")
        if use_global_integrator:
            config_str.append("Hierarchical Aggregator")

        if config_str:
            print(f"🔧 Registering: {' + '.join(config_str)}")
        else:
            print("⚠️ No memory modules enabled!")

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
    if rank == 0:
        print(f"   Model dtype: {model_dtype}")

    # ✅ 获取预训练嵌入权重用于初始化记忆模块（方案 2）
    embed_weight = llama_model.embed_tokens.weight.data  # [vocab_size, hidden_size]

    # 🆕 自动检测 ZeRO-3：从环境变量或已传入的配置路径获取
    use_zero3_init = False
    zero3_init_context = None
    detected_ds_config = ds_config_path

    # 如果没有显式传入，尝试从环境变量获取
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
                        f"   🔧 Using deepspeed.zero.Init() for Memory module sharding"
                    )
        except Exception as e:
            if rank == 0:
                print(f"   ⚠️ Failed to load DeepSpeed config: {e}")

    # Register modules to each attention layer
    # 如果是 ZeRO-3，使用 context manager 确保参数被正确分片
    if use_zero3_init:
        import deepspeed

        # 🔧 修复：从原始配置读取，但替换 "auto" 占位符为实际值
        # 保留原始配置中的 ZeRO-3 参数（如 stage3_param_persistence_threshold 等）
        import copy

        zero3_config = copy.deepcopy(ds_config)  # 深拷贝，避免修改原始配置

        # 替换 batch size 相关的 "auto" 值
        # DeepSpeed 要求: train_batch_size = micro_batch * grad_acc * world_size
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

        # Module 1: LocalConstructor (always register)
        # ✅ 使用预训练嵌入初始化（方案 2）
        # ✅ 使用 Flash Attention 实现高效的 cross-attention（支持 100k+ 序列）
        if use_local_constructor:
            if use_local_constructor_flash:
                # 原版：有独立的 Q/K/V 投影
                # 配合 forward_flashattn_optimized 使用
                attn.local_constructor = LocalConstructorFlash(
                    hidden_size=hidden_size,
                    num_local_slots=num_local_slots,
                    num_heads=num_heads,
                    init_from_embeddings=embed_weight,
                    init_from_llama_attn=attn if use_llama_init else None,
                    use_bottleneck=use_bottleneck,
                    bottleneck_dim=bottleneck_dim,
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
                    compress_dim=bottleneck_dim,  # 最终压缩维度 或者bottleneck_dim
                    shared_compress_dim=shared_compress_dim,  # 共享压缩层的中间维度（默认128）
                    num_heads=num_heads,
                    # num_heads=8,  # 固定为8头，减少参数
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
                    compress_dim=compress_dim,  # ← 统计量压缩维度  此处控制
                    num_heads=num_heads,
                    init_from_embeddings=embed_weight,
                    use_high_norm_init=True,  # ← 使用高范数初始化提升稳定性 此处控制
                ).to(model_dtype)  # ✅ 转换为模型的 dtype

    # 🆕 关闭 ZeRO-3 Init context manager
    if use_zero3_init and zero3_init_context is not None:
        zero3_init_context.__exit__(None, None, None)
        if rank == 0:
            print(f"   ✅ ZeRO-3 Memory module sharding complete")

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
        print("✅ Memory Module Registration Complete")
        print("=" * 80)

        # 模型总参数
        print(f"Model: {total_params:,} params ({total_params / 1e9:.2f}B)")
        print(f"Layers: {len(llama_model.layers)}")

        # 注册的模块和参数统计
        if use_local_constructor and use_global_integrator:
            total_memory_params = local_constructor_params + aggregator_params
            print(f"\nRegistered Modules:")
            print(f"  ✓ Local Memory ({local_constructor_params:,} params)")
            print(f"  ✓ Hierarchical Aggregator ({aggregator_params:,} params)")
            print(
                f"\nTotal Memory Params: {total_memory_params:,} ({total_memory_params / total_params * 100:.2f}%)"
            )

        elif use_local_constructor and not use_global_integrator:
            print(f"\nRegistered Modules:")
            print(f"  ✓ Local Memory ({local_constructor_params:,} params)")
            print(
                f"\nTotal Memory Params: {local_constructor_params:,} ({local_constructor_params / total_params * 100:.2f}%)"
            )

        elif not use_local_constructor and use_global_integrator:
            print(f"\n⚠️ Warning: Hierarchical registered without Local Memory!")
            print(f"  ✓ Hierarchical Aggregator ({aggregator_params:,} params)")
            print(
                f"\nTotal Memory Params: {aggregator_params:,} ({aggregator_params / total_params * 100:.2f}%)"
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
