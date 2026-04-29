#!/bin/bash
# source ~/venv/zxy/bin/activate
# bash train_fine_tune_memory_inject_cache_sft.sh 2>&1 | tee Train_out_sft/Llama-2-7b-chat-hf-sft-1.19.txt

pkill -9 -f "fine-tune_memory_inject_cache_sft.py"
fuser -k 38493/tcp 2>/dev/null || echo "✅ Port 38493 not in use"
sleep 2

# 基础配置
MODEL_PATH="./models/Llama-2-7b-chat-hf"
RESUME_CHECKPOINT="./checkpoints/Llama-2-7b-16k-FTM-NEW-8-bothhigher_multi_clip_2e_clean_share/checkpoint-1000"
OUTPUT_DIR="./checkpoints/Llama-2-7b-chat-hf-sft"
MAX_LENGTH=16384  # SFT通常使用 8192 或 16384 不需要 32768
DATA_PATH="./data/sft/LongAlpaca-12k.json"

# 训练超参数
nproc_per_node=8
WARMUP_STEPS=20
NUM_EPOCHS=20 
MAX_STEPS=3500  # -1 表示根据 epochs 自动计算；也可设置固定值如 1000
low_rank_training=True  # 是否使用低秩训练 LongLoRA

# Memory 模块配置
use_local_summary=True  # 是否使用本地摘要记忆机制
use_hierarchical_memory=True  # 是否使用高层压缩HierarchicalMemory
num_chunks=4  # chunk数量（仅在使用高层记忆时有效）
Local_MEMORY_SLOTS=8  # local memory slots
global_slots=4  # global memory slots
num_heads=8  # number of attention heads
use_bottleneck=True  # whether to use bottleneck in hierarchical memory aggregator
bottleneck_dim=512  # bottleneck dimension
recurrence_size=128  # 缓存大小

# Memory 学习率和梯度裁剪
global_memory_lr=2e-4
memory_grad_clip=0.3

# 可训练参数
# TRAINABLE_PARAMS="embed,norm"
TRAINABLE_PARAMS="embed,norm,global_memory,hierarchical_aggregator"

# Memory 模块类型选择
use_llama_init=False  # qkv的参数是否从llama初始化
memory_use_flash=False
use_hierarchical_forward=True  # 是否使用综合函数 局部+全局
use_flash_plus_norope=False
use_flash_plus=False  # 是否使用 GlobalMemoryModule_Flash_plus 复用kv投影
forward_flashattn_optimized=True  # 使用 forward_flashattn_hybrid (原forward_flashattn_optimized)

deepspeed_config="ds_configs/stage2.json"  # Stage 2: 24GB VRAM; Stage 3: 16GB VRAM

echo "========================================================================"
echo "🔥 Supervised Fine-Tuning (SFT) for Memory-Augmented LongLoRA"
echo "========================================================================"
echo ""
echo "📦 基础配置:"
echo "  - 基础模型: $MODEL_PATH"
echo "  - 恢复检查点: $RESUME_CHECKPOINT"
echo "  - 输出目录: $OUTPUT_DIR"
echo "  - 📊 数据集: $DATA_PATH"
echo "  - 🤖 GPU数目: $nproc_per_node"
echo "  - 📏 最大长度: $MAX_LENGTH"
echo "  - 🔄 训练轮数: $NUM_EPOCHS"
echo "  - 📈 最大步数: $MAX_STEPS (如为-1则根据epochs自动计算)"
echo "  - 🔥 预热步数: $WARMUP_STEPS"
echo "  - ⚙️ 可训练参数: $TRAINABLE_PARAMS"
echo "  - 💾 DeepSpeed配置: $deepspeed_config"
echo "  - 🎯 使用低秩训练 LongLoRA: $low_rank_training"
echo ""
echo "🧠 Memory 模块配置:"
echo "  - 📝 使用局部摘要记忆: $use_local_summary"
echo "  - 🔁 使用高层全局记忆: $use_hierarchical_memory"
echo "  - 🌐 Global Memory Slots: $global_slots"
echo "  - 🧠 Local Memory Slots: $Local_MEMORY_SLOTS"
echo "  - 🔢 Chunk分组数: $num_chunks"
echo "  - 💡 记忆学习率: $global_memory_lr"
echo "  - 🤖 记忆梯度剪裁: $memory_grad_clip"
echo "  - 🧩 Recurrence Cache大小: $recurrence_size"
echo "  - 🎯 使用 Bottleneck: $use_bottleneck"
echo "  - 📊 Bottleneck Dimension: $bottleneck_dim"
echo "  - 🧠 记忆kv初始化: llama_init=$use_llama_init"
echo ""
echo "⚙️ 前馈函数配置:"
echo "  - 🧠 调用记忆类: GlobalMemoryModule_Flash_plus kv复用: $use_flash_plus"
echo "  - 🧠 调用记忆类: GlobalMemoryModule_Flash kv独立: $memory_use_flash"
echo "  - 📝 调用函数: forward_flashattn_hierarchical 局部+全局: $use_hierarchical_forward"
echo "  - 📝 调用函数: forward_flashattn_optimized_plus_norope 无ROPE: $use_flash_plus_norope"
echo "  - 📝 调用函数: forward_flashattn_optimized_plus 含ROPE: $use_flash_plus"
echo "  - 📝 调用函数: forward_flashattn_hybrid (原optimized): $forward_flashattn_optimized"
echo ""
echo "========================================================================"

# --resume_from_checkpoint $RESUME_CHECKPOINT \
torchrun --nproc_per_node $nproc_per_node \
      --master_port=38493 \
      fine-tune_memory_inject_cache_sft.py \
      --model_name_or_path $MODEL_PATH \
      --data_path $DATA_PATH \
      --bf16 True \
      --output_dir $OUTPUT_DIR \
      --cache_dir ./cache \
      --model_max_length $MAX_LENGTH \
      --use_flash_attn True \
      --low_rank_training $low_rank_training \
      --num_train_epochs $NUM_EPOCHS \
      --per_device_train_batch_size 1 \
      --per_device_eval_batch_size 2 \
      --gradient_accumulation_steps 8 \
      --evaluation_strategy "no" \
      --save_strategy "steps" \
      --save_steps 500 \
      --save_total_limit 4 \
      --learning_rate 2e-5 \
      --weight_decay 0.0 \
      --warmup_steps $WARMUP_STEPS \
      --lr_scheduler_type "constant_with_warmup" \
      --logging_steps 1 \
      --deepspeed $deepspeed_config \
      --tf32 True \
      --max_steps $MAX_STEPS \
      --num_memory_slots $Local_MEMORY_SLOTS \
      --global_slots $global_slots \
      --num_chunks $num_chunks \
      --use_local_summary $use_local_summary \
      --use_hierarchical_memory $use_hierarchical_memory \
      --num_heads $num_heads \
      --use_bottleneck $use_bottleneck \
      --bottleneck_dim $bottleneck_dim \
      --recurrence_size $recurrence_size \
      --trainable_params $TRAINABLE_PARAMS \
      --global_memory_lr $global_memory_lr \
      --memory_grad_clip $memory_grad_clip \
      --memory_use_flash $memory_use_flash \
      --use_flash_plus $use_flash_plus \
      --use_flash_plus_norope $use_flash_plus_norope \
      --use_llama_init $use_llama_init \
      --use_hierarchical_forward $use_hierarchical_forward \
      --forward_flashattn_optimized $forward_flashattn_optimized

echo ""
echo "========================================================================"
echo "✅ SFT Training Completed!"
echo "========================================================================"
echo "📁 Checkpoints saved to: $OUTPUT_DIR"