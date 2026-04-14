#!/bin/bash
# ============================================================================
# NCI H200 x4 训练脚本 — 因果消融实验
# bash train_fine_tune_hici_nci.sh 2>&1 | tee Train_out_Re/Llama-2-7b-8k-hici-causal_shift-G8.txt
# bash train_fine_tune_hici_nci.sh 2>&1 | tee Train_out_Re/Llama-2-7b-8k-hici-causal_gi-G8.txt
# bash train_fine_tune_hici_nci.sh 2>&1 | tee Train_out_Re/Llama-2-7b-8k-hici-none-G8.txt
# bash train_fine_tune_hici_nci.sh 2>&1 | tee Train_out_Re/Llama-3-8b-32k-hici-causal_g-G4.txt
# ============================================================================
eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate hici
# export DS_SKIP_CUDA_CHECK=1
# export OMP_NUM_THREADS=12   # 96核 / 4 GPU / 2 (留余量) = 12 线程/进程
# export MKL_NUM_THREADS=12
cd /scratch/sx11/sx0401/workspace/xiangyu/llm-memory

pkill -9 -f "fine-tune_hici.py" 2>/dev/null

# ============================================================================
# 基础配置
# ============================================================================
# MODEL_PATH="./models/Llama-2-7b-hf"
MODEL_PATH="./models/Meta-Llama-3-8B"
OUTPUT_DIR="./checkpoints/Llama-3-8b-32k-hici-causal_g-G4"
MAX_LENGTH=32768     # 8192 16384 32768 65536  100000 131072  262144 
WARMUP_STEPS=20
nproc_per_node=4    # NCI: 4x H200

# ============================================================================
# 记忆配置
# ============================================================================
hici_lr=2e-4
hici_grad_clip=0.3
low_rank_training=True
use_local_constructor=True
use_global_integrator=True
num_chunks=8                # 8段 x 1024 tokens
NUM_LOCAL_SLOTS=8
global_slots=4
num_heads=8
use_bottleneck=True
bottleneck_dim=512
TRAINABLE_PARAMS="embed,norm,local_constructor,global_integrator"
use_llama_init=False
recurrence_size=128
use_local_constructor_flash=False

# 前馈函数
use_hierarchical_forward=True
deepspeed_config="ds_configs/stage2.json"

# ============================================================================
echo "NCI H200 x4 Training"
echo "================================"
echo "📦 基础模型: $MODEL_PATH"
echo "📁 模型输出目录: $OUTPUT_DIR"
echo "🤖 GPU数目: $nproc_per_node"
echo "📊 最大长度: $MAX_LENGTH"
echo "🔢 Chunk分组数: $num_chunks"
echo "⚙️ 可训练参数: $TRAINABLE_PARAMS"
echo "🔥 预热步数: $WARMUP_STEPS"
echo "💡 记忆学习率: $hici_lr"
echo "💡 deepspeed配置: $deepspeed_config"
echo "🤖 记忆模块梯度剪裁: $hici_grad_clip"
echo "🎯 使用低秩训练 LongLoRA: $low_rank_training"

echo "--------------记忆属性设置-----------------"
echo "📝 使用局部摘要记忆机制: $use_local_constructor"
echo "🔁 使用高层全局记忆机制: $use_global_integrator"
echo "🌐 Global Representation Slots: $global_slots"
echo "🧠 Local Representation Slots: $NUM_LOCAL_SLOTS"
echo "🧠 记忆的kv是否从Llama参数初始化: $use_llama_init"
echo "🎯 使用 Bottleneck: $use_bottleneck"
echo "🔢 Memory Attention Heads: $num_heads"
echo "🧩 Bottleneck Dimension: $bottleneck_dim"
echo "🧩 缓存大小: $recurrence_size"
echo "--------------前馈函数设置-----------------"
echo "🧠 LocalConstructorFlash (use_local_constructor_flash): $use_local_constructor_flash"
echo "📝 forward_flashattn_hierarchical (use_hierarchical_forward): $use_hierarchical_forward"
echo "================================"
echo ""

# 清理端口
fuser -k 29500/tcp 2>/dev/null || true
sleep 1

torchrun --nproc_per_node $nproc_per_node \
      --master_port=29500 \
      fine-tune_hici.py \
      --model_name_or_path $MODEL_PATH \
      --bf16 True \
      --output_dir $OUTPUT_DIR \
      --cache_dir ./cache \
      --model_max_length $MAX_LENGTH \
      --use_flash_attn True \
      --low_rank_training $low_rank_training \
      --num_train_epochs 1  \
      --per_device_train_batch_size 1 \
      --per_device_eval_batch_size 2 \
      --gradient_accumulation_steps 16 \
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
      --max_steps 2000 \
      --num_chunks $num_chunks \
      --num_local_slots $NUM_LOCAL_SLOTS \
      --global_slots $global_slots \
      --recurrence_size $recurrence_size \
      --use_local_constructor $use_local_constructor \
      --use_global_integrator $use_global_integrator \
      --trainable_params $TRAINABLE_PARAMS \
      --hici_lr $hici_lr \
      --num_heads $num_heads \
      --use_bottleneck $use_bottleneck \
      --bottleneck_dim $bottleneck_dim \
      --use_llama_init $use_llama_init \
      --use_hierarchical_forward $use_hierarchical_forward \
      --hici_grad_clip $hici_grad_clip \
      --use_local_constructor_flash $use_local_constructor_flash

echo ""
echo "================================"
echo "Training completed!"
echo "Checkpoints: $OUTPUT_DIR"
echo "================================"
