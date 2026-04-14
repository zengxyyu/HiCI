#!/bin/bash
source ~/venv/zxy/bin/activate
# bash train_fine_tune_hici_sft.sh 2>&1 | tee Train_out_sft/Llama-2-7b-16k-SFT-clean-share-hici-16.txt

# GPU抢占脚本 (训练结束后自动启动)
trap 'echo ""; echo "🎯 训练脚本退出，开始在 tmux 中抢占GPU..."; tmux new-session -d -s gpu_grab "bash /mnt/bn/strategy-mllm-train/user/xuqi/grab.sh" 2>/dev/null || tmux send-keys -t gpu_grab "bash /mnt/bn/strategy-mllm-train/user/xuqi/grab.sh" C-m; echo "✅ 抢占脚本已在 tmux session \"gpu_grab\" 中启动"; echo "💡 查看抢占状态: tmux attach -t gpu_grab"' EXIT
pkill -9 -f "fine-tune_hici_sft.py"
fuser -k 38493/tcp 2>/dev/null || echo "✅ Port 38493 not in use"
sleep 2

# 基础配置
MODEL_PATH="/mnt/bn/strategy-mllm-train/user/xuqi/repos/zxy/llm-memory/data1/pretrained-models/llama-7b-hf/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9"
# MODEL_PATH="./models/Llama-2-7b-chat-hf"
RESUME_CHECKPOINT="./checkpoints/Llama-2-7b-16k-FTM-NEW-8-bothhigher_multi_clip_2e_clean_share/checkpoint-1000"
# RESUME_CHECKPOINT="./checkpoints/Llama-2-7b-16k-FTM-NEW-75-bothhigher_multi_clip_2e_clean_share_woO/checkpoint-1000"
OUTPUT_DIR="./checkpoints/Llama-2-7b-16k-SFT-clean-share-hici-16"
MAX_LENGTH=16384  # SFT通常使用 8192 或 16384 不需要 32768
DATA_PATH="./data/sft/LongAlpaca-12k.json"
# DATA_PATH="./data/sft/LongAlpaca-16k-length/LongAlpaca-16k-length.json"

# 训练超参数
nproc_per_node=8
WARMUP_STEPS=20
NUM_EPOCHS=15 
MAX_STEPS=3000  # -1 表示根据 epochs 自动计算；也可设置固定值如 1000
low_rank_training=True  # 是否使用低秩训练 LongLoRA

# Memory 模块配置
use_local_constructor=True  # 是否使用本地摘要记忆机制
use_global_integrator=True  # 是否使用高层压缩HierarchicalMemory
num_chunks=4  # chunk数量（仅在使用高层记忆时有效）
NUM_LOCAL_SLOTS=8  # Local Representation Slots
global_slots=4  # Global Representation Slots
num_heads=8  # number of attention heads
use_bottleneck=True  # whether to use bottleneck in hierarchical memory aggregator
bottleneck_dim=512  # bottleneck dimension
shared_compress_dim=128  # 共享压缩层维度（7B用128, 13B用160）

# Memory 学习率和梯度裁剪
hici_lr=2e-4
hici_grad_clip=0.3

# 可训练参数
# TRAINABLE_PARAMS="embed,norm"
TRAINABLE_PARAMS="embed,norm,local_constructor,global_integrator"

# LocalConstructor 类型选择
use_llama_init=False  # qkv的参数是否从llama初始化
use_local_constructor_flash=False
use_hierarchical_forward=True

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
echo "  - 📝 使用局部摘要记忆: $use_local_constructor"
echo "  - 🔁 使用高层全局记忆: $use_global_integrator"
echo "  - 🌐 Global Representation Slots: $global_slots"
echo "  - 🧠 Local Representation Slots: $NUM_LOCAL_SLOTS"
echo "  - 🔢 Chunk分组数: $num_chunks"
echo "  - 💡 记忆学习率: $hici_lr"
echo "  - 🤖 记忆梯度剪裁: $hici_grad_clip"
echo "  - 🎯 使用 Bottleneck: $use_bottleneck"
echo "  - 📊 Bottleneck Dimension: $bottleneck_dim"
echo "  - 🧩 Shared Compress Dim: $shared_compress_dim"
echo "  - 🧠 记忆kv初始化: llama_init=$use_llama_init"
echo ""
echo "⚙️ 前馈函数配置:"
echo "  - 🧠 LocalConstructorFlash (use_local_constructor_flash): $use_local_constructor_flash"
echo "  - 📝 forward_flashattn_hierarchical (use_hierarchical_forward): $use_hierarchical_forward"
echo ""
echo "========================================================================"

# --resume_from_checkpoint $RESUME_CHECKPOINT \
torchrun --nproc_per_node $nproc_per_node \
      --master_port=38493 \
      fine-tune_hici_sft.py \
      --model_name_or_path $MODEL_PATH \
      --resume_from_checkpoint $RESUME_CHECKPOINT \
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
      --num_local_slots $NUM_LOCAL_SLOTS \
      --global_slots $global_slots \
      --num_chunks $num_chunks \
      --use_local_constructor $use_local_constructor \
      --use_global_integrator $use_global_integrator \
      --num_heads $num_heads \
      --use_bottleneck $use_bottleneck \
      --bottleneck_dim $bottleneck_dim \
      --shared_compress_dim $shared_compress_dim \
      --trainable_params $TRAINABLE_PARAMS \
      --hici_lr $hici_lr \
      --hici_grad_clip $hici_grad_clip \
      --use_local_constructor_flash $use_local_constructor_flash \
      --use_llama_init $use_llama_init \
      --use_hierarchical_forward $use_hierarchical_forward \

echo ""
echo "========================================================================"
echo "✅ SFT Training Completed!"
echo "========================================================================"
echo "📁 Checkpoints saved to: $OUTPUT_DIR"