pkill -9 -f "fine-tune_hici.py"
# Train_out_13b_baseline
# bash train_fine_tune_hici.sh 2>&1 | tee Train_out_baseline/Llama-2-7b-16k-FTM-NEW-84-bothhigher_multi_clip_2e_clean_share_G16.txt
# bash train_fine_tune_hici.sh 2>&1 | tee Training_out_fuxian/Llama-2-13b-hici-16k-none.txt
# New_Training_out Train_out_baseline
# 8192 16384 32768 65536  100000 131072  262144 
MODEL_PATH="./models/Llama-2-13b-hf"
# MODEL_PATH="/scratch/sh89/xz2053/projects/llm-memory/models/Meta-Llama-3-8B"
OUTPUT_DIR="./checkpoints/Llama-2-13b-hici-16k-none"
MAX_LENGTH=16384  # 8192 32768 16384 65536 100000 131072 262144
WARMUP_STEPS=20
hici_lr=2e-4
nproc_per_node=4
hici_grad_clip=0.3
low_rank_training=True  # 是否使用低秩训练 LongLoRA
# HiCI configuration ================
use_local_constructor=True
use_global_integrator=True
NUM_LOCAL_SLOTS=8
global_slots=4  # Global Representation Slots
num_heads=8  # number of attention heads
use_bottleneck=True
bottleneck_dim=512
TRAINABLE_PARAMS="embed,norm,local_constructor,global_integrator"
# TRAINABLE_PARAMS="embed,norm,local_constructor"
# TRAINABLE_PARAMS="embed,norm"
use_llama_init=False  #qkv的参数是否从llama初始化
shared_compress_dim=128  # 共享压缩层维度（7B用128, 13B用160）

use_local_constructor_flash=False

# 控制前馈函数调用的地方
#forward_flashattn_hierarchical_with_cache 或者 forward_flashattn_hierarchical
use_hierarchical_forward=True  # 是否使用综合函数 局部+全局
# ================
deepspeed_config="ds_configs/stage2.json"

echo "================================"
echo "📦 基础模型: $MODEL_PATH"
echo "📁 模型输出目录: $OUTPUT_DIR"
echo "🤖 GPU数目: $nproc_per_node"
echo "📊 最大长度: $MAX_LENGTH"
echo "⚙️ 可训练参数: $TRAINABLE_PARAMS"
echo "🔥 预热步数: $WARMUP_STEPS"
echo "💡 HiCI learning rate: $hici_lr"
echo "💡 deepspeed配置: $deepspeed_config"
echo "🤖 HiCI gradient clip: $hici_grad_clip"
echo "🎯 使用低秩训练 LongLoRA: $low_rank_training"

echo "--------------HiCI configuration-----------------"
echo "📝 LocalConstructor: $use_local_constructor"
echo "🔁 GlobalIntegrator: $use_global_integrator"
echo "🌐 Global Representation Slots: $global_slots"
echo "🧠 Local Representation Slots: $NUM_LOCAL_SLOTS"
echo "🧠 Init from LLaMA weights: $use_llama_init"
echo "🎯 使用 Bottleneck: $use_bottleneck"
echo "🔢 HiCI Attention Heads: $num_heads"
echo "🧩 Bottleneck Dimension: $bottleneck_dim"
echo "🧩 Shared Compress Dim: $shared_compress_dim"
echo "--------------Module and forward function config-----------------"
echo "🧠 LocalConstructorFlash: $use_local_constructor_flash"
echo "📝 调用函数 forward_flashattn_hierarchical 局部+全局: $use_hierarchical_forward"
echo "================================"
echo ""

# 清理端口
fuser -k 38493/tcp 2>/dev/null || echo "Port 38493 not in use"
sleep 2
pkill -9 -f "fine-tune_hici.py"
# --recurrence_size 256 \
# --use_yarn_rope True \
torchrun --nproc_per_node $nproc_per_node \
      --master_port=38493 \
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
      --save_steps 250 \
      --save_total_limit 4 \
      --learning_rate 2e-5 \
      --weight_decay 0.0 \
      --warmup_steps $WARMUP_STEPS \
      --lr_scheduler_type "constant_with_warmup" \
      --logging_steps 1 \
      --deepspeed $deepspeed_config \
      --tf32 True \
      --max_steps 1000\
      --num_local_slots $NUM_LOCAL_SLOTS \
      --global_slots $global_slots \
      --use_local_constructor $use_local_constructor \
      --use_global_integrator $use_global_integrator \
      --trainable_params $TRAINABLE_PARAMS \
      --hici_lr $hici_lr \
      --num_heads $num_heads \
      --use_bottleneck $use_bottleneck \
      --bottleneck_dim $bottleneck_dim \
      --shared_compress_dim $shared_compress_dim \
      --use_llama_init $use_llama_init \
      --use_hierarchical_forward $use_hierarchical_forward \
      --hici_grad_clip $hici_grad_clip \
      --use_local_constructor_flash $use_local_constructor_flash \

echo ""
echo "================================"
echo "✅ Training completed!"
echo "📁 Checkpoints saved to: $OUTPUT_DIR"
echo ""