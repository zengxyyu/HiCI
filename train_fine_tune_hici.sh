pkill -9 -f "fine-tune_hici.py"
# bash train_fine_tune_hici.sh 2>&1 | tee Train_out_baseline/Llama-2-7b-16k-FTM-NEW-84-bothhigher_multi_clip_2e_clean_share_G16.txt
# bash train_fine_tune_hici.sh 2>&1 | tee Training_out_fuxian/Llama-2-13b-hici-16k-none-4gpus.txt
# 8192 16384 32768 65536  100000 131072  262144
MODEL_PATH="./models/Llama-2-13b-hf"
# MODEL_PATH="/scratch/sh89/xz2053/projects/llm-memory/models/Meta-Llama-3-8B"
OUTPUT_DIR="./checkpoints/Llama-2-13b-hici-16k-none-sub"
MAX_LENGTH=16384  # 8192 32768 16384 65536 100000 131072 262144
WARMUP_STEPS=20
hici_lr=2e-4
nproc_per_node=4
gradient_accumulation_steps=16
hici_grad_clip=0.3
low_rank_training=True  # whether to use low-rank training (LongLoRA)

# HiCI configuration ================
use_local_constructor=True
use_global_integrator=True
NUM_LOCAL_SLOTS=8
global_slots=4  # Global Representation Slots
num_heads=8     # number of attention heads
use_bottleneck=True
bottleneck_dim=512
TRAINABLE_PARAMS="embed,norm,local_constructor,global_integrator"
# TRAINABLE_PARAMS="embed,norm,local_constructor"
# TRAINABLE_PARAMS="embed,norm"
use_llama_init=False      # whether to init Q/K/V from LLaMA weights
shared_compress_dim=128   # shared compress dim (128 for 7B, 160 for 13B)

use_local_constructor_flash=False

# controls which forward function is used
# forward_flashattn_hierarchical_with_cache or forward_flashattn_hierarchical
use_hierarchical_forward=True  # use hierarchical forward: local + global
# ================
deepspeed_config="ds_configs/stage2.json"

echo "========================================"
echo "🚀 LLaMA-2/3 HiCI Training"
echo "========================================"
echo "📦 Base model:        $MODEL_PATH"
echo "📁 Output dir:        $OUTPUT_DIR"
echo "🤖 GPUs:              $nproc_per_node"
echo "📈 Grad accumulation: $gradient_accumulation_steps"
echo "📊 Max length:        $MAX_LENGTH"
echo "⚙️  Trainable params:  $TRAINABLE_PARAMS"
echo "🔥 Warmup steps:      $WARMUP_STEPS"
echo "💡 HiCI LR:           $hici_lr"
echo "💡 DeepSpeed:         $deepspeed_config"
echo "🤖 HiCI grad clip:    $hici_grad_clip"
echo "🎯 LoRA training:     $low_rank_training"
echo ""
echo "── HiCI Configuration ───────────────────"
echo "📝 LocalConstructor:  $use_local_constructor"
echo "🔁 GlobalIntegrator:  $use_global_integrator"
echo "🌐 Global slots:      $global_slots"
echo "🧠 Local slots:       $NUM_LOCAL_SLOTS"
echo "🧠 Init from LLaMA:   $use_llama_init"
echo "🎯 Bottleneck:        $use_bottleneck"
echo "🔢 Attention heads:   $num_heads"
echo "🧩 Bottleneck dim:    $bottleneck_dim"
echo "🧩 Shared compress:   $shared_compress_dim"
echo ""
echo "── Forward Function ─────────────────────"
echo "🧠 LocalConstructorFlash: $use_local_constructor_flash"
echo "📝 Hierarchical fwd:  $use_hierarchical_forward"
echo "========================================"
echo ""

# Free port
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
      --gradient_accumulation_steps $gradient_accumulation_steps \
      --evaluation_strategy "no" \
      --save_strategy "steps" \
      --save_steps 500 \
      --save_total_limit 2 \
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
echo "========================================"
echo "✅ Training completed!"
echo "📁 Checkpoints saved to: $OUTPUT_DIR"
echo ""
