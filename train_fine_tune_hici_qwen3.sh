# bash train_fine_tune_hici_qwen3.sh 2>&1 | tee Train_out_qwen3/Qwen3-84-bothhigher_multi_clip_2e_clean_share.txt
# bash train_fine_tune_hici_qwen3.sh 2>&1 | tee Train_out_qwen3/Qwen3-8b-hici-64k-G8-stage3_causal_gi.txt

pkill -9 -f "fine-tune_hici_qwen3.py"

# Qwen3-8B HiCI Training on Gadi (4x H200)
MODEL_PATH="./models/Qwen3-8B"
OUTPUT_DIR="./checkpoints/Qwen3-8b-hici-64k-G8-stage3"
MAX_LENGTH=49152  # 8192 32768 49152 51200 65536 131072
WARMUP_STEPS=20
hici_lr=2e-4
nproc_per_node=2
hici_grad_clip=0.3
low_rank_training=True

# HiCI configuration
use_local_constructor=True
use_global_integrator=True
NUM_LOCAL_SLOTS=8
global_slots=4        # HiCI global context slots
num_heads=8            # HiCI HiCI module attention heads
use_bottleneck=True
bottleneck_dim=512
TRAINABLE_PARAMS="embed,norm,local_constructor,global_integrator"

# HiCI module and forward function

use_hierarchical_forward=True
use_local_constructor_flash=False
use_attn_init=False    # Whether to init HiCI Q/K/V from pretrained weights

deepspeed_config="ds_configs/stage2.json"

echo "================================"
echo "Qwen3-8B HiCI Training"
echo "================================"
echo "Model: $MODEL_PATH"
echo "Output: $OUTPUT_DIR"
echo "GPUs: $nproc_per_node"
echo "Max length: $MAX_LENGTH"
echo "Trainable params: $TRAINABLE_PARAMS"
echo "HiCI LR: $hici_lr"
echo "DeepSpeed: $deepspeed_config"
echo "HiCI grad clip: $hici_grad_clip"
echo "LoRA: $low_rank_training"
echo ""
echo "--- HiCI Settings ---"
echo "use local constructor: $use_local_constructor"
echo "use global integrator: $use_global_integrator"
echo "Global Representation Slots: $global_slots"
echo "Local Representation Slots: $NUM_LOCAL_SLOTS"
echo "Num heads: $num_heads"
echo "Bottleneck: $use_bottleneck (dim=$bottleneck_dim)"
echo "Flash plus: $use_local_constructor_flash"
echo "Attn init: $use_attn_init"
echo "================================"
echo ""

# Clean port
fuser -k 38493/tcp 2>/dev/null || echo "Port 38493 not in use"
sleep 2
pkill -9 -f "fine-tune_hici_qwen3.py"

torchrun --nproc_per_node $nproc_per_node \
      --master_port=38493 \
      fine-tune_hici_qwen3.py \
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
      --gradient_accumulation_steps 32 \
      --eval_strategy "no" \
      --save_strategy "steps" \
      --save_steps 100 \
      --save_total_limit 4 \
      --learning_rate 2e-5 \
      --weight_decay 0.0 \
      --warmup_steps $WARMUP_STEPS \
      --lr_scheduler_type "constant_with_warmup" \
      --logging_steps 1 \
      --deepspeed $deepspeed_config \
      --tf32 True \
      --max_steps 1000 \
      --num_local_slots $NUM_LOCAL_SLOTS \
      --global_slots $global_slots \
      --use_local_constructor $use_local_constructor \
      --use_global_integrator $use_global_integrator \
      --trainable_params $TRAINABLE_PARAMS \
      --hici_lr $hici_lr \
      --num_heads $num_heads \
      --use_bottleneck $use_bottleneck \
      --bottleneck_dim $bottleneck_dim \
      --use_local_constructor_flash $use_local_constructor_flash \
      --use_attn_init $use_attn_init \
      --use_hierarchical_forward $use_hierarchical_forward \
      --hici_grad_clip $hici_grad_clip

echo ""
echo "================================"
echo "Training completed!"
echo "Checkpoints saved to: $OUTPUT_DIR"
echo ""