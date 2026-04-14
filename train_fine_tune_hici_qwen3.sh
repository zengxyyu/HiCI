# bash train_fine_tune_hici_qwen3.sh 2>&1 | tee Train_out_qwen3/Qwen3-84-bothhigher_multi_clip_2e_clean_share.txt
# bash train_fine_tune_hici_qwen3.sh 2>&1 | tee Training_out_fuxian/Qwen3-8b-hici-48k-test_none.txt
pkill -9 -f "fine-tune_hici_qwen3.py"

# Qwen3-8B HiCI Training on Gadi (4x H200)
MODEL_PATH="/scratch/sh89/xz2053/projects/llm-memory/models/Qwen3-8B"
OUTPUT_DIR="./checkpoints/Qwen3-8b-hici-48k-test"
MAX_LENGTH=49152  # 8192 32768 49152 51200 65536 131072
WARMUP_STEPS=20
hici_lr=2e-4
nproc_per_node=4
hici_grad_clip=0.3
low_rank_training=True

# Memory configuration
use_local_constructor=True
use_global_integrator=True
num_chunks=8
NUM_LOCAL_SLOTS=8
global_slots=4        # HiCI Global Representation Slots
num_heads=8            # HiCI memory module attention heads
use_bottleneck=True
bottleneck_dim=512
shared_compress_dim=128  # 共享压缩层维度（7B/8B用128, 13B用160）
TRAINABLE_PARAMS="embed,norm,local_constructor,global_integrator"

# Memory module and forward function
use_local_constructor_flash=False
use_hierarchical_forward=True
use_flash_plus=False
use_attn_init=False    # Whether to init memory Q/K/V from pretrained weights

deepspeed_config="ds_configs/stage2.json"

echo "================================"
echo "Qwen3-8B HiCI Training"
echo "================================"
echo "Model: $MODEL_PATH"
echo "Output: $OUTPUT_DIR"
echo "GPUs: $nproc_per_node"
echo "Max length: $MAX_LENGTH"
echo "Chunks: $num_chunks"
echo "Trainable params: $TRAINABLE_PARAMS"
echo "Memory LR: $hici_lr"
echo "DeepSpeed: $deepspeed_config"
echo "Memory grad clip: $hici_grad_clip"
echo "LoRA: $low_rank_training"
echo ""
echo "--- Memory Settings ---"
echo "Local summary: $use_local_constructor"
echo "Hierarchical memory: $use_global_integrator"
echo "Global slots: $global_slots"
echo "Local slots: $NUM_LOCAL_SLOTS"
echo "Num heads: $num_heads"
echo "Bottleneck: $use_bottleneck (dim=$bottleneck_dim)"
echo "Shared compress dim: $shared_compress_dim"
echo "Flash plus: $use_flash_plus"
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
      --gradient_accumulation_steps 16 \
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
      --num_chunks $num_chunks \
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
      --use_flash_plus $use_flash_plus \
      --use_attn_init $use_attn_init \
      --use_hierarchical_forward $use_hierarchical_forward \
      --hici_grad_clip $hici_grad_clip \
      --use_local_constructor_flash $use_local_constructor_flash

echo ""
echo "================================"
echo "Training completed!"
echo "Checkpoints saved to: $OUTPUT_DIR"
echo ""
