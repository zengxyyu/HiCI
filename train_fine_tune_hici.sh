source ~/venv/env/bin/activate

MODEL_PATH="/path/to/Llama-2-7b-hf"
OUTPUT_DIR="./checkpoints/Llama-2-7b-8k-HiCI"
MAX_LENGTH=8192
WARMUP_STEPS=20
NUM_LOCAL_SLOTS=32
TRAINABLE_PARAMS="embed,norm,local_constructor,global_integrator"

echo "================================"
echo "Base model: $MODEL_PATH"
echo "Output dir: $OUTPUT_DIR"
echo "Max length: $MAX_LENGTH"
echo "Local slots: $NUM_LOCAL_SLOTS"
echo "Trainable params: $TRAINABLE_PARAMS"
echo "================================"

fuser -k 38493/tcp 2>/dev/null || echo "Port 38493 not in use"
sleep 2

torchrun --nproc_per_node=8 \
      --master_port=38493 \
      fine-tune_hici.py \
      --model_name_or_path $MODEL_PATH \
      --bf16 True \
      --output_dir $OUTPUT_DIR \
      --cache_dir ./cache \
      --model_max_length $MAX_LENGTH \
      --use_flash_attn True \
      --low_rank_training True \
      --num_train_epochs 1  \
      --per_device_train_batch_size 1 \
      --per_device_eval_batch_size 2 \
      --gradient_accumulation_steps 8 \
      --evaluation_strategy "no" \
      --save_strategy "steps" \
      --save_steps 1000 \
      --save_total_limit 2 \
      --learning_rate 2e-5 \
      --weight_decay 0.0 \
      --warmup_steps $WARMUP_STEPS \
      --lr_scheduler_type "constant_with_warmup" \
      --logging_steps 1 \
      --deepspeed "ds_configs/stage2.json" \
      --tf32 True \
      --max_steps 1000 \
      --num_local_slots $NUM_LOCAL_SLOTS \
      --trainable_params $TRAINABLE_PARAMS \
      --hici_lr 2e-4

echo ""
echo "================================"
echo "Training completed!"
echo "Checkpoints saved to: $OUTPUT_DIR"
echo ""
