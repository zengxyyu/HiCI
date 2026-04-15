#!/bin/bash
# Qwen3-8B HiCI SFT Training (UltraChat 200k)
#
# Stage 2: Supervised Fine-Tuning on instruction-following data
# After HiCI pre-training (stage 1), this teaches the model to follow instructions.
#
# Prerequisites:
#   pip install trl>=0.12.0
#   # Pre-download UltraChat on login node (compute nodes may not have internet):
#   python -c "from datasets import load_dataset; load_dataset('HuggingFaceH4/ultrachat_200k', split='train_sft')"
#
# Usage:
#   Single node:  bash train_fine_tune_hici_qwen3_sft.sh
#   Multi-node:   bash train_fine_tune_hici_qwen3_sft.sh <node_rank>

NODE_RANK=${1:-0}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
cd "$SCRIPT_DIR"
export DS_SKIP_CUDA_CHECK=1

# Activate environment
if [ -d "${PBS_JOBFS}/hici-qwen3/bin" ]; then
    export PATH="${PBS_JOBFS}/hici-qwen3/bin:$PATH"
    export CONDA_PREFIX="${PBS_JOBFS}/hici-qwen3"
else
    source ~/venv/zxy/bin/activate 2>/dev/null || conda activate hici-qwen3 2>/dev/null
fi

# ============================================================
# Configuration
# ============================================================

# Model: base Qwen3-8B or merged HiCI model from stage 1
MODEL_PATH="./models/Qwen3-8B"

# Output
OUTPUT_DIR="./checkpoints/Qwen3-8b-hici-sft-ultrachat"

# Dataset (local Arrow format, pre-downloaded)
DATASET_NAME="HuggingFaceH4/ultrachat_200k"
DATASET_SPLIT="train_sft"
DATA_PATH="./data/ultrachat_200k_train_sft"

# Sequence length for SFT (shorter than pre-training context)
MAX_SEQ_LENGTH=8192

# RoPE: set if model was pre-trained for longer context
# E.g., 49152 if HiCI pre-training extended to 48K
# Leave empty/0 for native 32K context
MODEL_MAX_LENGTH=0

# Optional: load HiCI weights from stage 1 pre-training
# PRETRAINED_HICI_PATH="./checkpoints/Qwen3-8b-hici-48k/checkpoint-2000/trainable_params.bin"
PRETRAINED_HICI_PATH=""

# Training
LEARNING_RATE=2e-5
MAX_STEPS=1000
WARMUP_STEPS=50
BATCH_SIZE=1
GRAD_ACCUM=8

# LoRA
LOW_RANK_TRAINING=True
LORA_R=8
LORA_ALPHA=16
TRAINABLE_PARAMS="embed,norm,local_constructor,global_integrator"

# HiCI module config (match stage 1 config!)
NUM_LOCAL_SLOTS=8
GLOBAL_SLOTS=4
NUM_HEADS=8
BOTTLENECK_DIM=512
HICI_LR=2e-4
HICI_GRAD_CLIP=0.3
RECURRENCE_SIZE=128

# Attention
USE_FLASH_ATTN=True
USE_HIERARCHICAL_FORWARD=True
USE_LOCAL_CONSTRUCTOR_FLASH=False

# DeepSpeed
DEEPSPEED_CONFIG="ds_configs/stage2.json"

# Multi-node (adjust for your cluster)
NNODES=1
NPROC_PER_NODE=4
MASTER_ADDR="localhost"
MASTER_PORT=29500

# ============================================================
# Logging
# ============================================================

echo "============================================"
echo "Qwen3-8B HiCI SFT Training"
echo "============================================"
echo "Model:         $MODEL_PATH"
echo "Output:        $OUTPUT_DIR"
echo "Dataset:       $DATASET_NAME ($DATASET_SPLIT)"
echo "Max seq len:   $MAX_SEQ_LENGTH"
echo "Steps:         $MAX_STEPS"
echo "Batch size:    $BATCH_SIZE x $GRAD_ACCUM (effective: $((BATCH_SIZE * GRAD_ACCUM * NPROC_PER_NODE * NNODES)))"
echo "HiCI config: slots=$NUM_LOCAL_SLOTS, global=$GLOBAL_SLOTS, heads=$NUM_HEADS"
echo "DeepSpeed:     $DEEPSPEED_CONFIG"
if [ -n "$PRETRAINED_HICI_PATH" ]; then
    echo "HiCI weights: $PRETRAINED_HICI_PATH"
fi
echo "============================================"
echo ""

# Clean up stale processes
pkill -9 -f "fine-tune_hici_qwen3_sft.py" 2>/dev/null
fuser -k $MASTER_PORT/tcp 2>/dev/null
sleep 2

# ============================================================
# Build command
# ============================================================

CMD="torchrun \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --nproc_per_node=$NPROC_PER_NODE \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    fine-tune_hici_qwen3_sft.py \
    --model_name_or_path $MODEL_PATH \
    --dataset_name $DATASET_NAME \
    --dataset_split $DATASET_SPLIT \
    --output_dir $OUTPUT_DIR \
    --max_seq_length $MAX_SEQ_LENGTH \
    --bf16 True \
    --per_device_train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --learning_rate $LEARNING_RATE \
    --max_steps $MAX_STEPS \
    --warmup_steps $WARMUP_STEPS \
    --lr_scheduler_type constant_with_warmup \
    --weight_decay 0.0 \
    --logging_steps 1 \
    --save_strategy steps \
    --save_steps 500 \
    --save_total_limit 3 \
    --gradient_checkpointing True \
    --deepspeed $DEEPSPEED_CONFIG \
    --use_flash_attn $USE_FLASH_ATTN \
    --use_hierarchical_forward $USE_HIERARCHICAL_FORWARD \
    --use_local_constructor_flash $USE_LOCAL_CONSTRUCTOR_FLASH \
    --low_rank_training $LOW_RANK_TRAINING \
    --lora_r $LORA_R \
    --lora_alpha $LORA_ALPHA \
    --trainable_params $TRAINABLE_PARAMS \
    --num_local_slots $NUM_LOCAL_SLOTS \
    --global_slots $GLOBAL_SLOTS \
    --num_heads $NUM_HEADS \
    --use_bottleneck True \
    --bottleneck_dim $BOTTLENECK_DIM \
    --recurrence_size $RECURRENCE_SIZE \
    --hici_lr $HICI_LR \
    --hici_grad_clip $HICI_GRAD_CLIP"

# Optional: RoPE extension
if [ "$MODEL_MAX_LENGTH" -gt 0 ] 2>/dev/null; then
    CMD="$CMD --model_max_length $MODEL_MAX_LENGTH"
fi

# Optional: pre-trained HiCI weights
if [ -n "$PRETRAINED_HICI_PATH" ] && [ -f "$PRETRAINED_HICI_PATH" ]; then
    CMD="$CMD --pretrained_hici_path $PRETRAINED_HICI_PATH"
fi

# Optional: local data path instead of HuggingFace Hub
if [ -n "$DATA_PATH" ] && [ -f "$DATA_PATH" ]; then
    CMD="$CMD --data_path $DATA_PATH"
fi

echo "Running: $CMD"
echo ""

eval $CMD

echo ""
echo "============================================"
echo "SFT Training completed!"
echo "Checkpoints: $OUTPUT_DIR"
echo ""
echo "Next steps:"
echo "  1. Extract weights:  python get_trainable_weights.py --checkpoint_path $OUTPUT_DIR/checkpoint-$MAX_STEPS --trainable_params $TRAINABLE_PARAMS"
echo "  2. Merge model:      python merge_lora_weights_hici.py --base_model $MODEL_PATH --peft_model $OUTPUT_DIR/checkpoint-$MAX_STEPS --save_path ./models/merged_models/Qwen3-8b-HiCI-SFT"
echo "  3. Evaluate:         python LongBench/pred.py --model Qwen3-8b-HiCI-SFT"
echo "============================================"
