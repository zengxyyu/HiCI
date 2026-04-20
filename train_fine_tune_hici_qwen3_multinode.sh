#!/bin/bash
# Qwen3-8B HiCI Multi-Node Training (2 nodes x 4 H200 = 8 GPUs)
#
# Usage:
#   Node 0 (gadi-gpu-h200-0009): bash train_fine_tune_hici_qwen3_multinode.sh 0 2>&1 | tee Train_out_qwen3/qwen3-8b-hici-48k-causal_gi_G4.txt
#   Node 1 (gadi-gpu-h200-0016): bash train_fine_tune_hici_qwen3_multinode.sh 1
#
# Both must be started within NCCL timeout (default 120s) of each other.

NODE_RANK=${1:?'Usage: bash train_fine_tune_hici_qwen3_multinode.sh <node_rank> (0=master, 1=worker)'}

# Environment - auto-detect account
# SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
# cd "$SCRIPT_DIR"
export DS_SKIP_CUDA_CHECK=1
# export CUDA_HOME=/apps/cuda/12.5.1


# Multi-node config
MASTER_ADDR="10.6.106.17"  # gadi-gpu-h200-0017
MASTER_PORT=29500
NNODES=2
NPROC_PER_NODE=4

# NCCL
export NCCL_TIMEOUT=7200
export NCCL_BLOCKING_WAIT=1

# Training config
MODEL_PATH="./models/Qwen3-8B"
OUTPUT_DIR="./checkpoints/Qwen3-8b-hici-48k"
MAX_LENGTH=49152  # 8192 32768 49152 65536
WARMUP_STEPS=20
hici_lr=2e-4
hici_grad_clip=0.3
gradient_accumulation_steps=8
low_rank_training=True

# HiCI configuration
use_local_constructor=True
use_global_integrator=True
NUM_LOCAL_SLOTS=8
global_slots=4
num_heads=8
use_bottleneck=True
bottleneck_dim=512
TRAINABLE_PARAMS="embed,norm,local_constructor,global_integrator"

# HiCI module and forward function
use_local_constructor_flash=False
use_hierarchical_forward=True
use_attn_init=False

deepspeed_config="ds_configs/stage2.json"

echo "========================================"
echo "🚀 Qwen3-8B HiCI Multi-Node Training"
echo "========================================"
echo "🖥️  Node:             $NODE_RANK / $((NNODES-1))"
echo "🌐 Master:           $MASTER_ADDR:$MASTER_PORT"
echo "🤖 GPUs per node:    $NPROC_PER_NODE"
echo "🤖 Total GPUs:       $((NNODES * NPROC_PER_NODE))"
echo "📈 Grad accumulation: $gradient_accumulation_steps"
echo "📦 Base model:       $MODEL_PATH"
echo "📁 Output dir:       $OUTPUT_DIR"
echo "📊 Max length:       $MAX_LENGTH"
echo "💡 DeepSpeed:        $deepspeed_config"
echo "========================================"
echo ""

# Clean up
pkill -9 -f "fine-tune_hici_qwen3.py"
fuser -k $MASTER_PORT/tcp 2>/dev/null
sleep 2

torchrun \
      --nnodes=$NNODES \
      --node_rank=$NODE_RANK \
      --nproc_per_node=$NPROC_PER_NODE \
      --master_addr=$MASTER_ADDR \
      --master_port=$MASTER_PORT \
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
      --gradient_accumulation_steps $gradient_accumulation_steps \
      --eval_strategy "no" \
      --save_strategy "steps" \
      --save_steps 250 \
      --save_total_limit 3 \
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
echo "========================================"
echo "✅ Training completed! (Node $NODE_RANK)"
echo "📁 Checkpoints saved to: $OUTPUT_DIR"
echo "========================================"
echo ""
