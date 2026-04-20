#!/bin/bash
# HiCI Qwen3 Distributed Evaluation — Multi-Node (2 nodes x 4 H200 = 8 GPUs)
#
# Usage:
#   Node 0 (master): bash eval_distributed_hici_qwen3_multinode.sh 0 2>&1 | tee eval_qwen3_Re/PG19_TEST_EVAL_Qwen3-8b_2k.txt
#   Node 1 (worker): bash eval_distributed_hici_qwen3_multinode.sh 1
#
# Both must be started within NCCL timeout (default 120s) of each other.

NODE_RANK=${1:?'Usage: bash eval_distributed_hici_qwen3_multinode.sh <node_rank> (0=master, 1=worker)'}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
cd "$SCRIPT_DIR"
export DS_SKIP_CUDA_CHECK=1

module load gcc/12.2.0
module unload cuda
module load cuda/12.5.1

source /g/data/hn98/Yang/miniconda3/etc/profile.d/conda.sh && conda activate hici-qwen3

CONDA_ENV="/g/data/hn98/Yang/miniconda3/envs/hici-qwen3"
export LD_LIBRARY_PATH="${CONDA_ENV}/lib/python3.11/site-packages/nvidia/nvjitlink/lib:${CONDA_ENV}/lib/python3.11/site-packages/nvidia/cusparse/lib:${LD_LIBRARY_PATH}"

# ============================================================
# Multi-node config
# ============================================================
MASTER_ADDR="gadi-gpu-h200-0024.gadi.nci.org.au"
MASTER_PORT=38493
NNODES=2
NPROC_PER_NODE=2

# NCCL
export NCCL_TIMEOUT=7200
export NCCL_BLOCKING_WAIT=0
export TORCH_NCCL_BLOCKING_WAIT=0

# ============================================================
# Model & Data
# ============================================================
# BASE_MODEL="./models/Qwen3-8B"
BASE_MODEL="./models/merged/Qwen3-8b-HiCI-48k-merged"
CHECKPOINT_PATH="./checkpoints/Qwen3-8b-HiCI-48k"

DATA_PATH="./data/pg19_qwen3/test.bin"
SEQ_LEN=32768  # 2048 4096 8192 16384 32768 49152
CONTEXT_SIZE=40960

# ============================================================
# Evaluation mode
# ============================================================
# Eval mode: None (chunked, same as training) or "full" (full attention, no HiCI)
eval_mode=full

# ============================================================
# HiCI config (must match training!)
# ============================================================
use_local_constructor=True
use_global_integrator=True
NUM_LOCAL_SLOTS=8
global_slots=4
num_heads=8
use_bottleneck=True
bottleneck_dim=512

# LocalConstructor type
use_local_constructor_flash=False

# Forward function
use_hierarchical_forward=True

# ============================================================
# Logging
# ============================================================
echo "========================================"
echo "🔍 Qwen3-8B HiCI Multi-Node Evaluation"
echo "========================================"
echo "📦 Base model:       $BASE_MODEL"
echo "📁 Checkpoint:       $CHECKPOINT_PATH"
echo "🤖 GPUs:             $((NNODES * NPROC_PER_NODE)) (${NNODES} nodes x ${NPROC_PER_NODE})"
echo "🖥️  Current node:     $NODE_RANK / $((NNODES-1))  Master: $MASTER_ADDR:$MASTER_PORT"
echo "🗃️  Dataset:          $DATA_PATH"
echo "📊 Context size:     $CONTEXT_SIZE"
echo "🔢 Eval seq len:     $SEQ_LEN"
echo "🎯 Eval mode:        $eval_mode"
echo ""
echo "── HiCI Configuration ───────────────────"
echo "📝 LocalConstructor: $use_local_constructor"
echo "🔁 GlobalIntegrator: $use_global_integrator"
echo "🌐 Global slots:     $global_slots"
echo "🧠 Local slots:      $NUM_LOCAL_SLOTS"
echo "🎯 Bottleneck:       $use_bottleneck"
echo "🔢 Attention heads:  $num_heads"
echo "🧩 Bottleneck dim:   $bottleneck_dim"
echo ""
echo "── Forward Function ─────────────────────"
echo "🧠 LocalConstructorFlash: $use_local_constructor_flash"
echo "📝 Hierarchical fwd: $use_hierarchical_forward"
echo "========================================"

# Clean up stale processes
pkill -9 -f "eval_distributed_hici_qwen3.py" 2>/dev/null
fuser -k $MASTER_PORT/tcp 2>/dev/null
sleep 2

# --peft_model $CHECKPOINT_PATH \
torchrun \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --nproc_per_node=$NPROC_PER_NODE \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    eval_distributed_hici_qwen3.py \
    --base_model $BASE_MODEL \
    --data_path $DATA_PATH \
    --seq_len $SEQ_LEN \
    --context_size $CONTEXT_SIZE \
    --batch_size 1 \
    --flash_attn True \
    --use_local_constructor $use_local_constructor \
    --use_global_integrator $use_global_integrator \
    --num_local_slots $NUM_LOCAL_SLOTS \
    --global_slots $global_slots \
    --num_heads $num_heads \
    --use_bottleneck $use_bottleneck \
    --bottleneck_dim $bottleneck_dim \
    --eval_mode $eval_mode \
    --use_local_constructor_flash $use_local_constructor_flash \
    --use_hierarchical_forward $use_hierarchical_forward

echo ""
echo "========================================"
echo "✅ Evaluation completed! (Node $NODE_RANK)"
echo "========================================"
