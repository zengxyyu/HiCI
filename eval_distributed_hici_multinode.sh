#!/bin/bash
# HiCI Distributed Evaluation — Multi-Node (2 nodes x 4 H200 = 8 GPUs)
#
# Usage:
#   Node 0 (master): bash eval_distributed_hici_multinode.sh 0 2>&1 | tee eval_llama2-7b-8k_Re/PG19_TEST_EVAL_Llama-2-Llama-2-7b-8k-hici-causal_gi-G8_2000_4k_astrain_S1024.txt
#   Node 1 (worker): bash eval_distributed_hici_multinode.sh 1
#
# Both must be started within NCCL timeout (default 120s) of each other.

NODE_RANK=${1:?'Usage: bash eval_distributed_hici_multinode.sh <node_rank> (0=master, 1=worker)'}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
cd "$SCRIPT_DIR"
export DS_SKIP_CUDA_CHECK=1

# Activate environment: prefer PBS_JOBFS extracted env, fallback to conda/venv
if [ -d "${PBS_JOBFS}/hici/bin" ]; then
    export PATH="${PBS_JOBFS}/hici/bin:$PATH"
    export CONDA_PREFIX="${PBS_JOBFS}/hici"
else
    source ~/venv/zxy/bin/activate 2>/dev/null || conda activate hici 2>/dev/null
fi

# ============================================================
# Multi-node config
# ============================================================
MASTER_ADDR="10.6.106.17"   # Update to master node IP (e.g., gadi-gpu-h200-XXXX)
MASTER_PORT=38493
NNODES=2
NPROC_PER_NODE=4

# NCCL
export NCCL_TIMEOUT=7200
export NCCL_BLOCKING_WAIT=1

# ============================================================
# Model & Data
# ============================================================
BASE_MODEL="./models/Llama-2-7b-hf"
# BASE_MODEL="./models/Llama-2-13b-hf"
# BASE_MODEL="./models/Meta-Llama-3-8B"
CHECKPOINT_PATH="./checkpoints/Llama-2-7b-8k-hici-causal_gi-G8/checkpoint-2000"

DATA_PATH="./data/pg19/test.bin"
SEQ_LEN=8192            # 2048 4096 8192 16384 32768 65536 100000
CONTEXT_SIZE=8192

# ============================================================
# Evaluation mode
# ============================================================
# Eval mode: None (chunked, same as training) or "full" (full attention, no HiCI)
eval_mode=None

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
echo "🔍 LLaMA-2/3 HiCI Multi-Node Evaluation"
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
pkill -9 -f "eval_distributed_hici.py" 2>/dev/null
fuser -k $MASTER_PORT/tcp 2>/dev/null
sleep 2

# ============================================================
# Run
# ============================================================
torchrun \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --nproc_per_node=$NPROC_PER_NODE \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    eval_distributed_hici.py \
    --base_model $BASE_MODEL \
    --peft_model $CHECKPOINT_PATH \
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
