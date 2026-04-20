# Qwen3-8B HiCI Evaluation on PG19
# bash eval_distributed_hici_qwen3.sh 2>&1 | tee eval_qwen3_Re/PG19_TEST_EVAL_Qwen3-8b_hici_merged_48k.txt

module load gcc/12.2.0
module load cuda/12.5.1

# Increase NCCL sync timeout (default 600s is insufficient; metric sync at eval end takes longer)
export NCCL_TIMEOUT=7200
export TORCH_NCCL_BLOCKING_WAIT=0

# Free port
fuser -k 38493/tcp 2>/dev/null || echo "Port 38493 not in use"
sleep 2
pkill -9 -f "eval_distributed_hici_qwen3.py"

BASE_MODEL="./models/Qwen3-8B"
# BASE_MODEL="./models/merged/Qwen3-8b-HiCI-48k-merged"
CHECKPOINT_PATH="./checkpoints/Qwen3-8b-HiCI-48k"
nproc_per_node=4
DATA_PATH="./data/pg19_qwen3/test.bin"  # validation or test
SEQ_LEN=2048  # 2048 4096 8192 16384 32768 49152
CONTEXT_SIZE=40960

# HiCI configuration (must match training!)
use_local_constructor=True
use_global_integrator=True
NUM_LOCAL_SLOTS=8
global_slots=4
num_heads=8
use_bottleneck=True
bottleneck_dim=512
shared_compress_dim=128
use_attn_init=False

# LocalConstructor type
use_local_constructor_flash=False

# Forward function
use_hierarchical_forward=True

# Eval mode: None (chunked, same as training) or "full" (full attention, no HiCI)
eval_mode=full

echo "========================================"
echo "🔍 Qwen3-8B HiCI Evaluation (PG-19)"
echo "========================================"
echo "📦 Base model:       $BASE_MODEL"
echo "📁 Checkpoint:       $CHECKPOINT_PATH"
echo "🤖 GPUs:             $nproc_per_node"
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

# --peft_model $CHECKPOINT_PATH \  # remove this line to evaluate the base model baseline
# --peft_model $CHECKPOINT_PATH \
torchrun --nproc_per_node=$nproc_per_node \
    --master_port=38493 \
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
    --use_hierarchical_forward $use_hierarchical_forward \
    --shared_compress_dim $shared_compress_dim \
    --use_attn_init $use_attn_init \
