# bash eval_distributed_hici.sh 2>&1 | tee eval_llama2-7b-8k_Re/PG19_TEST_EVAL_Llama-2-7b-8k-hici-causal_shift_g-G8_2000_2k_ori.txt

# Free port
fuser -k 38493/tcp 2>/dev/null || echo "Port 38493 not in use"
sleep 2
pkill -9 -f "eval_distributed_hici.py"

# ./data/pg19/validation.bin   ./data/pg19/test.bin
# ./data/proof-pile/test_sampled_data.bin
BASE_MODEL="./models/Llama-2-7b-hf"
# BASE_MODEL="./models/Llama-2-13b-hf"
# BASE_MODEL="./models/Meta-Llama-3-8B"
CHECKPOINT_PATH="./checkpoints/Llama-2-7b-8k-hici-causal_shift_g-G8/checkpoint-2000"
nproc_per_node=4
DATA_PATH="./data/pg19/test.bin"
SEQ_LEN=2048  # 2048 4096 8192 16384 32768 65536 100000
CONTEXT_SIZE=8192

# HiCI configuration (must match training!)
use_local_constructor=True
use_global_integrator=True
NUM_LOCAL_SLOTS=8
global_slots=4  # Global Representation Slots
num_heads=8     # number of attention heads
use_bottleneck=True
bottleneck_dim=512
shared_compress_dim=128

# Eval mode: None (chunked, same as training) or "full" (full attention, no HiCI)
eval_mode=None

# LocalConstructor type
use_local_constructor_flash=False

# Forward function
use_hierarchical_forward=True

echo "========================================"
echo "🔍 LLaMA-2/3 HiCI Evaluation (PG-19)"
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

# --peft_model $CHECKPOINT_PATH \
torchrun --nproc_per_node=$nproc_per_node \
    --master_port=38493 \
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
    --use_hierarchical_forward $use_hierarchical_forward \
    --shared_compress_dim $shared_compress_dim \
