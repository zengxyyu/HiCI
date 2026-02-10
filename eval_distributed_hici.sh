fuser -k 38493/tcp 2>/dev/null || echo "Port 38493 not in use"
sleep 2
pkill -9 -f "eval_distributed_hici.py"

BASE_MODEL="/path/to/models/Llama-2-13b-hf"
CHECKPOINT_PATH="./checkpoints/Llama-2-13b-32k-HiCI/checkpoint-1000"
nproc_per_node=8
DATA_PATH="./data/pg19/test.bin"
SEQ_LEN=8192
CONTEXT_SIZE=32768
use_local_repr=True
use_global_repr=True
NUM_LOCAL_SLOTS=8
global_slots=4
num_heads=10
use_bottleneck=True
bottleneck_dim=640
eval_mode=None

use_flash_attn_in_hici=False
use_hierarchical_forward=True
use_flash_plus_norope=False
use_flash_plus=False
forward_flashattn_optimized=True

echo "================================"
echo "Base model: $BASE_MODEL"
echo "Checkpoint: $CHECKPOINT_PATH"
echo "GPUs: $nproc_per_node"
echo "Data: $DATA_PATH"
echo "Context size: $CONTEXT_SIZE"
echo "Sequence length: $SEQ_LEN"
echo "Eval mode: $eval_mode"
echo "--- HiCI Config ---"
echo "use_local_repr: $use_local_repr"
echo "use_global_repr: $use_global_repr"
echo "Global slots: $global_slots"
echo "Local slots: $NUM_LOCAL_SLOTS"
echo "use_bottleneck: $use_bottleneck"
echo "num_heads: $num_heads"
echo "bottleneck_dim: $bottleneck_dim"
echo "--- Forward Config ---"
echo "use_flash_plus (LocalConstructorFlashPlus): $use_flash_plus"
echo "use_flash_attn_in_hici (LocalConstructorFlash): $use_flash_attn_in_hici"
echo "use_hierarchical_forward: $use_hierarchical_forward"
echo "use_flash_plus_norope: $use_flash_plus_norope"
echo "forward_flashattn_optimized: $forward_flashattn_optimized"
echo "================================"

torchrun --nproc_per_node=8 \
    --master_port=38493 \
    eval_distributed_hici.py \
    --base_model $BASE_MODEL \
    --peft_model $CHECKPOINT_PATH \
    --data_path $DATA_PATH \
    --seq_len $SEQ_LEN \
    --context_size $CONTEXT_SIZE \
    --batch_size 1 \
    --flash_attn True \
    --use_local_repr $use_local_repr \
    --use_global_repr $use_global_repr \
    --num_local_slots $NUM_LOCAL_SLOTS \
    --global_slots $global_slots \
    --num_heads $num_heads \
    --use_bottleneck $use_bottleneck \
    --bottleneck_dim $bottleneck_dim \
    --eval_mode $eval_mode \
    --use_flash_plus $use_flash_plus \
    --use_flash_attn_in_hici $use_flash_attn_in_hici \
    --use_hierarchical_forward $use_hierarchical_forward \
    --use_flash_plus_norope $use_flash_plus_norope \
    --forward_flashattn_optimized $forward_flashattn_optimized \
