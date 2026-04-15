# Qwen3-8B HiCI Evaluation on PG19
# bash eval_distributed_hici_qwen3.sh 2>&1 | tee eval_qwen3_Re/PG19_TEST_EVAL_Qwen3-8b_hici_merged_8k.txt

module load gcc/12.2.0
module load cuda/12.5.1

# 增大 NCCL 同步超时（默认 600s 不够，评估结束时 metric sync 耗时长）
export NCCL_TIMEOUT=7200
export TORCH_NCCL_BLOCKING_WAIT=0

# 清理端口
fuser -k 38493/tcp 2>/dev/null || echo "Port 38493 not in use"
sleep 2
pkill -9 -f "eval_distributed_hici_qwen3.py"

# BASE_MODEL="./models/Qwen3-8B"
BASE_MODEL="./models/merged/Qwen3-8b-HiCI-48k-merged"
CHECKPOINT_PATH="./checkpoints/Qwen3-8b-HiCI-48k"
nproc_per_node=2
DATA_PATH="./data/pg19_qwen3/test.bin" #validation
SEQ_LEN=8192  # 2048 4096 8192 16384 32768 49152
CONTEXT_SIZE=40960

# HiCI 参数（必须和训练时一致）
use_local_constructor=True
use_global_integrator=True
NUM_LOCAL_SLOTS=8
global_slots=4
num_heads=8
use_bottleneck=True
bottleneck_dim=512
use_local_constructor_flash=False
use_hierarchical_forward=True

# 评估模式: None (chunked, same as training), "full" (full attention no HiCI)
eval_mode=full

echo "================================"
echo "📦 基础模型: $BASE_MODEL"
echo "📁 评估模型的目录: $CHECKPOINT_PATH"
echo "🤖 GPU数目: $nproc_per_node"
echo "🗃️ 评估数据集: $DATA_PATH"
echo "📊 最大长度: $CONTEXT_SIZE"
echo "🔢 评估的序列长度: $SEQ_LEN"
echo "🎯 评估方式: $eval_mode"
echo "--------------记忆属性设置-----------------"
echo "📝 使用局部摘要记忆机制: $use_local_constructor"
echo "🔁 使用高层全局记忆机制: $use_global_integrator"
echo "🌐 Global Representation Slots: $global_slots"
echo "🧠 Local Representation Slots: $NUM_LOCAL_SLOTS"
echo "🎯 使用 Bottleneck: $use_bottleneck"
echo "🔢 HiCI Attention Heads: $num_heads"
echo "🧩 Bottleneck Dimension: $bottleneck_dim"
echo "--------------记忆类和前馈函数设置-----------------"
echo "🧠 LocalConstructorFlash (flash attn): $use_local_constructor_flash"
echo "📝 调用函数 forward_flashattn_hierarchical 局部+全局: $use_hierarchical_forward"
echo "================================"

# --peft_model $CHECKPOINT_PATH \  # 去掉此行则评估原始模型 baseline 
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
