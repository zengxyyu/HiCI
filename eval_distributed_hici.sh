conda activate hici
cd /scratch/sx11/sx0401/workspace/xiangyu/llm-memory
# eval_llama2-7b-8k_basline visual_7b
# bash eval_distributed_hici.sh 2>&1 | tee eval_llama2-7b-8k_Re/PG19_TEST_EVAL_Llama-2-7b-8k-hici-causal_shift_g-G8_2000_2k_ori.txt
# PG19_EVAL_Llama-2-7b-8k-FTM-NEW-8-only1localsummary_noR_4e_2000_astrain.txt
GRAB_SCRIPT="$(cd "$(dirname "$0")" && pwd)/grab_gpu.sh"
trap 'echo ""; echo "🎯 脚本退出，开始在 tmux 中抢占GPU..."; tmux new-session -d -s gpu_grab "bash $GRAB_SCRIPT" 2>/dev/null || tmux send-keys -t gpu_grab "bash $GRAB_SCRIPT" C-m; echo "✅ 抢占脚本已在 tmux session \"gpu_grab\" 中启动"; echo "💡 查看抢占状态: tmux attach -t gpu_grab"' EXIT
# 清理端口
fuser -k 38493/tcp 2>/dev/null || echo "Port 38493 not in use"
sleep 2
pkill -9 -f "eval_distributed_hici.py"
# ./data/pg19/validation.bin   ./data/pg19/test.bin
# ./data/proof-pile/test_sampled_data.bin
# Llama-2-7b-8k-memory-inject-cache  Llama-2-7b-8k-memory-inject
# Llama-2-7b-8k-FTM-NEW-84-bothhigher_multi_clip_2e_clean_share
# Llama-2-7b-16k-FTM-NEW-75-bothhigher_multi_clip_2e_clean_share_woO
# Llama-2-7b-8k-longlora-ori
# Llama-2-13b-32k-FTM-NEW-84-bothhigher_multi_clip_2e_clean_share
# Llama-2-7b-100k-FTM-NEW-84-bothhigher_multi_clip_2e_clean_share_G10
# BASE_MODEL="./models/merged_models/Llama-2-13b-64k-FTM-NEW-75-merged-1000-hici-S2048"
BASE_MODEL="./models/Llama-2-7b-hf"
# BASE_MODEL="./models/Llama-2-13b-hf"
# BASE_MODEL="./models/Meta-Llama-3-8B"
CHECKPOINT_PATH="./checkpoints/Llama-2-7b-8k-hici-causal_shift_g-G8/checkpoint-2000"
nproc_per_node=4
DATA_PATH="./data/pg19/test.bin"
SEQ_LEN=2048  # 2048 4096 8192 16384 32768 65536 100000
CONTEXT_SIZE=8192
use_local_constructor=True  # 是否使用本地摘要记忆机制 False
use_global_integrator=True  # 是否使用高层压缩HierarchicalMemoryAggregatorSingleHead
NUM_LOCAL_SLOTS=8  # loca memory slots
global_slots=4  # Global Representation Slots
num_heads=8  # number of attention heads
use_bottleneck=True  # whether to use bottleneck in hierarchical memory aggregator
bottleneck_dim=512  # bottleneck dimension for hierarchical memory aggregator
eval_mode=None   # 评估方式: None (chunked, same as training) or "full" (full attention, no memory)

# LocalConstructor 类型选择
use_local_constructor_flash=False

# 前馈函数
use_hierarchical_forward=True

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
# echo "🧠 记忆的kv是否从Llama参数初始化: $use_llama_init"
echo "🎯 使用 Bottleneck: $use_bottleneck"
echo "🔢 Memory Attention Heads: $num_heads"  #（目前没有使用这个参数）
echo "🧩 Bottleneck Dimension: $bottleneck_dim"
echo "--------------前馈函数设置-----------------"
echo "🧠 LocalConstructorFlash (use_local_constructor_flash): $use_local_constructor_flash"
echo "📝 forward_flashattn_hierarchical (use_hierarchical_forward): $use_hierarchical_forward"
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



    