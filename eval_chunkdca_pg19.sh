#!/bin/bash
# PG19 perplexity evaluation — baseline or DCA
# Usage:
#   bash eval_chunkdca_pg19.sh llama3            # DCA mode (default)
#   bash eval_chunkdca_pg19.sh qwen3
#   bash eval_chunkdca_pg19.sh llama3 baseline   # original model, no DCA
#   bash eval_chunkdca_pg19.sh qwen3  baseline

MODEL_TYPE=${1:-llama3}
MODE=${2:-dca}          # dca | baseline

source /scratch/iu02/xz2053/hici_env/bin/activate
cd /scratch/sh89/xz2053/projects/llm-memory

if [ "$MODEL_TYPE" = "llama3" ]; then
    MODEL_PATH="./models/Meta-Llama-3-8B"
    DATA_PATH="./data/pg19_llama3/test.bin"
    PRETRAINING_LENGTH=8192
    PPL_SCRIPT="ChunkLlama/ppl/test_ppl_distributed.py"
    LABEL="LLaMA-3-8B"
    NGPUS=4
    if [ "$MODE" = "baseline" ]; then
        SEQ_LENS="2048 4096 8192 16384 32768"
        NGPUS=3
    else
        SEQ_LENS="2048 4096 8192 16384 32768 49152 65536"
    fi
elif [ "$MODEL_TYPE" = "qwen3" ]; then
    MODEL_PATH="./models/Qwen3-8B"
    DATA_PATH="./data/pg19_qwen3/test.bin"
    PRETRAINING_LENGTH=40960
    PPL_SCRIPT="ChunkLlama/ppl/test_ppl_distributed_qwen3.py"
    LABEL="Qwen3-8B"
    NGPUS=4
    if [ "$MODE" = "baseline" ]; then
        SEQ_LENS="2048 4096 8192 16384 32768 40960"
    else
        SEQ_LENS="2048 4096 8192 16384 32768 65536 98304 131072"
    fi
else
    echo "ERROR: unknown model type '$MODEL_TYPE'. Use: llama3 or qwen3"
    exit 1
fi

if [ "$MODE" = "baseline" ]; then
    EXTRA_ARGS="--no_chunk"
    MODE_LABEL="Baseline (no DCA)"
    PORT=29601
else
    EXTRA_ARGS="--pretraining_length $PRETRAINING_LENGTH"
    MODE_LABEL="DCA"
    PORT=29600
fi

echo "Model:  $MODEL_PATH"
echo "Mode:   $MODE_LABEL"
echo "Data:   $DATA_PATH"
echo ""

for SEQ_LEN in $SEQ_LENS; do
    echo "============================================"
    echo "$MODE_LABEL | $LABEL | seq_len=$SEQ_LEN | ${NGPUS} GPUs"
    echo "============================================"
    torchrun --nproc_per_node=$NGPUS --master_port=$PORT \
        $PPL_SCRIPT \
        --model_path $MODEL_PATH \
        --data_path $DATA_PATH \
        --seq_len $SEQ_LEN \
        $EXTRA_ARGS
    echo ""
done
