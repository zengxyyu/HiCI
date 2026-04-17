#!/bin/bash
# ====================================================
# Topic Retrieval Evaluation Script (LongLoRA / HiCI)
# ====================================================
# bash eval_topic_retrieval_predict.sh 2>&1 | tee ./eval_topic_retrieval/Llama-2-13b-HiCI-16k-S2048-full.log

# Cleanup function: kill eval processes on exit
cleanup() {
    echo ""
    echo "Cleaning up eval.py processes..."
    pkill -9 -f "eval.py" 2>/dev/null || true
    pkill -9 -f "longeval" 2>/dev/null || true
    sleep 1
    echo "Cleanup done."
}

trap cleanup EXIT

set -e  # Exit immediately on error

# ====================================================
# Configuration
# ====================================================
MODEL_BASE_DIR="./models"

# Model name — change this to switch models
# MODEL_NAME="Llama-2-13b-longlora-18k-ft"
# MODEL_NAME="Llama-2-13b-longlora-16k-merged"
# MODEL_NAME="merged_models/Llama-2-13b-HiCI-16k"
# MODEL_NAME="merged_models/Llama-2-13b-HiCI-16k-S2048"
# MODEL_NAME="merged_models/Llama-2-7b-HiCI-32k"
MODEL_NAME="merged/Llama-2-13b-HiCI-16k-wO-merged"

# Resolve to absolute path so it stays valid after cd
MODEL_PATH="$(realpath ${MODEL_BASE_DIR}/${MODEL_NAME})"

# ====================================================
# HiCI attention mode
# ====================================================
# true  = HiCI grouped attention (matches training)
# false = standard full attention
USE_HICI_GROUPED_ATTN=false
HICI_SEGMENT_SIZE=4096   # tokens per segment (must match training)
# num_groups varies dynamically: num_groups = input_len / segment_size

LONGCHAT_DIR="./LongChat"

# ====================================================
# GPU configuration
# ====================================================
# Leave CUDA_DEVICES empty to use all visible GPUs
# e.g. "0,1,2,3" or "2,3,4,5"
CUDA_DEVICES="0,1"
NUM_GPUS=2     # must match number of GPUs in CUDA_DEVICES
MAX_GPU_MEMORY=80

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Topic Retrieval Evaluation (LongLoRA/HiCI)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Model name : $MODEL_NAME"
echo "Model path : $MODEL_PATH"
echo "LongChat   : $LONGCHAT_DIR"
echo "GPUs       : ${CUDA_DEVICES:-all visible}"
echo "Num GPUs   : $NUM_GPUS"
echo "GPU memory : ${MAX_GPU_MEMORY}GB each"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# ====================================================
# Step 1: Check model files
# ====================================================
echo "Step 1: Checking model files..."
if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: Model path not found: $MODEL_PATH"
    exit 1
fi

if [ ! -f "$MODEL_PATH/config.json" ]; then
    echo "ERROR: config.json not found: $MODEL_PATH/config.json"
    exit 1
fi

echo "Model files OK."
echo ""

# ====================================================
# Step 2: Check LongChat directory
# ====================================================
echo "Step 2: Checking LongChat repo..."
if [ ! -d "$LONGCHAT_DIR" ]; then
    echo "ERROR: LongChat directory not found: $LONGCHAT_DIR"
    exit 1
fi

cd "$LONGCHAT_DIR"

echo "LongChat directory OK."
echo ""

# ====================================================
# Step 3: Run Topic Retrieval evaluation
# ====================================================
echo "Step 3: Starting Topic Retrieval evaluation..."
echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

cd longeval

# Add HiCI root to PYTHONPATH so Python can find llama_flash_attn_fixed.py
export PYTHONPATH="/g/data/hn98/Yang/llm-mem/HiCI:$PYTHONPATH"

if [ -n "$CUDA_DEVICES" ]; then
    export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"
    echo "Using GPUs: $CUDA_DEVICES"
else
    echo "Using all visible GPUs."
fi

if [ "$USE_HICI_GROUPED_ATTN" = true ]; then
    echo "Mode: HiCI grouped attention (segment_size=$HICI_SEGMENT_SIZE)"
    python3 eval.py \
        --model-name-or-path "$MODEL_PATH" \
        --task topics \
        --num_gpus $NUM_GPUS \
        --max_gpu_memory $MAX_GPU_MEMORY \
        --enable_hici_grouped_attn \
        --hici_segment_size $HICI_SEGMENT_SIZE &
else
    echo "Mode: Full attention"
    python3 eval.py \
        --model-name-or-path "$MODEL_PATH" \
        --task topics \
        --num_gpus $NUM_GPUS \
        --max_gpu_memory $MAX_GPU_MEMORY \
        --enable_longlora_flash_attn &
fi

EVAL_PID=$!
echo "Eval PID: $EVAL_PID"

wait $EVAL_PID
EVAL_EXIT_CODE=$?

echo ""
echo "End time : $(date '+%Y-%m-%d %H:%M:%S')"
echo "Exit code: $EVAL_EXIT_CODE"
echo ""

# ====================================================
# Step 4: Show results
# ====================================================
echo "Step 4: Evaluation finished. Checking results..."
echo ""

MODEL_NAME_BASE=$(basename "$MODEL_PATH")
if [ "$USE_HICI_GROUPED_ATTN" = true ]; then
    OUTPUT_DIR="$LONGCHAT_DIR/longeval/evaluation/topics/predictions/${MODEL_NAME_BASE}_grouped"
else
    OUTPUT_DIR="$LONGCHAT_DIR/longeval/evaluation/topics/predictions/${MODEL_NAME_BASE}_full"
fi

if [ -d "$OUTPUT_DIR" ]; then
    echo "Results saved to: $OUTPUT_DIR"
    echo ""
    echo "Generated files:"
    ls -lh "$OUTPUT_DIR"
    echo ""

    for file in "$OUTPUT_DIR"/*.txt; do
        if [ -f "$file" ]; then
            echo "--- $(basename $file) preview ---"
            head -3 "$file"
            echo ""
        fi
    done
else
    echo "WARNING: Output directory not found: $OUTPUT_DIR"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Evaluation complete."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Next step: GPT auto-scoring"
echo ""
echo "If you have an OpenAI API key:"
echo "  export OPENAI_API_KEY='your-api-key'"
echo "  cd $LONGCHAT_DIR/longeval"
echo "  python3 auto_topic_eval.py --test_file $OUTPUT_DIR/*.txt"
echo ""
echo "Or manually inspect the result .txt files."
echo ""
