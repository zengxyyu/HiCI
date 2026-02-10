#!/bin/bash
# HiCI inference script
#
# Four modes:
# 1. Use training function (slow but consistent): --use_training_function
# 2. Use KV cache + repr (efficient): --include_repr_in_kv_cache
# 3. No HiCI (original inference): --ori
# 4. Prefill with HiCI but no KV cache injection (default)
#
# Examples:
#   bash run_hici_train_func.sh --use_training_function --suffix "_train_func"
#   bash run_hici_train_func.sh --include_repr_in_kv_cache --suffix "_kv_repr"
#   bash run_hici_train_func.sh --ori --suffix "_no_hici"
#   bash run_hici_train_func.sh --suffix "_hici_no_inject"

source ~/venv/env/bin/activate
cd /path/to/LongBench/LongBench

MODEL="hici-7b-chat-sft-16k-no-5epoch"
GPUS="0,1"
SUFFIX=""
USE_TRAINING_FUNC=""
INCLUDE_REPR=""
DISABLE_HICI=""
USE_HICI_ATTN="--use_hici_attn"

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --gpus)
            GPUS="$2"
            shift 2
            ;;
        --suffix)
            SUFFIX="$2"
            shift 2
            ;;
        --use_training_function)
            USE_TRAINING_FUNC="--use_training_function"
            shift
            ;;
        --include_repr_in_kv_cache)
            INCLUDE_REPR="--include_repr_in_kv_cache"
            shift
            ;;
        --disable_hici_in_prefill)
            DISABLE_HICI="--disable_hici_in_prefill"
            shift
            ;;
        --ori)
            USE_HICI_ATTN=""
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [ -n "$SUFFIX" ]; then
    OUTPUT_SUFFIX="--output_suffix"
    OUTPUT_SUFFIX_VALUE="$SUFFIX"
else
    OUTPUT_SUFFIX=""
    OUTPUT_SUFFIX_VALUE=""
fi

if [ -z "$USE_HICI_ATTN" ]; then
    MODE="Mode 3: No HiCI (original inference)"
elif [ -n "$USE_TRAINING_FUNC" ]; then
    MODE="Mode 1: Training function (no KV cache)"
elif [ -n "$INCLUDE_REPR" ]; then
    MODE="Mode 2: KV cache + repr"
elif [ -n "$DISABLE_HICI" ]; then
    MODE="Mode 3b: No HiCI in prefill, use KV cache"
else
    MODE="Mode 4: HiCI in prefill, no KV cache injection (default)"
fi

echo "========================================"
echo "HiCI Inference"
echo "========================================"
echo "Model: $MODEL"
echo "GPUs: $GPUS"
echo "Mode: $MODE"
echo "Use HiCI Attn: $USE_HICI_ATTN"
echo "Options: $USE_TRAINING_FUNC $INCLUDE_REPR $DISABLE_HICI"
echo "Output suffix: $SUFFIX"
echo "========================================"

ARGS=("--model" "$MODEL")

if [ -n "$USE_HICI_ATTN" ]; then
    ARGS+=("--use_hici_attn")
fi

if [ -n "$SUFFIX" ]; then
    ARGS+=("--output_suffix=$SUFFIX")
fi

if [ -n "$USE_TRAINING_FUNC" ]; then
    ARGS+=("--use_training_function")
fi

if [ -n "$INCLUDE_REPR" ]; then
    ARGS+=("--include_repr_in_kv_cache")
fi

if [ -n "$DISABLE_HICI" ]; then
    ARGS+=("--disable_hici_in_prefill")
fi

echo "Command: CUDA_VISIBLE_DEVICES=$GPUS python pred.py ${ARGS[*]}"
CUDA_VISIBLE_DEVICES=$GPUS python pred.py "${ARGS[@]}"
