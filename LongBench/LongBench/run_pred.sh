#!/bin/bash
# LongBench evaluation script for HiCI models
#
# Two modes (corresponding to paper results):
#   1. Baseline — standard full attention, no HiCI:
#      bash run_pred.sh --model <model-name> --ori --suffix "-ori"
#
#   2. HiCI — HiCI hierarchical attention in prefill (entire sequence as one group, no segmentation):
#      bash run_pred.sh --model <model-name> --suffix "_hici"

source ~/venv/env/bin/activate
cd /path/to/LongBench/LongBench

MODEL="hici-7b-chat-sft-16k-no-5epoch"
GPUS="0,1"
SUFFIX=""
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

if [ -z "$USE_HICI_ATTN" ]; then
    MODE="Baseline: standard full attention (no HiCI)"
else
    MODE="HiCI: hierarchical attention in prefill (entire sequence as one group)"
fi

echo "========================================"
echo "LongBench Evaluation"
echo "========================================"
echo "Model:  $MODEL"
echo "GPUs:   $GPUS"
echo "Mode:   $MODE"
echo "Suffix: $SUFFIX"
echo "========================================"

ARGS=("--model" "$MODEL")

if [ -n "$USE_HICI_ATTN" ]; then
    ARGS+=("--use_hici_attn")
fi

if [ -n "$SUFFIX" ]; then
    ARGS+=("--output_suffix=$SUFFIX")
fi

echo "Command: CUDA_VISIBLE_DEVICES=$GPUS python pred.py ${ARGS[*]}"
CUDA_VISIBLE_DEVICES=$GPUS python pred.py "${ARGS[@]}"
