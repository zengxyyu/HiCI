#!/bin/bash
# ====================================================
# Topic Retrieval Scoring Script
# Reads prediction files and computes accuracy via topic_retrieval_manual_eval.py.
# ====================================================
# Usage:
#   bash eval_topic_retrieval_score.sh              # score baseline models (no suffix)
#   bash eval_topic_retrieval_score.sh full         # score _full mode (LongLoRA/HiCI)
#   bash eval_topic_retrieval_score.sh grouped      # score _grouped mode (HiCI)
# ====================================================

HICI_DIR="/g/data/hn98/Yang/llm-mem/HiCI"
PRED_DIR="$HICI_DIR/LongChat/longeval/evaluation/topics/predictions"
MANUAL_EVAL="$HICI_DIR/topic_retrieval_manual_eval.py"
OUTPUT_DIR="$HICI_DIR/eval_topic_retrieval"

MODE_SUFFIX="${1:-}"
if [ -n "$MODE_SUFFIX" ]; then
    MODE_SUFFIX="_${MODE_SUFFIX}"
fi

mkdir -p "$OUTPUT_DIR"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Topic Retrieval Scoring"
echo "Mode suffix : ${MODE_SUFFIX:-(none)}"
echo "Pred dir    : $PRED_DIR"
echo "Output dir  : $OUTPUT_DIR"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# ====================================================
# score_model <MODEL_NAME>
# Scores all topic counts (5/10/15/20/25) for one model.
# ====================================================
score_model() {
    local MODEL_NAME="$1"
    local PRED_MODEL_DIR="${PRED_DIR}/${MODEL_NAME}${MODE_SUFFIX}"
    local OUTPUT_FILE="${OUTPUT_DIR}/${MODEL_NAME}${MODE_SUFFIX}_score.txt"

    echo "Scoring : $MODEL_NAME${MODE_SUFFIX}"
    echo "Pred dir: $PRED_MODEL_DIR"
    echo "Output  : $OUTPUT_FILE"

    if [ ! -d "$PRED_MODEL_DIR" ]; then
        echo "WARNING: Prediction directory not found, skipping: $PRED_MODEL_DIR"
        echo ""
        return
    fi

    echo "Model: $MODEL_NAME${MODE_SUFFIX}" > "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE"

    for num_topics in 5 10 15 20 25; do
        test_file="$PRED_MODEL_DIR/${num_topics}_response.txt"

        if [ -f "$test_file" ]; then
            echo "━━━ ${num_topics} topics ━━━" | tee -a "$OUTPUT_FILE"
            python3 "$MANUAL_EVAL" "$test_file" | tee -a "$OUTPUT_FILE"
            echo "" | tee -a "$OUTPUT_FILE"
        else
            echo "SKIP: ${num_topics}_response.txt not found" | tee -a "$OUTPUT_FILE"
            echo "" | tee -a "$OUTPUT_FILE"
        fi
    done

    echo "Done: results saved to $OUTPUT_FILE"
    echo ""
}

# ====================================================
# Models to score — add/comment as needed
# ====================================================

# HiCI / LongLoRA models (use with: full or grouped argument)
score_model "Llama-2-13b-HiCI-16k-wO-merged_full"
# score_model "Llama-2-13b-longlora-16k-merged"
# score_model "Llama-2-13b-HiCI-16k-S2048"

# Baseline models (use without argument)
# score_model "mpt-30b-chat"
# score_model "longchat-13b-16k"
# score_model "mpt-7b-storywriter"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "All scoring complete."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
