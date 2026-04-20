#!/bin/bash
# HiCI environment setup script
# Usage: source setup_hici_env.sh [hici|qwen3]
# Default env: hici; extracts to $PBS_JOBFS (compute node local SSD, not subject to scratch quota)

# Determine extraction target directory
DEST="${PBS_JOBFS:-$TMPDIR}"
if [ -z "$DEST" ]; then
    echo "ERROR: PBS_JOBFS and TMPDIR are both unset — make sure you are running on a compute node"
    return 1 2>/dev/null || exit 1
fi

ENV_NAME="${1:-hici}"

if [ "$ENV_NAME" = "qwen3" ]; then
    TAR_FILE="/scratch/jp09/sx0401/hici_qwen3_env.tar.gz"
    ENV_DIR="$DEST/hici-qwen3"
else
    TAR_FILE="/scratch/jp09/sx0401/hici_env.tar.gz"
    ENV_DIR="$DEST/hici"
fi

# Check if already extracted
if [ -d "$ENV_DIR/bin" ]; then
    echo "Environment already exists: $ENV_DIR — skipping extraction"
else
    echo "Extracting $TAR_FILE to $DEST ..."
    tar xzf "$TAR_FILE" -C "$DEST"
    echo "Extraction complete"
fi

# Set environment variables
export PATH="$ENV_DIR/bin:$PATH"
export CONDA_PREFIX="$ENV_DIR"
export CUDA_HOME=/apps/cuda/12.5.1
export DS_SKIP_CUDA_CHECK=1

echo "========================================"
echo "✅ Environment ready"
echo "📁 Env dir:  $ENV_DIR"
echo "🐍 Python:   $(which python)"
echo "🔥 PyTorch:  $(python -c 'import torch; print(torch.__version__)' 2>/dev/null)"
echo "⚡ CUDA:     $CUDA_HOME"
echo "========================================"
