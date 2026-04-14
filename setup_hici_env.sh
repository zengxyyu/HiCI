#!/bin/bash
# HiCI 环境初始化脚本
# 用法: source setup_hici_env.sh [hici|qwen3]
# 默认加载 hici 环境，解压到 $PBS_JOBFS（计算节点本地SSD，不受scratch配额限制）

# 确定解压目标目录
DEST="${PBS_JOBFS:-$TMPDIR}"
if [ -z "$DEST" ]; then
    echo "错误: PBS_JOBFS 和 TMPDIR 都未设置，请确认在计算节点上运行"
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

# 检查是否已解压
if [ -d "$ENV_DIR/bin" ]; then
    echo "环境已存在: $ENV_DIR，跳过解压"
else
    echo "正在解压 $TAR_FILE 到 $DEST ..."
    tar xzf "$TAR_FILE" -C "$DEST"
    echo "解压完成"
fi

# 设置环境变量
export PATH="$ENV_DIR/bin:$PATH"
export CONDA_PREFIX="$ENV_DIR"
export CUDA_HOME=/apps/cuda/12.5.1
export DS_SKIP_CUDA_CHECK=1

echo "================================"
echo "环境: $ENV_DIR"
echo "Python: $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null)"
echo "CUDA_HOME: $CUDA_HOME"
echo "================================"
