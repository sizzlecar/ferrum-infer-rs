#!/bin/bash
# GPU 机器一键初始化脚本
# 用法: curl -sSf https://raw.githubusercontent.com/sizzlecar/ferrum-infer-rs/main/scripts/setup_gpu.sh | bash
# 或: bash scripts/setup_gpu.sh

set -e

echo "=== Ferrum GPU Machine Setup ==="

# 1. 安装 Rust
if ! command -v rustc &>/dev/null; then
    echo "[1/5] Installing Rust..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
else
    echo "[1/5] Rust already installed: $(rustc --version)"
    source "$HOME/.cargo/env" 2>/dev/null || true
fi

# 2. 系统依赖
echo "[2/5] Installing system dependencies..."
apt-get update -qq && apt-get install -y -qq pkg-config libssl-dev git > /dev/null 2>&1

# 3. 检查 CUDA
echo "[3/5] Checking CUDA..."
if [ -f /usr/local/cuda/include/cuda.h ]; then
    export CUDA_HOME=/usr/local/cuda
    echo "  CUDA_HOME=$CUDA_HOME"
    nvcc --version 2>/dev/null | head -1 || echo "  nvcc not in PATH"
elif [ -f /usr/local/cuda-12/include/cuda.h ]; then
    export CUDA_HOME=/usr/local/cuda-12
    echo "  CUDA_HOME=$CUDA_HOME"
else
    echo "  WARNING: CUDA not found. Install CUDA Toolkit first."
fi

# 检查 GPU
nvidia-smi --query-gpu=name,memory.total,count --format=csv,noheader 2>/dev/null || echo "  No GPU detected"
GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader 2>/dev/null | head -1 || echo "0")
echo "  GPU count: $GPU_COUNT"

# 4. 克隆仓库
echo "[4/5] Cloning repository..."
cd /workspace 2>/dev/null || cd ~
if [ -d ferrum-infer-rs ]; then
    cd ferrum-infer-rs
    git fetch origin
    git checkout main
    git pull origin main
    echo "  Repository updated"
else
    git clone https://github.com/sizzlecar/ferrum-infer-rs.git
    cd ferrum-infer-rs
    echo "  Repository cloned"
fi

# 切换到需要测试的分支
for branch in feat/qwen2-cuda-runner feat/batched-flash-decode feat/tensor-parallel; do
    git branch -D $branch 2>/dev/null || true
done

# 5. 编译
echo "[5/5] Building (this takes ~5 minutes)..."
export HF_HOME=${HF_HOME:-/workspace/.hf_home}
mkdir -p $HF_HOME

cargo build --release -p ferrum-cli --features cuda 2>&1 | tail -3

echo ""
echo "=== Setup Complete ==="
echo "CUDA_HOME=$CUDA_HOME"
echo "HF_HOME=$HF_HOME"
echo "Binary: $(pwd)/target/release/ferrum"
echo ""
echo "Quick test:"
echo "  ./target/release/ferrum pull tinyllama"
echo "  ./target/release/ferrum bench tinyllama --rounds 2"
echo ""
echo "Run full test suite:"
echo "  bash scripts/test_gpu.sh"
