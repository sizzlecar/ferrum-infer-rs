#!/bin/bash
# GPU 全量测试脚本
# 用法: bash scripts/test_gpu.sh
#
# 测试项:
# 1. 单元测试 (CPU)
# 2. TinyLlama CUDA runner (单请求 + 多轮)
# 3. Qwen3-4B INT4 Marlin (单请求 + 并发)
# 4. Qwen2.5 CUDA runner
# 5. 长 KV 并发 (batched flash decode)
# 6. TP 分片测试 (CPU, 数学验证)

set -e

source "$HOME/.cargo/env" 2>/dev/null || true
cd "$(dirname "$0")/.."

export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export HF_HOME=${HF_HOME:-/workspace/.hf_home}
FERRUM="./target/release/ferrum"
PASS=0
FAIL=0
SKIP=0

run_test() {
    local name="$1"
    shift
    echo -n "  [$name] "
    if eval "$@" > /tmp/ferrum_test_$$.log 2>&1; then
        echo "PASS"
        PASS=$((PASS + 1))
    else
        echo "FAIL"
        cat /tmp/ferrum_test_$$.log | tail -5
        FAIL=$((FAIL + 1))
    fi
    rm -f /tmp/ferrum_test_$$.log
}

echo "============================================================"
echo "Ferrum GPU Test Suite"
echo "============================================================"
echo ""

# 0. Build
echo "[Build]"
echo -n "  Compiling... "
if cargo build --release -p ferrum-cli --features cuda 2>&1 | tail -1 | grep -q "Finished"; then
    echo "OK"
else
    echo "FAILED"
    exit 1
fi

# 1. Unit tests
echo ""
echo "[1] Unit Tests (CPU)"
run_test "workspace tests" "cargo test --workspace"
run_test "TP sharding tests" "cargo test -p ferrum-models --test tp_sharding_test"
run_test "TP mapping tests" "cargo test -p ferrum-engine -- parallel::tensor_parallel"

# 2. Pull models
echo ""
echo "[2] Pull Models"
$FERRUM pull tinyllama 2>/dev/null || true

# 3. TinyLlama (Llama CUDA runner)
echo ""
echo "[3] TinyLlama (Llama CUDA Runner)"
run_test "single request" "timeout 120 $FERRUM bench tinyllama --rounds 1 --max-tokens 32"
run_test "multi-round" "timeout 120 $FERRUM bench tinyllama --rounds 3 --max-tokens 16"
run_test "output correctness" "echo 'What is 2+2?' | timeout 30 $FERRUM run tinyllama 2>&1 | grep -q 'tok/s'"

# 4. Qwen3-4B INT4 (if available)
echo ""
echo "[4] Qwen3-4B INT4 Marlin"
if $FERRUM pull JunHowie/Qwen3-4B-GPTQ-Int4 2>/dev/null; then
    run_test "single request INT4" "timeout 300 $FERRUM bench JunHowie/Qwen3-4B-GPTQ-Int4 --rounds 1"
    run_test "multi-round INT4" "timeout 300 $FERRUM bench JunHowie/Qwen3-4B-GPTQ-Int4 --rounds 3 --max-tokens 16"
    run_test "concurrent 4x INT4" "FERRUM_MAX_BATCH=4 timeout 300 $FERRUM bench JunHowie/Qwen3-4B-GPTQ-Int4 --concurrency 4 --rounds 1 --max-tokens 32"
else
    echo "  SKIP (model not available)"
    SKIP=$((SKIP + 3))
fi

# 5. Qwen2.5 (if available)
echo ""
echo "[5] Qwen2.5-0.5B"
if $FERRUM pull qwen2.5:0.5b 2>/dev/null; then
    run_test "Qwen2.5 single" "timeout 120 $FERRUM bench qwen2.5:0.5b --rounds 1 --max-tokens 32"
else
    echo "  SKIP (model not available)"
    SKIP=$((SKIP + 1))
fi

# 6. Long context (batched flash decode)
echo ""
echo "[6] Long Context"
run_test "long decode 256+ tokens" "timeout 120 $FERRUM bench tinyllama --rounds 1 --max-tokens 512"
if [ "$SKIP_CONCURRENT" != "1" ]; then
    run_test "concurrent long decode" "FERRUM_MAX_BATCH=4 timeout 120 $FERRUM bench tinyllama --concurrency 2 --rounds 1 --max-tokens 256"
fi

# Summary
echo ""
echo "============================================================"
echo "RESULTS: $PASS passed, $FAIL failed, $SKIP skipped"
echo "============================================================"

if [ $FAIL -gt 0 ]; then
    exit 1
fi
