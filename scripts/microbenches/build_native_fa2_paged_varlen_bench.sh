#!/usr/bin/env bash
set -euo pipefail
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
NVCC="${NVCC:-$CUDA_HOME/bin/nvcc}"
OUT="${OUT:-/tmp/native_fa2_paged_varlen_bench}"
SRC="${SRC:-scripts/microbenches/native_fa2_paged_varlen_bench.cu}"
KERNEL="${KERNEL:-crates/ferrum-kernels/kernels/fa2_source/ferrum_fa2_paged_varlen.cu}"
"$NVCC" -std=c++17 -O3 --use_fast_math --expt-relaxed-constexpr --expt-extended-lambda \
  -arch=sm_${CUDA_COMPUTE_CAP:-89} "$SRC" "$KERNEL" -o "$OUT"
echo "$OUT"
