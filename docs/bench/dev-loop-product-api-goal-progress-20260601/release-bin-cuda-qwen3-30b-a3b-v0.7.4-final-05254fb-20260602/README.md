# Final CUDA release-bin regression: v0.7.4 / Qwen3-30B-A3B GPTQ Int4

Date: 2026-06-02  
Tag: `v0.7.4`  
Commit: `05254fb`  
Binary: public GitHub Release asset `ferrum-linux-x86_64-cuda-sm89.tar.gz`  
Version: `ferrum 0.7.4`  
Hardware: Vast RTX 4090, NVIDIA driver `580.65.06`

This run was added after the final tokenizer/model-path CI fix, because code changed after the previous CUDA release-bin smoke. It is a quick release regression gate, not the full same-pod N=5 vLLM comparison.

## Command shape

```bash
BIN=/workspace/ferrum-final-regression/ferrum \
MODEL_DIR=/workspace/hf-cache/hub/models--Qwen--Qwen3-30B-A3B-GPTQ-Int4/snapshots/9b534e4318b7ebc3c961a839f13eb18b1833f441 \
HF_MODEL=Qwen/Qwen3-30B-A3B-GPTQ-Int4 \
FA2_SOURCE=1 \
FA2_EXTRA_LD_LIBRARY_PATH="" \
BUILD=0 \
CONCURRENCY=32 \
NUM_PROMPTS=16 \
WARMUP_REQUESTS=4 \
REPEATS=1 \
bash scripts/m3_fa2_direct_ffi_ab.sh
```

## Correctness gates

| Case | Paris | Multi-turn | Three-round multi-turn | Bench completion |
|---|---:|---:|---:|---:|
| `fa2_source` | pass | pass | pass | 16/16, 0 errors |
| `fa_layout` | pass | pass | pass | 16/16, 0 errors |

Observed gate contents:

| Case | Paris response | Multi-turn response | Three-round response |
|---|---|---|---|
| `fa2_source` | `The capital of France is **Paris**.` | `Paris` | `Paris` |
| `fa_layout` | `The capital of France is **Paris**.` | `Paris` | `Paris` |

## Performance smoke

| Case | Concurrency | Throughput tok/s | TTFT p50 ms | TPOT p50 ms | ITL p95 ms | Status |
|---|---:|---:|---:|---:|---:|---|
| `fa2_source` | 32 | 916.1 | 407.4 | 14.07 | 17.86 | pass |
| `fa_layout` | 32 | 724.7 | 580.7 | 17.09 | 19.28 | pass |

`fa2_source` throughput was `+26.40%` versus `fa_layout` on this quick smoke run. This is release-regression evidence only; headline M3 performance remains the N=5 comparison in `docs/bench/cuda-rtx4090-2026-05-30-m3-80pct-confirmed/`.

## Artifact integrity

```text
a3432ce7a4602f65cb88062edbac2526bcca383eddf4476d0e276c84a75f2874  /workspace/ferrum-final-regression/ferrum-linux-x86_64-cuda-sm89.tar.gz
```
