# CUDA release-bin regression: Qwen3-30B-A3B GPTQ Int4

Date: 2026-06-02  
Commit: `9012ece`  
Binary: GitHub Actions CUDA dry-run artifact `ferrum-linux-x86_64-cuda-sm89.tar.gz` from run `26772092582`  
Version: `ferrum 0.7.4`  
Hardware: Vast RTX 4090, NVIDIA driver `580.82.07`, CUDA runtime reported by driver `13.0`

This is a release-binary smoke/regression gate, not a replacement for the full same-pod N=5 vLLM comparison in `docs/bench/cuda-rtx4090-2026-05-30-m3-80pct-confirmed/`.

## Command shape

The test used the packaged release binary directly:

```bash
BIN=/workspace/ferrum-release-test/bin/ferrum \
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

The runner compared `fa2_source` against the `fa_layout` baseline using the same release binary.

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
| `fa2_source` | 32 | 938.0 | 381.4 | 13.89 | 16.54 | pass |
| `fa_layout` | 32 | 960.8 | 408.3 | 13.55 | 15.95 | pass |

`fa2_source` throughput was `-2.37%` versus `fa_layout`, within the runner's `-3%` smoke threshold. This is acceptable as a release-binary regression gate on this host, but it is not used as headline performance evidence.

## Artifact integrity

Release tarball sha256 on the GPU host:

```text
73ccd2f4d70fba7951ab244d172b0d56774a3034c29e90730faf240a4ae398b3  /workspace/ferrum-release-test/ferrum-linux-x86_64-cuda-sm89.tar.gz
```

`ldd-runtime.txt` confirms CUDA runtime libraries resolved on the GPU host, including `libcuda.so.1`, `libcudart.so.12`, `libcurand.so.10`, `libcublas.so.12`, and `libcublasLt.so.12`.
