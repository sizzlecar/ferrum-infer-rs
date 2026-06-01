# GGUF 8B Release Benchmarks - 2026-06-01

Purpose: saved Ferrum/vLLM GGUF-vs-GGUF comparison artifacts for release preparation.

Artifact mirror:

- `docs/bench/dev-loop-product-api-goal-progress-20260601/gguf-8b-remote-artifacts/`

Caveats:

- These are GGUF-vs-GGUF comparisons, not GPTQ/Marlin comparisons.
- vLLM GGUF support is experimental and under-optimized.
- Ferrum CUDA GGUF currently uses the eager-dequant/fp16 dense fallback path, not native CUDA k-quant kernels.
- Qwen3-8B vLLM required the base tokenizer/config path to avoid GGUF tokenizer-cache collisions.
- LLaMA-3.1-8B vLLM required `--hf-config-path NousResearch/Meta-Llama-3.1-8B-Instruct` and `--max-model-len 4096` to fit on RTX 4090 24GB.

## Qwen3-8B Q4_K_M GGUF

Artifacts:

- Ferrum/vLLM reports: `gguf-8b-remote-artifacts/release-bench-20260601-gguf-8b-a075e64/qwen3-8b-q4_k_m/`
- vLLM config-fix log: `gguf-8b-remote-artifacts/logs/qwen3-8b-vllm-only-20260601-hfconfig.log`

| c | Ferrum tok/s | vLLM tok/s | Ferrum/vLLM |
|---:|---:|---:|---:|
| 1 | `54.5 ± 0.5` | `114.3 ± 0.3` | `0.477x` |
| 4 | `177.6 ± 3.7` | `241.5 ± 2.0` | `0.735x` |
| 16 | `442.6 ± 22.8` | `315.9 ± 0.6` | `1.40x` |
| 32 | `568.9 ± 98.4` | `332.4 ± 1.1` | `1.71x` |

## LLaMA-3.1-8B Q4_K_M GGUF

Artifacts:

- Ferrum/vLLM reports: `gguf-8b-remote-artifacts/release-bench-20260601-gguf-8b-d7d3ebb-llama/llama31-8b-q4_k_m/`
- max-model-len/config log: `gguf-8b-remote-artifacts/logs/llama31-8b-vllm-only-20260601-maxlen4096.log`

| c | Ferrum tok/s | vLLM tok/s | Ferrum/vLLM |
|---:|---:|---:|---:|
| 1 | `55.4 ± 1.3` | `117.6 ± 0.1` | `0.471x` |
| 4 | `188.6 ± 1.9` | `240.0 ± 0.9` | `0.786x` |
| 16 | `486.4 ± 60.2` | `313.2 ± 7.7` | `1.55x` |
| 32 | `677.0 ± 52.5` | `323.4 ± 28.4` | `2.09x` |

Release-note interpretation:

- The 8B GGUF rows should be used as compatibility/benchmark evidence with the caveats above.
- Do not present c=1 as a win; Ferrum is behind vLLM at c=1 for both models.
- The high-concurrency rows show Ferrum outperforming vLLM's current GGUF path under this workload.
