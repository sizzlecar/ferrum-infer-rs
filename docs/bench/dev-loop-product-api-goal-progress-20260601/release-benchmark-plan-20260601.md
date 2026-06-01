# Release Benchmark Plan - 2026-06-01

Purpose: produce saved benchmark artifacts for the formal release announcement.

Common benchmark shape:

- dataset: `random`
- input/output: `256/128`
- concurrency cells: `1,4,16,32` where the model fits
- repeats: `N>=3` for release tables; prefer `N=5` for headline rows
- save all raw logs, server commands, model revisions, env, and summary JSON
  under `/workspace/release-bench-20260601-*` and mirror summary docs under
  `docs/bench/dev-loop-product-api-goal-progress-20260601/`.

## 1. M3 Qwen3-30B-A3B GPTQ Int4

Goal: apples-to-apples Ferrum vs vLLM confirmation for the M3 release gate.

Ferrum:

- model: `Qwen/Qwen3-30B-A3B-GPTQ-Int4`
- path: cached snapshot under `/workspace/hf-cache/...`
- candidate env: `FERRUM_FA_LAYOUT_VARLEN=1,FERRUM_FA2_SOURCE=1`
- current release evidence: `/workspace/m3-fa2-source-current-allcells-n3-20260601`

vLLM:

- vLLM 0.20.2 same-pod baseline
- quantization: `gptq_marlin`
- same random `256/128` workload

Current release ratios from saved evidence:

| c | Ferrum source FA2 | vLLM | ratio |
|---:|---:|---:|---:|
| 1 | `157.18` | `183.9` | `0.855x` |
| 4 | `448.36` | `512.5` | `0.875x` |
| 16 | `1115.58` | `1331.9` | `0.838x` |
| 32 | `1488.08` | `1972.9` | `0.754x` |

## 2. Qwen3-8B GGUF

Goal: Ferrum GGUF vs vLLM GGUF saved comparison.

Ferrum model:

- alias: `qwen3:8b-q4_k_m`
- repo/file: `Qwen/Qwen3-8B-GGUF` / `Qwen3-8B-Q4_K_M.gguf`
- CUDA serve smoke: `/workspace/release-qwen3-8b-gguf-cuda-smoke-42ffbe2`
  passed health + OpenAI chat on RTX 4090.
- Implementation caveat: commit `42ffbe2` uses the CUDA GGUF eager-dequant
  fallback path. This is compatibility evidence, not a native CUDA k-quant
  performance claim.

vLLM model:

- vLLM GGUF support is experimental and under-optimized.
- Use single-file GGUF only.
- Preferred command shape:

```bash
vllm serve Qwen/Qwen3-8B-GGUF:Q4_K_M --tokenizer Qwen/Qwen3-8B
```

If repo-id quant loading fails, download the GGUF file and run:

```bash
vllm serve ./Qwen3-8B-Q4_K_M.gguf --tokenizer Qwen/Qwen3-8B
```

## 3. LLaMA-3.1-8B GGUF

Goal: Ferrum GGUF vs vLLM GGUF saved comparison.

Ferrum model:

- alias: `llama3.1:8b-q4_k_m`
- repo/file: `bartowski/Meta-Llama-3.1-8B-Instruct-GGUF` /
  `Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf`
- CUDA serve smoke: `/workspace/release-llama31-8b-gguf-cuda-smoke-42ffbe2`
  passed health + OpenAI chat on RTX 4090.
- Implementation caveat: commit `42ffbe2` uses the CUDA GGUF eager-dequant
  fallback path. This is compatibility evidence, not a native CUDA k-quant
  performance claim.

vLLM model:

- Use single-file GGUF only.
- Preferred command shape:

```bash
vllm serve bartowski/Meta-Llama-3.1-8B-Instruct-GGUF:Q4_K_M \
  --tokenizer meta-llama/Meta-Llama-3.1-8B-Instruct
```

If repo-id quant loading fails, download the GGUF file and run:

```bash
vllm serve ./Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf \
  --tokenizer meta-llama/Meta-Llama-3.1-8B-Instruct
```

## Release note caveat

vLLM GGUF support is experimental and may be under-optimized. Release tables
must label GGUF comparisons as GGUF-vs-GGUF and keep them separate from the M3
GPTQ-Marlin table.

## Current benchmark blocker

The Ferrum side of the 8B GGUF comparisons is now smoke-unblocked. The saved
Ferrum/vLLM benchmark tables are still pending a runnable GPU instance:

- Vast instance `38872161` could not be restarted after stopping; repeated
  `state=running` attempts returned `resources_unavailable`.
- `38872161` was destroyed per the recovery rule.
- Replacement RTX 4090 creation attempt on offer `32736582` failed with
  `insufficient_credit`.
