# Release Benchmark Plan - 2026-06-01

Purpose: produce saved benchmark artifacts for the formal release announcement.

Status update: the 8B GGUF comparison tables have now been measured and mirrored
locally. This file remains as the runbook plus result index.

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

Status: measured.

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

Measured table:

| c | Ferrum tok/s | vLLM tok/s | Ferrum/vLLM |
|---:|---:|---:|---:|
| 1 | `54.5 ± 0.5` | `114.3 ± 0.3` | `0.477x` |
| 4 | `177.6 ± 3.7` | `241.5 ± 2.0` | `0.735x` |
| 16 | `442.6 ± 22.8` | `315.9 ± 0.6` | `1.40x` |
| 32 | `568.9 ± 98.4` | `332.4 ± 1.1` | `1.71x` |

## 3. LLaMA-3.1-8B GGUF

Goal: Ferrum GGUF vs vLLM GGUF saved comparison.

Status: measured.

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

Measured table:

| c | Ferrum tok/s | vLLM tok/s | Ferrum/vLLM |
|---:|---:|---:|---:|
| 1 | `55.4 ± 1.3` | `117.6 ± 0.1` | `0.471x` |
| 4 | `188.6 ± 1.9` | `240.0 ± 0.9` | `0.786x` |
| 16 | `486.4 ± 60.2` | `313.2 ± 7.7` | `1.55x` |
| 32 | `677.0 ± 52.5` | `323.4 ± 28.4` | `2.09x` |

## Release note caveat

vLLM GGUF support is experimental and may be under-optimized. Release tables
must label GGUF comparisons as GGUF-vs-GGUF and keep them separate from the M3
GPTQ-Marlin table.

## Reproduction command

The release GGUF benchmark wrapper is now:

```bash
source /workspace/vllm-venv/bin/activate
cargo build --release -p ferrum-cli \
  --features cuda,marlin,vllm-paged-attn-v2,vllm-moe-marlin,fa2-source
OUT_ROOT=/workspace/release-bench-20260601-gguf-8b \
  bash scripts/release_gguf_8b_vs_vllm.sh
```

Defaults:

- models: `qwen3:8b-q4_k_m` and `llama3.1:8b-q4_k_m`
- vLLM models: `Qwen/Qwen3-8B-GGUF:Q4_K_M` and
  `bartowski/Meta-Llama-3.1-8B-Instruct-GGUF:Q4_K_M`
- tokenizer models: `Qwen/Qwen3-8B` and
  `NousResearch/Meta-Llama-3.1-8B-Instruct`
- workload: random `256/128`, c=`1,4,16,32`, `N=3`,
  `128` prompts, `10` warmup requests.
