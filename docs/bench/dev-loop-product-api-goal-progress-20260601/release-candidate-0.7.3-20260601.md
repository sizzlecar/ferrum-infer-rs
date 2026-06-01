# Ferrum 0.7.3 Release Candidate - 2026-06-01

Scope: formal release candidate for `docs/dev-loop-product-api-goal-2026-05-30.md`.

Current release-candidate code checkpoint:

- `be0596c bench: add release gguf 8b vllm wrapper`
- Workspace package version: `0.7.3`

## Release gates that are satisfied

| Gate | Status | Evidence |
|---|---|---|
| CUDA release iteration loop | pass | `/workspace/m3-release-touch-probe-thinlto-20260601-20260601_064127`, p50 `33.164s`, p95 `34.454s`, all `39` CUDA rows cache-hit |
| Native source FA2 build boundary | pass | `scripts/check_fa2_source_native.py` and `scripts/check_runtime_snapshot_boundary.py` passed in `docs/bench/dev-loop-product-api-goal-progress-20260601/local-static-boundary-guards-20260601.md` |
| M3 release performance gate | pass | `/workspace/m3-fa2-source-current-allcells-n3-20260601`, c1/c4/c16/c32 all above the user-adjusted `0.75x vLLM` release threshold |
| M3 correctness gates | pass | same all-cell packet passed Paris, multi-turn, and three-turn recall gates |
| Real-model OpenAI-compatible API smoke | pass | `/workspace/m3-real-model-api-direct-smoke-20260601`, health/chat/usage/streaming/json_object/three-turn recall all passed |
| `qwen3:0.6b` alias serve | pass | `/workspace/release-alias-serve-qwen3-06b-8ec0858`, no manual `FERRUM_MODEL_PATH`, health/chat passed |
| Qwen3-8B GGUF CUDA serve | pass smoke | `/workspace/release-qwen3-8b-gguf-cuda-smoke-42ffbe2`, health/chat passed |
| LLaMA-3.1-8B GGUF CUDA serve | pass smoke | `/workspace/release-llama31-8b-gguf-cuda-smoke-42ffbe2`, health/chat passed |
| q2 native FA2 experiment | rejected safely | microbench-positive but full-model c32 regressed; reverted by `2197077` |

## M3 release performance table

Release threshold: `0.75x vLLM`.

| c | Ferrum source FA2 tok/s | vLLM tok/s | ratio | release gate |
|---:|---:|---:|---:|---|
| 1 | `157.18` | `183.9` | `0.855x` | pass |
| 4 | `448.36` | `512.5` | `0.875x` | pass |
| 16 | `1115.58` | `1331.9` | `0.838x` | pass |
| 32 | `1488.08` | `1972.9` | `0.754x` | pass |

The old `0.80x` goal remains a stretch target; it is not satisfied at c32.

## Product compatibility changes included in this RC

- Native in-repo FA2 source path restored; product build no longer requires external FlashAttention/CUTLASS source checkout for the tested path.
- `serve qwen3:0.6b` now carries typed `model_path` into tokenizer/model factories instead of relying on process `FERRUM_MODEL_PATH`.
- GGUF pull now fetches tokenizer sidecars without pulling full sibling safetensors repositories.
- `llama3.1:8b-q4_k_m` uses a public tokenizer source for sidecars.
- CUDA GGUF serve path is enabled through an eager-dequant/fp16 dense fallback. This is a compatibility path, not a native CUDA k-quant performance path.
- `scripts/release_gguf_8b_vs_vllm.sh` is ready for saved Qwen3-8B and LLaMA-3.1-8B Ferrum/vLLM GGUF benchmark packets.

## Remaining release decisions

These do not invalidate the `0.7.3` release candidate, but they must be explicit in the final release note:

- Source FA2 is release-supported and benchmarked as an opt-in path. If the final release requires it to be the default selector for M3, that policy change still needs an explicit code/default decision.
- The ignored SDK cargo wrapper smoke remains blocked by a debug CUDA build-script path. The direct release-binary real-model API smoke is the accepted release evidence for this RC.
- Saved 8B Ferrum/vLLM GGUF comparison tables are pending GPU availability. Ferrum-side serve smoke is complete; vLLM comparison execution is blocked by Vast credit.

## Current external blocker

There is currently no runnable Vast GPU instance.

- Instance `38872161` could not be restarted after stop; repeated start attempts returned `resources_unavailable`.
- `38872161` was destroyed.
- Replacement RTX 4090 creation attempts failed with `insufficient_credit`, including fresh offer `38712898`.

When credit is restored, run:

```bash
source /workspace/vllm-venv/bin/activate
cargo build --release -p ferrum-cli \
  --features cuda,marlin,vllm-paged-attn-v2,vllm-moe-marlin,fa2-source
OUT_ROOT=/workspace/release-bench-20260601-gguf-8b \
  bash scripts/release_gguf_8b_vs_vllm.sh
```

## Release-note wording constraints

- M3 table may claim Ferrum reaches the formal `0.75x vLLM` release gate on RTX 4090 for Qwen3-30B-A3B GPTQ Int4.
- Do not claim the old `0.80x` stretch goal is complete.
- 8B GGUF support may be described as smoke-validated on CUDA, but performance comparison vs vLLM is not yet measured.
- Any GGUF-vs-vLLM table must be labeled `GGUF-vs-GGUF`; vLLM GGUF is experimental/under-optimized per vLLM docs.
- CUDA GGUF in Ferrum is currently eager-dequant fallback, not native CUDA k-quant.
