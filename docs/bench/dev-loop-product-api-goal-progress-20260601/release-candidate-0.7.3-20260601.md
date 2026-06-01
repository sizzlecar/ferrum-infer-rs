# Ferrum 0.7.3 Release Candidate - 2026-06-01

Scope: formal release candidate for `docs/dev-loop-product-api-goal-2026-05-30.md`.

Current release-candidate code checkpoint:

- Current evidence checkpoint before the pre-tag checklist:
  `2ecd2bb docs: add metal qwen3 0.6b server smoke`
- Last runtime-affecting checkpoint: `1e3ce42 fix: sync qwen3 moe prefill logits before readback`
- Packaging checkpoint in current branch: dedicated CUDA GitHub Release
  workflow plus Homebrew `ferrum-cuda` tap generation, pending final tag-run
  proof.
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
| 8B GGUF Ferrum/vLLM comparison | pass with caveats | `gguf-8b-release-benchmarks-20260601.md`, Qwen3-8B and LLaMA-3.1-8B GGUF-vs-GGUF tables saved |
| Metal Qwen3-MoE prefill readback | pass smoke | `metal-qwen3-30b-a3b-prefill-syncfix-20260601.md`, Paris output is sane and encoder assertion no longer reproduces |
| Metal Qwen3-0.6B smoke | pass | `metal-qwen3-06b-smoke-20260601`, Paris correctness passed; 64-token decode median `43.0 tok/s`; OpenAI server multi-turn returned `basalt`; concurrent c4/r8 chat completed `8/8` with aggregate `54.614 tok/s`; local swap active |
| Metal Qwen3-8B GGUF smoke | pass | `metal-qwen3-8b-q4km-smoke-20260601`, Paris correctness passed; 64-token decode median `23.5 tok/s` with local swap active |
| Post-fix GPU quick regression | pass | `m3-quick-regress-1e3ce42-c32-20260601.md`, c32 source FA2 `1403.98 tok/s`, correctness/multi-turn passed |
| q2 native FA2 experiment | rejected safely | microbench-positive but full-model c32 regressed; reverted by `2197077` |

## Release packaging status

- GitHub Release CPU and Metal artifacts continue to be produced by
  `release.yml`.
- GitHub Release CUDA artifact is now produced by `release-cuda.yml` as
  `ferrum-linux-x86_64-cuda-sm89.tar.gz`, built with
  `cuda,vllm-moe-marlin,vllm-paged-attn-v2,fa2-source`.
- The CUDA workflow records `.ldd.txt` and fails if the release binary links
  Torch, Python, or vLLM.
- Homebrew tap update now writes both `Formula/ferrum.rb` and
  `Formula/ferrum-cuda.rb`.
- `brew install ferrum` remains normal CPU/Metal. `brew install ferrum-cuda`
  is Linux x86_64 CUDA and downloads the CUDA tarball.
- This packaging path is wired but not yet final evidence; the proof point is
  the final tag workflow run plus tap push.

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
- Qwen3-MoE Metal prefill now synchronizes before reading logits back to host.

## Remaining release decisions

These do not invalidate the `0.7.3` release candidate, but they must be explicit in the final release note:

- Source FA2 is release-supported and benchmarked as an opt-in path. If the final release requires it to be the default selector for M3, that policy change still needs an explicit code/default decision.
- The ignored SDK cargo wrapper smoke remains blocked by a debug CUDA build-script path. The direct release-binary real-model API smoke is the accepted release evidence for this RC.
- The final tag workflow must still publish CPU, Metal, and CUDA release assets,
  and the Homebrew tap must be updated with both `ferrum` and `ferrum-cuda`.
- Saved 8B Ferrum/vLLM GGUF comparison tables are complete but must be labelled GGUF-vs-GGUF and caveated: Ferrum uses eager-dequant/fp16 dense CUDA fallback; vLLM GGUF is experimental.
- The Metal correctness fix is validated by smoke only. The local Mac had active swap, so no clean Metal performance claim should be made from this run.
- The Metal Qwen3-0.6B and Qwen3-8B smoke results are useful regression
  evidence, but local swap was active, so neither should be used as a clean
  headline benchmark.
- The Metal Qwen3-0.6B smoke covers single-turn correctness, CLI decode
  throughput, OpenAI-compatible multi-turn recall, and concurrent chat
  completions. Treat the throughput numbers as local regression evidence only.

## Latest post-code-change regression

Commit `1e3ce42` changed runtime readback behavior for Qwen3-MoE prefill. A fast
RTX 4090 regression was run instead of a full all-cell rerun:

| Case | c | prompts | throughput tok/s | correctness | perf gate |
|---|---:|---:|---:|---|---|
| `fa2_source` | 32 | 64 | `1403.98` | Paris + multi-turn + 3-round recall pass | pass |
| `fa_layout` | 32 | 64 | `1230.54` | Paris + multi-turn + 3-round recall pass | control |

Artifact:

- Remote: `/workspace/m3-quick-regress-1e3ce42-c32-20260601`
- Local mirror: `docs/bench/dev-loop-product-api-goal-progress-20260601/m3-quick-regress-1e3ce42-c32-20260601/`

## 8B GGUF saved benchmark tables

| Model | c1 | c4 | c16 | c32 |
|---|---:|---:|---:|---:|
| Qwen3-8B GGUF Ferrum/vLLM | `0.477x` | `0.735x` | `1.40x` | `1.71x` |
| LLaMA-3.1-8B GGUF Ferrum/vLLM | `0.471x` | `0.786x` | `1.55x` | `2.09x` |

Full tables and raw report mirrors are in
`docs/bench/dev-loop-product-api-goal-progress-20260601/gguf-8b-release-benchmarks-20260601.md`.

## Release-note wording constraints

- M3 table may claim Ferrum reaches the formal `0.75x vLLM` release gate on RTX 4090 for Qwen3-30B-A3B GPTQ Int4.
- Do not claim the old `0.80x` stretch goal is complete.
- 8B GGUF performance comparison is measured, but it is GGUF-vs-GGUF and should not be used as a native CUDA k-quant claim.
- Any GGUF-vs-vLLM table must be labeled `GGUF-vs-GGUF`; vLLM GGUF is experimental/under-optimized per vLLM docs.
- CUDA GGUF in Ferrum is currently eager-dequant fallback, not native CUDA k-quant.
