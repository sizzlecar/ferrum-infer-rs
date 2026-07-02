# W2 batch prefill fallback reason diagnostic

Status: diagnostic pass, not release-grade evidence.

Artifact: `docs/goals/model-coverage-2026-06-12/artifacts/w2_batch_prefill_fallback_reason_diag_2026-06-16/`

## Scope

- Lane: W2 batch-prefill fallback-reason diagnostic.
- Vast instance: `40826362`, 1x RTX 4090.
- Correctness checks: CUDA build rc `0`, server ready, deterministic chat smoke pass, c16 bench rc `0`.
- Performance shape: c16, 16 requests, `n_repeats=1`, diagnostic only.
- Shutdown: Vast final state `cur_state=stopped`, `actual_status=exited`.

## Key observation

- `FERRUM_BATCH_PREFILL_PROF` was active from typed config.
- No `[batch-prefill]` or `fallback_reason=` lines were emitted by
  `LlmExecutor::batch_prefill`.
- Continuous unified execution did batch prefill:
  - `iter#0 items=1 prefill=1 total=147324us`
  - `iter#3 items=10 prefill=10 total=946123us`

This rejects the current "LlmExecutor batch_prefill serial fallback is the W2 c16
TTFT bottleneck" hypothesis for this product path. The c16 path is already using
the continuous engine unified prefill route; the heavy TTFT is inside unified
prefill itself.

## Diagnostic numbers

- Chat smoke: pass, content `5`.
- Bench: 16 completed, 0 errored.
- Output token count source: `usage`.
- Output throughput: `284.90049780836483` tok/s.
- Orientation-only vLLM c16 baseline: `518.7959572662905` tok/s.
- Orientation-only Ferrum/vLLM ratio: `0.549157127803387`.
- Prefill profile lines: `297`.
- Batched op profile lines: `128`.
- Latest dense Marlin decode tail sample:
  - gate-up kernel: `8384us / 62 calls`
  - down kernel: `4354us / 62 calls`
  - gate-up workspace zero: `379us / 62 calls`
  - down workspace zero: `329us / 62 calls`

## Next lever

Do not spend more time on `LlmExecutor::batch_prefill` fallback for this c16
path unless a new artifact proves that path is called. The next high-signal
work is to profile and reduce the unified prefill wall time directly, with
attention to the dense GPTQ Marlin MLP kernels and prefill attention cost.

## Release-grade status

No `model_release_grade_manifest.json` was produced, and no
`MODEL_RELEASE_GRADE_W2 PASS: <out_dir>` line exists. W2 remains not
release-grade.
