# W2 full-logits unified prefill CUDA diagnostic

Status: diagnostic pass, not release-grade evidence.

Artifact: `docs/goals/model-coverage-2026-06-12/artifacts/w2_full_logits_unified_prefill_cuda_diag_2026-06-16/`

## Scope

- Lane: W2 full-logits unified-prefill CUDA diagnostic.
- Source checkpoint: `40186c75e393ef58e81b9f5acfe529186505a0bc`.
- Vast instance: `40826362`, 1x RTX 4090.
- Correctness checks: CUDA build rc `0`, server ready, deterministic chat smoke pass, c16 bench rc `0`.
- Performance shape: c16, 16 requests, `n_repeats=1`, diagnostic only.
- Shutdown: Vast final state `cur_state=stopped`, `actual_status=exited`.

## Key result

The source change was correct locally, but it did not remove the product-path
serial prefill behavior on CUDA.

- `prefill-profile tokens=122` lines: `26`.
- First c16 prefill cohort still shows ten serial 122-token prefill profiles
  before the engine's unified-post line.
- Engine line: `iter#3 items=10 prefill=10 total=927027us model=924576us`.
- Previous comparable diagnostic line was `iter#3 items=10 prefill=10 total=946123us model=943620us`.

This means the previous full-logits guard was not the only reason
`model.unified_forward` is not being used for the c16 prefill cohort. The next
source step must expose `LlmExecutor::unified_decode` fallback reason for the
mixed/unified API path, not only for `batch_prefill`.

## Diagnostic numbers

- Chat smoke: pass, content `5`.
- Bench: 16 completed, 0 errored.
- Output token count source: `usage`.
- Output throughput: `298.24957600538823` tok/s.
- Orientation-only vLLM c16 baseline: `518.7959572662905` tok/s.
- Orientation-only Ferrum/vLLM ratio: `0.5748880110341743`.
- Previous diagnostic throughput: `284.90049780836483` tok/s.
- Throughput delta vs previous diagnostic: about `+4.7%`.

## Interpretation

The full-logits unified-prefill source change is not enough to reach W2-P2. It
may have produced a small throughput improvement, but the main TTFT/prefill
wall time remains. The immediate next lever is observability in
`LlmExecutor::unified_decode`: record whether it skipped unified forward due to
full-logits capability, received `Unsupported`, or fell back for another
reason. Only then should another CUDA run be started.

## Release-grade status

No `model_release_grade_manifest.json` was produced, and no
`MODEL_RELEASE_GRADE_W2 PASS: <out_dir>` line exists. W2 remains not
release-grade.
