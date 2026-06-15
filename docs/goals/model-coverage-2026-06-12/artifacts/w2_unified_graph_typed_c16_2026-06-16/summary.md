# W2 typed unified graph c16 diagnostic

Status: FAIL, diagnostic only. No release-grade PASS line.

- Artifact: `docs/goals/model-coverage-2026-06-12/artifacts/w2_unified_graph_typed_c16_2026-06-16`
- Git: `7f15a3ef9a57e2c23d889975ab629d25e8638803`, source status clean for `crates/`, `scripts/`, `Cargo.toml`, `Cargo.lock`, and `ferrum.toml`
- Binary SHA256: `05f18a4cd8d8f34530758584122afad9e12f0bb929b450fc283449bb7d3180bd`
- Runtime: `ferrum run --unified-graph` and `ferrum serve --unified-graph`; decision trace selected `decode_graph_policy=unified_decode_graph` from CLI `FERRUM_UNIFIED_GRAPH`
- Correctness smoke:
  - `RUN_SMOKE_PASS content='5' tokens=3`
  - `SERVE_SMOKE_PASS content='5' completion_tokens=3`
- Bench command: c16 only, ShareGPT ASCII, `--fail-on-error --require-ci --seed 9271 --n-repeats 3`
- Bench progress before stop:
  - repeat 1/3: 16 completed, 0 errored, 3.1s
  - repeat 2/3: 16 completed, 0 errored, 3.1s
  - repeat 3/3 started
- Failure:
  - `[unified-graph] replay err: Unsupported operation: post-launch sync: CUDA_ERROR_ILLEGAL_ADDRESS`
  - `CudaBackend: load_function(rms_norm_f32_to_f16): DriverError(CUDA_ERROR_ILLEGAL_ADDRESS, "an illegal memory access was encountered")`
- Vast cleanup: instance `40826362` stopped, final `cur_state=stopped actual_status=exited`.

Interpretation: typed unified graph is not usable for W2 product perf yet. It passes one-shot run/serve smoke but fails under c16 bench, so it is a correctness blocker and cannot be used for performance evidence.
