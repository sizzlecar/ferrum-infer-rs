# W2 fused sandwich residual-add minimal CUDA validation

- Artifact: `docs/goals/model-coverage-2026-06-12/artifacts/w2_fused_sandwich_residual_2026-06-15/`.
- Commit under test: `4eeea0ba76a2ac8b0671941bcba0d66020c31ed4`.
- Instance: Vast `40826362`, 1x RTX 4090, cache-retained native CUDA machine.
- Scope: minimal CUDA product-path diagnostic for the fused Gemma sandwich branch RMSNorm + F32 residual-add source checkpoint.
- Release status: diagnostic only. This did not run the release-grade validator and did not produce `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`.

Validation results:

- CUDA release build PASS.
- Binary SHA256: `3e28a4cf37b2e25b127dbd591e8891b141863d8082d1757486707c785e6869ce`.
- `ferrum run gemma3:27b-gptq --backend cuda ...` PASS with assistant content `5`, `finish_reason=stop`, `run.rc=0`.
- `ferrum serve` readiness PASS at poll 29.
- `bench-serve --fail-on-error` diagnostic PASS, `bench.rc=0`.
- c=16: 16 completed, 0 errored, output token count source `usage`, throughput `306.061 tok/s`.
- c=32: 16 completed, 0 errored, output token count source `usage`, throughput `307.373 tok/s`.
- Sensitive scan of this artifact found no credential, startup-script, or private-key markers.
- Shutdown: Vast API poll 5 verified `cur_state=stopped`, `actual_status=exited`.

Profile interpretation:

- Compared with `w2_tail_gate_down_profile_2026-06-15`, batch=16 `tail_norm_us_mean` moved from about `806.5us` to `685.2us`, and `tail_resid_us_mean` from about `567.0us` to `494.8us`.
- Batch=16 total decode step moved from about `28.08ms` to `27.82ms`, roughly a `0.9%` diagnostic improvement.
- The largest remaining batch=16 buckets are still `tail_gate_up` at about `9.01ms` and `tail_down` at about `4.70ms`.
- No new product correctness issue was found, but this checkpoint is not enough to change W2 release-grade status.
