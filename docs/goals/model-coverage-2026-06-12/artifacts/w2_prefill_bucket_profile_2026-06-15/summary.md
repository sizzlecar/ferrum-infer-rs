# W2 Prefill Bucket Profile Diagnostic

- Artifact: `docs/goals/model-coverage-2026-06-12/artifacts/w2_prefill_bucket_profile_2026-06-15/`
- Lane: W2 Gemma3 CUDA prefill profile buckets validation.
- Instance: Vast `40826362`, 1x RTX 4090, reused cache-retained native CUDA machine.
- Source commit: `3c407faf25eed833fbb785057c6a7f39d0578e5b`.
- Binary SHA256: `5873e674ed0aff9a301af532e0f38c898595d02fd12441125240cf24abea9403`.
- Cleanup: instance stopped; `vast_shutdown/poll_3.json` reached `actual_status=exited`.

## Gate Result

- `cargo_build.rc=0`.
- `run.status=PASS`.
- `bench-serve.rc=0`.
- Chat smoke content: `5`; usage `completion_tokens=3`.
- c16 ShareGPT diagnostic: `16 completed / 0 errored`, bad output `[0]`.

## Bench Diagnostic

This run used `FERRUM_PREFILL_OP_PROFILE=1`, so the throughput number includes profiler overhead and is not release evidence.

- Output throughput: `321.551 tok/s`.
- Request throughput: `5.024 req/s`.
- TTFT p50/p95: `925.570ms` / `1516.331ms`.
- TPOT p50: `35.035ms`.
- ITL p50/p99: `26.578ms` / `295.802ms`.
- `output_token_count_source=usage`.

## Prefill Buckets

For the 26 ShareGPT prefills with `tokens=122`:

- Total prefill mean: `83.577ms` (`83-92ms` range).
- `tail_mlp`: `37.654ms`, about `45.1%` of total.
- `flash_attn`: `30.192ms`, about `36.1%` of total.
- `matmuls`: `6.000ms`, about `7.2%` of total.
- `qk_norm_rope`: `1.000ms`, about `1.2%` of total.
- Inside `tail_mlp`, `tail_gate_up` averaged `23.115ms` and `tail_down` averaged `13.115ms`.

## Interpretation

The profiler fix is effective: prefill bucket data is now non-empty and stable across ShareGPT requests. The prefill/TTFT path is dominated by MLP tail and flash attention, not ordinary QKV/O matmul timing. That narrows the next source lever to Gemma GPTQ MLP tail implementation and prefill attention behavior; typed vLLM paged attention was already tested and did not improve the end-to-end ShareGPT cells.

Remote `git status --short` is not clean because the sync intentionally omitted historical docs artifact files to avoid copying large unrelated evidence. The build-relevant source tree was synchronized to commit `3c407faf`; this run remains diagnostic-only and is not a formal performance claim.

## Release-Grade Status

No `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>` was produced. This is not release-grade evidence.
