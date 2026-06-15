# W2 unified_decode fallback_reason diagnosis

- artifact: `docs/goals/model-coverage-2026-06-12/artifacts/w2_unified_decode_fallback_reason_cuda_diag_2026-06-16`
- source commit: `5e9ae1514247e1ea7c34459dae3e2d1198c5e77d`
- build/chat/bench: PASS (`build.rc=0`, `run_profile.rc=0`, `bench.rc=0`)
- correctness: chat smoke returned `5`, completed 16/16 requests, 0 errors, usage token counts
- throughput: `315.39451845233344 tok/s`, orientation ratio vs vLLM c16 `0.6079355747378077`
- key finding: all observed `[unified-decode]` lines fall back with `fallback_reason=paged_kv_required`
- first c16 prefill cohort: `[unified-decode] call#3 items=10 prefill=10 decode=0 total_q=1220 attempted_unified=true fallback=true fallback_reason=paged_kv_required elapsed=890406us`
- release-grade: no `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`; diagnostic only
