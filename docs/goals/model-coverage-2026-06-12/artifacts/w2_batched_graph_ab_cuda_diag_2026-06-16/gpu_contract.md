# W2 batched-graph default-path A/B diagnostic

- lane: W2 batched-graph default-path A/B diagnostic
- instance: Vast 40826362, 1x RTX 4090, cache-retained CUDA machine
- expected runtime/cost: 15-30 minutes, about USD 0.425/hr while running
- stop condition: start/SSH/binary check/ferrum run/ferrum serve first failure, or run+serve correctness evidence plus one c16 diagnostic collected, then stop instance
- correctness gate: no remote source patch; product `ferrum run --batched-graph` and `ferrum serve --batched-graph` both return expected first-token behavior, with no panic/error/NaN log scan findings
- performance command: after correctness passes, one diagnostic `bench-serve --fail-on-error --require-ci` c16 run; not release evidence
- comparison target: previous default-path c16 artifact `w2_paged_unified_default_path_cuda_smoke_2026-06-16` at the same source checkpoint/binary
