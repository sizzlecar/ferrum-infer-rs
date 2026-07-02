# W2 default-path paged-unified correctness smoke

- lane: W2 default-path paged-unified correctness smoke
- instance: Vast 40826362, 1x RTX 4090, cache-retained CUDA machine
- expected runtime/cost: 20-40 minutes, about USD 0.425/hr while running
- stop condition: start/SSH/CUDA/source sync/build/ferrum run/ferrum serve first failure, or run+serve correctness evidence collected; if correctness is clean, optionally run one c16 diagnostic before stopping
- correctness gate: no remote diagnostic source patch; product `ferrum run` and `ferrum serve` both return expected first-token behavior, with no panic, no NaN/Inf logits, and no unified fallback
- performance command: only after correctness passes, one diagnostic `bench-serve --fail-on-error` c16 run; not release evidence
