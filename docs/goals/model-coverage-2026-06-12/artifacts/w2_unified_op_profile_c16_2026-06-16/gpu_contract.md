W2 Gemma3 CUDA unified op profile minimal diagnostic

- Lane: W2 Gemma3 CUDA unified-op profile minimal diagnostic on existing Vast 1x RTX 4090 instance `41187356`.
- Expected runtime/cost: about 20-40 minutes on the cached instance at about `$0.38/hour`.
- Stop condition: collect `[unified-op-profile]` lines for smoke plus c16 bench, or stop at the first build/startup/smoke/bench failure after copying logs.
- Correctness gate: `ferrum serve` streaming smoke must pass before treating c16 profile output as useful.
- Performance command: diagnostic-only `ferrum bench-serve ... --fail-on-error --seed 9271 --n-repeats 1` at concurrency 16. This is not release performance evidence.
- Cleanup: stop the Vast instance after artifacts are copied back unless more targeted work is immediately required.
