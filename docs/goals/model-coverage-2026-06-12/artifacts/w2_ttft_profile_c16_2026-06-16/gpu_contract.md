W2 Gemma3 CUDA TTFT profile minimal diagnostic

- Lane: W2 Gemma3 CUDA TTFT profile minimal diagnostic on existing Vast 1x RTX 4090 instance `41187356`.
- Expected runtime/cost: 10-20 minutes, hard stop at 30 minutes; instance rate recorded as about USD 0.38/hour.
- Stop condition: start/SSH/CUDA/source-sync/build failure, product smoke failure, diagnostic c16 profile collected, or hard stop at 30 minutes.
- Correctness gate: `ferrum serve` streaming smoke before treating profile output as useful.
- Performance command: diagnostic-only `ferrum bench-serve ... --fail-on-error --seed 9271 --n-repeats 1` at concurrency 16.
- Claim status: diagnostic only; not release performance evidence and not a W2 PASS artifact.
