# W2 Gemma3 CUDA prefill profile buckets validation

- Lane: W2 Gemma3 CUDA prefill profile buckets validation.
- Instance: reuse Vast instance `40826362`, 1x RTX 4090, about USD 0.424888/hour.
- Expected runtime/cost: 15-35 minutes, hard cap 45 minutes, expected cost under USD 0.32.
- Stop condition: stop on instance start/SSH/CUDA/source sync/build failure, serve readiness failure, chat smoke failure, c16 diagnostic completion, or the hard cap.
- Correctness gate: CUDA release build, serve readiness, non-stream chat smoke, then `bench-serve --fail-on-error`.
- Performance command: c16 ShareGPT diagnostic only, `--seed 9271 --n-repeats 1 --num-prompts 16`, with `FERRUM_PREFILL_OP_PROFILE=1`.
- Evidence scope: diagnostic only; not release-grade evidence and not an official performance claim.
