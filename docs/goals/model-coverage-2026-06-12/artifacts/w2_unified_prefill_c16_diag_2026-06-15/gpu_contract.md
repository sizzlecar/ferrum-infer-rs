# W2 Gemma3 CUDA unified-prefill c16 diagnostic

- lane: W2 Gemma3 CUDA unified-prefill c16 diagnostic
- expected runtime/cost: 10-20 minutes, hard cap 30 minutes, approximately USD 0.425/hr on existing 1x RTX 4090 Vast instance 40826362
- stop condition: instance start/SSH/source sync/server readiness first failure, chat smoke first failure, bench-serve first failure, malformed output or missing usage, or c16 diagnostic complete after artifacts copied back
- correctness gate: `ferrum serve` readiness, non-stream chat smoke, then `bench-serve --fail-on-error`
- performance command: diagnostic-only `bench-serve --fail-on-error --seed 9271 --n-repeats 1 --concurrency-sweep 16 --num-prompts 16`; no `--require-ci`, no release performance claim
- baseline: compare informally against existing same-hardware vLLM c16 baseline artifact; this run does not create release-grade ratio evidence
