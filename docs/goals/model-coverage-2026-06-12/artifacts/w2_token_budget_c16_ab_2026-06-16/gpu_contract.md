# GPU contract - W2 token-budget c16 A/B diagnostic

- Lane: W2 Gemma3 27B GPTQ c16 typed token-budget diagnostic.
- Instance: attempted reuse of Vast `41187356`, then discarded loading-only
  instance `41210668`; actual run used Vast `41212840`, 1x RTX 4090,
  offer `36846332`, California, US, quoted at USD `0.4044/h`.
- Expected runtime/cost: 1.5-3 hours for a fresh instance including setup,
  model prefetch, build, smoke, and A/B; stop early on SSH/CUDA failure.
- Stop condition: collect both `max_num_batched_tokens=1024` and `512`
  smoke+bench artifacts, or stop on first serve/correctness/bench failure.
- Correctness gate: for each cell, `ferrum serve` readiness, streaming smoke
  for `2+3`, and `bench-serve --fail-on-error` with zero request errors.
- Performance command: diagnostic-only c16 random 64/16, `n_repeats=1`,
  `seed=9271`, no `--require-ci`; this is not release-grade evidence.
- Product-path constraint: use typed CLI flags and saved config snapshot, not
  hidden env behavior.
- Cleanup: actual run artifact copied back locally; instance `41212840`
  deleted by Vast API, `DELETE` returned HTTP 200 with `success=true`.
