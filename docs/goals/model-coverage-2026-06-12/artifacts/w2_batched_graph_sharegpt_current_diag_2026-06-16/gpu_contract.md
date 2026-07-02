# W2 Batched Graph ShareGPT Diagnostic GPU Contract

- Lane: W2 current HEAD `--batched-graph` ShareGPT same-dataset diagnostic.
- Hardware: cached Vast instance `40826362`, 1x RTX 4090.
- Expected runtime/cost: 15-30 minutes, about USD 0.11-0.22 at
  USD 0.42488888888888887/h.
- Stop condition: startup, SSH, CUDA, sync, serve, or bench first failure, or
  c16/c32 diagnostic artifact collected, then stop the instance.
- Correctness gate: server readiness, chat smoke response `5` with usage,
  bench rc 0, completed requests equal expected prompts, zero request errors,
  zero bad output, zero zero-output responses, zero HTTP 500, and clean server
  error scan.
- Performance command: product CLI `ferrum serve --batched-graph` plus
  `ferrum bench-serve --dataset sharegpt --sharegpt-path ascii_sharegpt.jsonl
  --random-output-len 64 --concurrency-sweep 16,32 --num-prompts 16
  --n-repeats 1 --fail-on-error --seed 9271`.
- Scope: diagnostic only. This run intentionally omits `--require-ci` and does
  not produce `MODEL_RELEASE_GRADE_W2 PASS`.
