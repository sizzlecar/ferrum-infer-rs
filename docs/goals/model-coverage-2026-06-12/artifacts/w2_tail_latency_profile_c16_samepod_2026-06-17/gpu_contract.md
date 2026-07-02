# W2 Tail-Latency Profile c16 Same-Pod GPU Contract

- Date: 2026-06-17
- Lane: W2 Gemma3 current-default c16 tail-latency/profile diagnostic.
- Scope: diagnostic only. This is not a release-grade performance claim and
  cannot satisfy `MODEL_RELEASE_GRADE_W2 PASS`.
- Hardware target: reuse stopped Vast instance `41241013`, 1x RTX 4090, with
  the model/source/build environment retained from the same-pod c16 A/B run.
- Expected runtime/cost: 10-20 minutes if the retained environment starts
  cleanly; hard stop at 35 minutes. The recorded running rate is about
  `0.47111111111111115 USD/h`.
- Stop condition: startup, SSH, CUDA, server readiness, smoke, or bench first
  failure; or profile artifact collected. Copy artifacts first, then stop the
  instance and record `actual_status=exited`.
- Correctness gate:
  - `ferrum serve` readiness through `/v1/models`
  - streaming chat smoke returns content `5` with exactly one `[DONE]`
  - usage is present with positive completion tokens
  - `bench-serve --fail-on-error` completes c16 without request errors
- Diagnostic command shape:
  - current default Ferrum product server with `FERRUM_DECODE_OP_PROFILE=1`
  - `ferrum bench-serve`
  - `--dataset sharegpt`
  - `--sharegpt-path ascii_sharegpt_w2_100.jsonl`
  - `--random-output-len 128`
  - `--concurrency-sweep 16`
  - `--num-prompts 100`
  - `--n-repeats 1`
  - `--fail-on-error`
  - `--seed 9271`
- Purpose: correlate the same-pod c16 p95 ITL failure with current default
  Ferrum decode profile buckets before choosing the next product code change.
