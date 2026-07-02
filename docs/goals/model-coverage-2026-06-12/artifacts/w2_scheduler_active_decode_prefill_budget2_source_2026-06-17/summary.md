# W2 scheduler source checkpoint: two active-decode prefill chunks

This is source-only evidence. It did not run CUDA product correctness or
performance, and did not produce `MODEL_RELEASE_GRADE_W2 PASS`.

## Motivation

The previous aggregate active-decode mixed-prefill cap validated the root
cause: c16 p95 ITL improved from `52.819ms` to `26.637ms`. It also over-
throttled throughput: c16 LCB fell from `414.592 tok/s` to `333.110 tok/s`,
below the 80% vLLM line.

## Change

`active_decode_prefill_chunk` still caps each prefill request's chunk size, but
the per-iteration aggregate active-decode mixed-prefill budget is now
`2 * active_decode_prefill_chunk` instead of `1 * active_decode_prefill_chunk`.

For the Gemma3 CUDA GPTQ typed default `active_decode_prefill_chunk=16`, this
allows at most two 16-token prefill chunks to mix into an active decode step.
The intent is to avoid the prior `prefill=7-11` tail-spike pattern while
recovering some throughput lost by the one-chunk cap.

## Local Validation

- `cargo test -p ferrum-scheduler active_decode_prefill_chunk -- --nocapture`
  PASS: `2 passed`.
- `cargo fmt --all -- --check` PASS.
- `cargo test -p ferrum-scheduler` PASS: `52 passed`.

## Required Next Validation

Run Gemma3 product `ferrum run` and `ferrum serve` smoke on native CUDA, then a
c16 ShareGPT minimum with:

```text
ferrum bench-serve --dataset sharegpt --sharegpt-path /workspace/ascii_sharegpt_w2_100.jsonl --random-output-len 128 --concurrency-sweep 16 --num-prompts 100 --n-repeats 3 --fail-on-error --require-ci --seed 9271
```

Acceptance for this diagnostic candidate is both:

- c16 throughput LCB / same-pod vLLM LCB >= `0.80`;
- c16 p95 ITL / same-pod vLLM p95 ITL <= `1.00`.
