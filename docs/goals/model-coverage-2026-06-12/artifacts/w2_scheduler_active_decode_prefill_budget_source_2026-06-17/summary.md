# W2 Scheduler Active-Decode Prefill Budget Source Checkpoint

Date: 2026-06-17

This is a source-only checkpoint for the W2 Gemma3 CUDA GPTQ release-grade
goal. It did not start a paid GPU instance and it did not produce
`MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`.

## Why This Lever

The latest same-pod c16 evidence shows the remaining W2 blocker is p95 ITL,
not average throughput:

- Artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_vllm_same_hw_c16_sharegpt_2026-06-17/`.
- Ferrum c16 throughput LCB / vLLM throughput LCB:
  `0.8666308262975292`.
- Ferrum p95 ITL / vLLM p95 ITL:
  `1.597218665188174`, which fails the `<= 1.25x` release-grade rule.

A local split of the c16 profile artifact
`w2_tail_latency_profile_c16_samepod_2026-06-17` showed:

- pure decode frames: `21` rows, `208` decoded tokens in those rows,
  total frame p95 `27.978ms`;
- mixed prefill+decode frames: `152` rows, `1666` decoded tokens in those rows,
  total frame p95 `74.050ms`;
- decode-token-weighted frame p95: `64.029ms`;
- token share by frame type:
  - pure decode: `11.1%`;
  - mixed with `1-2` prefills: `20.0%`;
  - mixed with `3-4` prefills: `42.8%`;
  - mixed with `>=5` prefills: `26.1%`.

The same split shows the current `active_decode_prefill_chunk=16` default
limited each prefill request chunk, but did not limit the aggregate prefill
work admitted into the same decode iteration. The worst mixed frames had
`prefill=9-11`, `decode=5-7`, and total frame time around `77-85ms`.

## Source Change

`crates/ferrum-scheduler/src/implementations/continuous.rs` now carries a
per-iteration `active_decode_prefill_tokens_remaining` budget when decode
requests were scheduled in the current iteration.

When `active_decode_prefill_chunk=16`, active decode iterations can now admit
at most `16` aggregate prefill tokens, not `16` tokens per waiting prefill
request. This preserves the existing typed product knob and avoids hidden
environment combinations.

Added regression test:

- `active_decode_prefill_chunk_caps_aggregate_mixed_prefill_tokens`

The test simulates one active decode plus four waiting 256-token prompts with
`active_decode_prefill_chunk=64`; the mixed batch schedules only the decode plus
one 64-token prefill chunk.

## Validation

Commands run locally:

```bash
cargo test -p ferrum-scheduler active_decode_prefill_chunk -- --nocapture
cargo test -p ferrum-scheduler newly_admitted_prefill_uses_remaining_budget_with_decode -- --nocapture
cargo test -p ferrum-scheduler
cargo fmt --all -- --check
```

Results:

- targeted scheduler tests: PASS;
- `cargo test -p ferrum-scheduler`: `52 passed`;
- `cargo fmt --all -- --check`: PASS.

## Expected CUDA Validation

This source change must still be validated on the cached/native CUDA lane before
any performance claim:

1. Product correctness smoke:
   `ferrum run` and `ferrum serve` for Gemma3 27B GPTQ.
2. Minimal c16 ShareGPT A/B on the same machine:
   - previous checkpoint binary vs this checkpoint, or previous artifact vs
     rerun if the same cached pod is reused;
   - `bench-serve --fail-on-error --require-ci --seed 9271 --n-repeats 3`.
3. Tail profile with `FERRUM_DECODE_OP_PROFILE=1` and
   `FERRUM_MARLIN_PROFILE=1` only if the A/B still fails p95.

Expected profile movement:

- mixed frames with active decode should no longer show `prefill=7-11`;
- p95 ITL should move toward the pure-decode / `1-2` prefill frame range;
- throughput may fall slightly, so the same-pod LCB ratio must be rechecked
  against the `>= 0.8` W2 release-grade threshold.
