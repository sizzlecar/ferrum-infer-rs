# W2 Prefill/TTFT Profile Diagnostic

Date: 2026-06-15

Scope: diagnostic-only native CUDA run on Vast instance `40826362`
(1x RTX 4090). This is not release-grade evidence: `n_repeats=1`, no
`--require-ci`, no `model_release_grade_manifest.json`, and no
`MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`.

## Contract

- lane: W2 Gemma3 CUDA prefill/TTFT profile diagnostic
- expected runtime/cost: 8-20min, hard cap 30min, about USD 0.425/hr while
  running
- stop condition: start/SSH/CUDA/server readiness first failure, chat smoke
  failure, c16 ShareGPT diagnostic complete and artifacts copied, or 30min
  hard cap
- correctness gate: `ferrum serve` readiness plus non-stream chat smoke before
  `bench-serve`; `bench-serve` used `--fail-on-error`
- performance command: diagnostic-only natural ASCII ShareGPT c16 with
  `bench-serve --fail-on-error --seed 9271 --n-repeats 1 --num-prompts 16`
- profile scope: server ran with `FERRUM_PREFILL_OP_PROFILE=1`

## Correctness

- `run.status`: `PASS`
- `bench-serve.rc`: `0`
- chat smoke content: `5`
- chat smoke usage: `prompt_tokens=23`, `completion_tokens=3`,
  `total_tokens=26`
- c16: `16 completed / 0 errored`, bad output `[0]`

No new Ferrum product correctness issue was found in this diagnostic.

## Performance

Ferrum c16 diagnostic:

| completed | errored | output throughput | TTFT p50 | TTFT p95 | TPOT p50 | ITL p50 | ITL p99 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 0 | 340.882 tok/s | 889.558 ms | 1452.948 ms | 32.804 ms | 24.678 ms | 281.837 ms |

Compared with clean vLLM ShareGPT baseline c16
(`518.796 tok/s`), this diagnostic is `0.657x` vLLM and remains about
`14.3` percentage points below the 80% release-grade line.

## Prefill Profile

The run captured 27 `prefill-profile` total rows:

- smoke request: `tokens=23`, total `29ms`
- ShareGPT request prefills: `tokens=122`, total mostly `80ms`
- summary: min `29ms`, max `88ms`, mean `79.1ms`, median `80ms`

The bucket breakdown was empty even though `FERRUM_PREFILL_OP_PROFILE=1` was
set. Source inspection showed why: the ordinary op timers in
`llama_family.rs` were gated only by `decode_op_profile`; the prefill profile
drained those counters but never enabled them. This checkpoint therefore adds
a source fix so prefill profile can collect op and tail buckets on the next
native CUDA diagnostic.

## Shutdown

Artifact copied back locally before stopping the instance. Vast shutdown poll 1
recorded `cur_state=stopped`, `actual_status=exited`.
