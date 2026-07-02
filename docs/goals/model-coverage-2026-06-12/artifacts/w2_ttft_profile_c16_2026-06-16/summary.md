# W2 TTFT profile c16 diagnostic — 2026-06-16

Diagnostic only. This is not release performance evidence and produced no
`MODEL_RELEASE_GRADE_W2 PASS`.

## Scope

- Reused Vast instance `41187356`, 1x RTX 4090, driver `580.95.05`, CUDA 12.4.
- Remote clean worktree: `8eccd1c33c752937cf903f63638eaa6d51bd643e`.
- Binary SHA256:
  `ae817f5b086275a9c8689c8c991d504bb79b73fa39eac8032cbb2368972d5cd1`.
- Diagnostic config source:
  `[runtime] batch_decode_prof=true`, `next_batch_prof=true`,
  `unified_post_prof=true` from the saved diagnostic `ferrum.toml`.
- Cleanup: instance stop confirmed with `actual_status=exited`.

## Runs

- First attempt:
  - `--kv-capacity 2048 --max-num-seqs 16`;
  - failed before correctness smoke due CUDA OOM;
  - failure: 128 MiB F16 allocation with only about 94 MiB free.
- Second attempt:
  - `--kv-capacity 512 --max-num-seqs 16 --max-num-batched-tokens 1024`;
  - `ferrum serve` streaming smoke passed (`SMOKE_OK True`);
  - `bench-serve --fail-on-error --seed 9271 --n-repeats 1` completed
    16/16 requests with 0 errors.

## Diagnostic result

`bench-serve` c16 random 64/16, n=1:

- TTFT p50 `674.9 ms`, p95 `781.6 ms`, p99 `844.5 ms`;
- TPOT p50 `53.8 ms`, p95 `86.1 ms`;
- throughput `167.9 tok/s`;
- output token count source: `usage`;
- errors: 0, malformed stream: 0, missing/duplicate DONE: 0.

`first-token-prof` / `stream-ttft-prof` split, 21 first-token events
(smoke + warmup + measured requests):

- `queue_to_model_start_us`: p50 `87,622`, mean `95,326`;
- `model_batch_us`: p50 `421,647`, mean `312,066`;
- `queue_to_first_token_us`: p50 `559,364`, mean `458,436`;
- stream TTFT: p50 `565,475`, mean `464,673`.

The heaviest observed prefill/unified call:

```text
[unified-decode] call#21 items=15 prefill=13 decode=2 total_q=968 attempted_unified=true fallback=false elapsed=421564us
```

## Interpretation

The current c16 TTFT is not primarily client/SSE buffering, nor mostly
waiting before model execution. The largest measured component is the mixed
unified model call for a roughly 1k-token prefill batch. The next W2 lever
should compare Ferrum's Gemma3 GPTQ unified prefill path against vLLM for the
same shape, then isolate whether attention, dense MLP/Marlin, or packing/logits
dominates that ~421 ms call.

