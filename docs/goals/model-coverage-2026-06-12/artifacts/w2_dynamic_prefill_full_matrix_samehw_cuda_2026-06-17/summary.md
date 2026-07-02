# W2 Dynamic Active-Decode Prefill Same-Hardware Checkpoint

Date: 2026-06-17

Status: failed correctness gate. This is not a W2 PASS artifact.

## Scope

- Ferrum commit: `bcdb3a6f2111d1b908a4daed60319ced760793ef`
- Ferrum binary SHA256: `9fd0ad75e76ed645259cda8e3da96cef122ab487cc47ac8f20650e8079ba686c`
- Hardware: Vast instance `41256521`, 1x RTX 4090
- Driver/CUDA visibility: driver `590.48.01`, CUDA runtime visible through CUDA 12.x stack, build used CUDA 12.4 toolchain
- Model: `circulus/gemma-3-27b-it-gptq`
- Dataset: 100 prompt ASCII ShareGPT fixture, SHA256 `58d5721d8389d7ed9ec4b8b2dbd8797faa61641c6ba023dd150a1a9d93c0a01e`
- Command family: `ferrum bench-serve --fail-on-error --require-ci --seed 9271 --n-repeats 3`

## Completed Evidence

Ferrum CUDA release build completed and produced the binary SHA above.

vLLM 0.10.1.1 baseline completed on the same instance with 0 request errors and usage-derived output token counts:

| concurrency | output tok/s mean | CI95 half-width | completed per run | errored per run |
| --- | ---: | ---: | --- | --- |
| 1 | 43.2983 | 0.0795 | 100,100,100 | 0,0,0 |
| 4 | 166.4720 | 3.6346 | 100,100,100 | 0,0,0 |
| 16 | 488.1906 | 57.1789 | 100,100,100 | 0,0,0 |
| 32 | 713.1608 | 125.1589 | 100,100,100 | 0,0,0 |

Ferrum `run` smoke completed:

```json
{
  "assistant_events": 1,
  "content": "5",
  "n_tokens": 3
}
```

## Failure

Ferrum `serve` correctness failed before Ferrum performance sweep could start.

The `serve` path started with `--max-num-seqs 32 --kv-capacity 512`; the effective config selected:

- `kv_block_count=1024`
- `max_sequences=32`
- `max_batched_tokens=2048`
- `scheduler_admission_policy=active_decode_prefill_chunk:16`

During the serving smoke request, CUDA allocation failed:

```text
CudaBackend::alloc failed: dtype=F16 elements=33554432 bytes=67108864 free=29032448 total=25252724736 label=<none>: DriverError(CUDA_ERROR_OUT_OF_MEMORY, "out of memory")
```

This is a correctness blocker. Ferrum performance comparison against vLLM is not valid until `ferrum serve` passes.

## Artifact Paths

- Remote artifact copy: `remote_retry2_failure/`
- vLLM baseline JSON: `remote_retry2_failure/perf/bench_vllm_sharegpt_sweep_100x3.json`
- Ferrum run summary: `remote_retry2_failure/correctness/run_summary.json`
- Ferrum serve OOM log: `remote_retry2_failure/correctness/serve.log`
- Ferrum serve effective config: `remote_retry2_failure/correctness/serve_effective_config.json`
- Vast stop verification: `local_vast/stop_41256521_poll.jsonl` and `local_vast/stop_41256521_final.raw.json`

## Cleanup

After collecting artifacts, the Ferrum server process was stopped and Vast instance `41256521` was stopped. Final Vast state:

- `cur_state=stopped`
- `actual_status=exited`

## Next Fix Direction

The next change should focus on CUDA serve memory sizing for 24GB cards. The current serve preset can reserve KV/scratch so aggressively that model load succeeds but the first serving request has only tens of MB free for a transient 64MB allocation. Fix the product default or auto-sizing guard first, then rerun the minimal `ferrum serve` smoke before any Ferrum performance sweep.
