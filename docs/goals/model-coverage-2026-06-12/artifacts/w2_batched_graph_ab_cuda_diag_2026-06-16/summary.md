# W2 batched-graph A/B CUDA diagnostic

## Verdict

`--batched-graph` is correct on the minimal product-path smoke, but it does not
improve c16 throughput for the current Gemma3-27B GPTQ path.

This is diagnostic evidence only, not release-grade evidence.

## Source And Binary

- Source checkpoint: `0adb292a` locally, reusing remote clean source at
  `d6d872c1e12fc364886117b0431aec752b2d78ac`
- Binary SHA256:
  `11b26df2b8dccf3138b2fe294e80ef618cc6255e56af626213e6aaabe8b2e48f`
- Vast instance: `40826362`, 1x RTX 4090
- CUDA: driver `565.77`, runtime-reported CUDA `12.7`, `nvcc 12.4.131`
- No rebuild was performed; this reused the binary from
  `w2_paged_unified_default_path_cuda_smoke_2026-06-16`.

## Correctness

`ferrum run --batched-graph`:

- rc: `0`
- Output:
  `{"event":"assistant","turn":0,"content":"5","finish_reason":"stop","n_tokens":3,"chunk_count":1,"ms":397.64146}`

`ferrum serve --batched-graph`:

- readiness: `/v1/models` ready on poll `8`
- chat rc: `0`
- Response:
  `{"choices":[{"message":{"content":"5"},"finish_reason":"length"}],"usage":{"prompt_tokens":23,"completion_tokens":1,"total_tokens":24}}`
- Health after bench: `331` successful requests, `0` failed requests
- Server log scan: `0` matches for panic/error/NaN/`<unk>`/`[PAD]`/invalid UTF,
  fallback, graph-failed, or capture-failed patterns used in this artifact

Effective server config:

- `selected_graph_mode`: `legacy_batched_decode_graph`
- `selected_kv_layout`: `paged`
- `selected_attention_impl`: `legacy_paged_decode`
- `selected_max_sequences`: `16`
- `selected_kv_capacity`: `512`
- `selected_max_batched_tokens`: `2048`

## Performance

Command shape:

```text
bench-serve --random-input-len 256 --random-output-len 128 --concurrency-sweep 16 --num-prompts 100 --n-repeats 3 --fail-on-error --require-ci --seed 9271
```

Result:

- rc: `0`
- completed per run: `[100, 100, 100]`
- errored per run: `[0, 0, 0]`
- output token count source: `usage`
- output throughput: `287.1167006548677 ± 41.632552793935645 tok/s`
- goodput: `2.251751298484382 ± 0.3173552733445153 req/s`
- TTFT p50: `798.3041205 ms`
- TPOT p50: `46.949957875328074 ms`

Comparison:

- Previous default-path same-binary c16:
  `295.8064415567493 ± 5.210666937312439 tok/s`
- Batched-graph / default ratio: `0.970624`
- Direct random-prompt vLLM diagnostic baseline:
  `381.3929242134927 ± 13.831767810454199 tok/s`
- Batched-graph / random vLLM diagnostic ratio: `0.752811`

## Interpretation

- The `FERRUM_BATCHED_GRAPH` product switch materializes correctly:
  `decode_graph_policy=legacy_batched_decode_graph`.
- It does not close the W2 c16 performance gap and should not be promoted as
  the current release-grade lever.
- The remaining bottleneck is more likely decode cadence, scheduler/admission,
  or per-token tail work above/beside graph replay.

## Shutdown

- Server process was stopped.
- `nvidia_smi_after_server_stop.txt` showed no running GPU processes.
- Vast shutdown poll verified `cur_state=stopped actual_status=exited`.
