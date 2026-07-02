# W2 Autosize Serve Smoke CUDA

## Result

PASS:

```text
W2 AUTOSIZE SERVE SMOKE PASS: /workspace/w2_autosize_serve_smoke_cuda_2026-06-17
```

This is correctness smoke evidence only. It is not W2 release-grade performance evidence.

## Scope

- Local source commit: `99859ce6e9e0c315ebf614f02a0ccec661a6c215`
- Remote checkout status: clean (`remote/env/git_status_short.txt` empty)
- Instance: Vast `41256521`
- Hardware: 1x NVIDIA GeForce RTX 4090, 24564 MiB
- Build command: `cargo build --release -p ferrum-cli --bin ferrum --features cuda,vllm-moe-marlin,vllm-paged-attn-v2,fa2-source`
- Build duration: 39m 02s
- Binary SHA256: `33a1999b0805c1daca1dd5b102eb9e5d91065951343d11e966a1816f3b418dc1`

## Correctness Smoke

Command shape:

```text
ferrum serve gemma3:27b-gptq --backend cuda --port 18456 --kv-capacity 512 --max-num-seqs 32
```

The server started and the streaming `/v1/chat/completions` smoke succeeded:

- content: `5\n`
- `data: [DONE]`: 1
- usage: `prompt_tokens=23`, `completion_tokens=3`, `total_tokens=26`
- log blocker scan: no `out of memory`, `oom`, `panic`, or `cuda_error_out_of_memory`

## Important Finding

The autosizer avoided the previous serve OOM by clamping the requested paged KV pool:

```text
[auto-size] requested paged KV pool requires 1024 blocks (max_seqs=32 kv_capacity=512) above memory budget 338; clamping to max_seqs=8 kv_capacity=512
```

Effective runtime keys:

- `FERRUM_PAGED_MAX_SEQS=8` from `memory_profile`
- `FERRUM_KV_CAPACITY=512` from CLI
- `FERRUM_MAX_BATCHED_TOKENS=2048` from `memory_profile`
- `FERRUM_ACTIVE_DECODE_PREFILL_CHUNK=16` from default

This confirms the correctness/OOM blocker is fixed for product `serve`, but it does not prove true c=32 capacity or c=32 performance. Any next same-hardware performance comparison must either use the actual effective cap or fix the memory budget/runtime shape so c=32 remains effective.

## Artifact Map

- Remote run log: `remote/run.log`
- Build log: `remote/build/cargo_build_release.log`
- Server log: `remote/correctness/serve.log`
- Effective config: `remote/correctness/serve_effective_config.json`
- Decision trace: `remote/correctness/serve_decision_trace.jsonl`
- Stream summary: `remote/correctness/smoke/stream_summary.json`
- Cleanup proof: `local_vast/stop_41256521_poll.jsonl`

Final Vast state:

```text
{"ts":"2026-06-17T02:19:10Z","id":41256521,"cur_state":"stopped","actual_status":"exited","intended_status":"stopped","gpu_name":"RTX 4090","num_gpus":1}
```
