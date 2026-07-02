# W2 dynamic KV c32 serve smoke

Status: PASS

PASS line:

```text
W2 DYNAMIC KV C32 SERVE SMOKE PASS: /workspace/w2_dynamic_kv_c32_serve_smoke_cuda_2026-06-17
```

Scope:

- Correctness smoke only. This is not release evidence and not a performance claim.
- Product entrypoint: `ferrum serve`.
- Model: `gemma3:27b-gptq` (`circulus/gemma-3-27b-it-gptq`).
- Backend: CUDA, 1x RTX 4090.
- Commit: `9fbfb77395dfbbbe38d90836463bc4dddaaeca78`.
- Remote git status: clean.
- Binary SHA256: `c0a48db82e77f98d7ebde3225b5be103997198a85e467cbdbbf75c2803e12749`.

Command:

```text
target/release/ferrum serve gemma3:27b-gptq --backend cuda --host 127.0.0.1 --port 18458 --max-num-seqs 32 --effective-config-json /workspace/w2_dynamic_kv_c32_serve_smoke_cuda_2026-06-17/correctness/serve_effective_config.json --decision-trace-jsonl /workspace/w2_dynamic_kv_c32_serve_smoke_cuda_2026-06-17/correctness/serve_decision_trace.jsonl
```

Runtime evidence:

- Build command completed in 2232 seconds.
- `FERRUM_PAGED_MAX_SEQS=32`, source `cli`.
- `FERRUM_KV_MAX_BLOCKS=338`, source `memory_profile`.
- `FERRUM_KV_CAPACITY=2048`, source `memory_profile`.
- `FERRUM_MAX_BATCHED_TOKENS=2048`, source `memory_profile`.
- Server log shows `Creating paged KV cache manager: device=CUDA(0), block_size=16, max_blocks=338`.
- Stream smoke returned `content="5\n"`, `done_count=1`, `json_error_count=0`, and usage `{prompt_tokens: 18, completion_tokens: 3, total_tokens: 21}`.
- Blocker scan file is empty for `out of memory|oom|panic|cuda_error_out_of_memory`.

Lifecycle:

- Vast instance `41241013` was stopped after artifact copy.
- Cleanup poll reached `cur_state=stopped`, `actual_status=exited`.
