# W2 unified graph capture admission CUDA smoke

Date: 2026-06-16

This is a diagnostic smoke artifact only. It is not release-grade evidence and
does not produce a release PASS line.

## Scope

- Lane: W2 Gemma3 27B GPTQ CUDA `--unified-graph --unified-graph-layers-only`
  c16 regression after bounding unified graph capture admission.
- Instance: Vast `40826362`, 1x RTX 4090.
- Stop condition: build failure, serve smoke failure, benchmark failure, error
  scan hit, or completion of one c16 diagnostic.
- Correctness gate before benchmark: `ferrum serve` Paris smoke.
- Benchmark: `ferrum bench-serve --fail-on-error --seed 9271`, ShareGPT 64
  prompts, c16, output length 128, one repeat. No `--require-ci`.

## Hardware and build

- GPU: NVIDIA GeForce RTX 4090, 24564 MiB.
- Driver: 565.77.
- CUDA visible to driver: 12.7.
- NVCC: 12.4.131.
- Git SHA: `88200ac3e63c4c431b59d20cbf47f4c7370d4fc5`.
- Remote dirty status: 0 lines in `remote/build/git_status_short.txt`.
- Binary SHA256:
  `9eb3d4b9670d15b639cae7d90971ebc38dccd88f9eab4e2624be6af74fd838ff`.

Build command:

```bash
cargo build --release -p ferrum-cli --bin ferrum \
  --features cuda,vllm-moe-marlin,vllm-paged-attn-v2,fa2-source
```

## Change Under Test

Commit `88200ac3` bounds unified graph capture admission:

- capture only all-decode unified batches where `m_total == num_seqs`;
- do not capture new unified graph keys after 16 cached keys in the current
  scratch generation;
- log skipped captures as `[unified-graph-skip]` with a reason.

This targets the previous diagnostic failure:

```text
[unified-graph] layers-only end_capture err: Unsupported operation: cuGraphInstantiate failed: CUDA_ERROR_OUT_OF_MEMORY
```

## Smoke Command

Server:

```bash
ferrum serve --model gemma3:27b-gptq --backend cuda \
  --host 127.0.0.1 --port 18164 \
  --max-num-seqs 16 --max-num-batched-tokens 2048 --kv-capacity 512 \
  --unified-graph --unified-graph-layers-only \
  --effective-config-json layers_only_c16/effective_config.json \
  --decision-trace-jsonl layers_only_c16/decision_trace.jsonl
```

Benchmark:

```bash
ferrum bench-serve \
  --base-url http://127.0.0.1:18164 \
  --model gemma3:27b-gptq \
  --tokenizer /root/.cache/huggingface/hub/models--circulus--gemma-3-27b-it-gptq/snapshots/70d89a3a6b401b5f56558cb5d4c0f1fd158980b2 \
  --dataset sharegpt \
  --sharegpt-path /workspace/w2_natural_prompt_baseline_probe_2026-06-15/dataset/ascii_sharegpt.jsonl \
  --random-output-len 128 \
  --concurrency-sweep 16 \
  --num-prompts 64 \
  --n-repeats 1 \
  --fail-on-error \
  --seed 9271 \
  --out layers_only_c16/bench_sharegpt_c16.json
```

## Results

- Status: `SMOKE_PASS`.
- Paris response: `The capital of France is Paris.`
- Paris usage: prompt 23, completion 8, total 31.
- Bench return code: 0.
- Completed / errored: 64 / 0.
- Output token count source: `usage`.
- Output throughput: 339.5 tok/s.
- Goodput: 4.56 req/s.
- TTFT p50 / p95: 299.7 ms / 864.0 ms.
- TPOT p50 / p95: 42.5 ms / 45.2 ms.
- Error scan: 0 lines.
- Unified graph log hits: 22.
- Unified graph skip hits: 9.

Observed skip evidence:

```text
[unified-graph-skip] reason=cache_full count=1 scope=layers_only key=16798760801528114760 m_total=10 num_seqs=10 max_kv_len=141 cached_keys=16
[unified-graph-skip] reason=mixed_or_prefill_batch count=128 scope=layers_only key=13055585321574199491 m_total=137 num_seqs=16 max_kv_len=135 cached_keys=16
[unified-graph-stats] scope=layers_only key=15422059161627171008 m_total=16 num_seqs=16 max_kv_len=157 replays=44 eagers=256 captures=37 skips=209
```

## Interpretation

The bounded admission patch fixes the immediate correctness-risk symptom from
the previous c16 diagnostic: the server no longer logs CUDA graph instantiate
OOM and the error scan is clean.

Performance is still diagnostic only. This run is a single repeat without
`--require-ci`, so it cannot support release-grade claims. It does show that the
new admission policy avoids the OOM while keeping layers-only throughput in the
same range as the prior default-path smoke.

The remaining release-grade performance work is still the deeper CUDA graph
shape problem: max-KV-sensitive attention launch shapes keep changing, so a
vLLM-like graph path needs fewer stable decode shapes rather than exact
`max_kv_len` graph keys.
