# W2 unified graph layers-only c16 diagnostic

Date: 2026-06-16

This is a diagnostic artifact only. It is not release-grade evidence and does not
produce a release PASS line.

## Scope

- Lane: W2 Gemma3 27B GPTQ CUDA default vs `--unified-graph --unified-graph-layers-only`
  c16 diagnostic.
- Instance: Vast `40826362`, 1x RTX 4090.
- Stop condition: build failure, serve smoke failure, benchmark failure, or completion of
  one default/layers-only c16 diagnostic pair.
- Correctness gate before benchmark: layers-only `ferrum serve` Paris smoke.
- Benchmark: `ferrum bench-serve --fail-on-error --seed 9271`, ShareGPT 64 prompts,
  c16, output length 128, one repeat. No `--require-ci`.

## Hardware and build

- GPU: NVIDIA GeForce RTX 4090, 24564 MiB.
- Driver: 565.77.
- CUDA visible to driver: 12.7.
- NVCC: 12.4.131.
- Git SHA: `260f4583eb9550445be470f462e6b4c9b10702d8`.
- Remote dirty status: 0 lines in `remote/build/git_status_short.txt`.
- Binary SHA256:
  `11b955f8722e8ddc5f9de9023ba5796274dbceeedc65ca03b6d578d8776ffb65`.

Build command:

```bash
cargo build --release -p ferrum-cli --bin ferrum \
  --features cuda,vllm-moe-marlin,vllm-paged-attn-v2,fa2-source
```

## Correctness smoke

Command shape:

```bash
ferrum serve --model gemma3:27b-gptq --backend cuda \
  --host 127.0.0.1 --port 18161 \
  --max-num-seqs 16 --max-num-batched-tokens 2048 --kv-capacity 512 \
  --unified-graph --unified-graph-layers-only \
  --effective-config-json smoke_layers_only/effective_config.json \
  --decision-trace-jsonl smoke_layers_only/decision_trace.jsonl
```

Result:

- Status: `SMOKE_PASS`.
- Response: `The capital of France is Paris.`
- Usage: prompt 23, completion 8, total 31.
- Effective graph mode: `unified_decode_graph_layers_only`.
- Error scan: empty.
- Unified graph log hits: 6.

## c16 diagnostic commands

Default server:

```bash
ferrum serve --model gemma3:27b-gptq --backend cuda \
  --host 127.0.0.1 --port 18162 \
  --max-num-seqs 16 --max-num-batched-tokens 2048 --kv-capacity 512 \
  --effective-config-json bench/default/effective_config.json \
  --decision-trace-jsonl bench/default/decision_trace.jsonl
```

Layers-only server:

```bash
ferrum serve --model gemma3:27b-gptq --backend cuda \
  --host 127.0.0.1 --port 18163 \
  --max-num-seqs 16 --max-num-batched-tokens 2048 --kv-capacity 512 \
  --unified-graph --unified-graph-layers-only \
  --effective-config-json bench/layers_only/effective_config.json \
  --decision-trace-jsonl bench/layers_only/decision_trace.jsonl
```

Benchmark command for each server:

```bash
ferrum bench-serve \
  --base-url http://127.0.0.1:<port> \
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
  --out bench/<case>/bench_sharegpt_c16.json
```

## Results

| Case | Completed | Errored | Output token source | Output tok/s | Goodput req/s | TTFT p50 ms | TPOT p50 ms | Error scan |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |
| default | 64 | 0 | usage | 344.1 | 4.70 | 211.0 | 42.7 | 0 lines |
| layers-only | 64 | 0 | usage | 323.1 | 4.35 | 307.9 | 43.6 | 1 line |

Diagnostic delta: layers-only was 6.1% slower on output tok/s in this single
repeat. Because `n_repeats=1` and the layers-only error scan was not clean, this
is not valid release performance evidence.

## Key finding

Layers-only graph capture and replay are happening, but the current exact-shape
graph key creates too many graph instantiations under c16 natural prompts.

Observed layers-only graph stats:

```text
[unified-graph-stats] scope=layers_only key=13114892978788168881 m_total=137 num_seqs=16 max_kv_len=197 replays=226 eagers=256 captures=171
```

The correctness risk is the CUDA graph instantiate failure:

```text
[unified-graph] layers-only end_capture err: Unsupported operation: cuGraphInstantiate failed: CUDA_ERROR_OUT_OF_MEMORY
```

The benchmark itself completed 64/64 requests with zero request errors, but the
server log error scan intentionally fails the diagnostic checkpoint.

## Interpretation

The current bottleneck is no longer "graph does not replay". It does replay.
The problem is graph cache shape cardinality and memory pressure. The exact
`m_total`/`num_seqs`/`max_kv_len` style key is too granular for c16 mixed prompt
lengths, which causes repeated capture/instantiate work and eventual graph
instantiate OOM. That overhead also explains why layers-only is slower than the
default path in this diagnostic.

The next implementation direction should follow the vLLM-style constraint more
closely: restrict CUDA graph capture to a small admitted set of stable decode
shapes, bucket or remove volatile length dimensions where safe, cap the graph
cache, and permanently fall back to eager for shapes that hit instantiate OOM.
