# W2 Unified Graph Attention-Key CUDA Smoke

Date: 2026-06-16

This is a diagnostic smoke artifact only. It is not release-grade evidence and
does not produce a release PASS line.

## Scope

- Lane: W2 Gemma3 27B GPTQ CUDA `--unified-graph --unified-graph-layers-only`
  c16 diagnostic after coalescing unified graph attention keys.
- Commit under test: `b990e93904f5b1b4bb1fe016573703a76b4902cc`.
- Instance: Vast `40826362`, 1x RTX 4090.
- Stop condition: build failure, `ferrum run`/`ferrum serve` correctness
  failure, graph OOM/illegal-address/error scan hit, or one c16 diagnostic.
- Correctness gates before benchmark: `ferrum run` arithmetic smoke and
  `ferrum serve` Paris smoke.
- Benchmark: `ferrum bench-serve --fail-on-error --seed 9271`, ShareGPT 64
  prompts, c16, output length 128, one repeat. No `--require-ci`.

## Hardware and Build

- GPU: NVIDIA GeForce RTX 4090, 24564 MiB.
- Driver: 565.77.
- CUDA visible to driver: 12.7.
- NVCC: 12.4.131.
- Remote dirty status: clean for source paths.
- Binary SHA256:
  `194c92163e12d7f8d0657cda1a5f2cd0b632382ee7814b4083e0f321d85362a2`.

Build command:

```bash
CUDA_COMPUTE_CAP=89 cargo build --release -p ferrum-cli --bin ferrum \
  --features cuda,vllm-moe-marlin,vllm-paged-attn-v2,fa2-source
```

## Results

- Status: `SMOKE_PASS`.
- `ferrum run`: `RUN_SMOKE_PASS content='5' tokens=3`.
- `ferrum serve`: `SERVE_SMOKE_PASS content='Paris' completion_tokens=2`.
- Bench completed / errored: 64 / 0.
- Output token count source: `usage`.
- Output throughput: 344.1 tok/s.
- Goodput: 4.61 req/s.
- TTFT p50 / p95: 293.1 ms / 838.4 ms.
- TPOT p50 / p95: 41.9 ms / 44.9 ms.
- Error scan: 0 lines.
- Sparse unified graph log hits: 5 capture, 9 replay, 5 skip.
- Attention keys in sparse graph logs: `8316016994327462400`,
  `8318267720516763747`.

The same-hardware vLLM c16 orientation baseline from the earlier diagnostic is
518.8 tok/s, so this smoke is about 66.3% of vLLM. This is orientation only,
because this run is a single repeat without release CI.

## Interpretation

The key change worked mechanically. The previous diagnostic produced many graph
keys for the same decode batch shape as `max_kv_len` advanced; this smoke shows
one layers-only graph key replaying across `max_kv_len=125..188`:

```text
[unified-graph-replay] origin=pure ... key=11007875343381310285 attention_key=8316016994327462400 m_total=10 num_seqs=10 max_kv_len=126
[unified-graph-replay] origin=pure ... key=11007875343381310285 attention_key=8316016994327462400 m_total=10 num_seqs=10 max_kv_len=188
```

The endpoint performance gain is small: 344.1 tok/s versus the previous
layers-only c16 smoke at 339.5 tok/s. That means raw `max_kv_len` key
fragmentation was real, but it is not the main remaining performance bottleneck.
Next work should move to the heavier decode-layer costs already identified in
the W2 bottleneck synthesis: Marlin gate/up/down MLP time, attention/prefill
mixed batches, and vLLM's broader CUDA graph/padding execution model.

The Vast instance was stopped after artifact collection and confirmed
`actual_status=exited`.
