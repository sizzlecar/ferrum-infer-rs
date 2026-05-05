# v0.2-cuda — Chunked-Prefill + CUDA Graph Capture (Phase 13)

**Date:** 2026-05-05
**Branch:** `bench/v0.2-cuda`
**HEAD:** `7561af2`
**Pod:** Vast.ai RTX 4090 (sm_89)
**Model:** Llama-3.1-8B GPTQ-INT4 (`/workspace/models/M2`, desc_act=true, gs=128)
**Bench harness:** `bench/v0.2-cuda/bench_clean_c16.sh`
**Client:** `vllm bench serve --random-input-len 256 --random-output-len 128 --num-prompts 32 --max-concurrency 16 --temperature 0 --ignore-eos`

## TL;DR

| Stage | Throughput | TPOT_p50 | vs vLLM 1986 |
|------|------|------|------|
| Pre-session baseline (Phase 11/12) | 510-520 tok/s | 23.6 ms | 26% |
| chunked-prefill paged + unified_forward | 650 | 20.3 | 33% |
| + varlen single-launch split-fused | 680 | 19.3 | 34% |
| **+ CUDA Graph capture** | **714** | **18.3** | **36%** |

**Net session win: +39% throughput (510→714 tok/s), -22% TPOT (23.6→18.3 ms).**

vLLM apples-to-apples baseline (same pod, same model, same client params, max-num-seqs 32, max-model-len 4096, gpu-memory-utilization 0.9, default torch.compile graph): 1986 tok/s, TPOT_p50 7.4 ms.

## What changed

15 commits on the chunked-prefill / graph capture series:

| Commit | What |
|---|---|
| `4c2f234` | Step 5b2: real `unified_forward_internal` impl (~645 LoC). Mixed prefill chunks + decode tokens in one [M_total, h] forward pass via the new `paged_varlen_attention` kernel. |
| `d804131` | Step 3b minimal: `LlmExecutor::{prefill, decode, batch_decode}` now try `model.unified_forward(...)` first, fall back to legacy paths on `Unsupported`. No engine refactor needed. |
| `29e1b0b` | Initial gate on `B::supports_paged_kv()` (later relaxed once CUDA varlen kernel landed). |
| `b902a26` | New CUDA kernel `split_qkv_norm_rope_into_paged_cache.cu` (per-item entry, byte_offset arg path). |
| `cfc0170` | Bug fix #1: Q output `head-major → token-major` (paged_varlen reads token-major; tokens>1 prefill returned garbage). Bug fix #2: byte_offsets used `size_of::<f32>()` on f16 buffers — 2× too large. |
| `cc0f976` | Bug fix #3: index buffers (`cu_seqlens_q`, `pos_offsets`, `block_tables`) were sized to FIRST call's `num_seqs`. Warmup at c=1 → bench at c=16 → CUDA_INVALID_VALUE. Pin to engine-side `paged_max_seqs` cap. |
| `3cf3bf6` | Panic guard: `unified_forward` returns `Err(ResourceExhausted)` instead of panicking when pool is exhausted (panic killed tokio workers; cascading failures). |
| `1645663` | **Critical leak fix:** engine `run_batch_decode` was passing `request_id` (UUID) as `UnifiedBatchItem.seq_id`, but the model keyed `kv_caches` by the executor-generated id (`llm-cache-N`). Every decode iter allocated a fresh model-side cache + 128 paged blocks. Pool exhausted in ~60 prompts; bench rep 1 OK, rep 2/3 collapsed. Pull `seq.model_cache_id` instead. |
| `fd75525` | Per-op profile dumper for unified_forward (`FERRUM_DECODE_OP_PROFILE=1`). |
| `c26cf36` | New CUDA kernel `split_qkv_norm_rope_into_paged_cache_varlen_f16` — single launch per layer covering all sequences. Reads `pos_offsets` / `cu_seqlens_q` / `block_tables` from device buffers (graph-capturable). Replaces the per-item dispatch loop (16 launches/layer at c=16 → 1). |
| `d9cb34e` | CUDA Graph capture for `unified_forward_internal`: layer loop + final rms_norm + per-item copy_slice + lm_head wrapped in a graph. Per-shape cache keyed by `(m_total, num_seqs)`. |
| `f1f0f0b` | Bug fix #4: missing post-capture replay. Capture only RECORDS launches; without an immediate replay the layer kernels never executed and downstream ops read garbage state. |
| `5f5166d` | Tighten env gate to `=1` (was `!= "0"`); document failure mode. |
| `1cde3af` | **Critical graph fix:** `with_marlin_gather_scratch` does an in-place `stream.alloc::<f16>` when its slot is too small. cuLaunchKernel after a runtime alloc inside a captured stream fails with `CUDA_ERROR_INVALID_VALUE`. New `Backend::pregrow_marlin_gather_scratch(ctx, m_total * intermediate_size)` called eagerly before `begin_capture` — pre-allocates the worst-case slot OUTSIDE capture. |
| `8b28adf` | Status doc: graph capture confirmed working at 714 tok/s. |
| `7561af2` | rustfmt over the four touched files. |

## Per-op time breakdown (`FERRUM_DECODE_OP_PROFILE=1`, c=16 steady)

Drain dump from `unified_forward_internal` (per 64 calls = 1 unified_forward × 32 layers):

| Op | Calls | Avg/call | Total/iter | % |
|---|---|---|---|---|
| matmul (qkv/o/gate_up/down × 4 per layer) | 128 | 70-99 us | 9-13 ms | 60-65% |
| paged_varlen_attention | 32 | 96-126 us | 3-4 ms | 20% |
| split_qkv_norm_rope_paged (varlen, 1 per layer) | 32 | 53-64 us | 1.7 ms | 8% |
| rms_norm + fused_add_rms | 64 | 8-11 us | 0.6 ms | 4% |
| silu/residual_add | 64 | 7-9 us | 0.5 ms | 3% |

Iter total 17-21 ms. With graph capture: 18.3 ms steady.

vLLM TPOT 7.4 ms — gap is mostly in:
- matmul kernel overhead (each call ~70-100 us at m=16; vLLM measured ~50 us)
- attention kernel time (vLLM has FlashInfer paged-attn with persistent threads)

## Bench JSONs

All under `/tmp/paged_bench_run/` on the pod:
- `bench_c16_r{1,2,3}.json` — per-rep vllm bench output

## Configuration

```
CUDA_VISIBLE_DEVICES=0
FERRUM_KV_CAPACITY=2048
FERRUM_PAGED_MAX_SEQS=32
FERRUM_METAL_PAGED_KV=1        # cross-backend env, opts paged on
FERRUM_UNIFIED_GRAPH=1         # opts CUDA Graph capture on
```

VRAM headroom: ~10 GB used of 24 GB (5 GB weights + 4 GB paged pool + 1 GB scratch).

## Known gaps vs vLLM (next-session ROI ranking)

1. **paged_varlen_attention FlashDecoding rewrite** (~3-4 ms/iter = 20% — rewrite with split-K + persistent threads + Q-in-shared-mem could halve it; +10% net throughput).
2. **Mixed-batch chunked prefill in engine** — current Step 3b minimal sends single-item unified batches. The full design (engine emits one UnifiedBatch with prefill chunks + decode tokens in same iter) predicts another +20-40%.
3. **Marlin tile per-m tune** — m=16 already bandwidth-bound at ~24% peak per existing analysis; low ROI without persistent kernel design.
4. **Higher-concurrency tune (c=32, c=64)** — graph capture has more headroom when more launches per iter; pool sizing already matches `FERRUM_PAGED_MAX_SEQS`.

## Validation

- Single-prompt smoke (paged + graph capture):
  > "In a moonlit forest, a cunning fox named Rusty led his pack on a midnight hunt, their bushy tails bobbing behind them like lanterns. As they stalked their prey, Rust"
  Same output as eager — confirms graph-replay is functionally identical to eager record path.

- 3-rep c=16 bench survives without VRAM exhaustion (cache_id leak fix verified). Server log clean (no panics, no `paged KV pool exhausted` messages).

- vLLM ground truth: same pod, same model, same client invocation, 1986 tok/s sustained.
