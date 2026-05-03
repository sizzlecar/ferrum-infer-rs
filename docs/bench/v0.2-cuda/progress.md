# v0.2 CUDA Bench — Optimization Progress

Live tracking of ferrum-vs-vllm gap closure on RTX 4090, Llama-3.1-8B.
Each row is a measurable bench result on the same pod (vast.ai 36047161).

## Baseline (commit `6112ce3` — bench harness only)

Median of 3 reps. INT4 = M2 (Llama-3.1-8B-GPTQ-INT4-Marlin), FP16 = M1.

| | ferrum INT4 tok/s | TPOT_p50 | vllm INT4 tok/s | ratio |
|---|---|---|---|---|
| c=1  | 94  | 9.9 ms  | 149  | 63% |
| c=4  | 186 | 18.7 ms | 517  | 36% |
| c=16 | 292 | 50.7 ms | 1597 | 18% |
| c=32 | 0   | crash   | 2275 | 0%  |

Gap analysis (c=4 example): 12 ms TPOT delta = ~9 ms launch overhead from
per-item dispatch (qk_norm_rope, copy_slice, flash_attn, kv_append) + ~3 ms
matmul efficiency loss.

## Phase 4 (foundation): device-buffer kv_lens + stable scratch (commits `3448269`, `7c15ac7`, `5ef6736`)

Refactor of the batched kernel API to take `&Self::Buffer` for cache_lens
/ kv_lens (instead of `&[u32]` host slice). Caller writes the device
buffer via `B::write_u32` before the call. Required for any future
CUDA-graph capture replay — the kernel-arg buffer addresses must be
stable across calls.

Hidden bug exposed during testing: CudaBackend was inheriting the
trait-default no-op `write_u32`. The refactored path read zeros from
the un-written device buffer → -15% c=16 regression. Fix: override
`write_u32` in CudaBackend with `cuMemcpyHtoDAsync_v2`.

| | Phase 3 | **Phase 4** | vs Phase 3 | vs vllm |
|---|---|---|---|---|
| c=4 INT4 tok/s | 198 | **227** (227/229/225) | **+15%** | 44% (was 38%) |
| c=4 TPOT_p50 | 14.9 ms | **13.4 ms** | **-10%** | (vllm 6.84) |
| c=16 INT4 tok/s | 410 | **428** (433/433/419) | +4% | **27%** (was 26%) |
| c=16 TPOT_p50 | 29.3 ms | **27.9 ms** | -5% | (vllm 8.65) |

The c=4 +15% likely from removing the per-call `alloc_zeros` overhead;
c=16 was already alloc-overhead-irrelevant so smaller delta. Diag log
confirms all three batched paths still take effect with the new API:
```
[batched-qkr]       first call: m=4 use_batched_qkr=true
[batched-kv-append] first call: m=4 ok=true
[batched-attn]      first call: m=4 ok=true
```

**Cumulative impact since baseline (commit `6112ce3`):**
- c=4 TPOT_p50: 18.7 ms → **13.4 ms** (-28%)
- c=4 tok/s: 186 → **227** (+22%)
- c=16 tok/s: 292 → **428** (**+47%**)
- c=16 TPOT_p50: 50.7 ms → **27.9 ms** (-45%)
- c=16 vs vllm: 18% → **27%** (+9 pp)

## Phase 3: batched kv_cache_append across M caches (commit `60057e0`)

New CUDA kernel `kv_cache_append_batched_per_cache_f16` writes M items'
K-or-V into M independent caches in a single launch. Replaces the
per-item kv_cache_append + the per-item K/V copy_slice into single
buffers that fed it. Saves 3M dispatches/layer (2× kv_append + 1× copy
each for K and V; the V transpose was already being skipped).

| | Phase 2 | **Phase 3** | vs Phase 2 | vs vllm |
|---|---|---|---|---|
| c=4 INT4 tok/s | 188 | 198 (203/207/186) | +5% | 38% (was 36%) |
| c=4 TPOT_p50 | 14.7 ms | 14.9 ms | flat | (vllm 6.84) |
| c=16 INT4 tok/s | 367 | **410** (422/406/401) | **+12%** | **26%** (was 23%) |
| c=16 TPOT_p50 | 32.7 ms | **29.3 ms** | **-10%** | (vllm 8.65) |

**Cumulative impact since baseline (commit `6112ce3`):**

| | baseline | Phase 3 | delta |
|---|---|---|---|
| c=4 TPOT_p50 | 18.7 ms | 14.9 ms | **-20%** |
| c=4 tok/s | 186 | 198 | +6% |
| c=16 tok/s | 292 | **410** | **+40%** |
| c=16 TPOT_p50 | 50.7 ms | 29.3 ms | **-42%** |
| c=16 vs vllm ratio | 18% | **26%** | +8pp |

Diag log confirms all three batched paths run with `ok=true`:
```
[batched-qkr]       first call: m=4 use_batched_qkr=true
[batched-kv-append] first call: m=4 ok=true k_err=None v_err=None
[batched-attn]      first call: m=4 ok=true err=None
```

## Phase 2: batched flash_attention across M caches (commit `447b009` + symbol fix `19a0f55`)

Single CUDA kernel covers all M items' attention in one launch (replaces
M sequential `flash_attention(q_len=1)` calls and the M attn-output
copy_slice dispatches that follow). Kernel `batched_decode_attention_f16`
already existed in `kernels/batched_decode_attention.cu` — the work was
the trait wire-up + per-cache device-pointer plumbing.

Bug: first commit had the wrong kernel symbol name
(`"batched_decode_attention"` vs the actual `"batched_decode_attention_f16"`).
Caused the panic cascade on first decode — fixed in `19a0f55`.

| | Phase 1 | **Phase 2** | vs Phase 1 | vs vllm |
|---|---|---|---|---|
| c=4 INT4 tok/s | 234 | 188 (192/196/177)* | -20% | 36% |
| c=4 TPOT_p50 | 15.9 ms | 14.7 ms | -7% | (vllm 6.84) |
| c=16 INT4 tok/s | 326 | **367** (366/364/370) | **+13%** | **23%** (was 20%) |
| c=16 TPOT_p50 | 44.4 ms | **32.7 ms** | **-26%** | (vllm 8.65) |

*c=4 throughput regression is **not** a kernel issue — `[batched-attn]`
diag confirms `ok=true` on the first call, but `m=2` (not 4). The
ContinuousBatchScheduler batches at most 2 of the 4 concurrent c=4
requests per decode step, so `forward_layer_batched_decode` fires twice
serially with m=2 each instead of once with m=4. TPOT itself improved;
total throughput drops because of the extra scheduling round-trip.
Tracking separately as a scheduler-tuning item.

Cumulative impact since the no-optimization baseline (commit `6112ce3`):

| | baseline | now | delta |
|---|---|---|---|
| c=4 TPOT_p50 | 18.7 ms | 14.7 ms | **-21%** |
| c=16 tok/s | 292 | 367 | **+26%** |
| c=16 TPOT_p50 | 50.7 ms | 32.7 ms | **-36%** |

## Phase 1: batched per-item qk_norm_rope (commit `510ebad` + diag `c2e9757`)

Single CUDA kernel processes all M items' Q/K/V in one launch, with each
item's RoPE position read from a device array. Drops 3M qk_norm_rope
dispatches per layer to 3.

Diagnostic confirms `use_batched_qkr=true` on every batched_decode call.

| | ferrum INT4 tok/s | TPOT_p50 | vs baseline | vs vllm |
|---|---|---|---|---|
| c=1  | unchanged (path: decode_internal, not batched) |
| c=4  | **234** (was 186) | **15.9 ms** (was 18.7) | **+26%** | 45% (was 36%) |
| c=16 | **326** (was 292) | **44.4 ms** (was 50.7) | **+12%** | 20% (was 18%) |

Implementation details:
- New kernel `qk_norm_rope_batched_decode_f16` in
  `crates/ferrum-kernels/kernels/qk_norm_rope.cu`
- New trait `Backend::qk_norm_rope_batched_per_item` (default unsupported,
  caller falls back to per-item loop on backends that don't implement it)
- New scratch buffers `q/k/v_normed_batched` (separate output — Rust
  trait API forbids `&` ↔ `&mut` aliasing on the same buffer)
- New scratch buffer `batch_positions: u32 [max_M]` for per-item RoPE

## MAX_BATCH=32 verification (commit `1ab6ccb`)

`FERRUM_MAX_BATCH=32` env baked into `run_sweep.sh`. Diagnostic
confirms c=4 now batches at m=4 (`[batched-attn] m=4 ok=true`).
Throughput at c=4 is unchanged (184 tok/s vs 188 with m=2 batching),
which **rules out scheduler batch size as the c=4 bottleneck**. The
c=4 plateau is something else — most likely prefill-decode overlap
or per-request setup overhead. Tracking as a separate item.

## Roadmap (ranked by ROI)

Next phases to ship as separate commits, each measured against the
current Phase-1 numbers:

1. **Batched flash_attention** — kernel `launch_batched_decode_attention`
   already exists in `cuda_decode.rs`, just needs Backend trait wire-up
   and per-cache device-pointer plumbing. Expected: c=16 ~3 ms TPOT savings.
2. **Batched kv_cache_append + post-attn copy_slice** — replace m
   sequential dispatches with one batched kernel each. Expected: c=16
   ~2 ms TPOT savings.
3. **CUDA Graph capture for batched_decode** — single-item decode already
   has graph capture (`decode_internal` lines 1615-1746). Replicate for
   `decode_batch_internal`. Expected: ~5 ms TPOT savings across all m.
4. **Marlin GEMM tile/pipeline tuning for m > 1** — current Marlin call
   uses auto-tuned thread_k/thread_n; benchmark indicates room (m=16 GEMM
   ~12 ms vs theoretical floor ~5.65 ms). Expected: c=16 ~3-5 ms.
5. **c=32 cliff investigation** — c=32 r1 produces ~25 tok/s and r2-3
   produce 0 tok/s with all requests "ok". Looks like KV pool exhaustion
   or scheduler stall, not a per-token slowdown. Needs separate trace.
6. **MoE CUDA wire-up** — Qwen3-30B-A3B (M3) fails ferrum boot
   ("Architecture Unknown"). Different model family, separate work.

## Bench harness fixes shipped (out of perf scope)

1. `prompts_subset.py` emits `.jsonl` sibling — vllm 0.20 CustomDataset
   only accepts JSONL.
2. `setup.sh` uses `vllm[bench]==0.20.0` — pulls in pandas, required by
   `vllm bench serve`.
3. `run_cell.sh` adds `--save-result` — vllm bench skips the JSON write
   without it.
4. `run_cell.sh` parses vllm 0.20's flat-key result schema
   (`output_throughput`, `median_tpot_ms`, `p99_*`).
5. `run_sweep.sh` sets `FERRUM_KV_CAPACITY=2048` — default 512 panics
   on >512-token prompts (panic cascade from `scratch.residual.take()`
   leaves the model state poisoned for the rest of the run).
6. ferrum chat schema accepts OpenAI typed-parts content array
   (`[{"type":"text","text":"..."}]`) — vllm bench sends this shape.
7. ferrum streaming final chunk now includes empty `delta` so OpenAI
   clients reading `choices[0]["delta"]` unconditionally don't crash.
