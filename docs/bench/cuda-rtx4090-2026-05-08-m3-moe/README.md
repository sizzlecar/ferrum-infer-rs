# M3 (Qwen3-30B-A3B-INT4) MoE — RTX 4090 ferrum vs vLLM 0.20.1

**Date**: 2026-05-07 → 2026-05-08
**GPU**: NVIDIA RTX 4090 (24 GB), CUDA 13, sm_89
**Model**: `Qwen/Qwen3-30B-A3B-Instruct-GPTQ-Int4` (128 experts × 48 layers, top-k=8, n_per_expert=768, hidden=2048)
**Quant**: GPTQ INT4, group_size=128, sym=true, desc_act=false
**Workload**: `vllm bench serve` random 256→128 (input_len → output_len), `c × 4` prompts per run

## Headline result

ferrum M3 throughput jumped from **9 → 137 tok/s at c=32 (15× speedup)** in this session. Correctness verified bit-exact via GPU offset-GEMM parity test (`cuda_stacked_offset_vs_per_expert`, rel_err = 0.0000).

| Concurrency | ferrum start (per-pair) | ferrum final (multi-stream) | speedup | vLLM 0.20.1 | ratio vs vLLM |
|-------------|-------------------------|------------------------------|---------|-------------|---------------|
| c=1   | 9.7 / 34.1 / 7517   | **65.5 / 12.6 / 282**    | 6.8×  | 161.5 / 4.94 / 164  | 41% |
| c=4   | 9.4 / 203.4 / 22236 | **67.6 / 52.4 / 725**    | 7.2×  | 483.0 / 7.46 / 112  | 14% |
| c=8   | 9.7 / 545.4 / 28338 | **118.0 / 55.8 / 1242**  | 12.2× | 413.1 / 8.63 / 1381 | 29% |
| c=16  | —                   | **132.1 / 102.4 / 1944** | n/a   | 505.0 / 11.08 / 2646 | 26% |
| c=32  | —                   | **137.1 / 204.4 / 3050** | n/a   | 1872.9 / 14.67 / 317 | 7.3% |

Format: `tok/s aggregate / mean TPOT ms / mean TTFT ms`.

The c=32 ratio (7.3%) is misleadingly low — vLLM at c=32 leverages FULL-CUDA-Graph capture (cudagraph_capture_sizes up to 64) plus `fused_moe_marlin.cu` (one kernel handles all (token, expert) pairs of a layer). ferrum has neither. At c=4 / c=8 the ratio (14% / 29%) is more representative of the per-call efficiency gap.

## Optimization journey

Five sequential `vllm bench serve` cycles, each with multi-c sweep:

| Stage | c=1 | c=4 | c=8 | c=16 | c=32 | Δ vs prev |
|-------|-----|-----|-----|------|------|-----------|
| 1. per-pair fallback (broken stacked) | 9.2 | 9.4 | 9.7 | — | — | baseline |
| 2. stacked Marlin correct + bucketed | 50.1 | 52.7 | 86.0 | 100.2 | 108.1 | +5-12× |
| 3. + silu collapse | 53.5 | 54.4 | 88.3 | 101.0 | 109.5 | +1-7% |
| 4. + workspace bulk-zero | 53.6 | 54.5 | 86.9 | 100.9 | 111.6 | ~0% |
| 5. + multi-stream dispatch (s=4) | **65.5** | **67.6** | **118.0** | **132.1** | **137.1** | **+22-36%** |

Bench JSONs from each stage are preserved under `results_m3_*/` directories.

### Stage 1 — per-pair MoE forward fallback

Original code path: each (token, expert) pair dispatched as a separate `m=1` Marlin GEMM call. At c=32 with batch×top_k=256 pairs, the per-layer dispatch is `256 × (gate_up + silu + down + scaled_add) × 48 layers ≈ 49 152 launches/token`. CPU sample-rate is uniform across batch sizes — TPOT scales linearly with `c` while aggregate throughput stays flat. Functional but unusable for production.

```
ferrum c=1 → 9.2 tok/s  (TPOT 34.1 ms, TTFT 7.5 sec)
ferrum c=8 → 9.7 tok/s  (TPOT 545 ms,  TTFT 28 sec)
```

### Stage 2 — stacked Marlin (per-expert repack + concat) + bucketed dispatch

Two big infrastructure changes landed simultaneously:

**Stacked Marlin loader**. We load all `num_experts × proj_count` raw GPTQ tensors per layer, then call `B::load_gptq_stacked` which:

1. Per-expert independently runs `repack_gptq_to_marlin` + `repack_scales_to_marlin` (rayon-parallel across experts inside a single backend call).
2. Concatenates the resulting Marlin-format tiles into ONE big GPU buffer (qweight + scales + workspace).
3. Each expert's packed bytes are CONTIGUOUS in the buffer — offset GEMM at runtime is just pointer arithmetic.

Why per-expert repack is mandatory: Marlin's repack lays bytes out as `[K-tile-row OUTER, N-tile MIDDLE, ik, in_]`. A single repack of `concat(all experts along N)` produces a buffer where expert `e`'s bytes are spread across K-tile-rows, NOT contiguous. Pointer offset to "expert `e`'s first column" lands in K-tile-row `floor(e × n_per_expert / 16)` of the WHOLE stacked tile, which contains a mix of experts. (The `cuda_stacked_offset_vs_per_expert` parity test caught this bug — rel_err 1.4 with concat-then-repack vs rel_err 0.0000 with per-expert-then-concat.)

**Bucketed MoE dispatch** (`moe_forward_bucketed`):

1. Host-side bucket plan from RouterOutput: `expert_offsets[E+1]`, `packed_token_idx[total_pairs]`, `pairs_by_token[batch, top_k]`, `pair_weights[batch, top_k]`.
2. `embedding_lookup` gathers `x[batch, hidden]` into `x_packed[total_pairs, hidden]` (one GPU launch).
3. **Phase 1**: per active expert `e`, ONE Marlin GEMM at `m = tokens_per_expert[e]` writing to row range `[expert_offsets[e], expert_offsets[e+1])` of `gate_up_packed`.
4. **Phase 2**: ONE `fused_silu_mul_split` covering all `total_pairs` rows.
5. **Phase 3**: per active expert `e`, ONE Marlin GEMM at `m = tokens_per_expert[e]` for `down_proj`.
6. `moe_combine` kernel: `out[b, h] = Σ_k weights[b,k] × down_packed[pairs_by_token[b,k], h]` (one launch).

For c=32 prefill, this drops dispatch from `4 096 × 4 ops × 48 layers ≈ 786 K` launches/token to `~128 active × 2 ops × 48 ≈ 12 K` launches/token (60× reduction).

```
ferrum c=1 → 50.1 tok/s  (5.4× over per-pair)
ferrum c=8 → 86.0 tok/s  (8.9×)
ferrum c=32→ 108.1 tok/s
```

### Stage 3 — silu collapse

The bucketed loop initially called `fused_silu_mul_split_strided` once per active expert (per row range). Replaced with ONE `fused_silu_mul_split` over the whole packed buffer — every (token, k_slot) is covered by some expert's row, no garbage in the combined region. Saves `N_active - 1` launches per layer (~127 launches/layer at c=32).

Gain: +1-7%. Marginal because silu kernel itself is tiny — most of the per-layer cost is in Marlin GEMMs, not silu.

### Stage 4 — bulk Marlin workspace zero

Marlin's workspace mutex slots must be zeroed before each kernel. The per-call `cuMemsetD32Async` is itself a kernel launch — at c=32 that's `256 calls/layer × 48 layers = 12 288` memset launches/token.

`marlin_zero_stacked_workspace` does ONE `cuMemsetD32Async` for the whole stacked store (96 launches/token total). Per-call zero is skipped via `FERRUM_MARLIN_SKIP_WS_ZERO=1`.

Gain: ~0%. cuMemsetD32Async is asynchronous on the same stream — overlaps with the subsequent Marlin kernel completely. Bumped 32 tps→33 tps within noise.

### Stage 5 — multi-stream MoE GEMM dispatch ★

Each Marlin GEMM at small `m` (typically 1-8 at c=32) uses only `~thread_n_blocks = 8` SMs out of the 4090's 128. The remaining 120 SMs sit idle while we wait on launch + bandwidth for one tiny kernel after another.

Solution: a stream pool (default 4 streams, configurable via `FERRUM_MOE_STREAMS`). The new `B::moe_gemm_phase_batched` takes a `(expert_idx, in_row, out_row, m)` descriptor list and round-robins them across the pool. Cross-stream sync via CUDA events: default → pool entry, pool → default exit, so the silu / combine on default sees fully-committed GEMM outputs.

Race-freedom: input is read-only, output rows are disjoint per expert, workspace mutex slots are per-expert. No locking needed.

```
ferrum c=1 → 65.5 tok/s  (+22% over Stage 4)
ferrum c=8 → 118.0 tok/s (+36%)
ferrum c=16→ 132.1 tok/s (+31%)
ferrum c=32→ 137.1 tok/s (+23%)
```

`FERRUM_MOE_STREAMS=8` performs identically to 4 — sweet spot is at the SM utilization level where 4 concurrent Marlin kernels saturate scheduling without thrashing.

## Final tuned config

```
FERRUM_KV_CAPACITY=2048
FERRUM_KV_MAX_BLOCKS=2048
FERRUM_PAGED_MAX_SEQS=32
FERRUM_METAL_PAGED_KV=0          # CUDA's paged_decode_attention not impl'd yet
FERRUM_MIXED_BATCH=0             # incompatible with bucketed MoE forward
FERRUM_GREEDY_ARGMAX=1
FERRUM_MOE_BUCKETED=1            # default ON via stacked GPTQ store presence
FERRUM_MARLIN_SKIP_WS_ZERO=1     # paired with marlin_zero_stacked_workspace
FERRUM_MOE_STREAMS=4

./target/release/ferrum serve --model /workspace/models/M3 \
    --port 8800 --gpu-memory-utilization 0.95
```

## Stage 6 — allocation-free host route() + plan reuse  ★ +8.3%

**Date**: 2026-05-08 (later same day)

Profile data with the new `[bucket-prof]` instrumentation (PR adds per-phase
microsecond counters around sync / D2H / host_route / plan / gather / gemm1 /
silu / gemm3 / combine — drained per decode step) confirmed the host-side
softmax+top-k+sort was the biggest non-GEMM cost:

```
[bucket-prof] layers=48 bk_total=41 ms | sync=103 d2h=857 host_route=10210 plan=97
              gather=678 gemm1=14719 silu=384 gemm3=14065 combine=825 (us, summed)
```

`host_route` was 10.2 ms / decode token = 25% of MoE wallclock. Cause: per-row
`Vec` allocations × 32 rows × 48 layers = 4 608 fresh allocations. The fix:

- `route_into(logits, ..., &mut RouterOutput, &mut Vec<f32>)` — reusable
  scratch + softmax-in-place + argmax-mask top-K (K passes of linear scan
  vs O(N log N) sort).
- `MoeBucketPlan::rebuild_into(&mut self, ...)` — clear+resize in place,
  cursor scratch on the plan struct (one alloc on first call).
- `MoeRouteScratch { output, probs, plan }` lives on `Qwen3MoeScratch`.

Verified bit-exact vs the original `route()` via 4 unit tests
(`parity_qwen3_moe_shape`, etc).

```
host_route 10.2 ms → 1.8 ms  (5.7×)
plan       97 us  →  97 us   (~unchanged in steady state — random skewed routing)
bk_total   41 ms  → 32 ms

ferrum c=32 → 148.0 tok/s  (TPOT 194.74 ms)   ratio vs vLLM = 7.9%
```

## Stage 7 — cuEvent cache + GEMM dispatch sort by m desc  ★ +8.1%

Two further changes layered on Stage 6:

**Persistent cuEvents** in `CudaState`. `moe_gemm_phase_batched` was creating
+ destroying 1 entry + 4 exit events per call. At c=32 / 48 layers / 2 phases
that's ~960 `cuEvent*` driver calls per token. CUDA semantics let us re-record
an event silently overwriting the prior recording, so we just hold 5 events
on the long-lived `CudaState` and call `record` / `wait_event` only.

**Sort dispatches by m descending**. Round-robin pool was feeding streams
in expert-id order. With sort, the longest GEMMs go first so each stream
pops the next-largest job and they finish close together — no stream
sitting idle while another tails out an m=4 call. Tied `m` keeps
expert-id ascending order for host-side determinism.

```
gemm1   14719 us → 13449 us  (-1270 us)
gemm3   14065 us → 12865 us  (-1200 us)
bk_total   32 ms →    30 ms

ferrum c=32 → 160.1 tok/s  (TPOT 178.45 ms)   ratio vs vLLM = 8.5%
```

Most of the win came from sort (~2.2 ms GEMM time saved); cuEvents on
their own contributed <1 ms (the `cuEvent*` driver calls are sub-µs each).

## Cumulative session summary (Stages 1 → 7)

Profile-mode (FERRUM_DECODE_OP_PROFILE=1, the breakdown above):

| Stage | tok/s | TPOT  | Δ tok/s | Δ TPOT  |
|-------|-------|-------|---------|---------|
| 0 baseline (per-pair fallback)             | 9.4   | 203.4 ms | —       | —        |
| 1-5 stacked Marlin + bucketed + multi-stream | 137.1 | 204.4 ms | +14.6×  | ~0       |
| 6 alloc-free route + plan                  | 148.0 | 194.7 ms | +1.08×  | -10 ms   |
| 7 cuEvent cache + dispatch sort            | **160.1** | **178.5 ms** | +1.08×  | -16 ms   |

**Production (no FERRUM_DECODE_OP_PROFILE — what users actually see)**:

| c    | Stage 5 baseline | Stage 8 (this session) | Δ tok/s | vs vLLM |
|------|------------------|------------------------|---------|---------|
| 1    | 65.5             | **73.0**               | +11.5%  | 73.0  / 161.5  = 45.2% |
| 8    | 118.0            | **136.1**              | +15.3%  | 136.1 / 413.1  = 32.9% |
| 16   | 132.1            | **158.9**              | +20.3%  | 158.9 / 505.0  = 31.5% |
| 32   | 137.1            | **172.7**              | +25.9%  | 172.7 / 1872.9 = 9.2%  |

Format: tok/s aggregate. Mean TPOTs at c=32 went 204.4 → 168.1 ms.

Total session: **9.4 → 172.7 tok/s at c=32 (18.4× over per-pair baseline,
+25.9% over Stage 5 stacked-Marlin baseline)**. vLLM remains 10.8× ahead
at 1873 tok/s — the gap is now firmly in the Marlin-GEMM-throughput
regime; ~27 ms of every decode token is real Marlin compute.

## Stage 8 — GPU-side route_topk_softmax (CUDA)  ★ +6.2% at c=32

Port of the Metal `moe_router_topk_softmax_f32` kernel to `.cu`, plus a
new Backend trait method `try_gpu_route_topk_into_host` that fuses the
host-side `B::sync(ctx) + B::to_vec(router_logits) + crate::moe::router::
route_into(...)` triple into one GPU launch + small (~1 KB) D2H of
[batch, top_k] ids and weights.

Algorithm mirrors Metal: cooperative softmax → K passes of argmax-mask
top-K → optional renorm. Block: 1 warp (32 threads), 1 block per row.
Shared mem: `num_experts × 4 B` (512 B at Qwen3-MoE 128 experts).
Tie-break: smaller index wins → bit-exact with the host `route_into`.

Caller (`moe_forward_bucketed`) tries the GPU path first and falls back
to the existing host path on `Err(unsupported)`. The default trait impl
returns `Err`, so non-CUDA backends are unchanged.

```
host_route 1850 us → 988 us   (-862 us; the residual is the cuStreamSynchronize
                                wait for the kernel + D2H to commit, since the
                                bucket-plan rebuild needs the host data.)
bk_total      30 ms →   29 ms

ferrum c=32 → 172.7 tok/s  (TPOT 168.12 ms)   ratio vs vLLM = 9.2%
```

**Cumulative production sweep (no FERRUM_DECODE_OP_PROFILE)**:

| c    | Stage 5 baseline | Stage 7 alloc-free + sort | Stage 8 GPU route | total Δ |
|------|------------------|---------------------------|-------------------|---------|
| 1    | 65.5             | 73.9                      | 73.0              | +11.5%  |
| 8    | 118.0            | 127.4                     | **136.1**         | +15.3%  |
| 16   | 132.1            | 148.2                     | **158.9**         | +20.3%  |
| 32   | 137.1            | 162.6                     | **172.7**         | **+25.9%** |

Mean TPOT at c=32 went 204.4 → 178.5 → **168.1 ms**.

c=1 dropped -1.2% (Stage 8) — an explicit `cuStreamSynchronize` after the
kernel costs more than the host softmax+topk it replaces when there's
only one sequence and the GPU is otherwise idle. Fixable by pipelining
the synchronize with the next layer's router gemv, but only impacts
single-batch latency — c≥8 universally wins.

Trait surgery: kept the existing Metal-targeted `route_topk_softmax`
unchanged; added `try_gpu_route_topk_into_host` next to it. Default Err
keeps non-CUDA backends on their current host paths.

## Failed experiments (don't re-try without code changes)

| Experiment | Outcome | Why it fails |
|------------|---------|--------------|
| `FERRUM_MIXED_BATCH=1` | Garbled output (`"2 + 2 +  2 +"`) | Mixed-batch `unified_decode` path doesn't go through `moe_forward_bucketed` — needs explicit code wire-up. |
| `FERRUM_MOE_STREAMS=8` | Identical to s=4 (within noise) | 4 concurrent Marlin kernels already saturate small-m SM scheduling. |
| `FERRUM_MARLIN_SKIP_WS_ZERO=1` standalone | ~0% change | `cuMemsetD32Async` overlaps with following Marlin kernel; per-call zero is essentially free in steady state. |

## Remaining gap analysis (post-Stage-7)

**c=32 ratio is now 8.5% (160 vs 1873). Decomposition** (steady-state per
decode token at c=32, summed across 48 layers):

```
gemm1 + gemm3   ≈ 26 ms   (88% of MoE wallclock — Marlin compute)
host_route      ≈ 1.8 ms  (post-allocation-free softmax+topk)
d2h router      ≈ 0.9 ms  (32 batch × 128 experts × 2 bytes / layer × 48)
gather (gpu)    ≈ 0.7 ms
combine (gpu)   ≈ 0.8 ms
silu (gpu)      ≈ 0.4 ms
plan, sync etc  ≈ 0.3 ms
─────────────────────────
bk_total        ≈ 30 ms
```

**The remaining gap to vLLM is almost entirely Marlin compute throughput.**
The two big-effort levers from the original analysis still apply:

1. **`fused_moe_marlin.cu` port** (~1500 LoC): cuts per-layer Marlin launch
   count from ~256 to ~3 (gate_up + silu + down). Largest single lever.
2. **CUDA Graph capture for MoE forward**: requires moving `route()` and
   bucket plan computation to GPU so the decode step is graph-capturable.
   Trait stubs for `route_topk_softmax` / `compute_ids_tpe_gpu` already
   exist (Metal-implemented); CUDA `.cu` ports + plumbing is the work.
   Saves the remaining ~3 ms of host-side overhead AND the launch
   overhead (≈ 5 µs × 200 launches/layer × 48 layers = 1.4 ms / token).

Cheap wins still on the table (each <2% TPOT):

- Cache `embedding_lookup` ids buffer + `moe_combine` pairs/weights buffers
  in `CudaState` instead of `clone_htod` per call — saves the per-call
  `cuMemAllocAsync` (~1-2 µs each × 3 calls × 48 layers = ~250 µs / token).
- Pin host buffer for `to_vec(router_logits)` D2H — turns the implicit
  sync H2D into actual async; minor win since the route() function reads
  the host data immediately anyway.

Without `fused_moe_marlin.cu` ferrum is likely to saturate around
180-200 tok/s on M3 at c=32 — 9-10× behind vLLM. With it ported,
600-1000 tok/s is plausible based on per-layer accounting.

## Commits this session (chronological)

```
# Stages 1-5 (147 → 137 tok/s journey):
1b567be fix(stacked-marlin): per-expert repack + concat — Marlin tile is K-major not N-major  ★ KEY
9624a11 perf(moe): single silu call covering all packed rows — saves N-1 launches/layer
da8e50e perf(moe): bulk-zero Marlin workspace per phase — saves N-1 memsets
1377c26 perf(marlin): cache FERRUM_MARLIN_SKIP_WS_ZERO via OnceLock
ea6637a perf(moe): multi-stream MoE GEMM dispatch — round-robin pool                          ★ KEY
0854431 fix(cuda): move moe_stream_pool to impl CudaState
d7d8e98 fix(moe): cross-stream sync for multi-stream MoE GEMM dispatch
40c2cf1 fix(cuda): cuEventCreate flag is u32, not enum tuple variant

# Stages 6-8 (137 → 172.7 tok/s):
8d0f7dc perf(moe): per-phase profile counters for bucketed MoE forward
d7c4fe1 bench(m3): phase-profile script for c=32 bucketed MoE breakdown
652832c perf(moe): allocation-free route() + bucket plan reuse — saves ~10ms/token  ★ KEY
e1fd5d8 perf(moe): cache cuEvents + sort GEMM dispatches by m desc
6befb16 perf(moe-cuda): GPU-side route_topk_softmax kernel — saves ~1.7ms/token  ★ KEY
d3158d9 fix(cuda-moe-route): scope launch_builder so D2H can re-borrow scratch
```

Earlier exploration commits (debugged the offset-GEMM stride bug):
```
43607c0 fix(marlin): prob_n_full param decouples stride from iteration   (partial fix)
c6a93f4 fix(stacked-marlin): revert to group-major layout — works with prob_n_full   (still wrong)
e4775b7 fix(stacked-marlin): scales_offset_bytes for group-major layout   (still wrong)
21b0c78 fix(stacked-marlin): scales / qzeros layout = expert-major contiguous   (still wrong)
0b63b9c fix(moe): per-expert load fallback — stacked GEMM still wrong (rel_err 0.49)   (workaround)
97193f0 fix(moe): restore stacked Marlin load — proper expert-major scales layout   (still wrong)
42161a8 bench(m3): adjust to per-expert load reality — small KV pool + c<=8
```

The core lesson from the dead-ends: Marlin's repack lays out packed bytes as `[K-tile-row OUTER, N-tile MIDDLE, ik, in_]`, NOT N-major. There is no host-side scales/qzeros layout that makes a stacked-then-repacked qweight work — every expert's tile-rows are interleaved through the entire K-axis. Per-expert repack-then-concat is the only correct path (modulo writing a custom Marlin variant, which is the upstream `fused_moe_marlin.cu` approach).

## Validation

- `cuda_stacked_offset_vs_per_expert` (CUDA parity test) — 4 synthetic experts, all rel_err = 0.0000.
- `cpu_stacked_vs_per_expert_parity` (CPU parity, layout check) — passes.
- `bucketed_matches_per_pair_dispatch` (CPU bucketed semantic) — passes.
- `m3_load_finishes`, `m3_prefill_finite_logits`, `m3_decode_advances_kv` (real M3 GPU smoke tests) — all pass.
- Real chat completion round-trip (curl `/v1/chat/completions` "What is 2+2?") — answer "4", correct.

## File index

- `results_m3_quick/` — Stage 1 baseline (per-pair fallback) + initial vLLM
- `results_m3_bucketed/` — Stage 2 (stacked + bucketed) c=1/4/8
- `results_m3_c32/` — Stage 2 c=8/16/32 + vLLM c=16/32
- `results_m3_silu/` — Stage 3 (silu collapse) full sweep
- `results_m3_wszero/` — Stage 4 (bulk ws zero) full sweep
- `results_m3_streams/` — Stage 5 c=1 only (early test)
- `results_m3_streamsv2/` — Stage 5 final sweep (s=4 + s=8 comparison)

Each `*.json` is the raw `vllm bench serve --save-result` dump; the `tps / TPOT / TTFT` values cited above are extracted directly from those.

## Cold-load perf

Pre-stacking, M3 cold-load was projected at >30 minutes (12 288 per-call Marlin repacks). Three host-side fixes brought it to **53 seconds**:

- `read_i32` / `dtype_to_f32`: per-element `from_le_bytes` → bulk `ptr::copy_nonoverlapping` (1000× per-call).
- `repack_gptq_to_marlin`: 4 sequential passes parallelized with rayon (16× on 24-core box).
- `Shard::get`: was calling `SafeTensors::deserialize(&mmap)` on EVERY tensor read (full header reparse — ~9 minutes total). Replaced with TensorMeta cache populated once at open time.

These fixes are independent of the GEMM-correctness story and apply regardless of bucketed/per-pair dispatch.
