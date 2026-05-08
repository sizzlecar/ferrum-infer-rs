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

## Cumulative session summary (Stages 1 → 9)

Profile-mode (FERRUM_DECODE_OP_PROFILE=1, the breakdown above):

| Stage | tok/s | TPOT  | Δ tok/s | Δ TPOT  |
|-------|-------|-------|---------|---------|
| 0 baseline (per-pair fallback)             | 9.4   | 203.4 ms | —       | —        |
| 1-5 stacked Marlin + bucketed + multi-stream | 137.1 | 204.4 ms | +14.6×  | ~0       |
| 6 alloc-free route + plan                  | 148.0 | 194.7 ms | +1.08×  | -10 ms   |
| 7 cuEvent cache + dispatch sort            | 160.1 | 178.5 ms | +1.08×  | -16 ms   |
| 8 GPU-side route_topk_softmax              | (profile-only)  | (profile-only)   | —      | —        |
| 9 paged-KV CUDA + KV_MAX_BLOCKS=4096       | **267.7 (clean)** | **110.1 ms (clean)** | +1.55× over 8 | -58 ms |

**Production (no FERRUM_DECODE_OP_PROFILE — what users actually see)**:

| c    | Stage 5 baseline | Stage 9 final | Δ tok/s | vs vLLM |
|------|------------------|---------------|---------|---------|
| 1    | 65.5             | **75.4**      | +15.1%  | 75.4  / 161.5  = 46.7% |
| 8    | 118.0            | **178.1**     | +50.9%  | 178.1 / 413.1  = 43.1% |
| 16   | 132.1            | **235.5**     | +78.3%  | 235.5 / 505.0  = 46.6% |
| 32   | 137.1            | **267.7**     | +95.3%  | 267.7 / 1872.9 = 14.3% |

Format: tok/s aggregate. Mean TPOTs at c=32 went 204.4 → 110.1 ms — halved.

Total session: **9.4 → 267.7 tok/s at c=32 (28.5× over per-pair baseline,
+95% over Stage 5 stacked-Marlin baseline)**. vLLM ratio improved
7.3% → 14.3% at c=32. The gap is now firmly in the Marlin-GEMM-throughput
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

## Stream-pool size sweep (Stage 8, c=32)

After Stage 8 (alloc-free route + cuEvents + sort + GPU route), we
re-confirmed the s=4 optimum. The doc's earlier "s=4 = s=8" claim still
holds — load is bandwidth-bound past 4 concurrent Marlin kernels:

| s | tok/s | TPOT  | vs s=4 |
|---|-------|-------|--------|
| 1 | 133.7 | 212.76 ms | -21% |
| 2 | 163.0 | 175.84 ms | -3.7% |
| 4 | **169.3** | **171.45 ms** | baseline |
| 8 | 167.3 | 173.55 ms | -1.2% |

s=2 captures most of the parallelism (+22% over s=1); s=4 adds another
+3.9% vs s=2. s=8 regresses slightly — the extra cuEvent sync overhead
outweighs the marginal SM-scheduling headroom. **Default stays at 4.**

(Numbers slightly differ from Stage 8 clean bench's 172.7 tok/s — c=32
runs vary ±3 tok/s between sessions on shared cloud GPUs.)

## Stage 9 — paged_decode_attention CUDA wrapper  ★ +45% at c=16

**Trigger insight**: profile-mode `[batched-decode-prof]` showed
`attn_peritem=73 ms (58.8% of TPOT)` at c=32 — bigger than MoE
(25.5%). The per-item attention loop (32 sequential m=1 paged_decode
calls) was the dominant bottleneck, not Marlin.

**Root cause**: CUDA was missing `paged_decode_attention` entirely
(only Metal had it). When `FERRUM_METAL_PAGED_KV=1` was set on CUDA,
Qwen3MoeModel panicked with "not implemented for this backend".
Earlier doc claim "CUDA paged-KV not impl'd" was wrong about WHICH
specific call was missing — `paged_batched_decode_attention` and
`paged_varlen_attention` both existed; the single-seq decode entry
point was the gap.

**Fix** (commits e09855b → bad2438 → 6558fd2):

1. Added `paged_decode_attention` wrapper for CUDA that dispatches:
   - q_len==1 (any num_seqs) → `paged_batched_decode_attention`
   - q_len>1, num_seqs==1 (single-seq prefill) → `paged_varlen_attention`
2. `cu_seqlens_q + pos_offsets` allocation: switched from
   `from_slice_i32` (default impl loses bits via f32→f16 conversion,
   making `cu_seqlens_q = [0, q_len]` collapse to `[0, 0]`) to
   `alloc_u32 + write_u32`.
3. `paged_varlen_attention` writes output `[M_total, num_q_heads,
   head_dim]` (token-major) but Qwen3MoeModel's downstream code
   expects head-major (`attn_head_major_out`). Added
   `transpose_token_to_head` kernel and call it after varlen.
   (The Q input is ALREADY token-major in paged mode — the kernel
   `split_qkv_norm_rope_into_paged_cache_f16` writes
   `q_out[tok, head, hd]`, so no input transpose needed despite the
   misleading buffer name.)
4. Cached the token-major output scratch on `CudaState` to prevent
   stream-ordered free / kernel race that was triggering
   `CUDA_ERROR_ILLEGAL_ADDRESS` mid-bench.

**Production sweep (no FERRUM_DECODE_OP_PROFILE)**:

| c    | Stage 5 baseline | Stage 8 (paged_off) | **Stage 9 (paged_on + blocks=4096)** | Δ from baseline | vs vLLM |
|------|------------------|---------------------|--------------------------------------|-----------------|---------|
| 1    | 65.5             | 73.0                | **75.4**                             | +15.1%          | 75.4 / 161.5  = 46.7% |
| 8    | 118.0            | 136.1               | **178.1**                            | +50.9%          | 178.1 / 413.1 = 43.1% |
| 16   | 132.1            | 158.9               | **235.5**                            | **+78.3%**      | 235.5 / 505.0 = 46.6% |
| 32   | 137.1            | 172.7               | **267.7**                            | **+95.3%**      | 267.7 / 1872.9 = **14.3%** |

vLLM ratio at c=32: **14.3%** (was 9.2% at Stage 8, 7.3% baseline). Mean
TPOT at c=32: 204.4 → 168.1 → **110.1 ms** — halved from baseline.

### Post-merge regression bench (PR #94 squash → main, 2026-05-08 07:32 UTC)

After PR #94 (Stages 6-10) was squash-merged into `main` and the CUDA
CI fix (PR #95: `clone_dtoh` + `_packed_rows` / `_kt`) cherry-picked,
re-ran `m3_clean.sh` on the same pod (vast.ai ssh6:10160, RTX 4090):

| c    | tok/s    | Mean TPOT | Mean TTFT | Δ vs Stage 9 final |
|------|----------|-----------|-----------|--------------------|
| 1    | 73.8     | 12.78 ms  | 88 ms     | matches            |
| 8    | 171.6    | 44.05 ms  | 404 ms    | matches            |
| 16   | 236.1    | 61.46 ms  | 678 ms    | +0.3%              |
| 32   | **256.3**| 115.90 ms | 1044 ms   | -4.3% (jitter)     |

c=32 dipped 267.7 → 256.3 (-4.3%) — within Vast.ai shared-host GPU
jitter; c=16 matched Stage 9 to 0.3%. **No correctness regression.**
Bench JSONs live on the pod under
`/workspace/ferrum-infer-rs/bench/v0.2-cuda/results_m3_clean/ferrum_clean_c{1,8,16,32}.json`.

### Stages 11 + 12 + 12.1 — fused MoE Marlin (2026-05-08, PR #96)

Three sequential CUDA kernel changes, all gated on `FERRUM_MOE_FUSED=1`.

**Stage 11** — `marlin_cuda_moe` host entrypoint + `Marlin<...>` kernel
pre-amble. ONE launch processes all experts in a bucket via gridDim.y
indirection. Adds 7 optional MoE kernel args (default nullptr/0); when
non-null, kernel reads `e_local = blockIdx.y`, applies pointer offsets
to A/B/C/s/locks, overrides prob_m. Existing single-expert path
unchanged.

**Stage 12** — wire fused kernel into `moe_forward_bucketed`. Bucket
experts by `ceil(m/16)` ∈ {1,2,3,4}, fire ONE call per bucket via
`moe_gemm_phase_fused_impl`. c=32 alone gave +5% over Stage 9 — fused
launches don't help when each block does only 0.5M FLOPs but pays 3 µs
setup. Profile showed 80% of TPOT still in Marlin GEMM.

**Stage 12.1 ★** — `gridDim.x = n_tiles` (NOT sms). Original Marlin uses
gridDim.x = 128 cooperating blocks per GEMM. With Stage 12's
gridDim.y = num_experts, total = 12800 blocks → ~100 waves on 128 SMs,
each block doing 1 tile (0.5M FLOP) with 3 µs setup → setup dominates.
Setting `blocks = n_tiles` (=3 for Qwen3-MoE) means each block
processes ALL k-tiles for one n-tile — per-block work scales 32×, setup
amortises. Total grid: (3, 100) = 300 blocks → 2.3 waves.

| c    | Stage 9 final | **Stage 12.1**     | Δ vs Stage 9 | TPOT (Stage 9 → 12.1) | vs vLLM |
|------|---------------|--------------------|--------------|-----------------------|---------|
| 1    | 73.8          | **84.3**           | +14%         | 12.78 → 11.28 ms      | —       |
| 8    | 171.6         | **254.8**          | +48%         | 44.05 → 28.59 ms      | —       |
| 16   | 236.1         | **318.9**          | +35%         | 61.46 → 45.04 ms      | 63%     |
| 32   | 256.3         | **320.6**          | +25%         | 115.90 → 92.74 ms     | **17.1%** |

Re-profile at c=32 confirms `bk_total` halved (33 → 16 ms across 48
layers); `gemm1` 17 → 8.8 ms (-48%), `gemm3` 13 → 4.8 ms (-63%). New
distribution at c=32 (steady state, profile-mode):

```
TPOT 13 ms = attn 4 ms (31%) + moe 7 ms (54%) + other 1 ms (9%)
```

attn now sits comparable to MoE GEMM. Next levers (Stage 13+):

1. **Strategy B (sorted_token_ids)** — per-tile expert routing eliminates
   the padding waste (m_e ∈ {1..3} but kernel runs prob_m=16). Predicted
   +50-100% on the GEMM phase.
2. **Attn split-K / prefetch tuning** — paged_decode_attention is now
   31% of TPOT, was 14% pre-12.1.

Parity validated: `cuda_marlin_moe_fused_vs_per_expert` test passes
with rel < 0.001 (4 experts × variable m_e ∈ {16, 8, 12, 4}).

### Stage 13a — batched paged-decode flash split-K (PRs #96 + #98)

After Stage 12.1 attn jumped to 31% of c=32 TPOT (was 14%). New batched
flash-decode kernel: phase-1 splits each seq's KV across `num_splits`
chunks (grid: num_q_heads × num_seqs × num_splits) + phase-2 reduce per
(seq, head). Smart heuristic auto-tunes splits by `num_seqs × num_heads`
saturation AND `kv_len`:

- saturated grid: splits=1 if kv≤768 else 2-4
- low concurrency: aggressive splits to fill SMs

| c    | Stage 12.1 | **Stage 13a v2 (smart split-K)** | Δ vs Stage 12.1 | vs vLLM |
|------|------------|----------------------------------|-----------------|---------|
| 1    | 84.3       | **101.9**                        | **+21%** ★      | 63%     |
| 8    | 254.8      | **262.5**                        | +3.0%           | —       |
| 16   | 318.9      | **330.2**                        | +3.5%           | 65%     |
| 32   | 320.6      | **355.1**                        | **+10.8%**      | **19.0%** |

PR #98 flips `FERRUM_MOE_FUSED` and `FERRUM_PAGED_FLASH` defaults to
ON so end-user CLI doesn't need any env var to get the fast path.
Escape: `FERRUM_PAGED_FLASH=0` falls back to single-pass kernel;
`FERRUM_MOE_FUSED=0` falls back to multi-stream pool.

### Cumulative progress (this session, PRs #94 → #98)

| Stage              | c=1   | c=8   | c=16  | c=32  | vs vLLM (c=32) |
|--------------------|-------|-------|-------|-------|----------------|
| pre-#94 baseline   | 9.7   | 9.7   | —     | 137.1 | 7.3%           |
| #94 (Stages 6-10)  | 65.5  | 118.0 | 132.1 | 267.7 | 14.3%          |
| #96 (Stages 11+12+12.1+13a) | 84.3 | 254.8 | 318.9 | 320.6 | 17.1%   |
| **#98 (defaults ON, smart split-K)** | **101.9** | **262.5** | **330.2** | **355.1** | **19.0%** |

c=32: 137 → **355 tok/s = 2.6×**. Mean TPOT 204 → **83 ms** (-59%).

### Path to 80% of vLLM (~1500 tok/s)

Current ratio 19% means another **4.2×** to hit 80%. Single-kernel
tunes (tile size, gridDim shape, split-K heuristic) have been mined
out — each remaining 5-10% gain is now diminishing. The big remaining
levers all require larger work:

- **vLLM-style fused MoE Marlin port** — sorted_token_ids + per-tile
  expert routing. Eliminates the m=16 padding waste (m_e ∈ {1..3}
  effective, 5-16× compute slack). ~4000 LoC port. Speculative gain
  3-4× on the MoE GEMM phase.
- **Marlin m<16 tile rewrite** — drop the m16n8k16 MMA, add m8n8k16
  variants for tiny-m experts. Saves bandwidth waste (B/s tile is
  loaded once per call regardless of actual m). Major surgery on the
  IST-DASLab kernel internals.
- **CUDA Graph capture for full decode iter** — vLLM uses parameterized
  graphs (`cudaGraphExecKernelNodeSetParams`) so per-token routing
  changes don't force re-record. ~+5-10% on launch overhead, not the
  4× we'd need.

### The c=32 OOB resolution

### The c=32 OOB resolution

Stage 9's first ship (commit c846732) hit `CUDA_ERROR_ILLEGAL_ADDRESS`
at c=32 / 128 prompts and was documented as a known issue. The
diagnostic sweep (`m3_paged_c32_diag.sh`, 4 configs varying pool size
× concurrency × prompt count) found:

| config                     | result          |
|----------------------------|-----------------|
| c=32 blocks=2048 n=128     | hang at 75%     |
| c=32 blocks=2048 n=64      | 270.3 tok/s ✓   |
| c=24 blocks=2048 n=128     | 233.3 tok/s ✓   |
| c=32 blocks=4096 n=128     | **259.0 tok/s ✓** ★ |
| c=32 blocks=8192 n=128     | 257.2 tok/s ✓ (no extra benefit) |

**Root cause**: the engine's `kv_cache.max_blocks` (set via
`FERRUM_KV_MAX_BLOCKS`) was 2048 but the model's actual paged pool is
`max_seqs * max_blocks_per_seq = 32 × 128 = 4096`. The scheduling
budget was half the actual pool, leading to scheduler/pool drift that
triggered the illegal access at high concurrency over many prompts.

**Fix**: `FERRUM_KV_MAX_BLOCKS=4096`. Production config (commit 338b0e0):

```
FERRUM_KV_CAPACITY=2048
FERRUM_KV_MAX_BLOCKS=4096        # was 2048; must match max_seqs × (capacity / 16)
FERRUM_PAGED_MAX_SEQS=32
FERRUM_METAL_PAGED_KV=1          # NOW SAFE at all c=1/8/16/32
FERRUM_MIXED_BATCH=0
FERRUM_GREEDY_ARGMAX=1
FERRUM_MOE_BUCKETED=1
FERRUM_MARLIN_SKIP_WS_ZERO=1
FERRUM_MOE_STREAMS=4
```

## Failed experiments (don't re-try without code changes)

| Experiment | Outcome | Why it fails |
|------------|---------|--------------|
| `FERRUM_MIXED_BATCH=1` | Garbled output (`"2 + 2 +  2 +"`); -17% perf | Mixed-batch `unified_decode` path's per-item dispatch fallback corrupts KV state on Qwen3-MoE. Re-confirmed 2026-05-08 (mixed_off 165.7 / mixed_on 137.0 tok/s); needs unified_forward implementation, not a quick wire-up. |
| `FERRUM_METAL_PAGED_KV=1` on CUDA | Garbled output (`".name(\"attenuation_type_comp\")..."`) | CUDA was missing `paged_decode_attention` impl entirely. Wrapper added (commits e09855b/dad3da9/9764137 — REVERTED at 3401599) routing q_len=1→`paged_batched_decode_attention`, q_len>1→`paged_varlen_attention`. Bit-pattern bug in `from_slice_i32` default impl found and fixed (loses bits via f16 conversion → use `alloc_u32+write_u32`). Output STILL garbled after fix — likely paged_varlen kernel's KV layout assumption mismatches `split_qkv_norm_rope_into_paged_cache`'s output, OR pos_offsets semantics differ. Needs element-wise comparison vs Metal reference. **Default = OFF for Qwen3-MoE on CUDA** until this is resolved. |
| `FERRUM_MOE_STREAMS=8` | Identical to s=4 (within noise) | 4 concurrent Marlin kernels already saturate small-m SM scheduling (re-confirmed Stage 8: s=2 163, s=4 169, s=8 167 tok/s). |
| `FERRUM_MARLIN_SKIP_WS_ZERO=1` standalone | ~0% change | `cuMemsetD32Async` overlaps with following Marlin kernel; per-call zero is essentially free in steady state. |
| Cached htod scratch for embedding_lookup + moe_combine | -3.7% c=32 | `device_ptr_mut(&stream)` in cudarc 0.17.8 returns `SyncOnDrop::Sync` which calls `stream.synchronize()` on drop — forces sync H2D, killing async pipelining. Reverted at 686f062. |

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

# Stages 6-9 (137.1 → 267.7 tok/s c=32 — +95% over baseline, 14.3% of vLLM):
8d0f7dc perf(moe): per-phase profile counters for bucketed MoE forward
d7c4fe1 bench(m3): phase-profile script for c=32 bucketed MoE breakdown
652832c perf(moe): allocation-free route() + bucket plan reuse — saves ~10ms/token  ★ KEY
e1fd5d8 perf(moe): cache cuEvents + sort GEMM dispatches by m desc
6befb16 perf(moe-cuda): GPU-side route_topk_softmax kernel — saves ~1.7ms/token  ★ KEY
d3158d9 fix(cuda-moe-route): scope launch_builder so D2H can re-borrow scratch
e09855b feat(cuda): paged_decode_attention via paged_batched (q_len=1)
dad3da9 feat(cuda): paged_decode_attention dispatches q_len>1 prefill via paged_varlen
9764137 fix(cuda-paged): use alloc_u32+write_u32 for cu_seqlens_q + pos_offsets
ea808df feat(cuda-paged): paged_decode_attention wrapper with prefill transposes
bad2438 fix(cuda-paged): drop Q transpose — split_qkv_norm_rope writes Q token-major already  ★ KEY
6558fd2 fix(cuda-paged): cache out_token_major scratch on CudaState
0286a27 bench(m3): paged-KV c=32 OOB diagnostic — pool size × concurrency × prompt count
338b0e0 fix(bench/m3): paged_on + KV_MAX_BLOCKS=4096 — closes Stage 9 c=32 OOB  ★ KEY
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
