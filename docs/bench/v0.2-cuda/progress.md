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

## Phase 9 — multi-slot batched graph cache (2026-05-04, commits `fb21ebd`, `8260d76`)

Native CUDA microbench (`scripts/graph_upload_bench.cu`) confirmed
multi-slot replay is stable: 320 launches × 500 iters alternating two
graph sizes ran clean at ~0.26 ms/iter with no degradation vs single
slot, and `cuGraphUpload`-per-replay had near-zero impact (-0.014 ms
single-slot, +0.002 ms multi-slot).

Refactored `static DECODE_GRAPH: Option<GraphSlotRaw>` →
`HashMap<u64, GraphSlotRaw>`. Backend trait now takes `key: u64` on
end_graph_capture / replay_graph / reset_graph. Single-item path uses
`SINGLE_ITEM_GRAPH_KEY=0`; batched path uses `m_padded as u64`.

**Bench (M2 INT4, RTX 4090, FERRUM_BATCHED_GRAPH=1):**

| Scenario | tok/s | Comment |
|---|---|---|
| Fresh server, c=16 alone | **479.4** | matches no-graph 484; multi-slot fully functional |
| Fresh server, c=4 alone | 261.6 | matches no-graph 264 |
| c=4 → c=16 sequence (same server) | 134 (c=16) | **state pollution** — see below |

Sequence stats (c=4 → c=16 → c=4) showed only 256 batched_decode calls
during c=16 instead of the expected ~1000+ that fresh c=16 alone has,
so most c=16 work routed through the single-item decode path. Some
state set during c=4 (likely single-item warmup counter or pool-
recycled cache buffer interaction) was forcing the engine to skip
batched decode for c=16. Fresh server vs c=16 alone has no issue.

**Stability win (was the primary blocker):** SIGSEGV that killed
graph mode in earlier sessions is GONE post Phase 8. cuBLAS GEMM in
captured path was the real culprit; Marlin (perm-aware) replaced it
and graph capture+multi-replay now run clean. Native repro confirmed
the failure mode.

**Open: c=4 → c=16 cell sequence pollution.** Workaround: restart
server between cells (the bench sweep already does this for major
runs). Real fix: trace what state the c=4 cell leaks that pushes
c=16 onto single-item path. Lower priority now that graph mode is
stable + matches no-graph at clean startup.

## Phase 8 — perm-aware Marlin INT4 GEMM (2026-05-04, commits `5ba7e38`, `48bf17b`)

Recovers INT4 GEMM speed for desc_act=true models. Phase 6's
dequant→DenseLinear fix was correct but ran FP16 dense GEMM (lost
the INT4 2-3× speedup). vLLM's gptq_marlin solves this via runtime
activation gather; we land the same approach.

**Implementation:**
- `kernels/gather_columns.cu`: `output[m, j] = input[m, perm[j]]` f16
- `marlin.rs::permute_gptq_qweight_rows`: unpack INT4 → permute rows
  by `argsort(g_idx)` → repack INT4 (one-time CPU work at load)
- `MarlinWeight.perm: Option<CudaSlice<i32>>`: device-side perm buffer
- `cuda.rs::load_gptq`: detects desc_act, builds perm + permutes qweight
  before standard Marlin repack
- `cuda.rs::marlin_gemm_with_perm`: gathers input by perm into f16 scratch
  (cudarc pool alloc), then standard `marlin_gemm` on gathered input
- `native_safetensors.rs`: cfg-gates the dequant→Dense fallback to
  non-cuda only — CUDA now passes g_idx through to GptqLinear

**Math correctness:** identical output to dequant→Dense path (Pavlov
+ Schrödinger joke matches byte-for-byte). Verified at c=1 with
`temperature=0`.

**Real bench (M2, RTX 4090, content correct, 16/16 c=4, 64/64 c=16):**

| c | Phase 6 (FP16-dense) | **Phase 8 (perm-aware Marlin)** | Δ | vLLM | ratio |
|---|---|---|---|---|---|
| 1  | 50.2 tok/s | **105.3 tok/s** | +109% | 149  | 71% (was 33%) |
| 4  | 151.5 | **262.6** | +73% | 517  | 51% (was 29%) |
| 16 | 350.9 | **480.7** | +37% | 1597 | 30% (was 22%) |

TPOT_p50 (ms): c=1 19.3→8.8 (-54%), c=4 22.5→11.4 (-49%), c=16 37.1→25.6 (-31%).

Memory: ~16 GB → ~5.5 GB (INT4 weights restored).

**Why c=1 wins more than c=16**: at c=1 the GEMM is bandwidth-bound
(loading weights dominates), so INT4 (4× less DRAM) helps most. At c=16
the GEMM has higher arithmetic intensity, FP16 cuBLAS already saturates
compute well. Still, vLLM's c=16 = 1597 tok/s suggests their Marlin
batched path is more efficient — likely tile tuning + kernel-level
gather (saves 1 launch/GEMM = ~1.3ms/iter at c=16).

**Next ROI items (post Phase 8):**
1. **cuBLAS replacement: try Marlin's batched paths for non-INT4 layers**
   — actually, the only INT4 layers ARE the linear projections, and they
   ALL use Marlin now. Other ops (rms_norm, attention, etc.) are
   bandwidth-bound and well-optimised.
2. **Multi-replay graph SIGSEGV** — graph capture at c=4 would shave
   ~1ms/iter; still blocked.
3. **HTTP/SSE/tokio overhead at c=16** — measured 36% loss, lower-ROI
   but easy investigation.
4. **In-kernel gather** — fold the gather into Marlin (vLLM's approach)
   to save the scratch alloc + 1 launch per GEMM. ~2-5% win.

## Phase 7 — batch scaling diagnosis (post g_idx fix, 2026-05-04)

c=4 vs c=16 per-iter timing (FERRUM_BATCH_DECODE_PROF=1, M2 desc_act
dequant→Dense, 32 layers, RTX 4090):

| | iter (median) | vLLM TPOT_p50 | iter ratio |
|---|---|---|---|
| c=4 | **21.98 ms** | 6.84 ms | 3.21× |
| c=16 | **29.03 ms** | 8.65 ms | 3.36× |

Δ iter time c=4→c=16 = 7.05 ms for 12 more sequences = 0.59 ms/seq.

Per-iter throughput capacity:
- c=4:  4 toks / 22 ms = **182 tok/s/iter**
- c=16: 16 toks / 29 ms = **552 tok/s/iter**

Bench-measured (real harness):
- c=4:  150 tok/s (= 82% of iter capacity, 18% loss to HTTP/SSE/tokio)
- c=16: 351 tok/s (= 64% of iter capacity, 36% loss)

**Conclusion**: batch scaling at the kernel level is FINE. The c=16 ratio
worsening (2.32× vs vLLM 3.09×) is dominated by:
1. **iter is 3.2-3.4× slower than vLLM** at BOTH c=4 and c=16. Not a
   scaling problem — a per-iter speed problem. Each layer's GEMMs run
   on FP16 dense (cuBLAS) instead of perm-aware Marlin INT4.
2. **HTTP/SSE/tokio overhead doubles at c=16** (18%→36%). Likely
   per-stream channel buffering or token-serialisation cost in the SSE
   emit path. Worth investigating — but smaller ROI than the iter speed.

**ROI ranking for closing the vLLM gap (revised):**

1. **Perm-aware Marlin INT4 GEMM** — recovers desc_act models' INT4 path.
   vLLM's gptq_marlin gathers activations via `perm = argsort(g_idx)`
   in shared-mem load; Marlin kernel takes `perm` as kernel arg. Effort:
   ~1 day to integrate into ferrum's `marlin_repack` + `gptq_gemm`. Win:
   ~2-3× iter speedup on M2 / Llama-3.1-INT4 → c=16 ~700 tok/s plausible.

2. **CUDA Graph capture for batched** — 32 layers × ~10 kernel launches
   = ~300 launches/iter. Each launch ~3-5us. Total ~1-1.5ms saved.
   BLOCKED on multi-replay SIGSEGV (commit `39f813c` documents finding).

3. **HTTP/SSE optimisation at c=16** — 36% loss is excessive. Profile the
   tokio task emit pattern + see if batching SSE chunks helps.

4. **cuBLAS tile tuning for thin-tall m × H @ H × N** at m=4..16. cuBLAS
   auto-pick may not be optimal. Lower ROI than (1) but quick win.

## Phase 6 — INT4 desc_act g_idx fix (commits `bcb076b`, `e9968d0`, `e83188a`, `dad43e3`)

**Bug** (silent): all M2_* models on this pod (Llama-3.1-8B-GPTQ-INT4) have
`desc_act=true` (act-order). ferrum's gemm_gptq path (Marlin AND Triton)
ignored `g_idx` entirely — it computed group from `k/group_size` instead of
`g_idx[k]`. With desc_act the group assignment is non-monotonic, so almost
every dequantised weight used the wrong scale/zero → garbage tokens.

**Symptom**: M2 c=1..16 produced `]\n]\n]\n...` repeated ~40× across all
prompts/concurrencies. Bench harness measured *throughput of garbage*.
**ALL prior M2 numbers (94/186/235/443 tok/s) were invalid.**

**Fix** (`dad43e3`): when `quantize_config.json` has `desc_act=true`,
dequantise INT4 → f32 on CPU at LOAD time using `g_idx[k]` for scale/zero
lookup, then build `DenseLinear<B>` (cuBLAS f16 GEMM). Disk row k IS
original column k — no row permutation. Verified against vLLM exllama
math (`gptq.py:368`'s `argsort` + `gptq_shuffle` reduces to the same
y[n] = Σⱼ x[j]·dequant(qweight[j,n], scales[g_idx[j],n])).

| | ferrum c=1 | ferrum c=4 | ferrum c=16 | vLLM c=1/4/16 |
|---|---|---|---|---|
| **post-fix M2 INT4 (now FP16-dense)** | 50.2 tok/s | 150.8 tok/s | 350.9 tok/s | 149/517/1597 |
| ratio | 33% | 29% | 22% | — |
| TPOT_p50 | 19.3 ms | 22.6 ms | 37.8 ms | — |
| **content correct (16/16 c=4, 64/64 c=16)** | ✓ | ✓ | ✓ | ✓ |

The ratio dropped vs the (garbage) M1 FP16 baseline because vLLM's gptq_marlin
runs perm-aware INT4 and gets ~1.5-2× over its own FP16. ferrum now runs FP16
dense for these models (loses INT4 memory savings: 5.5 GB → ~16 GB; loses
INT4 GEMM speedup). Performance roughly matches M1 FP16 (153/361 tok/s),
which is the expected dequant→Dense ceiling.

**Future work**: perm-aware Marlin dispatch — vLLM's gptq_marlin takes
`perm = argsort(g_idx)` as a kernel arg and gathers activations in shared
memory load. Adding this to ferrum's Marlin call recovers INT4 perf for
desc_act=true models.

## Phase 5 — c=4 plateau investigation (commits `1afdd2c`, `da2f725`)

Profiling per-iter at c=4 to find where the throughput-flat-but-TPOT-down
inconsistency lives:

| Phase | Time | % |
|---|---|---|
| scheduler.next_batch | 4μs | 0% |
| process_batch (= run_batch_decode) | **10.7ms** | 96% |
| ↳ decode (forward + dtoh) | 10.2ms | 96% of run_batch_decode |
| ↳ post (logits 67μs + sample 326μs + emit 19μs) | 0.42ms | 4% |
| iter total | 10.7ms | |
| TPOT_p50 measured | 13.0ms | |
| **Gap (HTTP/SSE/tokio)** | **2.3ms** | 18% of TPOT |

**Conclusion**: engine internal is fine. Real bottleneck = decode forward
(10.2ms) which is dominated by Marlin matmuls (~9ms across 4 INT4
GEMMs × 32 layers at m=4). Marlin currently uses auto-tile (`thread_k=-1,
thread_n=-1`) which may not be optimal for m=4. **Next**: tile tuning
sweep at m=4.

Theoretical headroom:
- Current: 226 tok/s at c=4 (TPOT 13.0ms)
- If decode 10.2 → 7ms (Marlin 1.5×): TPOT ~9.3ms → ~430 tok/s (84% of vLLM)
- If decode 10.2 → 5ms (Marlin 2×): TPOT ~7.3ms → ~550 tok/s (**beats vLLM**)

## Phase 4d (CUDA Graph capture wiring) — shelved (commits `96eae06`, `b00ae94`, `156dcab`, `9b79001`, `08b0b46`)

Wires `decode_batch_internal` with begin_graph_capture / replay around
the layer loop + final norm + lm_head. Per-m_padded graph cache,
embedding_lookup stays outside (host tokens). All variable args
(positions, kv_lens_pre, kv_lens_post) hoisted to once-per-step writes
into stable scratch device buffers. cache.len bumping moved
post-forward (replay's lack of Rust execution would otherwise desync).

**Root cause (commit `9b79001`):** the batched kernels memcpy
device-pointer arrays from a STABLE host array into device scratch
inside graph capture. Each layer overwrote the SAME shared host
region. CUDA graph capture records the host POINTER, not its
contents — so on replay all 32 captured memcpys re-read the same
host slice (= layer 31's pointers), and layers 0..30 launched with
the wrong cache pointers → MISALIGNED at first sync.

Verified at the cudarc level by `cudarc_graph_shared_host_array_multi_memcpy`:
two captured memcpys reading from one Box<[u64;4]> both yield
[5,6,7,8] (the last write) on replay, proving the pattern is broken
regardless of driver version.

**Fix:** per-call `slot: usize` argument on
`kv_cache_append_batched_per_cache` and
`flash_attention_batched_per_cache`. Host arrays grow from
`BATCHED_SCRATCH_CAP` (64) to `2 * MAX_LAYERS_FOR_GRAPH * BATCHED_SCRATCH_CAP`
(8192 u64 = 64 KB each — trivial). CudaBackend uses
`host_array[slot*CAP..]` and `dev_scratch.slice(slot*CAP..)` for both
the staging memcpy and the kernel arg. Caller passes:
- K-append: `slot = li`
- V-append: `slot = li + MAX_LAYERS_FOR_GRAPH` (cache_ptrs is shared)
- flash_attn: `slot = li`

**Status: build21 in progress. To validate: rerun
`bash bench/v0.2-cuda/run_sweep.sh` with FERRUM_BATCHED_GRAPH=1, expect
non-zero throughput at c=4/c=16. Post-fix theoretical TPOT win ≈ 5 ms
(removing the per-iter Rust overhead between layers — graph dispatches
the whole forward in one go).**

## Phase 4 (foundation): device-buffer kv_lens + stable scratch (commits `3448269`, `7c15ac7`, `5ef6736`, `b00ae94`, `156dcab`)

Refactor of the batched kernel API to take `&Self::Buffer` for cache_lens
/ kv_lens (instead of `&[u32]` host slice). Caller writes the device
buffer via `B::write_u32` before the call. Required for any future
CUDA-graph capture replay — the kernel-arg buffer addresses must be
stable across calls.

Hidden bug exposed during testing: CudaBackend was inheriting the
trait-default no-op `write_u32`. The refactored path read zeros from
the un-written device buffer → -15% c=16 regression. Fix: override
`write_u32` in CudaBackend with `cuMemcpyHtoDAsync_v2`.

| | Phase 3 | **Phase 4 (post-fix)** | **Phase 4 (build13)** | vs vllm |
|---|---|---|---|---|
| c=4 INT4 tok/s | 198 | 227 | **235** (228/237/241) | **44%** (was 38%) |
| c=4 TPOT_p50 | 14.9 ms | 13.4 ms | **13.0 ms** | (vllm 6.84) |
| c=16 INT4 tok/s | 410 | 428 | **443** (446/440/442) | **27.7%** (was 26%) |
| c=16 TPOT_p50 | 29.3 ms | 27.9 ms | **27.0 ms** | (vllm 8.65) |

build13 includes the alloc_u32 sizing fix (`b00ae94`) and sync memcpy
fix (`156dcab`) — small additional gains over Phase 4 post-fix.

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
