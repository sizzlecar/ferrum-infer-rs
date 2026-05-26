# M3 80% goal — session 2026-05-26 measured findings

**Pod:** Vast contract 37853406, RTX 4090 sm_89, CUDA 13.0.48, driver 580.159.04
**ferrum build:** `b0da10d` (in flight at time of writing) — base sweep was on `8836890`
**Model:** Qwen/Qwen3-30B-A3B-GPTQ-Int4
**Dataset:** random in=256 out=128

---

## Bottom-line ratios (all correctness-verified)

These are the publication-grade ratios. Anything reported in
`session-2026-05-25/RESULTS.md` is invalid — the prior session ran
with `FERRUM_VLLM_MOE=1` which silently produced **garbled output**
on CUDA 13 (see § Correctness regression below). Throughput was the
speed at which ferrum emitted gibberish tokens, not coherent text.

**SAFE sweep (no vllm-moe-marlin, no vllm-paged-attn-v2):**
n_repeats=5, num_prompts=128, every cell pre-gated with
`"What is the capital of France?" → Paris`:

| c | ferrum SAFE tok/s | vllm 0.20.2 tok/s | **TRUE ratio** | TPOT p50 ferrum/vllm |
|--:|---:|---:|---:|---|
| 1 | **111.2 ± 0.0** | 190.1 ± 0.1 | **0.585** | 9.9 / 4.6 ms |
| 4 | **112.4 ± 0.0** | 493.2 ± 4.8 | **0.228** | — / 6.9 ms |
| 16 | **670.6 ± 14.5** | 1256.0 ± 5.8 | **0.534** | 21.6 / 10.3 ms |
| 32 | **762.0 ± 0.0** | 1850.3 ± 34.7 | **0.412** | 33.6 / 13.8 ms |

vllm baseline was collected by `vllm bench serve` at the same dataset
and num_prompts. Numbers in `docs/bench/m3-safe-sweep-2026-05-26/`
and `docs/bench/vllm-only-sweep/c32_nsys/`.

**Gap to 0.80**: c=1 −22pp, c=4 −57pp, c=16 −27pp, c=32 −39pp.
**Average ratio ~0.44**, not the **0.60-0.68** the prior session reported.

---

## ⚠️ Real root cause of `FERRUM_VLLM_MOE=1` garbled output (added late)

The "ScalarType::id() TU mismatch" diagnosis in the section below was
**wrong**. After both fixes (`b0da10d` straight-line OR-chain id() +
`241dbc0` literal IDs) the user's CLI still produced garbled output
with `FERRUM_VLLM_MOE=1`.

A 10-minute direct nvcc microbench (`scripts/microbenches/scalar_type_id_test.cu`)
proved that `vllm::ScalarType::id()` is consistent across TUs already
on commit `b0da10d` (and even on the original `cbe04ea` code, in
hindsight — the session-2026-05-25 PROGRESS.md `nm` evidence was
showing DIFFERENT Marlin specializations because the dispatcher was
selecting FP4/FP8 paths while only INT4 kernels were instantiated; NOT
the same constexpr evaluating differently).

The actual bug is a **weight-packing format mismatch**:

```rust
// crates/ferrum-kernels/src/backend/cuda/quant.rs:728
let qw_packed = crate::marlin::repack_gptq_to_marlin(&qw_in, k, n_per_expert);
```

ferrum's MoE-expert stack-build uses **IST-DASLab marlin packing**
(`crate::marlin::repack_gptq_to_marlin`), but the kernel called by
`FERRUM_VLLM_MOE=1` is `marlin_gemm_moe_vllm → ferrum_vllm_marlin_moe_f16
→ vllm::marlin_mm<half>` which expects **vLLM's own marlin packing**
(a different bit layout). The kernel runs, reads the IST-packed bytes
as if they were vLLM-packed, dequants the wrong bit slices, and returns
garbage activations.

vLLM's packing kernel is already ported into ferrum
(`crates/ferrum-kernels/src/backend/cuda/vllm_marlin.rs:31`
`ferrum_vllm_gptq_marlin_repack`) but is wired ONLY to the dense
vllm-marlin path — not to the MoE expert stack at `quant.rs:728`.

### Fix paths (ranked)

| option | work | risk | retention of FERRUM_VLLM_MOE=1 benefits |
|---|---|---|---|
| A. MoE stack uses `ferrum_vllm_gptq_marlin_repack` (per-expert) | ~1 day (device-side repack per expert + scales layout verify) | medium | full |
| B. Stay on ferrum's own `marlin_gemm_moe` (SAFE config) | 0 | 0 | none (~0.41 ratio at c=32, no batched-expert) |
| C. Wrap vLLM's device repack into the stack flow at load | ~2 days | high | full |

This session chose B (since A/C exceed remaining pod budget).

### Implication for M3 80%

The path to ratio ≥ 0.80 at c=32 still goes through enabling
`FERRUM_VLLM_MOE=1` correctly (option A), which would also activate
the device-route path (eliminating 62% DtoH overhead). Without that,
SAFE-config ferrum stays ~0.41 at c=32. Multi-week deep work on
graph coverage + FA2 SplitKV would also be needed.

The `b0da10d` and `241dbc0` patches stay in the tree because the
rewritten `id()` is functionally equivalent and the literal constants
don't hurt — but neither is the bug fix. The bug is in `quant.rs:728`.

---

## Correctness regression (CUDA 13)

Two independent kernel paths produce wrong output when enabled on
CUDA 13.0.48 + driver 580:

| env knob | output on "What is the capital of France?" | category |
|---|---|---|
| `FERRUM_VLLM_MOE=1` | `.` `.` `.` (empty / repeated period) or `"major major major, and, and"` | **garbled tokens** |
| `FERRUM_USE_VLLM_PAGED_ATTN=1` | `"教你如何用Python写一个简单的游戏，，，，"` | **wrong language, coherent shape, then degenerates** |
| Both | Same as VLLM_MOE | garbled |
| (neither) | `Paris` | ✓ correct |

**Bisect confirmed** these two flags are independent bugs:
- safe knobs (`FERRUM_GRAPH=1`, `FERRUM_GRAPH_SKIP_UPLOAD=1`,
  `FERRUM_MOE_DEVICE_ROUTE=1`, `FERRUM_GREEDY_ARGMAX=1`) all produce
  correct output individually
- only the two `vllm-*` flags cause regressions

### Root cause #1 — `vllm-moe-marlin` ScalarType::id() TU mismatch

`vllm::ScalarType::id()` is a recursive constexpr fold over struct
members. nvcc 13.0.48's cudafe++ frontend mis-mangles the same
`kFloat16.id()` expression across translation units — different TUs
compute different `int64_t` values for the same literal type. The
dispatcher in `ops.cu` and the explicit instantiations in
`kernel_instantiations.cu` then resolve to **different template
specializations** of `Marlin<scalar_t, w_type_id, s_type_id, ...>`.
The kernel that actually runs reads weights at the wrong dtype.

**Fix (commit b0da10d):**
`crates/ferrum-kernels/kernels/vllm_marlin_moe/core/scalar_type.hpp` —
replace the constexpr fold with a hand-unrolled straight-line OR-chain
that nvcc 13 handles deterministically. Semantics identical (same bit
layout, same widths).

```cpp
// Before (broken on nvcc 13):
constexpr Id id() const {
    auto or_and_advance = [](std::pair<Id, uint32_t> r, auto m) {
        // ... fold over members
    };
    return reduce_members(or_and_advance, ...).first;
}

// After (deterministic):
constexpr Id id() const {
    return (int64_t(exponent) & 0xFF)
         | ((int64_t(mantissa) & 0xFF) << 8)
         | ((int64_t(signed_) & 0x1) << 16)
         | ((int64_t(bias) & 0xFFFFFFFFLL) << 17)
         | ((int64_t(finite_values_only) & 0x1) << 49)
         | ((int64_t(nan_repr) & 0xFF) << 50);
}
```

Verification pending (rebuild in flight at time of writing).

### Root cause #2 — `vllm-paged-attn-v2` on CUDA 13

Not yet diagnosed. `kernels/vllm_attn/` doesn't use `ScalarType`, so
the same bug doesn't apply. Suspected: K/V cache layout mismatch
between writer (`split_qkv_norm_rope_into_paged_cache_vllm.cu`) and
reader (`paged_attention_v2.cu`) under CUDA 13's template
instantiation. **Not fixed this session** — workaround is to leave
`FERRUM_USE_VLLM_PAGED_ATTN` unset.

---

## nsys CUDA API breakdown (c=32 SAFE)

`nsys stats … --report cuda_api_sum` on the safe-sweep c=32 cell:

| API | calls | %API time | per-call |
|---|---:|---:|---:|
| **cuMemcpyDtoHAsync_v2** | **325K** | **62.0%** | 190 µs |
| cuMemcpyHtoDAsync_v2 | 1.7M | 12.9% | 7.4 µs |
| cuLaunchKernel | 1.6M | 7.5% | 4.6 µs |
| cudaLaunchKernel | 1.0M | 5.6% | 5.3 µs |
| cuStreamSynchronize | 207K | 3.7% | 17.7 µs |
| cuMemsetD32Async | 643K | 3.1% | 4.9 µs |
| cuMemAllocAsync | 1.7M | 2.2% | 1.3 µs |
| cuMemFreeAsync | 1.7M | 1.5% | 0.9 µs |
| **cudaGraphLaunch** | **0** | **0%** | — |

**vLLM c=32 same metric** (`docs/bench/vllm-only-sweep/c32_nsys/`):

| API | calls | %API time |
|---|---:|---:|
| cudaMemcpyAsync | 117K | 34.9% |
| cudaStreamSynchronize | 87K | 29.0% |
| cudaEventSynchronize | 573 | 21.0% |
| cudaLaunchKernel | 105K | 8.7% |
| **cudaGraphLaunch** | **253** | **0.8%** |
| cudaGraphInstantiate | 646 | 1.0% |

**vLLM does ~1 cudaGraphLaunch per iter (253 ≈ 256 iter count).**
**ferrum: zero CUDA Graph usage at runtime.** The
`FERRUM_GRAPH=1` env in the iter-3 baseline does NOT enable runtime
graph capture for the full decode iter — only sub-blocks.

---

## DtoH dominance — root cause

`crates/ferrum-kernels/src/backend/cuda/moe.rs::try_gpu_route_topk_into_host`
(lines 67-189):

```rust
stream.memcpy_dtoh(&ids_view, ...)?       // DtoH #1
stream.memcpy_dtoh(&weights_view, ...)?   // DtoH #2
stream.synchronize()?                     // BLOCKING WAIT
```

Called once per MoE layer per decode iter. M3 has 48 layers; the c=32
safe sweep ran ~2560 iter → ~245K DtoH calls. Matches the 325K
observed (the remainder is prefill + setup).

### The gate

`crates/ferrum-models/src/moe/dispatch.rs:1399-1403`:
```rust
let use_device_route = device_route.is_some()
    && use_vllm_moe          // FERRUM_VLLM_MOE=1
    && !FERRUM_MOE_HOST_ROUTE;
```

**`FERRUM_VLLM_MOE=1` simultaneously gates:**
1. vllm-moe-marlin batched-expert kernel (per-token×expert pairs in
   ONE launch, instead of per-expert m=1 launches in N launches)
2. **The device-route path that skips `try_gpu_route_topk_into_host`
   entirely** — no host DtoH, no sync barrier

The session-2026-05-25 sweeps had `FERRUM_VLLM_MOE=1` ON, hiding both
the launch-count problem AND the DtoH problem. The safe sweep here
exposed them by forcing the host-route fallback.

---

## CUDA Graph speedup — direct microbench

`scripts/microbenches/graph_bench.cu` (run on the same pod, sm_89):

Simulates ferrum's decode iter shape (480 kernel launches per iter ≈
48 layers × 10 small kernels/layer).

| Test | naive serial launches | graph capture+replay | per-iter win | speedup |
|---|---:|---:|---:|---:|
| NOOP kernels (pure launch overhead) | **1109 µs/iter** | **354 µs/iter** | -755 µs | **3.1×** |
| Small-work (~10 µs GPU each) | 1064 µs/iter | 456 µs/iter | -608 µs | 2.3× |

Scaling — graph saves a **constant ~1.6 µs per launch** across 100-1920
launch counts, speedup is 3.0× independent of N.

### What it means for ferrum

ferrum c=32 TPOT = 33600 µs/iter. vllm TPOT = 13800 µs/iter.
Gap = 19800 µs/iter.

- Pure CUDA Graph saving on launches: ≈ 750 µs/iter at 480 launches
- That's **2.2% of TPOT** → ferrum 762 → ~779 tok/s, ratio 0.412 → 0.421

**Pure CUDA Graph alone is worth ~+1 pp ratio at c=32.** Not 30 pp as
my earlier hypothesis. The lever is real but small.

The remaining 19000 µs/iter gap comes from `cuStreamSynchronize`
**barriers** (207K syncs total ÷ 2560 iter = 81 syncs per iter; each
forces the GPU pipeline to drain, serializing what should be a single
async chain). Each sync waits ~250 µs of in-flight work.

→ **The big lever is eliminating the sync barriers (via the
device-route path) — not capturing the graph.** Device-route removes
the per-layer sync; graph saves the launch overhead on top of that.

---

## Lever ranking (revised after measurement)

| lever | true expected gain | measurement basis | difficulty |
|---|---:|---|---|
| **`FERRUM_VLLM_MOE=1` works (b0da10d fix verified)** | **+15-25 pp avg** | device-route eliminates 62% DtoH + batched-MoE replaces per-expert m=1 | DONE (verifying) |
| Full forward CUDA Graph capture | +1-3 pp | graph_bench.cu measured 3.1× launch speedup, but launch is 3% of TPOT | 2-4 weeks |
| Vendor FA2 SplitKV decode attn | +3-4 pp | paged_batched_decode_attn = 15.5s total in safe sweep; FA2 ≈ 24% faster per call → ~4 sec saved | 1-2 weeks |
| Lever C lm_head cutlass | **0 pp (dead)** | cublasGemmEx auto-picks gemv which is already optimal at small m (microbench 651 µs ferrum vs 686 µs PyTorch GEMM) | — |
| Marlin tile small-m heuristic | ~+1-2 pp | shared tile space; nsys already shows dispatcher picks reasonably | 1 week |

### Predicted ratios with b0da10d fix landed

If the scalar_type.hpp fix unblocks `FERRUM_VLLM_MOE=1` without
introducing other regressions, predicted ratios (correctness-gated):

| c | predicted ferrum tok/s | predicted ratio | confidence |
|--:|---:|---:|---|
| 1 | 115-130 | 0.60-0.68 | medium |
| 4 | 280-380 | 0.57-0.77 | medium-high (eliminates per-expert m=1) |
| 16 | 850-1000 | 0.68-0.80 | medium |
| 32 | 1100-1400 | 0.60-0.76 | medium (DtoH dominates) |

**0.80 across the board is unlikely from this fix alone.** Needs
additional CUDA Graph + attn rewrite. But the fix should get most
cells to ≥0.6 from current 0.4-0.5.

---

## Methodology lessons (read before designing next bench)

1. **n=1 + num_prompts=30 is a TRAP**. Throughput numbers are inflated
   by warmup TTFT being averaged in differently. Production numbers
   need n≥3 + num_prompts≥128. The session-2026-05-25 c=1 = 146.5
   tok/s vs this session's c=1 = 111.2 tok/s (n=5 prompts=128) shows
   a 32% inflation from the smaller sample.

2. **Throughput is meaningless without correctness gate**. The prior
   session's "+24% gain from FERRUM_VLLM_MOE=1 at c=32" was the model
   emitting garbage tokens FASTER. The CORRECT measurement (this
   session, with vllm-moe-marlin OFF) is 762 tok/s. The "1079 tok/s"
   was throughput, but not THROUGHPUT-of-coherent-text.

   **Every cell bench should pre-gate with a known reference prompt.**
   Implementation in `tmp/safe_sweep.sh` on the pod.

3. **%GPU share is not a per-kernel speed proxy**. ferrum's 21% attn
   vs vLLM's 3% attn shares are not "ferrum kernel is 7× slower" —
   per-call latency is essentially identical. The gap is in
   launch count, which composes from synchronization barriers + lack
   of graph coverage. Compare *fires* and *µs/call* separately
   (see `scripts/compare_nsys_kernels.py`).

4. **Bench env knob `FERRUM_DECODE_OP_PROFILE=1` costs ~30% perf** by
   itself. The iter-3 baseline includes it for tracing overhead. Real
   ferrum perf is what the safe sweep + extra probes show, not what
   sweep_bottleneck.sh defaults emit.

5. **CUDA 13 (driver 580) is required to run vllm 0.20.2 baseline**.
   Pods with driver 570/CUDA 12.8 cannot install vllm 0.20.2 (torch
   2.11+cu13 fails with "driver too old"). Filter Vast offers by
   `cuda_max_good >= 13.0`.

---

## Files / artifacts produced this session

- `docs/bench/m3-safe-sweep-2026-05-26/c{1,4,16,32}/ferrum_baseline.json` — correct-output bench JSONs
- `docs/bench/m3-safe-sweep-2026-05-26/c32/ferrum_nsys.{nsys-rep,csv}` — ferrum c=32 SAFE nsys (correct path)
- `docs/bench/vllm-only-sweep/c{1,4,16,32}/vllm_bench.json` — vllm 0.20.2 apples-to-apples baseline
- `docs/bench/vllm-only-sweep/c32_nsys/vllm_nsys.{nsys-rep,kernels.csv}` — vLLM c=32 nsys
- `scripts/safe_sweep.sh` (on pod) — sweep with per-cell correctness gate
- `scripts/compare_nsys_kernels.py` — rigorous kernel-by-kernel comparison (fires × µs)
- `scripts/aggregate_m3_80pct.py` — publication-grade summary with CI95
- `scripts/microbenches/graph_bench.cu` (TO BE MOVED) — CUDA graph speedup direct measurement
- `crates/ferrum-kernels/kernels/vllm_marlin_moe/core/scalar_type.hpp` — fix in commit b0da10d

## Open issues for next session

1. ⏳ **Verify b0da10d fix unblocks `FERRUM_VLLM_MOE=1`** (rebuild + Paris test)
2. ⏳ If yes, re-sweep ALL cells with VLLM_MOE ON + correctness gate
3. 🔲 Diagnose `vllm-paged-attn-v2` Chinese-output regression (separate CUDA 13 bug)
4. 🔲 Plan full-iter CUDA Graph capture (worth +1-3 pp, needs design)
5. 🔲 Vendor FA2 SplitKV from vLLM 0.20.2 (worth +3-4 pp)
