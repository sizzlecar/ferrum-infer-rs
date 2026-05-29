# Goal: M3(Qwen3-30B-A3B-GPTQ-Int4) → 80% of vLLM throughput across c=1/4/16/32

**Status:** draft @ 2026-05-25, awaiting first sweep
**Owner:** ferrum core (this PR opens the work; subsequent PRs ship sub-targets)
**Validation harness:** `scripts/sweep_bottleneck.sh qwen3-moe-30b-int4` + `scripts/aggregate_sweep.py`

---

## Target

For **Qwen/Qwen3-30B-A3B-GPTQ-Int4** on RTX 4090 (sm_89, 24 GB, CUDA 13.x, locked clock per `scripts/lock_gpu.sh`):

| c | ferrum tok/s ≥ |
|---:|---:|
| 1 | 0.80 × vLLM_c1 |
| 4 | 0.80 × vLLM_c4 |
| 16 | 0.80 × vLLM_c16 |
| 32 | 0.80 × vLLM_c32 |

**Apples-to-apples**: ShareGPT-distribution prompts via `--dataset random --random-input-len 256 --random-output-len 128`, c × 30 requests, vLLM 0.20.2 with `--quantization gptq_marlin --no-enable-prefix-caching --enable-chunked-prefill --dtype float16 --gpu-memory-utilization 0.85`. ferrum env locked to the iter-3 set established in PR #206 (`FERRUM_GRAPH=1 FERRUM_MOE_DEVICE_ROUTE=1 FERRUM_MOE_STREAMS=4 FERRUM_GREEDY_ARGMAX=1 FERRUM_KV_MAX_BLOCKS=2048 FERRUM_PAGED_MAX_SEQS=32 FERRUM_USE_VLLM_PAGED_ATTN=1`).

## Update -- 2026-05-30 opt-in FA2 direct confirmation

The opt-in `FERRUM_FA2_DIRECT_FFI=1` path clears the 0.80× throughput target on
all four cells in a same-pod N=5 confirmation sweep. Artifact directory:
[`../cuda-rtx4090-2026-05-30-m3-80pct-confirmed/`](../cuda-rtx4090-2026-05-30-m3-80pct-confirmed/).

| c | ferrum FA2 direct tok/s | vLLM 0.20.2 tok/s | ratio |
|---:|---:|---:|---:|
| 1 | `160.4 ± 0.2` | `183.9 ± 0.2` | `0.872×` |
| 4 | `446.3 ± 7.0` | `512.5 ± 2.8` | `0.871×` |
| 16 | `1185.1 ± 12.3` | `1331.9 ± 5.7` | `0.890×` |
| 32 | `1641.9 ± 4.8` | `1972.9 ± 18.6` | `0.832×` |

This is a performance-confirmed opt-in path, not yet the default-runtime state.
It depends on a runtime shim around the vLLM/Torch FA2 extension and uses the
extra FA-compatible K/V pool. Treat it as diagnostic evidence, not an acceptable
default or final completion path. The target remains open until the same
attention win is reproduced with a source-built/Ferrum-owned FA2 wrapper or
kernel and no vLLM/Torch runtime dependency.

Follow-up source-built smoke: commits `066bd7c` / `b7defce` added a
FlashAttention-source-built shim that exports the same C ABI without linking
vLLM/Torch/Python. Remote `ldd` on `/workspace/libferrum_fa2_source_shim.so`
showed only system C/C++ libraries. Smoke
`/workspace/m3-fa2-source-shim-smoke-20260529_182244/` passed Paris and a
multi-turn gate, and c32 N=1/64 prompts measured source FA2 `1553.7 tok/s`
versus FA-layout `1310.8 tok/s` (`+18.53%`). This validates the dependency-free
direction but still is not final/default because the source shim is runtime
loaded and not yet integrated into the in-repo `ferrum-kernels` build/link path.

## Current baseline (2026-05-25 sweep)

Sweep dir: [`docs/bench/sweep-2026-05-25-1631-qwen3-moe-30b-int4/`](../sweep-2026-05-25-1631-qwen3-moe-30b-int4/) · ferrum commit `cbe04ea`, n_repeats=1, num_prompts=30, warmup=5, random in=256 out=128.

ferrum measured this session (RTX 4090, all iter-3 env knobs on):

| c | ferrum tok/s | TTFT p50 | TPOT p50 |
|---:|---:|---:|---:|
| 1 | 86.2 | 59.0 ms | 11.2 ms |
| 4 | 88.4 | 154.0 ms | 44.1 ms |
| 16 | 620.5 | 277.8 ms | 21.7 ms |
| 32 | 812.2 | 770.1 ms | 30.0 ms |

vLLM 0.20.2 **did not run this session** — the rented pod's driver (570.153.02, max CUDA 12.8) is older than vLLM 0.20.2's PyTorch CUDA 13 minimum. For ratio computation we reference the apples baseline from [`bench/v0.2-cuda/REPORT_2026-05-13.md`](../../../bench/v0.2-cuda/REPORT_2026-05-13.md) (same hardware, same model, ShareGPT dataset; ferrum env knobs differed at that time):

| c | this-session ferrum | historical vLLM (2026-05-13, ShareGPT) | indicative ratio |
|---:|---:|---:|---:|
| 1 | 86.2 | 205.4 | 0.42 |
| 4 | 88.4 | 505.8 | 0.17 |
| 16 | 620.5 | 1253.2 | 0.50 |
| 32 | 812.2 | 1883.2 | 0.43 |

**Caveats** — these ratios are **indicative, not authoritative**:
1. dataset differs: this session is `random 256/128`, the 2026-05-13 baseline is ShareGPT (≤200 input avg). Random-256 is roughly 2× heavier on prefill which suppresses the throughput denominator (especially c=4).
2. n_repeats=1 only (sweep_bottleneck.sh runs each cell once for trace+nsys); CI95 not computable.
3. ferrum at `cbe04ea` (this session) ≠ ferrum at the 2026-05-13 baseline commit.

A re-baseline against fresh apples-to-apples vLLM is required before quoting these ratios externally. Re-baseline is blocked until a Vast pod with driver ≥ 575 (CUDA 13.x) is available; track separately.

**c=4 anomaly**: this session's c=4 reports 88.4 tok/s — essentially identical to c=1. Hypothesis: with `num_prompts=30 / random_in=256`, prefill cost dominates wall time at small c and the 4 concurrent prefills serialize on the iteration lock. **Re-run c=4 with `num_prompts=128` and ShareGPT data** before treating that row as load-bearing.

## Bottleneck per c — measured chrome-trace breakdown

Framework data (BackendTimer GPU events, NOT CPU timing — chrome trace format):

| c | moe | attention | decode_step (embed/lm_head/final_norm) |
|---:|---:|---:|---:|
| 1 | **73.4%** | 19.1% | 7.5% |
| 16 | **73.4%** | 19.4% | 7.1% |
| 32 | **73.4%** | 19.5% | 7.1% |

(c=4 trace omitted — bench was truncated and partial events were patched out; category split mirrors c=1.)

**Headline finding**: MoE dominates at all c. The category split is essentially invariant across c=1/16/32 — the ferrum bottleneck shape is **kernel-level (MoE Marlin matmul), not scheduler-level**. Increasing c does not shift cost into the framework overhead bucket; it widens absolute moe µs proportionally.

This contradicts the prior hypothesis that c=1 vs c=32 had different shaped gaps. The kernel-quality lever (vllm-moe-marlin or our own Marlin-moe v2) is the single dominant cost item at every c.

| c | dominant cost | direct lever |
|---|---|---|
| 1 | MoE matmul (73%) + attention (19%) | vllm-moe-marlin OR ferrum Marlin tile heuristic |
| 4 | (re-run needed, see anomaly above) | — |
| 16 | MoE matmul (73%) + attention (19%) | vllm-moe-marlin OR ferrum Marlin tile heuristic |
| 32 | MoE matmul (73%) + attention (19%) | vllm-moe-marlin OR ferrum Marlin tile heuristic |

Per-c trace details: [bottleneck-c1.md](bottleneck-c1.md) · [bottleneck-c4.md](bottleneck-c4.md) · [bottleneck-c16.md](bottleneck-c16.md) · [bottleneck-c32.md](bottleneck-c32.md).

**nsys ground truth not captured this session** — `sweep_bottleneck.sh` wraps `bench-serve` (HTTP client, no GPU work) instead of `ferrum serve` under nsys. Bug to fix in follow-up; framework-validation session's nsys data ([`framework-validation-2026-05-25/nsys_kernels.csv`](../framework-validation-2026-05-25/nsys_kernels.csv)) is the current kernel-level reference and confirms Marlin <256,1,8,8> = 49% of GPU, cublas gemv (lm_head) = 10.2%, cutlass wmma (attn) = 11.5%.

## Prioritized levers

Ordered by **(expected ratio gain × probability of success on CUDA 13)**:

### 1. Fix `vllm-moe-marlin` build on CUDA 13 — **biggest single lever**

Currently blocked: rust-lld reports `undefined hidden symbol marlin::Marlin<...>` even with `-Xcompiler -fvisibility=default` and `__attribute__((visibility("default")))` on the template definition.

**Tried (PR #206 commits 1f9c3cb, 9eb46d7, b5dc156):**
- `-Xcompiler -fvisibility=default` — no effect (likely only affects host code, not nvcc-generated device stubs)
- `__attribute__((visibility("default")))` on template decl in `kernel.h` — no effect
- Same attribute on template definition in `marlin_template.h` — no effect

**Next things to try (decreasing order of confidence):**
1. `-rdc=true --device-link` workflow: build ops.cu with `-rdc=true`, run `nvcc --device-link` on the .o, then link the resulting `__cudart_*` symbols into the final binary. Restructures `build.rs::compile_vllm_moe_marlin` non-trivially.
2. Refresh vendored `crates/ferrum-kernels/kernels/vllm_marlin_moe/` from current vLLM HEAD (vendored circa v0.7-pre per Cargo.toml comments; may have been fixed upstream for CUDA 13).
3. As a fallback: manually instantiate all referenced `Marlin<...>` template specializations in a separate `.cu` file forcing default visibility. Tedious but deterministic.

**Expected gain**: ferrum_moe (49% of GPU time) → vllm_moe_marlin kernel which is empirically ~2× faster on this exact workload. **~25-30 pp ratio gain** at c=16/32, **~15 pp** at c=1/4.

### 2. lm_head: swap cublas gemv → cutlass GEMM (vLLM-style)

nsys shows `cublasGemvParamsEx<...,(int)6>` = **10.2% of GPU time** at c=16. vLLM uses tuned cutlass `Kernel2<...wmma_tensorop...>`. Per nsys, cutlass on the same workload runs in roughly **65% the time** of cublas gemv for vocab-sized projections.

**Approach**: in `ferrum-quantization` Linear impl for the lm_head case (m ≤ 32, n = vocab ≥ 150k, k = hidden), dispatch to a cutlass-based GEMM rather than cublas. Requires the cutlass GEMM op to be exposed via `Backend::gemm_cutlass()` or as a Linear adapter.

**Expected gain**: ~3 pp ratio gain across all c (the lm_head fires per emitted token regardless of batch size).

### 3. MoE Marlin tile heuristic — small-m optimization

Even without vLLM's marlin_moe, ferrum's own Marlin can be tuned. Per nsys at c=16, the dominant kernel is `Marlin<256,1,8,8,4,8>` (49%, `thread_m_blocks=1`). For c=32 it should pick `thread_m_blocks=2` for higher utilization but probably doesn't (autotune predicate gates).

**Approach**: extend `crates/ferrum-kernels/src/backend/cuda/marlin.rs` tile selection to consider effective per-expert m (= `tokens × top_k / num_active_experts`). For Qwen3-MoE that's typically ≤8 even at c=32. The right tile is `<256,2,8,8>` for c≤16 effective-m and `<256,4,16,4>` for c≥32. Currently the heuristic picks `<256,1,8,8>` for all.

**Expected gain**: ~5-8 pp at c=4/16 where small-m suboptimality bites hardest.

### 4. Phase 3 chunked-prefill token budget tuning

Per `memory/project_phase3_token_budget.md`: M3 c=16 +11%, c=32 +5% from setting `FERRUM_MAX_BATCHED_TOKENS=2048`. Already on in the iter-3 baseline. Could try `4096` or `8192` for larger cohort fusion, but `--max-num-batched-tokens` interacts with KV memory.

**Expected gain**: small (~1-2 pp), and only at c≥16.

### 5. CUDA graph capture quality at c=32

`memory/project_moe_phase3_graph_bug.md`: full graph at c=32 has been net **-6%** because graph re-bind cost dominates. Currently we ship with `FERRUM_GRAPH=1` always-on. At c=32 the iter-3 number with graph ON is 970 tok/s; with graph OFF it should be ~1030 tok/s. **Try this immediately** — it might be free 6%.

**Expected gain**: ~3 pp at c=32 only, may regress c=1/4 (where graph helps).

## Milestone roadmap

| Phase | Goal | What ships | Est. wall |
|---|---|---|---|
| **A. Locked baseline** | Sweep data per c, framework-validated, this PR | sweep dir + GOAL.md + per-cell bottleneck-<c>.md | 1-2h GPU |
| **B. Easy wins** | ratio ≥ 0.60 all c (graph-c32-off + Phase 3 token-budget verify) | env tuning PR | 1d |
| **C. lm_head cutlass** | ratio ≥ 0.65 all c | `Backend::gemm_lmhead()` or Linear adapter | 1 week |
| **D. Marlin tile heuristic** | ratio ≥ 0.70 all c | dispatch fix in `marlin.rs` autotune | 1 week |
| **E. vllm-moe-marlin CUDA 13 fix** | ratio ≥ 0.80 all c (final target) | -rdc=true link OR upstream refresh OR explicit template instantiation .cu | 1-2 weeks (high uncertainty) |

If E proves intractable, an alternative path to 0.80 is "ferrum's own marlin_moe v2" — write a Marlin-MoE kernel from scratch tuned for sm_89. ~3-4 weeks engineering. Out of scope for this goal doc; track separately if E stalls.

## Validation criteria

Goal is **achieved** when:

1. `scripts/sweep_bottleneck.sh qwen3-moe-30b-int4 1,4,16,32` produces a `ratio ≥ 0.80` row in `aggregate_sweep.py` output for **all four cells**, with `n_repeats ≥ 5` so CI95 doesn't overlap 0.80
2. Sweep run is on the locked GPU (`scripts/lock_gpu.sh`) with a committed `env.commit_sha` referenceable in main
3. The validation sweep result lives in `docs/bench/cuda-rtx4090-<date>-m3-80pct-confirmed/`

Goal is **provably blocked** when:

1. After E (or alternative) ships, sweep still reports `< 0.80` at any c
2. nsys profile shows a single kernel that consumes > 30% of GPU time and is also the dominant cost in vLLM's nsys — i.e. we're already at the same kernel quality and the gap is elsewhere (scheduler, queueing, framework overhead)
3. PR description must include the nsys evidence

---

## Per-cell bottleneck files

(Populated by sweep + aggregate_sweep.py — links land here)

- [`bottleneck-c1.md`](bottleneck-c1.md) — *pending sweep*
- [`bottleneck-c4.md`](bottleneck-c4.md) — *pending sweep*
- [`bottleneck-c16.md`](bottleneck-c16.md) — *pending sweep*
- [`bottleneck-c32.md`](bottleneck-c32.md) — *pending sweep with nsys*
- [`sweep-summary.md`](sweep-summary.md) — *full aggregate, generated by `aggregate_sweep.py`*

---

## Update — 2026-05-25 session

Lever #1 (vllm-moe-marlin) unblocked. See
[`session-2026-05-25/RESULTS.md`](session-2026-05-25/RESULTS.md) +
[`session-2026-05-25/PROGRESS.md`](session-2026-05-25/PROGRESS.md).

> ⚠️ **Data-quality caveats — read before quoting any number below**
>
> Per the same caveats applied to the original baseline table earlier
> in this doc, the session-2026-05-25 numbers are **preliminary, not
> publication-grade**:
>
> 1. **vLLM ratios are apples-to-oranges**. The ratio column compares
>    this session's ferrum runs (`--dataset random --random-input-len
>    256 --random-output-len 128`) against the historical vLLM column
>    (2026-05-13 ShareGPT, average input ≤ 200 tokens). random-256 is
>    roughly 2× heavier on prefill — it suppresses ferrum's throughput
>    denominator and depresses the ratio (especially at small c where
>    prefill dominates). The Δ% column (OFF→ON) is sound; the
>    ratio_OFF / ratio_ON columns are **indicative only**.
> 2. **n_repeats=1**. Single-shot per cell, no Student-t CI. PLAYBOOK
>    § 0.4 needs n ≥ 3 before quoting outside the team.
> 3. **c=4 marked anomaly** is inherited from the previous session's
>    `num_prompts=30` caveat — **not re-verified** this session. Treat
>    the +7% Δ as a lower bound; re-run with `num_prompts ≥ 128`
>    before reading anything into it.
> 4. **c=32 + FERRUM_GRAPH=0 = "−1.5% vs GRAPH=1"** (in `RESULTS.md`)
>    sits inside single-shot noise. Don't read it as "graph-off doesn't
>    help with vllm-moe-marlin" until repeated n ≥ 3.
> 5. **No nsys / chrome-trace captured for the ON path this session**.
>    `bottleneck-c1/c4/c16/c32.md` in this directory still describe
>    the OFF (pre-vllm-moe-marlin) state. We don't yet know which
>    kernel owns the remaining 23 pp gap at c=32 ON — `moe` may no
>    longer be 73% of GPU time once vllm-moe-marlin is active.
>
> Re-baseline against apples-to-apples vLLM (random 256/128, vLLM
> 0.20.2 on driver ≥ 575) **and** re-run `sweep_bottleneck.sh
> qwen3-moe-30b-int4` with `FERRUM_VLLM_MOE=1` before any external
> communication of this lever's outcome.

| c  | OFF tput | ON tput | Δ%   | ratio_OFF → ratio_ON ⚠️ | gap to 0.80 ⚠️ |
|---:|---------:|--------:|-----:|---:|---:|
| 1  | 128.0    | 146.5   | +14% | 0.62 → **0.71** | 9 pp |
| 4  | 130.1    | 138.8   | +7%  | 0.26 → 0.27 ⚠️ | re-test (anomaly inherited, not re-verified) |
| 16 | 673.6    | 818.3   | +22% | 0.54 → **0.65** | 15 pp |
| 32 | 871.1    | 1079.4  | +24% | 0.46 → **0.57** | 23 pp |

⚠️ markers on a column mean "see caveat #1 — apples-to-oranges with
the vLLM denominator". The OFF→ON Δ% column itself is internally
consistent (same hardware, same dataset).

Phase B is partially done — vllm-moe-marlin shipped, but the c=32
graph-off A/B and Phase 3 chunked-prefill 8192 verify are still
outstanding. Phases C / D / E still required to close the remaining
gap on c=16/c=32. The encoding bug GOAL.md hinted at went deeper
than visibility flags — see
[`session-2026-05-25/PROGRESS.md#findings`](session-2026-05-25/PROGRESS.md#findings)
for the symbol-level diagnosis.

### Next-session must-do (to retire the caveats above)

1. Rent pod with driver ≥ 575, install vLLM 0.20.2, run `vllm bench
   serve --dataset random --random-input-len 256 --random-output-len
   128 --num-prompts 256 --n-repeats 5` for c=1/4/16/32 to land the
   apples-to-apples denominator.
2. `FERRUM_VLLM_MOE=1 bash scripts/sweep_bottleneck.sh
   qwen3-moe-30b-int4 1,4,16,32` (this captures chrome trace + nsys)
   and replace `bottleneck-c*.md` with ON-state data.
3. Re-run ferrum bench with `num_prompts=128, n_repeats=5` so the
   throughput column lands with CI95.
4. Replace this Update block with the publication-grade table once
   (1)+(3) are in hand.

---

## Update — 2026-05-27 iteration 2 (perf + correctness, GPU-verified)

See [`session-2026-05-27-iteration2/SESSION-REPORT.md`](session-2026-05-27-iteration2/SESSION-REPORT.md)
for the full session writeup. Shipped (commits on `fix/moe-align-block-size-packed-row`):

- **`37f5dda` fix(paged-attn)**: opt-in to extended dynamic shared
  when `FERRUM_KV_CAPACITY > 12K`. Fixes the 3rd-chat-turn panic
  `paged_varlen_attn: CUDA_ERROR_INVALID_VALUE` on `ferrum run`
  (Chat-profile autosizer sets 64 KB shared > sm_89 48 KB default).
- **`a873d63` perf(autosize)**: default-ON `FERRUM_MOE_GRAPH=1` for
  Qwen3-MoE decode layer-loop CUDA graph capture. Memory's
  documented "c=32 -6% regression" was on the pre-fix
  garbage-emission path; with PR #216's moe_align fix landed, graph
  replay produces correct sorted_token_ids and `cuGraphLaunch`
  overhead is amortized by eliminating ~480 per-iter kernel launches.

GPU-verified at c=32 (RTX 4090, random 256/128, n=1, prompts=30):

| Config | tok/s | ratio vs hist vLLM 1883 |
|---|---:|---:|
| SAFE (`FERRUM_VLLM_MOE=0`) | 848 | 0.450 |
| + `FERRUM_VLLM_MOE=1` (post-fix device-route) | 976 | 0.518 |
| + autosizer `FERRUM_MOE_GRAPH=1` | **1006** | **0.534** |

Still **27 pp short of 0.80**. Per nsys post-fix profile (file
`session-2026-05-27-iteration2/api.csv`): top costs are
`cuStreamSynchronize` 44.7% (5288 calls — ~38/iter, target ~1-2 like
vLLM), `cuLaunchKernel` 28.8% (~400K launches), `cuMemcpyHtoDAsync`
15.5% (50K H2D). Closing the gap requires multi-day levers
(sync-source bisect, small-m fused MoE kernel port, full forward
graph extension) — none fit a single bench-iteration session.

Same caveat as the 2026-05-25 Update block: the vLLM denominator is
the 2026-05-13 ShareGPT historical baseline. Re-baseline against
apples-to-apples vLLM 0.20.2 still required to retire "indicative
ratio" status.
