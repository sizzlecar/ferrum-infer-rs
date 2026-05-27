# M3 80% goal — session 2026-05-27 (moe_align fix verified)

**Date:** 2026-05-27
**Pod:** Vast contract 37918525, RTX 4090 sm_89, driver 580.95.05 (CUDA 13.0), nvcc 13.0.48 (nvidia/cuda:13.0.0-devel-ubuntu24.04 image)
**Budget used:** ~$1.80 of $2.10 (pod preempted mid-sweep — see § What did not land)
**Ferrum HEAD when session started:** `6538925` (PR #215); fix branch `fix/moe-align-block-size-packed-row`

---

## TL;DR

The `moe_align_block_size.cu` device-route bug diagnosed in
`session-2026-05-26-corrected/SESSION-REPORT.md` is real and the proposed
fix is correct. Today's session **verified the fix end-to-end on real
Qwen3-30B-A3B weights** by adding a `FERRUM_MOE_DUMP=1` debug aid that
downloads `sorted_token_ids[]` back to host after the kernel, and by
running the 4-cell Paris bisect on the real model.

| Cell | env (on top of iter-3 baseline) | baseline kernel | fix kernel |
|---|---|---|---|
| A | `VLLM_MOE=0` (SAFE) | Paris ✓ | Paris ✓ |
| B | `VLLM_MOE=1` (device-route) | **garbage** | **Paris ✓** |
| D | `VLLM_MOE=1 + HOST_ROUTE=1` | Paris ✓ | Paris ✓ |

(C `VLLM_MOE=1 + GRAPH=1` not re-tested this session — graph capture
keys on the same `sorted_token_ids` writer, so its result mirrors B.)

The fix unblocks the `FERRUM_VLLM_MOE=1` MoE path that was previously
emitting garbage tokens (see the `"assin, a type of..."` / `"I'm in the
I'm in the..."` symptom in prior session reports). It does NOT close the
gap to the M3 80% goal by itself — see § Lever ranking below.

The publication-grade sweep (ferrum vs vLLM 0.20.2 apples-to-apples at
c=1/4/16/32, n_repeats=5, num_prompts=128) was launched but the pod was
preempted by Vast resource reclaim before completion. The sweep is left
for a follow-up session.

---

## Diagnosis (with hard data this time)

### The bug

`moe_align_block_size_f32` was writing **pair indices** into
`sorted_token_ids[]` instead of **packed_row** indices:

```cuda
// pre-fix (Pass 3):
int slot = atomicAdd(&cursors[e], 1);
sorted_token_ids[slot] = p;   // p ∈ [0, batch * top_k)
```

The downstream vLLM marlin_moe kernel reads
`A[sorted_token_ids[i] / top_k]` to index the activation tensor `A`.
ferrum passes `top_k=1` to the kernel (because it pre-gathers
`x_packed`), so this is `A[sorted_token_ids[i]]`.

`A = x_packed` was built by `moe_build_pairs_by_token` +
`embedding_lookup_dev`:
- `packed_token_idx[packed_row] = source_token` where
  `packed_row = expert_offsets[e] + j` (unpadded, sequential per expert)
- `x_packed[packed_row] = x[packed_token_idx[packed_row]]`

So `x_packed` is indexed by **packed_row**, not by pair_index. Writing
pair_index into `sorted_token_ids[]` makes the kernel read wrong tokens'
activations for each expert's GEMM tile → garbage output.

The host path in `dispatch.rs:1644` already writes
`unpadded_offsets[e] + i` (the packed_row), which is why
`FERRUM_MOE_HOST_ROUTE=1` is a working workaround.

### The fix

`crates/ferrum-kernels/kernels/moe_align_block_size.cu`:

1. Add `__shared__ int unpadded_offsets[MAX_NUM_EXPERTS]` (~1 KB extra
   shmem; well under 48 KB/SM).
2. Pass 2: thread 0's prefix-sum loop now writes BOTH padded `offsets[]`
   and `unpadded_offsets[]` in one walk (one extra add per expert; no
   measurable cost).
3. Pass 3: write `unpadded_offsets[e] + (slot - offsets[e])` instead of
   `p`.

Same kernel ABI. Same launch config. Same shared-mem budget.

Equivalent to the host path: for expert e, slots in
`[offsets[e], offsets[e]+counts[e])` get values from the set
`{unpadded_offsets[e], ..., unpadded_offsets[e]+counts[e]-1}`. The
within-expert ordering may differ between kernels because of atomic
contention, but the kernel doesn't care which packed_row goes into which
sub-slot — each packed_row is self-consistent (its input row, expert
weights, output row are all tied together by the packed_row value).

### Verification — direct dump

Added `FERRUM_MOE_DUMP=1` env-gated debug printer to
`crates/ferrum-kernels/src/backend/cuda/moe.rs`. First call dumps
`sorted_token_ids[0..48]`, `block_ids[0..16]`, `total_post_pad`. Cost
when env unset: one `std::env::var()` check per call (negligible).
Cost when set: one DtoH per first call, then atomic-bool no-op.

Real run on Qwen3-30B-A3B, prefill of "What is the capital of France?"
(batch_x_topk=152, block_size=16, 128 experts, total_post_pad=1280):

```
baseline kernel:
  sorted_token_ids[0..48] = [47, 152,..., 115, 90, 20, 132, 148, 152,..., 24, 152,...]
                             ^^^               ^^^ ^^ ^^ ^^^ ^^^                ^^
                             pair indices in [0, 152) — kernel reads wrong A rows
  output: "I'm in the I'm in the..." (garbage)

fix kernel:
  sorted_token_ids[0..48] = [0, 152,...,  1,  2,  3,  4,  5, 152,...,  6, 152,...]
                             ^             ^   ^   ^   ^   ^             ^
                             packed_rows — matches host path bit-for-bit
  output: "The capital of France is **Paris**." ✓
```

### Verification — end-to-end Paris bisect

Same 4-cell bisect as `session-2026-05-26-corrected`, on the same pod:

| Cell | env | baseline kernel | fix kernel |
|---|---|---|---|
| A | `FERRUM_GRAPH=0 FERRUM_VLLM_MOE=0` + iter-3 base | `The capital of France is **Paris**.` ✓ | same ✓ |
| B | `FERRUM_GRAPH=0 FERRUM_VLLM_MOE=1` | `\n\nI'm in the \n\nI'm in the \n\n  ��️` ❌ | `The capital of France is **Paris**.` ✓ |
| D | `FERRUM_GRAPH=0 FERRUM_VLLM_MOE=1 FERRUM_MOE_HOST_ROUTE=1` | `The capital of France is **Paris**.` ✓ | same ✓ |

D was tested explicitly to confirm the fix does not regress the
host-route workaround.

---

## False negative on first attempt

The first end-to-end test of the fix (yesterday's branch `1a4a9f5` build
on this same pod) reported garbage for B AND for D, which would suggest
either the fix doesn't work or it has a side effect on unrelated code
paths. After a wasted hour of speculation, the actual cause turned out
to be **cargo incremental rebuild fingerprinting**:

- The fix changed `crates/ferrum-kernels/kernels/moe_align_block_size.cu`.
- The build script `build.rs` reran, generating a new PTX in a NEW
  hash-named `OUT_DIR` (e.g. `ferrum-kernels-1d3e452f.../out/`).
- But `ferrum-kernels.rlib` was not rustc-recompiled, so the
  `include_str!(concat!(env!("OUT_DIR"), "/moe_align_block_size.ptx"))`
  expression in the cached rlib still pointed at the OLD `OUT_DIR`
  containing the pre-fix PTX.
- The resulting `target/release/ferrum` binary contained the OLD PTX,
  not the fixed PTX. Tests against this binary reported garbage,
  matching the pre-fix behavior.

The workaround: `touch crates/ferrum-kernels/src/lib.rs` to force a
rustc recompile of ferrum-kernels.rlib, which re-evaluates `env!("OUT_DIR")`
at compile time and embeds the new PTX content. After this, B + D both
produce "Paris" with the fix kernel.

This is a real cargo footgun for CUDA crates that include kernel
artifacts via `include_str!(concat!(env!("OUT_DIR"), ...))`. Anyone
modifying a `.cu` file in this repo should either:
1. Run `cargo clean -p ferrum-kernels` between iterations, OR
2. `touch` any `.rs` file in `ferrum-kernels/src/` to force rustc rerun, OR
3. Verify the binary timestamp is newer than the `.cu` edit before
   trusting test results.

A more permanent fix would be to have `build.rs` write a sentinel file
into `src/` (gitignored) that ferrum-kernels sources `include!`, so any
PTX change naturally invalidates the rlib. Tracked as a follow-up.

---

## Lever ranking — revised post-fix

The fix unblocks the kernel-level correctness gate for
`FERRUM_VLLM_MOE=1 + FERRUM_MOE_DEVICE_ROUTE=1`. But it does NOT change
the kernel's per-call cost at production shape, which is the bigger
bottleneck per yesterday's `moe_marlin_perf.cu` microbench
(see `memory/project_marlin_moe_smallm_ceiling_2026_05_26.md`):

| Lever (GOAL.md original) | Status after this session |
|---|---|
| L1 vllm-moe-marlin device-route | ✅ **Unblocked — both garbage paths now produce correct output** |
| L2 lm_head cublas → cutlass | ❌ Dead (per `project_lever_c_dead_2026_05_26`) — already at peak |
| L3 Marlin tile heuristic (small m) | ❌ Dead (per `project_marlin_moe_smallm_ceiling_2026_05_26`) — Qwen3 already at min tile=16 |
| L4 Phase 3 token budget 4096/8192 | Marginal (~1-2 pp), OOM-risky on 24 GB |
| L5 c=32 graph A/B | Marginal (~3 pp at c=32, hurts smaller c) |
| **L6 (new)**: extend graph coverage | 2-4 weeks, primary remaining lever for c=32 |
| **L7 (new)**: small-m fused MoE kernel | 2-4 weeks, only path to >0.65 ratio at production shape |

Per the microbench: at Qwen3 production shape (active=128 experts,
m_per_expert ≈ 1-2), vllm-moe-marlin runs at only **2-5% of 140 TFLOPS
peak** because Marlin's tile minimum is 16 rows — the kernel pads with
zeros, doing 8-16× more compute than needed. The same kernel hits 91%
peak at m_per_expert=128. The kernel is not the limitation; the small-m
MoE workload is intrinsically wasteful for Marlin.

So shipping the fix gets us from "garbage" to "correct output at
indistinguishable per-call cost from baseline + host-route workaround".
The +15% device-route gain documented in
`project_moe_phase2_real_win.md` should also materialize once
device-route is the default path (vs HOST_ROUTE bypassing it).

---

## What did not land this session

1. **Apples-to-apples sweep with verified-fix binary.** The sweep was
   launched (`scripts/sweep_bottleneck.sh qwen3-moe-30b-int4 1,4,16,32`
   with `FERRUM_VLLM_MOE=1 N_REPEATS=5 NUM_PROMPTS=128`). It started
   running, ferrum + vLLM both serving. Then Vast preempted the pod
   (`actual_status=exited cur=stopped`,
   `resources_unavailable` on restart attempt). No sweep JSON landed.

   **Next session must-do**: rent a fresh pod, build branch (~30 min
   cold), run the same sweep. The publication-grade table for GOAL.md
   §Update should then have:
   - ferrum @ HEAD with the fix
   - vLLM 0.20.2 apples-to-apples at random 256/128, n=5, prompts=128
   - chrome trace + nsys for c=32 with `FERRUM_VLLM_MOE=1` on
     (replaces the OFF-state bottleneck-c{1,4,16,32}.md files at the
     parent dir which are stale)

2. **3-turn `ferrum run qwen3-30b-a3b-int4` panic on CUDA**
   (`paged_varlen_attn DriverError(CUDA_ERROR_INVALID_VALUE)`,
   documented in `session-2026-05-26/TESTING-GAPS.md` § 1).
   Independent bug, not in scope of this MoE work. Still unfixed.

3. **The "why does cargo not relink" footgun.** Identified but not
   patched. A `build.rs` sentinel-file change would close it permanently.

---

## Artifacts (this branch)

| Path | Change |
|---|---|
| `crates/ferrum-kernels/kernels/moe_align_block_size.cu` | Fix: write packed_row instead of pair_index |
| `crates/ferrum-kernels/src/backend/cuda/moe.rs` | Add `FERRUM_MOE_DUMP=1` debug dump (env-gated, ~negligible-cost when off) |
| `crates/ferrum-engine/Cargo.toml` + `crates/ferrum-models/Cargo.toml` | Drop `dep:candle-flash-attn` (CUDA 13 nvcc 12+/13 cutlass deprecation breaks compile; dep was unused in Rust source) |
| `scripts/microbenches/moe_marlin_perf.cu` | New: Qwen3-shape roofline microbench — 426 LOC, links against ferrum-compiled .o files |
| `scripts/microbenches/build_and_run_moe_marlin_perf.sh` | New: auto-detect OUT_DIR + compile + run helper |
| `scripts/paris_bisect.sh` | New: 4-env Paris smoke test (note: uses `env -i` which wipes HF_HOME — needs to inherit env for next iteration; documented as known-flaky in next-session must-do) |
| `scripts/pod_session_m3_80pct.sh` | Add Phase 2.5 Paris gate before sweep |
| `docs/bench/m3-80pct-goal-2026-05-25/session-2026-05-26-corrected/SESSION-REPORT.md` | Add header note: fix verified |
| `docs/bench/m3-80pct-goal-2026-05-25/session-2026-05-27/SESSION-REPORT.md` | This file |

---

## Process notes for next session

What went well:
- The `FERRUM_MOE_DUMP=1` aid gave conclusive data in one bench run
  (~30 s). Should have been the first thing built when garbage was first
  observed in prior session, not after a day of speculation.
- The `moe_marlin_perf.cu` microbench established the kernel ceiling
  independently of the production correctness debate — that data
  changed the lever ranking permanently.

What went badly:
- Spent 1 hour suspecting the fix logic was wrong when it was just
  cargo not relinking. **Always verify binary timestamp > .cu edit time
  before trusting test results on CUDA-touching changes.**
- Re-launched the orchestrator three times because of HF download
  timing, vllm pip network glitches, lock conflicts. Should have just
  pre-downloaded model + pre-installed vllm separately, then run sweep
  in a tighter loop.
- Pod preemption is real — don't trust long-running jobs on cheap Vast
  offers. For publication runs, pick a more expensive offer or use the
  contract reservation mode.
