# M3 80% goal — session 2026-05-26 (corrected diagnosis + fix)

**Date:** 2026-05-26
**Pod:** Vast contract 37894338, RTX 4090 sm_89, CUDA 13.0.48 driver / nvcc 12.4
**Budget used:** ~$1.90 / $2.10 (90% of session credit)
**Ferrum HEAD when session started:** `6538925` (PR #215 "session(2026-05-26): M3 80% findings + CUDA 13 timer fix + microbench infra")

---

## TL;DR

`session-2026-05-26/FINDINGS.md`'s root-cause diagnosis ("packing format mismatch at `quant.rs:728`") is **wrong**. The kernel, repack, c_tmp size, workspace size, scales permute, and ScalarType TU mangling are **all correct** on CUDA 13.

**The real bug** is in `crates/ferrum-kernels/kernels/moe_align_block_size.cu`: the device kernel writes `pair_index` into `sorted_token_ids[slot]`, but the vLLM marlin MoE kernel expects the value to be **unpadded packed_row** (matching what `moe_build_pairs.cu` writes into `pairs_by_token` and what `embedding_lookup_dev` uses to gather `x_packed`).

The static workaround `FERRUM_MOE_HOST_ROUTE=1` bypasses the buggy device kernel and produces correct output — at the cost of the +15% c=32 device-route perf gain.

**A one-line fix** has been applied locally (not GPU-verified due to budget exhaustion). See § Fix.

---

## Setup

| Component | Value |
|---|---|
| GPU | RTX 4090 (sm_89, 24 GB, 128 SMs) |
| Driver / CUDA | 580.126.09 / driver-side CUDA 13.0 |
| nvcc | 12.4.131 (from `nvidia/cuda:12.4.0-devel-ubuntu22.04` image) |
| Pod $/hr | $0.67 (Vast offer 13430355) |
| Model | `Qwen/Qwen3-30B-A3B-GPTQ-Int4` (16 GB on disk) |
| Disk | 22 GB total; 19 GB used after build + model (fit, with 3 GB slack) |

Disk fits — earlier claim that ferrum-cli + Qwen3-30B needs > 22 GB was wrong.

---

## What I tested + what was disproven

### 1. Microbench `scripts/microbenches/moe_marlin_correctness.cu` (487 lines)

Standalone `.cu` that compiles against ferrum's vendored `vllm_marlin_moe/` and `vllm_marlin/` sources **bypassing cargo** (≤10 min compile vs 30+ min ferrum cargo).

Calls `ferrum_vllm_gptq_marlin_repack` + `ferrum_vllm_marlin_moe_f16` with hand-built moe_align outputs, compares against FP16 reference.

**14 test cases — ALL PASS**:

| # | Configuration | Result |
|---|---|---|
| T1 | 1 expert, non-uniform weights, K=128 N=128 | PASS |
| T2 | 1 expert, scales differ per group | PASS |
| T3 | 2 experts, different uniform weights | PASS |
| T4 | 2 experts, non-uniform weights, tokens split | PASS |
| T5 | 8 experts, stored 8..15 (tests stacked offset arithmetic) | PASS |
| T6 | Qwen3-shape (K=2048 N=1536 128 experts), ATOMIC_ADD | PASS |
| T7 | Qwen3-shape, FP32_REDUCE with ferrum's 2M c_tmp | PASS |
| T8 | Qwen3-shape, FP32_REDUCE with vLLM full-bound c_tmp | PASS |
| T9 | Qwen3-shape, ferrum's "too-small" workspace formula | PASS |
| T10 | Qwen3-shape, FP32_REDUCE ferrum c_tmp + ferrum workspace (= production) | PASS |
| T11 | Qwen3-shape, FP32_REDUCE vllm c_tmp + ferrum workspace | PASS |
| T12 | M=2048 stress (production-scale batched prefill), ferrum c_tmp + vllm ws | PASS |
| T13 | M=2048, vllm c_tmp + vllm ws | PASS |
| T14 | M=2048, ferrum c_tmp + ferrum ws (= production stress) | PASS |

**Conclusion**: kernel + repack + c_tmp + workspace + scales permute (within tested patterns) + multi-expert stacked offset arithmetic are all correct at production scale.

### 2. Ferrum's own parity test (`cargo test ... cuda_marlin_moe_vllm`)

`crates/ferrum-quantization/tests/marlin_moe_vllm_parity_test.rs` runs ferrum's actual `load_stacked_gptq_vllm_marlin` + `gemm_phase_vllm` with **random non-uniform GPTQ data** and asserts max_rel < 1e-2 across 4 experts.

**Result on CUDA 13: PASS**.
- expert 0 max diff 0.0312 rel 0.0008
- expert 1 max diff 0.0156 rel 0.0005
- expert 2 max diff 0.0156 rel 0.0004
- expert 3 max diff 0.0156 rel 0.0005

This eliminates: scales permute (covered by random scales), packing format, ScalarType TU mangling.

### 3. End-to-end Paris bisect

Built ferrum-cli with `cuda,vllm-moe-marlin`. Ran the prompt "What is the capital of France?" under 4 env combos:

| Test | env vars | Output | Verdict |
|---|---|---|---|
| **A-SAFE** | `FERRUM_GRAPH=0 FERRUM_VLLM_MOE=0` | `The capital of France is **Paris**.` | ✅ |
| **B-VLLM_MOE-only** | `FERRUM_GRAPH=0 FERRUM_VLLM_MOE=1` | `I have the ability to provide a detailed, comprehensive, and accurate I have the` | ❌ garbled |
| **C-VLLM_MOE+GRAPH** | `FERRUM_GRAPH=1 FERRUM_VLLM_MOE=1` | (same garbage as B) | ❌ garbled |
| **D-VLLM_MOE+HOST_ROUTE** | `FERRUM_GRAPH=0 FERRUM_VLLM_MOE=1 FERRUM_MOE_HOST_ROUTE=1` | `The capital of France is **Paris**.` | ✅ |

**Conclusion**: bug is **strictly in the device-route path**. `FERRUM_MOE_HOST_ROUTE=1` bypasses it and restores correctness. Graph capture (`FERRUM_GRAPH=1`) is innocent — B and C produce identical garbage.

(One incidental build issue: `candle-flash-attn` fails to compile on CUDA 13 nvcc 12.4. ferrum has no actual Rust callers of `candle_flash_attn`, so I removed the `dep:candle-flash-attn` lines from `ferrum-engine/Cargo.toml` and `ferrum-models/Cargo.toml` to unblock. **This change is in the working tree on Mac but NOT committed.** Either land that hack or upgrade nvcc when re-running.)

---

## Root cause

Compare `sorted_token_ids[slot]` semantics across the two paths:

| Path | Value written into `sorted_token_ids[slot]` |
|---|---|
| Host (`dispatch.rs:1644`) | `plan.expert_offsets[e] + i` = **unpadded packed_row** |
| Device (`moe_align_block_size.cu:111`) | `p` = **pair index** in `[0, batch * top_k)` |

These are different values. The downstream pipeline expects packed_row:

1. `moe_build_pairs.cu:79-82` (device): writes `packed_token_idx[packed_row] = token_id` where `packed_row = expert_offsets[e] + slot` (unpadded). This is what `embedding_lookup_dev` uses to gather `x` into `x_packed`. So `x_packed[packed_row]` = `x[token_id]`.
2. `embedding_lookup_dev(x, packed_token_idx, x_packed, ...)`: produces `x_packed` in unpadded-packed-row order.
3. vLLM marlin MoE kernel reads `A[sorted_token_ids[i] / top_k]`. With ferrum's `top_k=1` kernel param, this is `A[sorted_token_ids[i]]` = `x_packed[sorted_token_ids[i]]`.

If `sorted_token_ids[i]` is **pair_index** (current device kernel), step 3 reads `x_packed[pair_index]` — but `x_packed` is not indexed by pair_index, it's indexed by packed_row. The kernel reads the wrong tokens' activations for each expert's GEMM tile → outputs garbage that happens to still be valid token-shaped logits → model emits "I have the ability to" repeatedly.

Host path works because it writes packed_row directly into `sorted_token_ids`.

### Verification by example (computed by hand, not run)

batch=2, top_k=2, num_experts=3, block_size=16
- `expert_ids_per_pair = [0, 1, 0, 2]`
- counts = `{e0:2, e1:1, e2:1}` → unpadded_offsets = `{e0:0, e1:2, e2:3}`
- counts_padded = `{e0:16, e1:16, e2:16}` → padded offsets = `{e0:0, e1:16, e2:32}`

| pair `p` | expert `e` | atomicAdd cursor slot | per_expert_pos | desired value (host) | current device value |
|---|---|---|---|---|---|
| 0 | 0 | 0 | 0 | 0 (= unpadded[0]+0) | 0 (= p) — coincidentally OK |
| 1 | 1 | 16 | 0 | 2 (= unpadded[1]+0) | 1 (= p) — **WRONG** |
| 2 | 0 | 1 | 1 | 1 (= unpadded[0]+1) | 2 (= p) — **WRONG** |
| 3 | 2 | 32 | 0 | 3 (= unpadded[2]+0) | 3 (= p) — coincidentally OK |

Top-1 routing with one token per expert happens to produce p == packed_row (because they both equal the slot index). But any non-trivial routing (top_k > 1 or skewed distribution) makes them diverge — which is why parity-test's 4-expert random GPTQ AND simple Paris on Qwen3-30B (128 experts, top_k=8) both expose the bug differently from how a simpler one-token-per-expert microbench would.

---

## Fix (applied, not GPU-verified)

`crates/ferrum-kernels/kernels/moe_align_block_size.cu`:

1. Added `unpadded_offsets[MAX_NUM_EXPERTS]` shared-mem array (~1 KB extra shmem, well under 48 KB/SM).
2. Pass 2's thread-0 prefix-sum loop now computes both padded `offsets` and `unpadded_offsets` in one pass (one extra add per expert; zero observable cost).
3. Pass 3 now writes `unpadded_offsets[e] + (slot - offsets[e])` instead of `p`.
4. File header comment updated to reflect new semantics.

Diff is in the working tree; **not committed**.

The fix preserves the kernel's ABI (no Rust-side change), so a rebuild of just `ferrum-kernels` should be enough to verify.

---

## What's left to verify

Required steps to call the M3 80% goal's correctness regression "fixed":

1. **Rebuild ferrum-cli with the patched kernel** on any CUDA 13 RTX 4090 pod.
2. **Re-run the Paris bisect** (`/root/test_paris.sh` from this session, or equivalent). Expect:
   - A (SAFE): `Paris` ✓
   - B (VLLM_MOE only): **`Paris` ✓** (previously garbled)
   - C (VLLM_MOE + GRAPH): **`Paris` ✓**
   - D (VLLM_MOE + HOST_ROUTE): `Paris` ✓ (unchanged)
3. **Re-run apples-to-apples sweep** vs vLLM 0.20.2 at c=1/4/16/32. Expect the +15% device-route perf gain to materialize on top of the SAFE baseline:
   - SAFE c=32 = 762 tok/s; goal of VLLM_MOE=1 device-route at c=32 ≈ 880-900 tok/s (= ratio ≈ 0.47-0.49).
   - Still below 0.80 — additional levers (FA2 SplitKV, full forward graph) remain.
4. **Move `scripts/microbenches/moe_marlin_correctness.cu` into the repo** as a permanent regression test artifact.
5. **Land the Cargo.toml change** that strips the `dep:candle-flash-attn` line, OR upgrade to nvcc 13 toolchain to fix the candle-flash-attn build failure on CUDA 13.

---

## Process notes (for next session)

What went well:
- Renting the cheapest 24 GB pod ($0.67/hr) was the right call — disk fit, model fit.
- Native nvcc microbench bypassing cargo (~10 min) is far faster than the ferrum cargo build (~25-30 min). For pure CUDA hypotheses this is the right tool.
- Running ferrum's own parity test on CUDA 13 immediately eliminated several FINDINGS.md hypotheses.

What went badly:
- I spent too long on static analysis before going to GPU. The user explicitly redirected mid-session: "你近期静态分析 太糟糕了，你直接去开一个机器 然后使用 native cuda做最小验证".
- I **destroyed the pod before applying + verifying the fix**. The bug was located, but the fix-on-pod + Paris-verify-on-pod + bench-verify-on-pod cycle was never closed. Next time: bug located → fix kernel ON THE POD → rebuild → re-run Paris → THEN destroy.
- I missed the silent cargo build failure (openssl-sys + candle-flash-attn) for ~10 min because I was tailing the wrong stream. Set up Monitors that catch BOTH ready and failed states.
- Twice I trusted FINDINGS.md's pointer (`quant.rs:728`) and twice it was wrong. Trust empirical reproduction over written diagnosis.

What this session changes for the M3 80% roadmap:
- Phase A (SAFE baseline) ratio 0.41-0.59 across c is the correct denominator. The session-2026-05-25 perf numbers that used FERRUM_VLLM_MOE=1 were measuring garbage emission rate.
- Once the patched device-route lands, `FERRUM_VLLM_MOE=1` becomes safe to default-on (no `FERRUM_MOE_HOST_ROUTE=1` needed). Expected gain ≈ +15% on top of SAFE baseline at c=32 per `project_moe_phase2_real_win.md`.
- Lever ranking in `session-2026-05-26/FINDINGS.md` § "Lever ranking (revised)" stands EXCEPT: line "lever: `FERRUM_VLLM_MOE=1` works (b0da10d fix verified)" — that "fix" was wrong; the actual unblock is the `moe_align_block_size.cu` patch in this session.

---

## Artifacts

| Path | What |
|---|---|
| `scripts/microbenches/moe_marlin_correctness.cu` (working tree) | 487-line standalone microbench, 14 test cases, all PASS on CUDA 13. Compile cmd in file header. |
| `crates/ferrum-kernels/kernels/moe_align_block_size.cu` (working tree) | The fix. Not committed. |
| `crates/ferrum-engine/Cargo.toml`, `crates/ferrum-models/Cargo.toml` (working tree) | `dep:candle-flash-attn` line removed from `cuda` feature. Not committed. |
| (this file) | Session report — corrects FINDINGS.md's diagnosis. |

`session-2026-05-26/FINDINGS.md` should be marked as superseded by this report. The Update block in `GOAL.md` still cites the session-2026-05-25 numbers — those should be re-baselined after the patched device-route is verified.
