# Handoff — test-architecture session (2026-06-10)

Honest handoff for the next agent (codex). Records what shipped, what's
verified, what's **NOT** verified, and the open issue, with the testing
caveats called out so nothing is taken on faith.

Branch: `goal/test-architecture-20260610` (all work below is **merged to
main** via PRs #230/#231/#232 except the final matrix-detector commit
`d0660265`, which is on the branch).

---

## 1. What shipped + is verified

### A. Test-architecture goal — `TEST_ARCH GOAL PASS` (PR #230)
The original 3 pain points (Metal/CUDA isolation, manual bug discovery, slow
regression). Gate A (env/cfg/supports/OnceLock = 0), Gate B (10/10 scenarios,
9/9 kills), Gate C (L0 10/10, Metal matrix 8/8, CUDA matrix 4/4, L1-cuda
cold 1906s/warm 18s/3-3 on the RTX 4090 pod). Validator:
`python3 scripts/release/test_arch_goal_gate.py --validate
docs/goals/test-architecture-2026-06-10/evidence/final`.

### B. CUDA MoE fast-path fix (PR #231) — VERIFIED on the pod
`ferrum run` on Qwen3-30B-A3B-GPTQ-Int4 ran the slow host-route MoE at ~9.7
tok/s. Three stacked bugs (NOT the goal migration; pre-migration default is
also 9.7):
1. auto_config gated `FERRUM_VLLM_MOE` on `is_m3_preset()`; `ferrum run` uses
   the serving-default workload → broadened to a capability check
   (`cuda_gptq_moe`: cuda + `model.moe.is_some()` + gptq/int4 + vllm_moe_marlin
   compiled). auto_config.rs.
2. `auto_config.resolve()` returned `runtime_config` as a bare input clone, so
   the resolved knob was a *decision only*, never an effective-config entry →
   now upserts `FERRUM_VLLM_MOE`/`MOE_DEVICE_ROUTE`/`VLLM_MOE_PAIR_IDS`.
3. `ferrum run` applied the raw snapshot, not the resolved auto-config → now
   materializes + applies `startup_auto_config.runtime_config` like `serve`.
**Pod-verified: default `ferrum run` 9.5 → 54.3 tok/s, coherent.**
Unit test `cuda_gptq_moe_enables_vllm_marlin_without_m3_preset` asserts the
config ENTRY, not just the decision.

### C. Greedy repetition-loop fix (PR #232) — VERIFIED on the pod
`ferrum run` chat default was greedy (temperature 0) + `repeat_penalty 1.0`
(no penalty) → deterministic token loops on some inputs (a user hit the
"2D/3D 2D/3D..." degeneration). Default now `1.1` (OpenAI/llama.cpp standard;
the code comment already said so). **Pod-proven on the real 30B: same primed
prompt, `--repeat-penalty 1.0` → "2D/3D" x25, `1.1` → x1.** Independent of
VLLM_MOE (single-turn A/B coherent both ways). `reference_match` uses serve +
explicit params (unaffected); `bench` is a separate command. Regression test
`chat_default_applies_repetition_penalty` parses the real CLI default.

### D. Perf-regression gate + degeneration detector (Gate C7)
- `perf_floors.json` + matrix `perf` tok/s; validator fails a cell below its
  floor. Proven to catch the MoE regression (cuda qwen3-moe 9.7 < 40 → FAIL).
- Matrix runner `max_substr_repeat` (longest back-to-back short-substring run)
  fails token loops the old leak-marker check missed. Commit `d0660265`.

---

## 2. OPEN ISSUE — decode slowdown over context ("越到后面越慢")

Multi-turn chat decays 54 → 29 → 21 → 16 tok/s as context grows. **Not fully
diagnosed; do NOT trust my numbers here — see the testing caveat below.**

### Honest analysis
- Part is inherent (c=1 single-stream decode attends a growing KV).
- The steep drop to ~16 tok/s is the deeper issue: **ferrum's decode attention
  is not yet vLLM-parity** — this is the unfinished **M3 goal**
  (`docs/bench/m3-80pct-goal-2026-05-25/`), where ferrum sits at ~40-55% of
  vLLM. It is NOT a config switch; it is multi-week kernel/graph work.
- `vllm-paged-attn-v2` (VPA, runtime-gated by `FERRUM_USE_VLLM_PAGED_ATTN=1`)
  is the relevant fast path for decode-over-context, but the M3 notes peg it at
  only +3–6%, and **it is a separate opt-in cargo feature, not in the default
  CUDA build.**

### ⚠️ CRITICAL testing caveat (read before trusting any perf number here)
The pod binary I built used `--features cuda,vllm-moe-marlin` — **VPA was NOT
compiled.** So:
- `ferrum serve` with `FERRUM_USE_VLLM_PAGED_ATTN=1` errors
  "vLLM paged attention is not compiled". (Someone has that env var set on the
  pod; unset it or build VPA.)
- My 54.3 tok/s and the 54→16 slowdown were measured on a binary **missing the
  paged-attention path**. They are not representative of a full-feature build.

### Next steps for codex
1. Rebuild the pod binary with the full set:
   `cargo build --release -p ferrum-cli --features cuda,vllm-moe-marlin,vllm-paged-attn-v2`.
2. Broaden `use_vllm_paged_attn` in `auto_config.rs:573` the same way
   `cuda_gptq_moe` broadened `vllm_moe` (and materialize
   `FERRUM_USE_VLLM_PAGED_ATTN` into the resolved config, like §1.B.2/3) — BUT
   verify correctness first (VPA has a c=1 history; Paris smoke is the min gate
   per CLAUDE.md / the M3 memory).
3. Run a PROPER multi-turn test (6+ turns, ~120 tok each) and report per-turn
   tok/s + coherence, on the full-feature binary, VPA on vs off.
4. Be honest with the user: VPA helps modestly; vLLM-parity decode-over-context
   is the M3 goal itself.

---

## 3. TODO left behind
- **Re-run the Metal matrix LLM cells** with the `repeat_penalty=1.1` binary:
  the committed `evidence/final/metal-matrix/llama-family.log` degenerated
  (`..` x13) under the old 1.0 default, so the degeneration detector (d0660265)
  would now FAIL it. The CPU unit test is the real guard; the matrix evidence
  just needs refreshing. (Local Mac binary must be rebuilt with the fix first.)
- The matrix runner's tier-aware length / detector has not been run end-to-end
  on a fresh matrix.

## 4. Reproduction / access
- Pod: Vast instance `40361123`, RTX 4090, `ssh -p 11122 root@ssh9.vast.ai`
  (the saved `Host vast` config points at a stale endpoint; query
  `console.vast.ai/api/v0/instances/` with the key in `.env.local`). Repo at
  `/root/ferrum-infer-rs`. **Still running at ~$0.46/hr — stop when done.**
- cargo on the pod: `/root/.cargo/bin` (not on the non-interactive PATH; nohup
  scripts must `source $HOME/.cargo/env`).
- Local Mac: no GPU; Metal only. `CARGO_TARGET_DIR=/Users/chejinxuan/rust_ws/ferrum-infer-rs/target`.

## 5. Lesson (mine, for the record)
I declared "fast + correct" off a single-turn joke and a binary missing VPA.
Single-turn coherence ≠ multi-turn long-generation correctness, and a partial
feature build ≠ the real engine. Verify on a full-feature build with realistic
multi-turn load before claiming anything.
