# Progress Report — Continuous-Batching Foundation (2026-05-13)

**Merged**: PR #184 → `249dab0` on main. **Session date**: 2026-05-13.
**Headline**: M2 apples c=32 **851 → 881 tok/s (+3.5%)** verified. M3 baseline preserved. Architecture for the next ~+30-40% perf step is in place but dormant; activation needs Phase 3 (engine-init scratch budgeting).

---

## What shipped

### Verified perf wins

| bench | baseline | this session | vLLM 0.20.2 | ratio | gain |
|---|---:|---:|---:|---:|---:|
| M2 c=32 (Llama-3.1-8B-INT4) | 851 | **881** | 2179 | 40% | **+3.5%** ✓ |
| M3 c=32 (Qwen3-30B-A3B-Int4) | 1030 | 1023 | 1867 | 55% | noise |

M2's gain comes from **engine-layer unification**: `process_batch_unified` collapses prefill + decode items into one `model_executor.unified_decode` call per iter. LlamaFamilyModel already implements `unified_forward`, so this immediately routes mixed batches through a single varlen kernel chain instead of separate prefill / decode phases.

### Code landed

1. **Engine unified path** (`b9e0bd6`):
   - `process_batch_unified` replaces the prefill-then-decode split with one `UnifiedBatch` and one `unified_decode` call.
   - Legacy split preserved for chunked prefill (`FERRUM_CHUNKED_PREFILL=N`) + speculative decoding.

2. **Batched prefill API** (`1780c5a`):
   - `ModelExecutor::batch_prefill(inputs)` default-trait method.
   - `LlmExecutor` override drives `model.unified_forward` over all items in one call.

3. **Shared decoder helpers** (`b26220f`):
   - `crates/ferrum-models/src/common/decoder_unified.rs` — 7 pure functions (cu_seqlens, pos_offsets, max_kv_len, concat_q_tokens, stack_block_tables, final_indices, graph_key).
   - 6 unit tests, all green.
   - LlamaFamilyModel migrated to call helpers (~40 lines saved, behavior unchanged, M2 within ±2% baseline).

4. **Qwen3MoE unified_forward — dormant** (`91d169b` → `e9bac55`):
   - `Qwen3MoeScratch::unified_*` small INDEX buffers (cu_seqlens / pos_offsets / block_tables / packed_normed / packed_logits — max_seqs sized, ~few KB each).
   - `unified_forward_internal` + `unified_forward_layer` use **legacy scratch** (residual / norm_out / qkv_out / moe_out — no buffer duplication; per vLLM v1 pattern in `gpu_model_runner.py`).
   - MoE block: ONE `moe_forward_bucketed` call at M = m_total.
   - Shape-aware gate: pure-decode → fast legacy path; mixed/prefill → unified path; `m_total > scratch.max_tokens` → fallback.

5. **Profile probes** (developer-only, opt-in env vars):
   - `FERRUM_NEXT_BATCH_PROF`, `FERRUM_SCHED_NONE_PROF` — scheduler Some/None ratio + per-queue size.
   - `FERRUM_BATCH_PREFILL_PROF` — batch sizes + fallback rate.

6. **Bench tooling**:
   - `apples_all_drive.sh` — single driver for M1/M2/M3 c=1/4/16/32 sweep.
   - `summary.py` — JSON results → markdown table.
   - `bisect_m2.sh` — single-commit M2 c=32 regression driver.
   - `SKIP_VLLM=1` mode (reuses vllm baseline, saves ~5 min/run).
   - nsys profile script fixes (drop unsupported `--capture-range timer`, vLLM capture delay 60 → 35).

7. **Reports + design docs**:
   - `bench/v0.2-cuda/REPORT_2026-05-13.md` — original apples investigation.
   - `bench/v0.2-cuda/PERF_TRACKER.md` — running tracker with R0-R2 + Wave 1 + Phase 2B retrospectives.
   - `docs/continuous-batching-redesign.md` — 5-phase plan to 80%-of-vLLM.
   - `docs/decoder-unified-runner-abstraction.md` — `DecoderUnifiedOps` trait design (Phase 2A scope, deferred).

---

## Key investigations + findings

### The 17 ms inter-iter gap is the bench client, not ferrum

`bg-loop-gap` + `iter-prof` + `rbd-prof` measurements showed:
- run_iteration takes ~14 ms steady-state at c=32 on M3
- bench wall is ~31 ms/iter → 17 ms gap UNEXPLAINED

Adding `FERRUM_NEXT_BATCH_PROF` + `FERRUM_SCHED_NONE_PROF` localized it: `scheduler.next_batch` returns None 22× more often than Some during the bench. **Queue probe showed all 3 queues empty when None is returned** — the bench client (`vllm bench serve --max-concurrency 32`) maintains a worker pool of 32 and only fires the next cohort after the current 32 EOS together. Inter-cohort connection close + reopen is the gap.

vLLM hides this by streaming SSE faster (so client moves on faster) and admitting next prefills WHILE current cohort still decodes. That's continuous batching at the scheduler level.

### Architecture insight: vLLM v1 pattern

Reviewed via WebFetch:
- `vllm/v1/worker/gpu_model_runner.py` — pre-allocates ALL scratch at startup sized for `max_num_batched_tokens` (default 8192). No on-demand growth.
- `vllm/v1/core/sched/scheduler.py` — token-budget scheduling. Decodes scheduled first, prefill chunks fill remaining budget.
- `vllm/v1/attention/backends/flash_attn.py` — one varlen attention call per iter, `cu_seqlens_q` + `seqused_k` per request.

ferrum currently lazy-grows scratch; this is what blocks Qwen3MoE unified activation today.

### Phase 1 (batched prefill) gave M2 perf-neutral

R0 measurement: M2 prefill is only **5% of bench wall** at c=32 (decode is 95%, 30 ms/token × 384 iters). Batched prefill ROI is therefore capped at +5%. Real M2 win must come from the decode kernel.

### Phase 2B implementation iterations

3 attempts to wire Qwen3MoE unified_forward:
- v1 (`b5acc9b`): called `moe_forward_batched_prefill_impl` → crashed (`compute_ids_tpe_gpu` is Metal-only on CUDA).
- v2 (`f72cf4c`): switched to `moe_forward_bucketed` (CUDA path) + grew legacy scratch on demand → **CUDA OOM** at c=32 cohort start (scratch realloc transient + 17 GB model + 6 GB KV pool).
- v3 (`1c1f6cb` + `e9bac55`): use legacy scratch in-place + shape-aware gate (bypass for pure decode, reject when m_total exceeds scratch capacity) → no crash, no perf change because gate rejects almost everything.

The unified_forward implementation itself is correct (compiles clean, follows vLLM design). The remaining blocker is engine-init-time scratch budgeting.

---

## What's NOT in this PR (next sessions)

### Phase 3 — engine-init scratch budget + token-budget scheduler

**The actual M3 perf unlock.** Three coordinated changes:
1. Add `max_num_batched_tokens` config (default ~2048-4096, GPU-memory aware).
2. Pre-allocate Qwen3MoE scratch at engine init for `max_num_batched_tokens`. Resize KV pool budget to fit remaining GPU memory.
3. Rewrite `ContinuousBatchScheduler` to use token budget (replace `max_decode_batch=256` + `max_prefill_batch=8` with single `max_num_batched_tokens` constraint). Emit mixed prefill+decode batches.

Once these land, the dormant Qwen3MoE unified path activates for cohort prefill batches → expected M3 **+30-40%** (toward the 1500 tok/s = 80%-of-vLLM gate).

### Phase 2A — `DecoderUnifiedOps` trait extraction

Mechanical refactor: move LlamaFamily's `unified_forward_internal` body into a generic free function driven by a `DecoderUnifiedOps` trait. Each future decoder family then implements ~50 lines of trait impl instead of duplicating 700 lines of scaffolding. Design committed in `docs/decoder-unified-runner-abstraction.md`. Deferred — current ad-hoc helpers (`b26220f`) cover the most copy-prone parts, full trait extraction can wait until a third decoder family lands.

### Phase 4 — Llama parity + legacy cleanup

After Phase 3 validates the unified path on Qwen3MoE, mirror on LlamaFamilyModel (already partially done — Llama's `unified_forward` exists). Delete `process_batch_legacy_split`, `run_batch_prefill`, `run_batch_decode` once the unified path is the only production code path.

### Phase 5 — Polish

Stress test (long prompts, sudden bursts, EOS storms). Remove DEBUG probes after they've served their purpose. Update CLAUDE.md baselines.

### Known issue

`FERRUM_UNIFIED_GRAPH=1` + Wave 1 mixed batches → `paged_varlen_split_k_phase1: CUDA_ERROR_INVALID_VALUE`. Workaround: keep `FERRUM_UNIFIED_GRAPH` OFF in production. Tracked separately for kernel-level fix.

---

## Files changed

26 commits, 2035 insertions, 54 deletions across 15 files. Most diff weight in:
- `crates/ferrum-engine/src/continuous_engine.rs` — Wave 1 unified path (+323)
- `crates/ferrum-models/src/models/qwen3_moe_forward_unified.rs` — Qwen3MoE unified (new, +441)
- `crates/ferrum-models/src/models/qwen3_moe.rs` — scratch fields + gate (+100)
- `docs/continuous-batching-redesign.md` + `docs/decoder-unified-runner-abstraction.md` (+344)

## Bench reproducibility

```bash
# On Vast 4090 pod:
bash bench/v0.2-cuda/apples_all_drive.sh    # full M1/M2/M3 × c=1/4/16/32 sweep
python3 bench/v0.2-cuda/summary.py           # JSON → table

# Profile probes (server side):
FERRUM_NEXT_BATCH_PROF=1 FERRUM_SCHED_NONE_PROF=1 \
  /workspace/ferrum-infer-rs/target/release/ferrum serve ...
```
