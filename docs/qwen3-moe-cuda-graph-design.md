# Qwen3MoeModel CUDA Graph capture — design

Status: design only, not implemented. The post-#173 baseline (c=32=717 tok/s, TPOT 39ms) is ~38% of vLLM 0.20.1. The canonical next perf gap per `CLAUDE.md`: Qwen3MoeModel decode has no CUDA Graph capture, while `LlamaFamilyModel` has two (unified + piecewise).

## Existing Llama graph patterns

LlamaFamilyModel has TWO CUDA Graph paths:

### 1. Unified graph (`FERRUM_UNIFIED_GRAPH=1`)
- Captures the entire layer loop + final norm + per-item logits slice + lm_head as ONE graph.
- Key: `(1u64 << 63) | (m_total << 32) | num_seqs` — separate graph per concurrency level.
- Warmup: 3 eager calls before first capture (lets dynamic scratch settle).
- State on model: `unified_graph_warmup: usize`, `unified_graph_failed: bool`, `unified_graph_keys_seen: HashSet<u64>`.
- Bench: +5% c=16 (714 vs 680 tok/s on Llama-3.1-8B GPTQ-INT4).

### 2. Piecewise per-layer graph (`FERRUM_BATCHED_GRAPH=1`)
- Captures per-layer "pre-attn" and "post-attn" blocks separately; attention itself stays eager because `kv_len` changes per step.
- L+1 graphs per layer config:
  - graphs[0] = pre_attn_0
  - graphs[i] = post_attn_{i-1} + pre_attn_i
  - graphs[L] = post_attn_{L-1} + final_norm + lm_head
- Better for varying kv_len; attention is the one node that always varies between calls.

## What blocks the port to Qwen3MoeModel

The MoE decode path has two characteristics that Llama doesn't:

### Blocker 1: Per-step expert routing changes the active-expert set
- `route_topk_softmax` produces `expert_ids[batch * top_k]` and `tokens_per_expert[n_exp]`.
- Active experts (`m_e > 0`) vary per token → `phase1_dispatches` list changes shape.
- The number and arrangement of `marlin_gemm_with_offset_strided` calls in `moe_gemm_phase_batched` is data-dependent.

CUDA Graphs require the **same sequence of kernel launches** per replay. We can't capture a graph for "all active experts dispatch" because the dispatch list isn't fixed.

### Mitigations
1. **Fused MoE Marlin (`FERRUM_MOE_FUSED=1`, default ON)** — collapses N per-expert calls into 1-4 bucketed `marlin_gemm_moe` launches. Kernel reads `expert_ids` from device to route blocks. Number of launches is FIXED (1 per `thread_m_blocks` bucket ∈ {1,2,3,4}). **The kernel itself IS capturable.**
2. **vLLM marlin_moe_wna16 (`FERRUM_VLLM_MOE=1`)** — single fused launch per phase. **The kernel itself IS capturable.**

### Blocker 1b (discovered 2026-05-12 in Phase 1 attempt): host-mediated routing
Even with `FERRUM_VLLM_MOE=1`, `moe_forward_bucketed` calls `B::try_gpu_route_topk_into_host` which does **D2H + stream.synchronize** per layer (CUDA impl at `cuda/moe.rs:170-186`). This:
- Reads top-K expert ids/weights to host so the bucket plan + `sorted_token_ids` can be built host-side.
- D2H + synchronize during graph capture raises `CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED`.

**This is the actual blocker, not the per-expert variation.** Even if we resolve the dispatch-count variation via FUSED/VLLM_MOE, the per-layer host→device round-trip kills capture.

### Prerequisite: device-side routing — kernels already exist, just needs plumbing
Investigation 2026-05-12 (post-Phase-1 attempt) — the rewrite is more tractable than feared:

| Need | Have today |
|------|-----------|
| Top-K softmax on device | ✅ `B::route_topk_softmax` — used by `moe_forward_stacked` non-bucketed path. Writes ids[i32] / weights[f32] device buffers. |
| sorted_token_ids / expert_ids / num_tokens_past_padded device-side | ✅ `B::moe_align_block_size` — converts `expert_ids_per_pair` to sorted/block layout entirely on device (CUDA impl at `cuda/moe.rs:190-237`). |
| Per-block expert routing inside kernel | ✅ `marlin_gemm_moe_vllm` (vendored vLLM `marlin_moe_wna16` under `vllm-moe-marlin` feature). |

The blocker is in `moe_forward_bucketed` (`moe/dispatch.rs:1249+`):
1. It calls `B::try_gpu_route_topk_into_host` which does the `route_topk_softmax` kernel BUT ALSO `memcpy_dtoh` + `synchronize` to read results host-side.
2. Bucket plan + sorted_token_ids construction happens host-side (`route_scratch.plan.rebuild_into`, `vllm_routing` builder block around line 1392-1431).
3. Then `B::upload_moe_routing` ships the host-built buffers back to device.

The refactor: replace step 1 with `B::route_topk_softmax` (no D2H), replace steps 2-3 with `B::moe_align_block_size` on the device-side ids. Then the entire MoE phase runs on-device without host round-trips, and the layer loop is capturable.

Scope estimate: ~300-500 lines in `moe/dispatch.rs`, plus possibly a new helper for the `padded_offsets` math that's currently host-side. The bucketed `phase1_dispatches` host-list is no longer needed under VLLM_MOE (the fused kernel routes internally) — so the rewrite can simplify the post-VLLM_MOE branch significantly.

### Additional blocker: moe_combine takes host slices

Investigation continued: `B::moe_combine` signature is `(ctx, packed_down, pairs_by_token: &[i32], pair_weights: &[f32], out, ...)` — both routing slices are host inputs. The CUDA impl `clone_htod`s them inside the function. Inside a graph capture, the H2D copy would record stale host pointers (host data is rebuilt each layer) → wrong results on replay.

Two ways to resolve:
1. Add device buffers `pairs_by_token_dev` / `pair_weights_dev` on the route scratch (built once per `decode_batch` from `route_topk_softmax` output via a new on-device kernel) and pass them through to `moe_combine`.
2. Replace `moe_combine` with a fused write inside the down-proj kernel (vLLM's `mul_topk_weights=true` path on `marlin_gemm_moe_vllm` does this — multiplies output by topk weights as part of the GEMM).

Option 2 is cleaner but requires the down-proj to know the routing → only works with the VLLM_MOE fused path (which already takes `topk_weights` and `mul_topk_weights` flags). Quick scan: `crates/ferrum-kernels/src/backend/cuda/quant.rs` `moe_gemm_phase_vllm_impl` currently passes `None` for `topk_weights` and `false` for `mul_topk_weights`. Flipping those on, with the device-side weights buffer, drops the entire moe_combine step.

### Revised scope

The full refactor to enable MoE CUDA Graph capture:
1. Device-side routing path in `moe_forward_bucketed` (skip `try_gpu_route_topk_into_host`, use `route_topk_softmax` + `moe_align_block_size`).
2. Device-side `pair_weights` buffer feeding `moe_gemm_phase_vllm`'s `topk_weights` arg, with `mul_topk_weights=true` — removes `moe_combine` from the layer.
3. Then re-enable the Phase 1 graph scaffold.

Total: probably 800-1200 lines, **multi-day focused**. Worth doing because it unlocks the path to vLLM-parity perf on MoE — but not autonomous-tick scope. Defer until user prioritises perf over other roadmap items.

### Blocker 2: Pre-grow scratch before begin_capture
LlamaFamilyModel uses `B::pregrow_marlin_gather_scratch(m_total * intermediate_size)` to eagerly grow the marlin scratch slot. Same pattern needed for MoE:
- Marlin gather scratch
- `vllm_moe_c_tmp` scratch (fp32 reduce)
- `moe_route_ids` / `moe_route_weights` (already typed CudaBuf::I32/F32 post Phase D)
- `paged_attn_out_tm` (already typed post B-2)

All currently lazy-grow inside the impl. Need pre-grow helpers OR pre-warm with the worst-case shape during warmup phase.

### Blocker 3: Attention stays eager
The `paged_decode_attention` call reads `valid_kv_lens` from device — same kv_len problem as Llama. The piecewise pattern is the right model: capture pre-attn + post-attn, run attention eager between.

## Recommended implementation

Match LlamaFamilyModel's **piecewise** pattern (capture-around-attention), not unified. Reasons:
- MoE's per-step active-expert variation means even fused dispatches read device-side `expert_ids` — only the FIXED-shape MoE kernels can capture.
- The attention call is already-eager; mirroring that boundary is the lowest-risk port.
- Piecewise allows incremental rollout (capture L=0 first, expand).

### Step plan
1. **Add state to Qwen3MoeModel**: `batched_graph_warmup`, `batched_graph_failed`, `batched_graph_pre_attn_seen: HashSet<u64>`, `batched_graph_post_attn_seen: HashSet<u64>`.
2. **Add pre-grow methods**: `Backend::pregrow_moe_scratch(m_total, n_experts, top_k, intermediate)` — eagerly size vllm_moe_c_tmp + moe_route_ids/weights + paged_attn_out_tm.
3. **Split `forward_layer_batched_decode` into `pre_attn` + `post_attn`** mirrors Llama. Bottom of pre_attn = before paged_decode_attention; top of post_attn = right after.
4. **Wrap with capture/replay** keyed on `(m_total, num_seqs, layer_idx)`.
5. **Bench gate**: ship behind `FERRUM_MOE_GRAPH=1` (off by default). Validate +X% TPOT improvement before flipping default.

### Expected gain
- LlamaFamily piecewise gives ~+5% at c=16. MoE has more launches per layer (MoE GEMMs + silu + combine) so the launch-overhead reduction should be proportionally larger, possibly +15-25% at c=32.
- This is what would push Qwen3-30B-A3B from current ~38% of vLLM to ~50%.

## Estimated effort
- Code change: ~500-800 lines (state + pre-grow + capture wrappers).
- Validation: 2-4 hours (warmup tuning, bench parity at each concurrency level, regression on Q4_K_M Metal too).
- **Total: ~2-3 days focused** before it's worth flipping the default.

## Risk
- CUDA Graph capture has many invariants — any kernel that calls a runtime alloc inside the captured stream returns `CUDA_ERROR_INVALID_VALUE` (this is what bit Llama's marlin gather scratch first time). Pre-grow EVERYTHING.
- The capture must run AFTER warmup; the first 3 eager calls are not just for cache settling — they're to ensure all lazy allocs have fired.
- Recovery path needed: if capture fails, set `batched_graph_failed=true` and never retry (avoid log spam). Llama has this; mirror it.
