# Continuous Batching Architecture Redesign

**Status**: design, 2026-05-13
**Goal**: bring apples M3 c=32 from 1030 tok/s → ~1700 tok/s (matching vLLM 0.20.2's 1867).

## What's wrong with the current design

`process_batch` (crates/ferrum-engine/src/continuous_engine.rs:379) splits a `BatchPlan` into two disjoint lists and runs them as **two distinct phases**:

```rust
// CURRENT — phase 1: prefill ALL prefill items, one at a time
for rid in &prefill_ids {
    self.run_prefill(rid).await?;  // 47.5 ms each, serial
}
// CURRENT — phase 2: batched decode of any decode items already past prefill
if decode_ids.len() > 1 {
    self.run_batch_decode(&decode_ids).await?;
}
```

Measured cost on M3 c=32 apples (R2 baseline, see PERF_TRACKER.md):
- 128 prefill calls × **47.5 ms avg** = **6.08 sec** total prefill time
- Bench wall = 12 sec
- Prefill is **50% of total wall time** and the only thing happening during that period
- decode_queue alternates 32 ↔ 0 (4 cohorts × 32) because new prompts can't start decoding until ALL prior prompts in their cohort finish prefill

Additional architectural symptoms:
- `model_executor.prefill()` API takes `[batch_size, sequence_length]` but every call passes `batch_size=1` — the multi-batch dimension exists but is unused
- `scheduler.create_iteration_batch` admits up to `max_prefill_batch=8` requests per iter, but the engine still loops over them serially
- `Qwen3MoeModel::forward_layer_batched_decode` assumes `q_len=1` per item in the batch (decode-only path); no mixed-q-len path

## What right looks like (vLLM's design)

vLLM treats each scheduler step as a **single mixed-shape forward pass**:

```
batch = [
  (req_A, kind=PREFILL_CHUNK, q_tokens=512, kv_len=0),
  (req_B, kind=PREFILL_CHUNK, q_tokens=256, kv_len=512),  # 2nd chunk of req_B
  (req_C, kind=DECODE,         q_tokens=1,   kv_len=143),
  (req_D, kind=DECODE,         q_tokens=1,   kv_len=89),
  ...
]
```

ONE kernel call per layer processes all `Σq_tokens` Q tokens, with `cu_seqlens_q` marking the per-request boundary and per-request `kv_len` driving the attention.

Three properties this gives:
1. **No cohort gap**: a new prefill request can enter ANY iter (its prefill chunk shares the batch with ongoing decodes)
2. **No serial prefill**: 32 prefill chunks become one kernel call instead of 32 serial calls
3. **Token-budget scheduling**: `max_num_batched_tokens` is a single budget across all kinds (decode = 1 token each, prefill chunks = many tokens each), avoiding pathological cases

## Proposed ferrum redesign

### New types (ferrum-interfaces)

```rust
// crates/ferrum-interfaces/src/model_executor.rs
pub struct UnifiedBatchItem {
    pub request_id: RequestId,
    pub kv_cache: Arc<dyn KvCacheHandle>,  // per-request paged KV cache
    pub q_tokens: Vec<TokenId>,            // tokens being inputted this iter
    pub kv_len_pre: usize,                 // existing KV length (=0 for first prefill chunk)
    pub is_final_chunk: bool,              // true iff this finishes prefill OR a decode step
}

pub struct UnifiedBatch {
    pub items: Vec<UnifiedBatchItem>,
}

#[async_trait]
pub trait LlmModelExecutor {
    /// New unified forward. Replaces `prefill` + `decode` + `batched_decode`.
    /// Returns one logits tensor per item (last token's logits for prefill,
    /// the single decode token's logits for decode).
    async fn forward_unified(&self, batch: &UnifiedBatch) -> Result<Vec<TensorRef>>;
}
```

### Scheduler changes (ferrum-scheduler)

- Drop the `prefill_queue` / `decode_queue` split into separate dispatch paths. Keep them as internal state for tracking lifecycle but **dispatch a single mixed batch** per call to `next_batch`.
- Replace `max_prefill_batch=8` and `max_decode_batch=256` with **`max_num_batched_tokens`** (token budget). A decode item contributes 1 token; a prefill chunk contributes its chunk size.
- Always include all active decodes (they have priority); fill remaining token budget with prefill chunks from prefill_queue.

### Engine changes (ferrum-engine)

- `process_batch` becomes:
  ```rust
  let unified = build_unified_batch(&batch.requests, &self.sequences, &self.kv_cache);
  let logits_per_item = self.model_executor.forward_unified(&unified).await?;
  // Post-process: sample, update KV pointers, send stream, check stop
  ```
- No separate prefill/decode branches. No serial loops.
- Per-item KV cache handle written through `UnifiedBatchItem` so the kernel routes K/V to the right pool.

### Kernel changes (ferrum-kernels + ferrum-models)

Already-present primitives that get extended:
- `split_qkv_norm_rope_into_paged_cache_varlen` already handles varlen q_tokens for decode (q_len=1 each) — extend to accept q_len>1 per item for prefill chunks
- `paged_batched_flash_dispatch` already handles per-sequence attention with `cu_seqlens_q` + `pos_offsets` + `block_tables` — should work for mixed q_len without change

New work:
- `Qwen3MoeModel::forward_layer_unified` — replaces `forward_layer_batched_decode`. Reads per-item q_len from cu_seqlens, dispatches the same varlen kernels, no special-case for q=1 vs q>1.
- `LlamaFamilyModel::forward_layer_unified` — same idea for dense.
- MoE routing must handle Σq_tokens tokens routed to experts (not 32 fixed). Already varlen-able since `moe_align_block_size_f32` uses cu_seqlens-style indexing.

## Phased implementation plan

Each phase ends with a green bench and a row appended to PERF_TRACKER.md.

### Phase 1 — Interface seam (1 day)
- Define `UnifiedBatchItem` / `UnifiedBatch` types in ferrum-interfaces
- Add `forward_unified` to LlmModelExecutor trait with a default impl that **delegates to existing prefill + batched_decode** (no perf change yet)
- Update engine to construct UnifiedBatch and call forward_unified instead of separate prefill / decode paths
- Existing kernels keep working via the default-impl decomposition
- **Gate**: apples M3 c=32 unchanged (1030 tok/s ±2%)

### Phase 2 — Real unified forward on Qwen3MoeModel (3-5 days)
- Implement `Qwen3MoeExecutor::forward_unified` natively (no decompose to prefill+decode)
- Reuse `split_qkv_norm_rope_into_paged_cache_varlen` for K/V write across mixed q_lens
- Reuse `paged_batched_flash_dispatch` for attention with cu_seqlens
- MoE expert dispatch: confirm `moe_align_block_size_f32` handles m = Σq_tokens for mixed batch
- **Gate**: apples M3 c=32 hits **≥ 1400 tok/s** (= eliminate cohort prefill gap, ~5 sec saving)

### Phase 3 — Scheduler token-budget rewrite (2 days)
- Replace `max_prefill_batch` + `max_decode_batch` with single `max_num_batched_tokens` (default 8192)
- next_batch builds a token-budget-respecting mixed batch (decodes first, fill with prefill chunks)
- Chunked prefill: split long prompts so a single iter's prefill doesn't dwarf decode tokens
- **Gate**: apples M3 c=32 hits **≥ 1700 tok/s** (overlapping prefill with decode, no idle gaps)

### Phase 4 — LlamaFamilyModel parity + cleanup (2 days)
- Mirror Phase 2's unified forward in LlamaFamilyModel (for M1/M2 dense paths)
- Delete `process_batch`'s old prefill/decode split, `run_prefill_inner`, `run_batch_decode`, etc.
- Keep `model_executor.prefill` and `model_executor.batched_decode` only as compatibility shims (or delete if unused)
- **Gate**: apples M2 c=32 hits **≥ 60% of vLLM** (currently 38%)

### Phase 5 — Polish (1 day)
- Stress test: long prompts, sudden bursts, EOS storms
- Update CLAUDE.md baselines + PERF_TRACKER.md run log
- Cleanup feature gates / instrumentation that's served its purpose

**Total**: ~9-13 days serial work, can parallelize Phase 2 + 3 partially.

## Risks + mitigations

| risk | mitigation |
|---|---|
| Varlen prefill kernel correctness on M3 (MoE has 128 experts; expert-routing for mixed q_len is fiddly) | Reuse moe_align_block_size which already handles arbitrary m. Validate against `forward_layer_batched_decode` outputs for q=1 case to confirm we don't regress decode-only path. |
| Phase 1's default-impl shim is slower than current (extra dispatch indirection) | Add a benchmark gate before merging Phase 1 — must be ±2% of baseline. Inline the shim if needed. |
| Scheduler token-budget breaks priority ordering | Keep decode-first priority; only fill remaining budget with prefill. Same semantics as current "decode wins". |
| Chunked prefill regresses TTFT for short prompts | Default chunk_size = 512 (= one prompt fits in one chunk for ShareGPT). Only chunks if prompt > 512 tokens. |

## Open questions

1. **Speculative decoding interaction**: current speculative-decode path also calls `model_executor.prefill`. After Phase 1, does it route through forward_unified? — Yes, same code path. The speculative draft engine becomes a separate UnifiedBatch.
2. **GGUF path**: GGUF models go through `LlamaFamilyModel`. Phase 4 covers this.
3. **KV cache quantization (Dim 5 INT8)**: orthogonal — KV layer K type param flows through unchanged.

## Done-criteria

- `apples-all-drive.sh M3 c=32` reports ferrum at **≥ 80% of vLLM 0.20.2** = ≥ 1500 tok/s on RTX 4090
- PERF_TRACKER.md run log shows the journey
- The old `prefill_ids` / `decode_ids` split is deleted from process_batch
- `cargo test --workspace` passes on Mac (Metal feature) and on Vast pod (cuda + vllm-moe-marlin)
