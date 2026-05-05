# Chunked Prefill — Design Doc

**Status:** Steps 1–5a + Step 4 landed (commits `f450eb9`, `0e49370`,
`957d0e5`, `00509e9`, `3cad614`). Step 5b (real model-side
`unified_forward` impl) and Step 3b (engine cutover) outstanding.

**Goal:** support vLLM v1-style chunked prefill — mix prefill chunks
and decode tokens for different sequences in a single model forward
pass — to recover the ~40% wall-clock that's currently spent in
prefill-blocked decode (per `docs/bench/v0.2-cuda/status-2026-05-05-c16-opprofile.md`).

## What's done

| Step | Layer | Commit | Behaviour |
|---|---|---|---|
| 1 | `ferrum-interfaces` | `f450eb9` | `UnifiedBatch` / `UnifiedBatchItem` types. |
| 2 | `ferrum-interfaces` + `ferrum-models::executor` | `0e49370` | `ModelExecutor::unified_decode` trait + `LlmExecutor` fallback (per-item prefill + batched decode — preserves behaviour). |
| 3a | `ferrum-engine::continuous_engine` | `957d0e5` | `run_batch_decode` dispatches via `unified_decode`. |
| 4 | `ferrum-kernels` | `00509e9` | `paged_varlen_attention` CUDA kernel + Backend trait method (`Err(unsupported)` default, real impl on `CudaBackend`). |
| 5a | `ferrum-models::common::DecoderOnlyLLM` + `LlmExecutor` | `3cad614` | `unified_forward` trait method (default `Err`); executor tries it before falling back. |

Net effect today: `engine.run_batch_decode` calls `unified_decode`,
which tries `model.unified_forward` (returns `Err`), then falls back
to the legacy per-item prefill + batched decode dispatch. Behaviour
unchanged; perf unchanged. **The plumbing is in place.**

## What Step 5b needs to do

Implement `LlamaFamilyModel<B>::unified_forward` on top of the new
`paged_varlen_attention` kernel. This is the kernel-level perf unlock.

### Signature

```rust
impl<B: Backend> LlamaFamilyModel<B> {
    pub(crate) fn unified_forward_internal(
        &mut self,
        items: &[(String, Vec<u32>, usize, bool)],
    ) -> Vec<Option<Vec<f32>>> { ... }
}

impl<B: Backend> DecoderOnlyLLM for LlamaFamilyModel<B> {
    fn unified_forward(
        &mut self,
        items: &[(String, Vec<u32>, usize, bool)],
    ) -> Result<Vec<Option<Vec<f32>>>> {
        // Bail out cleanly on the contig-KV path until we add a contig
        // varlen kernel — the engine fallback handles it.
        if self.paged_pools.is_none() {
            return Err(FerrumError::unsupported(
                "unified_forward requires paged KV; engine will fall back",
            ));
        }
        Ok(self.unified_forward_internal(items))
    }
}
```

### Per-iter algorithm

Let `M_total = sum(items[i].q_tokens.len())`.

1. **Build per-item bookkeeping** (host side):
   - `q_lens[num_seqs]` = per-item q_token counts.
   - `cu_seqlens_q[num_seqs+1]` = prefix sum (0, q_lens[0], q_lens[0]+q_lens[1], …).
   - `pos_offsets[num_seqs]` = each item's `pos_offset`.
   - For each item, look up `cache.block_table` and build a stacked
     `block_tables[num_seqs * max_blocks_per_seq]` host array.
   - `final_chunk_token_idx[num_sampled]` = list of global token
     positions where logits should be extracted (last token of each
     `is_final_chunk` item).

2. **Upload to scratch** (one-time per iter):
   - `B::write_u32(scratch.cu_seqlens_q, &cu_seqlens_q)`
   - `B::write_u32(scratch.pos_offsets, &pos_offsets)`
   - `B::write_u32(scratch.block_tables, &block_tables_flat)`
   - `B::write_u32(scratch.final_chunk_token_idx, &final_chunk_token_idx)`

3. **Embed all q-tokens at once**:
   - Flat `tokens[M_total]` array.
   - `B::embedding_lookup(embed, tokens, &mut residual_M_total, h)` —
     produces `[M_total, h]` directly.

4. **Layer loop** (32× for Llama-3.1-8B):
   - `B::rms_norm(residual, ln_w, eps, norm_out, M_total, h)` — kernel
     is M-naive.
   - `qkv_proj.forward(norm_out, qkv_out, M_total)` — single GEMM with
     m=M_total.
   - `B::split_qkv(qkv_out, q_buf, k_buf, v_buf, M_total, q_dim, kv_dim)`.
   - **Per-seq `qk_norm_rope`** loop. Current `qk_norm_rope` takes
     uniform `pos_offset` for `tokens` consecutive positions, so for
     each item `(start, end, pos_offset)` from `cu_seqlens_q` and
     `pos_offsets`, slice `q_buf[start*nh*hd .. end*nh*hd]` and call
     `qk_norm_rope(slice, q_norm_w, cos, sin, q_normed_slice,
                   end-start, nh, hd, pos_offset, eps, qk_mode)`.
     Same for K and V. Total launches/layer: `3 * num_seqs` (=48 at
     c=16) — small. *Future optimisation: write a varlen variant that
     consumes `cu_seqlens_q` + `pos_offsets` and does one launch.*
   - **Per-seq `kv_cache_append`** loop. Each item appends its
     `q_len` K/V tokens at its `pos_offset`. Use existing
     `B::kv_cache_append_head_major` per item.
   - `B::paged_varlen_attention(q_normed, k_pool, v_pool, attn_out,
        cu_seqlens_q, pos_offsets, block_tables, num_seqs, M_total,
        max_kv_len, nh, nkv, hd, block_size, max_blocks_per_seq)`.
     Single launch (Step 4 kernel).
   - `o_proj.forward(attn_out, o_proj_out, M_total)`.
   - `B::fused_add_rms_norm(residual, o_proj_out, post_ln_w, eps,
        norm_out, residual_post, M_total, h)`.
   - `gate_up_proj.forward(...)` / `B::fused_silu_mul_split(...)` /
     `down_proj.forward(...)` / `B::add_inplace(residual, mlp_out,
                                                M_total * h)`.

5. **Final norm**: `B::rms_norm(residual, final_norm_w, eps,
     last_normed, M_total, h)`.

6. **Logits extraction** — only at last token of each
   `is_final_chunk` item:
   - `M_extract = num_sampled_items` (= count of `is_final_chunk`).
   - Gather rows from `last_normed` indexed by `final_chunk_token_idx`
     into a packed `[M_extract, h]` buffer (need a small
     `gather_rows` kernel — could reuse existing `gather_columns`
     transposed).
   - `lm_head.forward(packed_normed, packed_logits, M_extract)`.
   - `B::to_vec(packed_logits, M_extract * vocab)` →
     `Vec<Vec<f32>>` (split into per-item slices).

7. **Build return**: `Vec<Option<Vec<f32>>>` aligned to `items`. Walk
   `items` and pop from extracted logits list whenever
   `is_final_chunk == true`, else `None`.

### Scratch buffer additions to `LlmScratch<B>`

```rust
// In LlmScratch:
pub unified_residual: Option<B::Buffer>,        // [M_total_max, h] f16
pub unified_qkv_out: Option<B::Buffer>,
pub unified_q_buf: Option<B::Buffer>,
pub unified_k_buf: Option<B::Buffer>,
pub unified_v_buf: Option<B::Buffer>,
pub unified_q_normed: Option<B::Buffer>,
pub unified_attn_out: Option<B::Buffer>,
pub unified_o_proj_out: Option<B::Buffer>,
pub unified_norm_out: Option<B::Buffer>,
pub unified_gate_up_out: Option<B::Buffer>,
pub unified_silu_out: Option<B::Buffer>,
pub unified_mlp_out: Option<B::Buffer>,

// Index buffers (i32 stored in f16 buffer per existing convention)
pub unified_cu_seqlens_q: Option<B::Buffer>,    // [max_seqs+1]
pub unified_pos_offsets: Option<B::Buffer>,     // [max_seqs]
pub unified_block_tables: Option<B::Buffer>,    // [max_seqs * max_blocks_per_seq]
pub unified_final_idx: Option<B::Buffer>,       // [max_seqs]

pub unified_packed_normed: Option<B::Buffer>,   // [max_seqs, h]
pub unified_packed_logits: Option<B::Buffer>,   // [max_seqs * vocab]
```

`ensure_unified_scratch(M_total_max)` allocates / grows these on
demand, similar to existing `ensure_scratch` pattern.

### Estimated effort

| Sub-task | LoC | Notes |
|---|---|---|
| Scratch struct + ensure_unified_scratch | ~80 | mechanical |
| `unified_forward_internal` body | ~250 | main work; per-seq dispatch loops in qkr / kv_append are the verbose parts |
| `gather_rows` kernel + launcher | ~80 | small dedicated kernel for logits extraction |
| Tests: c=1 unified_forward content matches `prefill_internal`+`decode_batch_internal` byte-for-byte | ~150 | correctness gate |
| **Total** | **~560** | 1 week focused work, on a GPU box |

### Open questions

1. **Paged-on-CUDA**: `Backend::supports_paged_kv()` returns false for
   `CudaBackend`, so the paged path is currently Metal-only by
   default. To exercise `unified_forward` on CUDA we either:
   - Flip `supports_paged_kv()` to true on CUDA (risky — paged code
     paths haven't been validated end-to-end on CUDA), or
   - Add a `ferrum.toml` setting `engine.paged_kv = true|false|auto`
     and let the user opt in.
   The latter is cleaner; doable as a follow-up.

2. **varlen `qk_norm_rope`**: per-seq dispatch is correct but adds
   `3 * num_seqs * num_layers` launches per iter. At c=16 + 32 layers
   that's 1536 extra launches ≈ 8 ms host dispatch (cudarc async →
   probably overlapped with GPU work; profile to confirm). Future
   optimisation if profile shows it bound: write a varlen
   `qk_norm_rope` that takes `cu_seqlens_q + pos_offsets` and does
   one launch.

3. **`gather_rows` kernel**: cheap but new. Could also implement
   logits-extract via N small `copy_slice` calls (one per
   `is_final_chunk` item) — slower for large num_sampled but simpler.
   Pick the simpler path for first cut.

## After Step 5b: Step 3b (engine cutover)

Currently `engine.process_batch` still loops `run_prefill` per
prefill_id then `run_batch_decode`. Step 3b consolidates: each engine
iter builds ONE `UnifiedBatch` containing prefill chunks + decode
items, calls `unified_decode` once. The model's `unified_forward`
processes them in a single forward; chunked-prefill perf unlocks.

Estimated effort: ~200 LoC, 2 days. Depends on Step 5b being correct.

## Bench validation

Once Step 5b + 3b land:

| Metric | Pre-chunked-prefill | Expected post |
|---|---|---|
| c=16 throughput | 520 tok/s | 700–900 tok/s (+30–70%) |
| Decode TPOT during prefill window | blocked → 0 | ~30–50 ms (slower than pure decode but non-zero) |
| Bench wall (32 prompts × 128 tokens) | ~7.9 s | ~5–6 s |

Numbers depend heavily on workload mix. Long-prompt low-output
workloads (vLLM v1's sweet spot) benefit most; our short-prompt
high-output bench is a moderate case.
