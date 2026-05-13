# DecoderUnifiedRunner — shared continuous-batching pipeline

**Status**: design, 2026-05-13
**Why**: today's `LlamaFamilyModel::unified_forward_internal` is ~700 lines of model-specific code, but **~80% of it is the same scaffolding** every decoder needs (cu_seqlens setup, embedding, layer loop, final norm + lm_head, graph capture). Implementing `Qwen3MoeModel::unified_forward_internal` by copy-paste would duplicate this scaffolding. Future families (any new decoder-only LLM) would copy it again. This is an architectural failure.

## What's shared vs what's family-specific

Steps in a unified mixed-batch forward:

| step | description | family-specific? |
|---|---|---:|
| 1 | Compute `q_lens`, `cu_seqlens_q`, `m_total`, `pos_offsets`, `max_kv_len` | **no** |
| 2 | `ensure_unified_scratch(m_total, max_seqs, max_blocks_per_seq)` | **no** (different fields per family) |
| 3 | Concat all q-tokens, `embedding_lookup` into unified residual | **no** |
| 4 | Write `cu_seqlens_q` / `pos_offsets` / stacked `block_tables` to device | **no** |
| 5 | Graph capture/replay key (`(m_total, num_seqs)`), warmup, begin/end_capture | **no** |
| 6 | Layer loop: per-layer forward | **yes** — MoE vs MLP differs |
| 7 | Final `rms_norm` on per-item last tokens | **no** (one buffer name differs) |
| 8 | `lm_head` (slice-by-final-indices), D2H read | **no** |

Only step 6 (and the FFN block inside it) is family-specific. Steps 1-5, 7-8 are pure scaffolding that should live ONCE.

## Proposed shape — `DecoderUnifiedOps` trait

```rust
/// Per-model hook for the shared unified-forward pipeline. Each
/// decoder-only family (LlamaFamily, Qwen3Moe, future Mistral-MoE, ...)
/// implements this trait once; the outer `decoder_unified_forward` free
/// function drives the cu_seqlens / embedding / layer-loop / final-norm
/// / lm_head scaffolding uniformly.
pub trait DecoderUnifiedOps<B: ...> {
    /// Stable config snapshot the shared pipeline reads.
    fn shape(&self) -> DecoderShape;  // hidden_size, num_heads, kv_heads,
                                       // head_dim, num_layers, vocab_size,
                                       // rms_norm_eps, qk_mode, ...

    /// Ensure scratch buffers (unified_residual, unified_norm_out,
    /// unified_qkv_out, unified_attn_out, unified_o_proj_out, ...) are
    /// sized for `m_total` tokens and `max_seqs` sequences. Idempotent.
    fn ensure_unified_scratch(
        &mut self,
        m_total: usize,
        max_seqs: usize,
        max_blocks_per_seq: usize,
    );

    /// Borrow the shared scratch slots. Each call returns a fresh struct
    /// of `&mut B::Buffer` references so the pipeline can pass them by
    /// reference without re-borrowing through `self` for each step.
    fn unified_scratch(&mut self) -> UnifiedScratchView<'_, B>;

    /// KV cache discovery. `ensure_kv(cid)` is idempotent allocation;
    /// `block_indices(cid, layer)` returns the per-(seq, layer) page
    /// table that gets stacked into the unified `block_tables` buffer.
    fn ensure_kv(&mut self, cid: &str);
    fn block_indices(&self, cid: &str, layer: usize) -> &[u32];
    fn paged_pools_mut(&mut self, layer: usize) -> (&mut B::Buffer, &mut B::Buffer);

    /// `embedding_lookup` over `tokens` into `dst`. Most models delegate
    /// to `B::embedding_lookup(ctx, &self.embed, tokens, dst, h)`.
    fn embedding_lookup(&self, ctx: &mut B::Context, tokens: &[u32], dst: &mut B::Buffer);

    /// Per-layer forward. THIS is the only family-specific piece.
    /// LlamaFamily impl: attention path (shared) + MLP block (gate_up,
    /// silu, down, residual).
    /// Qwen3Moe impl: attention path (shared) + MoE block (route,
    /// expert dispatch, combine, residual).
    /// The shared attention path could be extracted further into a
    /// `attention_step()` helper if we want even less duplication.
    fn forward_layer_unified(
        &mut self,
        ctx: &mut B::Context,
        li: usize,
        m_total: usize,
        num_seqs: usize,
        max_kv_len: usize,
        max_blocks_per_seq: usize,
        block_size: usize,
        residual: &mut B::Buffer,
    ) -> Result<()>;

    /// Final `rms_norm` + `lm_head` on the last token of each
    /// `is_final_chunk` item. Returns host-side logits per item (None
    /// for items whose final_chunk is false — they only advanced KV).
    fn finalize_unified(
        &mut self,
        ctx: &mut B::Context,
        residual: &B::Buffer,
        cu_seqlens_q: &[u32],
        items: &[(String, Vec<u32>, usize, bool)],
    ) -> Result<Vec<Option<Vec<f32>>>>;

    /// CUDA graph cache state (key set, warmup counter, failed flag).
    /// Default impl returns "graph disabled" — opt-in per model.
    fn unified_graph_state(&mut self) -> Option<&mut UnifiedGraphState> { None }
}

/// Free function — the shared pipeline. ~120 lines (vs 700-line per-model
/// copy). Reads cfg + scratch via the trait, does setup/embedding/
/// layer-loop/finalize, returns per-item logits.
pub fn decoder_unified_forward<B, M: DecoderUnifiedOps<B>>(
    model: &mut M,
    items: &[(String, Vec<u32>, usize, bool)],
) -> Vec<Option<Vec<f32>>> {
    // 1-5: setup, embed, index upload, graph capture/replay decision
    // 6: for li in 0..num_layers { model.forward_layer_unified(...) }
    // 7-8: model.finalize_unified(...)
}
```

## Migration plan

Step 1 — **Extract**:
- Pull Llama's `unified_forward_internal` body into a free function
  `decoder_unified_forward<M: DecoderUnifiedOps<B>>(model, items)`.
- Define `DecoderUnifiedOps`, `DecoderShape`, `UnifiedScratchView`,
  `UnifiedGraphState`.
- Implement `DecoderUnifiedOps` for `LlamaFamilyModel`. `forward_layer_unified`
  for Llama is the current `unified_forward_layer` body.

Step 2 — **Validate Llama**:
- Run M2 apples (and the existing CUDA tests) — should be byte-identical.
- Gate: M2 c=32 = 881 tok/s (±2%). No regression.

Step 3 — **Implement for Qwen3Moe**:
- Add the same unified scratch fields to `Qwen3MoeScratch` (residual,
  norm_out, qkv_out, packed_q, attn_out, o_proj_out, etc).
- Implement `DecoderUnifiedOps` for `Qwen3MoeModel`.
- `forward_layer_unified` reuses the attention sub-steps (same kernels
  as Llama — `split_qkv_norm_rope_into_paged_cache_varlen`,
  `paged_varlen_attention`) but calls `moe_forward_batched_prefill_impl`
  for the FFN block instead of MLP gate_up/silu/down.

Step 4 — **Validate Qwen3Moe**:
- Run M3 apples — bench should be 1500+ tok/s (Phase 2 gate).
- No regression on M1 / older Qwen models.

Step 5 — **(Future families)**: a new decoder family only needs to:
- Define its layer structs
- `impl DecoderUnifiedOps for NewModel { ... }`  — mostly accessor boilerplate
- Write its `forward_layer_unified` (just the attention + family-specific FFN)
- No scaffolding duplication.

## Where attention itself goes

Steps inside `forward_layer_unified` that **don't** differ family-to-family:
- rms_norm (input)
- qkv_proj GEMM
- `split_qkv_norm_rope_into_paged_cache_varlen`
- `paged_varlen_attention`
- o_proj GEMM
- fused_add_rms_norm (post-norm + residual)
- residual_add (after FFN)

These could be lifted further into an `attention_step(...)` helper in
the same module so Qwen3Moe's `forward_layer_unified` becomes:
```rust
fn forward_layer_unified(...) {
    attention_step::<B>(ctx, layer, scratch, ..., qk_mode);
    moe_ffn_step::<B>(ctx, &self.moe_layers[li], scratch, m_total);
}
```
Llama's becomes:
```rust
fn forward_layer_unified(...) {
    attention_step::<B>(ctx, layer, scratch, ..., qk_mode);
    mlp_ffn_step::<B>(ctx, layer, scratch, m_total);
}
```

~30 lines of model-specific code per family. The rest is shared.

## Done-criteria

- `decoder_unified_forward` is in `ferrum-models::common::decoder_unified` (or similar).
- LlamaFamily + Qwen3Moe both call it, no duplicated scaffolding.
- M2 c=32 ≥ baseline 881 tok/s (Llama path regression check)
- M3 c=32 ≥ 1500 tok/s (Phase 2 gate, MoE path live)
- New family adding-doc shows a ~50-line `DecoderUnifiedOps` impl is sufficient
