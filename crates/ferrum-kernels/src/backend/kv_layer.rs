//! `KvLayer<B>` — per-K-dtype trait that picks the cache layout type and
//! the K-specific paged write / read launchers (Dim 5 PR C trait-based
//! dispatch).
//!
//! ## Why a trait, not an enum
//!
//! - `K` carries an associated `Layer` type — FP16 → `KvCache<B, KvFp16>`,
//!   INT8 → `KvCacheQuant<B, KvInt8>`.
//! - K-specific launchers (paged write + paged decode attention; contig
//!   write + contig decode for FP16) are trait methods. The model bound
//!   `where K: KvLayer<B>` lets `K::method(layer, ...)` dispatch directly
//!   to the right backend launcher per (B, K) at monomorphization time —
//!   no runtime tag, no enum match, no panicking accessors.
//! - `LlamaFamilyModel<CpuBackend, KvInt8>` is a compile error because
//!   `KvInt8: KvLayer<CpuBackend>` doesn't hold (CPU backend has no
//!   `BackendInt8KvOps` impl).

use ferrum_types::{FerrumError, Result};

use crate::backend::{Backend, BackendInt8KvOps, KvCache, KvCacheQuant};
use ferrum_interfaces::kv_dtype::{KvDtypeKind, KvFp16, KvInt8};

/// Per-K-dtype dispatch trait.
#[allow(clippy::too_many_arguments)]
pub trait KvLayer<B: Backend>: KvDtypeKind {
    /// Per-layer cache type (FP16 → `KvCache`, INT8 → `KvCacheQuant`).
    type Layer: Send + Sync;

    /// Allocate a paged cache layer for one sequence.
    fn alloc_paged(
        max_blocks_per_seq: usize,
        block_size: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Self::Layer;

    /// Allocate a contiguous cache layer (FP16 only; INT8 panics).
    fn alloc_contig(
        capacity: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Self::Layer;

    // Metadata accessors (variant-agnostic).
    fn len(layer: &Self::Layer) -> usize;
    fn set_len(layer: &mut Self::Layer, new_len: usize);
    fn capacity(layer: &Self::Layer) -> usize;
    fn block_size(layer: &Self::Layer) -> usize;
    fn num_kv_heads(layer: &Self::Layer) -> usize;
    fn head_dim(layer: &Self::Layer) -> usize;
    fn block_table(layer: &Self::Layer) -> Option<&B::Buffer>;
    fn block_table_mut(layer: &mut Self::Layer) -> Option<&mut B::Buffer>;
    fn context_lens(layer: &Self::Layer) -> Option<&B::Buffer>;
    fn context_lens_mut(layer: &mut Self::Layer) -> Option<&mut B::Buffer>;
    fn paged_block_indices(layer: &Self::Layer) -> &[u32];
    fn paged_block_indices_mut(layer: &mut Self::Layer) -> &mut Vec<u32>;

    fn is_paged(layer: &Self::Layer) -> bool {
        Self::block_size(layer) > 0
    }

    /// Paged write: split QKV → norm → RoPE → write K/V into the paged
    /// pool. FP16 uses `B::split_qkv_norm_rope_into_paged_cache`. INT8
    /// uses `B::split_qkv_norm_rope` + `B::int8_kv_append_paged`.
    fn paged_write(
        ctx: &mut B::Context,
        layer: &mut Self::Layer,
        qkv: &B::Buffer,
        q_norm_w: &B::Buffer,
        k_norm_w: &B::Buffer,
        cos: &B::Buffer,
        sin: &B::Buffer,
        q_out: &mut B::Buffer,
        k_scratch: &mut B::Buffer,
        v_scratch: &mut B::Buffer,
        pool_k: &mut B::Buffer,
        pool_v: &mut B::Buffer,
        tokens: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        pos_offset: usize,
        eps: f32,
        qk_mode: i32,
    ) -> Result<()>;

    /// Paged decode attention. Reads from the per-layer cache, writes the
    /// attended output to `output`. FP16 reads from `pool_k`/`pool_v`;
    /// INT8 reads from layer-internal INT8 buffers (pool args ignored).
    fn paged_decode_attention(
        ctx: &mut B::Context,
        layer: &mut Self::Layer,
        q: &B::Buffer,
        pool_k: &B::Buffer,
        pool_v: &B::Buffer,
        output: &mut B::Buffer,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        final_kv_len: usize,
        tokens: usize,
    ) -> Result<()>;

    /// Contig write: FP16 only. INT8 inherits the panic default —
    /// `KvInt8::alloc_contig` panics in `ensure_kv`, so this branch is
    /// dead code on the INT8 path.
    fn contig_write(
        _ctx: &mut B::Context,
        _layer: &mut Self::Layer,
        _qkv: &B::Buffer,
        _q_norm_w: &B::Buffer,
        _k_norm_w: &B::Buffer,
        _cos: &B::Buffer,
        _sin: &B::Buffer,
        _q_out: &mut B::Buffer,
        _k_scratch: &mut B::Buffer,
        _v_scratch: &mut B::Buffer,
        _q_buf: &mut B::Buffer,
        _k_buf: &mut B::Buffer,
        _v_buf: &mut B::Buffer,
        _tokens: usize,
        _num_q_heads: usize,
        _num_kv_heads: usize,
        _head_dim: usize,
        _pos_offset: usize,
        _eps: f32,
        _qk_mode: i32,
    ) -> Result<()> {
        unimplemented!("contig_write: not supported for this K dtype")
    }

    /// Contig decode attention: FP16 only.
    fn contig_decode_attention(
        _ctx: &mut B::Context,
        _layer: &Self::Layer,
        _q: &B::Buffer,
        _output: &mut B::Buffer,
        _attn_cfg: crate::backend::AttnConfig,
        _tokens: usize,
        _pos_offset: usize,
    ) -> Result<()> {
        unimplemented!("contig_decode_attention: not supported for this K dtype")
    }
}

// ─────────────────────────────────────────────────────────────────────
// FP16 impl
// ─────────────────────────────────────────────────────────────────────

impl<B: Backend + crate::backend::BackendPagedKv> KvLayer<B> for KvFp16 {
    type Layer = KvCache<B, KvFp16>;

    fn alloc_paged(
        max_blocks_per_seq: usize,
        block_size: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Self::Layer {
        let block_table = B::alloc_u32(max_blocks_per_seq);
        let mut context_lens = B::alloc_u32(1);
        let mut bt_ctx = B::new_context();
        B::write_u32(&mut bt_ctx, &mut context_lens, &[0u32]);
        B::sync(&mut bt_ctx);
        KvCache {
            k: B::alloc(1),
            v: B::alloc(1),
            len: 0,
            capacity: max_blocks_per_seq * block_size,
            num_kv_heads,
            head_dim,
            block_size,
            block_table: Some(block_table),
            context_lens: Some(context_lens),
            paged_block_indices: Vec::new(),
            _kv_dtype: std::marker::PhantomData,
        }
    }

    fn alloc_contig(capacity: usize, num_kv_heads: usize, head_dim: usize) -> Self::Layer {
        KvCache {
            k: B::alloc(num_kv_heads * capacity * head_dim),
            v: B::alloc(num_kv_heads * capacity * head_dim),
            len: 0,
            capacity,
            num_kv_heads,
            head_dim,
            block_size: 0,
            block_table: None,
            context_lens: None,
            paged_block_indices: Vec::new(),
            _kv_dtype: std::marker::PhantomData,
        }
    }

    fn len(layer: &Self::Layer) -> usize { layer.len }
    fn set_len(layer: &mut Self::Layer, new_len: usize) { layer.len = new_len; }
    fn capacity(layer: &Self::Layer) -> usize { layer.capacity }
    fn block_size(layer: &Self::Layer) -> usize { layer.block_size }
    fn num_kv_heads(layer: &Self::Layer) -> usize { layer.num_kv_heads }
    fn head_dim(layer: &Self::Layer) -> usize { layer.head_dim }
    fn block_table(layer: &Self::Layer) -> Option<&B::Buffer> { layer.block_table.as_ref() }
    fn block_table_mut(layer: &mut Self::Layer) -> Option<&mut B::Buffer> { layer.block_table.as_mut() }
    fn context_lens(layer: &Self::Layer) -> Option<&B::Buffer> { layer.context_lens.as_ref() }
    fn context_lens_mut(layer: &mut Self::Layer) -> Option<&mut B::Buffer> { layer.context_lens.as_mut() }
    fn paged_block_indices(layer: &Self::Layer) -> &[u32] { &layer.paged_block_indices }
    fn paged_block_indices_mut(layer: &mut Self::Layer) -> &mut Vec<u32> { &mut layer.paged_block_indices }

    fn paged_write(
        ctx: &mut B::Context,
        layer: &mut Self::Layer,
        qkv: &B::Buffer,
        q_norm_w: &B::Buffer,
        k_norm_w: &B::Buffer,
        cos: &B::Buffer,
        sin: &B::Buffer,
        q_out: &mut B::Buffer,
        _k_scratch: &mut B::Buffer,
        _v_scratch: &mut B::Buffer,
        pool_k: &mut B::Buffer,
        pool_v: &mut B::Buffer,
        tokens: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        pos_offset: usize,
        eps: f32,
        qk_mode: i32,
    ) -> Result<()> {
        let block_size = layer.block_size;
        let cache_len_before = layer.len;
        let num_blocks_per_seq = layer.capacity / block_size;
        let bt = layer
            .block_table
            .as_ref()
            .ok_or_else(|| FerrumError::model("FP16 paged_write: missing block_table"))?;
        B::split_qkv_norm_rope_into_paged_cache(
            ctx, qkv, 0, q_norm_w, k_norm_w, cos, sin, q_out, 0,
            pool_k, pool_v, bt,
            tokens, num_q_heads, num_kv_heads, head_dim,
            pos_offset, eps, qk_mode,
            cache_len_before, block_size, num_blocks_per_seq,
        )
    }

    fn paged_decode_attention(
        ctx: &mut B::Context,
        layer: &mut Self::Layer,
        q: &B::Buffer,
        pool_k: &B::Buffer,
        pool_v: &B::Buffer,
        output: &mut B::Buffer,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        final_kv_len: usize,
        tokens: usize,
    ) -> Result<()> {
        let block_size = layer.block_size;
        let num_blocks_per_seq = layer.capacity / block_size;
        let bt_ptr = layer
            .block_table
            .as_ref()
            .ok_or_else(|| FerrumError::model("FP16 paged_decode: missing block_table"))?
            as *const B::Buffer;
        let cl_buf = layer
            .context_lens
            .as_mut()
            .ok_or_else(|| FerrumError::model("FP16 paged_decode: missing context_lens"))?;
        B::write_u32(ctx, cl_buf, &[final_kv_len as u32]);
        // SAFETY: block_table outlives the call.
        let bt = unsafe { &*bt_ptr };
        let cl = layer.context_lens.as_ref().unwrap();
        B::paged_decode_attention(
            ctx, q, pool_k, pool_v, output, bt, cl,
            1, num_q_heads, num_kv_heads, head_dim,
            block_size, num_blocks_per_seq, tokens,
        )
    }

    fn contig_write(
        ctx: &mut B::Context,
        layer: &mut Self::Layer,
        qkv: &B::Buffer,
        q_norm_w: &B::Buffer,
        k_norm_w: &B::Buffer,
        cos: &B::Buffer,
        sin: &B::Buffer,
        q_out: &mut B::Buffer,
        k_scratch: &mut B::Buffer,
        v_scratch: &mut B::Buffer,
        q_buf: &mut B::Buffer,
        k_buf: &mut B::Buffer,
        v_buf: &mut B::Buffer,
        tokens: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        pos_offset: usize,
        eps: f32,
        qk_mode: i32,
    ) -> Result<()> {
        let cache_len_before = layer.len;
        let cache_capacity = layer.capacity;
        let used_into_cache = B::split_qkv_norm_rope_into_cache(
            ctx, qkv, q_norm_w, k_norm_w, cos, sin, q_out,
            &mut layer.k, &mut layer.v,
            tokens, num_q_heads, num_kv_heads, head_dim,
            pos_offset, eps, qk_mode,
            cache_len_before, cache_capacity,
        )
        .is_ok();
        if used_into_cache {
            return Ok(());
        }
        let used_fused_qkv = B::split_qkv_norm_rope(
            ctx, qkv, q_norm_w, k_norm_w, cos, sin,
            q_out, k_scratch, v_scratch,
            tokens, num_q_heads, num_kv_heads, head_dim,
            pos_offset, eps, qk_mode,
        )
        .is_ok();
        if !used_fused_qkv {
            let q_dim = num_q_heads * head_dim;
            let kv_dim = num_kv_heads * head_dim;
            B::split_qkv(ctx, qkv, q_buf, k_buf, v_buf, tokens, q_dim, kv_dim);
            B::qk_norm_rope(ctx, q_buf, q_norm_w, cos, sin, q_out, tokens, num_q_heads, head_dim, pos_offset, eps, qk_mode);
            B::qk_norm_rope(ctx, k_buf, k_norm_w, cos, sin, k_scratch, tokens, num_kv_heads, head_dim, pos_offset, eps, qk_mode);
            B::qk_norm_rope(ctx, v_buf, q_norm_w, cos, sin, v_scratch, tokens, num_kv_heads, head_dim, pos_offset, eps, 0);
        }
        B::kv_cache_append_head_major(
            ctx, &mut layer.k, &mut layer.v,
            cache_len_before, cache_capacity,
            k_scratch, v_scratch,
            tokens, num_kv_heads, head_dim,
        );
        Ok(())
    }

    fn contig_decode_attention(
        ctx: &mut B::Context,
        layer: &Self::Layer,
        q: &B::Buffer,
        output: &mut B::Buffer,
        attn_cfg: crate::backend::AttnConfig,
        tokens: usize,
        pos_offset: usize,
    ) -> Result<()> {
        let kv_len = layer.len;
        B::flash_attention(
            ctx, q, &layer.k, &layer.v, output,
            1, tokens, kv_len, pos_offset, &attn_cfg,
        );
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────
// INT8 impl
// ─────────────────────────────────────────────────────────────────────

impl<B: Backend + BackendInt8KvOps> KvLayer<B> for KvInt8 {
    type Layer = KvCacheQuant<B, KvInt8>;

    fn alloc_paged(
        max_blocks_per_seq: usize,
        block_size: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Self::Layer {
        B::alloc_paged_int8_layer(max_blocks_per_seq, block_size, num_kv_heads, head_dim)
    }

    fn alloc_contig(_capacity: usize, _num_kv_heads: usize, _head_dim: usize) -> Self::Layer {
        panic!("KvInt8::alloc_contig: INT8 KV is paged-only")
    }

    fn len(layer: &Self::Layer) -> usize { layer.len }
    fn set_len(layer: &mut Self::Layer, new_len: usize) { layer.len = new_len; }
    fn capacity(layer: &Self::Layer) -> usize { layer.capacity }
    fn block_size(layer: &Self::Layer) -> usize { layer.block_size }
    fn num_kv_heads(layer: &Self::Layer) -> usize { layer.num_kv_heads }
    fn head_dim(layer: &Self::Layer) -> usize { layer.head_dim }
    fn block_table(layer: &Self::Layer) -> Option<&B::Buffer> { layer.block_table.as_ref() }
    fn block_table_mut(layer: &mut Self::Layer) -> Option<&mut B::Buffer> { layer.block_table.as_mut() }
    fn context_lens(layer: &Self::Layer) -> Option<&B::Buffer> { layer.context_lens.as_ref() }
    fn context_lens_mut(layer: &mut Self::Layer) -> Option<&mut B::Buffer> { layer.context_lens.as_mut() }
    fn paged_block_indices(layer: &Self::Layer) -> &[u32] { &layer.paged_block_indices }
    fn paged_block_indices_mut(layer: &mut Self::Layer) -> &mut Vec<u32> { &mut layer.paged_block_indices }

    fn paged_write(
        ctx: &mut B::Context,
        layer: &mut Self::Layer,
        qkv: &B::Buffer,
        q_norm_w: &B::Buffer,
        k_norm_w: &B::Buffer,
        cos: &B::Buffer,
        sin: &B::Buffer,
        q_out: &mut B::Buffer,
        _k_scratch: &mut B::Buffer,
        _v_scratch: &mut B::Buffer,
        _pool_k: &mut B::Buffer,
        _pool_v: &mut B::Buffer,
        tokens: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        pos_offset: usize,
        eps: f32,
        qk_mode: i32,
    ) -> Result<()> {
        // Single fused launch: split-QKV + norm + RoPE + INT8 quantize +
        // paged-pool write. Replaces the 4-kernel chain (split_qkv +
        // qk_norm_rope×3 + int8_kv_append_paged) used by the legacy path.
        // Backend computes slot addresses inline from `block_table` +
        // `cache_len_before` — no slot_mapping H2D.
        let cache_len_before = layer.len;
        let block_size = layer.block_size;
        let bt_ptr = layer
            .block_table
            .as_ref()
            .ok_or_else(|| FerrumError::model("INT8 paged_write: missing block_table"))?
            as *const B::Buffer;
        // SAFETY: block_table outlives this call; no concurrent access.
        let block_table = unsafe { &*bt_ptr };
        B::fused_split_qkv_norm_rope_into_int8_paged_cache(
            ctx,
            qkv,
            q_norm_w,
            k_norm_w,
            cos,
            sin,
            q_out,
            &mut layer.k,
            &mut layer.v,
            &mut layer.k_scales,
            &mut layer.v_scales,
            block_table,
            tokens,
            num_q_heads,
            num_kv_heads,
            head_dim,
            pos_offset,
            eps,
            qk_mode,
            cache_len_before,
            block_size,
        )
    }

    fn paged_decode_attention(
        ctx: &mut B::Context,
        layer: &mut Self::Layer,
        q: &B::Buffer,
        _pool_k: &B::Buffer,
        _pool_v: &B::Buffer,
        output: &mut B::Buffer,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        final_kv_len: usize,
        _tokens: usize,
    ) -> Result<()> {
        let block_size = layer.block_size;
        let cl_buf = layer
            .context_lens
            .as_mut()
            .ok_or_else(|| FerrumError::model("INT8 paged_decode: missing context_lens"))?;
        B::write_u32(ctx, cl_buf, &[final_kv_len as u32]);
        let bt = layer
            .block_table
            .as_ref()
            .ok_or_else(|| FerrumError::model("INT8 paged_decode: missing block_table"))?;
        let scale = (head_dim as f32).sqrt().recip();
        B::int8_paged_decode_attention(
            ctx, q,
            &layer.k, &layer.v, &layer.k_scales, &layer.v_scales,
            bt, output,
            num_q_heads, num_kv_heads, head_dim,
            final_kv_len, block_size, scale,
        )
    }
}
