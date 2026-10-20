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

use crate::backend::{
    Backend, BackendInt8KvOps, KvCache, KvCacheQuant, KvTokenLayout, QuantizedPagedKvPool,
};
use ferrum_interfaces::kv_dtype::{KvDtypeKind, KvFp16, KvInt8};

/// Type-bound storage and contiguous-replay contract for one backend/KV implementation.
/// Safe code cannot reuse a proof for another backend's Layer/PagedPool methods.
/// This is not permission to capture every KV helper: the paged single-item
/// helpers may upload temporary host values and are eager-only. Batched/unified
/// graph callers use persistent views with their own device-staged operations.
///
/// A proof can be moved without changing either implementation type:
/// ```
/// use ferrum_kernels::backend::{cpu::CpuBackend, Backend, KvGraphCaptureContract, KvLayer};
/// use ferrum_interfaces::kv_dtype::KvFp16;
/// fn same_backend<B: Backend, K: ?Sized>(
///     proof: KvGraphCaptureContract<B, K>,
/// ) -> KvGraphCaptureContract<B, K> { proof }
/// let proof = <KvFp16 as KvLayer<CpuBackend>>::graph_capture_contract().unwrap();
/// let _: KvGraphCaptureContract<CpuBackend, KvFp16> = same_backend(proof);
/// ```
/// Qualification for B1 cannot be borrowed by a different B2 implementation.
/// This example requires no optional GPU feature:
/// ```compile_fail,E0308
/// use ferrum_kernels::backend::{Backend, KvGraphCaptureContract};
/// fn wrong_backend<B1: Backend, B2: Backend, K: ?Sized>(
///     proof: KvGraphCaptureContract<B1, K>,
/// ) -> KvGraphCaptureContract<B2, K> { proof }
/// ```
pub struct KvGraphCaptureContract<B: Backend, K: ?Sized> {
    _implementation: std::marker::PhantomData<fn(B, *const K) -> (B, *const K)>,
}

impl<B: Backend, K: ?Sized> KvGraphCaptureContract<B, K> {
    /// # Safety
    /// Persistent device storage exposed by K's KvLayer<B> Layer/PagedPool
    /// must remain owned and stable until checked graph retirement. Growth or
    /// rebinding must retire graphs before freeing their old storage; ordinary
    /// submitted copies keep their source/destination owners through completion.
    ///
    /// If contiguous execution is supported, contig_write and
    /// contig_decode_attention must be replay-equivalent under the backend's
    /// dynamic decode state: no required host side effect or temporary host
    /// upload may be captured. Custom implementations must uphold this constraint.
    /// Batched/unified callers separately qualify the actual backend operations
    /// over the retained views. paged_write and paged_decode_attention are not
    /// capture-qualified by this proof: their eager-only host uploads must stay
    /// outside any capture region.
    pub const unsafe fn new() -> Self {
        Self {
            _implementation: std::marker::PhantomData,
        }
    }
}

/// Per-K-dtype dispatch trait.
#[allow(clippy::too_many_arguments)]
pub trait KvLayer<B: Backend>: KvDtypeKind {
    /// Explicit storage/attention capabilities for this backend and KV dtype.
    /// A custom implementation must opt in to every layout it implements.
    fn supports_contiguous() -> bool {
        false
    }
    fn supports_windowed_paged_attention() -> bool {
        false
    }

    /// Safe custom KV implementations remain eager-only by default.
    fn graph_capture_contract() -> Option<KvGraphCaptureContract<B, Self>> {
        None
    }

    /// Per-layer cache type (FP16 → `KvCache`, INT8 → `KvCacheQuant`).
    type Layer: Send + Sync;

    /// Physical storage shared across sequences in one model layer. Its block
    /// count is the model allocator's capacity, not a sequence's table length.
    type PagedPool: Send + Sync;

    fn alloc_paged_pool(
        ctx: &mut B::Context,
        num_blocks: usize,
        block_size: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Self::PagedPool;

    /// Allocate storage for a paged cache layer for one sequence. The caller
    /// must initialize its block table and context length on its own retained
    /// preparation context before using the layer. Allocator RAII still covers
    /// backend-internal zero initialization.
    fn alloc_paged(
        ctx: &mut B::Context,
        max_blocks_per_seq: usize,
        block_size: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Self::Layer;

    /// Allocate a contiguous cache layer (FP16 only; INT8 panics).
    fn alloc_contig(
        ctx: &mut B::Context,
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

    /// Ensure a contiguous cache has physical room for `required_capacity`
    /// KV positions. Copies are encoded on the caller's retained context;
    /// the caller must observe it before exposing the new capacity as ready.
    /// Captured executables must retire before old buffers can be replaced.
    /// Paged caches and fixed-size cache dtypes may no-op.
    fn ensure_contig_capacity(
        _ctx: &mut B::Context,
        _layer: &mut Self::Layer,
        _required_capacity: usize,
        _num_kv_heads: usize,
        _head_dim: usize,
    ) -> Result<()>
    where
        B: crate::backend::BackendGraph,
    {
        Ok(())
    }

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
        pool: &mut Self::PagedPool,
        tokens: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        pos_offset: usize,
        eps: f32,
        qk_mode: i32,
    ) -> Result<()>;

    /// Paged decode attention. Reads from the per-layer cache, writes the
    /// attended output to `output`. Both dtypes read the physical storage
    /// shared by the model's sequences, through the sequence's block table.
    fn paged_decode_attention(
        ctx: &mut B::Context,
        layer: &mut Self::Layer,
        q: &B::Buffer,
        pool: &Self::PagedPool,
        output: &mut B::Buffer,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        final_kv_len: usize,
        tokens: usize,
        sliding_window: usize,
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

    /// Contig decode attention: FP16 only. The encoded attention extent is
    /// explicit so recording work need not publish a completed host frontier.
    fn contig_decode_attention(
        _ctx: &mut B::Context,
        _layer: &Self::Layer,
        _q: &B::Buffer,
        _output: &mut B::Buffer,
        _attn_cfg: crate::backend::AttnConfig,
        _tokens: usize,
        _pos_offset: usize,
        _final_kv_len: usize,
    ) -> Result<()> {
        unimplemented!("contig_decode_attention: not supported for this K dtype")
    }
}

// ─────────────────────────────────────────────────────────────────────
// FP16 impl
// ─────────────────────────────────────────────────────────────────────

impl<B: Backend + crate::backend::BackendPagedKv> KvLayer<B> for KvFp16 {
    fn supports_contiguous() -> bool {
        true
    }
    fn supports_windowed_paged_attention() -> bool {
        B::supports_windowed_paged_attention()
    }

    fn graph_capture_contract() -> Option<KvGraphCaptureContract<B, Self>> {
        // SAFETY: layer/pool owners retain backing allocations, with checked
        // storage retirement before pointer changes. Contiguous decode delegates
        // to graph-aware backend operations; paged single helpers remain eager.
        Some(unsafe { KvGraphCaptureContract::new() })
    }

    type Layer = KvCache<B, KvFp16>;
    type PagedPool = (B::Buffer, B::Buffer);

    fn alloc_paged_pool(
        ctx: &mut B::Context,
        num_blocks: usize,
        block_size: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Self::PagedPool {
        let elements = num_blocks
            .checked_mul(block_size)
            .and_then(|n| n.checked_mul(num_kv_heads))
            .and_then(|n| n.checked_mul(head_dim))
            .expect("FP16 paged pool dimensions overflow");
        (
            B::alloc_in_context(ctx, elements),
            B::alloc_in_context(ctx, elements),
        )
    }

    fn alloc_paged(
        ctx: &mut B::Context,
        max_blocks_per_seq: usize,
        block_size: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Self::Layer {
        let block_table =
            B::alloc_typed_in_context(ctx, crate::backend::Dtype::U32, max_blocks_per_seq);
        let context_lens = B::alloc_typed_in_context(ctx, crate::backend::Dtype::U32, 1);
        KvCache {
            k: B::alloc_in_context(ctx, 1),
            v: B::alloc_in_context(ctx, 1),
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

    fn alloc_contig(
        ctx: &mut B::Context,
        capacity: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Self::Layer {
        KvCache {
            k: B::alloc_in_context(ctx, num_kv_heads * capacity * head_dim),
            v: B::alloc_in_context(ctx, num_kv_heads * capacity * head_dim),
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

    fn len(layer: &Self::Layer) -> usize {
        layer.len
    }
    fn set_len(layer: &mut Self::Layer, new_len: usize) {
        layer.len = new_len;
    }
    fn capacity(layer: &Self::Layer) -> usize {
        layer.capacity
    }
    fn block_size(layer: &Self::Layer) -> usize {
        layer.block_size
    }
    fn num_kv_heads(layer: &Self::Layer) -> usize {
        layer.num_kv_heads
    }
    fn head_dim(layer: &Self::Layer) -> usize {
        layer.head_dim
    }
    fn block_table(layer: &Self::Layer) -> Option<&B::Buffer> {
        layer.block_table.as_ref()
    }
    fn block_table_mut(layer: &mut Self::Layer) -> Option<&mut B::Buffer> {
        layer.block_table.as_mut()
    }
    fn context_lens(layer: &Self::Layer) -> Option<&B::Buffer> {
        layer.context_lens.as_ref()
    }
    fn context_lens_mut(layer: &mut Self::Layer) -> Option<&mut B::Buffer> {
        layer.context_lens.as_mut()
    }
    fn paged_block_indices(layer: &Self::Layer) -> &[u32] {
        &layer.paged_block_indices
    }
    fn paged_block_indices_mut(layer: &mut Self::Layer) -> &mut Vec<u32> {
        &mut layer.paged_block_indices
    }

    fn ensure_contig_capacity(
        ctx: &mut B::Context,
        layer: &mut Self::Layer,
        required_capacity: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Result<()>
    where
        B: crate::backend::BackendGraph,
    {
        if layer.block_size > 0 || required_capacity <= layer.capacity {
            return Ok(());
        }
        B::retire_graphs_for_storage_mutation(ctx)?;
        let old_capacity = layer.capacity;
        let old_len = layer.len;
        // These owners are outside the unwind boundary: an allocator/copy
        // panic cannot drop a destination that earlier commands may still use.
        let mut new_k = None;
        let mut new_v = None;
        let copied = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            new_k = Some(B::alloc_in_context(
                ctx,
                num_kv_heads * required_capacity * head_dim,
            ));
            new_v = Some(B::alloc_in_context(
                ctx,
                num_kv_heads * required_capacity * head_dim,
            ));
            if old_len > 0 {
                let copy_len = old_len * head_dim;
                for head in 0..num_kv_heads {
                    let old_offset = head * old_capacity * head_dim;
                    let new_offset = head * required_capacity * head_dim;
                    B::copy_slice(
                        ctx,
                        &layer.k,
                        old_offset,
                        new_k.as_mut().unwrap(),
                        new_offset,
                        copy_len,
                    );
                    B::copy_slice(
                        ctx,
                        &layer.v,
                        old_offset,
                        new_v.as_mut().unwrap(),
                        new_offset,
                        copy_len,
                    );
                }
            }
        }));
        if let Err(panic) = copied {
            if let Some(buffer) = new_k {
                B::retain_buffer_until_completion(ctx, buffer);
            }
            if let Some(buffer) = new_v {
                B::retain_buffer_until_completion(ctx, buffer);
            }
            std::panic::resume_unwind(panic);
        }
        // Publish both new buffers only after every copy was encoded. Keep
        // the old sources in the original context until actual completion,
        // including CUDA's non-async-allocation fallback on an unknown result.
        let old_k = std::mem::replace(&mut layer.k, new_k.unwrap());
        B::retain_buffer_until_completion(ctx, old_k);
        let old_v = std::mem::replace(&mut layer.v, new_v.unwrap());
        B::retain_buffer_until_completion(ctx, old_v);
        layer.capacity = required_capacity;
        Ok(())
    }

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
        pool: &mut Self::PagedPool,
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
            ctx,
            qkv,
            0,
            q_norm_w,
            k_norm_w,
            cos,
            sin,
            q_out,
            0,
            &mut pool.0,
            &mut pool.1,
            bt,
            tokens,
            num_q_heads,
            num_kv_heads,
            head_dim,
            pos_offset,
            eps,
            qk_mode,
            cache_len_before,
            block_size,
            num_blocks_per_seq,
        )
    }

    fn paged_decode_attention(
        ctx: &mut B::Context,
        layer: &mut Self::Layer,
        q: &B::Buffer,
        pool: &Self::PagedPool,
        output: &mut B::Buffer,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        final_kv_len: usize,
        tokens: usize,
        sliding_window: usize,
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
        let final_lens = [u32::try_from(final_kv_len)
            .map_err(|_| FerrumError::model("FP16 paged context length exceeds u32"))?];
        // Both views are projected from this exact original cache frontier.
        B::write_typed::<u32>(ctx, cl_buf, &final_lens);
        // SAFETY: block_table outlives the call.
        let bt = unsafe { &*bt_ptr };
        let cl = layer.context_lens.as_ref().unwrap();
        B::paged_decode_attention(
            ctx,
            q,
            &pool.0,
            &pool.1,
            output,
            bt,
            cl,
            1,
            num_q_heads,
            num_kv_heads,
            head_dim,
            block_size,
            num_blocks_per_seq,
            tokens,
            sliding_window,
            Some(&final_lens),
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
            ctx,
            qkv,
            q_norm_w,
            k_norm_w,
            cos,
            sin,
            q_out,
            &mut layer.k,
            &mut layer.v,
            tokens,
            num_q_heads,
            num_kv_heads,
            head_dim,
            pos_offset,
            eps,
            qk_mode,
            cache_len_before,
            cache_capacity,
        )
        .is_ok();
        if used_into_cache {
            return Ok(());
        }
        let used_fused_qkv = B::split_qkv_norm_rope(
            ctx,
            qkv,
            q_norm_w,
            k_norm_w,
            cos,
            sin,
            q_out,
            k_scratch,
            v_scratch,
            tokens,
            num_q_heads,
            num_kv_heads,
            head_dim,
            pos_offset,
            eps,
            qk_mode,
        )
        .is_ok();
        if !used_fused_qkv {
            let q_dim = num_q_heads * head_dim;
            let kv_dim = num_kv_heads * head_dim;
            B::split_qkv(ctx, qkv, q_buf, k_buf, v_buf, tokens, q_dim, kv_dim);
            B::qk_norm_rope(
                ctx,
                q_buf,
                q_norm_w,
                cos,
                sin,
                q_out,
                tokens,
                num_q_heads,
                head_dim,
                pos_offset,
                eps,
                qk_mode,
            );
            B::qk_norm_rope(
                ctx,
                k_buf,
                k_norm_w,
                cos,
                sin,
                k_scratch,
                tokens,
                num_kv_heads,
                head_dim,
                pos_offset,
                eps,
                qk_mode,
            );
            B::qk_norm_rope(
                ctx,
                v_buf,
                q_norm_w,
                cos,
                sin,
                v_scratch,
                tokens,
                num_kv_heads,
                head_dim,
                pos_offset,
                eps,
                0,
            );
        }
        B::kv_cache_append_head_major(
            ctx,
            &mut layer.k,
            &mut layer.v,
            cache_len_before,
            cache_capacity,
            k_scratch,
            v_scratch,
            tokens,
            num_kv_heads,
            head_dim,
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
        final_kv_len: usize,
    ) -> Result<()> {
        B::flash_attention(
            ctx,
            q,
            &layer.k,
            &layer.v,
            output,
            1,
            tokens,
            final_kv_len,
            pos_offset,
            &attn_cfg,
        );
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────
// INT8 impl
// ─────────────────────────────────────────────────────────────────────

impl<B: Backend + BackendInt8KvOps> KvLayer<B> for KvInt8 {
    fn graph_capture_contract() -> Option<KvGraphCaptureContract<B, Self>> {
        // SAFETY: INT8 owns its persistent paged views and has no contiguous
        // path. Batch/unified callers use their graph-aware backend operations;
        // the single-item paged helpers are not capture-qualified.
        Some(unsafe { KvGraphCaptureContract::new() })
    }

    type Layer = KvCacheQuant<B, KvInt8>;
    type PagedPool = QuantizedPagedKvPool<B, KvInt8>;

    fn alloc_paged_pool(
        ctx: &mut B::Context,
        num_blocks: usize,
        block_size: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Self::PagedPool {
        B::alloc_paged_int8_pool(ctx, num_blocks, block_size, num_kv_heads, head_dim)
    }

    fn alloc_paged(
        ctx: &mut B::Context,
        max_blocks_per_seq: usize,
        block_size: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Self::Layer {
        B::alloc_paged_int8_layer(ctx, max_blocks_per_seq, block_size, num_kv_heads, head_dim)
    }

    fn alloc_contig(
        _ctx: &mut B::Context,
        _capacity: usize,
        _num_kv_heads: usize,
        _head_dim: usize,
    ) -> Self::Layer {
        panic!("KvInt8::alloc_contig: INT8 KV is paged-only")
    }

    fn len(layer: &Self::Layer) -> usize {
        layer.len
    }
    fn set_len(layer: &mut Self::Layer, new_len: usize) {
        layer.len = new_len;
    }
    fn capacity(layer: &Self::Layer) -> usize {
        layer.capacity
    }
    fn block_size(layer: &Self::Layer) -> usize {
        layer.block_size
    }
    fn num_kv_heads(layer: &Self::Layer) -> usize {
        layer.num_kv_heads
    }
    fn head_dim(layer: &Self::Layer) -> usize {
        layer.head_dim
    }
    fn block_table(layer: &Self::Layer) -> Option<&B::Buffer> {
        layer.block_table.as_ref()
    }
    fn block_table_mut(layer: &mut Self::Layer) -> Option<&mut B::Buffer> {
        layer.block_table.as_mut()
    }
    fn context_lens(layer: &Self::Layer) -> Option<&B::Buffer> {
        layer.context_lens.as_ref()
    }
    fn context_lens_mut(layer: &mut Self::Layer) -> Option<&mut B::Buffer> {
        layer.context_lens.as_mut()
    }
    fn paged_block_indices(layer: &Self::Layer) -> &[u32] {
        &layer.paged_block_indices
    }
    fn paged_block_indices_mut(layer: &mut Self::Layer) -> &mut Vec<u32> {
        &mut layer.paged_block_indices
    }

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
        pool: &mut Self::PagedPool,
        tokens: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        pos_offset: usize,
        eps: f32,
        qk_mode: i32,
    ) -> Result<()> {
        // 1. split + norm + RoPE → FP16 head-major scratch (k/v_scratch).
        B::split_qkv_norm_rope(
            ctx,
            qkv,
            q_norm_w,
            k_norm_w,
            cos,
            sin,
            q_out,
            k_scratch,
            v_scratch,
            tokens,
            num_q_heads,
            num_kv_heads,
            head_dim,
            pos_offset,
            eps,
            qk_mode,
        )?;
        // 2. quantize FP16 → INT8 + per-token scales, paged append.
        // `paged_block_indices` is the host-side mirror populated at
        // `ensure_kv` time — passing it directly avoids the D2H + sync
        // barrier that would otherwise dominate per-token overhead.
        let cache_len_before = layer.len;
        let block_size = layer.block_size;
        B::int8_kv_append_paged(
            ctx,
            k_scratch,
            v_scratch,
            pool,
            &layer.paged_block_indices,
            cache_len_before,
            tokens,
            KvTokenLayout::HeadMajor,
            block_size,
            num_kv_heads,
            head_dim,
        )
    }

    fn paged_decode_attention(
        ctx: &mut B::Context,
        layer: &mut Self::Layer,
        q: &B::Buffer,
        pool: &Self::PagedPool,
        output: &mut B::Buffer,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        final_kv_len: usize,
        tokens: usize,
        sliding_window: usize,
    ) -> Result<()> {
        if sliding_window != 0 {
            return Err(FerrumError::unsupported(
                "this INT8 KV implementation does not support windowed paged attention",
            ));
        }
        let block_size = layer.block_size;
        let cl_buf = layer
            .context_lens
            .as_mut()
            .ok_or_else(|| FerrumError::model("INT8 paged_decode: missing context_lens"))?;
        B::write_typed::<u32>(ctx, cl_buf, &[final_kv_len as u32]);
        let bt = layer
            .block_table
            .as_ref()
            .ok_or_else(|| FerrumError::model("INT8 paged_decode: missing block_table"))?;
        let scale = (head_dim as f32).sqrt().recip();
        B::int8_paged_decode_attention(
            ctx,
            q,
            pool,
            bt,
            &layer.paged_block_indices,
            output,
            num_q_heads,
            num_kv_heads,
            head_dim,
            final_kv_len,
            tokens,
            KvTokenLayout::HeadMajor,
            block_size,
            scale,
        )
    }
}
