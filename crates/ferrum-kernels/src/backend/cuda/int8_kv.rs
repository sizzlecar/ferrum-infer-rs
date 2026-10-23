//! INT8 KV cache (Dim 5) — CUDA backend implementation.
//!
//! Extracted from the monolithic `cuda/mod.rs` for maintainability:
//! - `BackendKvDtype<KvInt8>` marker (`KvBuffer = OptionalCudaInt8`,
//!   `KvScales = OptionalCudaScalesF16`).
//! - `OptionalCudaInt8` / `OptionalCudaScalesF16` lazy-allocated buffers
//!   (Default produces an empty placeholder; real allocation happens
//!   via `alloc(...)` once a CUDA stream is in scope).
//! - `impl BackendInt8KvOps for CudaBackend` — paged INT8 append (FP16 →
//!   INT8 + per-(token, kv_head) FP16 scale) + paged INT8 decode
//!   attention. Both delegate to the launchers in `crate::int8_kv`.
//! - Model-layer shared INT8 pools and sequence-local block tables.
//!
//! Tests: `crates/ferrum-kernels/tests/int8_kv_parity.rs` checks stream
//! ownership, round-trip quantization and attention against an FP32 host reference.

use crate::backend::{KvInt8, KvTokenLayout, QuantizedPagedKvPool};
use ferrum_types::{FerrumError, Result};

use super::{CudaBackend, CudaState};

// CUDA: INT8 KV cache (vLLM-style scale-per-token symmetric quantization).
// Kernel-side dispatch is exposed via [`crate::int8_kv::launch_int8_paged_decode_attention`]
// and [`crate::int8_kv::launch_int8_kv_cache_append`]. See
// `tests/int8_kv_parity.rs` for a host-reference parity check
// (cos sim ≈ 0.99999 vs FP32 ref). With the associated types declared
// here, `KvCache<CudaBackend, KvInt8>` carries `CudaSlice<i8>` for K/V
// and `CudaSlice<f16>` for scales — distinct types from the FP16 path.
//
// Note: `KvScales = OptionalCudaScalesF16` rather than a bare
// `CudaSlice<f16>` so the `Default` bound on `KvScales` can be
// satisfied without holding a CUDA stream at struct-default time.
impl crate::backend::BackendKvDtype<crate::backend::KvInt8> for CudaBackend {
    type KvBuffer = OptionalCudaInt8;
    type KvScales = OptionalCudaScalesF16;
}

/// Lazily-allocated INT8 KV buffer. `Default` produces an empty
/// placeholder; the real allocation happens via the `init` method
/// once a CUDA stream is in scope.
#[derive(Default)]
pub struct OptionalCudaInt8(pub Option<cudarc::driver::CudaSlice<i8>>);

impl OptionalCudaInt8 {
    /// Allocate `len` zeroed `int8_t` elements on the owning context's CUDA stream.
    pub fn alloc(ctx: &mut CudaState, len: usize) -> Self {
        assert!(
            !ctx.capture_in_flight,
            "INT8 allocation during CUDA capture"
        );
        ctx.completion_observed = false;
        let stream = &ctx.stream;
        let buf = stream.alloc_zeros::<i8>(len).expect("alloc int8 KV buffer");
        Self(Some(buf))
    }

    pub fn buffer(&self) -> &cudarc::driver::CudaSlice<i8> {
        self.0.as_ref().expect("OptionalCudaInt8 not allocated")
    }

    pub fn buffer_mut(&mut self) -> &mut cudarc::driver::CudaSlice<i8> {
        self.0.as_mut().expect("OptionalCudaInt8 not allocated")
    }
}

/// Lazily-allocated INT8 scales buffer (FP16 storage on CUDA).
#[derive(Default)]
pub struct OptionalCudaScalesF16(pub Option<cudarc::driver::CudaSlice<half::f16>>);

impl OptionalCudaScalesF16 {
    /// Allocate `len` zeroed FP16 scales on the owning context's CUDA stream.
    pub fn alloc(ctx: &mut CudaState, len: usize) -> Self {
        assert!(
            !ctx.capture_in_flight,
            "INT8 allocation during CUDA capture"
        );
        ctx.completion_observed = false;
        let stream = &ctx.stream;
        let buf = stream
            .alloc_zeros::<half::f16>(len)
            .expect("alloc int8 KV scales");
        Self(Some(buf))
    }

    pub fn buffer(&self) -> &cudarc::driver::CudaSlice<half::f16> {
        self.0
            .as_ref()
            .expect("OptionalCudaScalesF16 not allocated")
    }

    pub fn buffer_mut(&mut self) -> &mut cudarc::driver::CudaSlice<half::f16> {
        self.0
            .as_mut()
            .expect("OptionalCudaScalesF16 not allocated")
    }
}

// Implement INT8 KV launchers as Backend trait methods so the model
// layer can dispatch via `B::int8_kv_append_paged(...)` /
// `B::int8_paged_decode_attention(...)` without reaching into
// cudarc primitives directly.
impl crate::backend::BackendInt8KvOps for CudaBackend {
    fn int8_kv_storage() -> Option<ferrum_types::KvCacheStorage> {
        Some(ferrum_types::KvCacheStorage::Int8PerTokenHeadF16Scales)
    }

    fn alloc_paged_int8_pool(
        ctx: &mut Self::Context,
        num_blocks: usize,
        block_size: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> QuantizedPagedKvPool<Self, KvInt8> {
        QuantizedPagedKvPool::new_cuda(ctx, num_blocks, block_size, num_kv_heads, head_dim)
    }

    fn alloc_paged_int8_layer(
        ctx: &mut Self::Context,
        max_blocks_per_seq: usize,
        block_size: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> crate::backend::KvCacheQuant<Self, crate::backend::KvInt8> {
        crate::backend::KvCacheQuant::<CudaBackend, crate::backend::KvInt8>::new_paged_cuda(
            ctx,
            max_blocks_per_seq,
            block_size,
            num_kv_heads,
            head_dim,
        )
    }

    fn int8_kv_append_paged(
        ctx: &mut Self::Context,
        k_in: &Self::Buffer,
        v_in: &Self::Buffer,
        pool: &mut QuantizedPagedKvPool<Self, KvInt8>,
        paged_block_indices: &[u32],
        cache_len_before: usize,
        tokens: usize,
        input_layout: KvTokenLayout,
        block_size: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Result<()> {
        if tokens == 0 {
            return Ok(());
        }
        let end = cache_len_before
            .checked_add(tokens)
            .ok_or_else(|| FerrumError::model("INT8 append token range overflow"))?;
        pool.validate_mapping(paged_block_indices, end, block_size, num_kv_heads, head_dim)?;
        // Compute flat slot indices: physical_block * block_size + slot.
        // Reads `paged_block_indices` directly (host mirror populated at
        // `ensure_kv`), avoiding the per-token D2H + sync barrier the
        // earlier version paid. H2D for the resulting `slot_mapping` uses
        // `cuMemcpyHtoDAsync` on the stream (no host wait), so the cost
        // collapses to the cudarc enqueue overhead.
        let stream = ctx.stream.clone();
        let mut slot_mapping_host = vec![0i32; tokens];
        for t in 0..tokens {
            let global_pos = cache_len_before + t;
            let block_logical = global_pos / block_size;
            let slot_in_block = global_pos % block_size;
            let block_physical = paged_block_indices[block_logical] as usize;
            slot_mapping_host[t] = (block_physical * block_size + slot_in_block) as i32;
        }
        let slot_mapping = stream
            .clone_htod(&slot_mapping_host)
            .map_err(|e| FerrumError::model(format!("htod slot_mapping: {e}")))?;

        crate::int8_kv::launch_int8_kv_cache_append(
            &ctx.stream,
            k_in.as_f16(),
            v_in.as_f16(),
            pool.k.buffer_mut(),
            pool.v.buffer_mut(),
            pool.k_scales.buffer_mut(),
            pool.v_scales.buffer_mut(),
            &slot_mapping,
            tokens,
            num_kv_heads,
            head_dim,
            input_layout,
        )
        .map_err(|e| FerrumError::model(format!("launch_int8_kv_cache_append: {e}")))?;
        Ok(())
    }

    fn int8_paged_decode_attention(
        ctx: &mut Self::Context,
        q: &Self::Buffer,
        pool: &QuantizedPagedKvPool<Self, KvInt8>,
        block_table: &Self::Buffer,
        paged_block_indices: &[u32],
        output: &mut Self::Buffer,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        valid_kv_len: usize,
        query_rows: usize,
        query_layout: KvTokenLayout,
        block_size: usize,
        scale: f32,
    ) -> Result<()> {
        pool.validate_mapping(
            paged_block_indices,
            valid_kv_len,
            block_size,
            num_kv_heads,
            head_dim,
        )?;
        // block_table is CudaBuf::U32 (typed since B-2). The launch
        // function takes `&CudaView<i32>` — same byte pattern, signed
        // view; transmute on CudaBuf dispatches to the U32 variant's
        // CudaSlice::transmute which reinterprets u32→i32 1:1.
        let n_blocks = valid_kv_len.div_ceil(block_size).max(1);
        let bt_i32_view = unsafe {
            block_table
                .transmute::<i32>(n_blocks)
                .ok_or_else(|| FerrumError::model("block_table transmute<i32> failed"))?
        };
        crate::int8_kv::launch_int8_paged_decode_attention(
            &ctx.stream,
            q.as_f16(),
            pool.k.buffer(),
            pool.v.buffer(),
            pool.k_scales.buffer(),
            pool.v_scales.buffer(),
            &bt_i32_view,
            output.as_f16_mut(),
            num_q_heads,
            num_kv_heads,
            head_dim,
            valid_kv_len,
            query_rows,
            query_layout,
            block_size,
            scale,
        )
        .map_err(|e| FerrumError::model(format!("launch_int8_paged_decode_attention: {e}")))?;
        Ok(())
    }
}

impl QuantizedPagedKvPool<CudaBackend, KvInt8> {
    /// Allocate physical storage once per model layer. Every sequence and
    /// cached prefix refers to the same block namespace and these same arrays.
    pub fn new_cuda(
        ctx: &mut CudaState,
        num_blocks: usize,
        block_size: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Self {
        assert!(num_blocks > 0 && block_size > 0 && num_kv_heads > 0 && head_dim > 0);
        let pool_tokens = num_blocks
            .checked_mul(block_size)
            .expect("INT8 pool token overflow");
        assert!(
            pool_tokens <= i32::MAX as usize,
            "INT8 pool exceeds kernel slot range"
        );
        let scale_count = pool_tokens
            .checked_mul(num_kv_heads)
            .expect("INT8 pool scale overflow");
        let elem_count = scale_count
            .checked_mul(head_dim)
            .expect("INT8 pool element overflow");
        Self {
            k: OptionalCudaInt8::alloc(ctx, elem_count),
            v: OptionalCudaInt8::alloc(ctx, elem_count),
            k_scales: OptionalCudaScalesF16::alloc(ctx, scale_count),
            v_scales: OptionalCudaScalesF16::alloc(ctx, scale_count),
            num_blocks,
            block_size,
            num_kv_heads,
            head_dim,
        }
    }

    fn validate_mapping(
        &self,
        indices: &[u32],
        tokens: usize,
        block_size: usize,
        heads: usize,
        dim: usize,
    ) -> Result<()> {
        if block_size == 0
            || block_size != self.block_size
            || heads != self.num_kv_heads
            || dim != self.head_dim
        {
            return Err(FerrumError::model(
                "INT8 pool layout does not match the sequence",
            ));
        }
        let count = tokens.div_ceil(block_size);
        if tokens == 0
            || indices.len() < count
            || indices[..count]
                .iter()
                .any(|&index| index as usize >= self.num_blocks)
        {
            return Err(FerrumError::model(
                "INT8 table references outside its shared physical pool",
            ));
        }
        let scales = self
            .num_blocks
            .checked_mul(block_size)
            .and_then(|n| n.checked_mul(heads))
            .ok_or_else(|| FerrumError::model("INT8 pool dimensions overflow"))?;
        let elements = scales
            .checked_mul(dim)
            .ok_or_else(|| FerrumError::model("INT8 pool dimensions overflow"))?;
        if self
            .k
            .0
            .as_ref()
            .is_none_or(|buffer| buffer.len() != elements)
            || self
                .v
                .0
                .as_ref()
                .is_none_or(|buffer| buffer.len() != elements)
            || self
                .k_scales
                .0
                .as_ref()
                .is_none_or(|buffer| buffer.len() != scales)
            || self
                .v_scales
                .0
                .as_ref()
                .is_none_or(|buffer| buffer.len() != scales)
        {
            return Err(FerrumError::model(
                "INT8 shared pool allocation does not match its dimensions",
            ));
        }
        Ok(())
    }
}

// Convenience constructor for paged INT8 KV caches on CUDA.
impl crate::backend::KvCacheQuant<CudaBackend, crate::backend::KvInt8> {
    /// Allocate one sequence's table and metadata, without private K/V storage.
    ///
    /// - `max_blocks_per_seq` × `block_size` = capacity in tokens
    /// - Physical K/V and scales belong to `QuantizedPagedKvPool`.
    /// - `block_table` is allocated as u32[max_blocks_per_seq] via `B::alloc_u32`
    /// - `context_lens` is allocated as u32[1] (single seq for now)
    pub fn new_paged_cuda(
        ctx: &mut CudaState,
        max_blocks_per_seq: usize,
        block_size: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Self {
        use crate::backend::Backend;
        let logical_capacity = max_blocks_per_seq
            .checked_mul(block_size)
            .expect("INT8 table capacity overflow");
        let block_table = <CudaBackend as Backend>::alloc_typed_in_context(
            ctx,
            crate::backend::Dtype::U32,
            max_blocks_per_seq,
        );
        // The caller initializes tables on its preparation context and observes
        // that context before publishing the cache as ready.
        let context_lens =
            <CudaBackend as Backend>::alloc_typed_in_context(ctx, crate::backend::Dtype::U32, 1);

        // Re-cast typed u32 buffer to the trait's Buffer (FP16) — same
        // pattern the FP16 paged path uses for block_table/context_lens
        // (they are u32 device tensors written through alloc_u32).
        let bt_buf = block_table;
        let cl_buf = context_lens;

        crate::backend::KvCacheQuant {
            len: 0,
            capacity: logical_capacity,
            num_kv_heads,
            head_dim,
            block_size,
            block_table: Some(bt_buf),
            context_lens: Some(cl_buf),
            paged_block_indices: Vec::new(),
            _kv_dtype: std::marker::PhantomData,
        }
    }
}
