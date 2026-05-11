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
//! - `KvCacheQuant<CudaBackend, KvInt8>::new_paged_cuda(...)` one-call
//!   constructor (K/V int8 pool + FP16 scales + block_table + context_lens).
//!
//! Tests: `crates/ferrum-kernels/tests/int8_kv_parity.rs`. All four
//! ignored tests pass cosine 0.99999 vs FP32 host reference.

use ferrum_types::{FerrumError, Result};

use super::{default_stream, CudaBackend};

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
    /// Allocate `len` zeroed `int8_t` elements on the default CUDA stream.
    pub fn alloc(len: usize) -> Self {
        let stream = default_stream();
        let buf = stream
            .alloc_zeros::<i8>(len)
            .expect("alloc int8 KV buffer");
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
    /// Allocate `len` zeroed FP16 scales on the default CUDA stream.
    pub fn alloc(len: usize) -> Self {
        let stream = default_stream();
        let buf = stream
            .alloc_zeros::<half::f16>(len)
            .expect("alloc int8 KV scales");
        Self(Some(buf))
    }

    pub fn buffer(&self) -> &cudarc::driver::CudaSlice<half::f16> {
        self.0.as_ref().expect("OptionalCudaScalesF16 not allocated")
    }

    pub fn buffer_mut(&mut self) -> &mut cudarc::driver::CudaSlice<half::f16> {
        self.0.as_mut().expect("OptionalCudaScalesF16 not allocated")
    }
}

// Implement INT8 KV launchers as Backend trait methods so the model
// layer can dispatch via `B::int8_kv_append_paged(...)` /
// `B::int8_paged_decode_attention(...)` without reaching into
// cudarc primitives directly.
impl crate::backend::BackendInt8KvOps for CudaBackend {
    fn alloc_paged_int8_layer(
        max_blocks_per_seq: usize,
        block_size: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> crate::backend::KvCacheQuant<Self, crate::backend::KvInt8> {
        crate::backend::KvCacheQuant::<CudaBackend, crate::backend::KvInt8>::new_paged_cuda(
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
        layer_k: &mut <Self as crate::backend::BackendKvDtype<crate::backend::KvInt8>>::KvBuffer,
        layer_v: &mut <Self as crate::backend::BackendKvDtype<crate::backend::KvInt8>>::KvBuffer,
        layer_k_scales: &mut <Self as crate::backend::BackendKvDtype<crate::backend::KvInt8>>::KvScales,
        layer_v_scales: &mut <Self as crate::backend::BackendKvDtype<crate::backend::KvInt8>>::KvScales,
        paged_block_indices: &[u32],
        cache_len_before: usize,
        tokens: usize,
        block_size: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Result<()> {
        if tokens == 0 {
            return Ok(());
        }
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
            .memcpy_stod(&slot_mapping_host)
            .map_err(|e| FerrumError::model(format!("htod slot_mapping: {e}")))?;

        // Lazily alloc INT8 buffers + scales on first call (the constructor
        // populates them already, but defensive in case callers clear).
        if layer_k.0.is_none() {
            return Err(FerrumError::model(
                "int8_kv_append_paged: layer_k not allocated",
            ));
        }
        if layer_v.0.is_none() || layer_k_scales.0.is_none() || layer_v_scales.0.is_none() {
            return Err(FerrumError::model(
                "int8_kv_append_paged: layer_v / scales not allocated",
            ));
        }

        crate::int8_kv::launch_int8_kv_cache_append(
            &ctx.ctx,
            k_in.as_f16(),
            v_in.as_f16(),
            layer_k.buffer_mut(),
            layer_v.buffer_mut(),
            layer_k_scales.buffer_mut(),
            layer_v_scales.buffer_mut(),
            &slot_mapping,
            tokens,
            num_kv_heads,
            head_dim,
        )
        .map_err(|e| FerrumError::model(format!("launch_int8_kv_cache_append: {e}")))?;
        Ok(())
    }

    fn int8_paged_decode_attention(
        ctx: &mut Self::Context,
        q: &Self::Buffer,
        layer_k: &<Self as crate::backend::BackendKvDtype<crate::backend::KvInt8>>::KvBuffer,
        layer_v: &<Self as crate::backend::BackendKvDtype<crate::backend::KvInt8>>::KvBuffer,
        layer_k_scales: &<Self as crate::backend::BackendKvDtype<crate::backend::KvInt8>>::KvScales,
        layer_v_scales: &<Self as crate::backend::BackendKvDtype<crate::backend::KvInt8>>::KvScales,
        block_table: &Self::Buffer,
        output: &mut Self::Buffer,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        valid_kv_len: usize,
        block_size: usize,
        scale: f32,
    ) -> Result<()> {
        // block_table is stored as f16 but holds i32 (alloc_u32 doubles
        // bytes). Reinterpret to i32 view of length max_blocks_per_seq.
        let n_blocks = valid_kv_len.div_ceil(block_size).max(1);
        let bt_i32_view = unsafe {
            block_table
                .transmute::<i32>(n_blocks)
                .ok_or_else(|| FerrumError::model("block_table transmute<i32> failed"))?
        };
        crate::int8_kv::launch_int8_paged_decode_attention(
            &ctx.ctx,
            q.as_f16(),
            layer_k.buffer(),
            layer_v.buffer(),
            layer_k_scales.buffer(),
            layer_v_scales.buffer(),
            &bt_i32_view,
            output.as_f16_mut(),
            num_q_heads,
            num_kv_heads,
            head_dim,
            valid_kv_len,
            block_size,
            scale,
        )
        .map_err(|e| FerrumError::model(format!("launch_int8_paged_decode_attention: {e}")))?;
        Ok(())
    }
}

// Convenience constructor for paged INT8 KV caches on CUDA.
impl crate::backend::KvCacheQuant<CudaBackend, crate::backend::KvInt8> {
    /// Allocate a paged INT8 KV cache for one sequence.
    ///
    /// - `max_blocks_per_seq` × `block_size` = capacity in tokens
    /// - K/V pool size: `max_blocks_per_seq * block_size * num_kv_heads * head_dim` int8 elems
    /// - scales pool size: `max_blocks_per_seq * block_size * num_kv_heads` FP16 elems
    /// - `block_table` is allocated as u32[max_blocks_per_seq] via `B::alloc_u32`
    /// - `context_lens` is allocated as u32[1] (single seq for now)
    pub fn new_paged_cuda(
        max_blocks_per_seq: usize,
        block_size: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Self {
        use crate::backend::Backend;
        let pool_tokens = max_blocks_per_seq * block_size;
        let elem_count = pool_tokens * num_kv_heads * head_dim;
        let scale_count = pool_tokens * num_kv_heads;
        let block_table = <CudaBackend as Backend>::alloc_u32(max_blocks_per_seq);
        let mut context_lens = <CudaBackend as Backend>::alloc_u32(1);
        let mut bt_ctx = <CudaBackend as Backend>::new_context();
        <CudaBackend as Backend>::write_u32(&mut bt_ctx, &mut context_lens, &[0u32]);
        <CudaBackend as Backend>::sync(&mut bt_ctx);

        // Re-cast typed u32 buffer to the trait's Buffer (FP16) — same
        // pattern the FP16 paged path uses for block_table/context_lens
        // (they are u32 device tensors written through alloc_u32).
        let bt_buf = block_table;
        let cl_buf = context_lens;

        crate::backend::KvCacheQuant {
            k: OptionalCudaInt8::alloc(elem_count),
            v: OptionalCudaInt8::alloc(elem_count),
            k_scales: OptionalCudaScalesF16::alloc(scale_count),
            v_scales: OptionalCudaScalesF16::alloc(scale_count),
            len: 0,
            capacity: pool_tokens,
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
