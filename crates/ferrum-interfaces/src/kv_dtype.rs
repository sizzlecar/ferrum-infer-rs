//! KV cache element-type markers (Dim 5 of the 5-dimension architecture).
//!
//! These are pure marker types with no GPU dependencies, so they live
//! in `ferrum-interfaces` rather than `ferrum-kernels`. The capability
//! trait that links them to a backend (`BackendKvDtype<K>: BackendPagedKv`)
//! does need GPU types, so it stays in `ferrum-kernels::backend`.
//!
//! Each model's KV cache has its own precision independent of the
//! model's compute precision. vLLM 0.6+ ships INT8 / FP8 KV caches that
//! halve KV memory at small (<1%) accuracy hit. ferrum's type system
//! exposes this axis via the `K: KvDtypeKind` parameter on
//! `KvCache<B, K>` (default `K = KvFp16`).

/// Marker trait + metadata for a KV cache element type.
pub trait KvDtypeKind: Send + Sync + 'static {
    /// Stable name for logging / debug (e.g. "fp16", "int8").
    const NAME: &'static str;
    /// Bytes per element on disk + in cache memory.
    const BYTES_PER_ELEM: usize;
}

/// FP16 KV cache (the existing default on CUDA + Metal).
pub struct KvFp16;
impl KvDtypeKind for KvFp16 {
    const NAME: &'static str = "fp16";
    const BYTES_PER_ELEM: usize = 2;
}

/// BF16 KV cache (drop-in replacement for FP16 on Ampere+ / Apple Silicon).
pub struct KvBf16;
impl KvDtypeKind for KvBf16 {
    const NAME: &'static str = "bf16";
    const BYTES_PER_ELEM: usize = 2;
}

/// INT8 KV cache — half the memory of FP16 with per-token / per-channel
/// scale factors. CUDA path planned via vLLM's quant_kv kernels.
pub struct KvInt8;
impl KvDtypeKind for KvInt8 {
    const NAME: &'static str = "int8";
    const BYTES_PER_ELEM: usize = 1;
}

/// FP8 KV cache — E4M3 by default. Hopper+ on CUDA, future on Metal.
pub struct KvFp8;
impl KvDtypeKind for KvFp8 {
    const NAME: &'static str = "fp8";
    const BYTES_PER_ELEM: usize = 1;
}
