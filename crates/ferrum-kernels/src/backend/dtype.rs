//! Runtime element type tag for typed device buffers.
//!
//! Used by the typed buffer handles (`MetalBuf`, `CpuBuf`, `CudaBuf`)
//! to carry a dtype alongside the raw byte storage. This lets the
//! existing Backend trait surface (which today exposes a monomorphic
//! `Self::Buffer` and a pile of `from_slice_i32` / `alloc_u32` /
//! `write_u32` / `from_slice_f32` etc. helpers) move toward a single
//! `Self::alloc(Dtype, n)` / `Self::write(buf, &[T])` API without
//! breaking callers in one PR.
//!
//! Phase A (PR A): defined here + Metal `MetalBuf` re-uses it (was
//! Metal-internal previously). CPU `CpuBuf` and CUDA `CudaBuf` not
//! yet wired — see Phase B for the migration.

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Dtype {
    /// 32-bit IEEE float. Default activation / weight dtype on the
    /// CPU path; fallback for backends without F16 hw support.
    F32,
    /// 16-bit IEEE half. Hot-path dtype on CUDA + Metal (decode q,
    /// K/V, GEMM outputs).
    F16,
    /// 32-bit unsigned integer. Block tables, context lens, sorted
    /// token ids, args buffers — anything previously tunneled through
    /// an FP buffer via `alloc_u32` / `write_u32`.
    U32,
    /// 32-bit signed integer. Expert ids, position offsets,
    /// `cu_seqlens_q`, `tpe` (tokens-per-expert). Same byte width as
    /// `U32`; separate variant so kernel signatures
    /// (`device const int*` vs `device const uint*`) can stay
    /// type-honest at runtime.
    I32,
    /// 8-bit signed integer. INT8 quantized KV cache cells. Used by
    /// `KvCacheQuant<B, KvInt8>`'s paged stores.
    I8,
}

impl Dtype {
    pub const fn bytes_per_elem(self) -> usize {
        match self {
            Dtype::F32 | Dtype::U32 | Dtype::I32 => 4,
            Dtype::F16 => 2,
            Dtype::I8 => 1,
        }
    }

    /// Human-readable tag for log lines / panic messages.
    pub const fn name(self) -> &'static str {
        match self {
            Dtype::F32 => "f32",
            Dtype::F16 => "f16",
            Dtype::U32 => "u32",
            Dtype::I32 => "i32",
            Dtype::I8 => "i8",
        }
    }
}

/// Marker trait connecting a host element type `T` to its runtime
/// `Dtype` tag. Used by the typed Backend allocator + uploader so the
/// trait surface has ONE `alloc_typed(Dtype, n)` /
/// `from_slice_typed<T>(&[T])` / `write_typed<T>(buf, &[T])` instead
/// of the per-dtype-named family (`alloc_u32`, `from_slice_i32`,
/// `write_i32_into`, `write_f32_into`, ...).
///
/// Implemented for all dtypes that backends store in `Self::Buffer`.
pub trait HostDtype: Copy + Send + Sync + 'static {
    const DTYPE: Dtype;
}

impl HostDtype for u32 {
    const DTYPE: Dtype = Dtype::U32;
}
impl HostDtype for i32 {
    const DTYPE: Dtype = Dtype::I32;
}
impl HostDtype for f32 {
    const DTYPE: Dtype = Dtype::F32;
}
impl HostDtype for half::f16 {
    const DTYPE: Dtype = Dtype::F16;
}
impl HostDtype for i8 {
    const DTYPE: Dtype = Dtype::I8;
}
