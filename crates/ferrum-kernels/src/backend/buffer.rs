//! Typed buffer wrappers — Phase B foundation.
//!
//! Each backend grows a wrapper struct that carries a runtime `Dtype`
//! tag alongside the raw storage. The wrappers are **not yet wired** to
//! `Backend::Buffer` (that's Phase B-2: switch `type Buffer = ...` per
//! backend + migrate all callsites). For now they exist as separate
//! types so Phase B-2 can be staged without breaking everything in
//! one go.
//!
//! Pattern mirrors the existing `MetalBuf` (see `backend::metal::MetalBuf`)
//! which has been carrying a dtype tag since the Metal backend shipped.
//! Generalising it to CPU + CUDA eliminates the `from_slice_i32` /
//! `alloc_u32` / `write_u32` family of helpers that tunnel integer data
//! through the FP-typed buffer.
//!
//! Migration plan (next PR):
//! 1. Replace each backend's `type Buffer = <concrete>` with the typed
//!    wrapper.
//! 2. Add forwarder methods so existing call sites compile unchanged
//!    (`buf.as_f16()` / `buf.as_f32_slice()` / `buf.as_u32_mut()` etc.).
//! 3. Replace `alloc_u32` / `from_slice_i32` / `write_u32` with one
//!    `Self::alloc(Dtype, n)` and one `Self::write_typed(buf, &[T])`.
//! 4. Delete the legacy helpers + their i32 bit-cast tunnels.

use crate::backend::dtype::Dtype;
use half::f16;

/// CPU-side typed buffer. Variants per dtype keep storage typed (no
/// `unsafe` bytemuck casting, no alignment concerns). Compared to a
/// `(Vec<u8>, Dtype, n)` triple this trades 16 bytes of discriminant
/// for type safety on a non-hot-path (CPU is the slow path; if you
/// care about throughput here, switch backends).
pub enum CpuBuf {
    F32(Vec<f32>),
    F16(Vec<f16>),
    U32(Vec<u32>),
    I32(Vec<i32>),
    I8(Vec<i8>),
}

impl CpuBuf {
    pub fn alloc(dtype: Dtype, n: usize) -> Self {
        match dtype {
            Dtype::F32 => CpuBuf::F32(vec![0.0; n]),
            Dtype::F16 => CpuBuf::F16(vec![f16::ZERO; n]),
            Dtype::U32 => CpuBuf::U32(vec![0u32; n]),
            Dtype::I32 => CpuBuf::I32(vec![0i32; n]),
            Dtype::I8 => CpuBuf::I8(vec![0i8; n]),
        }
    }

    pub fn dtype(&self) -> Dtype {
        match self {
            CpuBuf::F32(_) => Dtype::F32,
            CpuBuf::F16(_) => Dtype::F16,
            CpuBuf::U32(_) => Dtype::U32,
            CpuBuf::I32(_) => Dtype::I32,
            CpuBuf::I8(_) => Dtype::I8,
        }
    }

    pub fn len(&self) -> usize {
        match self {
            CpuBuf::F32(v) => v.len(),
            CpuBuf::F16(v) => v.len(),
            CpuBuf::U32(v) => v.len(),
            CpuBuf::I32(v) => v.len(),
            CpuBuf::I8(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Typed accessor — panics on dtype mismatch. Catches the silent
    /// type-tunnel bugs the old `from_slice_i32` route would have
    /// papered over.
    pub fn as_f32(&self) -> &[f32] {
        match self {
            CpuBuf::F32(v) => v,
            _ => panic!("CpuBuf::as_f32 on dtype {}", self.dtype().name()),
        }
    }
    pub fn as_f32_mut(&mut self) -> &mut [f32] {
        match self {
            CpuBuf::F32(v) => v,
            _ => panic!("CpuBuf::as_f32_mut on dtype {}", self.dtype().name()),
        }
    }
    pub fn as_f16(&self) -> &[f16] {
        match self {
            CpuBuf::F16(v) => v,
            _ => panic!("CpuBuf::as_f16 on dtype {}", self.dtype().name()),
        }
    }
    pub fn as_u32(&self) -> &[u32] {
        match self {
            CpuBuf::U32(v) => v,
            _ => panic!("CpuBuf::as_u32 on dtype {}", self.dtype().name()),
        }
    }
    pub fn as_u32_mut(&mut self) -> &mut [u32] {
        match self {
            CpuBuf::U32(v) => v,
            _ => panic!("CpuBuf::as_u32_mut on dtype {}", self.dtype().name()),
        }
    }
    pub fn as_i32(&self) -> &[i32] {
        match self {
            CpuBuf::I32(v) => v,
            _ => panic!("CpuBuf::as_i32 on dtype {}", self.dtype().name()),
        }
    }
    pub fn as_i32_mut(&mut self) -> &mut [i32] {
        match self {
            CpuBuf::I32(v) => v,
            _ => panic!("CpuBuf::as_i32_mut on dtype {}", self.dtype().name()),
        }
    }
    pub fn as_i8(&self) -> &[i8] {
        match self {
            CpuBuf::I8(v) => v,
            _ => panic!("CpuBuf::as_i8 on dtype {}", self.dtype().name()),
        }
    }

    pub fn from_f32(data: Vec<f32>) -> Self {
        CpuBuf::F32(data)
    }
    pub fn from_u32(data: Vec<u32>) -> Self {
        CpuBuf::U32(data)
    }
    pub fn from_i32(data: Vec<i32>) -> Self {
        CpuBuf::I32(data)
    }
}

/// CUDA-side typed buffer. Enum over typed `CudaSlice<T>` variants —
/// keeps cudarc's typed alloc / dtoh / htod APIs working unchanged
/// when consumers call `buf.as_f16()` / `buf.as_u32_mut()` etc.
///
/// Phase B-1: defined here, not yet wired to `CudaBackend::Buffer`.
/// Phase B-2 switches `CudaBackend::Buffer = CudaBuf` and migrates
/// all `CudaSlice<f16>` consumers to the typed accessor pattern.
#[cfg(feature = "cuda")]
pub enum CudaBuf {
    F32(cudarc::driver::CudaSlice<f32>),
    F16(cudarc::driver::CudaSlice<f16>),
    U32(cudarc::driver::CudaSlice<u32>),
    I32(cudarc::driver::CudaSlice<i32>),
    I8(cudarc::driver::CudaSlice<i8>),
}

#[cfg(feature = "cuda")]
impl CudaBuf {
    pub fn dtype(&self) -> Dtype {
        match self {
            CudaBuf::F32(_) => Dtype::F32,
            CudaBuf::F16(_) => Dtype::F16,
            CudaBuf::U32(_) => Dtype::U32,
            CudaBuf::I32(_) => Dtype::I32,
            CudaBuf::I8(_) => Dtype::I8,
        }
    }

    pub fn len(&self) -> usize {
        match self {
            CudaBuf::F32(s) => s.len(),
            CudaBuf::F16(s) => s.len(),
            CudaBuf::U32(s) => s.len(),
            CudaBuf::I32(s) => s.len(),
            CudaBuf::I8(s) => s.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn as_f16(&self) -> &cudarc::driver::CudaSlice<f16> {
        match self {
            CudaBuf::F16(s) => s,
            _ => panic!("CudaBuf::as_f16 on dtype {}", self.dtype().name()),
        }
    }
    pub fn as_f16_mut(&mut self) -> &mut cudarc::driver::CudaSlice<f16> {
        match self {
            CudaBuf::F16(s) => s,
            _ => panic!("CudaBuf::as_f16_mut on dtype {}", self.dtype().name()),
        }
    }
    pub fn as_f32(&self) -> &cudarc::driver::CudaSlice<f32> {
        match self {
            CudaBuf::F32(s) => s,
            _ => panic!("CudaBuf::as_f32 on dtype {}", self.dtype().name()),
        }
    }
    pub fn as_u32(&self) -> &cudarc::driver::CudaSlice<u32> {
        match self {
            CudaBuf::U32(s) => s,
            _ => panic!("CudaBuf::as_u32 on dtype {}", self.dtype().name()),
        }
    }
    pub fn as_u32_mut(&mut self) -> &mut cudarc::driver::CudaSlice<u32> {
        match self {
            CudaBuf::U32(s) => s,
            _ => panic!("CudaBuf::as_u32_mut on dtype {}", self.dtype().name()),
        }
    }
    pub fn as_i32(&self) -> &cudarc::driver::CudaSlice<i32> {
        match self {
            CudaBuf::I32(s) => s,
            _ => panic!("CudaBuf::as_i32 on dtype {}", self.dtype().name()),
        }
    }
    pub fn as_i8(&self) -> &cudarc::driver::CudaSlice<i8> {
        match self {
            CudaBuf::I8(s) => s,
            _ => panic!("CudaBuf::as_i8 on dtype {}", self.dtype().name()),
        }
    }
}
