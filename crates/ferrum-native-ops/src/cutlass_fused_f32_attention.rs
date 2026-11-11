//! Allocation-free ABI for locked CUTLASS full MHA with F32 boundaries.
//! These geometry/span checks confer no device authority. The caller retains
//! the current context, stream, code and all allocations through completion.

use std::mem::{offset_of, size_of};

pub const CUTLASS_FUSED_F32_ATTENTION_ABI: u32 = 1;
pub const CUTLASS_FUSED_F32_ATTENTION_OPERATOR: &str = "ferrum.cuda.cutlass_fused_f32_attention";
pub const CUTLASS_FUSED_F32_ATTENTION_EXPORTS: [&str; 2] = [
    "ferrum_cutlass_fused_f32_attention_prepare_v1",
    "ferrum_cutlass_fused_f32_attention_run_v1",
];
pub const CUTLASS_FUSED_F32_ATTENTION_NUMERICAL_IDENTITY: &str = concat!(
    "cutlass-cb4247394dd82148787aed73e5dc7cef33cbf862-example41;",
    "F32-QKVY-BSHD;Hq=Hkv;full-bidirectional-all-valid;explicit-F32-scale;",
    "Sm80-OpMultiplyAddFastF32-tensorop;F32-accumulator-softmax-exp2f;",
    "D64=tile64x64x64;D72=tile64x128x128-logicalD72;",
    "no-bias-dropout;zero-global-workspace;no-fast-math;",
    "finite-domain-qualified;different-association-and-intermediate-rounding;",
    "not-strict-online-bitwise;not-uniform-error-bound;target-qualification-required"
);

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CutlassFusedF32AttentionRequestV1 {
    pub abi: u32,
    pub size: u32,
    pub batches: u32,
    pub sequence: u32,
    pub heads: u32,
    pub dimension: u32,
    pub scale: f32,
    pub reserved: u32,
}

impl CutlassFusedF32AttentionRequestV1 {
    pub fn new(
        batches: u32,
        sequence: u32,
        heads: u32,
        dimension: u32,
        scale: f32,
    ) -> Result<Self, CutlassFusedF32AttentionError> {
        let value = Self {
            abi: CUTLASS_FUSED_F32_ATTENTION_ABI,
            size: size_of::<Self>() as u32,
            batches,
            sequence,
            heads,
            dimension,
            scale,
            reserved: 0,
        };
        value.validate()?;
        Ok(value)
    }
    pub fn validate(&self) -> Result<(), CutlassFusedF32AttentionError> {
        if self.abi != CUTLASS_FUSED_F32_ATTENTION_ABI
            || self.size as usize != size_of::<Self>()
            || self.reserved != 0
            || !self.scale.is_finite()
            || self.scale <= 0.0
        {
            return Err(CutlassFusedF32AttentionError::InvalidArgument);
        }
        if !(1..=65535).contains(&self.batches)
            || !(1..=65535).contains(&self.sequence)
            || !(1..=65535).contains(&self.heads)
            || !matches!(self.dimension, 64 | 72)
            || u64::from(self.sequence) * u64::from(self.heads) * u64::from(self.dimension)
                > i32::MAX as u64
        {
            return Err(CutlassFusedF32AttentionError::UnsupportedGeometry);
        }
        self.elements()?;
        Ok(())
    }
    pub fn elements(&self) -> Result<u64, CutlassFusedF32AttentionError> {
        u64::from(self.batches)
            .checked_mul(u64::from(self.sequence))
            .and_then(|n| n.checked_mul(u64::from(self.heads)))
            .and_then(|n| n.checked_mul(u64::from(self.dimension)))
            .filter(|n| *n > 0 && *n <= isize::MAX as u64 / 4)
            .ok_or(CutlassFusedF32AttentionError::Extent)
    }
    pub fn fingerprint_material(&self) -> [u8; 32] {
        words([
            self.abi,
            self.size,
            self.batches,
            self.sequence,
            self.heads,
            self.dimension,
            self.scale.to_bits(),
            self.reserved,
        ])
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CutlassFusedF32AttentionSpanV1 {
    pub address: u64,
    pub elements: u64,
}
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CutlassFusedF32AttentionBindingsV1 {
    pub query: CutlassFusedF32AttentionSpanV1,
    pub key: CutlassFusedF32AttentionSpanV1,
    pub value: CutlassFusedF32AttentionSpanV1,
    pub output: CutlassFusedF32AttentionSpanV1,
}
impl CutlassFusedF32AttentionBindingsV1 {
    pub fn validate(
        &self,
        request: &CutlassFusedF32AttentionRequestV1,
    ) -> Result<(), CutlassFusedF32AttentionError> {
        request.validate()?;
        let elements = request.elements()?;
        let bytes = elements
            .checked_mul(4)
            .ok_or(CutlassFusedF32AttentionError::Extent)?;
        for s in [self.query, self.key, self.value, self.output] {
            if s.address == 0
                || s.address % 16 != 0
                || s.elements != elements
                || s.address.checked_add(bytes).is_none()
            {
                return Err(CutlassFusedF32AttentionError::Span);
            }
        }
        for s in [self.query, self.key, self.value] {
            if self.output.address < s.address + bytes && s.address < self.output.address + bytes {
                return Err(CutlassFusedF32AttentionError::Span);
            }
        }
        Ok(())
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CutlassFusedF32AttentionInfoV1 {
    pub abi: u32,
    pub size: u32,
    pub shared_bytes: u32,
    pub threads: u32,
    pub query_tile: u32,
    pub key_tile: u32,
    pub global_workspace_bytes: u64,
}
impl CutlassFusedF32AttentionInfoV1 {
    /// Also compare shared_bytes with the actual device's opt-in limit before
    /// admission. This structural check does not certify a CUDA device.
    pub fn validate_for(
        &self,
        request: &CutlassFusedF32AttentionRequestV1,
    ) -> Result<(), CutlassFusedF32AttentionError> {
        request.validate()?;
        let (threads, key_tile) = if request.dimension == 64 {
            (128, 64)
        } else {
            (256, 128)
        };
        if self.abi != CUTLASS_FUSED_F32_ATTENTION_ABI
            || self.size as usize != size_of::<Self>()
            || self.shared_bytes == 0
            || self.threads != threads
            || self.query_tile != 64
            || self.key_tile != key_tile
            || self.global_workspace_bytes != 0
        {
            return Err(CutlassFusedF32AttentionError::Plan);
        }
        Ok(())
    }
    pub fn fingerprint_material(&self) -> [u8; 32] {
        words([
            self.abi,
            self.size,
            self.shared_bytes,
            self.threads,
            self.query_tile,
            self.key_tile,
            self.global_workspace_bytes as u32,
            (self.global_workspace_bytes >> 32) as u32,
        ])
    }
}

pub type CutlassFusedF32AttentionPrepareFn = unsafe extern "C" fn(
    *const CutlassFusedF32AttentionRequestV1,
    *mut CutlassFusedF32AttentionInfoV1,
) -> i32;
pub type CutlassFusedF32AttentionRunFn = unsafe extern "C" fn(
    *const CutlassFusedF32AttentionRequestV1,
    *const CutlassFusedF32AttentionBindingsV1,
    *mut std::ffi::c_void,
) -> i32;

fn words(values: [u32; 8]) -> [u8; 32] {
    let mut bytes = [0; 32];
    for (i, value) in values.into_iter().enumerate() {
        bytes[i * 4..i * 4 + 4].copy_from_slice(&value.to_le_bytes());
    }
    bytes
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum CutlassFusedF32AttentionError {
    #[error("invalid CUTLASS fused F32 argument")]
    InvalidArgument,
    #[error("unsupported CUTLASS fused F32 geometry")]
    UnsupportedGeometry,
    #[error("CUTLASS fused F32 extent overflows")]
    Extent,
    #[error("CUTLASS fused F32 span extent/alignment/alias violation")]
    Span,
    #[error("CUTLASS fused F32 native plan differs from its geometry")]
    Plan,
    #[error("locked CUTLASS fused F32 native artifact is unavailable")]
    Unavailable,
    #[error("CUTLASS fused F32 native returned status {0}")]
    Native(i32),
}

const _: () = {
    assert!(size_of::<CutlassFusedF32AttentionRequestV1>() == 32);
    assert!(offset_of!(CutlassFusedF32AttentionRequestV1, scale) == 24);
    assert!(size_of::<CutlassFusedF32AttentionBindingsV1>() == 64);
    assert!(size_of::<CutlassFusedF32AttentionInfoV1>() == 32);
    assert!(offset_of!(CutlassFusedF32AttentionInfoV1, global_workspace_bytes) == 24);
};

#[cfg(test)]
mod tests;
