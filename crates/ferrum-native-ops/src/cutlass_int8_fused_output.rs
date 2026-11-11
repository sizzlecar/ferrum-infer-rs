//! Allocation-free INT8 GEMM with row/channel scaling and a BF16 output.
//! The caller owns the CUDA context, stream, code and all buffer lifetimes.

use std::mem::{offset_of, size_of};

pub const CUTLASS_INT8_FUSED_OUTPUT_OPERATOR: &str = "ferrum.cuda.cutlass_int8_fused_output";
pub const CUTLASS_INT8_FUSED_OUTPUT_EXPORTS: [&str; 2] = [
    "ferrum_int8_fused_output_prepare_v1",
    "ferrum_int8_fused_output_run_v1",
];
pub const CUTLASS_INT8_FUSED_OUTPUT_NUMERICAL_IDENTITY: &str = concat!(
    "cutlass-cb4247394dd82148787aed73e5dc7cef33cbf862;Sm80-int8-tensorop;",
    "tile128x128x64;warp64x64x64;instruction16x8x32;stages3;",
    "bounded-exact-I32-dot;F32-RNE-conversion;row-scale-then-channel-scale;",
    "BF16-RNE-store;no-splitK;zero-global-workspace;no-fast-math"
);

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CutlassInt8FusedOutputShapeV1 {
    pub rows: u32,
    pub outputs: u32,
    pub inputs: u32,
    pub reserved: u32,
}
impl CutlassInt8FusedOutputShapeV1 {
    pub fn new(rows: u64, outputs: u64, inputs: u64) -> Result<Self, CutlassInt8FusedOutputError> {
        let mut dims = [0; 3];
        for (out, input) in dims.iter_mut().zip([rows, outputs, inputs]) {
            *out = input
                .try_into()
                .map_err(|_| CutlassInt8FusedOutputError::Geometry)?;
        }
        let result = Self {
            rows: dims[0],
            outputs: dims[1],
            inputs: dims[2],
            reserved: 0,
        };
        result.validate()?;
        Ok(result)
    }
    pub fn validate(&self) -> Result<(), CutlassInt8FusedOutputError> {
        if self.reserved != 0
            || self.rows == 0
            || self.outputs == 0
            || self.inputs == 0
            || self.rows > i32::MAX as u32
            || self.outputs > i32::MAX as u32
            || self.inputs > i32::MAX as u32 / (128 * 128)
            || self.inputs % 16 != 0
            || self.outputs % 8 != 0
        {
            return Err(CutlassInt8FusedOutputError::Geometry);
        }
        Ok(())
    }
    pub fn output_elements(&self) -> u64 {
        u64::from(self.rows) * u64::from(self.outputs)
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CutlassInt8FusedOutputBindingsV1 {
    pub activation: u64,
    pub weight: u64,
    pub row_scale: u64,
    pub channel_scale: u64,
    pub output: u64,
}
impl CutlassInt8FusedOutputBindingsV1 {
    /// Geometry-derived extents are also checked against retained buffer regions
    /// by the provider; integer addresses alone do not confer device authority.
    pub fn validate(
        &self,
        shape: &CutlassInt8FusedOutputShapeV1,
    ) -> Result<(), CutlassInt8FusedOutputError> {
        shape.validate()?;
        let m = u64::from(shape.rows);
        let n = u64::from(shape.outputs);
        let k = u64::from(shape.inputs);
        let reads = [
            (self.activation, m * k, 16),
            (self.weight, n * k, 16),
            (self.row_scale, m * 4, 4),
            (self.channel_scale, n * 4, 4),
        ];
        let output_bytes = m * n * 2;
        let checked = |address: u64, bytes: u64, alignment: u64| {
            if address == 0 || address % alignment != 0 {
                return Err(CutlassInt8FusedOutputError::Span);
            }
            address
                .checked_add(bytes)
                .ok_or(CutlassInt8FusedOutputError::Span)
        };
        let output_end = checked(self.output, output_bytes, 16)?;
        for (address, bytes, alignment) in reads {
            let end = checked(address, bytes, alignment)?;
            if self.output < end && address < output_end {
                return Err(CutlassInt8FusedOutputError::Span);
            }
        }
        Ok(())
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CutlassInt8FusedOutputInfoV1 {
    pub workspace_bytes: u64,
    pub tile_m: u32,
    pub tile_n: u32,
    pub tile_k: u32,
    pub threads: u32,
}
impl CutlassInt8FusedOutputInfoV1 {
    pub fn validate(&self) -> Result<(), CutlassInt8FusedOutputError> {
        if self.workspace_bytes != 0
            || [self.tile_m, self.tile_n, self.tile_k, self.threads] != [128, 128, 64, 128]
        {
            return Err(CutlassInt8FusedOutputError::Plan);
        }
        Ok(())
    }
}

pub type CutlassInt8FusedOutputPrepareFn = unsafe extern "C" fn(
    *const CutlassInt8FusedOutputShapeV1,
    *mut CutlassInt8FusedOutputInfoV1,
) -> i32;
pub type CutlassInt8FusedOutputRunFn = unsafe extern "C" fn(
    *const CutlassInt8FusedOutputShapeV1,
    *const CutlassInt8FusedOutputBindingsV1,
    *mut std::ffi::c_void,
) -> i32;

#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum CutlassInt8FusedOutputError {
    #[error("unsupported fused INT8 geometry or possible I32 accumulation overflow")]
    Geometry,
    #[error("fused INT8 span extent, alignment or output alias violation")]
    Span,
    #[error("fused INT8 native workspace or tile differs from the declared plan")]
    Plan,
}

const _: () = {
    assert!(size_of::<CutlassInt8FusedOutputShapeV1>() == 16);
    assert!(size_of::<CutlassInt8FusedOutputBindingsV1>() == 40);
    assert!(offset_of!(CutlassInt8FusedOutputBindingsV1, output) == 32);
    assert!(size_of::<CutlassInt8FusedOutputInfoV1>() == 24);
    assert!(offset_of!(CutlassInt8FusedOutputInfoV1, tile_m) == 8);
};

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn geometry_accepts_row_tails_and_rejects_overflow_or_vector_misalignment() {
        assert!(CutlassInt8FusedOutputShapeV1::new(3, 8, 16).is_ok());
        assert!(CutlassInt8FusedOutputShapeV1::new(1, 8, 131056).is_ok());
        for dims in [
            (0, 8, 16),
            (1, 7, 16),
            (1, 8, 15),
            (1, 8, 131072),
            (u64::MAX, 8, 16),
        ] {
            assert!(CutlassInt8FusedOutputShapeV1::new(dims.0, dims.1, dims.2).is_err());
        }
    }
    #[test]
    fn buffers_reject_output_alias_and_address_wrap() {
        let shape = CutlassInt8FusedOutputShapeV1::new(3, 8, 16).unwrap();
        let valid = CutlassInt8FusedOutputBindingsV1 {
            activation: 256,
            weight: 512,
            row_scale: 768,
            channel_scale: 1024,
            output: 1280,
        };
        valid.validate(&shape).unwrap();
        for output in [0, 257, 256, 512, 768, 1024, u64::MAX - 15] {
            assert!(CutlassInt8FusedOutputBindingsV1 { output, ..valid }
                .validate(&shape)
                .is_err());
        }
    }
}
