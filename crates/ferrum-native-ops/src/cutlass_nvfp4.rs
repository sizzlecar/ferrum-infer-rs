//! Allocation-free packed NVFP4 C ABI. Byte validation grants no resource authority.
use std::mem::{offset_of, size_of};
mod quantize;
pub use quantize::*;

pub const CUTLASS_NVFP4_ABI: u32 = 1;
pub const CUTLASS_NVFP4_OPERATOR: &str = "ferrum.cuda.cutlass_nvfp4_matmul";
pub const CUTLASS_NVFP4_EXPORTS: [&str; 6] = [
    "ferrum_cutlass_nvfp4_plan_v1",
    "ferrum_cutlass_nvfp4_run_v1",
    "ferrum_cutlass_nvfp4_validate_v1",
    "ferrum_nvfp4_quantize_bf16_plan_v1",
    "ferrum_nvfp4_quantize_bf16_validate_v1",
    "ferrum_nvfp4_quantize_bf16_run_v1",
];
pub const CUTLASS_NVFP4_ALIGNMENT: u64 = 256;
pub const CUTLASS_NVFP4_NUMERICAL_IDENTITY: &str = concat!(
    "cutlass-cb4247394dd82148787aed73e5dc7cef33cbf862;sm120a;",
    "tile128x128x128,cluster1x1x1,static-persistent,no-stream-k,no-swap-ab,no-pdl;",
    "A=E2M1-rowmajor;B=E2M1-OI;low-nibble-first;group16;scales=nonnegative-E4M3FN-128x4;",
    "acc=blockscaled-tensorcore-f32;epilogue=BF16-RNE(alpha-f32*acc);",
    "alpha=device-f32-positive-finite;no-bias;no-quantize;no-fast-math;qualification=not-established"
);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CutlassNvfp4Device {
    pub ordinal: u32,
    pub architecture: u32,
    pub multiprocessors: u32,
    pub maximum_dynamic_shared_bytes: u64,
}
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[repr(C)]
pub struct CutlassNvfp4RequestV1 {
    pub abi: u32,
    pub size: u32,
    pub rows: u32,
    pub inputs: u32,
    pub outputs: u32,
    pub device_ordinal: u32,
    pub architecture: u32,
    pub multiprocessors: u32,
    pub maximum_dynamic_shared_bytes: u64,
}
impl CutlassNvfp4RequestV1 {
    pub fn new(
        rows: u32,
        inputs: u32,
        outputs: u32,
        device: CutlassNvfp4Device,
    ) -> Result<Self, CutlassNvfp4Error> {
        let value = Self {
            abi: CUTLASS_NVFP4_ABI,
            size: size_of::<Self>() as u32,
            rows,
            inputs,
            outputs,
            device_ordinal: device.ordinal,
            architecture: device.architecture,
            multiprocessors: device.multiprocessors,
            maximum_dynamic_shared_bytes: device.maximum_dynamic_shared_bytes,
        };
        value.validate()?;
        Ok(value)
    }
    pub fn validate(&self) -> Result<(), CutlassNvfp4Error> {
        if self.abi != CUTLASS_NVFP4_ABI
            || self.size as usize != size_of::<Self>()
            || [self.rows, self.inputs, self.outputs, self.multiprocessors]
                .iter()
                .any(|v| *v == 0 || *v > i32::MAX as u32)
            || self.device_ordinal > i32::MAX as u32
            || self.maximum_dynamic_shared_bytes == 0
            || self.maximum_dynamic_shared_bytes > u32::MAX as u64
        {
            return Err(CutlassNvfp4Error::InvalidArgument);
        }
        // No K tail: inactive scale padding need not be numerically meaningful.
        // BF16 epilogue requires 16-byte row alignment. Broader K-tail behavior
        // requires separate numerical qualification.
        if self.architecture != 1200 || self.inputs % 128 != 0 || self.outputs % 8 != 0 {
            return Err(CutlassNvfp4Error::UnsupportedGeometry);
        }
        self.input_bytes()?;
        self.weight_bytes()?;
        self.input_scale_bytes()?;
        self.weight_scale_bytes()?;
        self.output_bytes()?;
        Ok(())
    }
    pub fn input_bytes(&self) -> Result<u64, CutlassNvfp4Error> {
        extent(self.rows.into(), u64::from(self.inputs) / 2)
    }
    pub fn weight_bytes(&self) -> Result<u64, CutlassNvfp4Error> {
        extent(self.outputs.into(), u64::from(self.inputs) / 2)
    }
    pub fn input_scale_bytes(&self) -> Result<u64, CutlassNvfp4Error> {
        scale_extent(self.rows, self.inputs)
    }
    pub fn weight_scale_bytes(&self) -> Result<u64, CutlassNvfp4Error> {
        scale_extent(self.outputs, self.inputs)
    }
    pub fn output_bytes(&self) -> Result<u64, CutlassNvfp4Error> {
        extent(self.rows.into(), u64::from(self.outputs) * 2)
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[repr(C)]
pub struct CutlassNvfp4PlanV1 {
    pub request: CutlassNvfp4RequestV1,
    pub abi: u32,
    pub size: u32,
    pub kernel_id: u32,
    pub shared_bytes: u32,
    pub workspace_bytes: u64,
}
impl CutlassNvfp4PlanV1 {
    /// Structural metadata only; native preparation determines actual sizes.
    pub fn checked(
        request: CutlassNvfp4RequestV1,
        shared_bytes: u32,
        workspace_bytes: u64,
    ) -> Result<Self, CutlassNvfp4Error> {
        request.validate()?;
        if shared_bytes == 0
            || u64::from(shared_bytes) > request.maximum_dynamic_shared_bytes
            || workspace_bytes > isize::MAX as u64
        {
            return Err(CutlassNvfp4Error::UnsupportedGeometry);
        }
        Ok(Self {
            request,
            abi: CUTLASS_NVFP4_ABI,
            size: size_of::<Self>() as u32,
            kernel_id: 1,
            shared_bytes,
            workspace_bytes,
        })
    }
    pub fn validate(&self) -> Result<(), CutlassNvfp4Error> {
        if *self != Self::checked(self.request, self.shared_bytes, self.workspace_bytes)? {
            return Err(CutlassNvfp4Error::Plan);
        }
        Ok(())
    }
    pub fn fingerprint_material(&self) -> [u8; 64] {
        let mut result = [0; 64];
        for (i, value) in [
            self.request.abi,
            self.request.size,
            self.request.rows,
            self.request.inputs,
            self.request.outputs,
            self.request.device_ordinal,
            self.request.architecture,
            self.request.multiprocessors,
        ]
        .into_iter()
        .enumerate()
        {
            result[i * 4..i * 4 + 4].copy_from_slice(&value.to_le_bytes());
        }
        result[32..40].copy_from_slice(&self.request.maximum_dynamic_shared_bytes.to_le_bytes());
        for (i, value) in [self.abi, self.size, self.kernel_id, self.shared_bytes]
            .into_iter()
            .enumerate()
        {
            result[40 + i * 4..44 + i * 4].copy_from_slice(&value.to_le_bytes());
        }
        result[56..64].copy_from_slice(&self.workspace_bytes.to_le_bytes());
        result
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[repr(C)]
pub struct CutlassNvfp4Span {
    pub address: u64,
    pub bytes: u64,
}
impl CutlassNvfp4Span {
    fn validate(self, required: u64, alignment: u64) -> Result<(), CutlassNvfp4Error> {
        if required == 0 {
            return if self == Self::default() {
                Ok(())
            } else {
                Err(CutlassNvfp4Error::Span)
            };
        }
        if self.address == 0
            || self.address % alignment != 0
            || self.bytes != required
            || self
                .address
                .checked_add(self.bytes)
                .and_then(|v| usize::try_from(v).ok())
                .is_none()
        {
            return Err(CutlassNvfp4Error::Span);
        }
        Ok(())
    }
}
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[repr(C)]
pub struct CutlassNvfp4BindingsV1 {
    pub input: CutlassNvfp4Span,
    pub weight: CutlassNvfp4Span,
    pub input_scales: CutlassNvfp4Span,
    pub weight_scales: CutlassNvfp4Span,
    pub alpha: CutlassNvfp4Span,
    pub output: CutlassNvfp4Span,
    pub workspace: CutlassNvfp4Span,
}
impl CutlassNvfp4BindingsV1 {
    pub fn validate(&self, plan: &CutlassNvfp4PlanV1) -> Result<(), CutlassNvfp4Error> {
        plan.validate()?;
        self.input.validate(plan.request.input_bytes()?, 16)?;
        self.weight.validate(plan.request.weight_bytes()?, 16)?;
        self.input_scales
            .validate(plan.request.input_scale_bytes()?, 16)?;
        self.weight_scales
            .validate(plan.request.weight_scale_bytes()?, 16)?;
        self.alpha.validate(4, 4)?;
        self.output.validate(plan.request.output_bytes()?, 16)?;
        self.workspace
            .validate(plan.workspace_bytes, CUTLASS_NVFP4_ALIGNMENT)?;
        for write in [self.output, self.workspace] {
            for read in [
                self.input,
                self.weight,
                self.input_scales,
                self.weight_scales,
                self.alpha,
            ] {
                disjoint(write, read)?;
            }
        }
        disjoint(self.output, self.workspace)
    }
}
fn disjoint(a: CutlassNvfp4Span, b: CutlassNvfp4Span) -> Result<(), CutlassNvfp4Error> {
    if a.bytes != 0
        && b.bytes != 0
        && a.address
            < b.address
                .checked_add(b.bytes)
                .ok_or(CutlassNvfp4Error::Span)?
        && b.address
            < a.address
                .checked_add(a.bytes)
                .ok_or(CutlassNvfp4Error::Span)?
    {
        return Err(CutlassNvfp4Error::Span);
    }
    Ok(())
}
fn extent(a: u64, b: u64) -> Result<u64, CutlassNvfp4Error> {
    a.checked_mul(b)
        .filter(|v| *v > 0 && *v <= isize::MAX as u64)
        .ok_or(CutlassNvfp4Error::Extent)
}
fn scale_extent(rows: u32, columns: u32) -> Result<u64, CutlassNvfp4Error> {
    extent(
        u64::from(rows).div_ceil(128) * 128,
        (u64::from(columns) / 16).div_ceil(4) * 4,
    )
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum CutlassNvfp4Error {
    #[error("invalid CUTLASS NVFP4 boundary argument")]
    InvalidArgument,
    #[error("unsupported CUTLASS NVFP4 geometry/device")]
    UnsupportedGeometry,
    #[error("CUTLASS NVFP4 extent exceeds address space")]
    Extent,
    #[error("CUTLASS NVFP4 span has wrong extent/alignment or aliases a write")]
    Span,
    #[error("CUTLASS NVFP4 plan differs from derivation")]
    Plan,
    #[error("locked CUTLASS NVFP4 artifact is not compiled")]
    Unavailable,
    #[error("CUTLASS NVFP4 native status {0}; submission may have started")]
    Native(i32),
}
const _: () = {
    assert!(size_of::<CutlassNvfp4RequestV1>() == 40);
    assert!(size_of::<CutlassNvfp4PlanV1>() == 64);
    assert!(offset_of!(CutlassNvfp4PlanV1, workspace_bytes) == 56);
    assert!(size_of::<CutlassNvfp4BindingsV1>() == 112);
};

#[cfg(test)]
mod tests;
