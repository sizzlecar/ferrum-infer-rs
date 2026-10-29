//! Allocation-free boundary for locked SM89/SM120 E4M3FN W8A8 kernels.
//!
//! These descriptors validate geometry and byte ranges. They confer no device
//! resource authority. The caller owns every span, the current context, opaque
//! native capacity and stream until completion (or quarantined retirement).

use std::mem::{offset_of, size_of};

pub const CUTLASS_FP8_ABI: u32 = 1;
pub const CUTLASS_FP8_OPERATOR: &str = "ferrum.cuda.cutlass_fp8_scaled_mm";
pub const CUTLASS_FP8_EXPORTS: [&str; 3] = [
    "ferrum_cutlass_fp8_plan_v1",
    "ferrum_cutlass_fp8_run_v1",
    "ferrum_cutlass_fp8_validate_v1",
];
pub const CUTLASS_FP8_SCRATCH_ALIGNMENT: u64 = 256;
pub const CUTLASS_FP8_KERNEL_128: u32 = 1;
pub const CUTLASS_FP8_KERNEL_SM89: u32 = 2;
pub const CUTLASS_FP8_NUMERICAL_IDENTITY: &str = concat!(
    "cutlass-cb4247394dd82148787aed73e5dc7cef33cbf862;",
    "sm120a=tile128x128x128,cluster1x1x1,persistent-no-stream-k;",
    "sm89=tile128x64x128,warp64x32x128,swizzle8,stages3,no-split-k,f32-scratch-scale;",
    "A=bf16-rowmajor;B=e4m3fn-OI;Y=bf16-rowmajor;",
    "row-amax=fmax(abs(x));scale=max(rn32(amax/448),rn32(1/(448*512)));",
    "quant=E4M3FN-RNE-SATFINITE(fmax(-448,fmin(rn32(x/scale),448)));",
    "sm89-acc=vendor-fp8-mma,empirical-fp22-local-k128,f32-staged-promotion;",
    "sm120-acc=f32;epilogue=B(rn32(weight-scale*acc)*row-scale);",
    "weight-scale=f32-positive-finite-scalar;no-bias;",
    "no-fast-math;qualification=not-established;not-deployment-bitwise-parity"
);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CutlassFp8Device {
    pub ordinal: u32,
    /// CUDA major*100 + minor*10; supported values are 890 and 1200.
    pub architecture: u32,
    pub multiprocessors: u32,
    pub maximum_dynamic_shared_bytes: u64,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[repr(C)]
pub struct CutlassFp8RequestV1 {
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

impl CutlassFp8RequestV1 {
    pub fn new(
        rows: u32,
        inputs: u32,
        outputs: u32,
        device: CutlassFp8Device,
    ) -> Result<Self, CutlassFp8Error> {
        let value = Self {
            abi: CUTLASS_FP8_ABI,
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

    pub fn validate(&self) -> Result<(), CutlassFp8Error> {
        if self.abi != CUTLASS_FP8_ABI
            || self.size as usize != size_of::<Self>()
            || self.rows == 0
            || self.inputs == 0
            || self.outputs == 0
            || self.rows > i32::MAX as u32
            || self.inputs > i32::MAX as u32
            || self.outputs > i32::MAX as u32
            || self.device_ordinal > i32::MAX as u32
            || self.multiprocessors == 0
            || self.multiprocessors > i32::MAX as u32
            || self.maximum_dynamic_shared_bytes == 0
            || self.maximum_dynamic_shared_bytes > u32::MAX as u64
        {
            return Err(CutlassFp8Error::InvalidArgument);
        }
        if !matches!(self.architecture, 890 | 1200)
            || self.inputs % 16 != 0
            || self.outputs % 16 != 0
        {
            return Err(CutlassFp8Error::UnsupportedGeometry);
        }
        self.input_bytes()?;
        self.weight_bytes()?;
        self.output_bytes()?;
        Ok(())
    }
    pub fn input_bytes(&self) -> Result<u64, CutlassFp8Error> {
        extent(self.rows, self.inputs, 2)
    }
    pub fn weight_bytes(&self) -> Result<u64, CutlassFp8Error> {
        extent(self.outputs, self.inputs, 1)
    }
    pub fn output_bytes(&self) -> Result<u64, CutlassFp8Error> {
        extent(self.rows, self.outputs, 2)
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[repr(C)]
pub struct CutlassFp8PlanV1 {
    pub request: CutlassFp8RequestV1,
    pub abi: u32,
    pub size: u32,
    pub kernel_id: u32,
    pub shared_bytes: u32,
    /// Quantized A starts at scratch offset zero.
    pub quantized_activation_bytes: u64,
    pub activation_scale_offset: u64,
    pub activation_scale_bytes: u64,
    pub gemm_workspace_offset: u64,
    pub gemm_workspace_bytes: u64,
    pub scratch_bytes: u64,
}

impl CutlassFp8PlanV1 {
    /// Build/check metadata only. Native preparation must obtain the actual
    /// shared/workspace requirement; this constructor does not certify it.
    pub fn checked(
        request: CutlassFp8RequestV1,
        shared_bytes: u32,
        gemm_workspace_bytes: u64,
    ) -> Result<Self, CutlassFp8Error> {
        request.validate()?;
        if shared_bytes == 0 || u64::from(shared_bytes) > request.maximum_dynamic_shared_bytes {
            return Err(CutlassFp8Error::UnsupportedGeometry);
        }
        let quantized_activation_bytes = extent(request.rows, request.inputs, 1)?;
        let activation_scale_offset = align(quantized_activation_bytes)?;
        let activation_scale_bytes = extent(request.rows, 1, 4)?;
        let gemm_workspace_offset = align(
            activation_scale_offset
                .checked_add(activation_scale_bytes)
                .ok_or(CutlassFp8Error::Extent)?,
        )?;
        if request.architecture == 890
            && gemm_workspace_bytes < align(extent(request.rows, request.outputs, 4)?)?
        {
            return Err(CutlassFp8Error::Extent);
        }
        let scratch_bytes = gemm_workspace_offset
            .checked_add(gemm_workspace_bytes)
            .filter(|n| *n <= isize::MAX as u64)
            .ok_or(CutlassFp8Error::Extent)?;
        Ok(Self {
            request,
            abi: CUTLASS_FP8_ABI,
            size: size_of::<Self>() as u32,
            kernel_id: if request.architecture == 890 {
                CUTLASS_FP8_KERNEL_SM89
            } else {
                CUTLASS_FP8_KERNEL_128
            },
            shared_bytes,
            quantized_activation_bytes,
            activation_scale_offset,
            activation_scale_bytes,
            gemm_workspace_offset,
            gemm_workspace_bytes,
            scratch_bytes,
        })
    }
    pub fn validate(&self) -> Result<(), CutlassFp8Error> {
        if *self != Self::checked(self.request, self.shared_bytes, self.gemm_workspace_bytes)? {
            return Err(CutlassFp8Error::Plan);
        }
        Ok(())
    }
    /// Canonical, explicitly little-endian facts for a provider's prepared
    /// implementation digest. Include the artifact and numerical identity too.
    pub fn fingerprint_material(&self) -> [u8; 104] {
        let mut bytes = [0; 104];
        let u32s = [
            self.request.abi,
            self.request.size,
            self.request.rows,
            self.request.inputs,
            self.request.outputs,
            self.request.device_ordinal,
            self.request.architecture,
            self.request.multiprocessors,
        ];
        for (index, value) in u32s.into_iter().enumerate() {
            bytes[index * 4..index * 4 + 4].copy_from_slice(&value.to_le_bytes());
        }
        bytes[32..40].copy_from_slice(&self.request.maximum_dynamic_shared_bytes.to_le_bytes());
        for (index, value) in [self.abi, self.size, self.kernel_id, self.shared_bytes]
            .into_iter()
            .enumerate()
        {
            bytes[40 + index * 4..44 + index * 4].copy_from_slice(&value.to_le_bytes());
        }
        for (index, value) in [
            self.quantized_activation_bytes,
            self.activation_scale_offset,
            self.activation_scale_bytes,
            self.gemm_workspace_offset,
            self.gemm_workspace_bytes,
            self.scratch_bytes,
        ]
        .into_iter()
        .enumerate()
        {
            bytes[56 + index * 8..64 + index * 8].copy_from_slice(&value.to_le_bytes());
        }
        bytes
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[repr(C)]
pub struct CutlassFp8Span {
    pub address: u64,
    pub bytes: u64,
}

impl CutlassFp8Span {
    fn validate(self, required: u64, alignment: u64) -> Result<(), CutlassFp8Error> {
        if required == 0
            || self.address == 0
            || self.address % alignment != 0
            || self.bytes != required
            || self.address.checked_add(self.bytes).is_none()
            || usize::try_from(self.address).is_err()
        {
            return Err(CutlassFp8Error::Span);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[repr(C)]
pub struct CutlassFp8BindingsV1 {
    pub input: CutlassFp8Span,
    pub weight: CutlassFp8Span,
    pub weight_scale: CutlassFp8Span,
    pub output: CutlassFp8Span,
    pub scratch: CutlassFp8Span,
}
impl CutlassFp8BindingsV1 {
    pub fn validate(&self, plan: &CutlassFp8PlanV1) -> Result<(), CutlassFp8Error> {
        plan.validate()?;
        self.input.validate(plan.request.input_bytes()?, 16)?;
        self.weight.validate(plan.request.weight_bytes()?, 16)?;
        self.weight_scale.validate(4, 4)?;
        self.output.validate(plan.request.output_bytes()?, 16)?;
        self.scratch
            .validate(plan.scratch_bytes, CUTLASS_FP8_SCRATCH_ALIGNMENT)?;
        for write in [self.output, self.scratch] {
            for read in [self.input, self.weight, self.weight_scale] {
                disjoint(write, read)?;
            }
        }
        disjoint(self.output, self.scratch)
    }
}
fn disjoint(a: CutlassFp8Span, b: CutlassFp8Span) -> Result<(), CutlassFp8Error> {
    if a.address
        < b.address
            .checked_add(b.bytes)
            .ok_or(CutlassFp8Error::Span)?
        && b.address
            < a.address
                .checked_add(a.bytes)
                .ok_or(CutlassFp8Error::Span)?
    {
        return Err(CutlassFp8Error::Span);
    }
    Ok(())
}
fn extent(rows: u32, columns: u32, element: u64) -> Result<u64, CutlassFp8Error> {
    u64::from(rows)
        .checked_mul(u64::from(columns))
        .and_then(|n| n.checked_mul(element))
        .filter(|n| *n > 0 && *n <= isize::MAX as u64)
        .ok_or(CutlassFp8Error::Extent)
}
fn align(bytes: u64) -> Result<u64, CutlassFp8Error> {
    bytes
        .checked_add(CUTLASS_FP8_SCRATCH_ALIGNMENT - 1)
        .map(|n| n & !(CUTLASS_FP8_SCRATCH_ALIGNMENT - 1))
        .ok_or(CutlassFp8Error::Extent)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum CutlassFp8Error {
    #[error("invalid CUTLASS FP8 boundary argument")]
    InvalidArgument,
    #[error("unsupported CUTLASS FP8 geometry/device")]
    UnsupportedGeometry,
    #[error("CUTLASS FP8 byte extent overflows")]
    Extent,
    #[error("CUTLASS FP8 span has wrong extent/alignment or aliases a write")]
    Span,
    #[error("CUTLASS FP8 prepared plan differs from its derivation")]
    Plan,
    #[error("locked CUTLASS FP8 native artifact is not compiled")]
    Unavailable,
    #[error("CUTLASS FP8 native boundary returned status {0}")]
    Native(i32),
}

const _: () = {
    assert!(size_of::<CutlassFp8RequestV1>() == 40);
    assert!(size_of::<CutlassFp8PlanV1>() == 104);
    assert!(offset_of!(CutlassFp8PlanV1, quantized_activation_bytes) == 56);
    assert!(size_of::<CutlassFp8BindingsV1>() == 80);
};

#[cfg(test)]
mod tests;
