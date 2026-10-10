//! One prepared leaf retains its exact native boundary type. The common
//! geometry below is metadata only: it is never passed across an FFI boundary.
use crate::native_ops::upstream_linear::{Arithmetic, DeviceSpan, Error, PreparedUpstreamLinear};
use crate::native_ops::upstream_q6_f16_linear::PreparedQ6F16Linear;
use std::ffi::c_void;

pub(in crate::backend::cuda::vnext_ops) enum PreparedProjectionNative {
    Upstream(PreparedUpstreamLinear),
    Q6F16(PreparedQ6F16Linear),
}

#[derive(Clone, Copy)]
pub(in crate::backend::cuda::vnext_ops) struct NativeRequestShape {
    pub format: u32,
    pub inputs: u32,
    pub outputs: u32,
}

#[derive(Clone, Copy)]
pub(in crate::backend::cuda::vnext_ops) struct NativeGeometry {
    pub request: NativeRequestShape,
    pub algorithm: u32,
    pub padded_inputs: u32,
    pub padded_outputs: u32,
    pub guard_blocks: u32,
    pub j: u32,
    pub i: u32,
    pub nthreads: u32,
    pub shared_bytes: u32,
    pub blocks: u32,
    pub fixup: u32,
    pub ncols: u32,
    pub channels: u32,
    pub nwarps: u32,
    pub rows_per_block: u32,
    pub weight_bytes: u64,
}

impl PreparedProjectionNative {
    pub fn operator(&self) -> &'static str {
        match self {
            Self::Upstream(plan) => plan.operator(),
            Self::Q6F16(plan) => plan.operator(),
        }
    }

    pub fn boundary_abi(&self) -> u32 {
        match self {
            Self::Upstream(_) => 1,
            Self::Q6F16(_) => ferrum_native_ops::upstream_q6_f16_linear::UPSTREAM_Q6_F16_LINEAR_ABI,
        }
    }

    pub fn arithmetic(&self) -> Arithmetic {
        match self {
            Self::Upstream(plan) => plan.arithmetic(),
            Self::Q6F16(_) => Arithmetic::MarkerV2,
        }
    }

    pub fn geometry(&self) -> NativeGeometry {
        macro_rules! metadata {
            ($p:expr) => {{
                let p = $p;
                NativeGeometry {
                    request: NativeRequestShape {
                        format: p.request.format,
                        inputs: p.request.inputs,
                        outputs: p.request.outputs,
                    },
                    algorithm: p.algorithm,
                    padded_inputs: p.padded_inputs,
                    padded_outputs: p.padded_outputs,
                    guard_blocks: p.guard_blocks,
                    j: p.j,
                    i: p.i,
                    nthreads: p.nthreads,
                    shared_bytes: p.shared_bytes,
                    blocks: p.blocks,
                    fixup: p.fixup,
                    ncols: p.ncols,
                    channels: p.channels,
                    nwarps: p.nwarps,
                    rows_per_block: p.rows_per_block,
                    weight_bytes: p.weight_bytes,
                }
            }};
        }
        match self {
            Self::Upstream(plan) => metadata!(plan.geometry()),
            Self::Q6F16(plan) => metadata!(plan.geometry()),
        }
    }

    pub unsafe fn check_weights(
        &self,
        weights: DeviceSpan,
        flag: DeviceSpan,
        stream: *mut c_void,
    ) -> Result<(), Error> {
        match self {
            Self::Upstream(plan) => unsafe { plan.check_weights(weights, flag, stream) },
            Self::Q6F16(plan) => unsafe { plan.check_weights(weights, flag, stream) },
        }
    }

    pub unsafe fn pack(
        &self,
        input: DeviceSpan,
        stride: u32,
        converted: DeviceSpan,
        packed: DeviceSpan,
        rows: DeviceSpan,
        stream: *mut c_void,
    ) -> Result<(), Error> {
        match self {
            Self::Upstream(plan) => unsafe {
                plan.pack(input, stride, converted, packed, rows, stream)
            },
            Self::Q6F16(plan) => unsafe {
                plan.pack(input, stride, converted, packed, rows, stream)
            },
        }
    }

    pub unsafe fn dot(
        &self,
        weights: DeviceSpan,
        packed: DeviceSpan,
        raw: DeviceSpan,
        fixup: DeviceSpan,
        stream: *mut c_void,
    ) -> Result<(), Error> {
        match self {
            Self::Upstream(plan) => unsafe { plan.dot(weights, packed, raw, fixup, stream) },
            Self::Q6F16(plan) => unsafe { plan.dot(weights, packed, raw, fixup, stream) },
        }
    }

    pub unsafe fn cast(
        &self,
        raw: DeviceSpan,
        output: DeviceSpan,
        stride: u32,
        rows: DeviceSpan,
        flag: DeviceSpan,
        stream: *mut c_void,
    ) -> Result<(), Error> {
        match self {
            Self::Upstream(plan) => unsafe { plan.cast(raw, output, stride, rows, flag, stream) },
            Self::Q6F16(plan) => unsafe { plan.cast(raw, output, stride, rows, flag, stream) },
        }
    }
}
