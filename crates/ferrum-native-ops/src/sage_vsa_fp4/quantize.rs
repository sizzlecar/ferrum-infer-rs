//! Sage's three distinct attention quantization layouts, without allocation.
//!
//! Metadata validation does not authorize device spans. Native execution still
//! checks the current device and pointer kinds, and Core retains every owner.
use super::{
    extent, SageVsaFp4Device as Device, SageVsaFp4Error as Error, SageVsaFp4SpanV1 as Span,
};
use std::mem::{offset_of, size_of};

#[cfg(test)]
mod tests;

pub const EXPORTS: [&str; 3] = [
    "ferrum_sage_vsa_fp4_quantize_plan_v1",
    "ferrum_sage_vsa_fp4_quantize_run_v1",
    "ferrum_sage_vsa_fp4_quantize_validate_v1",
];

pub const NUMERICAL_IDENTITY: &str = concat!(
    "sage-fp4-quantization-4d.f04880acf1e73c48;sm120a;BF16;D128;tile128;",
    "candidate-CUDA12.8.1-source-build;16-elements-per-thread;",
    "group16-BF16-amax-to-F32;amax-over-6=div.approx.ftz.f32;",
    "SF=cvt.rn.satfinite.e4m3x2.f32-to-F16-to-F32;",
    "zero-scale-inverse-zero;inverse=rcp.approx.ftz.f32;",
    "input-times-inverse=mul.ftz.f32;E2M1-RN-satfinite;",
    "Q-row-major;K-within32-row-permutation;V-transposed-sequence-groups;",
    "SF-64row-4column-swizzle;no-centering;no-prefix-slicing"
);

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct Request {
    pub abi: u32,
    pub size: u32,
    pub batches: u32,
    pub heads: u32,
    pub query_rows: u32,
    pub key_rows: u32,
    pub head_dim: u32,
    pub profile: u32,
    pub device_ordinal: u32,
    pub architecture: u32,
    pub reserved: [u32; 2],
    /// BF16 element strides in [batch, head, sequence] order.
    pub query_strides: [u64; 3],
    pub key_strides: [u64; 3],
    pub value_strides: [u64; 3],
}
impl Request {
    pub fn new(
        batches: u32,
        heads: u32,
        query_rows: u32,
        key_rows: u32,
        device: Device,
    ) -> Result<Self, Error> {
        let value = Self {
            abi: 1,
            size: size_of::<Self>() as u32,
            batches,
            heads,
            query_rows,
            key_rows,
            head_dim: 128,
            profile: 1,
            device_ordinal: device.ordinal,
            architecture: device.architecture,
            reserved: [0; 2],
            query_strides: strides(heads, query_rows)?,
            key_strides: strides(heads, key_rows)?,
            value_strides: strides(heads, key_rows)?,
        };
        value.validate()?;
        Ok(value)
    }
    pub fn validate(&self) -> Result<(), Error> {
        if self.abi != 1
            || self.size as usize != size_of::<Self>()
            || self.reserved != [0; 2]
            || self.device_ordinal > i32::MAX as u32
        {
            return Err(Error::InvalidArgument);
        }
        if self.profile != 1
            || self.architecture != 1200
            || self.head_dim != 128
            || [self.batches, self.heads, self.query_rows, self.key_rows].contains(&0)
            || self.query_rows % 128 != 0
            || self.key_rows % 128 != 0
        {
            return Err(Error::UnsupportedGeometry);
        }
        for rows in [self.query_rows, self.key_rows] {
            if extent(&[self.batches.into(), rows.into(), self.heads.into(), 128])?
                > i32::MAX as u64
            {
                return Err(Error::Extent);
            }
        }
        if self.query_strides != strides(self.heads, self.query_rows)?
            || self.key_strides != strides(self.heads, self.key_rows)?
            || self.value_strides != strides(self.heads, self.key_rows)?
        {
            return Err(Error::UnsupportedGeometry);
        }
        Ok(())
    }
    pub fn buffer_sizes(&self) -> Result<BufferSizes, Error> {
        self.validate()?;
        let [b, h, q, k] =
            [self.batches, self.heads, self.query_rows, self.key_rows].map(u64::from);
        Ok(BufferSizes {
            query_input: extent(&[b, h, q, 128, 2])?,
            key_input: extent(&[b, h, k, 128, 2])?,
            value_input: extent(&[b, h, k, 128, 2])?,
            query: extent(&[b, h, q, 64])?,
            key: extent(&[b, h, k, 64])?,
            value: extent(&[b, h, 128, k / 2])?,
            query_scales: extent(&[b, h, q, 8])?,
            key_scales: extent(&[b, h, k, 8])?,
            value_scales: extent(&[b, h, 128, k / 16])?,
        })
    }
}
fn strides(heads: u32, rows: u32) -> Result<[u64; 3], Error> {
    let width = extent(&[heads.into(), 128])?;
    Ok([extent(&[rows.into(), width])?, 128, width])
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct BufferSizes {
    pub query_input: u64,
    pub key_input: u64,
    pub value_input: u64,
    pub query: u64,
    pub key: u64,
    pub value: u64,
    pub query_scales: u64,
    pub key_scales: u64,
    pub value_scales: u64,
}
impl BufferSizes {
    pub fn all(self) -> [u64; 9] {
        [
            self.query_input,
            self.key_input,
            self.value_input,
            self.query,
            self.key,
            self.value,
            self.query_scales,
            self.key_scales,
            self.value_scales,
        ]
    }
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct Plan {
    pub request: Request,
    pub abi: u32,
    pub size: u32,
    pub query_grid_x: u32,
    pub key_grid_x: u32,
    pub grid_y: u32,
    pub grid_z: u32,
    pub threads: u32,
    pub launch_count: u32,
    pub query_shared_bytes: u32,
    pub key_shared_bytes: u32,
    pub value_shared_bytes: u32,
    pub reserved: u32,
    pub bytes: BufferSizes,
}
impl Plan {
    pub fn validate(&self) -> Result<(), Error> {
        self.request.validate()?;
        if self.abi != 1
            || self.size as usize != size_of::<Self>()
            || self.reserved != 0
            || self.query_grid_x != self.request.query_rows / 128
            || self.key_grid_x != self.request.key_rows / 128
            || self.grid_y != self.request.batches
            || self.grid_z != self.request.heads
            || self.threads != 1024
            || self.launch_count != 3
            || self.query_shared_bytes != 0
            || self.key_shared_bytes != 0
            || self.value_shared_bytes != 128 * 128 * 2
            || self.bytes != self.request.buffer_sizes()?
        {
            return Err(Error::Plan);
        }
        Ok(())
    }
    /// Explicit little-endian fields; never hash raw Rust/C padding.
    pub fn fingerprint_material(&self) -> [u8; 240] {
        let mut bytes = [0u8; 240];
        let mut offset = 0;
        let mut append = |part: &[u8]| {
            bytes[offset..offset + part.len()].copy_from_slice(part);
            offset += part.len();
        };
        let r = self.request;
        for value in [
            r.abi,
            r.size,
            r.batches,
            r.heads,
            r.query_rows,
            r.key_rows,
            r.head_dim,
            r.profile,
            r.device_ordinal,
            r.architecture,
            r.reserved[0],
            r.reserved[1],
        ] {
            append(&value.to_le_bytes());
        }
        for value in r
            .query_strides
            .into_iter()
            .chain(r.key_strides)
            .chain(r.value_strides)
        {
            append(&value.to_le_bytes());
        }
        for value in [
            self.abi,
            self.size,
            self.query_grid_x,
            self.key_grid_x,
            self.grid_y,
            self.grid_z,
            self.threads,
            self.launch_count,
            self.query_shared_bytes,
            self.key_shared_bytes,
            self.value_shared_bytes,
            self.reserved,
        ] {
            append(&value.to_le_bytes());
        }
        for value in self.bytes.all() {
            append(&value.to_le_bytes());
        }
        bytes
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct Bindings {
    pub query_input: Span,
    pub key_input: Span,
    pub value_input: Span,
    pub query: Span,
    pub key: Span,
    pub value: Span,
    pub query_scales: Span,
    pub key_scales: Span,
    pub value_scales: Span,
}
impl Bindings {
    pub fn validate(&self, plan: &Plan) -> Result<(), Error> {
        plan.validate()?;
        let spans = [
            self.query_input,
            self.key_input,
            self.value_input,
            self.query,
            self.key,
            self.value,
            self.query_scales,
            self.key_scales,
            self.value_scales,
        ];
        for ((span, bytes), alignment) in spans
            .into_iter()
            .zip(plan.bytes.all())
            .zip([4, 4, 4, 8, 8, 8, 1, 1, 1])
        {
            span.validate(bytes, alignment)?;
        }
        for i in 3..spans.len() {
            for other in &spans[..i] {
                if spans[i].address < other.address + other.bytes
                    && other.address < spans[i].address + spans[i].bytes
                {
                    return Err(Error::Span);
                }
            }
        }
        Ok(())
    }
}
const _: () = {
    assert!(size_of::<Request>() == 120);
    assert!(offset_of!(Request, query_strides) == 48);
    assert!(size_of::<BufferSizes>() == 72);
    assert!(size_of::<Plan>() == 240);
    assert!(offset_of!(Plan, bytes) == 168);
    assert!(size_of::<Bindings>() == 144);
};
