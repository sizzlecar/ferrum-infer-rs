//! Checked CPU-list entry to the fixed SM120 sparse FP4 attention ABI.
//!
//! Pure metadata validation neither pins Host memory nor admits device buffers.
//! Native Run revalidates caller-owned pinned Host lists, uploads them to its
//! explicit metadata write spans, then launches on the supplied stream. Core
//! must retain all Host/device/code owners through completion, including errors
//! after the first upload. No arbitrary device I32 tensor is a trusted list.

use std::mem::{offset_of, size_of};

mod lists;
pub use lists::{CheckedCpuBlockLists, SageVsaFp4ListShape};
pub mod quantize;
pub mod selection;
#[cfg(test)]
mod tests;

pub const SAGE_VSA_FP4_ABI: u32 = 1;
pub const SAGE_VSA_FP4_OPERATOR: &str = "ferrum.cuda.sage_vsa_fp4_cpu_lists";
pub const SAGE_VSA_FP4_PROFILE: u32 = 1;
pub const SAGE_VSA_FP4_EXPORTS: [&str; 12] = [
    "ferrum_sage_vsa_fp4_plan_v1",
    "ferrum_sage_vsa_fp4_validate_v1",
    "ferrum_sage_vsa_fp4_run_v1",
    "ferrum_sage_vsa_fp4_region_plan_v1",
    "ferrum_sage_vsa_fp4_region_validate_v1",
    "ferrum_sage_vsa_fp4_region_run_v1",
    "ferrum_sage_vsa_fp4_region_kernel_plan_v1",
    "ferrum_sage_vsa_fp4_region_kernel_validate_v1",
    "ferrum_sage_vsa_fp4_region_kernel_run_v1",
    "ferrum_sage_vsa_fp4_quantize_plan_v1",
    "ferrum_sage_vsa_fp4_quantize_validate_v1",
    "ferrum_sage_vsa_fp4_quantize_run_v1",
];
pub const SAGE_VSA_FP4_NUMERICAL_IDENTITY: &str = concat!(
    "fastvideo-866a3cdb91a85aef8561847365fc8564add9e4c8-attn_qat_infer;",
    "cutlass-4.3.0-deployed-header-closure;sm120a;D128;tile128x128;",
    "noncausal;Q-unsmoothed;deltaS-zero;two-level-P;BF16-output;",
    "QKV=E2M1-group16-E4M3,distinct-Q-K-V-physical-layouts;",
    "use-fast-math;cpu-checked-pinned-lists-upload;no-allocator;",
    "qualification=not-established"
);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SageVsaFp4Device {
    pub ordinal: u32,
    pub architecture: u32,
    pub multiprocessors: u32,
    pub maximum_dynamic_shared_bytes: u64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct SageVsaFp4RequestV1 {
    pub abi: u32,
    pub size: u32,
    pub batches: u32,
    pub heads: u32,
    pub query_blocks: u32,
    pub key_blocks: u32,
    pub head_dim: u32,
    pub tile_rows: u32,
    pub profile: u32,
    pub softmax_scale_bits: u32,
    pub device_ordinal: u32,
    pub architecture: u32,
    pub multiprocessors: u32,
    pub reserved: u32,
    pub maximum_dynamic_shared_bytes: u64,
}
impl SageVsaFp4RequestV1 {
    pub fn new(
        batches: u32,
        heads: u32,
        query_blocks: u32,
        key_blocks: u32,
        softmax_scale: f32,
        device: SageVsaFp4Device,
    ) -> Result<Self, SageVsaFp4Error> {
        let value = Self {
            abi: 1,
            size: size_of::<Self>() as u32,
            batches,
            heads,
            query_blocks,
            key_blocks,
            head_dim: 128,
            tile_rows: 128,
            profile: SAGE_VSA_FP4_PROFILE,
            softmax_scale_bits: softmax_scale.to_bits(),
            device_ordinal: device.ordinal,
            architecture: device.architecture,
            multiprocessors: device.multiprocessors,
            reserved: 0,
            maximum_dynamic_shared_bytes: device.maximum_dynamic_shared_bytes,
        };
        value.validate()?;
        Ok(value)
    }
    pub fn validate(&self) -> Result<(), SageVsaFp4Error> {
        let scale = f32::from_bits(self.softmax_scale_bits);
        if self.abi != 1
            || self.size as usize != size_of::<Self>()
            || self.reserved != 0
            || !scale.is_finite()
            || scale <= 0.0
            || self.device_ordinal > i32::MAX as u32
            || [
                self.batches,
                self.heads,
                self.query_blocks,
                self.key_blocks,
                self.multiprocessors,
            ]
            .iter()
            .any(|&v| v == 0 || v > i32::MAX as u32)
            || self.maximum_dynamic_shared_bytes == 0
            || self.maximum_dynamic_shared_bytes > u32::MAX as u64
        {
            return Err(SageVsaFp4Error::InvalidArgument);
        }
        if self.architecture != 1200
            || self.head_dim != 128
            || self.tile_rows != 128
            || self.profile != SAGE_VSA_FP4_PROFILE
            || self.query_blocks > self.key_blocks
        {
            return Err(SageVsaFp4Error::UnsupportedGeometry);
        }
        let q = u64::from(self.query_blocks) * 128;
        let k = u64::from(self.key_blocks) * 128;
        let bh = u64::from(self.batches) * u64::from(self.heads);
        // Native scheduler and counts indexing are int; LSE head stride is
        // formed by int multiplication in the unchanged epilogue adapter.
        if q > i32::MAX as u64 || k > i32::MAX as u64
            // The persistent scheduler also adds gridDim.x after its last
            // valid tile; that terminal advance must remain representable.
            || bh.checked_mul(u64::from(self.query_blocks)).is_none_or(|n| n > i32::MAX as u64 - u64::from(self.multiprocessors) + 1)
            || u64::from(self.heads) * q > i32::MAX as u64
        {
            return Err(SageVsaFp4Error::Extent);
        }
        // The unchanged CuTe scale/delta layouts build their logical shapes
        // and strides with int32_t. Bound that arithmetic before native layout
        // construction, even though Q/K/V byte offsets themselves use int64_t.
        if extent(&[bh, k, 128])? > i32::MAX as u64 {
            return Err(SageVsaFp4Error::Extent);
        }
        self.buffer_sizes()?;
        Ok(())
    }
    pub fn buffer_sizes(&self) -> Result<SageVsaFp4BufferSizesV1, SageVsaFp4Error> {
        let b = u64::from(self.batches);
        let h = u64::from(self.heads);
        let q = u64::from(self.query_blocks) * 128;
        let k = u64::from(self.key_blocks) * 128;
        Ok(SageVsaFp4BufferSizesV1 {
            query: extent(&[b, h, q, 64])?,
            key: extent(&[b, h, k, 64])?,
            value: extent(&[b, h, 128, k / 2])?,
            query_scales: extent(&[b, h, q, 8])?,
            key_scales: extent(&[b, h, k, 8])?,
            value_scales: extent(&[b, h, 128, k / 16])?,
            output: extent(&[b, h, q, 128, 2])?,
            lse: extent(&[b, h, q, 4])?,
            zero_delta: extent(&[b, h, k, 4])?,
            indices: extent(&[
                b,
                h,
                u64::from(self.query_blocks),
                u64::from(self.key_blocks),
                4,
            ])?,
            counts: extent(&[b, h, u64::from(self.query_blocks), 4])?,
            half_valid: extent(&[u64::from(self.key_blocks), 2, 4])?,
        })
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct SageVsaFp4BufferSizesV1 {
    pub query: u64,
    pub key: u64,
    pub value: u64,
    pub query_scales: u64,
    pub key_scales: u64,
    pub value_scales: u64,
    pub output: u64,
    pub lse: u64,
    pub zero_delta: u64,
    pub indices: u64,
    pub counts: u64,
    pub half_valid: u64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct SageVsaFp4PlanV1 {
    pub request: SageVsaFp4RequestV1,
    pub abi: u32,
    pub size: u32,
    pub kernel_id: u32,
    pub shared_bytes: u32,
    pub threads: u32,
    pub grid_blocks: u32,
    pub upload_count: u32,
    pub memset_count: u32,
    pub bytes: SageVsaFp4BufferSizesV1,
}
impl SageVsaFp4PlanV1 {
    pub fn fingerprint_material(&self) -> [u8; 192] {
        let mut bytes = [0u8; 192];
        let mut offset = 0;
        let mut append = |value: &[u8]| {
            bytes[offset..offset + value.len()].copy_from_slice(value);
            offset += value.len();
        };
        let r = self.request;
        for v in [
            r.abi,
            r.size,
            r.batches,
            r.heads,
            r.query_blocks,
            r.key_blocks,
            r.head_dim,
            r.tile_rows,
            r.profile,
            r.softmax_scale_bits,
            r.device_ordinal,
            r.architecture,
            r.multiprocessors,
            r.reserved,
        ] {
            append(&v.to_le_bytes());
        }
        append(&r.maximum_dynamic_shared_bytes.to_le_bytes());
        for v in [
            self.abi,
            self.size,
            self.kernel_id,
            self.shared_bytes,
            self.threads,
            self.grid_blocks,
            self.upload_count,
            self.memset_count,
        ] {
            append(&v.to_le_bytes());
        }
        let s = self.bytes;
        for v in [
            s.query,
            s.key,
            s.value,
            s.query_scales,
            s.key_scales,
            s.value_scales,
            s.output,
            s.lse,
            s.zero_delta,
            s.indices,
            s.counts,
            s.half_valid,
        ] {
            append(&v.to_le_bytes());
        }
        bytes
    }
    /// Checks returned metadata; native Plan alone determines compiled kernel
    /// shared-memory size. This does not certify a device or permit submission.
    pub fn validate(&self) -> Result<(), SageVsaFp4Error> {
        self.request.validate()?;
        if self.abi != 1
            || self.size as usize != size_of::<Self>()
            || self.kernel_id != 1
            || self.shared_bytes == 0
            || u64::from(self.shared_bytes) > self.request.maximum_dynamic_shared_bytes
            || self.threads != 384
            || self.grid_blocks != self.request.multiprocessors
            || self.upload_count != 3
            || self.memset_count != 1
            || self.bytes != self.request.buffer_sizes()?
        {
            return Err(SageVsaFp4Error::Plan);
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct SageVsaFp4SpanV1 {
    pub address: u64,
    pub bytes: u64,
}
impl SageVsaFp4SpanV1 {
    fn validate(self, bytes: u64, alignment: u64) -> Result<(), SageVsaFp4Error> {
        if self.address == 0
            || self.address % alignment != 0
            || self.bytes != bytes
            || self
                .address
                .checked_add(bytes)
                .and_then(|n| usize::try_from(n).ok())
                .is_none()
        {
            return Err(SageVsaFp4Error::Span);
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct SageVsaFp4BindingsV1 {
    pub query: SageVsaFp4SpanV1,
    pub key: SageVsaFp4SpanV1,
    pub value: SageVsaFp4SpanV1,
    pub query_scales: SageVsaFp4SpanV1,
    pub key_scales: SageVsaFp4SpanV1,
    pub value_scales: SageVsaFp4SpanV1,
    pub output: SageVsaFp4SpanV1,
    pub lse: SageVsaFp4SpanV1,
    pub zero_delta: SageVsaFp4SpanV1,
    pub indices: SageVsaFp4SpanV1,
    pub counts: SageVsaFp4SpanV1,
    pub half_valid: SageVsaFp4SpanV1,
}
impl SageVsaFp4BindingsV1 {
    pub fn validate(&self, plan: &SageVsaFp4PlanV1) -> Result<(), SageVsaFp4Error> {
        plan.validate()?;
        let sizes = plan.bytes;
        let reads = [
            self.query,
            self.key,
            self.value,
            self.query_scales,
            self.key_scales,
            self.value_scales,
        ];
        let writes = [
            self.output,
            self.lse,
            self.zero_delta,
            self.indices,
            self.counts,
            self.half_valid,
        ];
        for (span, bytes) in reads.into_iter().zip([
            sizes.query,
            sizes.key,
            sizes.value,
            sizes.query_scales,
            sizes.key_scales,
            sizes.value_scales,
        ]) {
            span.validate(bytes, 16)?;
        }
        for (span, bytes, alignment) in [
            (self.output, sizes.output, 16),
            (self.lse, sizes.lse, 4),
            (self.zero_delta, sizes.zero_delta, 16),
            (self.indices, sizes.indices, 4),
            (self.counts, sizes.counts, 4),
            (self.half_valid, sizes.half_valid, 4),
        ] {
            span.validate(bytes, alignment)?;
        }
        for (i, write) in writes.iter().enumerate() {
            for other in reads.iter().chain(writes[..i].iter()) {
                if write.address < other.address + other.bytes
                    && other.address < write.address + write.bytes
                {
                    return Err(SageVsaFp4Error::Span);
                }
            }
        }
        Ok(())
    }
}

/// Host addresses, never device input bindings. All three allocations must be
/// pinned and Core-admitted; native Validate verifies their CUDA Host memory
/// kind and reads their contents. Caller guarantees valid immutable extents.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct SageVsaFp4HostListsV1 {
    pub indices: SageVsaFp4SpanV1,
    pub counts: SageVsaFp4SpanV1,
    pub half_valid: SageVsaFp4SpanV1,
}

fn extent(values: &[u64]) -> Result<u64, SageVsaFp4Error> {
    values
        .iter()
        .try_fold(1u64, |n, &v| n.checked_mul(v))
        .filter(|&n| n > 0 && n <= isize::MAX as u64)
        .ok_or(SageVsaFp4Error::Extent)
}

#[derive(Debug, thiserror::Error, Clone, Copy, PartialEq, Eq)]
pub enum SageVsaFp4Error {
    #[error("VSA native artifact is unavailable")]
    Unavailable,
    #[error("VSA native boundary returned status {0}")]
    Native(i32),
    #[error("invalid VSA ABI argument")]
    InvalidArgument,
    #[error("unsupported VSA kernel geometry/profile")]
    UnsupportedGeometry,
    #[error("VSA extent exceeds native representation")]
    Extent,
    #[error("VSA plan metadata mismatch")]
    Plan,
    #[error("VSA byte span mismatch or alias")]
    Span,
    #[error("invalid VSA sparse list contents")]
    Lists,
}

const _: () = {
    assert!(size_of::<SageVsaFp4RequestV1>() == 64);
    assert!(offset_of!(SageVsaFp4RequestV1, maximum_dynamic_shared_bytes) == 56);
    assert!(size_of::<SageVsaFp4BufferSizesV1>() == 96);
    assert!(size_of::<SageVsaFp4PlanV1>() == 192);
    assert!(offset_of!(SageVsaFp4PlanV1, bytes) == 96);
    assert!(size_of::<SageVsaFp4BindingsV1>() == 192);
    assert!(size_of::<SageVsaFp4HostListsV1>() == 48);
};
