//! Allocation-free F16 XQA boundary. Device metadata and KV page ownership remain
//! the caller's responsibility; these host descriptors validate only byte ranges.
use std::mem::size_of;

pub const FLASHINFER_XQA_ABI: u32 = 1;
pub const FLASHINFER_XQA_OPERATOR: &str = "ferrum.cuda.flashinfer_xqa_decode";
pub const FLASHINFER_XQA_EXPORTS: [&str; 3] = [
    "ferrum_flashinfer_xqa_plan_v1",
    "ferrum_flashinfer_xqa_run_v1",
    "ferrum_flashinfer_xqa_validate_v1",
];
pub const FLASHINFER_XQA_SCRATCH_ALIGNMENT: u64 = 1024;
pub const FLASHINFER_XQA_UPSTREAM_REVISION: &str = "946200de1ae94fc93fdd0926f0a13afd1fa7f0f1";
pub const FLASHINFER_XQA_NUMERICAL_IDENTITY: &str = concat!(
    "flashinfer-946200de1ae94fc93fdd0926f0a13afd1fa7f0f1;",
    "sm120;F16-QKV-output;F32-sinks-from-F16;h64;gqa8;kvheads8;page32;",
    "independent-interleaved-KV-pages;scale=rsqrt(64);window=full-or128;",
    "vendor-tensorcore-softmax;no-fast-math;not-bitwise-scalar-equivalent"
);

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[repr(C)]
pub struct XqaRequestV1 {
    pub abi: u32,
    pub size: u32,
    pub batch: u32,
    pub max_pages_per_sequence: u32,
    /// Each row contains i32[page_count, position, 1, sequence_tokens] then u64 page addresses.
    pub binding_stride_bytes: u32,
    /// Zero means full attention; 128 includes the current token.
    pub sliding_window: u32,
    pub device_ordinal: u32,
    /// CUDA major*100 + minor*10. This artifact is qualified only for 1200.
    pub architecture: u32,
    pub multiprocessors: u32,
    pub reserved: u32,
    pub maximum_dynamic_shared_bytes: u64,
}
impl XqaRequestV1 {
    pub fn validate(&self) -> Result<(), XqaError> {
        if self.abi != FLASHINFER_XQA_ABI
            || self.size as usize != size_of::<Self>()
            || self.batch == 0
            || self.batch > 65535
            || self.max_pages_per_sequence == 0
            || self.max_pages_per_sequence > i32::MAX as u32 / 32
            || u64::from(self.binding_stride_bytes)
                < 16 + 8 * u64::from(self.max_pages_per_sequence)
            || self.binding_stride_bytes % 8 != 0
            || self.reserved != 0
            || self.multiprocessors == 0
            || self.multiprocessors > i32::MAX as u32
            || self.device_ordinal > i32::MAX as u32
            || self.maximum_dynamic_shared_bytes == 0
        {
            return Err(XqaError::InvalidArgument);
        }
        if self.architecture != 1200 || !matches!(self.sliding_window, 0 | 128) {
            return Err(XqaError::UnsupportedGeometry);
        }
        self.metadata_bytes()?;
        Ok(())
    }
    pub fn query_bytes(&self) -> u64 {
        u64::from(self.batch) * 64 * 64 * 2
    }
    pub fn metadata_bytes(&self) -> Result<u64, XqaError> {
        u64::from(self.batch)
            .checked_mul(u64::from(self.binding_stride_bytes))
            .filter(|n| *n <= isize::MAX as u64)
            .ok_or(XqaError::Extent)
    }
    pub fn subsequences(&self) -> u32 {
        (self.multiprocessors / (self.batch * 8))
            .max(1)
            .min(self.max_pages_per_sequence.div_ceil(8))
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[repr(C)]
pub struct XqaPlanV1 {
    pub request: XqaRequestV1,
    pub shared_bytes: u32,
    pub subsequences: u32,
    pub semaphore_bytes: u64,
    pub scratch_bytes: u64,
}
impl XqaPlanV1 {
    /// Canonical metadata. Native preparation additionally verifies the active
    /// device, compiled type sizes, and exact dynamic shared-memory requirement.
    pub fn checked(request: XqaRequestV1, shared_bytes: u32) -> Result<Self, XqaError> {
        request.validate()?;
        if shared_bytes == 0 || u64::from(shared_bytes) > request.maximum_dynamic_shared_bytes {
            return Err(XqaError::UnsupportedGeometry);
        }
        let subsequences = request.subsequences();
        let count = u64::from(request.batch) * 8 * u64::from(subsequences);
        let scratch_bytes = if subsequences > 1 { count * 1280 } else { 0 };
        // Upstream MemSegmenter uses uint32 offsets.
        if scratch_bytes > u64::from(u32::MAX) {
            return Err(XqaError::Extent);
        }
        Ok(Self {
            request,
            shared_bytes,
            subsequences,
            semaphore_bytes: u64::from(request.batch) * 32,
            scratch_bytes,
        })
    }
    pub fn validate(&self) -> Result<(), XqaError> {
        if *self != Self::checked(self.request, self.shared_bytes)? {
            return Err(XqaError::Plan);
        }
        Ok(())
    }
    pub fn fingerprint_material(&self) -> Vec<u8> {
        let mut result = Vec::with_capacity(size_of::<Self>());
        for value in [
            self.request.abi,
            self.request.size,
            self.request.batch,
            self.request.max_pages_per_sequence,
            self.request.binding_stride_bytes,
            self.request.sliding_window,
            self.request.device_ordinal,
            self.request.architecture,
            self.request.multiprocessors,
            self.request.reserved,
        ] {
            result.extend(value.to_le_bytes());
        }
        result.extend(self.request.maximum_dynamic_shared_bytes.to_le_bytes());
        result.extend(self.shared_bytes.to_le_bytes());
        result.extend(self.subsequences.to_le_bytes());
        result.extend(self.semaphore_bytes.to_le_bytes());
        result.extend(self.scratch_bytes.to_le_bytes());
        result
    }
}
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[repr(C)]
pub struct XqaSpan {
    pub address: u64,
    pub bytes: u64,
}
impl XqaSpan {
    fn validate(self, required: u64, alignment: u64) -> Result<(), XqaError> {
        if required == 0 {
            return if self.address == 0 && self.bytes == 0 {
                Ok(())
            } else {
                Err(XqaError::Span)
            };
        }
        if self.address == 0
            || self.address % alignment != 0
            || self.bytes != required
            || self.address.checked_add(self.bytes).is_none()
            || usize::try_from(self.address).is_err()
        {
            return Err(XqaError::Span);
        }
        Ok(())
    }
}
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[repr(C)]
pub struct XqaBindingsV1 {
    pub query: XqaSpan,
    pub sinks_f32: XqaSpan,
    pub packed_metadata: XqaSpan,
    pub output: XqaSpan,
    pub semaphores: XqaSpan,
    pub scratch: XqaSpan,
}
impl XqaBindingsV1 {
    pub fn validate(&self, plan: &XqaPlanV1) -> Result<(), XqaError> {
        plan.validate()?;
        self.query.validate(plan.request.query_bytes(), 16)?;
        self.sinks_f32.validate(256, 4)?;
        self.packed_metadata
            .validate(plan.request.metadata_bytes()?, 8)?;
        self.output.validate(plan.request.query_bytes(), 16)?;
        self.semaphores.validate(plan.semaphore_bytes, 4)?;
        self.scratch
            .validate(plan.scratch_bytes, FLASHINFER_XQA_SCRATCH_ALIGNMENT)?;
        let writes = [self.output, self.semaphores, self.scratch];
        for (i, write) in writes.into_iter().enumerate() {
            for read in [self.query, self.sinks_f32, self.packed_metadata] {
                disjoint(write, read)?;
            }
            for other in &writes[i + 1..] {
                disjoint(write, *other)?;
            }
        }
        Ok(())
    }
}
fn disjoint(a: XqaSpan, b: XqaSpan) -> Result<(), XqaError> {
    if a.bytes > 0
        && b.bytes > 0
        && a.address < b.address + b.bytes
        && b.address < a.address + a.bytes
    {
        return Err(XqaError::Span);
    }
    Ok(())
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum XqaError {
    #[error("invalid XQA boundary argument")]
    InvalidArgument,
    #[error("unsupported XQA geometry/device")]
    UnsupportedGeometry,
    #[error("XQA byte extent overflows")]
    Extent,
    #[error("XQA span has wrong extent/alignment or aliases a write")]
    Span,
    #[error("XQA prepared plan differs from its derivation")]
    Plan,
    #[error("locked XQA native artifact is not compiled")]
    Unavailable,
    #[error("XQA native boundary returned status {0}")]
    Native(i32),
}
const _: () = {
    assert!(size_of::<XqaRequestV1>() == 48);
    assert!(size_of::<XqaPlanV1>() == 72);
    assert!(size_of::<XqaBindingsV1>() == 96);
};
#[cfg(test)]
mod tests;
