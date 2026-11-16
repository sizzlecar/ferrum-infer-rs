//! Immutable charged geometry for GPU-produced sparse lists.
#[cfg(any(feature = "cuda", test))]
use super::DeviceSpan;
use super::{host_pattern::integers, Error, ListShape, Request};
pub use ferrum_native_ops::sage_vsa_fp4::selection::{
    CheckedRegionSelection, RegionBindingsV1, RegionHostV1, RegionKernelPlanV1, RegionPlanV1,
    RegionRequestV1, REGION_KERNEL_EXPORTS, REGION_SELECTION_EXPORTS,
    REGION_SELECTION_NUMERICAL_IDENTITY, REGION_SELECTION_PROFILE,
};
use ferrum_runtime::{HostMemoryClass, ManagedHostBytes};
use sha2::{Digest, Sha256};
use std::ffi::c_void;
mod kernel_upload;
pub use kernel_upload::{kernel_upload_compiled, PreparedSageVsaFp4KernelRegions};
#[cfg(test)]
mod tests;

pub struct CheckedVsaRegions {
    shape: ListShape,
    payloads: [ManagedHostBytes; 2],
    digest: String,
    selected_tiles: u64,
}
impl CheckedVsaRegions {
    /// The caller's original immutable TransferStaging owners include every
    /// physical padding byte. This constructor allocates no payload storage.
    pub fn new(
        shape: ListShape,
        regions: ManagedHostBytes,
        half_valid: ManagedHostBytes,
    ) -> Result<Self, String> {
        let payloads = [regions, half_valid];
        if payloads
            .iter()
            .any(|p| p.class() != HostMemoryClass::TransferStaging)
        {
            return Err("VSA regions require charged immutable TransferStaging owners".into());
        }
        let checked =
            CheckedRegionSelection::new(shape, integers(&payloads[0])?, integers(&payloads[1])?)
                .map_err(|e| e.to_string())?;
        let selected_tiles = u64::from(shape.batches())
            * u64::from(shape.heads())
            * u64::from(shape.query_blocks())
            * u64::from(checked.selected_per_row());
        let mut hash = Sha256::new();
        hash.update(REGION_SELECTION_NUMERICAL_IDENTITY.as_bytes());
        for value in [
            shape.batches(),
            shape.heads(),
            shape.query_blocks(),
            shape.key_blocks(),
        ] {
            hash.update(value.to_le_bytes());
        }
        for payload in &payloads {
            hash.update((payload.len() as u64).to_le_bytes());
            hash.update(payload.bytes());
        }
        Ok(Self {
            shape,
            payloads,
            digest: format!("{:x}", hash.finalize()),
            selected_tiles,
        })
    }
    pub const fn shape(&self) -> ListShape {
        self.shape
    }
    pub fn digest(&self) -> &str {
        &self.digest
    }
    pub const fn selected_tiles(&self) -> u64 {
        self.selected_tiles
    }
    pub fn checked(&self) -> Result<CheckedRegionSelection<'_>, String> {
        CheckedRegionSelection::new(
            self.shape,
            integers(&self.payloads[0])?,
            integers(&self.payloads[1])?,
        )
        .map_err(|e| e.to_string())
    }
    #[cfg(any(feature = "cuda", test))]
    pub(crate) fn payloads(&self) -> &[ManagedHostBytes; 2] {
        &self.payloads
    }
}

#[derive(Clone, Debug)]
pub struct PreparedSageVsaFp4Regions {
    plan: RegionPlanV1,
}
impl PreparedSageVsaFp4Regions {
    pub fn new(attention: Request, region_count: u32) -> Result<Self, Error> {
        let request = RegionRequestV1 {
            attention,
            region_count,
            selection_profile: REGION_SELECTION_PROFILE,
        };
        request.validate()?;
        if !compiled() {
            return Err(Error::Unavailable);
        }
        #[cfg(feature = "cuda-sage-vsa-fp4")]
        {
            let mut plan = RegionPlanV1::default();
            // SAFETY: Host ABI structs only, no enqueue or payload access.
            super::status(unsafe { ferrum_sage_vsa_fp4_region_plan_v1(&request, &mut plan) })?;
            plan.validate()?;
            if plan.request() != request {
                return Err(Error::Plan);
            }
            Ok(Self { plan })
        }
        #[cfg(not(feature = "cuda-sage-vsa-fp4"))]
        {
            Err(Error::Unavailable)
        }
    }
    pub fn plan(&self) -> &RegionPlanV1 {
        &self.plan
    }
    /// # Safety
    /// Exact Host extents are pinned, readable, immutable and retained. Device
    /// extents belong to this context; this call performs no enqueue.
    pub unsafe fn validate(
        &self,
        bindings: &RegionBindingsV1,
        host: &RegionHostV1,
    ) -> Result<(), Error> {
        self.plan.validate()?;
        bindings.attention.validate(&self.plan.attention)?;
        #[cfg(feature = "cuda-sage-vsa-fp4")]
        {
            super::status(unsafe {
                ferrum_sage_vsa_fp4_region_validate_v1(&self.plan, bindings, host)
            })
        }
        #[cfg(not(feature = "cuda-sage-vsa-fp4"))]
        {
            let _ = host;
            Err(Error::Unavailable)
        }
    }
    /// # Safety
    /// Caller admits every span and retains Host/device/module owners to device
    /// completion/quarantine, including errors after the first geometry copy.
    /// Scores and packed QKV are ready on the given explicit stream. All writes
    /// are exclusive. Lists are produced internally and never supplied inputs.
    pub unsafe fn run(
        &self,
        bindings: &RegionBindingsV1,
        host: &RegionHostV1,
        stream: *mut c_void,
    ) -> Result<(), Error> {
        self.plan.validate()?;
        bindings.attention.validate(&self.plan.attention)?;
        if stream.is_null() || matches!(stream as usize, 1 | 2) {
            return Err(Error::InvalidArgument);
        }
        #[cfg(feature = "cuda-sage-vsa-fp4")]
        {
            super::status(unsafe {
                ferrum_sage_vsa_fp4_region_run_v1(&self.plan, bindings, host, stream)
            })
        }
        #[cfg(not(feature = "cuda-sage-vsa-fp4"))]
        {
            let _ = host;
            Err(Error::Unavailable)
        }
    }
}
pub fn compiled() -> bool {
    super::compiled()
        && super::super::compiled_native_operator_artifacts()
            .iter()
            .any(|a| {
                a.operator == super::SAGE_VSA_FP4_OPERATOR
                    && REGION_SELECTION_EXPORTS
                        .iter()
                        .all(|name| a.exports.iter().any(|s| s == name))
            })
}
#[cfg(feature = "cuda-sage-vsa-fp4")]
unsafe extern "C" {
    fn ferrum_sage_vsa_fp4_region_plan_v1(
        request: *const RegionRequestV1,
        plan: *mut RegionPlanV1,
    ) -> i32;
    fn ferrum_sage_vsa_fp4_region_validate_v1(
        plan: *const RegionPlanV1,
        bindings: *const RegionBindingsV1,
        host: *const RegionHostV1,
    ) -> i32;
    fn ferrum_sage_vsa_fp4_region_run_v1(
        plan: *const RegionPlanV1,
        bindings: *const RegionBindingsV1,
        host: *const RegionHostV1,
        stream: *mut c_void,
    ) -> i32;
}

#[cfg(any(feature = "cuda", test))]
#[derive(Clone, Copy, Debug)]
pub(crate) struct RegionWorkspaceLayout {
    attention: super::layout::WorkspaceLayout,
    region_bytes: u64,
    bytes: u64,
}
#[cfg(any(feature = "cuda", test))]
impl RegionWorkspaceLayout {
    pub(crate) fn new(plan: &RegionPlanV1) -> Result<Self, Error> {
        plan.validate()?;
        let attention = super::layout::WorkspaceLayout::new(&plan.attention)?;
        let bytes = attention
            .bytes()
            .checked_add(plan.regions_bytes)
            .and_then(|n| n.checked_add(15))
            .map(|n| n & !15)
            .filter(|&n| n <= isize::MAX as u64)
            .ok_or(Error::Extent)?;
        Ok(Self {
            attention,
            region_bytes: plan.regions_bytes,
            bytes,
        })
    }
    pub(crate) const fn bytes(self) -> u64 {
        self.bytes
    }
    pub(crate) fn spans(self, span: DeviceSpan) -> Result<([DeviceSpan; 5], DeviceSpan), Error> {
        if span.address == 0
            || span.address % 16 != 0
            || span.bytes != self.bytes
            || span.address.checked_add(span.bytes).is_none()
        {
            return Err(Error::Span);
        }
        let attention = self.attention.spans(DeviceSpan {
            bytes: self.attention.bytes(),
            ..span
        })?;
        Ok((
            attention,
            DeviceSpan {
                address: span.address + self.attention.bytes(),
                bytes: self.region_bytes,
            },
        ))
    }
}
