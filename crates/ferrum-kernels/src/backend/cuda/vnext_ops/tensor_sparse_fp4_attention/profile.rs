//! Two explicit metadata producers, sharing one Core preparation/retirement path.
use super::*;
use crate::native_ops::sage_vsa_fp4::{
    regions::{
        PreparedSageVsaFp4KernelRegions, RegionBindingsV1, RegionWorkspaceLayout,
        REGION_SELECTION_NUMERICAL_IDENTITY,
    },
    CheckedVsaRegions,
};
use ferrum_runtime::{
    sparse_fp4_attention_region_topk_contract, ManagedHostBytes, StandardOperationContract,
    SPARSE_FP4_ATTENTION_REGION_TOPK_CAPABILITY_ID, SPARSE_FP4_ATTENTION_REGION_TOPK_OPERATION_ID,
};

#[derive(Clone)]
pub(super) enum Pattern {
    Host(Arc<CheckedHostVsaPattern>),
    Regions(Arc<CheckedVsaRegions>),
}
impl Pattern {
    pub(super) fn shape(&self) -> sage_vsa_fp4::ListShape {
        match self {
            Self::Host(p) => p.shape(),
            Self::Regions(p) => p.shape(),
        }
    }
    pub(super) fn digest(&self) -> &str {
        match self {
            Self::Host(p) => p.digest(),
            Self::Regions(p) => p.digest(),
        }
    }
    pub(super) fn selected_tiles(&self) -> u64 {
        match self {
            Self::Host(p) => p.selected_tiles(),
            Self::Regions(p) => p.selected_tiles(),
        }
    }
    pub(super) fn payloads(&self) -> &[ManagedHostBytes] {
        match self {
            Self::Host(p) => p.payloads(),
            Self::Regions(p) => p.payloads(),
        }
    }
    pub(super) fn same_owner(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Host(a), Self::Host(b)) => Arc::ptr_eq(a, b),
            (Self::Regions(a), Self::Regions(b)) => Arc::ptr_eq(a, b),
            _ => false,
        }
    }
    pub(super) fn inputs(&self) -> usize {
        if matches!(self, Self::Host(_)) {
            6
        } else {
            7
        }
    }
    pub(super) fn operation(&self) -> &'static str {
        match self {
            Self::Host(_) => SPARSE_FP4_ATTENTION_CHECKED_HOST_OPERATION_ID,
            Self::Regions(_) => SPARSE_FP4_ATTENTION_REGION_TOPK_OPERATION_ID,
        }
    }
    pub(super) fn capability(&self) -> &'static str {
        match self {
            Self::Host(_) => SPARSE_FP4_ATTENTION_CHECKED_HOST_CAPABILITY_ID,
            Self::Regions(_) => SPARSE_FP4_ATTENTION_REGION_TOPK_CAPABILITY_ID,
        }
    }
    pub(super) fn native_capability(&self) -> &'static str {
        match self {
            Self::Host(_) => sage_vsa_fp4::CAPABILITY_ID,
            Self::Regions(_) => sage_vsa_fp4::REGION_CAPABILITY_ID,
        }
    }
    pub(super) fn provider(&self) -> &'static str {
        match self {
            Self::Host(_) => sage_vsa_fp4::PROVIDER_ID,
            Self::Regions(_) => sage_vsa_fp4::REGION_PROVIDER_ID,
        }
    }
    pub(super) fn numerical_identity(&self) -> &'static str {
        match self {
            Self::Host(_) => sage_vsa_fp4::CHECKED_HOST_PATTERN_SOURCE,
            Self::Regions(_) => REGION_SELECTION_NUMERICAL_IDENTITY,
        }
    }
    pub(super) fn contract(&self) -> Result<StandardOperationContract, VNextError> {
        match self {
            Self::Host(_) => sparse_fp4_attention_checked_host_contract(),
            Self::Regions(_) => sparse_fp4_attention_region_topk_contract(),
        }
    }
}

pub(super) enum Execution {
    Host {
        native: PreparedSageVsaFp4,
        workspace: WorkspaceLayout,
    },
    Regions {
        native: PreparedSageVsaFp4KernelRegions,
        workspace: RegionWorkspaceLayout,
    },
}
impl Execution {
    pub(super) fn prepare(pattern: &Pattern, request: Request) -> Result<Self, String> {
        match pattern {
            Pattern::Host(p) => {
                p.checked_lists()?;
                let native = PreparedSageVsaFp4::new(request).map_err(|e| e.to_string())?;
                let workspace = WorkspaceLayout::new(native.plan()).map_err(|e| e.to_string())?;
                Ok(Self::Host { native, workspace })
            }
            Pattern::Regions(p) => {
                let count = p.checked()?.regions().len() / 3;
                let native = PreparedSageVsaFp4KernelRegions::new(request, count as u32)
                    .map_err(|e| e.to_string())?;
                let workspace = RegionWorkspaceLayout::new(&native.plan().geometry)
                    .map_err(|e| e.to_string())?;
                Ok(Self::Regions { native, workspace })
            }
        }
    }
    pub(super) fn bytes(&self) -> u64 {
        match self {
            Self::Host { workspace, .. } => workspace.bytes(),
            Self::Regions { workspace, .. } => workspace.bytes(),
        }
    }
    pub(super) fn fingerprint(&self) -> String {
        match self {
            Self::Host { native, .. } => {
                implementation_fingerprint(&[&native.plan().fingerprint_material()])
            }
            Self::Regions { native, .. } => {
                implementation_fingerprint(&[&native.plan().fingerprint_material()])
            }
        }
    }
    pub(super) fn dispatch_counts(&self) -> (u64, u64) {
        match self {
            Self::Host { native, .. } => (
                1,
                u64::from(native.plan().upload_count) + u64::from(native.plan().memset_count),
            ),
            Self::Regions { native, .. } => native.plan().dispatch_counts(),
        }
    }
    fn spans(
        &self,
        regions: &[CudaBufferRegion],
    ) -> Result<(Bindings, Option<RegionBindingsV1>), String> {
        let inputs = match self {
            Self::Host { .. } => 6,
            Self::Regions { .. } => 7,
        };
        if regions.len() != inputs + 2 {
            return Err("VSA operand/scratch extent mismatch".into());
        }
        let span = |i: usize| DeviceSpan {
            address: regions[i].device_ptr(),
            bytes: regions[i].length_bytes(),
        };
        let (writes, region_span) = match self {
            Self::Host { workspace, .. } => (
                workspace
                    .spans(span(inputs + 1))
                    .map_err(|e| e.to_string())?,
                None,
            ),
            Self::Regions { workspace, .. } => {
                let (a, r) = workspace
                    .spans(span(inputs + 1))
                    .map_err(|e| e.to_string())?;
                (a, Some(r))
            }
        };
        let [zero_delta, indices, counts, half_valid, lse] = writes;
        let attention = Bindings {
            query: span(0),
            key: span(1),
            value: span(2),
            query_scales: span(3),
            key_scales: span(4),
            value_scales: span(5),
            output: span(inputs),
            lse,
            zero_delta,
            indices,
            counts,
            half_valid,
        };
        Ok((
            attention,
            region_span.map(|r| RegionBindingsV1 {
                attention,
                scores: span(6),
                regions: r,
            }),
        ))
    }
    pub(super) unsafe fn validate(
        &self,
        regions: &[CudaBufferRegion],
        pattern: &Pattern,
    ) -> Result<(), String> {
        let (attention, region) = self.spans(regions)?;
        match (self, pattern) {
            (Self::Host { native, .. }, Pattern::Host(p)) => {
                unsafe { native.validate(&attention, &p.checked_lists()?.host_addresses()) }
                    .map_err(|e| e.to_string())
            }
            (Self::Regions { native, .. }, Pattern::Regions(p)) => unsafe {
                native.validate(
                    &region.ok_or("VSA region bindings missing")?,
                    &p.checked()?.host_addresses(),
                )
            }
            .map_err(|e| e.to_string()),
            _ => Err("VSA prepared producer profile differs".into()),
        }
    }
    pub(super) unsafe fn run(
        &self,
        regions: &[CudaBufferRegion],
        pattern: &Pattern,
        stream: *mut std::ffi::c_void,
    ) -> Result<(), String> {
        let (attention, region) = self.spans(regions)?;
        match (self, pattern) {
            (Self::Host { native, .. }, Pattern::Host(p)) => {
                unsafe { native.run(&attention, &p.checked_lists()?.host_addresses(), stream) }
                    .map_err(|e| e.to_string())
            }
            (Self::Regions { native, .. }, Pattern::Regions(p)) => unsafe {
                native.run(
                    &region.ok_or("VSA region bindings missing")?,
                    &p.checked()?.host_addresses(),
                    stream,
                )
            }
            .map_err(|e| e.to_string()),
            _ => Err("VSA prepared producer profile differs".into()),
        }
    }
}
