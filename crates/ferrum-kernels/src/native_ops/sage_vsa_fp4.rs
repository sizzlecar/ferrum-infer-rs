//! Native span adapter. Calling the unsafe ABI supplies no admission authority.
pub use ferrum_native_ops::sage_vsa_fp4::{
    SageVsaFp4BindingsV1 as Bindings, SageVsaFp4Device as Device, SageVsaFp4Error as Error,
    SageVsaFp4HostListsV1 as HostLists, SageVsaFp4ListShape as ListShape, SageVsaFp4PlanV1 as Plan,
    SageVsaFp4RequestV1 as Request, SageVsaFp4SpanV1 as DeviceSpan, SAGE_VSA_FP4_EXPORTS,
    SAGE_VSA_FP4_NUMERICAL_IDENTITY, SAGE_VSA_FP4_OPERATOR,
};
use std::ffi::c_void;
mod host_pattern;
pub use host_pattern::{CheckedHostVsaPattern, CHECKED_HOST_PATTERN_SOURCE};
pub mod quantize;
pub mod regions;
pub use regions::CheckedVsaRegions;
#[cfg(any(feature = "cuda", test))]
mod layout;
#[cfg(feature = "cuda")]
pub(crate) use layout::WorkspaceLayout;
mod catalog;
pub use catalog::{
    provider_catalog_declaration, source_declaration_fingerprint, SageVsaFp4Config, CAPABILITY_ID,
    PROVIDER_ID, QUANTIZE_CAPABILITY_ID, QUANTIZE_PROVIDER_ID, REGION_CAPABILITY_ID,
    REGION_PROVIDER_ID,
};

#[derive(Clone, Debug)]
pub struct PreparedSageVsaFp4 {
    plan: Plan,
}
impl PreparedSageVsaFp4 {
    pub fn new(request: Request) -> Result<Self, Error> {
        request.validate()?;
        if !compiled() {
            return Err(Error::Unavailable);
        }
        #[cfg(feature = "cuda-sage-vsa-fp4")]
        {
            let mut plan = Plan::default();
            // SAFETY: host ABI structs; no enqueue/payload dereference.
            status(unsafe { ferrum_sage_vsa_fp4_plan_v1(&request, &mut plan) })?;
            plan.validate()?;
            if plan.request != request {
                return Err(Error::Plan);
            }
            Ok(Self { plan })
        }
        #[cfg(not(feature = "cuda-sage-vsa-fp4"))]
        {
            Err(Error::Unavailable)
        }
    }
    pub fn plan(&self) -> &Plan {
        &self.plan
    }
    /// # Safety
    /// Host spans are immutable readable allocations for their full extents.
    /// The caller retains their original charged owners throughout this call.
    /// Device spans belong to the prepared current context. No enqueue occurs.
    pub unsafe fn validate(&self, bindings: &Bindings, host: &HostLists) -> Result<(), Error> {
        bindings.validate(&self.plan)?;
        #[cfg(feature = "cuda-sage-vsa-fp4")]
        {
            status(unsafe { ferrum_sage_vsa_fp4_validate_v1(&self.plan, bindings, host) })
        }
        #[cfg(not(feature = "cuda-sage-vsa-fp4"))]
        {
            let _ = host;
            Err(Error::Unavailable)
        }
    }
    /// # Safety
    /// All spans are admitted live storage. The Host inputs are pinned and
    /// immutable, device inputs ready on the supplied stream, writes exclusive.
    /// Retain all Host/device/module owners until completion or quarantine,
    /// including after any error, which may follow one or more submitted copies.
    pub unsafe fn run(
        &self,
        bindings: &Bindings,
        host: &HostLists,
        stream: *mut c_void,
    ) -> Result<(), Error> {
        bindings.validate(&self.plan)?;
        if stream.is_null() || matches!(stream as usize, 1 | 2) {
            return Err(Error::InvalidArgument);
        }
        #[cfg(feature = "cuda-sage-vsa-fp4")]
        {
            status(unsafe { ferrum_sage_vsa_fp4_run_v1(&self.plan, bindings, host, stream) })
        }
        #[cfg(not(feature = "cuda-sage-vsa-fp4"))]
        {
            let _ = host;
            Err(Error::Unavailable)
        }
    }
}
pub fn compiled() -> bool {
    cfg!(feature = "cuda-sage-vsa-fp4")
        && super::compiled_native_operator_artifacts().iter().any(|a| {
            a.operator == SAGE_VSA_FP4_OPERATOR
                && a.backend == ferrum_types::NativeOperatorBackend::Cuda
                && a.linkage == ferrum_types::NativeOperatorLinkage::Static
                && a.operator_abi_version == "1"
                && SAGE_VSA_FP4_EXPORTS
                    .iter()
                    .all(|name| a.exports.iter().any(|s| s == name))
        })
}
#[cfg(feature = "cuda-sage-vsa-fp4")]
fn status(value: i32) -> Result<(), Error> {
    if value == 0 {
        Ok(())
    } else {
        Err(Error::Native(value))
    }
}
#[cfg(feature = "cuda-sage-vsa-fp4")]
unsafe extern "C" {
    fn ferrum_sage_vsa_fp4_plan_v1(request: *const Request, plan: *mut Plan) -> i32;
    fn ferrum_sage_vsa_fp4_validate_v1(
        plan: *const Plan,
        bindings: *const Bindings,
        host: *const HostLists,
    ) -> i32;
    fn ferrum_sage_vsa_fp4_run_v1(
        plan: *const Plan,
        bindings: *const Bindings,
        host: *const HostLists,
        stream: *mut c_void,
    ) -> i32;
}
