//! Allocation-free adapter. Resource admission and readiness belong to the
//! execution owner; this unsafe boundary neither allocates nor retains them.
pub use ferrum_native_ops::cutlass_nvfp4::{
    CutlassNvfp4BindingsV1 as Bindings, CutlassNvfp4Device as Device, CutlassNvfp4Error as Error,
    CutlassNvfp4PlanV1 as Plan, CutlassNvfp4RequestV1 as Request, CutlassNvfp4Span as DeviceSpan,
    CUTLASS_NVFP4_ALIGNMENT, CUTLASS_NVFP4_NUMERICAL_IDENTITY, CUTLASS_NVFP4_OPERATOR,
};
use std::ffi::c_void;
mod catalog;
mod quantize;
pub use catalog::{
    provider_catalog_declaration, source_declaration_fingerprint, CutlassNvfp4Config,
    CAPABILITY_ID, PROVIDER_ID, QUANTIZE_CAPABILITY_ID, QUANTIZE_PROVIDER_ID,
};
pub use quantize::*;

#[derive(Debug, Clone)]
pub struct PreparedCutlassNvfp4 {
    plan: Plan,
}
impl PreparedCutlassNvfp4 {
    /// Query with the caller's context already bound. Native preparation reads
    /// device attributes only; it neither enqueues nor selects a context.
    pub fn new(request: Request) -> Result<Self, Error> {
        request.validate()?;
        if !compiled() {
            return Err(Error::Unavailable);
        }
        #[cfg(feature = "cuda-cutlass-nvfp4")]
        {
            let mut plan = Plan::default();
            // SAFETY: exact host ABI structures, no device payload access.
            status(unsafe { ferrum_cutlass_nvfp4_plan_v1(&request, &mut plan) })?;
            plan.validate()?;
            if plan.request != request {
                return Err(Error::Plan);
            }
            Ok(Self { plan })
        }
        #[cfg(not(feature = "cuda-cutlass-nvfp4"))]
        {
            Err(Error::Unavailable)
        }
    }
    pub fn plan(&self) -> &Plan {
        &self.plan
    }
    pub fn workspace_bytes(&self) -> u64 {
        self.plan.workspace_bytes
    }
    pub fn alignment(&self) -> u64 {
        CUTLASS_NVFP4_ALIGNMENT
    }
    pub fn fingerprint_material(&self) -> [u8; 64] {
        self.plan.fingerprint_material()
    }
    /// Structural/native can-implement checks, no payload read or enqueue.
    pub fn validate(&self, bindings: &Bindings) -> Result<(), Error> {
        bindings.validate(&self.plan)?;
        #[cfg(feature = "cuda-cutlass-nvfp4")]
        {
            status(unsafe { ferrum_cutlass_nvfp4_validate_v1(&self.plan, bindings) })
        }
        #[cfg(not(feature = "cuda-cutlass-nvfp4"))]
        {
            Err(Error::Unavailable)
        }
    }
    /// Enqueue one packed matrix multiplication, without activation conversion.
    ///
    /// # Safety
    /// All spans must be admitted live device storage on the prepared context.
    /// Inputs/scales/alpha must satisfy the operation value domain and already
    /// be ready on this explicit stream. The caller retains every read, write,
    /// workspace and admitted module/opaque capacity through completion, also
    /// after any error (which may follow enqueue). Workspace is exclusive until
    /// completion. Native module capacity lives until context destruction.
    /// This API supplies no resource, stream, event or submission authority.
    pub unsafe fn run(&self, bindings: &Bindings, stream: *mut c_void) -> Result<(), Error> {
        bindings.validate(&self.plan)?;
        if stream.is_null() || matches!(stream as usize, 1 | 2) {
            return Err(Error::InvalidArgument);
        }
        #[cfg(feature = "cuda-cutlass-nvfp4")]
        {
            status(unsafe { ferrum_cutlass_nvfp4_run_v1(&self.plan, bindings, stream) })
        }
        #[cfg(not(feature = "cuda-cutlass-nvfp4"))]
        {
            Err(Error::Unavailable)
        }
    }
}
pub fn compiled() -> bool {
    use ferrum_native_ops::cutlass_nvfp4::CUTLASS_NVFP4_EXPORTS;
    cfg!(feature = "cuda-cutlass-nvfp4")
        && super::compiled_native_operator_artifacts().iter().any(|a| {
            a.operator == CUTLASS_NVFP4_OPERATOR
                && a.backend == ferrum_types::NativeOperatorBackend::Cuda
                && a.linkage == ferrum_types::NativeOperatorLinkage::Static
                && a.operator_abi_version == "1"
                && CUTLASS_NVFP4_EXPORTS
                    .iter()
                    .all(|name| a.exports.iter().any(|actual| actual == name))
        })
}
#[cfg(feature = "cuda-cutlass-nvfp4")]
fn status(value: i32) -> Result<(), Error> {
    if value == 0 {
        Ok(())
    } else {
        Err(Error::Native(value))
    }
}
#[cfg(feature = "cuda-cutlass-nvfp4")]
unsafe extern "C" {
    fn ferrum_cutlass_nvfp4_plan_v1(request: *const Request, plan: *mut Plan) -> i32;
    fn ferrum_cutlass_nvfp4_validate_v1(plan: *const Plan, bindings: *const Bindings) -> i32;
    fn ferrum_cutlass_nvfp4_run_v1(
        plan: *const Plan,
        bindings: *const Bindings,
        stream: *mut c_void,
    ) -> i32;
}
