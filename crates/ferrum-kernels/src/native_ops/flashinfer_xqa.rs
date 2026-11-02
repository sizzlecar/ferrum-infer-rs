//! Allocation-free adapter for an optional, locked SM120 XQA artifact.
//! The caller owns every span, KV page and CUDA module through completion.
pub use ferrum_native_ops::flashinfer_xqa::{
    XqaBindingsV1 as Bindings, XqaError as Error, XqaPlanV1 as Plan, XqaRequestV1 as Request,
    XqaSpan as DeviceSpan, FLASHINFER_XQA_ABI, FLASHINFER_XQA_NUMERICAL_IDENTITY,
    FLASHINFER_XQA_OPERATOR, FLASHINFER_XQA_SCRATCH_ALIGNMENT, FLASHINFER_XQA_UPSTREAM_REVISION,
};
use std::ffi::c_void;

mod catalog;
pub use catalog::{
    provider_catalog_declaration, provider_source_fingerprint, GPT_OSS_ATTENTION_PROVIDER_ID,
};

#[derive(Debug, Clone)]
pub struct PreparedXqa {
    plan: Plan,
}
impl PreparedXqa {
    /// Preparation requires the existing caller context to be current. It sets
    /// the native kernel's shared-memory attribute; call before stream capture.
    /// It never creates a context or allocates device payload storage.
    pub fn new(request: Request) -> Result<Self, Error> {
        request.validate()?;
        if !compiled() {
            return Err(Error::Unavailable);
        }
        #[cfg(ferrum_native_flashinfer_xqa)]
        {
            let mut plan = Plan::default();
            // SAFETY: ABI-identical live host structures.
            status(unsafe { ferrum_flashinfer_xqa_plan_v1(&request, &mut plan) })?;
            plan.validate()?;
            if plan.request != request {
                return Err(Error::Plan);
            }
            Ok(Self { plan })
        }
        #[cfg(not(ferrum_native_flashinfer_xqa))]
        {
            Err(Error::Unavailable)
        }
    }
    pub fn plan(&self) -> &Plan {
        &self.plan
    }
    pub fn fingerprint_material(&self) -> Vec<u8> {
        self.plan.fingerprint_material()
    }
    /// Structural checks only: no device reads, context switch, or enqueue.
    pub fn validate(&self, bindings: &Bindings) -> Result<(), Error> {
        bindings.validate(&self.plan)?;
        #[cfg(ferrum_native_flashinfer_xqa)]
        {
            // SAFETY: only host descriptors are accessed.
            status(unsafe { ferrum_flashinfer_xqa_validate_v1(&self.plan, bindings) })
        }
        #[cfg(not(ferrum_native_flashinfer_xqa))]
        {
            Err(Error::Unavailable)
        }
    }
    /// Enqueue semaphore reset and XQA on the explicit stream. Capture-safe;
    /// no allocation, synchronization, fallback, or implicit default stream.
    ///
    /// # Safety
    /// Caller retains all spans and every referenced page on the prepared
    /// context until stream completion (also after partial enqueue failures).
    /// Each metadata row is i32[page_count,position,1,sequence_tokens] followed
    /// by page_count aligned u64 addresses, with 0<=position<page_count*32,
    /// page_count<=max_pages_per_sequence, and sequence_tokens=position+1.
    /// Each page owns 64KiB containing F16[token,kind,8,64]. Every referenced
    /// page is live and disjoint from output/scratch/semaphores. Scratch and
    /// semaphores are exclusive to this invocation until completion. Sinks
    /// are 64 finite F32 values converted from the existing rounded F16 weights.
    /// Nonzero status can follow a queued memset or kernel; it is not proof
    /// that nothing was submitted.
    pub unsafe fn run(&self, bindings: &Bindings, stream: *mut c_void) -> Result<(), Error> {
        bindings.validate(&self.plan)?;
        if stream.is_null() {
            return Err(Error::InvalidArgument);
        }
        #[cfg(ferrum_native_flashinfer_xqa)]
        {
            // SAFETY: caller establishes device ownership and stream lifetime.
            status(unsafe { ferrum_flashinfer_xqa_run_v1(&self.plan, bindings, stream) })
        }
        #[cfg(not(ferrum_native_flashinfer_xqa))]
        {
            Err(Error::Unavailable)
        }
    }
}
pub fn compiled() -> bool {
    cfg!(ferrum_native_flashinfer_xqa)
        && super::compiled_native_operator_artifacts().iter().any(|a| {
            a.operator == FLASHINFER_XQA_OPERATOR
                && a.operator_abi_version == "1"
                && a.backend == ferrum_types::NativeOperatorBackend::Cuda
                && a.linkage == ferrum_types::NativeOperatorLinkage::Static
                && ferrum_native_ops::flashinfer_xqa::FLASHINFER_XQA_EXPORTS
                    .iter()
                    .all(|name| a.exports.iter().any(|export| export == name))
        })
}
pub fn source_declaration_fingerprint() -> String {
    use sha2::{Digest, Sha256};
    let mut digest = Sha256::new();
    for bytes in [
        FLASHINFER_XQA_NUMERICAL_IDENTITY.as_bytes(),
        include_bytes!("flashinfer_xqa.rs"),
        include_bytes!("../../native/flashinfer-xqa/abi.h"),
        include_bytes!("../../native/flashinfer-xqa/boundary.cu"),
        include_bytes!("../../native/flashinfer-xqa/flashinfer-xqa.seed.json"),
        include_bytes!("../../native/flashinfer-xqa/source-overlay.json"),
        include_bytes!("../../native/flashinfer-xqa/overrides/mha.cu"),
        include_bytes!("../../native/flashinfer-xqa/overrides/mha.h"),
        include_bytes!("../../native/flashinfer-xqa/overrides/mhaUtils.cuh"),
        include_bytes!("../../native/flashinfer-xqa/overrides/hostUtils.h"),
    ] {
        digest.update((bytes.len() as u64).to_le_bytes());
        digest.update(bytes);
    }
    format!("{:x}", digest.finalize())
}
#[cfg(ferrum_native_flashinfer_xqa)]
fn status(value: i32) -> Result<(), Error> {
    match value {
        0 => Ok(()),
        1 => Err(Error::InvalidArgument),
        2 => Err(Error::UnsupportedGeometry),
        3 => Err(Error::Extent),
        4 => Err(Error::Span),
        5 => Err(Error::Plan),
        n => Err(Error::Native(n)),
    }
}
#[cfg(ferrum_native_flashinfer_xqa)]
unsafe extern "C" {
    fn ferrum_flashinfer_xqa_plan_v1(request: *const Request, plan: *mut Plan) -> i32;
    fn ferrum_flashinfer_xqa_validate_v1(plan: *const Plan, bindings: *const Bindings) -> i32;
    fn ferrum_flashinfer_xqa_run_v1(
        plan: *const Plan,
        bindings: *const Bindings,
        stream: *mut c_void,
    ) -> i32;
}
