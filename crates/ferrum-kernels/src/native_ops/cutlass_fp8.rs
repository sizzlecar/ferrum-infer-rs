//! Static-link, allocation-free adapter for the locked SM120 FP8 artifact.
//! Native modules may remain loaded until context destruction. The caller must
//! keep its separately admitted native opaque owner for that full lifetime.

pub use ferrum_native_ops::cutlass_fp8::{
    CutlassFp8BindingsV1 as Bindings, CutlassFp8Device as Device, CutlassFp8Error as Error,
    CutlassFp8PlanV1 as Plan, CutlassFp8RequestV1 as Request, CutlassFp8Span as DeviceSpan,
    CUTLASS_FP8_NUMERICAL_IDENTITY, CUTLASS_FP8_OPERATOR, CUTLASS_FP8_SCRATCH_ALIGNMENT,
};
use std::ffi::c_void;

mod catalog;
pub use catalog::{
    provider_catalog_declaration, source_declaration_fingerprint, CutlassFp8CatalogConfig,
    CUTLASS_FP8_PROVIDER_ID,
};

#[derive(Debug, Clone)]
pub struct PreparedCutlassFp8 {
    plan: Plan,
}
impl PreparedCutlassFp8 {
    /// Query the actual locked kernel with the caller's existing context bound.
    /// Does not select/create a CUDA context, allocate payloads or enqueue work.
    pub fn new(request: Request) -> Result<Self, Error> {
        request.validate()?;
        if !compiled() {
            return Err(Error::Unavailable);
        }
        #[cfg(feature = "cuda-cutlass-fp8")]
        {
            let mut plan = Plan::default();
            // SAFETY: native code reads/writes these exact ABI-sized host values.
            status(unsafe { ferrum_cutlass_fp8_plan_v1(&request, &mut plan) })?;
            plan.validate()?;
            if plan.request != request {
                return Err(Error::Plan);
            }
            Ok(Self { plan })
        }
        #[cfg(not(feature = "cuda-cutlass-fp8"))]
        {
            Err(Error::Unavailable)
        }
    }
    pub fn plan(&self) -> &Plan {
        &self.plan
    }
    pub fn scratch_bytes(&self) -> u64 {
        self.plan.scratch_bytes
    }
    pub fn alignment(&self) -> u64 {
        CUTLASS_FP8_SCRATCH_ALIGNMENT
    }
    pub fn fingerprint_material(&self) -> [u8; 104] {
        self.plan.fingerprint_material()
    }

    /// Host structural/native can-implement validation only; no payload read,
    /// device enqueue or synchronization. Addresses must name the same context
    /// whose current device was used for preparation.
    pub fn validate(&self, bindings: &Bindings) -> Result<(), Error> {
        bindings.validate(&self.plan)?;
        #[cfg(feature = "cuda-cutlass-fp8")]
        {
            status(unsafe { ferrum_cutlass_fp8_validate_v1(&self.plan, bindings) })
        }
        #[cfg(not(feature = "cuda-cutlass-fp8"))]
        {
            Err(Error::Unavailable)
        }
    }

    /// Enqueue quantization followed by the prepared GEMM on one explicit stream.
    ///
    /// # Safety
    /// All spans must be admitted, live device storage on the bound context.
    /// The caller retains input/weight/scale/output/scratch and native opaque
    /// ownership through stream completion, including every error/unwind path.
    /// A nonzero status may follow quantization enqueue: it is NOT proof that
    /// nothing was submitted. No capture, implicit default stream or fallback.
    pub unsafe fn run(&self, bindings: &Bindings, stream: *mut c_void) -> Result<(), Error> {
        bindings.validate(&self.plan)?;
        if stream.is_null() {
            return Err(Error::InvalidArgument);
        }
        #[cfg(feature = "cuda-cutlass-fp8")]
        {
            status(unsafe { ferrum_cutlass_fp8_run_v1(&self.plan, bindings, stream) })
        }
        #[cfg(not(feature = "cuda-cutlass-fp8"))]
        {
            Err(Error::Unavailable)
        }
    }
}

pub fn compiled() -> bool {
    use ferrum_native_ops::cutlass_fp8::CUTLASS_FP8_EXPORTS;
    cfg!(feature = "cuda-cutlass-fp8")
        && super::compiled_native_operator_artifacts()
            .iter()
            .any(|artifact| {
                artifact.operator == CUTLASS_FP8_OPERATOR
                    && artifact.backend == ferrum_types::NativeOperatorBackend::Cuda
                    && artifact.linkage == ferrum_types::NativeOperatorLinkage::Static
                    && artifact.operator_abi_version == "1"
                    && CUTLASS_FP8_EXPORTS
                        .iter()
                        .all(|name| artifact.exports.iter().any(|actual| actual == name))
            })
}

#[cfg(feature = "cuda-cutlass-fp8")]
fn status(value: i32) -> Result<(), Error> {
    match value {
        0 => Ok(()),
        other => Err(Error::Native(other)),
    }
}

#[cfg(feature = "cuda-cutlass-fp8")]
unsafe extern "C" {
    fn ferrum_cutlass_fp8_plan_v1(request: *const Request, plan: *mut Plan) -> i32;
    fn ferrum_cutlass_fp8_validate_v1(plan: *const Plan, bindings: *const Bindings) -> i32;
    fn ferrum_cutlass_fp8_run_v1(
        plan: *const Plan,
        bindings: *const Bindings,
        stream: *mut c_void,
    ) -> i32;
}
