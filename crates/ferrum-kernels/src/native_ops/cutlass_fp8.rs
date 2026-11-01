//! Static-link, allocation-free adapter for the locked SM89/SM120 FP8 artifact.
//! Native modules may remain loaded until context destruction. The caller must
//! keep its separately admitted native opaque owner for that full lifetime.

pub use ferrum_native_ops::cutlass_fp8::{
    CutlassFp8BindingsV1 as Bindings, CutlassFp8Device as Device, CutlassFp8Error as Error,
    CutlassFp8PlanV1 as Plan, CutlassFp8RequestV1 as Request, CutlassFp8Span as DeviceSpan,
    CUTLASS_FP8_KERNEL_128, CUTLASS_FP8_KERNEL_SM89, CUTLASS_FP8_NUMERICAL_IDENTITY,
    CUTLASS_FP8_OPERATOR, CUTLASS_FP8_SCRATCH_ALIGNMENT,
};
use std::ffi::c_void;

#[cfg(any(feature = "cuda", test))]
pub(crate) mod cache;
mod catalog;
pub use catalog::{
    group_provider_id, provider_catalog_declaration, source_declaration_fingerprint,
    CutlassFp8CatalogConfig, CUTLASS_FP8_PAIR_PROVIDER_ID, CUTLASS_FP8_PROVIDER_ID,
    CUTLASS_FP8_TRIPLE_PROVIDER_ID,
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

    /// Validate all branches before enqueue. The same input and scratch spans
    /// feed two or three independent, equal-shape projections. This does not
    /// read payloads or acquire device memory.
    pub fn validate_group(&self, bindings: &[Bindings]) -> Result<(), Error> {
        Bindings::validate_group(&self.plan, bindings)?;
        #[cfg(feature = "cuda-cutlass-fp8")]
        {
            status(unsafe {
                ferrum_cutlass_fp8_validate_group_v1(
                    &self.plan,
                    bindings.as_ptr(),
                    bindings.len() as u32,
                )
            })
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

    /// Quantize one shared input, then enqueue every independent GEMM in order.
    /// Native validation of the whole group completes before the first enqueue.
    ///
    /// # Safety
    /// The same ownership/context/completion rules as [`Self::run`] apply to
    /// every branch. Input and scratch must remain live and immutable to other
    /// work until all branches complete. Scratch may not be concurrently reused.
    /// An error may follow partial enqueue and never proves NotSubmitted.
    pub unsafe fn run_group(
        &self,
        bindings: &[Bindings],
        stream: *mut c_void,
    ) -> Result<(), Error> {
        Bindings::validate_group(&self.plan, bindings)?;
        if stream.is_null() {
            return Err(Error::InvalidArgument);
        }
        #[cfg(feature = "cuda-cutlass-fp8")]
        {
            status(unsafe {
                ferrum_cutlass_fp8_run_group_v1(
                    &self.plan,
                    bindings.as_ptr(),
                    bindings.len() as u32,
                    stream,
                )
            })
        }
        #[cfg(not(feature = "cuda-cutlass-fp8"))]
        {
            Err(Error::Unavailable)
        }
    }

    /// Opt-in command-event diagnostic: records before quantization, after it,
    /// and after all GEMMs. No event creation, synchronization or device payload
    /// allocation occurs. Intervals may include host enqueue gaps; they are not
    /// kernel-active cycles. Ordinary [`Self::run_group`] has no event calls.
    ///
    /// # Safety
    /// All [`Self::run_group`] requirements apply. `events` must contain three
    /// distinct, live timing-enabled CUDA events on the same bound context.
    /// The caller retains the events and every payload through completion,
    /// including partial errors, and reads timing only after completion.
    pub unsafe fn run_group_profiled(
        &self,
        bindings: &[Bindings],
        stream: *mut c_void,
        events: [*mut c_void; 3],
    ) -> Result<(), Error> {
        Bindings::validate_group(&self.plan, bindings)?;
        if stream.is_null()
            || events.iter().any(|event| event.is_null())
            || events[0] == events[1]
            || events[0] == events[2]
            || events[1] == events[2]
        {
            return Err(Error::InvalidArgument);
        }
        #[cfg(feature = "cuda-cutlass-fp8")]
        {
            status(unsafe {
                ferrum_cutlass_fp8_run_group_profiled_v1(
                    &self.plan,
                    bindings.as_ptr(),
                    bindings.len() as u32,
                    stream,
                    events[0],
                    events[1],
                    events[2],
                )
            })
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
    fn ferrum_cutlass_fp8_validate_group_v1(
        plan: *const Plan,
        bindings: *const Bindings,
        count: u32,
    ) -> i32;
    fn ferrum_cutlass_fp8_run_group_v1(
        plan: *const Plan,
        bindings: *const Bindings,
        count: u32,
        stream: *mut c_void,
    ) -> i32;
    fn ferrum_cutlass_fp8_run_group_profiled_v1(
        plan: *const Plan,
        bindings: *const Bindings,
        count: u32,
        stream: *mut c_void,
        start: *mut c_void,
        after_quant: *mut c_void,
        after_gemm: *mut c_void,
    ) -> i32;
}
