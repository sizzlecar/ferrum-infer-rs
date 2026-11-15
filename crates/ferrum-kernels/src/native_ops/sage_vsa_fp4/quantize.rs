//! Three launches on caller-owned spans; no workspace or metadata upload.
use super::Error;
pub use ferrum_native_ops::sage_vsa_fp4::quantize::{
    Bindings, BufferSizes, Plan, Request, EXPORTS, NUMERICAL_IDENTITY,
};
use std::ffi::c_void;

#[derive(Clone, Debug)]
pub struct PreparedSageVsaFp4Quantize {
    plan: Plan,
}
impl PreparedSageVsaFp4Quantize {
    pub fn new(request: Request) -> Result<Self, Error> {
        request.validate()?;
        if !compiled() {
            return Err(Error::Unavailable);
        }
        #[cfg(feature = "cuda-sage-vsa-fp4")]
        {
            let mut plan = Plan::default();
            // SAFETY: pure Host metadata; no payload access or enqueue.
            super::status(unsafe { ferrum_sage_vsa_fp4_quantize_plan_v1(&request, &mut plan) })?;
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
    /// Every span belongs to this prepared current context and its admitted
    /// owner remains live throughout this call. This call enqueues no work.
    pub unsafe fn validate(&self, bindings: &Bindings) -> Result<(), Error> {
        bindings.validate(&self.plan)?;
        #[cfg(feature = "cuda-sage-vsa-fp4")]
        {
            super::status(unsafe { ferrum_sage_vsa_fp4_quantize_validate_v1(&self.plan, bindings) })
        }
        #[cfg(not(feature = "cuda-sage-vsa-fp4"))]
        {
            Err(Error::Unavailable)
        }
    }
    /// # Safety
    /// Inputs are ready on the explicit stream, outputs exclusively writable,
    /// and all spans are admitted live storage in the prepared current context.
    /// Retain every input/output/module owner until completion or quarantine.
    /// An error may follow a successfully enqueued earlier Q/K/V launch.
    pub unsafe fn run(&self, bindings: &Bindings, stream: *mut c_void) -> Result<(), Error> {
        bindings.validate(&self.plan)?;
        if stream.is_null() || matches!(stream as usize, 1 | 2) {
            return Err(Error::InvalidArgument);
        }
        #[cfg(feature = "cuda-sage-vsa-fp4")]
        {
            super::status(unsafe {
                ferrum_sage_vsa_fp4_quantize_run_v1(&self.plan, bindings, stream)
            })
        }
        #[cfg(not(feature = "cuda-sage-vsa-fp4"))]
        {
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
                    && EXPORTS
                        .iter()
                        .all(|name| a.exports.iter().any(|s| s == name))
            })
}
#[cfg(feature = "cuda-sage-vsa-fp4")]
unsafe extern "C" {
    fn ferrum_sage_vsa_fp4_quantize_plan_v1(request: *const Request, plan: *mut Plan) -> i32;
    fn ferrum_sage_vsa_fp4_quantize_validate_v1(
        plan: *const Plan,
        bindings: *const Bindings,
    ) -> i32;
    fn ferrum_sage_vsa_fp4_quantize_run_v1(
        plan: *const Plan,
        bindings: *const Bindings,
        stream: *mut c_void,
    ) -> i32;
}
