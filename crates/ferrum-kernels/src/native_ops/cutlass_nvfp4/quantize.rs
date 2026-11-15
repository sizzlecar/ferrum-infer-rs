//! Allocation-free quantization binding, sharing the locked NVFP4 module.
use super::{compiled, Error};
pub use ferrum_native_ops::cutlass_nvfp4::{
    Nvfp4QuantizeBf16BindingsV1 as QuantizeBindings, Nvfp4QuantizeBf16PlanV1 as QuantizePlan,
    Nvfp4QuantizeBf16RequestV1 as QuantizeRequest, NVFP4_QUANTIZE_BF16_NUMERICAL_IDENTITY,
};
use std::ffi::c_void;

#[derive(Debug, Clone)]
pub struct PreparedNvfp4QuantizeBf16 {
    plan: QuantizePlan,
}
impl PreparedNvfp4QuantizeBf16 {
    pub fn new(request: QuantizeRequest) -> Result<Self, Error> {
        request.validate()?;
        if !compiled() {
            return Err(Error::Unavailable);
        }
        #[cfg(feature = "cuda-cutlass-nvfp4")]
        {
            let mut plan = QuantizePlan::default();
            // SAFETY: fixed host ABI structures, no device payload reads.
            super::status(unsafe { ferrum_nvfp4_quantize_bf16_plan_v1(&request, &mut plan) })?;
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
    pub fn plan(&self) -> &QuantizePlan {
        &self.plan
    }
    pub fn fingerprint_material(&self) -> [u8; 64] {
        self.plan.fingerprint_material()
    }
    pub fn validate(&self, bindings: &QuantizeBindings) -> Result<(), Error> {
        bindings.validate(&self.plan)?;
        #[cfg(feature = "cuda-cutlass-nvfp4")]
        {
            super::status(unsafe { ferrum_nvfp4_quantize_bf16_validate_v1(&self.plan, bindings) })
        }
        #[cfg(not(feature = "cuda-cutlass-nvfp4"))]
        {
            Err(Error::Unavailable)
        }
    }
    /// # Safety
    /// The caller admits and retains module capacity, all four exact device
    /// spans and the explicit stream on the prepared context until completion,
    /// including partial launch errors. The source satisfies the profile's
    /// finite input/positive global domain and is ready before submission.
    /// No workspace, internal allocation or stream authority is supplied here.
    pub unsafe fn run(
        &self,
        bindings: &QuantizeBindings,
        stream: *mut c_void,
    ) -> Result<(), Error> {
        bindings.validate(&self.plan)?;
        if stream.is_null() || matches!(stream as usize, 1 | 2) {
            return Err(Error::InvalidArgument);
        }
        #[cfg(feature = "cuda-cutlass-nvfp4")]
        {
            super::status(unsafe {
                ferrum_nvfp4_quantize_bf16_run_v1(&self.plan, bindings, stream)
            })
        }
        #[cfg(not(feature = "cuda-cutlass-nvfp4"))]
        {
            Err(Error::Unavailable)
        }
    }
}
#[cfg(feature = "cuda-cutlass-nvfp4")]
unsafe extern "C" {
    fn ferrum_nvfp4_quantize_bf16_plan_v1(
        request: *const QuantizeRequest,
        plan: *mut QuantizePlan,
    ) -> i32;
    fn ferrum_nvfp4_quantize_bf16_validate_v1(
        plan: *const QuantizePlan,
        bindings: *const QuantizeBindings,
    ) -> i32;
    fn ferrum_nvfp4_quantize_bf16_run_v1(
        plan: *const QuantizePlan,
        bindings: *const QuantizeBindings,
        stream: *mut c_void,
    ) -> i32;
}
