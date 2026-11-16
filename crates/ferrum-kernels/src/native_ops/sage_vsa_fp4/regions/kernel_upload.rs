//! Checked geometry copied through bounded CUDA kernel parameters. Device
//! scratch, numerical selection and retirement remain the original owners.
use super::*;

#[derive(Clone, Debug)]
pub struct PreparedSageVsaFp4KernelRegions {
    plan: RegionKernelPlanV1,
}
impl PreparedSageVsaFp4KernelRegions {
    pub fn new(attention: Request, region_count: u32) -> Result<Self, Error> {
        let request = RegionRequestV1 {
            attention,
            region_count,
            selection_profile: REGION_SELECTION_PROFILE,
        };
        request.validate()?;
        if !kernel_upload_compiled() {
            return Err(Error::Unavailable);
        }
        #[cfg(feature = "cuda-sage-vsa-fp4")]
        {
            let mut plan = RegionKernelPlanV1::default();
            // SAFETY: metadata only; this entry neither reads payloads nor enqueues.
            super::super::status(unsafe {
                ferrum_sage_vsa_fp4_region_kernel_plan_v1(&request, &mut plan)
            })?;
            plan.validate()?;
            if plan.geometry.request() != request {
                return Err(Error::Plan);
            }
            Ok(Self { plan })
        }
        #[cfg(not(feature = "cuda-sage-vsa-fp4"))]
        Err(Error::Unavailable)
    }
    pub fn plan(&self) -> &RegionKernelPlanV1 {
        &self.plan
    }
    /// # Safety
    /// Host extents are readable and immutable during this call. They need not
    /// be pinned. Device spans belong to this context and satisfy the original
    /// exclusive-write contract. Validation enqueues nothing.
    pub unsafe fn validate(
        &self,
        bindings: &RegionBindingsV1,
        host: &RegionHostV1,
    ) -> Result<(), Error> {
        self.plan.validate()?;
        bindings.attention.validate(&self.plan.geometry.attention)?;
        #[cfg(feature = "cuda-sage-vsa-fp4")]
        {
            super::super::status(unsafe {
                ferrum_sage_vsa_fp4_region_kernel_validate_v1(&self.plan, bindings, host)
            })
        }
        #[cfg(not(feature = "cuda-sage-vsa-fp4"))]
        {
            let _ = host;
            Err(Error::Unavailable)
        }
    }
    /// # Safety
    /// Host geometry is readable and immutable through this call. Native Run
    /// copies it by value into each launch before returning. Device/module
    /// owners, explicit stream ordering and exclusive writes must remain valid
    /// through completion/quarantine, including errors after a partial upload.
    pub unsafe fn run(
        &self,
        bindings: &RegionBindingsV1,
        host: &RegionHostV1,
        stream: *mut c_void,
    ) -> Result<(), Error> {
        self.plan.validate()?;
        bindings.attention.validate(&self.plan.geometry.attention)?;
        if stream.is_null() || matches!(stream as usize, 1 | 2) {
            return Err(Error::InvalidArgument);
        }
        #[cfg(feature = "cuda-sage-vsa-fp4")]
        {
            super::super::status(unsafe {
                ferrum_sage_vsa_fp4_region_kernel_run_v1(&self.plan, bindings, host, stream)
            })
        }
        #[cfg(not(feature = "cuda-sage-vsa-fp4"))]
        {
            let _ = host;
            Err(Error::Unavailable)
        }
    }
}
pub fn kernel_upload_compiled() -> bool {
    compiled()
        && super::super::super::compiled_native_operator_artifacts()
            .iter()
            .any(|a| {
                a.operator == super::super::SAGE_VSA_FP4_OPERATOR
                    && REGION_KERNEL_EXPORTS
                        .iter()
                        .all(|name| a.exports.iter().any(|s| s == name))
            })
}
#[cfg(feature = "cuda-sage-vsa-fp4")]
unsafe extern "C" {
    fn ferrum_sage_vsa_fp4_region_kernel_plan_v1(
        request: *const RegionRequestV1,
        plan: *mut RegionKernelPlanV1,
    ) -> i32;
    fn ferrum_sage_vsa_fp4_region_kernel_validate_v1(
        plan: *const RegionKernelPlanV1,
        bindings: *const RegionBindingsV1,
        host: *const RegionHostV1,
    ) -> i32;
    fn ferrum_sage_vsa_fp4_region_kernel_run_v1(
        plan: *const RegionKernelPlanV1,
        bindings: *const RegionBindingsV1,
        host: *const RegionHostV1,
        stream: *mut c_void,
    ) -> i32;
}
