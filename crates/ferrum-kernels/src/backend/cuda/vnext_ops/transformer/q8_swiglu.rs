//! Explicit, opt-in native SwiGLU with Q8 activation projections.
use super::super::native_blocks::q8_f32scale::Q8F32ScaleKernels;
use super::*;
use ferrum_interfaces::vnext::{
    dense_swiglu_q8_f32scale_contract, DENSE_SWIGLU_Q8_F32SCALE_CAPABILITY_ID,
    DENSE_SWIGLU_Q8_F32SCALE_OPERATION_ID,
};

pub(in crate::backend::cuda::vnext_ops) struct CudaQ8SwiGluProvider {
    descriptor: OperationProviderDescriptor,
    native: super::super::native_blocks::CudaNativeBlockKernels,
    q8: Q8F32ScaleKernels,
    silu: CudaFunction,
}

impl CudaQ8SwiGluProvider {
    pub(in crate::backend::cuda::vnext_ops) fn new(
        runtime: &CudaDeviceRuntime,
    ) -> Result<Self, CudaDeviceRuntimeError> {
        let contract = dense_swiglu_q8_f32scale_contract().map_err(contract_error)?;
        let fingerprint = implementation_fingerprint(&[
            include_str!("q8_swiglu.rs").as_bytes(),
            include_str!("native_swiglu.rs").as_bytes(),
            include_str!("../native_blocks/q8_f32scale.rs").as_bytes(),
            include_str!("../native_blocks.rs").as_bytes(),
            include_str!("../native_blocks/weights.rs").as_bytes(),
            crate::ptx::VNEXT_GGUF.as_bytes(),
            crate::ptx::FUSED_SILU_MUL.as_bytes(),
        ]);
        let descriptor = provider_descriptor_with_formats(
            runtime,
            &contract,
            "provider.cuda.dense_swiglu.q8-f32scale",
            DENSE_SWIGLU_Q8_F32SCALE_CAPABILITY_ID,
            "resource-estimator.cuda.dense_swiglu.q8-f32scale",
            contiguous_bindings(3),
            BTreeSet::from([
                WeightFormatId::new("weight-format.gguf.native-block").map_err(contract_error)?
            ]),
            native_linear::quantization_formats().map_err(contract_error)?,
            fingerprint,
        )?;
        let module = runtime
            .context()
            .load_module(Ptx::from_src(crate::ptx::FUSED_SILU_MUL.to_owned()))
            .map_err(|e| CudaDeviceRuntimeError::driver("Q8 SwiGLU SiLU module", e))?;
        let silu = module
            .load_function(SILU_MUL_FUNCTION_NAME)
            .map_err(|e| CudaDeviceRuntimeError::driver("Q8 SwiGLU SiLU function", e))?;
        Ok(Self {
            descriptor,
            silu,
            native: super::super::native_blocks::CudaNativeBlockKernels::load(runtime.context())?,
            q8: Q8F32ScaleKernels::load(runtime.context())?,
        })
    }
}

impl OperationResourceEstimator for CudaQ8SwiGluProvider {
    fn descriptor(&self) -> &OperationProviderDescriptor {
        &self.descriptor
    }

    fn estimate_resources(
        &self,
        request: OperationResourceEstimateRequest<'_>,
    ) -> Result<OperationResourceEstimate, VNextError> {
        ensure_estimator_request(
            &self.descriptor,
            &request,
            DENSE_SWIGLU_Q8_F32SCALE_OPERATION_ID,
        )?;
        let intermediate =
            unsigned_attribute(request.attributes(), "intermediate_size").map_err(invalid_plan)?;
        let packed =
            native_swiglu::q8_workspace_per_token(request.values()).map_err(invalid_plan)?;
        let bytes = intermediate
            .checked_mul(6)
            .and_then(|bytes| bytes.checked_add(packed))
            .ok_or_else(|| invalid_plan("Q8 SwiGLU workspace overflows"))?;
        let scratch = ProviderWorkspaceRequirement::from_formula(
            ProviderWorkspaceSizeFormula::tokens(bytes)?,
            VALUE_ALIGNMENT_BYTES,
            ProviderWorkspaceScope::Invocation,
            ProviderWorkspaceReusePolicy::OverwriteBeforeRead,
            DynamicStorageRequirement::contiguous(),
        )?;
        Ok(estimate(
            &self.descriptor,
            request.input_fingerprint(),
            Some(scratch),
        ))
    }
}

impl OperationProvider<CudaDeviceRuntime> for CudaQ8SwiGluProvider {
    fn reusable_execution_topology(
        &self,
        request: ReusableExecutionTopologyRequest<'_>,
    ) -> Result<ReusableExecutionTopology, VNextError> {
        static_contiguous_reusable_topology(&request, 3, &[CapturedProviderWorkspace::Scratch])
    }

    fn encode_selected(
        &self,
        invocation: BatchedOperationInvocation<'_, CudaDeviceBuffer>,
    ) -> Result<EncodedDeviceOperation<CudaDeviceCommand>, OperationFailure> {
        let identity = invocation.participants()[0].identity().clone();
        native_swiglu::encode_with_q8(
            self.descriptor.provider_implementation_fingerprint(),
            &self.native,
            &self.silu,
            Some(&self.q8),
            invocation,
        )
        .map(EncodedDeviceOperation::compute)
        .map_err(|message| {
            provider_failure(identity, "cuda.dense_swiglu.q8-f32scale.encode", message)
        })
    }
}
