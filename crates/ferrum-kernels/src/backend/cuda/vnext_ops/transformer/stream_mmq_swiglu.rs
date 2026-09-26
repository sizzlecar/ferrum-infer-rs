//! Opt-in B8 gate/up Q4 integer MMA with strict fallback and strict down.
use super::super::native_blocks::stream_mmq::{self, StreamMmq};
use super::*;
use ferrum_interfaces::vnext::{
    dense_swiglu_q8_gate_up_stream_mmq_contract, dense_swiglu_q8_residual2_ffn_m2to8_contract,
    DENSE_SWIGLU_Q8_GATE_UP_STREAM_MMQ_CAPABILITY_ID,
    DENSE_SWIGLU_Q8_RESIDUAL2_FFN_M2_TO8_CAPABILITY_ID,
};

pub(in crate::backend::cuda::vnext_ops) struct CudaStreamMmqSwiGluProvider {
    descriptor: OperationProviderDescriptor,
    structured_capture: ferrum_types::SloStructuredCostCapture,
    native: super::super::native_blocks::CudaNativeBlockKernels,
    mmq: StreamMmq,
    silu: CudaFunction,
}

impl CudaStreamMmqSwiGluProvider {
    pub(in crate::backend::cuda::vnext_ops) fn new(
        runtime: &CudaDeviceRuntime,
    ) -> Result<Self, CudaDeviceRuntimeError> {
        Self::with_residual2(runtime, false)
    }
    pub(in crate::backend::cuda::vnext_ops) fn residual2_m2to8(
        runtime: &CudaDeviceRuntime,
    ) -> Result<Self, CudaDeviceRuntimeError> {
        Self::with_residual2(runtime, true)
    }
    fn with_residual2(
        runtime: &CudaDeviceRuntime,
        residual2: bool,
    ) -> Result<Self, CudaDeviceRuntimeError> {
        let contract = if residual2 {
            dense_swiglu_q8_residual2_ffn_m2to8_contract()
        } else {
            dense_swiglu_q8_gate_up_stream_mmq_contract()
        }
        .map_err(contract_error)?;
        let fingerprint = implementation_fingerprint(&[
            include_str!("stream_mmq_swiglu.rs").as_bytes(),
            include_str!("../transformer.rs").as_bytes(),
            include_str!("native_swiglu.rs").as_bytes(),
            include_bytes!("native_swiglu/prepared.rs"),
            include_bytes!("native_swiglu/selected.rs"),
            include_bytes!("../native_blocks/linear_launch.rs"),
            include_bytes!("../native_blocks/selected.rs"),
            include_bytes!("../native_blocks/hadamard.rs"),
            include_str!("native_swiglu/route_selection.rs").as_bytes(),
            include_str!("native_swiglu/cost_route.rs").as_bytes(),
            include_str!("../native_blocks/stream_mmq.rs").as_bytes(),
            include_bytes!("../native_blocks/stream_mmq/launch_plan.rs"),
            include_bytes!("../native_blocks/stream_mmq/selected.rs"),
            include_str!("../native_blocks.rs").as_bytes(),
            include_str!("../native_blocks/weights.rs").as_bytes(),
            stream_mmq::PTX.as_bytes(),
            crate::ptx::VNEXT_GGUF.as_bytes(),
            crate::ptx::FUSED_SILU_MUL.as_bytes(),
        ]);
        let descriptor = provider_descriptor_with_formats(
            runtime,
            &contract,
            if residual2 {
                "provider.cuda.dense_swiglu.q8-residual2-ffn-m2to8-stream-mmq"
            } else {
                "provider.cuda.dense_swiglu.q8-gate-up-stream-mmq-f32scale"
            },
            if residual2 {
                DENSE_SWIGLU_Q8_RESIDUAL2_FFN_M2_TO8_CAPABILITY_ID
            } else {
                DENSE_SWIGLU_Q8_GATE_UP_STREAM_MMQ_CAPABILITY_ID
            },
            if residual2 {
                "resource-estimator.cuda.dense_swiglu.q8-residual2-ffn-m2to8-stream-mmq"
            } else {
                "resource-estimator.cuda.dense_swiglu.q8-gate-up-stream-mmq-f32scale"
            },
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
            .map_err(|e| CudaDeviceRuntimeError::driver("Stream-MMQ SiLU module", e))?;
        let silu = module
            .load_function(SILU_MUL_FUNCTION_NAME)
            .map_err(|e| CudaDeviceRuntimeError::driver("Stream-MMQ SiLU function", e))?;
        Ok(Self {
            structured_capture: runtime.structured_capture(),
            descriptor,
            silu,
            native: super::super::native_blocks::CudaNativeBlockKernels::load(runtime.context())?,
            mmq: if residual2 {
                StreamMmq::load_residual2(runtime.context())?
            } else {
                StreamMmq::load(runtime.context())?
            },
        })
    }
}

impl OperationResourceEstimator for CudaStreamMmqSwiGluProvider {
    fn descriptor(&self) -> &OperationProviderDescriptor {
        &self.descriptor
    }
    fn estimate_resources(
        &self,
        request: OperationResourceEstimateRequest<'_>,
    ) -> Result<OperationResourceEstimate, VNextError> {
        ensure_estimator_request(&self.descriptor, &request, self.mmq.operation_id())?;
        let hidden =
            unsigned_attribute(request.attributes(), "hidden_size").map_err(invalid_plan)?;
        let intermediate =
            unsigned_attribute(request.attributes(), "intermediate_size").map_err(invalid_plan)?;
        let fixed = native_swiglu::stream_mmq_workspace(
            request.values(),
            checked_u32(hidden, "Stream-MMQ hidden").map_err(invalid_plan)?,
            checked_u32(intermediate, "Stream-MMQ intermediate").map_err(invalid_plan)?,
            &self.mmq,
        )
        .map_err(invalid_plan)?;
        let per_token = intermediate
            .checked_mul(6)
            .and_then(|bytes| {
                bytes.checked_add(
                    super::super::native_blocks::hadamard::workspace_bytes_per_token(
                        request.values(),
                    )
                    .ok()?,
                )
            })
            .ok_or_else(|| {
                invalid_plan("Stream-MMQ/strict scratch overflow or invalid transform")
            })?;
        // The fixed bound is shared by gate/up and all participants, retained
        // even for fallback shapes. No allocation takes place during encoding.
        let scratch = ProviderWorkspaceRequirement::from_formula(
            ProviderWorkspaceSizeFormula::affine(fixed, 0, per_token)?,
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

impl OperationProvider<CudaDeviceRuntime> for CudaStreamMmqSwiGluProvider {
    fn replayed_compute_cost_evidence(
        &self,
        invocation: &BatchedOperationInvocation<'_, CudaDeviceBuffer>,
    ) -> Result<Option<ferrum_interfaces::execution_cost::SelectedCommandCostEvidenceV1>, VNextError>
    {
        native_swiglu::replay_evidence(
            self.descriptor.provider_implementation_fingerprint(),
            None,
            Some(&self.mmq),
            self.structured_capture,
            invocation,
        )
    }

    fn eager_cost_route(
        &self,
        request: ferrum_interfaces::vnext::OperationCostRouteRequest<'_>,
    ) -> Result<Option<ferrum_interfaces::vnext::OperationCostRoute>, VNextError> {
        native_swiglu::eager_stream_mmq_cost_route(request, &self.mmq, self.structured_capture)
    }
    fn reusable_execution_cost_topology(
        &self,
        request: ferrum_interfaces::vnext::OperationCostRouteRequest<'_>,
    ) -> Result<Option<ReusableExecutionTopology>, VNextError> {
        self.shared_reusable_topology(&request).map(Some)
    }
    fn reusable_execution_topology(
        &self,
        request: ReusableExecutionTopologyRequest<'_>,
    ) -> Result<ReusableExecutionTopology, VNextError> {
        self.shared_reusable_topology(&request)
    }
    fn encode_selected(
        &self,
        invocation: BatchedOperationInvocation<'_, CudaDeviceBuffer>,
    ) -> Result<EncodedDeviceOperation<CudaDeviceCommand>, OperationFailure> {
        let identity = invocation.participants()[0].identity().clone();
        native_swiglu::encode_with_stream_mmq(
            self.descriptor.provider_implementation_fingerprint(),
            &self.native,
            &self.silu,
            &self.mmq,
            self.structured_capture,
            invocation,
        )
        .map(EncodedDeviceOperation::compute)
        .map_err(|message| {
            provider_failure(identity, "cuda.dense_swiglu.stream-mmq.encode", message)
        })
    }
}

impl CudaStreamMmqSwiGluProvider {
    fn shared_reusable_topology(
        &self,
        request: &impl ReusableExecutionTopologyView,
    ) -> Result<ReusableExecutionTopology, VNextError> {
        static_contiguous_reusable_topology(request, 3, &[CapturedProviderWorkspace::Scratch])
    }
}
