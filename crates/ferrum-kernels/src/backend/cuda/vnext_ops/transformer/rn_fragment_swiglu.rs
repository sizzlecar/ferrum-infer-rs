//! Explicit dual-representation RN-F16 FFN. Physical M=1..8 uses fragment MMA;
//! larger M keeps the original dense GemmEx sequence and every F16 stage point.
use super::*;
use cublas_api::{CublasHandleApiIdentity, GemmF16ApiPlan};
use ferrum_interfaces::execution_cost::SelectedCommandCostEvidenceV1;
use ferrum_interfaces::vnext::{
    dense_swiglu_gguf_rn_f16_fragment_m1to8_contract, CheckpointBoundaryConstraint,
    CheckpointInputDependency, CheckpointPartitionNumerics, DeviceCommandPhase,
    OperationCostCommand, OperationCostRoute, OperationCostRouteRequest,
    ProviderCheckpointCapability, ProviderCheckpointContract, RnF16FragmentPlanV1,
    RnF16FragmentSourceFormatV1, DENSE_SWIGLU_GGUF_RN_F16_FRAGMENT_M1_TO8_CAPABILITY_ID,
    DENSE_SWIGLU_GGUF_RN_F16_FRAGMENT_M1_TO8_OPERATION_ID,
};
use ferrum_types::SloStructuredCostCapture;

mod execution;
mod plan;
mod weights;
pub(super) use plan::Shape;

const ENTRY: &str = "vnext_rn_fragment_mma";
const LABEL: &str = "vnext_rn_fragment_swiglu";

/// Bind eligibility to the actual embedded module, including builds targeting
/// pre-MMA devices. An unknown or ambiguous target never advertises this op.
pub(in crate::backend::cuda::vnext_ops) fn compiled_mma_target(ptx: &str) -> bool {
    let mut targets = ptx.lines().filter_map(|line| {
        let mut words = line.split("//").next()?.split_whitespace();
        (words.next()? == ".target").then(|| words.next())
    });
    let Some(Some(target)) = targets.next() else {
        return false;
    };
    if targets.next().is_some() {
        return false;
    }
    let Some(suffix) = target.trim_end_matches(',').strip_prefix("sm_") else {
        return false;
    };
    let digits = suffix.strip_suffix('a').unwrap_or(suffix);
    !digits.is_empty()
        && digits.bytes().all(|b| b.is_ascii_digit())
        && digits.parse::<u32>().is_ok_and(|target| target >= 80)
}

pub(in crate::backend::cuda::vnext_ops) struct CudaRnFragmentSwiGluProvider {
    descriptor: OperationProviderDescriptor,
    capture: SloStructuredCostCapture,
    cublas_identity: cublas_api::CublasCostIdentitySource,
    mma: CudaFunction,
    silu: CudaFunction,
}

impl CudaRnFragmentSwiGluProvider {
    pub(in crate::backend::cuda::vnext_ops) fn new(
        runtime: &CudaDeviceRuntime,
    ) -> Result<Self, CudaDeviceRuntimeError> {
        let contract =
            dense_swiglu_gguf_rn_f16_fragment_m1to8_contract().map_err(contract_error)?;
        let fingerprint = implementation_fingerprint(&[
            include_bytes!("rn_fragment_swiglu.rs"),
            include_bytes!("rn_fragment_swiglu/plan.rs"),
            include_bytes!("rn_fragment_swiglu/weights.rs"),
            include_bytes!("rn_fragment_swiglu/execution.rs"),
            include_bytes!("replay_cost.rs"),
            include_bytes!("cublas_api.rs"),
            include_bytes!("native_swiglu/selected.rs"),
            crate::ptx::VNEXT_GGUF.as_bytes(),
            crate::ptx::FUSED_SILU_MUL.as_bytes(),
        ]);
        let descriptor = provider_descriptor_with_formats(
            runtime,
            &contract,
            "provider.cuda.dense_swiglu.gguf-rn-f16-fragment-m1to8",
            DENSE_SWIGLU_GGUF_RN_F16_FRAGMENT_M1_TO8_CAPABILITY_ID,
            "resource-estimator.cuda.dense_swiglu.gguf-rn-f16-fragment-m1to8",
            contiguous_bindings(3),
            BTreeSet::from([WeightFormatId::new(
                crate::gguf_rn_fragment_materializer::GGUF_RN_FRAGMENT_FORMAT_ID,
            )
            .map_err(contract_error)?]),
            [
                RnF16FragmentSourceFormatV1::Q4K,
                RnF16FragmentSourceFormatV1::Q5K,
                RnF16FragmentSourceFormatV1::Q6K,
            ]
            .into_iter()
            .map(|format| {
                let encoding = RnF16FragmentPlanV1::new(format, 1, 256)?.packed_encoding();
                let ferrum_interfaces::vnext::WeightEncoding::BlockQuantized(spec) = encoding
                else {
                    unreachable!()
                };
                Ok(spec.format_id)
            })
            .collect::<Result<BTreeSet<_>, VNextError>>()
            .map_err(contract_error)?,
            fingerprint,
        )?
        // Both physical M routes read only the current inputs and immutable
        // weights; gate/up and activation scratch is overwritten in this call.
        // The FFN contributes no durable state. Restored complete upstream
        // state therefore preserves identical-suffix continuation with the
        // same partitions and implementation, not cross-partition equality.
        .with_checkpoint_capability(ProviderCheckpointCapability::CompletedBoundary(
            ProviderCheckpointContract::new(
                CheckpointInputDependency::ExactTokenPrefix,
                CheckpointBoundaryConstraint::any_positive(),
                CheckpointPartitionNumerics::CapturedExecutionContinuation,
            ),
        ));
        let mma = runtime
            .context()
            .load_module(Ptx::from_src(crate::ptx::VNEXT_GGUF))
            .map_err(|e| CudaDeviceRuntimeError::driver("RN fragment module", e))?
            .load_function(ENTRY)
            .map_err(|e| CudaDeviceRuntimeError::driver("RN fragment function", e))?;
        let silu = runtime
            .context()
            .load_module(Ptx::from_src(crate::ptx::FUSED_SILU_MUL))
            .map_err(|e| CudaDeviceRuntimeError::driver("RN fragment SiLU module", e))?
            .load_function(SILU_MUL_FUNCTION_NAME)
            .map_err(|e| CudaDeviceRuntimeError::driver("RN fragment SiLU function", e))?;
        Ok(Self {
            descriptor,
            capture: runtime.structured_capture(),
            cublas_identity: runtime.cublas_cost_identity_source(),
            mma,
            silu,
        })
    }
}

impl OperationResourceEstimator for CudaRnFragmentSwiGluProvider {
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
            DENSE_SWIGLU_GGUF_RN_F16_FRAGMENT_M1_TO8_OPERATION_ID,
        )?;
        // Both physical representations are validated even when only one will
        // execute. Their resident bytes belong to the weight resource plan.
        let shape =
            Shape::from_values(request.values(), request.attributes(), 1).map_err(invalid_plan)?;
        let scratch = ProviderWorkspaceRequirement::from_formula(
            ProviderWorkspaceSizeFormula::tokens(shape.scratch_bytes)?,
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

impl OperationProvider<CudaDeviceRuntime> for CudaRnFragmentSwiGluProvider {
    fn uses_captured_replay_cost_recipe(&self) -> bool {
        true
    }
    fn replayed_compute_cost_evidence(
        &self,
        invocation: &BatchedOperationInvocation<'_, CudaDeviceBuffer>,
    ) -> Result<Option<SelectedCommandCostEvidenceV1>, VNextError> {
        if self.capture.is_disabled() {
            return Ok(None);
        }
        let prepared = execution::prepare(invocation).map_err(invalid_plan)?;
        Ok(prepared
            .shape
            .selected(self.capture, self.cublas_identity.frozen()))
    }
    fn reusable_execution_cost_topology(
        &self,
        request: OperationCostRouteRequest<'_>,
    ) -> Result<Option<ReusableExecutionTopology>, VNextError> {
        static_contiguous_reusable_topology(&request, 3, &[CapturedProviderWorkspace::Scratch])
            .map(Some)
    }
    fn reusable_execution_topology(
        &self,
        request: ReusableExecutionTopologyRequest<'_>,
    ) -> Result<ReusableExecutionTopology, VNextError> {
        static_contiguous_reusable_topology(&request, 3, &[CapturedProviderWorkspace::Scratch])
    }
    fn eager_cost_route(
        &self,
        request: OperationCostRouteRequest<'_>,
    ) -> Result<Option<OperationCostRoute>, VNextError> {
        if self.capture.is_disabled() {
            return Ok(None);
        }
        if request.operation_id().as_str() != DENSE_SWIGLU_GGUF_RN_F16_FRAGMENT_M1_TO8_OPERATION_ID
        {
            return Ok(None);
        }
        let shape = Shape::from_values(
            request.bindings(),
            request.attributes(),
            request.immediate_tokens(),
        )
        .map_err(invalid_plan)?;
        if request.rows().len() > 1
            && (!request.binding_uses_packed_batch_coordinates(ResolvedValueRole::Input, 0)?
                || !request.binding_uses_packed_batch_coordinates(ResolvedValueRole::Output, 0)?)
        {
            return Ok(None);
        }
        let Some(evidence) = shape.selected(self.capture, self.cublas_identity.frozen()) else {
            return Ok(None);
        };
        let command = OperationCostCommand::new(
            LABEL,
            DeviceCommandPhase::Compute,
            DeviceBatchingForm::Packed,
            0,
            checked_u32(request.rows().len() as u64, "RN fragment participants")
                .map_err(invalid_plan)?,
            shape.tokens,
            3,
            0,
        )?
        .with_statistical_evidence(evidence)
        .map_err(|reason| invalid_plan(format!("RN fragment cost evidence: {reason:?}")))?;
        Ok(Some(OperationCostRoute::new(vec![command])?))
    }
    fn encode_selected(
        &self,
        invocation: BatchedOperationInvocation<'_, CudaDeviceBuffer>,
    ) -> Result<EncodedDeviceOperation<CudaDeviceCommand>, OperationFailure> {
        let identity = invocation.participants()[0].identity().clone();
        execution::encode(
            self.descriptor.provider_implementation_fingerprint(),
            &self.mma,
            &self.silu,
            self.capture,
            self.cublas_identity.frozen(),
            invocation,
        )
        .map(EncodedDeviceOperation::compute)
        .map_err(|message| provider_failure(identity, "cuda.rn_fragment_swiglu.encode", message))
    }
}

#[cfg(test)]
mod tests;
