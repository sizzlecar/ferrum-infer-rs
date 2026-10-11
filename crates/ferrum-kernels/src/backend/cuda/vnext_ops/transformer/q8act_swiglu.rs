//! Explicit SwiGLU provider; the retained plan is the sole leaf-route authority.
use super::super::native_blocks::q8act::{self, Q8ActKernels};
use super::*;
use ferrum_interfaces::vnext::{PreparedProjectionNumerics, Q8ActSwiGluProfile};

pub(in crate::backend::cuda::vnext_ops) struct CudaQ8ActSwiGluProvider {
    descriptor: OperationProviderDescriptor,
    native: super::super::native_blocks::CudaNativeBlockKernels,
    q8: Q8ActKernels,
    silu: CudaFunction,
}

impl CudaQ8ActSwiGluProvider {
    pub(in crate::backend::cuda::vnext_ops) fn new(
        runtime: &CudaDeviceRuntime,
        profile: Q8ActSwiGluProfile,
    ) -> Result<Self, CudaDeviceRuntimeError> {
        let contract = profile.contract().map_err(contract_error)?;
        let (provider_id, estimator_id) = match profile {
            Q8ActSwiGluProfile::Iq4Xs => (
                "provider.cuda.dense_swiglu.iq4xs-q8act-g32",
                "resource-estimator.cuda.dense_swiglu.iq4xs-q8act-g32",
            ),
            Q8ActSwiGluProfile::Q4KQ5KIq4Xs => (
                "provider.cuda.dense_swiglu.q4k-q5k-iq4xs-q8act-g32",
                "resource-estimator.cuda.dense_swiglu.q4k-q5k-iq4xs-q8act-g32",
            ),
        };
        let fingerprint = implementation_fingerprint(&[
            include_bytes!("q8act_swiglu.rs"),
            include_bytes!("native_swiglu.rs"),
            include_bytes!("../native_blocks/q8act.rs"),
            include_bytes!("../native_blocks.rs"),
            include_bytes!("../native_blocks/weights.rs"),
            include_bytes!("../native_blocks/hadamard.rs"),
            crate::ptx::VNEXT_GGUF.as_bytes(),
            crate::ptx::FUSED_SILU_MUL.as_bytes(),
        ]);
        let descriptor = provider_descriptor_with_formats(
            runtime,
            &contract,
            provider_id,
            profile.capability_id(),
            estimator_id,
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
            .map_err(|error| CudaDeviceRuntimeError::driver("Q8act SiLU module", error))?;
        let silu = module
            .load_function(SILU_MUL_FUNCTION_NAME)
            .map_err(|error| CudaDeviceRuntimeError::driver("Q8act SiLU function", error))?;
        Ok(Self {
            descriptor,
            silu,
            native: super::super::native_blocks::CudaNativeBlockKernels::load(runtime.context())?,
            q8: Q8ActKernels::load_for_profile(runtime.context(), profile)?,
        })
    }
}

impl OperationResourceEstimator for CudaQ8ActSwiGluProvider {
    fn descriptor(&self) -> &OperationProviderDescriptor {
        &self.descriptor
    }

    fn estimate_resources(
        &self,
        request: OperationResourceEstimateRequest<'_>,
    ) -> Result<OperationResourceEstimate, VNextError> {
        ensure_estimator_request(&self.descriptor, &request, self.q8.profile().operation_id())?;
        let prepared =
            PreparedProjectionNumerics::prepare(&self.q8.profile().arithmetic(), request.values())
                .map_err(invalid_plan)?;
        // Qualify strict as well as staged parts at preparation. Unsupported
        // native layouts/transform ABIs are startup errors, never lazy fallback.
        for projection in prepared.projections() {
            let value = binding(
                request.values(),
                ResolvedValueRole::Input,
                projection.weight_input_ordinal(),
            )
            .map_err(invalid_plan)?;
            let parts = super::super::native_blocks::weights::matrix_parts(
                value
                    .weight()
                    .ok_or_else(|| invalid_plan("Q8act projection weight is absent"))?,
                value.tensor().dimensions(),
            )
            .map_err(invalid_plan)?;
            Q8ActKernels::validate_parts_for_profile(self.q8.profile(), projection, &parts)
                .map_err(invalid_plan)?;
        }
        let intermediate =
            unsigned_attribute(request.attributes(), "intermediate_size").map_err(invalid_plan)?;
        let pack = q8act::workspace_per_token(&prepared).map_err(invalid_plan)?;
        let bytes = intermediate
            .checked_mul(6)
            .and_then(|bytes| bytes.checked_add(pack))
            .and_then(|bytes| {
                bytes.checked_add(
                    super::super::native_blocks::hadamard::workspace_bytes_per_token(
                        request.values(),
                    )
                    .ok()?,
                )
            })
            .ok_or_else(|| {
                invalid_plan("Q8act SwiGLU workspace overflows or transform is unsupported")
            })?;
        // At most 15 bytes align the shared qword/F32-scale region after the
        // actual fused/transform scratch. No weight expansion or persistent pack.
        let scratch = ProviderWorkspaceRequirement::from_formula(
            ProviderWorkspaceSizeFormula::affine(if pack > 0 { 15 } else { 0 }, 0, bytes)?,
            VALUE_ALIGNMENT_BYTES,
            ProviderWorkspaceScope::Invocation,
            ProviderWorkspaceReusePolicy::OverwriteBeforeRead,
            DynamicStorageRequirement::contiguous(),
        )?;
        Ok(
            estimate(&self.descriptor, request.input_fingerprint(), Some(scratch))
                .with_projection_numerics(prepared),
        )
    }
}

impl OperationProvider<CudaDeviceRuntime> for CudaQ8ActSwiGluProvider {
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
            provider_failure(identity, "cuda.dense_swiglu.q8act-g32.encode", message)
        })
    }
}
