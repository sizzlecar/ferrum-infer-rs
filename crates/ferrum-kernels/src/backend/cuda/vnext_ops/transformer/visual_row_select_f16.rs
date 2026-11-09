//! Bit-copy selection over actual token spans, without reading masked visual rows.
use super::*;
const OP: &str = ferrum_runtime::ROW_SELECT_F16_OPERATION_ID;
const KERNEL: &str = "vnext_visual_row_select_f16";
pub(in crate::backend::cuda::vnext_ops) struct CudaVisualRowSelectF16Provider {
    descriptor: OperationProviderDescriptor,
    function: CudaFunction,
    element_type: ElementType,
}
impl CudaVisualRowSelectF16Provider {
    pub(in crate::backend::cuda::vnext_ops) fn new(
        runtime: &CudaDeviceRuntime,
    ) -> Result<Self, CudaDeviceRuntimeError> {
        Self::with_element_type(runtime, ElementType::F16)
    }
    pub(in crate::backend::cuda::vnext_ops) fn new_bf16(
        runtime: &CudaDeviceRuntime,
    ) -> Result<Self, CudaDeviceRuntimeError> {
        Self::with_element_type(runtime, ElementType::Bf16)
    }
    fn with_element_type(
        runtime: &CudaDeviceRuntime,
        element_type: ElementType,
    ) -> Result<Self, CudaDeviceRuntimeError> {
        let bf16 = element_type == ElementType::Bf16;
        let contract = if bf16 {
            ferrum_runtime::row_select_bf16_contract()
        } else {
            ferrum_runtime::row_select_f16_contract()
        }
        .map_err(contract_error)?;
        let descriptor = weightless_provider_descriptor(
            runtime,
            &contract,
            if bf16 {
                "provider.cuda.row_select.bf16"
            } else {
                "provider.cuda.row_select.f16"
            },
            if bf16 {
                ferrum_runtime::ROW_SELECT_BF16_CAPABILITY_ID
            } else {
                ferrum_runtime::ROW_SELECT_F16_CAPABILITY_ID
            },
            if bf16 {
                "resource-estimator.cuda.row_select.bf16"
            } else {
                "resource-estimator.cuda.row_select.f16"
            },
            contiguous_bindings(3),
            implementation_fingerprint(&[
                include_bytes!("visual_row_select_f16.rs"),
                crate::ptx::RESIDUAL_ADD.as_bytes(),
            ]),
        )?;
        let descriptor = descriptor.with_candidate_state_effects(
            ferrum_runtime::ProviderCandidateStateEffects::PrefixCausal { ports: Vec::new() },
        );
        let module = runtime
            .context()
            .load_module(Ptx::from_src(crate::ptx::RESIDUAL_ADD.to_owned()))
            .map_err(|e| CudaDeviceRuntimeError::driver("visual row-selection module load", e))?;
        let function = module
            .load_function(if bf16 {
                "vnext_visual_row_select_bf16"
            } else {
                KERNEL
            })
            .map_err(|e| CudaDeviceRuntimeError::driver("visual row-selection function load", e))?;
        Ok(Self {
            descriptor,
            function,
            element_type,
        })
    }
}
impl OperationResourceEstimator for CudaVisualRowSelectF16Provider {
    fn descriptor(&self) -> &OperationProviderDescriptor {
        &self.descriptor
    }
    fn estimate_resources(
        &self,
        request: OperationResourceEstimateRequest<'_>,
    ) -> Result<OperationResourceEstimate, VNextError> {
        estimate_without_workspace(
            &self.descriptor,
            &request,
            self.descriptor.operation_id().as_str(),
        )
    }
}
impl OperationProvider<CudaDeviceRuntime> for CudaVisualRowSelectF16Provider {
    fn reusable_execution_topology(
        &self,
        _: ReusableExecutionTopologyRequest<'_>,
    ) -> Result<ReusableExecutionTopology, VNextError> {
        Ok(ReusableExecutionTopology::EagerBoundary)
    }
    fn encode_selected(
        &self,
        invocation: BatchedOperationInvocation<'_, CudaDeviceBuffer>,
    ) -> Result<EncodedDeviceOperation<CudaDeviceCommand>, OperationFailure> {
        let identity = invocation.participants()[0].identity().clone();
        encode(&self.function, &invocation, self.element_type)
            .map(EncodedDeviceOperation::compute)
            .map_err(|e| provider_failure(identity, "cuda.row_select.f16.encode", e))
    }
}
fn encode(
    function: &CudaFunction,
    invocation: &BatchedOperationInvocation<'_, CudaDeviceBuffer>,
    element_type: ElementType,
) -> Result<CudaDeviceCommand, String> {
    ensure_invocation(
        invocation,
        if element_type == ElementType::Bf16 {
            ferrum_runtime::ROW_SELECT_BF16_OPERATION_ID
        } else {
            OP
        },
    )?;
    let mut regions = Vec::new();
    let mut launches = Vec::new();
    let first = &invocation.participants()[0];
    let columns = unsigned_attribute(first.attributes(), "hidden_size")?;
    if columns == 0 {
        return Err("visual row selection requires a positive hidden width".into());
    }
    for (index, participant) in invocation.participants().iter().enumerate() {
        if participant.attributes() != first.attributes() {
            return Err("visual row-selection attributes differ".into());
        }
        let range = invocation
            .participant_token_ranges()
            .get(index)
            .ok_or("visual row selection lacks actual token spans")?;
        let source = range.source_token_range();
        let tokens = range.immediate_tokens();
        let bytes = tokens
            .checked_mul(columns)
            .and_then(|n| n.checked_mul(2))
            .ok_or("visual row-selection extent overflows")?;
        let base = regions.len();
        for (role, ordinal, dtype, expected) in [
            (ResolvedValueRole::Input, 0, element_type, bytes),
            (ResolvedValueRole::Input, 1, element_type, bytes),
            (ResolvedValueRole::Input, 2, ElementType::U8, tokens),
            (ResolvedValueRole::Output, 0, element_type, bytes),
        ] {
            let packed = super::token_binding_is_packed(invocation, role, ordinal)?;
            let start = if packed {
                range.immediate_token_range().start
            } else {
                source.start
            };
            let region = contiguous_token_region(
                participant,
                binding(participant.bindings(), role, ordinal)?,
                dtype,
                start,
                tokens,
            )?;
            if region.length_bytes() != expected {
                return Err("visual row-selection physical extent differs from token work".into());
            }
            regions.push(region);
        }
        let elements = tokens
            .checked_mul(columns)
            .ok_or("visual row-selection size overflows")?;
        let grid = checked_u32(
            elements.div_ceil(u64::from(THREADS_PER_BLOCK)),
            "visual row-selection grid",
        )?;
        launches.push((base, elements, grid));
    }
    let function = function.clone();
    let participants = checked_u32(launches.len() as u64, "visual row-selection participants")?;
    let tokens = invocation.work_shape().immediate_tokens();
    CudaDeviceCommand::operation(KERNEL, regions, move |stream, regions| {
        for &(base, elements, grid) in &launches {
            let pointers = [
                regions[base].device_ptr(),
                regions[base + 1].device_ptr(),
                regions[base + 2].device_ptr(),
                regions[base + 3].device_ptr(),
            ];
            let mut launch = stream.launch_builder(&function);
            for p in &pointers {
                launch.arg(p);
            }
            launch.arg(&elements).arg(&columns);
            unsafe {
                launch.launch(LaunchConfig {
                    grid_dim: (grid, 1, 1),
                    block_dim: (THREADS_PER_BLOCK, 1, 1),
                    shared_mem_bytes: 0,
                })
            }
            .map_err(|e| CudaDeviceRuntimeError::driver("visual row-selection launch", e))?;
        }
        Ok(())
    })
    .and_then(|command| {
        command.with_work_attribution(
            if participants == 1 {
                DeviceBatchingForm::Scalar
            } else {
                DeviceBatchingForm::ParticipantLoop
            },
            participants,
            tokens,
            u64::from(participants),
            0,
        )
    })
    .map_err(|e| e.to_string())
}
