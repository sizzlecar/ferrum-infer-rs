//! Independent projections share one admitted activation-quantization scratch.
//! The scalar and grouped providers retain one module owner and one plan cache.
use super::*;
use ferrum_runtime::{
    dense_linear_fp8_group_bf16_contract, DenseLinearFp8Group, DenseLinearFp8GroupBf16Geometry,
    ProviderStorageBindingRequirement,
};

const NUMERICAL_PROFILE: &str = "cuda.cutlass.w8a8.bf16.shared_quant.rowwise_e4m3fn.v1";

pub(in super::super) fn providers(
    runtime: &CudaDeviceRuntime,
    config: &CudaCutlassFp8Config,
) -> Result<Vec<Box<dyn OperationProvider<CudaDeviceRuntime>>>, CudaDeviceRuntimeError> {
    for group in [DenseLinearFp8Group::Pair, DenseLinearFp8Group::Triple] {
        let capability = CapabilityId::new(group.capability_id()).map_err(contract_error)?;
        if !runtime.descriptor().capabilities.contains(&capability) {
            return Err(CudaDeviceRuntimeError::contract(
                "grouped FP8 runtime capability absent",
            ));
        }
    }
    let shared = Arc::new(Provider::new(runtime, config)?);
    Ok(vec![
        Box::new(shared.as_ref().clone()),
        Box::new(GroupProvider::new(
            runtime,
            Arc::clone(&shared),
            DenseLinearFp8Group::Pair,
        )?),
        Box::new(GroupProvider::new(
            runtime,
            shared,
            DenseLinearFp8Group::Triple,
        )?),
    ])
}

struct GroupProvider {
    descriptor: OperationProviderDescriptor,
    shared: Arc<Provider>,
    group: DenseLinearFp8Group,
}

impl GroupProvider {
    fn new(
        runtime: &CudaDeviceRuntime,
        shared: Arc<Provider>,
        group: DenseLinearFp8Group,
    ) -> Result<Self, CudaDeviceRuntimeError> {
        let contract = dense_linear_fp8_group_bf16_contract(group).map_err(contract_error)?;
        let provider = cutlass_fp8::group_provider_id(group);
        let implementation = implementation_fingerprint(&[
            shared
                .descriptor
                .provider_implementation_fingerprint()
                .as_bytes(),
            NUMERICAL_PROFILE.as_bytes(),
            group.operation_id().as_bytes(),
            include_bytes!("grouped.rs"),
        ]);
        let estimator_id = format!("resource-estimator.{provider}");
        let estimator =
            implementation_fingerprint(&[estimator_id.as_bytes(), implementation.as_bytes()]);
        let storage = (0..=group.count() as u32)
            .map(|i| {
                ProviderStorageBindingRequirement::new(
                    ResolvedValueRole::Input,
                    i,
                    DynamicStorageRequirement::contiguous(),
                )
            })
            .chain((0..group.count() as u32).map(|i| {
                ProviderStorageBindingRequirement::new(
                    ResolvedValueRole::Output,
                    i,
                    DynamicStorageRequirement::contiguous(),
                )
            }))
            .collect();
        let descriptor = OperationProviderDescriptor::new(
            ProviderId::new(provider).map_err(contract_error)?,
            contract.descriptor().id.clone(),
            contract
                .descriptor()
                .fingerprint()
                .map_err(contract_error)?,
            implementation,
            ProviderExecutionSemantics::bitwise_eager_only(),
            contract.descriptor().version,
            runtime.descriptor().id.clone(),
            BTreeSet::from([CapabilityId::new(group.capability_id()).map_err(contract_error)?]),
            BTreeSet::from([
                WeightFormatId::new(super::super::DENSE_SAFETENSORS_FORMAT_ID)
                    .map_err(contract_error)?,
            ]),
            BTreeSet::from([
                QuantizationFormatId::new(FP8_E4M3FN_WHOLE_TENSOR_F32_FORMAT_ID)
                    .map_err(contract_error)?,
            ]),
            storage,
            &estimator_id,
            ContractVersion::new(1, 0),
            estimator,
        )
        .map_err(contract_error)?
        .with_fixed_tensor_work(true);
        Ok(Self {
            descriptor,
            shared,
            group,
        })
    }

    fn geometry(
        &self,
        values: &[ResolvedValueBinding],
        attrs: &BTreeMap<AttributeId, SemanticValue>,
    ) -> Result<DenseLinearFp8GroupBf16Geometry, String> {
        if values.len() != 1 + 2 * self.group.count() || attrs.len() != 2 {
            return Err("grouped FP8 requires its exact ports and two shape attributes".into());
        }
        let x = binding(values, ResolvedValueRole::Input, 0)?;
        let mut ws = Vec::with_capacity(self.group.count());
        let mut ys = Vec::with_capacity(self.group.count());
        for i in 0..self.group.count() as u32 {
            let w = binding(values, ResolvedValueRole::Input, i + 1)?;
            weights::validate(w)?;
            ws.push(w.tensor());
            ys.push(binding(values, ResolvedValueRole::Output, i)?.tensor());
        }
        DenseLinearFp8GroupBf16Geometry::new(
            self.group,
            x.tensor(),
            &ws,
            &ys,
            super::super::unsigned_attribute(attrs, "in_features")?,
            super::super::unsigned_attribute(attrs, "out_features")?,
        )
        .map_err(|e| e.to_string())
    }

    fn prepared_fingerprint(&self, plan: &PreparedCutlassFp8) -> String {
        implementation_fingerprint(&[
            self.descriptor
                .provider_implementation_fingerprint()
                .as_bytes(),
            &plan.fingerprint_material(),
        ])
    }

    fn encode(
        &self,
        invocation: TensorOperationInvocation<'_, CudaDeviceBuffer>,
    ) -> Result<CudaDeviceCommand, String> {
        let [participant] = invocation.participants() else {
            return Err("grouped FP8 requires one participant".into());
        };
        if invocation.operation().id != *self.descriptor.operation_id()
            || !matches!(participant.work(), NodeWorkContract::Fixed)
            || !invocation.resource_work_shape().resource_work().is_tensor()
        {
            return Err("grouped FP8 requires its fixed tensor contract".into());
        }
        let geometry = self.geometry(participant.bindings(), participant.attributes())?;
        let plan = self.shared.plan(geometry.branch())?;
        if invocation.prepared_implementation_fingerprint()
            != Some(self.prepared_fingerprint(&plan).as_str())
        {
            return Err("grouped FP8 native plan differs from admitted implementation".into());
        }
        let input = weights::plain(
            participant,
            binding(participant.bindings(), ResolvedValueRole::Input, 0)?,
            plan.plan()
                .request
                .input_bytes()
                .map_err(|e| e.to_string())?,
        )?;
        let scratch = weights::scratch(participant, plan.scratch_bytes(), plan.alignment())?;
        let mut regions = vec![input, scratch];
        for i in 0..self.group.count() as u32 {
            let [weight, scale] = weights::retain(
                participant,
                binding(participant.bindings(), ResolvedValueRole::Input, i + 1)?,
            )?;
            let output = weights::plain(
                participant,
                binding(participant.bindings(), ResolvedValueRole::Output, i)?,
                plan.plan()
                    .request
                    .output_bytes()
                    .map_err(|e| e.to_string())?,
            )?;
            regions.extend([weight, scale, output]);
        }
        let owner = Arc::clone(&self.shared.owner);
        if regions
            .iter()
            .any(|region| !Arc::ptr_eq(region.context(), owner.context()))
        {
            return Err("grouped FP8 binding belongs to another CUDA context".into());
        }
        owner
            .context()
            .bind_to_thread()
            .map_err(|e| e.to_string())?;
        plan.validate_group(&spans(&regions, self.group)?)
            .map_err(|e| e.to_string())?;
        let group = self.group;
        CudaDeviceCommand::operation(
            cutlass_fp8::CUTLASS_FP8_OPERATOR,
            regions,
            move |stream, regions| {
                owner
                    .context()
                    .bind_to_thread()
                    .map_err(|e| CudaDeviceRuntimeError::driver("grouped FP8 context bind", e))?;
                let bindings = spans(regions, group).map_err(CudaDeviceRuntimeError::contract)?;
                // All input/weight/scale/output/scratch spans survive partial enqueue errors.
                unsafe { plan.run_group(&bindings, stream.cu_stream().cast()) }
                    .map_err(|e| CudaDeviceRuntimeError::contract(e.to_string()))
            },
        )
        .and_then(|command| {
            command.with_tensor_work(
                geometry.output_elements(),
                geometry.logical_arithmetic_operations(),
                group.count() as u64 + 1,
                0,
            )
        })
        .map_err(|e| e.to_string())
    }
}

fn spans(
    regions: &[CudaBufferRegion],
    group: DenseLinearFp8Group,
) -> Result<Vec<Bindings>, String> {
    if regions.len() != 2 + 3 * group.count() {
        return Err("grouped FP8 retained ranges differ from group".into());
    }
    let span = |r: &CudaBufferRegion| DeviceSpan {
        address: r.device_ptr(),
        bytes: r.length_bytes(),
    };
    Ok(regions[2..]
        .chunks_exact(3)
        .map(|branch| Bindings {
            input: span(&regions[0]),
            scratch: span(&regions[1]),
            weight: span(&branch[0]),
            weight_scale: span(&branch[1]),
            output: span(&branch[2]),
        })
        .collect())
}

impl OperationResourceEstimator for GroupProvider {
    fn descriptor(&self) -> &OperationProviderDescriptor {
        &self.descriptor
    }
    fn estimate_resources(
        &self,
        request: OperationResourceEstimateRequest<'_>,
    ) -> Result<OperationResourceEstimate, VNextError> {
        if request.operation().id != *self.descriptor.operation_id()
            || request.operation().fingerprint()? != self.descriptor.operation_fingerprint()
        {
            return Err(invalid("grouped FP8 estimator received another operation"));
        }
        let g = self
            .geometry(request.values(), request.attributes())
            .map_err(invalid)?;
        let plan = self.shared.plan(g.branch()).map_err(invalid)?;
        // The shared activation and serial GEMM workspace need one scalar-sized lease.
        let scratch = ProviderWorkspaceRequirement::from_formula(
            ProviderWorkspaceSizeFormula::fixed(plan.scratch_bytes())?,
            plan.alignment(),
            ProviderWorkspaceScope::Invocation,
            ProviderWorkspaceReusePolicy::OverwriteBeforeRead,
            DynamicStorageRequirement::contiguous(),
        )?;
        OperationResourceEstimate::new(
            self.descriptor.resource_estimator_id(),
            self.descriptor.resource_estimator_version(),
            self.descriptor
                .resource_estimator_implementation_fingerprint(),
            request.input_fingerprint(),
            16,
            Some(scratch),
            None,
        )
        .with_prepared_implementation_fingerprint(self.prepared_fingerprint(&plan))
    }
}

impl OperationProvider<CudaDeviceRuntime> for GroupProvider {
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
        Err(provider_failure(
            invocation.participants()[0].identity().clone(),
            "cuda.dense.fp8.group.encode",
            "grouped FP8 requires fixed tensor work".into(),
        ))
    }
    fn encode_tensor_selected(
        &self,
        invocation: TensorOperationInvocation<'_, CudaDeviceBuffer>,
    ) -> Result<Option<EncodedDeviceOperation<CudaDeviceCommand>>, OperationFailure> {
        let identity = invocation.participants()[0].identity().clone();
        self.encode(invocation)
            .map(EncodedDeviceOperation::compute)
            .map(Some)
            .map_err(|error| provider_failure(identity, "cuda.dense.fp8.group.encode", error))
    }
}
