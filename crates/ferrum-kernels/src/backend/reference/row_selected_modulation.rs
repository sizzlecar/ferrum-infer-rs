use super::dense_linear::{
    binding, contiguous_bindings, contiguous_region, implementation_fingerprint,
};
use super::runtime::{
    ReferenceDeviceBuffer, ReferenceDeviceCommand, ReferenceDeviceRuntime,
    ReferenceDeviceRuntimeError, ReferenceRowSelectedModulationLaunch,
};
use ferrum_runtime::*;
use std::collections::{BTreeMap, BTreeSet};
pub const REFERENCE_ROW_SELECTED_MODULATION_BF16_F32_NUMERICAL_PROFILE_ID: &str =
    "reference.row_selected_modulation.bf16_f32.separate_rne.gradual.v1";
pub fn reference_vnext_row_selected_modulation_bf16_f32_provider(
    runtime: &ReferenceDeviceRuntime,
) -> Result<Box<dyn OperationProvider<ReferenceDeviceRuntime>>, ReferenceDeviceRuntimeError> {
    let contract = row_selected_modulation_bf16_f32_contract().map_err(error)?;
    let cap = CapabilityId::new(ROW_SELECTED_MODULATION_BF16_F32_CAPABILITY_ID).map_err(error)?;
    if !runtime.descriptor().capabilities.contains(&cap) {
        return Err(ReferenceDeviceRuntimeError::contract(
            "Reference lacks row-selected BF16/F32 modulation",
        ));
    }
    let fingerprint = implementation_fingerprint(&[
        REFERENCE_ROW_SELECTED_MODULATION_BF16_F32_NUMERICAL_PROFILE_ID.as_bytes(),
        include_bytes!("row_selected_modulation.rs"),
        include_bytes!("runtime/row_selected_modulation.rs"),
    ]);
    let estimator = "resource-estimator.reference.row_selected_modulation.bf16_f32";
    let descriptor = OperationProviderDescriptor::new(
        ProviderId::new("provider.reference.row_selected_modulation.bf16_f32").map_err(error)?,
        contract.descriptor().id.clone(),
        contract.descriptor().fingerprint().map_err(error)?,
        fingerprint.clone(),
        ProviderExecutionSemantics::bitwise_eager_only(),
        ContractVersion::new(1, 0),
        runtime.descriptor().id.clone(),
        BTreeSet::from([cap]),
        BTreeSet::from([WeightFormatId::new("weight-format.safetensors.dense").map_err(error)?]),
        BTreeSet::new(),
        contiguous_bindings(2),
        estimator,
        ContractVersion::new(1, 0),
        implementation_fingerprint(&[estimator.as_bytes(), fingerprint.as_bytes()]),
    )
    .map_err(error)?
    .with_fixed_tensor_work(true);
    Ok(Box::new(Provider { descriptor }))
}
struct Provider {
    descriptor: OperationProviderDescriptor,
}
impl OperationResourceEstimator for Provider {
    fn descriptor(&self) -> &OperationProviderDescriptor {
        &self.descriptor
    }
    fn estimate_resources(
        &self,
        r: OperationResourceEstimateRequest<'_>,
    ) -> Result<OperationResourceEstimate, VNextError> {
        if r.operation().id != *self.descriptor.operation_id()
            || r.operation().fingerprint()? != self.descriptor.operation_fingerprint()
        {
            return Err(invalid("row-selected estimator received another operation"));
        }
        geometry(r.values(), r.attributes()).map_err(invalid)?;
        Ok(OperationResourceEstimate::new(
            self.descriptor.resource_estimator_id(),
            self.descriptor.resource_estimator_version(),
            self.descriptor
                .resource_estimator_implementation_fingerprint(),
            r.input_fingerprint(),
            16,
            None,
            None,
        ))
    }
}
impl OperationProvider<ReferenceDeviceRuntime> for Provider {
    fn reusable_execution_topology(
        &self,
        _: ReusableExecutionTopologyRequest<'_>,
    ) -> Result<ReusableExecutionTopology, VNextError> {
        Ok(ReusableExecutionTopology::EagerBoundary)
    }
    fn encode_selected(
        &self,
        i: BatchedOperationInvocation<'_, ReferenceDeviceBuffer>,
    ) -> Result<EncodedDeviceOperation<ReferenceDeviceCommand>, OperationFailure> {
        Err(failure(
            i.participants()[0].identity().clone(),
            "requires fixed row-selected tensor invocation".into(),
        ))
    }
    fn encode_tensor_selected(
        &self,
        i: TensorOperationInvocation<'_, ReferenceDeviceBuffer>,
    ) -> Result<Option<EncodedDeviceOperation<ReferenceDeviceCommand>>, OperationFailure> {
        let identity = i.participants()[0].identity().clone();
        self.encode(i)
            .map(EncodedDeviceOperation::compute)
            .map(Some)
            .map_err(|e| failure(identity, e))
    }
}
impl Provider {
    fn encode(
        &self,
        i: TensorOperationInvocation<'_, ReferenceDeviceBuffer>,
    ) -> Result<ReferenceDeviceCommand, String> {
        let [p] = i.participants() else {
            return Err("requires one row-selected participant".into());
        };
        if i.operation().id.as_str() != ROW_SELECTED_MODULATION_BF16_F32_OPERATION_ID
            || !matches!(p.work(), NodeWorkContract::Fixed)
            || !i.resource_work_shape().resource_work().is_tensor()
        {
            return Err("requires selected fixed row modulation contract".into());
        }
        let geometry = geometry(p.bindings(), p.attributes())?;
        ReferenceDeviceCommand::row_selected_modulation_tensor(
            ReferenceRowSelectedModulationLaunch {
                input: contiguous_region(
                    p,
                    binding(p.bindings(), ResolvedValueRole::Input, 0)?,
                    ElementType::Bf16,
                )?,
                parameters: contiguous_region(
                    p,
                    binding(p.bindings(), ResolvedValueRole::Input, 1)?,
                    ElementType::F32,
                )?,
                output: contiguous_region(
                    p,
                    binding(p.bindings(), ResolvedValueRole::Output, 0)?,
                    ElementType::F32,
                )?,
                geometry,
            },
        )
        .map_err(|e| e.to_string())
    }
}
fn geometry(
    b: &[ResolvedValueBinding],
    attrs: &BTreeMap<AttributeId, SemanticValue>,
) -> Result<RowSelectedModulationBf16F32Geometry, String> {
    if b.len() != 3
        || b.iter().any(|v| {
            v.weight()
                .is_some_and(|w| !matches!(w.physical_layout(), PhysicalWeightLayout::Dense { .. }))
        })
    {
        return Err("row modulation requires exact dense operands".into());
    }
    RowSelectedModulationBf16F32Geometry::new(
        binding(b, ResolvedValueRole::Input, 0)?.tensor(),
        binding(b, ResolvedValueRole::Input, 1)?.tensor(),
        binding(b, ResolvedValueRole::Output, 0)?.tensor(),
        attrs,
    )
    .map_err(|e| e.to_string())
}
fn invalid(reason: impl Into<String>) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: reason.into(),
    }
}
fn error(e: VNextError) -> ReferenceDeviceRuntimeError {
    ReferenceDeviceRuntimeError::contract(e.to_string())
}
fn failure(identity: ExecutionIdentityEnvelope, message: String) -> OperationFailure {
    OperationFailure::new(
        identity,
        ProfilePhase::Forward,
        "reference.row_selected_modulation.encode",
        message.chars().take(2048).collect::<String>(),
        false,
    )
    .expect("bounded failure")
}
