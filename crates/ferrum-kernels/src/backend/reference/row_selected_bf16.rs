use super::dense_linear::{
    binding, contiguous_bindings, contiguous_region, implementation_fingerprint,
};
use super::runtime::{
    ReferenceDeviceBuffer, ReferenceDeviceCommand, ReferenceDeviceRuntime,
    ReferenceDeviceRuntimeError, ReferenceRowSelectedBf16Launch,
};
use ferrum_runtime::*;
use std::collections::{BTreeMap, BTreeSet};
pub const REFERENCE_ROW_SELECTED_MODULATION_BF16_NUMERICAL_PROFILE_ID: &str =
    "reference.row_selected_modulation.bf16.staged_rne.gradual.v1";
pub const REFERENCE_ROW_SELECTED_ADDCMUL_BF16_NUMERICAL_PROFILE_ID: &str =
    "reference.row_selected_addcmul.bf16.separate_rne.gradual.v1";
pub fn reference_vnext_row_selected_bf16_provider(
    runtime: &ReferenceDeviceRuntime,
    kind: RowSelectedBf16Kind,
) -> Result<Box<dyn OperationProvider<ReferenceDeviceRuntime>>, ReferenceDeviceRuntimeError> {
    let contract = match kind {
        RowSelectedBf16Kind::Modulation => row_selected_modulation_bf16_contract(),
        RowSelectedBf16Kind::Addcmul => row_selected_addcmul_bf16_contract(),
    }
    .map_err(error)?;
    let cap = CapabilityId::new(kind.capability_id()).map_err(error)?;
    if !runtime.descriptor().capabilities.contains(&cap) {
        return Err(ReferenceDeviceRuntimeError::contract(
            "Reference lacks selected compact BF16 capability",
        ));
    }
    let (suffix, profile) = match kind {
        RowSelectedBf16Kind::Modulation => (
            "modulation",
            REFERENCE_ROW_SELECTED_MODULATION_BF16_NUMERICAL_PROFILE_ID,
        ),
        RowSelectedBf16Kind::Addcmul => (
            "addcmul",
            REFERENCE_ROW_SELECTED_ADDCMUL_BF16_NUMERICAL_PROFILE_ID,
        ),
    };
    let fingerprint = implementation_fingerprint(&[
        profile.as_bytes(),
        include_bytes!("row_selected_bf16.rs"),
        include_bytes!("runtime/row_selected_bf16.rs"),
    ]);
    let estimator = format!("resource-estimator.reference.row_selected_{suffix}.bf16");
    let descriptor = OperationProviderDescriptor::new(
        ProviderId::new(format!("provider.reference.row_selected_{suffix}.bf16")).map_err(error)?,
        contract.descriptor().id.clone(),
        contract.descriptor().fingerprint().map_err(error)?,
        fingerprint.clone(),
        ProviderExecutionSemantics::bitwise_eager_only(),
        ContractVersion::new(1, 0),
        runtime.descriptor().id.clone(),
        BTreeSet::from([cap]),
        BTreeSet::from([
            WeightFormatId::new("weight-format.safetensors.dense").map_err(error)?,
            WeightFormatId::new("weight-format.comfy.convrot.canonical.v1").map_err(error)?,
        ]),
        BTreeSet::new(),
        contiguous_bindings(kind.input_count() as u32),
        estimator.clone(),
        ContractVersion::new(1, 0),
        implementation_fingerprint(&[estimator.as_bytes(), fingerprint.as_bytes()]),
    )
    .map_err(error)?
    .with_fixed_tensor_work(true);
    Ok(Box::new(Provider { descriptor, kind }))
}
struct Provider {
    descriptor: OperationProviderDescriptor,
    kind: RowSelectedBf16Kind,
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
            return Err(invalid("estimator received another operation"));
        }
        geometry(self.kind, r.values(), r.attributes()).map_err(invalid)?;
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
            "requires fixed tensor invocation".into(),
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
            .map_err(|s| failure(identity, s))
    }
}
impl Provider {
    fn encode(
        &self,
        i: TensorOperationInvocation<'_, ReferenceDeviceBuffer>,
    ) -> Result<ReferenceDeviceCommand, String> {
        let [p] = i.participants() else {
            return Err("requires one fixed tensor participant".into());
        };
        if i.operation().id.as_str() != self.kind.operation_id()
            || !matches!(p.work(), NodeWorkContract::Fixed)
            || !i.resource_work_shape().resource_work().is_tensor()
        {
            return Err("requires exact selected fixed tensor operation".into());
        }
        let g = geometry(self.kind, p.bindings(), p.attributes())?;
        let region = |ordinal| {
            contiguous_region(
                p,
                binding(p.bindings(), ResolvedValueRole::Input, ordinal)?,
                ElementType::Bf16,
            )
        };
        ReferenceDeviceCommand::row_selected_bf16_tensor(ReferenceRowSelectedBf16Launch {
            input: region(0)?,
            parameters: region(1)?,
            update: if self.kind == RowSelectedBf16Kind::Addcmul {
                Some(region(2)?)
            } else {
                None
            },
            output: contiguous_region(
                p,
                binding(p.bindings(), ResolvedValueRole::Output, 0)?,
                ElementType::Bf16,
            )?,
            geometry: g,
        })
        .map_err(|e| e.to_string())
    }
}
fn geometry(
    kind: RowSelectedBf16Kind,
    b: &[ResolvedValueBinding],
    attrs: &BTreeMap<AttributeId, SemanticValue>,
) -> Result<RowSelectedBf16Geometry, String> {
    if b.len() != kind.input_count() + 1
        || b.iter().any(|v| {
            v.weight()
                .is_some_and(|w| !matches!(w.physical_layout(), PhysicalWeightLayout::Dense { .. }))
        })
    {
        return Err("requires exact dense physical operands".into());
    }
    let inputs = (0..kind.input_count())
        .map(|n| binding(b, ResolvedValueRole::Input, n as u32).map(|v| v.tensor()))
        .collect::<Result<Vec<_>, _>>()?;
    RowSelectedBf16Geometry::new(
        kind,
        &inputs,
        binding(b, ResolvedValueRole::Output, 0)?.tensor(),
        attrs,
    )
    .map_err(|e| e.to_string())
}
fn invalid(s: impl Into<String>) -> VNextError {
    VNextError::InvalidExecutionPlan { reason: s.into() }
}
fn error(e: VNextError) -> ReferenceDeviceRuntimeError {
    ReferenceDeviceRuntimeError::contract(e.to_string())
}
fn failure(identity: ExecutionIdentityEnvelope, s: String) -> OperationFailure {
    OperationFailure::new(
        identity,
        ProfilePhase::Forward,
        "reference.row_selected_bf16.encode",
        s.chars().take(2048).collect::<String>(),
        false,
    )
    .expect("bounded failure")
}
