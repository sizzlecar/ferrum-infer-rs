//! Managed F32 LayerNorm over original admitted operands.
use super::dense_linear::{
    binding, contiguous_bindings, contiguous_region, implementation_fingerprint,
};
use super::runtime::{
    ReferenceDeviceBuffer, ReferenceDeviceCommand, ReferenceDeviceRuntime,
    ReferenceDeviceRuntimeError, ReferenceLayerNormF32Launch,
};
use ferrum_runtime::*;
use std::collections::{BTreeMap, BTreeSet};
pub const REFERENCE_LAYER_NORM_F32_NUMERICAL_PROFILE_ID: &str =
    "reference.layer_norm.f32.lane256_two_pass.separate_affine.rn_sqrt_div.v1";
struct Provider {
    descriptor: OperationProviderDescriptor,
}
pub fn reference_vnext_layer_norm_f32_provider(
    runtime: &ReferenceDeviceRuntime,
) -> Result<Box<dyn OperationProvider<ReferenceDeviceRuntime>>, ReferenceDeviceRuntimeError> {
    let c = layer_norm_f32_contract().map_err(contract_error)?;
    let cap = CapabilityId::new(LAYER_NORM_F32_CAPABILITY_ID).map_err(contract_error)?;
    if !runtime.descriptor().capabilities.contains(&cap) {
        return Err(ReferenceDeviceRuntimeError::contract(
            "Reference runtime lacks F32 LayerNorm capability",
        ));
    }
    let fp = implementation_fingerprint(&[
        REFERENCE_LAYER_NORM_F32_NUMERICAL_PROFILE_ID.as_bytes(),
        include_bytes!("layer_norm_f32.rs"),
        include_bytes!("runtime/layer_norm_f32.rs"),
    ]);
    let estimator = "resource-estimator.reference.layer_norm.f32";
    let descriptor = OperationProviderDescriptor::new(
        ProviderId::new("provider.reference.layer_norm.f32").map_err(contract_error)?,
        c.descriptor().id.clone(),
        c.descriptor().fingerprint().map_err(contract_error)?,
        fp.clone(),
        ProviderExecutionSemantics::bitwise_eager_only(),
        ContractVersion::new(1, 0),
        runtime.descriptor().id.clone(),
        BTreeSet::from([cap]),
        BTreeSet::from([
            WeightFormatId::new(super::composition::REFERENCE_DENSE_SAFETENSORS_FORMAT_ID)
                .map_err(contract_error)?,
            WeightFormatId::new("weight-format.comfy.convrot.canonical.v1")
                .map_err(contract_error)?,
        ]),
        BTreeSet::new(),
        contiguous_bindings(3),
        estimator,
        ContractVersion::new(1, 0),
        implementation_fingerprint(&[estimator.as_bytes(), fp.as_bytes()]),
    )
    .map_err(contract_error)?
    .with_fixed_tensor_work(true);
    Ok(Box::new(Provider { descriptor }))
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
            "LayerNorm requires fixed tensor work".into(),
        ))
    }
    fn encode_tensor_selected(
        &self,
        i: TensorOperationInvocation<'_, ReferenceDeviceBuffer>,
    ) -> Result<Option<EncodedDeviceOperation<ReferenceDeviceCommand>>, OperationFailure> {
        let identity = i.participants()[0].identity().clone();
        encode(i)
            .map(EncodedDeviceOperation::compute)
            .map(Some)
            .map_err(|e| failure(identity, e))
    }
}
fn encode(
    i: TensorOperationInvocation<'_, ReferenceDeviceBuffer>,
) -> Result<ReferenceDeviceCommand, String> {
    let [p] = i.participants() else {
        return Err("F32 LayerNorm requires one participant".into());
    };
    if i.operation().id.as_str() != LAYER_NORM_F32_OPERATION_ID
        || !matches!(p.work(), NodeWorkContract::Fixed)
        || !i.resource_work_shape().resource_work().is_tensor()
    {
        return Err("F32 LayerNorm requires its fixed tensor contract".into());
    }
    let g = geometry(p.bindings(), p.attributes())?;
    let region = |role, n| contiguous_region(p, binding(p.bindings(), role, n)?, ElementType::F32);
    ReferenceDeviceCommand::layer_norm_f32_tensor(ReferenceLayerNormF32Launch {
        input: region(ResolvedValueRole::Input, 0)?,
        gamma: region(ResolvedValueRole::Input, 1)?,
        beta: region(ResolvedValueRole::Input, 2)?,
        output: region(ResolvedValueRole::Output, 0)?,
        geometry: g,
    })
    .map_err(|e| e.to_string())
}
fn geometry(
    b: &[ResolvedValueBinding],
    attrs: &BTreeMap<AttributeId, SemanticValue>,
) -> Result<LayerNormF32Geometry, String> {
    if b.len() != 4
        || attrs.len() != 1
        || b.iter().any(|v| {
            v.weight()
                .is_some_and(|w| !matches!(w.physical_layout(), PhysicalWeightLayout::Dense { .. }))
        })
    {
        return Err("F32 LayerNorm requires exact dense operands and epsilon".into());
    }
    let Some(SemanticValue::Rational(epsilon)) = attrs
        .iter()
        .find(|(k, _)| k.as_str() == "epsilon")
        .map(|(_, v)| v)
    else {
        return Err("requires rational epsilon".into());
    };
    let x = |n| binding(b, ResolvedValueRole::Input, n).map(|v| v.tensor());
    LayerNormF32Geometry::new(
        x(0)?,
        x(1)?,
        x(2)?,
        binding(b, ResolvedValueRole::Output, 0)?.tensor(),
        *epsilon,
    )
    .map_err(|e| e.to_string())
}
fn failure(identity: ExecutionIdentityEnvelope, message: String) -> OperationFailure {
    OperationFailure::new(
        identity,
        ProfilePhase::Forward,
        "reference.layer_norm_f32.encode",
        message.chars().take(2048).collect::<String>(),
        false,
    )
    .expect("bounded provider failure")
}
fn contract_error(e: VNextError) -> ReferenceDeviceRuntimeError {
    ReferenceDeviceRuntimeError::contract(e.to_string())
}
fn invalid(reason: impl Into<String>) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: format!("F32 LayerNorm: {}", reason.into()),
    }
}
