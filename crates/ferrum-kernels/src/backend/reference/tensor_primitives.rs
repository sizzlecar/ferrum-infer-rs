//! Portable, explicitly selected tensor primitive profiles. Payloads are read
//! directly from admitted regions; bounded metadata is the only owned vector.
use super::composition::REFERENCE_DENSE_SAFETENSORS_FORMAT_ID;
use super::dense_linear::{
    binding, contiguous_bindings, contiguous_region, implementation_fingerprint,
};
use super::runtime::{
    ReferenceDeviceBuffer, ReferenceDeviceCommand, ReferenceDeviceRuntime,
    ReferenceDeviceRuntimeError, ReferenceTensorPrimitiveLaunch,
};
use ferrum_runtime::*;
use std::collections::{BTreeMap, BTreeSet};

pub fn reference_tensor_primitive_bf16_profile(kind: TensorPrimitiveBf16) -> String {
    format!("reference.tensor_primitive.{}.bf16.v1", kind.name())
}
pub(super) struct ReferenceTensorPrimitiveProvider {
    kind: TensorPrimitiveBf16,
    descriptor: OperationProviderDescriptor,
}
impl ReferenceTensorPrimitiveProvider {
    pub(super) fn new(
        runtime: &ReferenceDeviceRuntime,
        kind: TensorPrimitiveBf16,
    ) -> Result<Self, ReferenceDeviceRuntimeError> {
        let contract = kind.contract().map_err(error)?;
        let cap = CapabilityId::new(kind.capability_id()).map_err(error)?;
        if !runtime.descriptor().capabilities.contains(&cap) {
            return Err(ReferenceDeviceRuntimeError::contract(
                "missing tensor primitive capability",
            ));
        }
        let provider = format!("provider.reference.tensor_primitive.{}.bf16", kind.name());
        let estimator = format!(
            "resource-estimator.reference.tensor_primitive.{}.bf16",
            kind.name()
        );
        let numerical = super::transcendental::numerical_implementation_fingerprint();
        let implementation = implementation_fingerprint(&[
            include_bytes!("tensor_primitives.rs"),
            include_bytes!("runtime/tensor_primitives.rs"),
            reference_tensor_primitive_bf16_profile(kind).as_bytes(),
            numerical.as_bytes(),
        ]);
        let estimator_fp =
            implementation_fingerprint(&[estimator.as_bytes(), implementation.as_bytes()]);
        let descriptor = OperationProviderDescriptor::new(
            ProviderId::new(provider).map_err(error)?,
            contract.descriptor().id.clone(),
            contract.descriptor().fingerprint().map_err(error)?,
            implementation,
            ProviderExecutionSemantics::bitwise_eager_only(),
            ContractVersion::new(1, 0),
            runtime.descriptor().id.clone(),
            BTreeSet::from([cap]),
            BTreeSet::from([
                WeightFormatId::new(REFERENCE_DENSE_SAFETENSORS_FORMAT_ID).map_err(error)?
            ]),
            BTreeSet::new(),
            contiguous_bindings(kind.input_count()),
            estimator,
            ContractVersion::new(1, 0),
            estimator_fp,
        )
        .map_err(error)?
        .with_fixed_tensor_work(true);
        Ok(Self { kind, descriptor })
    }
}
impl OperationResourceEstimator for ReferenceTensorPrimitiveProvider {
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
            return Err(invalid("estimator operation differs"));
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
impl OperationProvider<ReferenceDeviceRuntime> for ReferenceTensorPrimitiveProvider {
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
            "requires fixed tensor work".into(),
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
impl ReferenceTensorPrimitiveProvider {
    fn encode(
        &self,
        i: TensorOperationInvocation<'_, ReferenceDeviceBuffer>,
    ) -> Result<ReferenceDeviceCommand, String> {
        let [p] = i.participants() else {
            return Err("requires one participant".into());
        };
        if i.operation().id.as_str() != self.kind.operation_id()
            || !matches!(p.work(), NodeWorkContract::Fixed)
            || !i.resource_work_shape().resource_work().is_tensor()
        {
            return Err("requires its own fixed tensor contract".into());
        }
        let shape = geometry(self.kind, p.bindings(), p.attributes())?;
        let inputs = (0..self.kind.input_count())
            .map(|ordinal| {
                let dtype = if self.kind == TensorPrimitiveBf16::HalfRotary && ordinal > 0 {
                    ElementType::F32
                } else {
                    ElementType::Bf16
                };
                contiguous_region(
                    p,
                    binding(p.bindings(), ResolvedValueRole::Input, ordinal)?,
                    dtype,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let output = contiguous_region(
            p,
            binding(p.bindings(), ResolvedValueRole::Output, 0)?,
            ElementType::Bf16,
        )?;
        ReferenceDeviceCommand::tensor_primitive(ReferenceTensorPrimitiveLaunch {
            inputs,
            output,
            shape,
        })
        .map_err(|e| e.to_string())
    }
}
fn geometry(
    kind: TensorPrimitiveBf16,
    values: &[ResolvedValueBinding],
    attrs: &BTreeMap<AttributeId, SemanticValue>,
) -> Result<TensorPrimitiveBf16Geometry, String> {
    let g = TensorPrimitiveBf16Geometry::new(kind, values, attrs).map_err(|e| e.to_string())?;
    for value in values {
        if value
            .weight()
            .is_some_and(|w| !matches!(w.physical_layout(), PhysicalWeightLayout::Dense { .. }))
        {
            return Err("requires dense physical weight storage".into());
        }
    }
    super::runtime::tensor_primitives::arithmetic(&g).map_err(|e| e.to_string())?;
    Ok(g)
}
fn error(e: VNextError) -> ReferenceDeviceRuntimeError {
    ReferenceDeviceRuntimeError::contract(e.to_string())
}
fn invalid(reason: impl Into<String>) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: reason.into(),
    }
}
fn failure(identity: ExecutionIdentityEnvelope, message: String) -> OperationFailure {
    OperationFailure::new(
        identity,
        ProfilePhase::Forward,
        "reference.tensor_primitive.encode",
        message.chars().take(2048).collect::<String>(),
        false,
    )
    .expect("bounded provider error")
}
