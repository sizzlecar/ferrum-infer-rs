use super::dense_linear::{
    binding, contiguous_bindings, contiguous_region, implementation_fingerprint,
};
use super::runtime::{
    ReferenceBatchedF32Launch, ReferenceDeviceBuffer, ReferenceDeviceCommand,
    ReferenceDeviceRuntime, ReferenceDeviceRuntimeError,
};
use ferrum_runtime::*;
use std::collections::{BTreeMap, BTreeSet};

pub const REFERENCE_BATCHED_F32_PROFILE_ID: &str =
    "reference.batched_f32.fma_ascending.softmax_exp_ascending.v1";
struct Provider {
    descriptor: OperationProviderDescriptor,
    op: BatchedF32Op,
}
pub fn reference_vnext_batched_f32_provider(
    runtime: &ReferenceDeviceRuntime,
    op: BatchedF32Op,
) -> Result<Box<dyn OperationProvider<ReferenceDeviceRuntime>>, ReferenceDeviceRuntimeError> {
    let contract = op.contract().map_err(error)?;
    let cap = CapabilityId::new(op.capability_id()).map_err(error)?;
    if !runtime.descriptor().capabilities.contains(&cap) {
        return Err(ReferenceDeviceRuntimeError::contract(
            "missing batched F32 capability",
        ));
    }
    let name = format!(
        "provider.reference.{}",
        op.operation_id().trim_start_matches("operation.")
    );
    let estimator = format!(
        "resource-estimator.reference.{}",
        op.operation_id().trim_start_matches("operation.")
    );
    let implementation = implementation_fingerprint(&[
        REFERENCE_BATCHED_F32_PROFILE_ID.as_bytes(),
        name.as_bytes(),
        include_bytes!("batched_f32.rs"),
        include_bytes!("runtime/batched_f32.rs"),
    ]);
    let estimate = implementation_fingerprint(&[estimator.as_bytes(), implementation.as_bytes()]);
    let descriptor = OperationProviderDescriptor::new(
        ProviderId::new(name).map_err(error)?,
        contract.descriptor().id.clone(),
        contract.descriptor().fingerprint().map_err(error)?,
        implementation,
        ProviderExecutionSemantics::bitwise_eager_only(),
        ContractVersion::new(1, 0),
        runtime.descriptor().id.clone(),
        BTreeSet::from([cap]),
        BTreeSet::from([WeightFormatId::new(
            super::composition::REFERENCE_DENSE_SAFETENSORS_FORMAT_ID,
        )
        .map_err(error)?]),
        BTreeSet::new(),
        contiguous_bindings(op.input_count()),
        estimator,
        ContractVersion::new(1, 0),
        estimate,
    )
    .map_err(error)?
    .with_fixed_tensor_work(true);
    Ok(Box::new(Provider { descriptor, op }))
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
        geometry(self.op, r.values(), r.attributes()).map_err(invalid)?;
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
impl Provider {
    fn encode(
        &self,
        i: TensorOperationInvocation<'_, ReferenceDeviceBuffer>,
    ) -> Result<ReferenceDeviceCommand, String> {
        let [p] = i.participants() else {
            return Err("requires one fixed participant".into());
        };
        if i.operation().id.as_str() != self.op.operation_id()
            || !matches!(p.work(), NodeWorkContract::Fixed)
            || !i.resource_work_shape().resource_work().is_tensor()
        {
            return Err("requires its fixed tensor contract".into());
        }
        let geometry = geometry(self.op, p.bindings(), p.attributes())?;
        let inputs = (0..self.op.input_count())
            .map(|n| {
                contiguous_region(
                    p,
                    binding(p.bindings(), ResolvedValueRole::Input, n)?,
                    ElementType::F32,
                )
            })
            .collect::<Result<_, String>>()?;
        ReferenceDeviceCommand::batched_f32_tensor(ReferenceBatchedF32Launch {
            inputs,
            output: contiguous_region(
                p,
                binding(p.bindings(), ResolvedValueRole::Output, 0)?,
                ElementType::F32,
            )?,
            geometry,
        })
        .map_err(|e| e.to_string())
    }
}
fn geometry(
    op: BatchedF32Op,
    b: &[ResolvedValueBinding],
    attrs: &BTreeMap<AttributeId, SemanticValue>,
) -> Result<BatchedF32Geometry, String> {
    if b.len() != op.input_count() as usize + 1
        || b.iter().any(|v| {
            v.weight()
                .is_some_and(|w| !matches!(w.physical_layout(), PhysicalWeightLayout::Dense { .. }))
        })
    {
        return Err("requires exact dense ports".into());
    }
    let inputs = (0..op.input_count())
        .map(|n| binding(b, ResolvedValueRole::Input, n).map(|v| v.tensor()))
        .collect::<Result<Vec<_>, String>>()?;
    BatchedF32Geometry::new(
        op,
        &inputs,
        binding(b, ResolvedValueRole::Output, 0)?.tensor(),
        attrs,
    )
    .map_err(|e| e.to_string())
}
fn failure(identity: ExecutionIdentityEnvelope, message: String) -> OperationFailure {
    OperationFailure::new(
        identity,
        ProfilePhase::Forward,
        "reference.batched_f32.encode",
        message.chars().take(2048).collect::<String>(),
        false,
    )
    .expect("bounded failure")
}
fn invalid(reason: impl Into<String>) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: reason.into(),
    }
}
fn error(e: VNextError) -> ReferenceDeviceRuntimeError {
    ReferenceDeviceRuntimeError::contract(e.to_string())
}
