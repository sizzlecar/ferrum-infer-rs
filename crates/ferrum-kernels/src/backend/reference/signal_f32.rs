use super::dense_linear::{
    binding, contiguous_bindings, contiguous_region, implementation_fingerprint,
};
use super::runtime::{
    ReferenceDeviceBuffer, ReferenceDeviceCommand, ReferenceDeviceRuntime,
    ReferenceDeviceRuntimeError, ReferenceSignalF32Launch,
};
use ferrum_runtime::*;
use std::collections::{BTreeMap, BTreeSet};
pub const REFERENCE_SIGNAL_F32_PROFILE_ID: &str =
    "reference.signal.f32.tap_fma_ascending.libm_separate_rn.v1";
struct Provider {
    descriptor: OperationProviderDescriptor,
    kind: SignalF32Op,
}
pub fn reference_vnext_signal_f32_provider(
    runtime: &ReferenceDeviceRuntime,
    kind: SignalF32Op,
) -> Result<Box<dyn OperationProvider<ReferenceDeviceRuntime>>, ReferenceDeviceRuntimeError> {
    let c = kind.contract().map_err(error)?;
    let cap = CapabilityId::new(kind.capability_id()).map_err(error)?;
    if !runtime.descriptor().capabilities.contains(&cap) {
        return Err(ReferenceDeviceRuntimeError::contract(
            "missing F32 signal capability",
        ));
    }
    let implementation = implementation_fingerprint(&[
        REFERENCE_SIGNAL_F32_PROFILE_ID.as_bytes(),
        kind.operation_id().as_bytes(),
        include_bytes!("signal_f32.rs"),
        include_bytes!("runtime/signal_f32.rs"),
    ]);
    let estimator = format!("resource-estimator.reference.{}", kind.operation_id());
    let fp = implementation_fingerprint(&[estimator.as_bytes(), implementation.as_bytes()]);
    let descriptor = OperationProviderDescriptor::new(
        ProviderId::new(format!("provider.reference.{}", kind.operation_id())).map_err(error)?,
        c.descriptor().id.clone(),
        c.descriptor().fingerprint().map_err(error)?,
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
        contiguous_bindings(kind.input_count() as u32),
        estimator,
        ContractVersion::new(1, 0),
        fp,
    )
    .map_err(error)?
    .with_fixed_tensor_work(true);
    Ok(Box::new(Provider { descriptor, kind }))
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
            return Err(invalid("signal estimator received another operation"));
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
            "signal requires fixed tensor work".into(),
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
            return Err("signal requires one participant".into());
        };
        if i.operation().id.as_str() != self.kind.operation_id()
            || !matches!(p.work(), NodeWorkContract::Fixed)
            || !i.resource_work_shape().resource_work().is_tensor()
        {
            return Err("requires selected fixed signal contract".into());
        }
        let geometry = geometry(self.kind, p.bindings(), p.attributes())?;
        let inputs = (0..self.kind.input_count())
            .map(|n| {
                contiguous_region(
                    p,
                    binding(p.bindings(), ResolvedValueRole::Input, n as u32)?,
                    ElementType::F32,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let output = contiguous_region(
            p,
            binding(p.bindings(), ResolvedValueRole::Output, 0)?,
            ElementType::F32,
        )?;
        ReferenceDeviceCommand::signal_f32_tensor(ReferenceSignalF32Launch {
            inputs,
            output,
            geometry,
        })
        .map_err(|e| e.to_string())
    }
}
fn geometry(
    kind: SignalF32Op,
    b: &[ResolvedValueBinding],
    a: &BTreeMap<AttributeId, SemanticValue>,
) -> Result<SignalF32Geometry, String> {
    if b.len() != kind.input_count() + 1
        || b.iter().any(|v| {
            v.weight()
                .is_some_and(|w| !matches!(w.physical_layout(), PhysicalWeightLayout::Dense { .. }))
        })
    {
        return Err("signal requires exact dense ports".into());
    }
    let inputs = (0..kind.input_count())
        .map(|n| binding(b, ResolvedValueRole::Input, n as u32).map(|v| v.tensor()))
        .collect::<Result<Vec<_>, _>>()?;
    SignalF32Geometry::new(
        kind,
        &inputs,
        binding(b, ResolvedValueRole::Output, 0)?.tensor(),
        a,
    )
    .map_err(|e| e.to_string())
}
fn failure(identity: ExecutionIdentityEnvelope, message: String) -> OperationFailure {
    OperationFailure::new(
        identity,
        ProfilePhase::Forward,
        "reference.signal_f32",
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
