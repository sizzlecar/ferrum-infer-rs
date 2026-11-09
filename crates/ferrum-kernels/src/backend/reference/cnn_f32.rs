use super::dense_linear::{
    binding, contiguous_bindings, contiguous_region, implementation_fingerprint,
};
use super::runtime::{
    ReferenceCnnF32Launch, ReferenceDeviceBuffer, ReferenceDeviceCommand, ReferenceDeviceRuntime,
    ReferenceDeviceRuntimeError,
};
use ferrum_runtime::*;
use std::collections::{BTreeMap, BTreeSet};
pub fn reference_cnn_f32_numerical_profile(kind: F32CnnOp) -> &'static str {
    match kind {
        F32CnnOp::Conv2d => "reference.conv2d.f32.sequential_mul_add.separate_bias.v1",
        F32CnnOp::GroupNormSilu => "reference.group_norm_silu.nchw.f32.tree256.two_pass.libm.v1",
        F32CnnOp::ResidualAdd => "reference.plain_residual_add.nchw.f32.separate_rn.v1",
    }
}
pub fn reference_vnext_cnn_f32_provider(
    runtime: &ReferenceDeviceRuntime,
    kind: F32CnnOp,
) -> Result<Box<dyn OperationProvider<ReferenceDeviceRuntime>>, ReferenceDeviceRuntimeError> {
    let contract = kind.contract().map_err(error)?;
    let cap = CapabilityId::new(kind.capability_id()).map_err(error)?;
    if !runtime.descriptor().capabilities.contains(&cap) {
        return Err(ReferenceDeviceRuntimeError::contract(
            "Reference lacks selected F32 CNN capability",
        ));
    }
    let fingerprint = implementation_fingerprint(&[
        reference_cnn_f32_numerical_profile(kind).as_bytes(),
        include_bytes!("cnn_f32.rs"),
        include_bytes!("runtime/cnn_f32.rs"),
    ]);
    let estimator = format!("resource-estimator.reference.{}", kind.operation_id());
    let descriptor = OperationProviderDescriptor::new(
        ProviderId::new(format!("provider.reference.{}", kind.operation_id())).map_err(error)?,
        contract.descriptor().id.clone(),
        contract.descriptor().fingerprint().map_err(error)?,
        fingerprint.clone(),
        ProviderExecutionSemantics::bitwise_eager_only(),
        ContractVersion::new(1, 0),
        runtime.descriptor().id.clone(),
        BTreeSet::from([cap]),
        BTreeSet::from([WeightFormatId::new("weight-format.safetensors.dense").map_err(error)?]),
        BTreeSet::new(),
        contiguous_bindings(kind.input_count() as u32),
        &estimator,
        ContractVersion::new(1, 0),
        implementation_fingerprint(&[estimator.as_bytes(), fingerprint.as_bytes()]),
    )
    .map_err(error)?
    .with_fixed_tensor_work(true);
    Ok(Box::new(Provider { descriptor, kind }))
}
struct Provider {
    descriptor: OperationProviderDescriptor,
    kind: F32CnnOp,
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
            return Err(invalid("CNN estimator received another operation"));
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
            "CNN requires fixed tensor invocation".into(),
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
            return Err("CNN requires one participant".into());
        };
        if i.operation().id.as_str() != self.kind.operation_id()
            || !matches!(p.work(), NodeWorkContract::Fixed)
            || !i.resource_work_shape().resource_work().is_tensor()
        {
            return Err("requires selected fixed CNN contract".into());
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
            .collect::<Result<_, _>>()?;
        let output = contiguous_region(
            p,
            binding(p.bindings(), ResolvedValueRole::Output, 0)?,
            ElementType::F32,
        )?;
        ReferenceDeviceCommand::cnn_f32_tensor(ReferenceCnnF32Launch {
            inputs,
            output,
            geometry,
        })
        .map_err(|e| e.to_string())
    }
}
fn geometry(
    kind: F32CnnOp,
    b: &[ResolvedValueBinding],
    attrs: &BTreeMap<AttributeId, SemanticValue>,
) -> Result<F32CnnGeometry, String> {
    if b.len() != kind.input_count() + 1
        || b.iter().any(|v| {
            v.weight()
                .is_some_and(|w| !matches!(w.physical_layout(), PhysicalWeightLayout::Dense { .. }))
        })
    {
        return Err("requires exact Dense F32 CNN operands".into());
    }
    let inputs = (0..kind.input_count())
        .map(|n| binding(b, ResolvedValueRole::Input, n as u32).map(|b| b.tensor()))
        .collect::<Result<Vec<_>, _>>()?;
    F32CnnGeometry::new(
        kind,
        &inputs,
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
        "reference.cnn_f32.encode",
        message.chars().take(2048).collect::<String>(),
        false,
    )
    .expect("bounded failure")
}
