//! F32 channel arithmetic and payload-preserving grouped column transport.
use super::dense_linear::{
    binding, contiguous_bindings, contiguous_region, implementation_fingerprint,
};
use super::runtime::{
    ReferenceDeviceBuffer, ReferenceDeviceCommand, ReferenceDeviceRuntime,
    ReferenceDeviceRuntimeError, ReferenceF32ChannelwiseLaunch,
};
use ferrum_runtime::*;
use std::collections::{BTreeMap, BTreeSet};
pub fn reference_channelwise_f32_numerical_profile(kind: F32ChannelwiseOp) -> &'static str {
    match kind {
        F32ChannelwiseOp::GroupedColumnSlice => "reference.grouped_column_slice.f32.exact_bits.v1",
        F32ChannelwiseOp::BiasAdd => "reference.channel_bias_add.f32.rne.gradual.v1",
        F32ChannelwiseOp::ScaledResidual => {
            "reference.channel_scaled_residual.f32.single_fma.gradual.v1"
        }
        F32ChannelwiseOp::NanToNum => "reference.nan_to_num.f32.exact_finite_bits.v1",
    }
}
pub fn reference_vnext_channelwise_f32_provider(
    runtime: &ReferenceDeviceRuntime,
    kind: F32ChannelwiseOp,
) -> Result<Box<dyn OperationProvider<ReferenceDeviceRuntime>>, ReferenceDeviceRuntimeError> {
    let contract = kind.contract().map_err(error)?;
    let cap = CapabilityId::new(kind.capability_id()).map_err(error)?;
    if !runtime.descriptor().capabilities.contains(&cap) {
        return Err(ReferenceDeviceRuntimeError::contract(
            "Reference lacks selected F32 channelwise capability",
        ));
    }
    let fingerprint = implementation_fingerprint(&[
        reference_channelwise_f32_numerical_profile(kind).as_bytes(),
        include_bytes!("channelwise_f32.rs"),
        include_bytes!("runtime/channelwise_f32.rs"),
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
        BTreeSet::from([
            WeightFormatId::new("weight-format.safetensors.dense").map_err(error)?,
            WeightFormatId::new("weight-format.comfy.convrot.canonical.v1").map_err(error)?,
        ]),
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
    kind: F32ChannelwiseOp,
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
            return Err(invalid("channelwise estimator received another operation"));
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
            "F32 channelwise requires fixed tensor invocation".into(),
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
            return Err("F32 channelwise requires one tensor participant".into());
        };
        if i.operation().id.as_str() != self.kind.operation_id()
            || !matches!(p.work(), NodeWorkContract::Fixed)
            || !i.resource_work_shape().resource_work().is_tensor()
        {
            return Err("requires selected fixed channelwise contract".into());
        }
        let geometry = geometry(self.kind, p.bindings(), p.attributes())?;
        let inputs = (0..self.kind.input_count())
            .map(|index| {
                contiguous_region(
                    p,
                    binding(p.bindings(), ResolvedValueRole::Input, index as u32)?,
                    ElementType::F32,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let output = contiguous_region(
            p,
            binding(p.bindings(), ResolvedValueRole::Output, 0)?,
            ElementType::F32,
        )?;
        ReferenceDeviceCommand::channelwise_f32_tensor(ReferenceF32ChannelwiseLaunch {
            inputs,
            output,
            geometry,
        })
        .map_err(|e| e.to_string())
    }
}
fn geometry(
    kind: F32ChannelwiseOp,
    b: &[ResolvedValueBinding],
    attrs: &BTreeMap<AttributeId, SemanticValue>,
) -> Result<F32ChannelwiseGeometry, String> {
    if b.len() != kind.input_count() + 1
        || b.iter().any(|v| {
            v.weight()
                .is_some_and(|w| !matches!(w.physical_layout(), PhysicalWeightLayout::Dense { .. }))
        })
    {
        return Err("F32 channelwise requires exactly its original dense ports".into());
    }
    let inputs = (0..kind.input_count())
        .map(|index| binding(b, ResolvedValueRole::Input, index as u32).map(|v| v.tensor()))
        .collect::<Result<Vec<_>, _>>()?;
    F32ChannelwiseGeometry::new(
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
        "reference.channelwise_f32.encode",
        message.chars().take(2048).collect::<String>(),
        false,
    )
    .expect("bounded failure")
}
