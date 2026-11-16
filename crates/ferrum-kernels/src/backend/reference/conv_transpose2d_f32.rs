use super::dense_linear::{
    binding, contiguous_bindings, contiguous_region, implementation_fingerprint,
};
use super::runtime::{
    ReferenceConvTranspose2dF32Launch, ReferenceDeviceBuffer, ReferenceDeviceCommand,
    ReferenceDeviceRuntime, ReferenceDeviceRuntimeError,
};
use ferrum_runtime::*;
use std::collections::{BTreeMap, BTreeSet};
pub const REFERENCE_CONV_TRANSPOSE2D_F32_PROFILE: &str =
    "reference.conv_transpose2d.f32.sequential_mul_add.separate_bias.v1";
pub fn reference_vnext_conv_transpose2d_f32_provider(
    runtime: &ReferenceDeviceRuntime,
) -> Result<Box<dyn OperationProvider<ReferenceDeviceRuntime>>, ReferenceDeviceRuntimeError> {
    let contract = conv_transpose2d_f32_contract().map_err(error)?;
    let cap = CapabilityId::new(CONV_TRANSPOSE2D_F32_CAPABILITY_ID).map_err(error)?;
    if !runtime.descriptor().capabilities.contains(&cap) {
        return Err(ReferenceDeviceRuntimeError::contract(
            "Reference lacks selected F32 transposed convolution capability",
        ));
    }
    let fingerprint = implementation_fingerprint(&[
        REFERENCE_CONV_TRANSPOSE2D_F32_PROFILE.as_bytes(),
        include_bytes!("conv_transpose2d_f32.rs"),
        include_bytes!("runtime/conv_transpose2d_f32.rs"),
    ]);
    let estimator = format!(
        "resource-estimator.reference.{}",
        CONV_TRANSPOSE2D_F32_OPERATION_ID
    );
    let descriptor = OperationProviderDescriptor::new(
        ProviderId::new(format!(
            "provider.reference.{}",
            CONV_TRANSPOSE2D_F32_OPERATION_ID
        ))
        .map_err(error)?,
        contract.descriptor().id.clone(),
        contract.descriptor().fingerprint().map_err(error)?,
        fingerprint.clone(),
        ProviderExecutionSemantics::bitwise_eager_only(),
        ContractVersion::new(1, 0),
        runtime.descriptor().id.clone(),
        BTreeSet::from([cap]),
        BTreeSet::from([WeightFormatId::new("weight-format.safetensors.dense").map_err(error)?]),
        BTreeSet::new(),
        contiguous_bindings(3),
        &estimator,
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
            return Err(invalid(
                "transposed convolution estimator received another operation",
            ));
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
            "transposed convolution requires fixed tensor invocation".into(),
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
            return Err("transposed convolution requires one participant".into());
        };
        if i.operation().id.as_str() != CONV_TRANSPOSE2D_F32_OPERATION_ID
            || !matches!(p.work(), NodeWorkContract::Fixed)
            || !i.resource_work_shape().resource_work().is_tensor()
        {
            return Err("requires selected fixed transposed convolution contract".into());
        }
        let geometry = geometry(p.bindings(), p.attributes())?;
        let inputs = (0..3)
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
        ReferenceDeviceCommand::conv_transpose2d_f32_tensor(ReferenceConvTranspose2dF32Launch {
            inputs,
            output,
            geometry,
        })
        .map_err(|e| e.to_string())
    }
}
fn geometry(
    b: &[ResolvedValueBinding],
    attrs: &BTreeMap<AttributeId, SemanticValue>,
) -> Result<ConvTranspose2dF32Geometry, String> {
    if b.len() != 3 + 1
        || b.iter().any(|v| {
            v.weight()
                .is_some_and(|w| !matches!(w.physical_layout(), PhysicalWeightLayout::Dense { .. }))
        })
    {
        return Err("requires exact Dense F32 transposed convolution operands".into());
    }
    let inputs = (0..3)
        .map(|n| binding(b, ResolvedValueRole::Input, n as u32).map(|b| b.tensor()))
        .collect::<Result<Vec<_>, _>>()?;
    ConvTranspose2dF32Geometry::new(
        inputs[0],
        inputs[1],
        inputs[2],
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
        "reference.conv_transpose2d_f32.encode",
        message.chars().take(2048).collect::<String>(),
        false,
    )
    .expect("bounded failure")
}
