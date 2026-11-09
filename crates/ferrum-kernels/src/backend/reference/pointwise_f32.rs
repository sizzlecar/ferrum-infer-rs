//! Full-F32 addition and ordinary SiLU*up, with no activation quantization.
use super::dense_linear::{
    binding, contiguous_bindings, contiguous_region, implementation_fingerprint,
};
use super::runtime::{
    ReferenceDeviceBuffer, ReferenceDeviceCommand, ReferenceDeviceRuntime,
    ReferenceDeviceRuntimeError, ReferenceF32PointwiseLaunch,
};
use ferrum_runtime::*;
use std::collections::{BTreeMap, BTreeSet};

pub const REFERENCE_PLAIN_RESIDUAL_ADD_F32_NUMERICAL_PROFILE_ID: &str =
    "reference.plain_residual_add.f32.rne.gradual.v1";
pub const REFERENCE_SILU_MUL_F32_NUMERICAL_PROFILE_ID: &str =
    "reference.silu_mul.f32.libm-0.2.15.separate.gradual.v1";
const ESTIMATOR: &str = "resource-estimator.reference.pointwise_f32";

pub fn reference_vnext_plain_residual_add_f32_provider(
    runtime: &ReferenceDeviceRuntime,
) -> Result<Box<dyn OperationProvider<ReferenceDeviceRuntime>>, ReferenceDeviceRuntimeError> {
    provider(runtime, F32PointwiseOp::ResidualAdd)
}
pub fn reference_vnext_silu_mul_f32_provider(
    runtime: &ReferenceDeviceRuntime,
) -> Result<Box<dyn OperationProvider<ReferenceDeviceRuntime>>, ReferenceDeviceRuntimeError> {
    provider(runtime, F32PointwiseOp::SiluMul)
}
fn provider(
    runtime: &ReferenceDeviceRuntime,
    kind: F32PointwiseOp,
) -> Result<Box<dyn OperationProvider<ReferenceDeviceRuntime>>, ReferenceDeviceRuntimeError> {
    let contract = kind.contract().map_err(contract_error)?;
    let cap = CapabilityId::new(kind.capability_id()).map_err(contract_error)?;
    if !runtime.descriptor().capabilities.contains(&cap) {
        return Err(ReferenceDeviceRuntimeError::contract(
            "Reference lacks F32 pointwise capability",
        ));
    }
    let profile = match kind {
        F32PointwiseOp::ResidualAdd => REFERENCE_PLAIN_RESIDUAL_ADD_F32_NUMERICAL_PROFILE_ID,
        F32PointwiseOp::SiluMul => REFERENCE_SILU_MUL_F32_NUMERICAL_PROFILE_ID,
    };
    let fingerprint = implementation_fingerprint(&[
        profile.as_bytes(),
        include_bytes!("pointwise_f32.rs"),
        include_bytes!("runtime/pointwise_f32.rs"),
    ]);
    let estimator = implementation_fingerprint(&[ESTIMATOR.as_bytes(), fingerprint.as_bytes()]);
    let descriptor = OperationProviderDescriptor::new(
        ProviderId::new(format!("provider.reference.{}", kind.operation_id()))
            .map_err(contract_error)?,
        contract.descriptor().id.clone(),
        contract
            .descriptor()
            .fingerprint()
            .map_err(contract_error)?,
        fingerprint,
        ProviderExecutionSemantics::bitwise_eager_only(),
        ContractVersion::new(1, 0),
        runtime.descriptor().id.clone(),
        BTreeSet::from([cap]),
        BTreeSet::from([WeightFormatId::new(
            super::composition::REFERENCE_DENSE_SAFETENSORS_FORMAT_ID,
        )
        .map_err(contract_error)?]),
        BTreeSet::new(),
        contiguous_bindings(2),
        ESTIMATOR,
        ContractVersion::new(1, 0),
        estimator,
    )
    .map_err(contract_error)?
    .with_fixed_tensor_work(true);
    Ok(Box::new(Provider { descriptor, kind }))
}
struct Provider {
    descriptor: OperationProviderDescriptor,
    kind: F32PointwiseOp,
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
                "F32 pointwise estimator received another operation",
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
            "F32 pointwise requires fixed tensors".into(),
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
            return Err("F32 pointwise requires one tensor participant".into());
        };
        if i.operation().id.as_str() != self.kind.operation_id()
            || !matches!(p.work(), NodeWorkContract::Fixed)
            || !i.resource_work_shape().resource_work().is_tensor()
        {
            return Err("F32 pointwise requires its selected fixed tensor contract".into());
        }
        let geometry = geometry(p.bindings(), p.attributes())?;
        ReferenceDeviceCommand::f32_pointwise_tensor(ReferenceF32PointwiseLaunch {
            input: contiguous_region(
                p,
                binding(p.bindings(), ResolvedValueRole::Input, 0)?,
                ElementType::F32,
            )?,
            update: contiguous_region(
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
            kind: self.kind,
        })
        .map_err(|e| e.to_string())
    }
}
fn geometry(
    bindings: &[ResolvedValueBinding],
    attrs: &BTreeMap<AttributeId, SemanticValue>,
) -> Result<F32SameShapeGeometry, String> {
    if bindings.len() != 3
        || !attrs.is_empty()
        || bindings.iter().any(|b| {
            b.weight()
                .is_some_and(|w| !matches!(w.physical_layout(), PhysicalWeightLayout::Dense { .. }))
        })
    {
        return Err("F32 pointwise requires three dense ports and no attributes".into());
    }
    F32SameShapeGeometry::new(
        binding(bindings, ResolvedValueRole::Input, 0)?.tensor(),
        binding(bindings, ResolvedValueRole::Input, 1)?.tensor(),
        binding(bindings, ResolvedValueRole::Output, 0)?.tensor(),
    )
    .map_err(|e| e.to_string())
}
fn contract_error(e: VNextError) -> ReferenceDeviceRuntimeError {
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
        "reference.pointwise_f32.encode",
        message.chars().take(2048).collect::<String>(),
        false,
    )
    .expect("bounded failure")
}
