//! Independent libm F32 GELU providers; no implicit half input/output boundary.
use super::dense_linear::{
    binding, contiguous_bindings, contiguous_region, implementation_fingerprint,
};
use super::runtime::{
    ReferenceDeviceBuffer, ReferenceDeviceCommand, ReferenceDeviceRuntime,
    ReferenceDeviceRuntimeError, ReferenceGeluF32Launch,
};
use ferrum_runtime::*;
use std::collections::{BTreeMap, BTreeSet};

pub const REFERENCE_GELU_TANH_F32_NUMERICAL_PROFILE_ID: &str =
    "reference.gelu_tanh.f32.libm-0.2.15.separate.gradual.v1";
pub const REFERENCE_GELU_ERF_F32_NUMERICAL_PROFILE_ID: &str =
    "reference.gelu_erf.f32.libm-0.2.15.separate.gradual.v1";
const ESTIMATOR: &str = "resource-estimator.reference.gelu_f32";
pub fn reference_vnext_gelu_tanh_f32_provider(
    runtime: &ReferenceDeviceRuntime,
) -> Result<Box<dyn OperationProvider<ReferenceDeviceRuntime>>, ReferenceDeviceRuntimeError> {
    provider(runtime, GeluF32Variant::Tanh)
}
pub fn reference_vnext_gelu_erf_f32_provider(
    runtime: &ReferenceDeviceRuntime,
) -> Result<Box<dyn OperationProvider<ReferenceDeviceRuntime>>, ReferenceDeviceRuntimeError> {
    provider(runtime, GeluF32Variant::Erf)
}
fn provider(
    runtime: &ReferenceDeviceRuntime,
    variant: GeluF32Variant,
) -> Result<Box<dyn OperationProvider<ReferenceDeviceRuntime>>, ReferenceDeviceRuntimeError> {
    let contract = variant.contract().map_err(contract_error)?;
    let cap = CapabilityId::new(variant.capability_id()).map_err(contract_error)?;
    if !runtime.descriptor().capabilities.contains(&cap) {
        return Err(ReferenceDeviceRuntimeError::contract(
            "Reference lacks F32 GELU capability",
        ));
    }
    let profile = match variant {
        GeluF32Variant::Tanh => REFERENCE_GELU_TANH_F32_NUMERICAL_PROFILE_ID,
        GeluF32Variant::Erf => REFERENCE_GELU_ERF_F32_NUMERICAL_PROFILE_ID,
    };
    let fingerprint = implementation_fingerprint(&[
        profile.as_bytes(),
        include_bytes!("gelu_f32.rs"),
        include_bytes!("runtime/gelu_f32.rs"),
    ]);
    let estimator = implementation_fingerprint(&[ESTIMATOR.as_bytes(), fingerprint.as_bytes()]);
    let descriptor = OperationProviderDescriptor::new(
        ProviderId::new(format!("provider.reference.{}", variant.operation_id()))
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
    Ok(Box::new(Provider {
        descriptor,
        variant,
    }))
}
struct Provider {
    descriptor: OperationProviderDescriptor,
    variant: GeluF32Variant,
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
            return Err(invalid("F32 GELU estimator received another operation"));
        }
        geometry(r.values(), r.attributes())
            .map_err(invalid)?
            .logical_operations(self.variant)?;
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
            "F32 GELU requires fixed tensors".into(),
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
            return Err("F32 GELU requires one tensor participant".into());
        };
        if i.operation().id.as_str() != self.variant.operation_id()
            || !matches!(p.work(), NodeWorkContract::Fixed)
            || !i.resource_work_shape().resource_work().is_tensor()
        {
            return Err("F32 GELU requires its selected fixed tensor contract".into());
        }
        let geometry = geometry(p.bindings(), p.attributes())?;
        ReferenceDeviceCommand::gelu_f32_tensor(ReferenceGeluF32Launch {
            input: contiguous_region(
                p,
                binding(p.bindings(), ResolvedValueRole::Input, 0)?,
                ElementType::F32,
            )?,
            output: contiguous_region(
                p,
                binding(p.bindings(), ResolvedValueRole::Output, 0)?,
                ElementType::F32,
            )?,
            geometry,
            variant: self.variant,
        })
        .map_err(|e| e.to_string())
    }
}
fn geometry(
    values: &[ResolvedValueBinding],
    attributes: &BTreeMap<AttributeId, SemanticValue>,
) -> Result<GeluF32Geometry, String> {
    if values.len() != 2
        || !attributes.is_empty()
        || values.iter().any(|v| {
            v.weight()
                .is_some_and(|w| !matches!(w.physical_layout(), PhysicalWeightLayout::Dense { .. }))
        })
    {
        return Err("F32 GELU requires two exact dense ports and no attributes".into());
    }
    GeluF32Geometry::new(
        binding(values, ResolvedValueRole::Input, 0)?.tensor(),
        binding(values, ResolvedValueRole::Output, 0)?.tensor(),
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
        "reference.gelu_f32.encode",
        message.chars().take(2048).collect::<String>(),
        false,
    )
    .expect("bounded failure")
}
