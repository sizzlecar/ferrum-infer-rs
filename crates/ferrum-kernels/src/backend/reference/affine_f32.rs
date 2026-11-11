//! Fixed full-row F32 affine provider, with no intermediate allocation.
use super::dense_linear::{
    binding, contiguous_bindings, contiguous_region, implementation_fingerprint,
};
use super::runtime::{
    ReferenceAffineF32Launch, ReferenceDenseLinearLaunch, ReferenceDenseScalar,
    ReferenceDeviceBuffer, ReferenceDeviceCommand, ReferenceDeviceRuntime,
    ReferenceDeviceRuntimeError,
};
use ferrum_runtime::*;
use std::collections::{BTreeMap, BTreeSet};

pub const REFERENCE_AFFINE_LINEAR_F32_NUMERICAL_PROFILE_ID: &str =
    "reference.affine_linear.f32.dot_then_bias.v1";
const PROVIDER: &str = "provider.reference.affine_linear.f32";
const ESTIMATOR: &str = "resource-estimator.reference.affine_linear.f32";
pub(super) struct ReferenceAffineF32Provider {
    descriptor: OperationProviderDescriptor,
}
impl ReferenceAffineF32Provider {
    pub(super) fn new(
        runtime: &ReferenceDeviceRuntime,
    ) -> Result<Self, ReferenceDeviceRuntimeError> {
        let contract = affine_linear_f32_contract().map_err(contract_error)?;
        let capability =
            CapabilityId::new(AFFINE_LINEAR_F32_CAPABILITY_ID).map_err(contract_error)?;
        if !runtime.descriptor().capabilities.contains(&capability) {
            return Err(ReferenceDeviceRuntimeError::contract(
                "Reference runtime lacks F32 affine capability",
            ));
        }
        let implementation = implementation_fingerprint(&[
            include_bytes!("affine_f32.rs"),
            include_bytes!("runtime/affine_f32.rs"),
            include_bytes!("runtime.rs"),
            REFERENCE_AFFINE_LINEAR_F32_NUMERICAL_PROFILE_ID.as_bytes(),
        ]);
        let estimator =
            implementation_fingerprint(&[ESTIMATOR.as_bytes(), implementation.as_bytes()]);
        let descriptor = OperationProviderDescriptor::new(
            ProviderId::new(PROVIDER).map_err(contract_error)?,
            contract.descriptor().id.clone(),
            contract
                .descriptor()
                .fingerprint()
                .map_err(contract_error)?,
            implementation,
            ProviderExecutionSemantics::bitwise_eager_only(),
            ContractVersion::new(1, 0),
            runtime.descriptor().id.clone(),
            BTreeSet::from([capability]),
            BTreeSet::from([
                WeightFormatId::new(super::composition::REFERENCE_DENSE_SAFETENSORS_FORMAT_ID)
                    .map_err(contract_error)?,
                WeightFormatId::new("weight-format.comfy.convrot.canonical.v1")
                    .map_err(contract_error)?,
            ]),
            BTreeSet::new(),
            contiguous_bindings(3),
            ESTIMATOR,
            ContractVersion::new(1, 0),
            estimator,
        )
        .map_err(contract_error)?
        .with_fixed_tensor_work(true);
        Ok(Self { descriptor })
    }
}
impl OperationResourceEstimator for ReferenceAffineF32Provider {
    fn descriptor(&self) -> &OperationProviderDescriptor {
        &self.descriptor
    }
    fn estimate_resources(
        &self,
        request: OperationResourceEstimateRequest<'_>,
    ) -> Result<OperationResourceEstimate, VNextError> {
        if request.operation().id != *self.descriptor.operation_id()
            || request.operation().fingerprint()? != self.descriptor.operation_fingerprint()
        {
            return Err(invalid("F32 affine estimator received another operation"));
        }
        geometry(request.values(), request.attributes()).map_err(invalid)?;
        Ok(OperationResourceEstimate::new(
            self.descriptor.resource_estimator_id(),
            self.descriptor.resource_estimator_version(),
            self.descriptor
                .resource_estimator_implementation_fingerprint(),
            request.input_fingerprint(),
            16,
            None,
            None,
        ))
    }
}
impl OperationProvider<ReferenceDeviceRuntime> for ReferenceAffineF32Provider {
    fn reusable_execution_topology(
        &self,
        _: ReusableExecutionTopologyRequest<'_>,
    ) -> Result<ReusableExecutionTopology, VNextError> {
        Ok(ReusableExecutionTopology::EagerBoundary)
    }
    fn encode_selected(
        &self,
        invocation: BatchedOperationInvocation<'_, ReferenceDeviceBuffer>,
    ) -> Result<EncodedDeviceOperation<ReferenceDeviceCommand>, OperationFailure> {
        Err(failure(
            invocation.participants()[0].identity().clone(),
            "F32 affine requires fixed tensors".into(),
        ))
    }
    fn encode_tensor_selected(
        &self,
        invocation: TensorOperationInvocation<'_, ReferenceDeviceBuffer>,
    ) -> Result<Option<EncodedDeviceOperation<ReferenceDeviceCommand>>, OperationFailure> {
        let identity = invocation.participants()[0].identity().clone();
        encode(invocation)
            .map(EncodedDeviceOperation::compute)
            .map(Some)
            .map_err(|error| failure(identity, error))
    }
}
fn encode(
    invocation: TensorOperationInvocation<'_, ReferenceDeviceBuffer>,
) -> Result<ReferenceDeviceCommand, String> {
    let [participant] = invocation.participants() else {
        return Err("F32 affine requires one participant".into());
    };
    if invocation.operation().id.as_str() != AFFINE_LINEAR_F32_OPERATION_ID
        || !matches!(participant.work(), NodeWorkContract::Fixed)
        || !invocation.resource_work_shape().resource_work().is_tensor()
    {
        return Err("F32 affine requires its fixed tensor contract".into());
    }
    let shape = geometry(participant.bindings(), participant.attributes())?;
    let region = |role, ordinal| {
        contiguous_region(
            participant,
            binding(participant.bindings(), role, ordinal)?,
            ElementType::F32,
        )
    };
    ReferenceDeviceCommand::affine_f32_tensor(ReferenceAffineF32Launch {
        dense: ReferenceDenseLinearLaunch {
            input: region(ResolvedValueRole::Input, 0)?,
            weight: region(ResolvedValueRole::Input, 1)?,
            output: region(ResolvedValueRole::Output, 0)?,
            rows: shape.rows() as usize,
            in_features: shape.inputs() as usize,
            out_features: shape.outputs() as usize,
            scalar: ReferenceDenseScalar::F32,
        },
        bias: region(ResolvedValueRole::Input, 2)?,
    })
    .map_err(|error| error.to_string())
}
fn geometry(
    bindings: &[ResolvedValueBinding],
    attributes: &BTreeMap<AttributeId, SemanticValue>,
) -> Result<AffineLinearF32Geometry, String> {
    if bindings.len() != 4
        || attributes.len() != 2
        || bindings.iter().any(|binding| {
            binding.weight().is_some_and(|weight| {
                !matches!(weight.physical_layout(), PhysicalWeightLayout::Dense { .. })
            })
        })
    {
        return Err(
            "F32 affine requires three exact dense ports, one output and two shape attributes"
                .into(),
        );
    }
    let attribute = |name: &str| -> Result<u64, String> {
        match attributes
            .iter()
            .find(|(key, _)| key.as_str() == name)
            .map(|(_, value)| value)
        {
            Some(SemanticValue::Unsigned(value)) => Ok(*value),
            _ => Err(format!("F32 affine lacks {name}")),
        }
    };
    AffineLinearF32Geometry::new(
        binding(bindings, ResolvedValueRole::Input, 0)?.tensor(),
        binding(bindings, ResolvedValueRole::Input, 1)?.tensor(),
        binding(bindings, ResolvedValueRole::Input, 2)?.tensor(),
        binding(bindings, ResolvedValueRole::Output, 0)?.tensor(),
        attribute("in_features")?,
        attribute("out_features")?,
    )
    .map_err(|error| error.to_string())
}
fn failure(identity: ExecutionIdentityEnvelope, message: String) -> OperationFailure {
    OperationFailure::new(
        identity,
        ProfilePhase::Forward,
        "reference.affine_f32.encode",
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
fn contract_error(error: VNextError) -> ReferenceDeviceRuntimeError {
    ReferenceDeviceRuntimeError::contract(error.to_string())
}

/// Explicit registration for custom TensorComponent registries.
pub fn reference_vnext_affine_linear_f32_provider(
    runtime: &ReferenceDeviceRuntime,
) -> Result<Box<dyn OperationProvider<ReferenceDeviceRuntime>>, ReferenceDeviceRuntimeError> {
    Ok(Box::new(ReferenceAffineF32Provider::new(runtime)?))
}
