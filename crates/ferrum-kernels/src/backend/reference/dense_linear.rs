use std::collections::{BTreeMap, BTreeSet};

use ferrum_runtime::{
    dense_linear_bf16_contract, dense_linear_contract, AttributeId, BatchedOperationInvocation,
    CapabilityId, ContractVersion, DeviceBatchingForm, DeviceRuntime, DynamicStorageRequirement,
    ElementType, EncodedDeviceOperation, ExecutionIdentityEnvelope, InvocationWorkProjection,
    NodeWorkContract, OperationContract, OperationFailure, OperationProvider,
    OperationProviderDescriptor, OperationResourceEstimate, OperationResourceEstimateRequest,
    OperationResourceEstimator, ProfilePhase, ProviderExecutionSemantics, ProviderId,
    ProviderStorageBindingRequirement, ResolvedValueBinding, ResolvedValueRole,
    ReusableExecutionTopology, ReusableExecutionTopologyRequest, SemanticValue,
    TensorOperationInvocation, VNextError, WeightFormatId, DENSE_LINEAR_BF16_CAPABILITY_ID,
    DENSE_LINEAR_BF16_OPERATION_ID, DENSE_LINEAR_F16_CAPABILITY_ID, DENSE_LINEAR_OPERATION_ID,
};
use sha2::{Digest, Sha256};

use super::composition::REFERENCE_DENSE_SAFETENSORS_FORMAT_ID;
use super::runtime::{
    ReferenceDenseLinearLaunch, ReferenceDenseScalar, ReferenceDeviceBuffer,
    ReferenceDeviceCommand, ReferenceDeviceRuntime, ReferenceDeviceRuntimeError,
};

const VALUE_ALIGNMENT_BYTES: u64 = 16;

fn operation_id(scalar: ReferenceDenseScalar) -> &'static str {
    match scalar {
        ReferenceDenseScalar::F16 => DENSE_LINEAR_OPERATION_ID,
        ReferenceDenseScalar::Bf16 => DENSE_LINEAR_BF16_OPERATION_ID,
    }
}

pub(super) struct ReferenceDenseLinearProvider {
    descriptor: OperationProviderDescriptor,
    scalar: ReferenceDenseScalar,
}

impl ReferenceDenseLinearProvider {
    pub(super) fn new(
        runtime: &ReferenceDeviceRuntime,
    ) -> Result<Self, ReferenceDeviceRuntimeError> {
        Self::new_for(runtime, ReferenceDenseScalar::F16)
    }

    pub(super) fn new_bf16(
        runtime: &ReferenceDeviceRuntime,
    ) -> Result<Self, ReferenceDeviceRuntimeError> {
        Self::new_for(runtime, ReferenceDenseScalar::Bf16)
    }

    fn new_for(
        runtime: &ReferenceDeviceRuntime,
        scalar: ReferenceDenseScalar,
    ) -> Result<Self, ReferenceDeviceRuntimeError> {
        let (contract, capability, provider_id, estimator_id) = match scalar {
            ReferenceDenseScalar::F16 => (
                dense_linear_contract(),
                DENSE_LINEAR_F16_CAPABILITY_ID,
                "provider.reference.dense_linear.f16",
                "resource-estimator.reference.dense_linear.f16",
            ),
            ReferenceDenseScalar::Bf16 => (
                dense_linear_bf16_contract(),
                DENSE_LINEAR_BF16_CAPABILITY_ID,
                "provider.reference.dense_linear.bf16",
                "resource-estimator.reference.dense_linear.bf16",
            ),
        };
        let contract = contract.map_err(contract_error)?;
        let capability = CapabilityId::new(capability).map_err(contract_error)?;
        if !runtime.descriptor().capabilities.contains(&capability) {
            return Err(ReferenceDeviceRuntimeError::contract(
                "reference runtime does not advertise dense-linear capability",
            ));
        }
        let provider_fingerprint = implementation_fingerprint(&[
            include_str!("dense_linear.rs").as_bytes(),
            provider_id.as_bytes(),
        ]);
        let estimator_fingerprint =
            implementation_fingerprint(&[estimator_id.as_bytes(), provider_fingerprint.as_bytes()]);
        let descriptor = OperationProviderDescriptor::new(
            ProviderId::new(provider_id).map_err(contract_error)?,
            contract.descriptor().id.clone(),
            contract
                .descriptor()
                .fingerprint()
                .map_err(contract_error)?,
            provider_fingerprint,
            ProviderExecutionSemantics::bitwise_eager_only(),
            contract.descriptor().version,
            runtime.descriptor().id.clone(),
            BTreeSet::from([capability]),
            BTreeSet::from([WeightFormatId::new(REFERENCE_DENSE_SAFETENSORS_FORMAT_ID)
                .map_err(contract_error)?]),
            BTreeSet::new(),
            contiguous_bindings(2),
            estimator_id,
            ContractVersion::new(1, 0),
            estimator_fingerprint,
        )
        .map_err(contract_error)?
        .with_fixed_tensor_work(true);
        Ok(Self { descriptor, scalar })
    }
}

impl OperationResourceEstimator for ReferenceDenseLinearProvider {
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
            return Err(invalid_plan(
                "reference dense-linear estimator received another operation",
            ));
        }
        Ok(OperationResourceEstimate::new(
            self.descriptor.resource_estimator_id(),
            self.descriptor.resource_estimator_version(),
            self.descriptor
                .resource_estimator_implementation_fingerprint(),
            request.input_fingerprint(),
            VALUE_ALIGNMENT_BYTES,
            None,
            None,
        ))
    }
}

impl OperationProvider<ReferenceDeviceRuntime> for ReferenceDenseLinearProvider {
    fn reusable_execution_topology(
        &self,
        _request: ReusableExecutionTopologyRequest<'_>,
    ) -> Result<ReusableExecutionTopology, VNextError> {
        Ok(ReusableExecutionTopology::EagerBoundary)
    }

    fn encode_selected(
        &self,
        invocation: BatchedOperationInvocation<'_, ReferenceDeviceBuffer>,
    ) -> Result<EncodedDeviceOperation<ReferenceDeviceCommand>, OperationFailure> {
        let identity = invocation.participants()[0].identity().clone();
        encode_dense_linear(invocation, self.scalar)
            .map(EncodedDeviceOperation::compute)
            .map_err(|message| provider_failure(identity, message))
    }

    fn encode_tensor_selected(
        &self,
        invocation: TensorOperationInvocation<'_, ReferenceDeviceBuffer>,
    ) -> Result<Option<EncodedDeviceOperation<ReferenceDeviceCommand>>, OperationFailure> {
        let identity = invocation.participants()[0].identity().clone();
        encode_tensor_dense_linear(invocation, self.scalar)
            .map(EncodedDeviceOperation::compute)
            .map(Some)
            .map_err(|message| provider_failure(identity, message))
    }
}

fn encode_tensor_dense_linear(
    invocation: TensorOperationInvocation<'_, ReferenceDeviceBuffer>,
    scalar: ReferenceDenseScalar,
) -> Result<ReferenceDeviceCommand, String> {
    let [participant] = invocation.participants() else {
        return Err("reference tensor dense-linear requires one participant".to_owned());
    };
    if invocation.operation().id.as_str() != operation_id(scalar)
        || !matches!(participant.work(), NodeWorkContract::Fixed)
        || !invocation.resource_work_shape().resource_work().is_tensor()
    {
        return Err("reference tensor dense-linear requires its fixed tensor contract".to_owned());
    }
    let in_features = dimension(participant.attributes(), "in_features")?;
    let out_features = dimension(participant.attributes(), "out_features")?;
    validate_participant_signature(participant, in_features, out_features, scalar)?;
    let input = binding(participant.bindings(), ResolvedValueRole::Input, 0)?;
    let weight = binding(participant.bindings(), ResolvedValueRole::Input, 1)?;
    let output = binding(participant.bindings(), ResolvedValueRole::Output, 0)?;
    let launch = reference_launch(
        contiguous_region(participant, input, scalar.element_type())?,
        contiguous_region(participant, weight, scalar.element_type())?,
        contiguous_region(participant, output, scalar.element_type())?,
        input.tensor().dimensions()[0],
        in_features,
        out_features,
        scalar,
    )?;
    ReferenceDeviceCommand::dense_tensor(launch).map_err(|error| error.to_string())
}

fn encode_dense_linear(
    invocation: BatchedOperationInvocation<'_, ReferenceDeviceBuffer>,
    scalar: ReferenceDenseScalar,
) -> Result<ReferenceDeviceCommand, String> {
    if invocation.participants().is_empty()
        || invocation.operation().id.as_str() != operation_id(scalar)
    {
        return Err(
            "reference dense-linear provider received another or empty operation".to_owned(),
        );
    }
    let first = &invocation.participants()[0];
    let in_features = dimension(first.attributes(), "in_features")?;
    let out_features = dimension(first.attributes(), "out_features")?;
    for participant in invocation.participants() {
        if dimension(participant.attributes(), "in_features")? != in_features
            || dimension(participant.attributes(), "out_features")? != out_features
        {
            return Err("reference dense-linear participant attributes disagree".to_owned());
        }
        validate_participant_signature(participant, in_features, out_features, scalar)?;
    }
    let token_ranges = invocation.participant_token_ranges();
    if token_ranges.len() != invocation.participants().len() {
        return Err("reference dense-linear participant ranges are incomplete".to_owned());
    }
    let weight_binding = binding(first.bindings(), ResolvedValueRole::Input, 1)?;
    let weight = contiguous_region(first, weight_binding, scalar.element_type())?;
    for participant in &invocation.participants()[1..] {
        let candidate = contiguous_region(
            participant,
            binding(participant.bindings(), ResolvedValueRole::Input, 1)?,
            scalar.element_type(),
        )?;
        if !weight.same_physical_region(&candidate) {
            return Err("reference dense-linear participants do not share one weight".to_owned());
        }
    }
    let input_packed = token_binding_is_packed(&invocation, ResolvedValueRole::Input, 0)?;
    let output_packed = token_binding_is_packed(&invocation, ResolvedValueRole::Output, 0)?;
    let mut launches = Vec::new();
    if input_packed && output_packed {
        let rows = invocation.work_shape().immediate_tokens();
        launches.push(reference_launch(
            shared_token_region(
                &invocation,
                ResolvedValueRole::Input,
                0,
                scalar.element_type(),
                rows,
            )?,
            weight,
            shared_token_region(
                &invocation,
                ResolvedValueRole::Output,
                0,
                scalar.element_type(),
                rows,
            )?,
            rows,
            in_features,
            out_features,
            scalar,
        )?);
    } else {
        for (participant, token_range) in invocation.participants().iter().zip(token_ranges) {
            let rows = token_range.immediate_tokens();
            let input_start = if input_packed {
                token_range.immediate_token_range().start
            } else {
                token_range.source_token_range().start
            };
            let output_start = if output_packed {
                token_range.immediate_token_range().start
            } else {
                token_range.source_token_range().start
            };
            launches.push(reference_launch(
                contiguous_token_region(
                    participant,
                    binding(participant.bindings(), ResolvedValueRole::Input, 0)?,
                    scalar.element_type(),
                    input_start,
                    rows,
                )?,
                weight.clone(),
                contiguous_token_region(
                    participant,
                    binding(participant.bindings(), ResolvedValueRole::Output, 0)?,
                    scalar.element_type(),
                    output_start,
                    rows,
                )?,
                rows,
                in_features,
                out_features,
                scalar,
            )?);
        }
    }
    let participant_count = u32::try_from(invocation.participants().len())
        .map_err(|_| "reference dense-linear participant count exceeds u32".to_owned())?;
    let token_count = invocation.work_shape().immediate_tokens();
    let batching_form = if participant_count == 1 {
        DeviceBatchingForm::Scalar
    } else if launches.len() == 1 {
        DeviceBatchingForm::Packed
    } else {
        DeviceBatchingForm::ParticipantLoop
    };
    ReferenceDeviceCommand::dense_linear(launches, batching_form, participant_count, token_count)
        .map_err(|error| error.to_string())
}

fn validate_participant_signature<'a, W: InvocationWorkProjection<'a>>(
    participant: &ferrum_runtime::OperationInvocation<'a, ReferenceDeviceBuffer, W>,
    in_features: usize,
    out_features: usize,
    scalar: ReferenceDenseScalar,
) -> Result<(), String> {
    let input = binding(participant.bindings(), ResolvedValueRole::Input, 0)?;
    let weight = binding(participant.bindings(), ResolvedValueRole::Input, 1)?;
    let output = binding(participant.bindings(), ResolvedValueRole::Output, 0)?;
    let input_dimensions = input.tensor().dimensions();
    let output_dimensions = output.tensor().dimensions();
    if input_dimensions.len() != 2
        || output_dimensions.len() != 2
        || input_dimensions[0] != output_dimensions[0]
        || input_dimensions[1] != in_features as u64
        || weight.tensor().dimensions() != [out_features as u64, in_features as u64]
        || output_dimensions[1] != out_features as u64
        || [input, weight, output]
            .iter()
            .any(|binding| binding.tensor().element_type() != scalar.element_type())
    {
        return Err(
            "reference dense-linear invocation differs from its resolved signature".to_owned(),
        );
    }
    Ok(())
}

fn reference_launch(
    input: super::runtime::ReferenceBufferRegion,
    weight: super::runtime::ReferenceBufferRegion,
    output: super::runtime::ReferenceBufferRegion,
    rows: u64,
    in_features: usize,
    out_features: usize,
    scalar: ReferenceDenseScalar,
) -> Result<ReferenceDenseLinearLaunch, String> {
    let rows = usize::try_from(rows)
        .map_err(|_| "reference dense-linear row count exceeds usize".to_owned())?;
    let launch = ReferenceDenseLinearLaunch {
        input,
        weight,
        output,
        rows,
        in_features,
        out_features,
        scalar,
    };
    launch.validate().map_err(|error| error.to_string())?;
    Ok(launch)
}

fn contiguous_region<'a, W: InvocationWorkProjection<'a>>(
    participant: &ferrum_runtime::OperationInvocation<'a, ReferenceDeviceBuffer, W>,
    binding: &ResolvedValueBinding,
    element_type: ElementType,
) -> Result<super::runtime::ReferenceBufferRegion, String> {
    let [component] = binding.storage().components() else {
        return Err("reference operation requires one storage component per value".to_owned());
    };
    if component.element_type() != element_type {
        return Err("reference dense-linear binding differs from its scalar contract".to_owned());
    }
    let view = participant
        .views()
        .iter()
        .find(|view| view.resource_id() == component.resource_id())
        .ok_or_else(|| "reference operation value has no resource view".to_owned())?;
    let translated = view
        .translate(component.offset_bytes(), component.length_bytes())
        .map_err(|error| error.to_string())?;
    let mut physical = translated.iter();
    let region = physical
        .next()
        .ok_or_else(|| "reference operation translated to no physical region".to_owned())?;
    if physical.next().is_some() {
        return Err("reference operation requires contiguous physical storage".to_owned());
    }
    let (buffer, range, retention) = region.buffer_and_physical_range();
    buffer
        .retained_region(range, retention)
        .map_err(|error| error.to_string())
}

fn contiguous_token_region(
    participant: &ferrum_runtime::OperationInvocation<'_, ReferenceDeviceBuffer>,
    binding: &ResolvedValueBinding,
    element_type: ElementType,
    token_start: u64,
    token_count: u64,
) -> Result<super::runtime::ReferenceBufferRegion, String> {
    let [component] = binding.storage().components() else {
        return Err("reference operation requires one storage component per value".to_owned());
    };
    let dimensions = binding.tensor().dimensions();
    let (canonical_extent, component_base_offset) = match participant
        .work()
        .token_projection(binding.role(), binding.ordinal())
    {
        Some(projection)
            if projection.axis() == 0
                && projection.rank() as usize == dimensions.len()
                && dimensions.first() == Some(&projection.canonical_extent())
                && component.offset_bytes() == 0 =>
        {
            (projection.canonical_extent(), 0)
        }
        Some(_) => {
            return Err(
                "reference token projection is not a canonical leading-axis tensor".to_owned(),
            );
        }
        None => (
            dimensions
                .first()
                .copied()
                .filter(|extent| *extent > 0)
                .ok_or_else(|| {
                    "reference canonical token binding has no leading extent".to_owned()
                })?,
            component.offset_bytes(),
        ),
    };
    if component.length_bytes() % canonical_extent != 0 {
        return Err("reference token binding has a fractional token stride".to_owned());
    }
    let bytes_per_token = component.length_bytes() / canonical_extent;
    let relative_offset = token_start
        .checked_mul(bytes_per_token)
        .ok_or_else(|| "reference token region offset overflows".to_owned())?;
    let offset = component_base_offset
        .checked_add(relative_offset)
        .ok_or_else(|| "reference token region base offset overflows".to_owned())?;
    let length = token_count
        .checked_mul(bytes_per_token)
        .ok_or_else(|| "reference token region length overflows".to_owned())?;
    contiguous_region_range(participant, binding, element_type, offset, length)
}

fn contiguous_region_range(
    participant: &ferrum_runtime::OperationInvocation<'_, ReferenceDeviceBuffer>,
    binding: &ResolvedValueBinding,
    element_type: ElementType,
    logical_offset_bytes: u64,
    logical_length_bytes: u64,
) -> Result<super::runtime::ReferenceBufferRegion, String> {
    let [component] = binding.storage().components() else {
        return Err("reference operation requires one storage component per value".to_owned());
    };
    if component.element_type() != element_type {
        return Err(
            "reference operation storage element type differs from its contract".to_owned(),
        );
    }
    let view = participant
        .views()
        .iter()
        .find(|view| view.resource_id() == component.resource_id())
        .ok_or_else(|| "reference operation value has no resource view".to_owned())?;
    let translated = view
        .translate(logical_offset_bytes, logical_length_bytes)
        .map_err(|error| error.to_string())?;
    let mut physical = translated.iter();
    let region = physical
        .next()
        .ok_or_else(|| "reference operation translated to no physical region".to_owned())?;
    if physical.next().is_some() {
        return Err("reference operation requires contiguous physical storage".to_owned());
    }
    let (buffer, range, retention) = region.buffer_and_physical_range();
    let retained = buffer
        .retained_region(range, retention)
        .map_err(|error| error.to_string())?;
    if retained.element_type() != element_type
        || u64::try_from(retained.length_bytes()).ok() != Some(logical_length_bytes)
    {
        return Err("reference operation retained the wrong physical region".to_owned());
    }
    Ok(retained)
}

fn shared_token_region(
    invocation: &BatchedOperationInvocation<'_, ReferenceDeviceBuffer>,
    role: ResolvedValueRole,
    ordinal: u32,
    element_type: ElementType,
    tokens: u64,
) -> Result<super::runtime::ReferenceBufferRegion, String> {
    let first = &invocation.participants()[0];
    let region = contiguous_token_region(
        first,
        binding(first.bindings(), role, ordinal)?,
        element_type,
        0,
        tokens,
    )?;
    for participant in &invocation.participants()[1..] {
        let candidate = contiguous_token_region(
            participant,
            binding(participant.bindings(), role, ordinal)?,
            element_type,
            0,
            tokens,
        )?;
        if !region.same_physical_region(&candidate) {
            return Err(format!(
                "reference batch {role:?} binding {ordinal} is not one shared token region"
            ));
        }
    }
    Ok(region)
}

fn token_binding_is_packed(
    invocation: &BatchedOperationInvocation<'_, ReferenceDeviceBuffer>,
    role: ResolvedValueRole,
    ordinal: u32,
) -> Result<bool, String> {
    invocation
        .binding_uses_packed_batch_coordinates(role, ordinal)
        .map_err(|error| error.to_string())
}

fn binding(
    bindings: &[ResolvedValueBinding],
    role: ResolvedValueRole,
    ordinal: u32,
) -> Result<&ResolvedValueBinding, String> {
    bindings
        .iter()
        .find(|binding| binding.role() == role && binding.ordinal() == ordinal)
        .ok_or_else(|| format!("reference operation lacks {role:?} binding {ordinal}"))
}

fn dimension(
    attributes: &BTreeMap<AttributeId, SemanticValue>,
    name: &str,
) -> Result<usize, String> {
    match attributes
        .iter()
        .find(|(attribute, _)| attribute.as_str() == name)
        .map(|(_, value)| value)
    {
        Some(SemanticValue::Unsigned(value)) if *value > 0 => usize::try_from(*value)
            .map_err(|_| format!("reference dense-linear attribute {name} exceeds usize")),
        _ => Err(format!(
            "reference dense-linear lacks positive attribute {name}"
        )),
    }
}

fn contiguous_bindings(input_count: u32) -> Vec<ProviderStorageBindingRequirement> {
    (0..input_count)
        .map(|ordinal| {
            ProviderStorageBindingRequirement::new(
                ResolvedValueRole::Input,
                ordinal,
                DynamicStorageRequirement::contiguous(),
            )
        })
        .chain(std::iter::once(ProviderStorageBindingRequirement::new(
            ResolvedValueRole::Output,
            0,
            DynamicStorageRequirement::contiguous(),
        )))
        .collect()
}

fn provider_failure(identity: ExecutionIdentityEnvelope, message: String) -> OperationFailure {
    OperationFailure::new(
        identity,
        ProfilePhase::Forward,
        "reference.dense_linear.encode",
        message.chars().take(2048).collect::<String>(),
        false,
    )
    .expect("reference provider failure metadata is bounded and static")
}

fn contract_error(error: VNextError) -> ReferenceDeviceRuntimeError {
    ReferenceDeviceRuntimeError::contract(error.to_string())
}

fn invalid_plan(reason: impl Into<String>) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: reason.into(),
    }
}

pub(super) fn implementation_fingerprint(parts: &[&[u8]]) -> String {
    let mut digest = Sha256::new();
    for part in parts {
        digest.update((part.len() as u64).to_le_bytes());
        digest.update(part);
    }
    format!("{:x}", digest.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::reference::{
        reference_vnext_dense_linear_bf16_provider, reference_vnext_dense_linear_provider,
        ReferenceVNextComposition,
    };
    use ferrum_runtime::DeviceId;

    #[test]
    fn precision_providers_have_independent_contracts_capabilities_and_estimators() {
        let composition = ReferenceVNextComposition::create(
            DeviceId::new("device.reference.precision-contracts").unwrap(),
        )
        .unwrap();
        let f16 = reference_vnext_dense_linear_provider(composition.runtime()).unwrap();
        let bf16 = reference_vnext_dense_linear_bf16_provider(composition.runtime()).unwrap();
        let a = f16.descriptor();
        let b = bf16.descriptor();
        assert_eq!(a.operation_id().as_str(), DENSE_LINEAR_OPERATION_ID);
        assert_eq!(b.operation_id().as_str(), DENSE_LINEAR_BF16_OPERATION_ID);
        assert_eq!(
            a.operation_fingerprint(),
            dense_linear_contract()
                .unwrap()
                .descriptor()
                .fingerprint()
                .unwrap()
        );
        assert_eq!(
            b.operation_fingerprint(),
            dense_linear_bf16_contract()
                .unwrap()
                .descriptor()
                .fingerprint()
                .unwrap()
        );
        assert_ne!(a.provider_id(), b.provider_id());
        assert_ne!(a.resource_estimator_id(), b.resource_estimator_id());
        assert_ne!(
            a.provider_implementation_fingerprint(),
            b.provider_implementation_fingerprint()
        );
        assert_eq!(
            a.capabilities(),
            &BTreeSet::from([CapabilityId::new(DENSE_LINEAR_F16_CAPABILITY_ID).unwrap()])
        );
        assert_eq!(
            b.capabilities(),
            &BTreeSet::from([CapabilityId::new(DENSE_LINEAR_BF16_CAPABILITY_ID).unwrap()])
        );
        assert!(a.supports_fixed_tensor_work());
        assert!(b.supports_fixed_tensor_work());
    }
}
