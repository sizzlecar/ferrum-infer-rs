use std::collections::{BTreeMap, BTreeSet};

use ferrum_interfaces::vnext::{
    dense_linear_contract, AttributeId, BatchedOperationInvocation, CapabilityId, ContractVersion,
    DeviceBatchingForm, DeviceRuntime, DynamicStorageRequirement, ElementType,
    EncodedDeviceOperation, ExecutionIdentityEnvelope, OperationContract, OperationFailure,
    OperationProvider, OperationProviderDescriptor, OperationResourceEstimate,
    OperationResourceEstimateRequest, OperationResourceEstimator, ProfilePhase,
    ProviderExecutionSemantics, ProviderId, ProviderStorageBindingRequirement,
    ResolvedValueBinding, ResolvedValueRole, ReusableExecutionTopology,
    ReusableExecutionTopologyRequest, SemanticValue, VNextError, WeightFormatId,
    DENSE_LINEAR_F16_CAPABILITY_ID, DENSE_LINEAR_OPERATION_ID,
};
use sha2::{Digest, Sha256};

use super::composition::REFERENCE_DENSE_SAFETENSORS_FORMAT_ID;
use super::runtime::{
    ReferenceDenseLinearLaunch, ReferenceDeviceBuffer, ReferenceDeviceCommand,
    ReferenceDeviceRuntime, ReferenceDeviceRuntimeError,
};

const PROVIDER_ID: &str = "provider.reference.dense_linear.f16";
const ESTIMATOR_ID: &str = "resource-estimator.reference.dense_linear.f16";
const VALUE_ALIGNMENT_BYTES: u64 = 16;

pub(super) struct ReferenceDenseLinearProvider {
    descriptor: OperationProviderDescriptor,
}

impl ReferenceDenseLinearProvider {
    pub(super) fn new(
        runtime: &ReferenceDeviceRuntime,
    ) -> Result<Self, ReferenceDeviceRuntimeError> {
        let contract = dense_linear_contract().map_err(contract_error)?;
        let capability =
            CapabilityId::new(DENSE_LINEAR_F16_CAPABILITY_ID).map_err(contract_error)?;
        if !runtime.descriptor().capabilities.contains(&capability) {
            return Err(ReferenceDeviceRuntimeError::contract(
                "reference runtime does not advertise dense-linear capability",
            ));
        }
        let provider_fingerprint = implementation_fingerprint(&[
            include_str!("dense_linear.rs").as_bytes(),
            PROVIDER_ID.as_bytes(),
        ]);
        let estimator_fingerprint =
            implementation_fingerprint(&[ESTIMATOR_ID.as_bytes(), provider_fingerprint.as_bytes()]);
        let descriptor = OperationProviderDescriptor::new(
            ProviderId::new(PROVIDER_ID).map_err(contract_error)?,
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
            ESTIMATOR_ID,
            ContractVersion::new(1, 0),
            estimator_fingerprint,
        )
        .map_err(contract_error)?;
        Ok(Self { descriptor })
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
        if request.operation().id.as_str() != DENSE_LINEAR_OPERATION_ID
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
        encode_dense_linear(invocation)
            .map(EncodedDeviceOperation::compute)
            .map_err(|message| provider_failure(identity, message))
    }
}

fn encode_dense_linear(
    invocation: BatchedOperationInvocation<'_, ReferenceDeviceBuffer>,
) -> Result<ReferenceDeviceCommand, String> {
    if invocation.participants().is_empty()
        || invocation.operation().id.as_str() != DENSE_LINEAR_OPERATION_ID
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
        validate_participant_signature(participant, in_features, out_features)?;
    }
    let token_ranges = invocation.participant_token_ranges();
    if token_ranges.len() != invocation.participants().len() {
        return Err("reference dense-linear participant ranges are incomplete".to_owned());
    }
    let weight_binding = binding(first.bindings(), ResolvedValueRole::Input, 1)?;
    let weight = contiguous_region(first, weight_binding)?;
    for participant in &invocation.participants()[1..] {
        let candidate = contiguous_region(
            participant,
            binding(participant.bindings(), ResolvedValueRole::Input, 1)?,
        )?;
        if !weight.same_physical_region(&candidate) {
            return Err("reference dense-linear participants do not share one weight".to_owned());
        }
    }
    let input_shared =
        token_binding_is_shared(&invocation, ResolvedValueRole::Input, 0, ElementType::F16)?;
    let output_shared =
        token_binding_is_shared(&invocation, ResolvedValueRole::Output, 0, ElementType::F16)?;
    let mut launches = Vec::new();
    if input_shared && output_shared {
        let rows = invocation.work_shape().immediate_tokens();
        launches.push(reference_launch(
            shared_token_region(
                &invocation,
                ResolvedValueRole::Input,
                0,
                ElementType::F16,
                rows,
            )?,
            weight,
            shared_token_region(
                &invocation,
                ResolvedValueRole::Output,
                0,
                ElementType::F16,
                rows,
            )?,
            rows,
            in_features,
            out_features,
        )?);
    } else {
        for (participant, token_range) in invocation.participants().iter().zip(token_ranges) {
            let rows = token_range.immediate_tokens();
            let input_start = if input_shared {
                token_range.immediate_token_range().start
            } else {
                token_range.source_token_range().start
            };
            let output_start = if output_shared {
                token_range.immediate_token_range().start
            } else {
                token_range.source_token_range().start
            };
            launches.push(reference_launch(
                contiguous_token_region(
                    participant,
                    binding(participant.bindings(), ResolvedValueRole::Input, 0)?,
                    ElementType::F16,
                    input_start,
                    rows,
                )?,
                weight.clone(),
                contiguous_token_region(
                    participant,
                    binding(participant.bindings(), ResolvedValueRole::Output, 0)?,
                    ElementType::F16,
                    output_start,
                    rows,
                )?,
                rows,
                in_features,
                out_features,
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

fn validate_participant_signature(
    participant: &ferrum_interfaces::vnext::OperationInvocation<'_, ReferenceDeviceBuffer>,
    in_features: usize,
    out_features: usize,
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
) -> Result<ReferenceDenseLinearLaunch, String> {
    let rows = usize::try_from(rows)
        .map_err(|_| "reference dense-linear row count exceeds usize".to_owned())?;
    let expected_input = rows
        .checked_mul(in_features)
        .and_then(|elements| elements.checked_mul(2))
        .ok_or_else(|| "reference dense-linear input size overflows".to_owned())?;
    let expected_weight = out_features
        .checked_mul(in_features)
        .and_then(|elements| elements.checked_mul(2))
        .ok_or_else(|| "reference dense-linear weight size overflows".to_owned())?;
    let expected_output = rows
        .checked_mul(out_features)
        .and_then(|elements| elements.checked_mul(2))
        .ok_or_else(|| "reference dense-linear output size overflows".to_owned())?;
    if input.length_bytes() != expected_input
        || weight.length_bytes() != expected_weight
        || output.length_bytes() != expected_output
        || input.element_type() != ElementType::F16
        || weight.element_type() != ElementType::F16
        || output.element_type() != ElementType::F16
    {
        return Err("reference dense-linear physical ranges differ from runtime work".to_owned());
    }
    Ok(ReferenceDenseLinearLaunch {
        input,
        weight,
        output,
        rows,
        in_features,
        out_features,
    })
}

fn contiguous_region(
    participant: &ferrum_interfaces::vnext::OperationInvocation<'_, ReferenceDeviceBuffer>,
    binding: &ResolvedValueBinding,
) -> Result<super::runtime::ReferenceBufferRegion, String> {
    let [component] = binding.storage().components() else {
        return Err("reference operation requires one storage component per value".to_owned());
    };
    if component.element_type() != ElementType::F16 {
        return Err("reference dense-linear binding is not F16".to_owned());
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
    participant: &ferrum_interfaces::vnext::OperationInvocation<'_, ReferenceDeviceBuffer>,
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
    participant: &ferrum_interfaces::vnext::OperationInvocation<'_, ReferenceDeviceBuffer>,
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

fn token_binding_is_shared(
    invocation: &BatchedOperationInvocation<'_, ReferenceDeviceBuffer>,
    role: ResolvedValueRole,
    ordinal: u32,
    element_type: ElementType,
) -> Result<bool, String> {
    let first = &invocation.participants()[0];
    let region = contiguous_token_region(
        first,
        binding(first.bindings(), role, ordinal)?,
        element_type,
        0,
        1,
    )?;
    for participant in &invocation.participants()[1..] {
        let candidate = contiguous_token_region(
            participant,
            binding(participant.bindings(), role, ordinal)?,
            element_type,
            0,
            1,
        )?;
        if !region.same_physical_region(&candidate) {
            return Ok(false);
        }
    }
    Ok(true)
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
