use super::{
    canonical_fingerprint, invalid_plan, is_canonical_sha256, AttributeId, BTreeMap, BTreeSet,
    BufferUsage, CapabilityId, ContractVersion, Digest, ElementType, NodeId, OperationDescriptor,
    OperationId, PreparedModelFamily, ProgramNode, ProgramTensorSpec, ProviderId,
    QuantizationFormatId, ResolvedTensorSpec, ResolvedValueBinding, ResolvedValueRole,
    ResolvedValueStorage, ResourceId, SemanticValue, Serialize, Sha256, TensorAccess, VNextError,
    WeightEncoding, WeightFormatId, WeightId,
};

#[derive(Serialize)]
pub(super) struct ProviderEstimatorInputMaterial<'a> {
    pub(super) prepared_family_fingerprint: &'a str,
    pub(super) operation_fingerprint: &'a str,
    pub(super) node_id: &'a NodeId,
    pub(super) operation_id: &'a OperationId,
    pub(super) operation_version: ContractVersion,
    pub(super) attributes: &'a BTreeMap<AttributeId, SemanticValue>,
    pub(super) provider_id: &'a ProviderId,
    pub(super) values: &'a [ResolvedValueBinding],
    pub(super) required_capabilities: &'a BTreeSet<CapabilityId>,
    pub(super) required_weight_formats: &'a BTreeSet<WeightFormatId>,
    pub(super) required_quantization_formats: &'a BTreeSet<QuantizationFormatId>,
}

/// Canonical input signature a provider estimator must bind into its trusted
/// result. Physical bindings remain part of the signature, but never come from
/// a serialized execution plan during validation.
pub(crate) fn provider_resource_estimator_input_fingerprint(
    family: &PreparedModelFamily,
    operation: &OperationDescriptor,
    node: &ProgramNode,
    provider_id: &ProviderId,
    values: &[ResolvedValueBinding],
    required_capabilities: &BTreeSet<CapabilityId>,
) -> Result<String, VNextError> {
    let (required_weight_formats, required_quantization_formats) =
        node_weight_requirements(family, values)?;
    let effective_required_capabilities = operation
        .provider
        .required_capabilities
        .union(required_capabilities)
        .cloned()
        .collect::<BTreeSet<_>>();
    let prepared_family_fingerprint = family.fingerprint()?;
    let operation_fingerprint = operation.fingerprint()?;
    canonical_fingerprint(
        &ProviderEstimatorInputMaterial {
            prepared_family_fingerprint: &prepared_family_fingerprint,
            operation_fingerprint: &operation_fingerprint,
            node_id: &node.id,
            operation_id: &node.operation_id,
            operation_version: node.required_version,
            attributes: &node.attributes,
            provider_id,
            values,
            required_capabilities: &effective_required_capabilities,
            required_weight_formats: &required_weight_formats,
            required_quantization_formats: &required_quantization_formats,
        },
        "fingerprint provider resource estimator input",
    )
}

pub(super) fn validate_program_bindings(
    node: &ProgramNode,
    bindings: &[ResolvedValueBinding],
) -> Result<(), VNextError> {
    let expected = node
        .inputs
        .iter()
        .enumerate()
        .map(|(ordinal, value)| (ResolvedValueRole::Input, ordinal as u32, value))
        .chain(
            node.outputs
                .iter()
                .enumerate()
                .map(|(ordinal, value)| (ResolvedValueRole::Output, ordinal as u32, value)),
        )
        .collect::<Vec<_>>();
    if expected.len() != bindings.len()
        || expected.iter().zip(bindings).any(|(expected, actual)| {
            expected.0 != actual.role()
                || expected.1 != actual.ordinal()
                || expected.2 != actual.value_id()
        })
    {
        return Err(invalid_plan(format!(
            "node `{}` bindings do not match semantic program values",
            node.id
        )));
    }
    Ok(())
}

pub(super) fn validate_semantic_binding(
    family: &PreparedModelFamily,
    binding: &ResolvedValueBinding,
) -> Result<(), VNextError> {
    if let Some(weight) = family
        .program()
        .weights()
        .iter()
        .find(|weight| &weight.value_id == binding.value_id())
    {
        if binding.usage() != BufferUsage::Weights || binding.access() != TensorAccess::Read {
            return Err(invalid_plan(format!(
                "weight value `{}` is not backed by immutable weight memory",
                binding.value_id()
            )));
        }
        validate_program_tensor(&weight.tensor, binding.tensor(), "weight")?;
        validate_weight_storage(family, &weight.weight_id, binding.storage())?;
        return Ok(());
    }
    if let Some(state) = family
        .program()
        .states()
        .iter()
        .find(|state| &state.value_id == binding.value_id())
    {
        if binding.usage() != BufferUsage::State {
            return Err(invalid_plan(format!(
                "state value `{}` is not backed by state memory",
                binding.value_id()
            )));
        }
        validate_program_tensor(&state.tensor, binding.tensor(), "state")?;
        return Ok(());
    }
    if binding.usage() != BufferUsage::Activations {
        return Err(invalid_plan(format!(
            "semantic value `{}` must use activation memory",
            binding.value_id()
        )));
    }
    Ok(())
}

pub(super) fn validate_program_tensor(
    expected: &ProgramTensorSpec,
    actual: &ResolvedTensorSpec,
    kind: &str,
) -> Result<(), VNextError> {
    if expected.dimensions != actual.dimensions()
        || expected.element_type != actual.element_type()
        || &expected.layout != actual.layout()
    {
        return Err(invalid_plan(format!(
            "{kind} binding shape, dtype, or layout differs from model semantics"
        )));
    }
    Ok(())
}

pub(super) fn validate_weight_storage(
    family: &PreparedModelFamily,
    weight_id: &WeightId,
    storage: &ResolvedValueStorage,
) -> Result<(), VNextError> {
    let expected = family
        .weight_schema()
        .physical_component_refs(weight_id)?
        .into_iter()
        .map(|component| (component.id.clone(), component))
        .collect::<BTreeMap<_, _>>();
    if storage.components().len() != expected.len() {
        return Err(invalid_plan(format!(
            "weight `{weight_id}` physical component count differs from its schema"
        )));
    }
    let mut seen = BTreeSet::new();
    for component in storage.components() {
        let component_id = component.component_id().ok_or_else(|| {
            invalid_plan(format!(
                "weight `{weight_id}` storage lacks a physical component identity"
            ))
        })?;
        let spec = expected.get(component_id).ok_or_else(|| {
            invalid_plan(format!(
                "weight `{weight_id}` binds unknown physical component `{component_id}`"
            ))
        })?;
        let expected_element_type = spec
            .encoding
            .dense_element_type()
            .unwrap_or(ElementType::U8);
        if !seen.insert(component_id.clone())
            || component.length_bytes() != spec.physical_bytes()?
            || component.element_type() != expected_element_type
        {
            return Err(invalid_plan(format!(
                "weight `{weight_id}` component `{component_id}` byte length or dtype differs from schema"
            )));
        }
    }
    if seen != expected.keys().cloned().collect() {
        return Err(invalid_plan(format!(
            "weight `{weight_id}` physical component identities are incomplete"
        )));
    }
    Ok(())
}
pub(super) fn node_weight_requirements(
    family: &PreparedModelFamily,
    bindings: &[ResolvedValueBinding],
) -> Result<(BTreeSet<WeightFormatId>, BTreeSet<QuantizationFormatId>), VNextError> {
    let mut weight_formats = BTreeSet::new();
    let mut quantization_formats = BTreeSet::new();
    for binding in bindings
        .iter()
        .filter(|binding| binding.usage() == BufferUsage::Weights)
    {
        let weight = family
            .program()
            .weights()
            .iter()
            .find(|weight| weight.value_id == *binding.value_id())
            .ok_or_else(|| {
                invalid_plan(format!(
                    "weight binding `{}` is not declared by the model program",
                    binding.value_id()
                ))
            })?;
        weight_formats.insert(family.weight_schema().format_id.clone());
        for component in family
            .weight_schema()
            .physical_component_refs(&weight.weight_id)?
        {
            if let WeightEncoding::Quantized(spec) = &component.encoding {
                quantization_formats.insert(spec.format_id.clone());
            }
        }
    }
    Ok((weight_formats, quantization_formats))
}
pub(super) fn workspace_base_id(
    node_id: &NodeId,
    kind: &str,
    estimate_fingerprint: &str,
) -> Result<ResourceId, VNextError> {
    if !is_canonical_sha256(estimate_fingerprint) {
        return Err(invalid_plan(
            "provider workspace identity has invalid estimate fingerprint",
        ));
    }
    let digest = Sha256::digest(format!("{kind}\0{node_id}\0{estimate_fingerprint}").as_bytes());
    ResourceId::new(format!("resource/{kind}/sha256/{digest:x}"))
}
