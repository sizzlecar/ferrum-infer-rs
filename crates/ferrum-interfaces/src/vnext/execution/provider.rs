use super::{
    canonical_fingerprint, invalid_plan, is_canonical_sha256, validate_active_sequence_ceiling,
    AttributeId, BTreeMap, BTreeSet, BufferUsage, CapabilityCatalog, CapabilityId, ContractVersion,
    Deserialize, Deserializer, DynamicStorageRequirement, ExecutionPlan, MemoryPlan, ModelFamilyId,
    NodeId, OperationDescriptor, OperationId, OperationPlanningHandle, OperationPlanningRegistry,
    OperationProviderDescriptor, OperationRegistryAuthority, OperationResourceEstimate,
    OperationResourceEstimateRequest, PlanBuildRequest, PlanExactAlias, PlanHash, PlanId, PlanNode,
    PlanProviderRejectReason, PlanSchemaVersion, PlanStateEffect, PreparedModelFamily, ProgramNode,
    ProviderCompatibilityRequest, ProviderEstimateFingerprintMaterial, ProviderId,
    ProviderSelection, ProviderWorkspaceRequirement, ProviderWorkspaceScope, QuantizationFormatId,
    ResolvedValueBinding, ResourceId, RuntimePolicy, SemanticValue, Serialize, VNextError,
    WeightFormatId, EXECUTION_PLAN_SCHEMA,
};

/// Trusted output from the selected provider's shape/attribute-specific
/// resource estimator. The core binds it to the exact estimator input and
/// selected provider before the values can enter an executable plan.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderResourcePlan {
    pub(super) provider_id: ProviderId,
    pub(super) estimator_id: String,
    pub(super) estimator_version: ContractVersion,
    pub(super) estimator_implementation_fingerprint: String,
    pub(super) estimator_input_fingerprint: String,
    pub(super) estimate_fingerprint: String,
    pub(super) value_alignment_bytes: u64,
    pub(super) scratch: Option<ProviderWorkspaceRequirement>,
    pub(super) persistent: Option<ProviderWorkspaceRequirement>,
}

impl ProviderResourcePlan {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn from_provider_output(
        descriptor: &OperationProviderDescriptor,
        estimator_input_fingerprint: &str,
        estimate: OperationResourceEstimate,
    ) -> Result<Self, VNextError> {
        if estimate.estimator_id() != descriptor.resource_estimator_id()
            || estimate.estimator_version() != descriptor.resource_estimator_version()
            || estimate.estimator_implementation_fingerprint()
                != descriptor.resource_estimator_implementation_fingerprint()
            || estimate.claimed_input_fingerprint() != estimator_input_fingerprint
        {
            return Err(invalid_plan(
                "provider raw resource estimate identity or input claim differs from the selected registered implementation",
            ));
        }
        let mut plan = Self {
            provider_id: descriptor.provider_id().clone(),
            estimator_id: descriptor.resource_estimator_id().to_owned(),
            estimator_version: descriptor.resource_estimator_version(),
            estimator_implementation_fingerprint: descriptor
                .resource_estimator_implementation_fingerprint()
                .to_owned(),
            estimator_input_fingerprint: estimator_input_fingerprint.to_owned(),
            estimate_fingerprint: String::new(),
            value_alignment_bytes: estimate.value_alignment_bytes(),
            scratch: estimate.scratch().cloned(),
            persistent: estimate.persistent().cloned(),
        };
        plan.validate_fields()?;
        plan.estimate_fingerprint = plan.compute_estimate_fingerprint()?;
        plan.validate_shape()?;
        Ok(plan)
    }

    pub(super) fn validate_fields(&self) -> Result<(), VNextError> {
        if self.estimator_id.is_empty()
            || self.estimator_id.len() > 160
            || !self.estimator_id.bytes().all(|byte| {
                byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-' | b':' | b'/')
            })
            || self.estimator_version.major == 0
            || !is_canonical_sha256(&self.estimator_implementation_fingerprint)
            || !is_canonical_sha256(&self.estimator_input_fingerprint)
            || self.value_alignment_bytes == 0
            || !self.value_alignment_bytes.is_power_of_two()
            || self.scratch.as_ref().is_some_and(|workspace| {
                workspace.scope != ProviderWorkspaceScope::Invocation
                    || ProviderWorkspaceRequirement::from_formula(
                        workspace.size_formula.clone(),
                        workspace.alignment_bytes,
                        workspace.scope,
                        workspace.storage.clone(),
                    )
                    .is_err()
            })
            || self.persistent.as_ref().is_some_and(|workspace| {
                workspace.scope == ProviderWorkspaceScope::Invocation
                    || ProviderWorkspaceRequirement::from_formula(
                        workspace.size_formula.clone(),
                        workspace.alignment_bytes,
                        workspace.scope,
                        workspace.storage.clone(),
                    )
                    .is_err()
            })
        {
            return Err(invalid_plan(
                "provider resource estimate identity, alignment, or scope is invalid",
            ));
        }
        Ok(())
    }

    pub(super) fn validate_shape(&self) -> Result<(), VNextError> {
        self.validate_fields()?;
        if !is_canonical_sha256(&self.estimate_fingerprint)
            || self.estimate_fingerprint != self.compute_estimate_fingerprint()?
        {
            return Err(invalid_plan(
                "provider resource estimate fingerprint does not match its typed fields",
            ));
        }
        Ok(())
    }

    pub(super) fn compute_estimate_fingerprint(&self) -> Result<String, VNextError> {
        canonical_fingerprint(
            &ProviderEstimateFingerprintMaterial {
                provider_id: &self.provider_id,
                estimator_id: &self.estimator_id,
                estimator_version: self.estimator_version,
                estimator_implementation_fingerprint: &self.estimator_implementation_fingerprint,
                estimator_input_fingerprint: &self.estimator_input_fingerprint,
                value_alignment_bytes: self.value_alignment_bytes,
                scratch: &self.scratch,
                persistent: &self.persistent,
            },
            "fingerprint provider resource estimate",
        )
    }

    pub fn provider_id(&self) -> &ProviderId {
        &self.provider_id
    }

    pub fn estimator_id(&self) -> &str {
        &self.estimator_id
    }

    pub const fn estimator_version(&self) -> ContractVersion {
        self.estimator_version
    }

    pub fn estimator_implementation_fingerprint(&self) -> &str {
        &self.estimator_implementation_fingerprint
    }

    pub fn estimator_input_fingerprint(&self) -> &str {
        &self.estimator_input_fingerprint
    }

    pub fn estimate_fingerprint(&self) -> &str {
        &self.estimate_fingerprint
    }

    pub const fn value_alignment_bytes(&self) -> u64 {
        self.value_alignment_bytes
    }

    pub fn scratch(&self) -> Option<&ProviderWorkspaceRequirement> {
        self.scratch.as_ref()
    }

    pub fn persistent(&self) -> Option<&ProviderWorkspaceRequirement> {
        self.persistent.as_ref()
    }
}

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
        ExecutionPlan::node_weight_requirements(family, values)?;
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

/// Per-node trusted physical resolution. It supplies physical bindings and a
/// provider estimator result, but cannot provide memory totals, compatibility
/// reports, plan identities, or hashes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlanNodeResolution {
    pub(super) operation_registry_authority: OperationRegistryAuthority,
    pub(super) node_id: NodeId,
    pub(super) values: Vec<ResolvedValueBinding>,
    pub(super) required_capabilities: BTreeSet<CapabilityId>,
    pub(super) preferred_provider: Option<ProviderId>,
    pub(super) provider_resource_candidates: Vec<ProviderResourcePlan>,
    pub(super) provider_resolution_rejections: BTreeMap<ProviderId, PlanProviderRejectReason>,
}

impl PlanNodeResolution {
    pub(super) fn from_provider_resolution(
        operation_registry_authority: OperationRegistryAuthority,
        node_id: NodeId,
        values: Vec<ResolvedValueBinding>,
        required_capabilities: BTreeSet<CapabilityId>,
        preferred_provider: Option<ProviderId>,
        mut provider_resource_candidates: Vec<ProviderResourcePlan>,
        provider_resolution_rejections: BTreeMap<ProviderId, PlanProviderRejectReason>,
    ) -> Result<Self, VNextError> {
        if values.is_empty() {
            return Err(invalid_plan(format!(
                "node `{node_id}` has no physical value resolution"
            )));
        }
        provider_resource_candidates
            .sort_by(|left, right| left.provider_id.cmp(&right.provider_id));
        if provider_resource_candidates.is_empty()
            || provider_resource_candidates
                .windows(2)
                .any(|pair| pair[0].provider_id == pair[1].provider_id)
        {
            return Err(invalid_plan(format!(
                "node `{node_id}` has empty or duplicate provider resource candidates"
            )));
        }
        for candidate in &provider_resource_candidates {
            candidate.validate_shape()?;
            if provider_resolution_rejections.contains_key(candidate.provider_id()) {
                return Err(invalid_plan(format!(
                    "provider `{}` is both a resource candidate and rejected resolution",
                    candidate.provider_id()
                )));
            }
        }
        for reason in provider_resolution_rejections.values() {
            match reason {
                PlanProviderRejectReason::StorageIncompatible { resource_ids }
                    if !resource_ids.is_empty()
                        && !resource_ids.windows(2).any(|pair| pair[0] >= pair[1]) => {}
                PlanProviderRejectReason::StorageIncompatible { .. } => {
                    return Err(invalid_plan(
                        "provider storage rejection resources are empty or non-canonical",
                    ))
                }
                PlanProviderRejectReason::NotRegistered
                | PlanProviderRejectReason::Incompatible(_) => {
                    return Err(invalid_plan(
                        "resolution-local rejection must describe storage incompatibility",
                    ))
                }
            }
        }
        Ok(Self {
            operation_registry_authority,
            node_id,
            values,
            required_capabilities,
            preferred_provider,
            provider_resource_candidates,
            provider_resolution_rejections,
        })
    }

    /// Resolves one node through the typed planning registry. Provider
    /// selection is core-owned; `preferred_provider` is only a compatibility
    /// preference and cannot inject a provider or resource estimate.
    #[allow(clippy::too_many_arguments)]
    pub fn resolve<P: RuntimePolicy>(
        family: &PreparedModelFamily,
        catalog: &CapabilityCatalog,
        policy: &P,
        registry: &OperationPlanningHandle<'_>,
        node_id: NodeId,
        values: Vec<ResolvedValueBinding>,
        required_capabilities: BTreeSet<CapabilityId>,
        preferred_provider: Option<ProviderId>,
    ) -> Result<Self, VNextError> {
        policy.validate()?;
        validate_active_sequence_ceiling(policy.maximum_active_sequences())?;
        if values.is_empty() {
            return Err(invalid_plan(format!(
                "node `{node_id}` has no physical value resolution"
            )));
        }
        let program_node = family
            .program()
            .blocks()
            .iter()
            .flat_map(|block| &block.nodes)
            .find(|node| node.id == node_id)
            .ok_or_else(|| invalid_plan(format!("program has no node `{node_id}`")))?;
        let operation = catalog.operation(&program_node.operation_id)?;

        let contracts = registry.contracts_for(&program_node.operation_id);
        if contracts.len() != 1 {
            return Err(invalid_plan(format!(
                "operation `{}` requires exactly one typed contract registration, found {}",
                program_node.operation_id,
                contracts.len()
            )));
        }
        let contract = contracts[0];
        if contract.descriptor() != operation {
            return Err(invalid_plan(format!(
                "typed contract for operation `{}` differs from the capability catalog",
                program_node.operation_id
            )));
        }
        contract.validate_signature(&operation.inputs, &operation.outputs)?;
        operation.validate_attributes(&program_node.attributes)?;
        operation.validate_resolved_bindings(&values)?;
        ExecutionPlan::validate_program_bindings(program_node, &values)?;
        for binding in &values {
            ExecutionPlan::validate_semantic_binding(family, binding)?;
        }

        let (required_weight_formats, required_quantization_formats) =
            ExecutionPlan::node_weight_requirements(family, &values)?;
        let compatibility_request = ProviderCompatibilityRequest::new(
            program_node.operation_id.clone(),
            program_node.required_version,
            operation
                .provider
                .required_capabilities
                .union(&required_capabilities)
                .cloned()
                .collect(),
            required_weight_formats,
            required_quantization_formats,
        )?;
        let report = catalog.provider_compatibility(compatibility_request)?;
        report.require_compatible(&catalog.device().id)?;
        let profile_available = |requirement: &DynamicStorageRequirement| {
            policy
                .dynamic_storage_profile_order()
                .iter()
                .any(|profile| {
                    catalog.device().dynamic_storage_profiles.contains(profile)
                        && requirement.accepts(*profile)
                })
        };
        let mut provider_resource_candidates = Vec::new();
        let mut provider_resolution_rejections = BTreeMap::new();
        for provider_id in report.compatible_provider_ids() {
            let descriptor = catalog
                .providers_for(&program_node.operation_id)?
                .iter()
                .find(|provider| provider.provider_id() == provider_id)
                .ok_or_else(|| invalid_plan("compatible provider is absent from the catalog"))?;
            let value_storage_conflicts = values
                .iter()
                .filter(|binding| binding.usage() != BufferUsage::Weights)
                .filter(|binding| {
                    descriptor
                        .dynamic_storage_for(binding.role(), binding.ordinal())
                        .is_none_or(|requirement| !profile_available(requirement))
                })
                .flat_map(|binding| binding.storage().components())
                .map(|component| component.resource_id().clone())
                .collect::<BTreeSet<_>>();
            if !value_storage_conflicts.is_empty() {
                provider_resolution_rejections.insert(
                    provider_id.clone(),
                    PlanProviderRejectReason::StorageIncompatible {
                        resource_ids: value_storage_conflicts.into_iter().collect(),
                    },
                );
                continue;
            }
            let estimators = registry.estimators_for(provider_id);
            if estimators.len() != 1 || estimators[0].descriptor() != descriptor {
                return Err(invalid_plan(format!(
                    "compatible provider `{provider_id}` lacks one exact resource estimator registration"
                )));
            }
            let estimator_input_fingerprint = provider_resource_estimator_input_fingerprint(
                family,
                operation,
                program_node,
                provider_id,
                &values,
                &required_capabilities,
            )?;
            let estimate_request = OperationResourceEstimateRequest::new(
                &program_node.id,
                operation,
                &values,
                &program_node.attributes,
                &estimator_input_fingerprint,
            )?;
            let estimate = estimators[0].estimate_resources(estimate_request)?;
            let provider_resources = ProviderResourcePlan::from_provider_output(
                descriptor,
                &estimator_input_fingerprint,
                estimate,
            )?;
            let minimum_alignment = operation.resources.minimum_value_alignment_bytes;
            if provider_resources.value_alignment_bytes() < minimum_alignment
                || provider_resources.value_alignment_bytes() % minimum_alignment != 0
                || !operation
                    .resources
                    .scratch
                    .accepts(provider_resources.scratch().is_some())
                || !operation
                    .resources
                    .persistent
                    .accepts(provider_resources.persistent().is_some())
            {
                return Err(invalid_plan(format!(
                    "compatible provider `{provider_id}` returned resources outside its operation contract"
                )));
            }
            let mut workspace_storage_conflicts = BTreeSet::new();
            for (kind, workspace) in [
                ("scratch", provider_resources.scratch()),
                ("persistent", provider_resources.persistent()),
            ] {
                if workspace.is_some_and(|workspace| !profile_available(workspace.storage())) {
                    workspace_storage_conflicts.insert(ExecutionPlan::workspace_base_id(
                        &program_node.id,
                        kind,
                        provider_resources.estimate_fingerprint(),
                    )?);
                }
            }
            if !workspace_storage_conflicts.is_empty() {
                provider_resolution_rejections.insert(
                    provider_id.clone(),
                    PlanProviderRejectReason::StorageIncompatible {
                        resource_ids: workspace_storage_conflicts.into_iter().collect(),
                    },
                );
                continue;
            }
            provider_resource_candidates.push(provider_resources);
        }
        if provider_resource_candidates.is_empty() {
            return Err(invalid_plan(format!(
                "node `{node_id}` has no provider whose binding and workspace storage requirements intersect runtime offers and policy"
            )));
        }
        Self::from_provider_resolution(
            registry.authority().clone(),
            node_id,
            values,
            required_capabilities,
            preferred_provider,
            provider_resource_candidates,
            provider_resolution_rejections,
        )
    }

    pub fn node_id(&self) -> &NodeId {
        &self.node_id
    }

    pub fn values(&self) -> &[ResolvedValueBinding] {
        &self.values
    }

    pub fn required_capabilities(&self) -> &BTreeSet<CapabilityId> {
        &self.required_capabilities
    }

    pub fn preferred_provider(&self) -> Option<&ProviderId> {
        self.preferred_provider.as_ref()
    }

    pub fn provider_resource_candidates(&self) -> &[ProviderResourcePlan] {
        &self.provider_resource_candidates
    }

    pub fn provider_resolution_rejections(
        &self,
    ) -> &BTreeMap<ProviderId, PlanProviderRejectReason> {
        &self.provider_resolution_rejections
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ExecutionPlanPayload {
    pub(super) schema: PlanSchemaVersion,
    pub(super) plan_id: PlanId,
    pub(super) family_id: ModelFamilyId,
    pub(super) device_id: super::DeviceId,
    pub(super) device_runtime_implementation_fingerprint: String,
    pub(super) prepared_family_fingerprint: String,
    pub(super) program_fingerprint: String,
    pub(super) capability_catalog_fingerprint: String,
    pub(super) policy_version: ContractVersion,
    pub(super) policy_fingerprint: String,
    pub(super) weight_format: WeightFormatId,
    pub(super) quantization_formats: BTreeSet<QuantizationFormatId>,
    pub(super) nodes: Vec<PlanNode>,
    pub(super) memory: MemoryPlan,
}

impl ExecutionPlanPayload {
    pub const fn schema(&self) -> PlanSchemaVersion {
        self.schema
    }

    pub fn plan_id(&self) -> &PlanId {
        &self.plan_id
    }

    pub fn family_id(&self) -> &ModelFamilyId {
        &self.family_id
    }

    pub fn device_id(&self) -> &super::DeviceId {
        &self.device_id
    }

    pub fn device_runtime_implementation_fingerprint(&self) -> &str {
        &self.device_runtime_implementation_fingerprint
    }

    pub fn prepared_family_fingerprint(&self) -> &str {
        &self.prepared_family_fingerprint
    }

    pub fn program_fingerprint(&self) -> &str {
        &self.program_fingerprint
    }

    pub fn capability_catalog_fingerprint(&self) -> &str {
        &self.capability_catalog_fingerprint
    }

    pub const fn policy_version(&self) -> ContractVersion {
        self.policy_version
    }

    pub fn policy_fingerprint(&self) -> &str {
        &self.policy_fingerprint
    }

    pub fn weight_format(&self) -> &WeightFormatId {
        &self.weight_format
    }

    pub fn quantization_formats(&self) -> &BTreeSet<QuantizationFormatId> {
        &self.quantization_formats
    }

    pub fn nodes(&self) -> &[PlanNode] {
        &self.nodes
    }

    pub fn memory(&self) -> &MemoryPlan {
        &self.memory
    }
}

#[derive(Serialize)]
pub(super) struct PlanHashMaterial<'a> {
    pub(super) schema: PlanSchemaVersion,
    pub(super) family_id: &'a ModelFamilyId,
    pub(super) device_id: &'a super::DeviceId,
    pub(super) device_runtime_implementation_fingerprint: &'a str,
    pub(super) prepared_family_fingerprint: &'a str,
    pub(super) program_fingerprint: &'a str,
    pub(super) capability_catalog_fingerprint: &'a str,
    pub(super) policy_version: ContractVersion,
    pub(super) policy_fingerprint: &'a str,
    pub(super) weight_format: &'a WeightFormatId,
    pub(super) quantization_formats: &'a BTreeSet<QuantizationFormatId>,
    pub(super) nodes: &'a [PlanNode],
    pub(super) memory: &'a MemoryPlan,
}

impl<'a> From<&'a ExecutionPlanPayload> for PlanHashMaterial<'a> {
    fn from(payload: &'a ExecutionPlanPayload) -> Self {
        Self {
            schema: payload.schema,
            family_id: &payload.family_id,
            device_id: &payload.device_id,
            device_runtime_implementation_fingerprint: &payload
                .device_runtime_implementation_fingerprint,
            prepared_family_fingerprint: &payload.prepared_family_fingerprint,
            program_fingerprint: &payload.program_fingerprint,
            capability_catalog_fingerprint: &payload.capability_catalog_fingerprint,
            policy_version: payload.policy_version,
            policy_fingerprint: &payload.policy_fingerprint,
            weight_format: &payload.weight_format,
            quantization_formats: &payload.quantization_formats,
            nodes: &payload.nodes,
            memory: &payload.memory,
        }
    }
}

/// A wire payload is deliberately not an executable plan. It must be rebuilt
/// against a typed model family, catalog, and runtime policy before use.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UnvalidatedProviderResourcePlan {
    pub(super) provider_id: ProviderId,
    pub(super) estimator_id: String,
    pub(super) estimator_version: ContractVersion,
    pub(super) estimator_implementation_fingerprint: String,
    pub(super) estimator_input_fingerprint: String,
    pub(super) estimate_fingerprint: String,
    pub(super) value_alignment_bytes: u64,
    pub(super) scratch: Option<ProviderWorkspaceRequirement>,
    pub(super) persistent: Option<ProviderWorkspaceRequirement>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UnvalidatedPlanNode {
    pub(super) id: NodeId,
    pub(super) dependencies: Vec<NodeId>,
    pub(super) operation_id: OperationId,
    pub(super) operation_version: ContractVersion,
    pub(super) operation_fingerprint: String,
    pub(super) provider_implementation_fingerprint: String,
    pub(super) required_capabilities: BTreeSet<CapabilityId>,
    pub(super) attributes: BTreeMap<AttributeId, SemanticValue>,
    pub(super) selection: ProviderSelection,
    pub(super) provider_resources: UnvalidatedProviderResourcePlan,
    pub(super) values: Vec<ResolvedValueBinding>,
    pub(super) exact_aliases: Vec<PlanExactAlias>,
    pub(super) state_effects: Vec<PlanStateEffect>,
    pub(super) scratch_resource: Option<ResourceId>,
    pub(super) persistent_resource: Option<ResourceId>,
    pub(super) resources: Vec<ResourceId>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct UnvalidatedExecutionPlanPayload {
    pub(super) schema: PlanSchemaVersion,
    pub(super) plan_id: PlanId,
    pub(super) family_id: ModelFamilyId,
    pub(super) device_id: super::DeviceId,
    pub(super) device_runtime_implementation_fingerprint: String,
    pub(super) prepared_family_fingerprint: String,
    pub(super) program_fingerprint: String,
    pub(super) capability_catalog_fingerprint: String,
    pub(super) policy_version: ContractVersion,
    pub(super) policy_fingerprint: String,
    pub(super) weight_format: WeightFormatId,
    pub(super) quantization_formats: BTreeSet<QuantizationFormatId>,
    pub(super) nodes: Vec<UnvalidatedPlanNode>,
    pub(super) memory: MemoryPlan,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnvalidatedExecutionPlan {
    pub(super) payload: UnvalidatedExecutionPlanPayload,
    pub(super) plan_hash: PlanHash,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub(crate) struct UnvalidatedExecutionPlanWire {
    pub(super) payload: UnvalidatedExecutionPlanPayload,
    pub(super) plan_hash: PlanHash,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct UnvalidatedExecutionPlanWireFields {
    pub(super) payload: UnvalidatedExecutionPlanPayload,
    pub(super) plan_hash: PlanHash,
}

impl<'de> Deserialize<'de> for UnvalidatedExecutionPlanWire {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let raw = serde_json::Value::deserialize(deserializer)?;
        let fields = UnvalidatedExecutionPlanWireFields::deserialize(&raw)
            .map_err(serde::de::Error::custom)?;
        let canonical = serde_json::to_value(&fields).map_err(serde::de::Error::custom)?;
        if canonical != raw {
            return Err(serde::de::Error::custom(
                "execution plan wire contains unknown or non-canonical nested fields",
            ));
        }
        Ok(Self {
            payload: fields.payload,
            plan_hash: fields.plan_hash,
        })
    }
}

impl From<UnvalidatedExecutionPlanWire> for UnvalidatedExecutionPlan {
    fn from(wire: UnvalidatedExecutionPlanWire) -> Self {
        Self {
            payload: wire.payload,
            plan_hash: wire.plan_hash,
        }
    }
}

impl UnvalidatedExecutionPlan {
    pub fn schema(&self) -> PlanSchemaVersion {
        self.payload.schema
    }

    pub fn revalidate<P: RuntimePolicy>(
        self,
        family: &PreparedModelFamily,
        capabilities: &CapabilityCatalog,
        policy: &P,
        node_resolutions: Vec<PlanNodeResolution>,
    ) -> Result<ExecutionPlan, VNextError> {
        if self.payload.schema != EXECUTION_PLAN_SCHEMA {
            return Err(VNextError::UnsupportedPlanSchema {
                expected_major: EXECUTION_PLAN_SCHEMA.major,
                expected_minor: EXECUTION_PLAN_SCHEMA.minor,
                actual_major: self.payload.schema.major,
                actual_minor: self.payload.schema.minor,
            });
        }
        let rebuilt = ExecutionPlan::build(PlanBuildRequest::new(
            family,
            capabilities,
            policy,
            node_resolutions,
        )?)?;
        let untrusted_payload =
            serde_json::to_value(&self.payload).map_err(|error| VNextError::Serialization {
                context: "serialize unvalidated execution plan payload",
                message: error.to_string(),
            })?;
        let rebuilt_payload =
            serde_json::to_value(&rebuilt.payload).map_err(|error| VNextError::Serialization {
                context: "serialize rebuilt execution plan payload",
                message: error.to_string(),
            })?;
        if untrusted_payload != rebuilt_payload {
            return Err(invalid_plan(
                "untrusted plan differs from a semantic rebuild against current dependencies",
            ));
        }
        if rebuilt.plan_hash != self.plan_hash {
            return Err(VNextError::PlanHashMismatch {
                expected: rebuilt.plan_hash.to_string(),
                actual: self.plan_hash.to_string(),
            });
        }
        Ok(rebuilt)
    }
}
