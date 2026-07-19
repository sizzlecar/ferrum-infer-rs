use super::{
    invalid_plan, node_weight_requirements, provider_resource_estimator_input_fingerprint,
    validate_active_sequence_ceiling, validate_program_bindings, validate_scheduled_token_ceiling,
    validate_semantic_binding, workspace_base_id, BTreeMap, BTreeSet, BufferUsage,
    CapabilityCatalog, CapabilityId, DynamicStorageRequirement, NodeId, OperationPlanningHandle,
    OperationPlanningRegistry, OperationRegistryAuthority, OperationResourceEstimateRequest,
    PlanNodeResolution, PlanProviderRejectReason, PreparedModelFamily,
    ProviderCompatibilityRequest, ProviderId, ProviderResourcePlan, ResolvedValueBinding,
    RuntimePolicy, VNextError,
};

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
        validate_scheduled_token_ceiling(policy.maximum_scheduled_tokens())?;
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
        validate_program_bindings(program_node, &values)?;
        for binding in &values {
            validate_semantic_binding(family, binding)?;
        }

        let (required_weight_formats, required_quantization_formats) =
            node_weight_requirements(family, &values)?;
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
                    .binding
                    .accepts(provider_resources.binding().is_some())
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
                ("binding", provider_resources.binding()),
                ("persistent", provider_resources.persistent()),
            ] {
                if workspace.is_some_and(|workspace| !profile_available(workspace.storage())) {
                    workspace_storage_conflicts.insert(workspace_base_id(
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
