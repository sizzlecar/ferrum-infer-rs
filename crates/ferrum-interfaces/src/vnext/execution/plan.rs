use super::{
    canonical_fingerprint, canonical_runtime_policy_fingerprint, invalid_plan, is_canonical_sha256,
    joint_candidate_components, joint_partial_precedes, node_weight_requirements,
    provider_resource_estimator_input_fingerprint, static_contiguous_storage_profile,
    storage_incompatible_resource_ids, tensor_storage_layout_fingerprint,
    validate_active_sequence_ceiling, validate_program_bindings, validate_scheduled_token_ceiling,
    validate_semantic_binding, workspace_base_id, workspace_storage_layout_fingerprint,
    AliasPolicy, AllocationKind, AllocationLifetime, BTreeMap, BTreeSet, BufferUsage,
    CanonicalValueBinding, CapabilityCatalog, CapabilityId, DimensionConstraint,
    DynamicResourceDescriptor, DynamicStorageContract, DynamicStorageProfile,
    DynamicStorageRequirement, ElementType, ExecutionPlanPayload, GlobalValueRange,
    JointComponentSolution, JointPartialSelection, JointProviderCandidate,
    JointProviderStorageSelection, JointSelectionObjective, MemoryPlan, NodeId,
    NodeTokenBindingProjection, NodeWorkContract, OperationDescriptor, OperationRegistryAuthority,
    PlanBuildRequest, PlanExactAlias, PlanExactAliasKind, PlanHash, PlanHashMaterial, PlanId,
    PlanNode, PlanNodeResolution, PlanProviderRejectReason, PlanStateEffect, PreparedModelFamily,
    ProgramNode, ProgramNodeWorkSpec, ProgramValueId, ProviderCompatibilityRequest, ProviderId,
    ProviderResourcePlan, ProviderSelection, ProviderSelectionReason, ProviderWorkspaceScope,
    QuantizationFormatId, RejectedProvider, ResolvedValueBinding, ResolvedValueRole,
    ResourceAllocation, ResourceId, ReusableExecutionMemoryPlan, ReusableExecutionPolicy,
    RuntimePolicy, Serialize, StateCapacityDemand, StateDependencyTracker, StateInitialization,
    StateLifetime, TensorAccess, UnvalidatedExecutionPlan, UnvalidatedExecutionPlanWire,
    VNextError, ValueAllocationAccumulator, ValueResourceDemand, WeightFormatId,
    EXECUTION_PLAN_SCHEMA, MAX_EXECUTION_PLAN_WIRE_BYTES,
};

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ExecutionPlan {
    pub(super) payload: ExecutionPlanPayload,
    pub(super) plan_hash: PlanHash,
    #[serde(skip)]
    pub(super) operation_registry_authority: OperationRegistryAuthority,
}

impl ExecutionPlan {
    pub fn build<P: RuntimePolicy>(request: PlanBuildRequest<'_, P>) -> Result<Self, VNextError> {
        request.policy.validate()?;
        let maximum_active_sequences = request.policy.maximum_active_sequences();
        validate_active_sequence_ceiling(maximum_active_sequences)?;
        let maximum_scheduled_tokens = request.policy.maximum_scheduled_tokens();
        validate_scheduled_token_ceiling(maximum_scheduled_tokens)?;
        let operation_registry_authority = request
            .node_resolutions
            .first()
            .ok_or_else(|| invalid_plan("plan build request has no node resolutions"))?
            .operation_registry_authority
            .clone();
        if request.node_resolutions.iter().any(|resolution| {
            resolution.operation_registry_authority != operation_registry_authority
        }) {
            return Err(invalid_plan(
                "node resolutions belong to different operation runtime registries",
            ));
        }
        let policy_capacity = request.policy.memory_capacity_bytes();
        let memory_reserve = request.policy.memory_reserve_bytes();
        let device_capacity = request.capabilities.device().total_memory_bytes;
        if policy_capacity == 0
            || policy_capacity > device_capacity
            || memory_reserve >= policy_capacity
        {
            return Err(invalid_plan(
                "runtime policy raw capacity, reserve, or typed admission concurrency is invalid for the device descriptor",
            ));
        }
        let family = request.family;
        let program = family.program();
        let prepared_family_fingerprint = family.fingerprint()?;
        let program_fingerprint = program.fingerprint()?;
        let capability_catalog_fingerprint = request.capabilities.fingerprint()?;
        let device_runtime_implementation_fingerprint = request
            .capabilities
            .device()
            .runtime_implementation_fingerprint
            .clone();
        let policy_fingerprint = canonical_runtime_policy_fingerprint(request.policy)?;
        let weight_format = family.weight_schema().format_id.clone();
        let quantization_formats = family.weight_schema().quantization_formats();

        let mut resolutions = BTreeMap::new();
        for resolution in request.node_resolutions {
            let node_id = resolution.node_id.clone();
            if resolutions.insert(node_id.clone(), resolution).is_some() {
                return Err(invalid_plan(format!(
                    "node `{node_id}` has duplicate physical resolutions"
                )));
            }
        }

        let program_nodes = program
            .blocks()
            .iter()
            .flat_map(|block| &block.nodes)
            .collect::<Vec<_>>();
        let joint_storage = Self::select_joint_provider_storage(
            &program_nodes,
            &resolutions,
            request.capabilities,
            request.policy,
        )?;
        let mut selected_node_resources = joint_storage.node_resources;
        let selected_resource_profiles = joint_storage.resource_profiles;
        let mut storage_rejections = joint_storage.storage_rejections;
        let producers = program_nodes
            .iter()
            .flat_map(|node| {
                node.outputs
                    .iter()
                    .map(move |output| (output.clone(), node.id.clone()))
            })
            .collect::<BTreeMap<_, _>>();
        let mut last_consumers = BTreeMap::<ProgramValueId, usize>::new();
        for (node_index, node) in program_nodes.iter().enumerate() {
            for input in &node.inputs {
                last_consumers.insert(input.clone(), node_index);
            }
        }
        let program_outputs = program.outputs().iter().cloned().collect::<BTreeSet<_>>();
        let mut canonical_values = BTreeMap::new();
        let mut bound_values = BTreeSet::new();
        let mut nodes = Vec::new();
        let mut state_dependencies = StateDependencyTracker::default();
        for (node_index, program_node) in program_nodes.into_iter().enumerate() {
            let resolution = resolutions.remove(&program_node.id).ok_or_else(|| {
                invalid_plan(format!(
                    "node `{}` has no physical resolution",
                    program_node.id
                ))
            })?;
            let provider_resources = selected_node_resources
                .remove(&program_node.id)
                .ok_or_else(|| invalid_plan("joint solver omitted one program node"))?;
            let storage_rejection = storage_rejections.remove(&program_node.id);
            let node = Self::build_node(
                family,
                program_node,
                resolution,
                provider_resources,
                storage_rejection,
                request.capabilities,
                &producers,
                node_index,
                &last_consumers,
                &program_outputs,
                &mut state_dependencies,
                &mut canonical_values,
                &mut bound_values,
            )?;
            nodes.push(node);
        }
        if !resolutions.is_empty() {
            return Err(invalid_plan(format!(
                "physical resolutions contain unknown nodes: {:?}",
                resolutions.keys().collect::<Vec<_>>()
            )));
        }
        if !selected_node_resources.is_empty() {
            return Err(invalid_plan(
                "joint solver returned resources for unknown program nodes",
            ));
        }
        if !storage_rejections.is_empty() {
            return Err(invalid_plan(
                "joint solver returned storage rejections for unknown program nodes",
            ));
        }
        Self::validate_semantic_coverage(family, &bound_values)?;
        Self::validate_global_storage_aliasing(&canonical_values, &nodes)?;
        let memory = Self::build_memory_plan(
            family,
            device_capacity,
            policy_capacity,
            memory_reserve,
            maximum_active_sequences,
            maximum_scheduled_tokens,
            &nodes,
            &selected_resource_profiles,
            request.policy.reusable_execution_policy(),
        )?;

        let mut payload = ExecutionPlanPayload {
            schema: EXECUTION_PLAN_SCHEMA,
            plan_id: PlanId::new("plan/unset")?,
            family_id: family.family_id().clone(),
            device_id: request.capabilities.device().id.clone(),
            device_runtime_implementation_fingerprint,
            prepared_family_fingerprint,
            program_fingerprint,
            capability_catalog_fingerprint,
            policy_version: request.policy.version(),
            policy_fingerprint,
            maximum_scheduled_tokens,
            weight_format,
            quantization_formats,
            nodes,
            memory,
        };
        let plan_hash = PlanHash::new(canonical_fingerprint(
            &PlanHashMaterial::from(&payload),
            "fingerprint execution plan",
        )?)?;
        payload.plan_id = Self::plan_id_for_hash(&plan_hash)?;
        let plan = Self {
            payload,
            plan_hash,
            operation_registry_authority,
        };
        plan.validate_internal()?;
        Ok(plan)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn build_node(
        family: &PreparedModelFamily,
        program_node: &ProgramNode,
        resolution: PlanNodeResolution,
        provider_resources: ProviderResourcePlan,
        storage_rejection: Option<RejectedProvider>,
        catalog: &CapabilityCatalog,
        producers: &BTreeMap<ProgramValueId, NodeId>,
        node_index: usize,
        last_consumers: &BTreeMap<ProgramValueId, usize>,
        program_outputs: &BTreeSet<ProgramValueId>,
        state_dependencies: &mut StateDependencyTracker,
        canonical_values: &mut BTreeMap<ProgramValueId, CanonicalValueBinding>,
        bound_values: &mut BTreeSet<ProgramValueId>,
    ) -> Result<PlanNode, VNextError> {
        let operation = catalog.operation_for_node(&program_node.id, &program_node.operation_id)?;
        if !operation.version.satisfies(program_node.required_version) {
            return Err(VNextError::IncompatibleOperationVersion {
                node_id: Some(program_node.id.to_string()),
                operation_id: program_node.operation_id.to_string(),
                required_major: program_node.required_version.major,
                required_minor: program_node.required_version.minor,
                available_major: operation.version.major,
                available_minor: operation.version.minor,
            });
        }
        operation.validate_attributes(&program_node.attributes)?;
        operation.validate_resolved_bindings(&resolution.values)?;
        validate_program_bindings(program_node, &resolution.values)?;
        let exact_aliases = Self::extract_exact_aliases(operation, &resolution.values)?;
        let work = Self::derive_node_work_contract(program_node, operation, &resolution.values)?;
        Self::validate_alias_liveness(
            family,
            program_node,
            node_index,
            last_consumers,
            program_outputs,
            &exact_aliases,
            &resolution.values,
        )?;
        let state_effects = Self::derive_state_effects(family, &resolution.values)?;
        for binding in &resolution.values {
            validate_semantic_binding(family, binding)?;
            Self::validate_cross_node_value(binding, canonical_values)?;
            bound_values.insert(binding.value_id().clone());
        }

        let (required_weight_formats, required_quantization_formats) =
            node_weight_requirements(family, &resolution.values)?;
        let selection = Self::select_provider(
            program_node,
            operation,
            catalog,
            &resolution.required_capabilities,
            resolution.preferred_provider.as_ref(),
            &provider_resources.provider_id,
            storage_rejection,
            &required_weight_formats,
            &required_quantization_formats,
        )?;
        provider_resources.validate_shape()?;
        if provider_resources.provider_id != selection.selected_provider {
            return Err(invalid_plan(format!(
                "node `{}` resource estimate belongs to provider `{}` instead of selected provider `{}`",
                program_node.id,
                provider_resources.provider_id,
                selection.selected_provider
            )));
        }
        let selected_provider = catalog
            .providers_for_node(&program_node.id, &program_node.operation_id)?
            .iter()
            .find(|provider| provider.provider_id() == &selection.selected_provider)
            .ok_or_else(|| {
                invalid_plan(format!(
                    "node `{}` selected provider is absent from the catalog",
                    program_node.id
                ))
            })?;
        if provider_resources.estimator_id != selected_provider.resource_estimator_id()
            || provider_resources.estimator_version
                != selected_provider.resource_estimator_version()
            || provider_resources.estimator_implementation_fingerprint
                != selected_provider.resource_estimator_implementation_fingerprint()
        {
            return Err(invalid_plan(format!(
                "node `{}` provider resource estimate is not issued by the selected catalog provider's estimator",
                program_node.id
            )));
        }
        let minimum_value_alignment = operation.resources.minimum_value_alignment_bytes;
        if provider_resources.value_alignment_bytes < minimum_value_alignment
            || provider_resources.value_alignment_bytes % minimum_value_alignment != 0
            || !operation
                .resources
                .scratch
                .accepts(provider_resources.scratch.is_some())
            || !operation
                .resources
                .binding
                .accepts(provider_resources.binding.is_some())
            || !operation
                .resources
                .persistent
                .accepts(provider_resources.persistent.is_some())
        {
            return Err(invalid_plan(format!(
                "node `{}` provider resource estimate violates the operation's alignment or workspace-presence contract",
                program_node.id
            )));
        }
        let expected_estimator_input = provider_resource_estimator_input_fingerprint(
            family,
            operation,
            program_node,
            &selection.selected_provider,
            &resolution.values,
            &resolution.required_capabilities,
        )?;
        if provider_resources.estimator_input_fingerprint != expected_estimator_input {
            return Err(invalid_plan(format!(
                "node `{}` provider resource estimate is not bound to its selected provider, shape, attributes, and bindings",
                program_node.id
            )));
        }
        let mut dependencies = program_node
            .inputs
            .iter()
            .filter_map(|input| producers.get(input))
            .cloned()
            .collect::<BTreeSet<_>>();
        Self::add_state_dependencies(
            &program_node.id,
            &state_effects,
            state_dependencies,
            &mut dependencies,
        );
        let dependencies = dependencies.into_iter().collect::<Vec<_>>();
        let scratch_resource = provider_resources
            .scratch
            .as_ref()
            .map(|_| {
                workspace_base_id(
                    &program_node.id,
                    "scratch",
                    &provider_resources.estimate_fingerprint,
                )
            })
            .transpose()?;
        let binding_resource = provider_resources
            .binding
            .as_ref()
            .map(|_| {
                workspace_base_id(
                    &program_node.id,
                    "binding",
                    &provider_resources.estimate_fingerprint,
                )
            })
            .transpose()?;
        let persistent_resource = provider_resources
            .persistent
            .as_ref()
            .map(|_| {
                workspace_base_id(
                    &program_node.id,
                    "persistent",
                    &provider_resources.estimate_fingerprint,
                )
            })
            .transpose()?;
        let resources = resolution
            .values
            .iter()
            .flat_map(|binding| binding.storage().components())
            .map(|component| component.resource_id().clone())
            .chain(scratch_resource.iter().cloned())
            .chain(binding_resource.iter().cloned())
            .chain(persistent_resource.iter().cloned())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        Ok(PlanNode {
            id: program_node.id.clone(),
            dependencies,
            operation_id: program_node.operation_id.clone(),
            operation_version: program_node.required_version,
            operation_fingerprint: operation.fingerprint()?,
            provider_implementation_fingerprint: selected_provider
                .provider_implementation_fingerprint()
                .to_owned(),
            required_capabilities: resolution.required_capabilities,
            attributes: program_node.attributes.clone(),
            work,
            selection,
            provider_resources,
            values: resolution.values,
            exact_aliases,
            state_effects,
            scratch_resource,
            binding_resource,
            persistent_resource,
            resources,
        })
    }

    pub(super) fn derive_node_work_contract(
        node: &ProgramNode,
        operation: &OperationDescriptor,
        bindings: &[ResolvedValueBinding],
    ) -> Result<NodeWorkContract, VNextError> {
        let ProgramNodeWorkSpec::Tokens {
            value_id,
            axis: source_axis,
        } = &node.work
        else {
            return Ok(NodeWorkContract::Fixed);
        };
        let source_binding = bindings
            .iter()
            .find(|binding| binding.value_id() == value_id)
            .ok_or_else(|| invalid_plan("node token work source has no resolved binding"))?;
        let source_contract = match source_binding.role() {
            ResolvedValueRole::Input => operation.inputs.get(source_binding.ordinal() as usize),
            ResolvedValueRole::Output => operation.outputs.get(source_binding.ordinal() as usize),
        }
        .ok_or_else(|| invalid_plan("node token work source ordinal is outside its operation"))?;
        let source_axis_index = usize::try_from(*source_axis)
            .map_err(|_| invalid_plan("node token work axis exceeds usize"))?;
        let source_symbol = match source_contract.dimensions().get(source_axis_index) {
            Some(DimensionConstraint::Symbol(symbol)) => symbol,
            _ => {
                return Err(invalid_plan(
                    "node token work source axis is not one symbolic operation dimension",
                ))
            }
        };
        if source_binding.usage() != BufferUsage::Activations
            || source_binding
                .tensor()
                .dimensions()
                .get(source_axis_index)
                .is_none()
        {
            return Err(invalid_plan(
                "node token work source is not an in-bounds activation axis",
            ));
        }

        let mut projections = Vec::new();
        for binding in bindings {
            let contract = match binding.role() {
                ResolvedValueRole::Input => operation.inputs.get(binding.ordinal() as usize),
                ResolvedValueRole::Output => operation.outputs.get(binding.ordinal() as usize),
            }
            .ok_or_else(|| {
                invalid_plan("resolved work binding ordinal is outside its operation")
            })?;
            let matching_axes = contract
                .dimensions()
                .iter()
                .enumerate()
                .filter(|(_, dimension)| {
                    matches!(dimension, DimensionConstraint::Symbol(symbol) if symbol == source_symbol)
                })
                .map(|(axis, _)| axis)
                .collect::<Vec<_>>();
            if matching_axes.len() > 1 {
                return Err(invalid_plan(
                    "one resolved binding repeats the node token work dimension",
                ));
            }
            let Some(axis) = matching_axes.first().copied() else {
                continue;
            };
            let dimensions = binding.tensor().dimensions();
            if binding.usage() != BufferUsage::Activations || dimensions.get(axis).is_none() {
                return Err(invalid_plan(
                    "node token work projection is not an in-bounds activation axis",
                ));
            }
            projections.push(NodeTokenBindingProjection {
                value_id: binding.value_id().clone(),
                role: binding.role(),
                ordinal: binding.ordinal(),
                axis: u32::try_from(axis)
                    .map_err(|_| invalid_plan("node token projection axis exceeds u32"))?,
                rank: u32::try_from(dimensions.len())
                    .map_err(|_| invalid_plan("node token projection rank exceeds u32"))?,
                canonical_extent: dimensions[axis],
            });
        }
        projections.sort();
        if projections.is_empty()
            || projections
                .windows(2)
                .any(|pair| pair[0].role == pair[1].role && pair[0].ordinal == pair[1].ordinal)
        {
            return Err(invalid_plan(
                "node token work projections are empty or non-canonical",
            ));
        }
        let source = projections
            .iter()
            .find(|projection| projection.value_id == *value_id && projection.axis == *source_axis)
            .cloned()
            .ok_or_else(|| invalid_plan("node token work source did not resolve exactly"))?;
        if projections
            .iter()
            .any(|projection| projection.canonical_extent != source.canonical_extent)
        {
            return Err(invalid_plan(
                "node token work projections disagree on canonical extent",
            ));
        }
        Ok(NodeWorkContract::Tokens {
            source,
            projections,
        })
    }

    fn validate_node_work_contract(node: &PlanNode) -> Result<(), VNextError> {
        let NodeWorkContract::Tokens {
            source,
            projections,
        } = &node.work
        else {
            return Ok(());
        };
        if projections.is_empty()
            || projections.windows(2).any(|pair| pair[0] >= pair[1])
            || !projections.contains(source)
            || projections
                .iter()
                .any(|projection| projection.canonical_extent != source.canonical_extent)
        {
            return Err(invalid_plan(format!(
                "node `{}` token work contract is empty or non-canonical",
                node.id
            )));
        }
        for projection in projections {
            let binding = node
                .values
                .iter()
                .find(|binding| {
                    binding.role() == projection.role
                        && binding.ordinal() == projection.ordinal
                        && binding.value_id() == &projection.value_id
                })
                .ok_or_else(|| {
                    invalid_plan(format!(
                        "node `{}` token projection has no exact value binding",
                        node.id
                    ))
                })?;
            let axis = usize::try_from(projection.axis)
                .map_err(|_| invalid_plan("node token projection axis exceeds usize"))?;
            if binding.usage() != BufferUsage::Activations
                || usize::try_from(projection.rank).ok()
                    != Some(binding.tensor().dimensions().len())
                || binding.tensor().dimensions().get(axis) != Some(&projection.canonical_extent)
            {
                return Err(invalid_plan(format!(
                    "node `{}` token projection differs from its resolved tensor",
                    node.id
                )));
            }
        }
        Ok(())
    }

    pub(super) fn extract_exact_aliases(
        operation: &OperationDescriptor,
        bindings: &[ResolvedValueBinding],
    ) -> Result<Vec<PlanExactAlias>, VNextError> {
        let inputs = &bindings[..operation.inputs.len()];
        let outputs = &bindings[operation.inputs.len()..];
        let mut aliases = Vec::new();
        for (output_ordinal, output) in outputs.iter().enumerate() {
            let (input_ordinal, kind) = match output.alias() {
                AliasPolicy::NoAlias => continue,
                AliasPolicy::MayAlias { tensor_index } => {
                    (*tensor_index, PlanExactAliasKind::MayAlias)
                }
                AliasPolicy::MustAlias { tensor_index } => {
                    (*tensor_index, PlanExactAliasKind::MustAlias)
                }
            };
            let input = inputs.get(input_ordinal as usize).ok_or_else(|| {
                invalid_plan(format!(
                    "operation `{}` alias input ordinal is out of range after validation",
                    operation.id
                ))
            })?;
            if output.storage() == input.storage() {
                aliases.push(PlanExactAlias {
                    output_value_id: output.value_id().clone(),
                    output_ordinal: output_ordinal as u32,
                    input_value_id: input.value_id().clone(),
                    input_ordinal,
                    kind,
                });
            } else if kind == PlanExactAliasKind::MustAlias {
                return Err(invalid_plan(format!(
                    "operation `{}` lost its mandatory exact alias proof",
                    operation.id
                )));
            }
        }
        Ok(aliases)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn validate_alias_liveness(
        family: &PreparedModelFamily,
        node: &ProgramNode,
        node_index: usize,
        last_consumers: &BTreeMap<ProgramValueId, usize>,
        program_outputs: &BTreeSet<ProgramValueId>,
        aliases: &[PlanExactAlias],
        bindings: &[ResolvedValueBinding],
    ) -> Result<(), VNextError> {
        for alias in aliases {
            let input = bindings
                .iter()
                .find(|binding| {
                    binding.role() == ResolvedValueRole::Input
                        && binding.ordinal() == alias.input_ordinal
                        && binding.value_id() == &alias.input_value_id
                })
                .ok_or_else(|| invalid_plan("exact alias input proof has no matching binding"))?;
            if family
                .program()
                .states()
                .iter()
                .any(|state| state.value_id == alias.input_value_id)
            {
                return Err(invalid_plan(format!(
                    "node `{}` output aliases state `{}` without a typed state transition contract",
                    node.id, alias.input_value_id
                )));
            }
            if input.usage() != BufferUsage::Activations
                || last_consumers.get(&alias.input_value_id) != Some(&node_index)
                || program_outputs.contains(&alias.input_value_id)
            {
                return Err(invalid_plan(format!(
                    "node `{}` aliases activation `{}` before its final legal consumer",
                    node.id, alias.input_value_id
                )));
            }
        }
        Ok(())
    }

    pub(super) fn derive_state_effects(
        family: &PreparedModelFamily,
        bindings: &[ResolvedValueBinding],
    ) -> Result<Vec<PlanStateEffect>, VNextError> {
        let mut effects = Vec::new();
        for state in family.program().states() {
            let mut reads = false;
            let mut writes = false;
            let state_bindings = bindings
                .iter()
                .filter(|binding| binding.value_id() == &state.value_id)
                .collect::<Vec<_>>();
            for binding in &state_bindings {
                match binding.access() {
                    TensorAccess::Read => reads = true,
                    TensorAccess::Write => writes = true,
                    TensorAccess::ReadWrite => {
                        reads = true;
                        writes = true;
                    }
                }
            }
            let access = match (reads, writes) {
                (false, false) => continue,
                (true, false) => TensorAccess::Read,
                (false, true) => TensorAccess::Write,
                (true, true) => TensorAccess::ReadWrite,
            };
            let lifetime = match state.lifetime {
                StateLifetime::Request => AllocationLifetime::Request,
                StateLifetime::Sequence => AllocationLifetime::Sequence,
                StateLifetime::Step => AllocationLifetime::Step,
            };
            let resource_ids = state_bindings
                .iter()
                .flat_map(|binding| binding.storage().components())
                .map(|component| component.resource_id().clone())
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect::<Vec<_>>();
            if resource_ids.is_empty() {
                return Err(invalid_plan(format!(
                    "state `{}` effect has no physical resource closure",
                    state.id
                )));
            }
            effects.push(PlanStateEffect {
                state_id: state.id.clone(),
                state_value_id: state.value_id.clone(),
                lifetime,
                access,
                resource_ids,
            });
        }
        if effects
            .windows(2)
            .any(|pair| pair[0].state_id >= pair[1].state_id)
        {
            return Err(invalid_plan("state effects are not canonical"));
        }
        Ok(effects)
    }

    pub(super) fn add_state_dependencies(
        node_id: &NodeId,
        effects: &[PlanStateEffect],
        tracker: &mut StateDependencyTracker,
        dependencies: &mut BTreeSet<NodeId>,
    ) {
        for effect in effects {
            let state_id = &effect.state_id;
            match effect.access {
                TensorAccess::Read => {
                    if let Some(writer) = tracker.last_writer.get(state_id) {
                        dependencies.insert(writer.clone());
                    }
                    tracker
                        .readers_since_write
                        .entry(state_id.clone())
                        .or_default()
                        .insert(node_id.clone());
                }
                TensorAccess::Write | TensorAccess::ReadWrite => {
                    if let Some(writer) = tracker.last_writer.get(state_id) {
                        dependencies.insert(writer.clone());
                    }
                    if let Some(readers) = tracker.readers_since_write.remove(state_id) {
                        dependencies.extend(readers);
                    }
                    tracker
                        .last_writer
                        .insert(state_id.clone(), node_id.clone());
                }
            }
        }
    }

    pub(super) fn validate_cross_node_value(
        binding: &ResolvedValueBinding,
        values: &mut BTreeMap<ProgramValueId, CanonicalValueBinding>,
    ) -> Result<(), VNextError> {
        let canonical = CanonicalValueBinding {
            tensor: binding.tensor().clone(),
            usage: binding.usage(),
            storage: binding.storage().clone(),
        };
        match values.get(binding.value_id()) {
            Some(previous) if previous != &canonical => Err(invalid_plan(format!(
                "value `{}` changes tensor or physical storage between nodes",
                binding.value_id()
            ))),
            Some(_) => Ok(()),
            None => {
                values.insert(binding.value_id().clone(), canonical);
                Ok(())
            }
        }
    }

    pub(super) fn validate_global_storage_aliasing(
        values: &BTreeMap<ProgramValueId, CanonicalValueBinding>,
        nodes: &[PlanNode],
    ) -> Result<(), VNextError> {
        let alias_classes = Self::alias_classes(nodes)?;
        let mut by_resource = BTreeMap::<ResourceId, Vec<GlobalValueRange>>::new();
        for (value_id, binding) in values {
            for component in binding.storage.components() {
                let end_bytes = component
                    .offset_bytes()
                    .checked_add(component.length_bytes())
                    .ok_or_else(|| invalid_plan("global value storage range overflows u64"))?;
                let ranges = by_resource
                    .entry(component.resource_id().clone())
                    .or_default();
                if let Some(previous) = ranges.iter().find(|previous| {
                    previous.value_id != *value_id
                        && previous.offset_bytes < end_bytes
                        && component.offset_bytes() < previous.end_bytes
                }) {
                    let same_alias_class = alias_classes.get(&previous.value_id).is_some()
                        && alias_classes.get(&previous.value_id) == alias_classes.get(value_id);
                    let previous_binding = values.get(&previous.value_id).ok_or_else(|| {
                        invalid_plan("global alias range has no canonical value binding")
                    })?;
                    if !same_alias_class || previous_binding.storage != binding.storage {
                        return Err(invalid_plan(format!(
                            "values `{}` and `{value_id}` have undeclared, partial, or non-equivalent overlap in physical resource `{}`",
                            previous.value_id,
                            component.resource_id()
                        )));
                    }
                }
                ranges.push(GlobalValueRange {
                    value_id: value_id.clone(),
                    offset_bytes: component.offset_bytes(),
                    end_bytes,
                });
            }
        }
        Ok(())
    }

    pub(super) fn alias_classes(
        nodes: &[PlanNode],
    ) -> Result<BTreeMap<ProgramValueId, ProgramValueId>, VNextError> {
        let mut graph = BTreeMap::<ProgramValueId, BTreeSet<ProgramValueId>>::new();
        for node in nodes {
            let mut previous_output_ordinal = None;
            for alias in &node.exact_aliases {
                if previous_output_ordinal.is_some_and(|ordinal| ordinal >= alias.output_ordinal) {
                    return Err(invalid_plan(format!(
                        "node `{}` exact aliases are not canonical",
                        node.id
                    )));
                }
                previous_output_ordinal = Some(alias.output_ordinal);
                let input = node
                    .values
                    .iter()
                    .find(|binding| {
                        binding.role() == ResolvedValueRole::Input
                            && binding.ordinal() == alias.input_ordinal
                            && binding.value_id() == &alias.input_value_id
                    })
                    .ok_or_else(|| invalid_plan("plan exact alias input binding is missing"))?;
                let output = node
                    .values
                    .iter()
                    .find(|binding| {
                        binding.role() == ResolvedValueRole::Output
                            && binding.ordinal() == alias.output_ordinal
                            && binding.value_id() == &alias.output_value_id
                    })
                    .ok_or_else(|| invalid_plan("plan exact alias output binding is missing"))?;
                let policy_matches = matches!(
                    (output.alias(), alias.kind),
                    (
                        AliasPolicy::MayAlias { tensor_index },
                        PlanExactAliasKind::MayAlias
                    ) if *tensor_index == alias.input_ordinal
                ) || matches!(
                    (output.alias(), alias.kind),
                    (
                        AliasPolicy::MustAlias { tensor_index },
                        PlanExactAliasKind::MustAlias
                    ) if *tensor_index == alias.input_ordinal
                );
                if !policy_matches
                    || input.storage() != output.storage()
                    || input.usage() != BufferUsage::Activations
                    || output.usage() != BufferUsage::Activations
                {
                    return Err(invalid_plan(format!(
                        "node `{}` exact alias proof differs from its bindings",
                        node.id
                    )));
                }
                graph
                    .entry(alias.input_value_id.clone())
                    .or_default()
                    .insert(alias.output_value_id.clone());
                graph
                    .entry(alias.output_value_id.clone())
                    .or_default()
                    .insert(alias.input_value_id.clone());
            }
        }

        let mut classes = BTreeMap::new();
        let mut visited = BTreeSet::new();
        for start in graph.keys() {
            if visited.contains(start) {
                continue;
            }
            let mut pending = vec![start.clone()];
            let mut members = BTreeSet::new();
            while let Some(value) = pending.pop() {
                if !visited.insert(value.clone()) {
                    continue;
                }
                members.insert(value.clone());
                if let Some(neighbors) = graph.get(&value) {
                    pending.extend(neighbors.iter().cloned());
                }
            }
            let representative = members
                .first()
                .cloned()
                .ok_or_else(|| invalid_plan("empty alias equivalence class"))?;
            for member in members {
                classes.insert(member, representative.clone());
            }
        }
        Ok(classes)
    }

    pub(super) fn validate_semantic_coverage(
        family: &PreparedModelFamily,
        bound: &BTreeSet<ProgramValueId>,
    ) -> Result<(), VNextError> {
        let required = family
            .program()
            .inputs()
            .iter()
            .cloned()
            .chain(
                family
                    .program()
                    .weights()
                    .iter()
                    .map(|weight| weight.value_id.clone()),
            )
            .chain(
                family
                    .program()
                    .states()
                    .iter()
                    .map(|state| state.value_id.clone()),
            )
            .chain(family.program().outputs().iter().cloned())
            .collect::<BTreeSet<_>>();
        if !required.is_subset(bound) {
            return Err(invalid_plan(format!(
                "semantic values lack physical bindings: {:?}",
                required.difference(bound).collect::<Vec<_>>()
            )));
        }
        Ok(())
    }

    pub(super) fn available_storage_profiles<P: RuntimePolicy>(
        requirement: &DynamicStorageRequirement,
        catalog: &CapabilityCatalog,
        policy: &P,
    ) -> BTreeSet<DynamicStorageProfile> {
        policy
            .dynamic_storage_profile_order()
            .iter()
            .copied()
            .filter(|profile| {
                catalog.device().dynamic_storage_profiles.contains(profile)
                    && requirement.accepts(*profile)
            })
            .collect()
    }

    pub(super) fn merge_storage_constraint(
        constraints: &mut BTreeMap<ResourceId, BTreeSet<DynamicStorageProfile>>,
        resource_id: ResourceId,
        accepted: BTreeSet<DynamicStorageProfile>,
    ) -> bool {
        if accepted.is_empty() {
            return false;
        }
        match constraints.get_mut(&resource_id) {
            Some(existing) => {
                existing.retain(|profile| accepted.contains(profile));
                !existing.is_empty()
            }
            None => {
                constraints.insert(resource_id, accepted);
                true
            }
        }
    }

    pub(super) fn select_joint_provider_storage<P: RuntimePolicy>(
        program_nodes: &[&ProgramNode],
        resolutions: &BTreeMap<NodeId, PlanNodeResolution>,
        catalog: &CapabilityCatalog,
        policy: &P,
    ) -> Result<JointProviderStorageSelection, VNextError> {
        let mut candidate_sets = Vec::with_capacity(program_nodes.len());
        for node in program_nodes {
            let resolution = resolutions.get(&node.id).ok_or_else(|| {
                invalid_plan(format!("node `{}` has no physical resolution", node.id))
            })?;
            let providers = catalog.providers_for_node(&node.id, &node.operation_id)?;
            let mut candidates = Vec::new();
            for resources in &resolution.provider_resource_candidates {
                let provider = providers
                    .iter()
                    .find(|provider| provider.provider_id() == resources.provider_id())
                    .ok_or_else(|| {
                        invalid_plan("provider resource candidate is absent from the catalog")
                    })?;
                let mut constraints = BTreeMap::new();
                let mut compatible = true;
                for binding in resolution
                    .values
                    .iter()
                    .filter(|binding| binding.usage() != BufferUsage::Weights)
                {
                    let Some(requirement) =
                        provider.dynamic_storage_for(binding.role(), binding.ordinal())
                    else {
                        compatible = false;
                        break;
                    };
                    let accepted = Self::available_storage_profiles(requirement, catalog, policy);
                    for component in binding.storage().components() {
                        if !Self::merge_storage_constraint(
                            &mut constraints,
                            component.resource_id().clone(),
                            accepted.clone(),
                        ) {
                            compatible = false;
                            break;
                        }
                    }
                    if !compatible {
                        break;
                    }
                }
                for (kind, workspace) in [
                    ("scratch", resources.scratch()),
                    ("binding", resources.binding()),
                    ("persistent", resources.persistent()),
                ] {
                    let Some(workspace) = workspace else {
                        continue;
                    };
                    let resource_id =
                        workspace_base_id(&node.id, kind, resources.estimate_fingerprint())?;
                    if !Self::merge_storage_constraint(
                        &mut constraints,
                        resource_id,
                        Self::available_storage_profiles(workspace.storage(), catalog, policy),
                    ) {
                        compatible = false;
                        break;
                    }
                }
                if compatible {
                    let is_preferred = resolution
                        .preferred_provider
                        .as_ref()
                        .is_some_and(|preferred| preferred == resources.provider_id());
                    candidates.push(JointProviderCandidate {
                        resources: resources.clone(),
                        allowed_profiles: constraints,
                        is_preferred,
                    });
                }
            }
            candidates.sort_by(|left, right| {
                let left_preferred = resolution
                    .preferred_provider
                    .as_ref()
                    .is_some_and(|preferred| preferred == left.resources.provider_id());
                let right_preferred = resolution
                    .preferred_provider
                    .as_ref()
                    .is_some_and(|preferred| preferred == right.resources.provider_id());
                right_preferred.cmp(&left_preferred).then(
                    left.resources
                        .provider_id()
                        .cmp(right.resources.provider_id()),
                )
            });
            if candidates.is_empty() {
                return Err(invalid_plan(format!(
                    "node `{}` has no provider candidate with an available storage profile",
                    node.id
                )));
            }
            candidate_sets.push(candidates);
        }

        let (chosen, resource_profiles) = Self::solve_joint_provider_candidates(
            &candidate_sets,
            policy.dynamic_storage_profile_order(),
        )?;

        let mut storage_rejections = BTreeMap::new();
        for (index, node) in program_nodes.iter().enumerate() {
            let resolution = resolutions
                .get(&node.id)
                .ok_or_else(|| invalid_plan("joint storage resolution disappeared"))?;
            let Some(preferred) = resolution.preferred_provider.as_ref() else {
                continue;
            };
            if chosen[index].provider_id() == preferred {
                continue;
            }
            let Some(preferred_candidate) = candidate_sets[index]
                .iter()
                .find(|candidate| candidate.resources.provider_id() == preferred)
            else {
                if let Some(reason) = resolution.provider_resolution_rejections.get(preferred) {
                    storage_rejections.insert(
                        node.id.clone(),
                        RejectedProvider {
                            provider_id: preferred.clone(),
                            reasons: reason.clone(),
                        },
                    );
                }
                continue;
            };
            let resource_ids =
                storage_incompatible_resource_ids(preferred_candidate, &resource_profiles);
            if resource_ids.is_empty() {
                return Err(invalid_plan(format!(
                    "preferred provider `{preferred}` was not selected for node `{}` without a storage conflict",
                    node.id
                )));
            }
            storage_rejections.insert(
                node.id.clone(),
                RejectedProvider {
                    provider_id: preferred.clone(),
                    reasons: PlanProviderRejectReason::StorageIncompatible { resource_ids },
                },
            );
        }

        let node_resources = program_nodes
            .iter()
            .zip(chosen)
            .map(|(node, resources)| (node.id.clone(), resources))
            .collect();
        Ok(JointProviderStorageSelection {
            node_resources,
            resource_profiles,
            storage_rejections,
        })
    }

    pub(super) fn solve_joint_provider_candidates(
        candidate_sets: &[Vec<JointProviderCandidate>],
        profile_order: &[DynamicStorageProfile],
    ) -> Result<
        (
            Vec<ProviderResourcePlan>,
            BTreeMap<ResourceId, DynamicStorageProfile>,
        ),
        VNextError,
    > {
        if candidate_sets.is_empty() || candidate_sets.iter().any(Vec::is_empty) {
            return Err(invalid_plan(
                "joint provider/storage search has an empty candidate set",
            ));
        }
        if profile_order.is_empty() {
            return Err(invalid_plan(
                "joint provider/storage search has an empty profile order",
            ));
        }

        let components = joint_candidate_components(candidate_sets);
        let mut chosen = vec![None; candidate_sets.len()];
        let mut resource_profiles = BTreeMap::new();
        for component in components {
            let solution =
                Self::solve_joint_provider_component(&component, candidate_sets, profile_order)?;
            for (node_index, resources) in component.iter().copied().zip(solution.chosen) {
                if chosen[node_index].replace(resources).is_some() {
                    return Err(invalid_plan(
                        "joint storage component assigned one node more than once",
                    ));
                }
            }
            for (resource_id, profile) in solution.resource_profiles {
                if resource_profiles.insert(resource_id, profile).is_some() {
                    return Err(invalid_plan(
                        "joint storage components overlap one resource",
                    ));
                }
            }
        }
        let chosen = chosen
            .into_iter()
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| invalid_plan("joint storage components omitted one node"))?;
        Ok((chosen, resource_profiles))
    }

    pub(super) fn solve_joint_provider_component(
        component: &[usize],
        candidate_sets: &[Vec<JointProviderCandidate>],
        profile_order: &[DynamicStorageProfile],
    ) -> Result<JointComponentSolution, VNextError> {
        let mut frontier = BTreeMap::from([(
            BTreeMap::<ResourceId, BTreeSet<DynamicStorageProfile>>::new(),
            JointPartialSelection::default(),
        )]);
        for node_index in component {
            let mut next = BTreeMap::<
                BTreeMap<ResourceId, BTreeSet<DynamicStorageProfile>>,
                JointPartialSelection,
            >::new();
            for (constraints, partial) in frontier {
                for candidate in &candidate_sets[*node_index] {
                    let mut next_constraints = constraints.clone();
                    if candidate
                        .allowed_profiles
                        .iter()
                        .any(|(resource_id, accepted)| {
                            !Self::merge_storage_constraint(
                                &mut next_constraints,
                                resource_id.clone(),
                                accepted.clone(),
                            )
                        })
                    {
                        continue;
                    }
                    let mut next_partial = partial.clone();
                    next_partial.chosen.push(candidate.resources.clone());
                    next_partial.preferred.push(candidate.is_preferred);
                    match next.entry(next_constraints) {
                        std::collections::btree_map::Entry::Vacant(entry) => {
                            entry.insert(next_partial);
                        }
                        std::collections::btree_map::Entry::Occupied(mut entry) => {
                            if joint_partial_precedes(&next_partial, entry.get()) {
                                entry.insert(next_partial);
                            }
                        }
                    }
                }
            }
            if next.is_empty() {
                return Err(invalid_plan(
                    "no joint provider/storage assignment satisfies shared resource constraints",
                ));
            }
            frontier = next;
        }

        let mut best: Option<(JointSelectionObjective, JointComponentSolution)> = None;
        for (constraints, partial) in frontier {
            let resource_profiles = constraints
                .into_iter()
                .map(|(resource_id, accepted)| {
                    let (rank, profile) = profile_order
                        .iter()
                        .copied()
                        .enumerate()
                        .find(|(_, profile)| accepted.contains(profile))
                        .ok_or_else(|| {
                            invalid_plan("joint storage solution lost policy-ordered profile")
                        })?;
                    Ok((resource_id, (rank, profile)))
                })
                .collect::<Result<BTreeMap<_, _>, VNextError>>()?;
            let objective = JointSelectionObjective::new(
                &partial,
                resource_profiles.values().map(|(rank, _)| *rank),
                profile_order.len(),
            )?;
            let solution = JointComponentSolution {
                chosen: partial.chosen,
                resource_profiles: resource_profiles
                    .into_iter()
                    .map(|(resource_id, (_, profile))| (resource_id, profile))
                    .collect(),
            };
            if best
                .as_ref()
                .is_none_or(|(current, _)| objective.precedes(current))
            {
                best = Some((objective, solution));
            }
        }
        best.map(|(_, solution)| solution).ok_or_else(|| {
            invalid_plan("no joint provider/storage assignment satisfies one component")
        })
    }

    pub(super) fn select_provider(
        node: &ProgramNode,
        operation: &OperationDescriptor,
        catalog: &CapabilityCatalog,
        resolution_required_capabilities: &BTreeSet<CapabilityId>,
        preferred_provider: Option<&ProviderId>,
        storage_selected_provider: &ProviderId,
        storage_rejection: Option<RejectedProvider>,
        required_weight_formats: &BTreeSet<WeightFormatId>,
        required_quantization_formats: &BTreeSet<QuantizationFormatId>,
    ) -> Result<ProviderSelection, VNextError> {
        let required_capabilities = operation
            .provider
            .required_capabilities
            .union(resolution_required_capabilities)
            .cloned()
            .collect::<BTreeSet<_>>();
        let request = ProviderCompatibilityRequest::new(
            node.operation_id.clone(),
            node.required_version,
            required_capabilities,
            required_weight_formats.clone(),
            required_quantization_formats.clone(),
        )?;
        let report = catalog.provider_compatibility(request)?;
        report.require_compatible_for_node(&catalog.device().id, &node.id)?;
        if !report
            .compatible_provider_ids()
            .contains(storage_selected_provider)
        {
            return Err(invalid_plan(format!(
                "joint storage solver selected incompatible provider `{storage_selected_provider}`"
            )));
        }
        let selected_provider = storage_selected_provider.clone();
        let selection_reason = match preferred_provider {
            Some(preferred) if preferred == &selected_provider => {
                ProviderSelectionReason::PreferredCompatible
            }
            Some(_) => ProviderSelectionReason::FallbackFromPreferred,
            None => ProviderSelectionReason::DeterministicCompatible,
        };
        let mut rejected_providers = report
            .rejected()
            .iter()
            .map(|rejection| RejectedProvider {
                provider_id: rejection.provider_id.clone(),
                reasons: PlanProviderRejectReason::Incompatible(rejection.reasons.clone()),
            })
            .collect::<Vec<_>>();
        if let Some(preferred) = preferred_provider {
            let registered = catalog
                .providers_for_node(&node.id, &node.operation_id)?
                .iter()
                .any(|provider| provider.provider_id() == preferred);
            if !registered {
                rejected_providers.push(RejectedProvider {
                    provider_id: preferred.clone(),
                    reasons: PlanProviderRejectReason::NotRegistered,
                });
            }
        }
        if let Some(rejection) = storage_rejection {
            if rejected_providers
                .iter()
                .any(|existing| existing.provider_id == rejection.provider_id)
            {
                return Err(invalid_plan(
                    "provider has duplicate compatibility and storage rejection evidence",
                ));
            }
            rejected_providers.push(rejection);
        }
        if let Some(preferred) =
            preferred_provider.filter(|preferred| *preferred != &selected_provider)
        {
            if !rejected_providers
                .iter()
                .any(|rejection| &rejection.provider_id == preferred)
            {
                return Err(invalid_plan(format!(
                    "preferred provider `{preferred}` fallback lacks typed rejection evidence"
                )));
            }
        }
        rejected_providers.sort_by(|left, right| left.provider_id.cmp(&right.provider_id));
        Ok(ProviderSelection {
            requested_provider: preferred_provider.cloned(),
            selected_provider,
            selection_reason,
            rejected_providers,
        })
    }

    pub(super) fn validate_provider_selection_evidence(
        selection: &ProviderSelection,
    ) -> Result<(), VNextError> {
        if selection
            .rejected_providers
            .windows(2)
            .any(|pair| pair[0].provider_id >= pair[1].provider_id)
            || selection
                .rejected_providers
                .iter()
                .any(|rejection| rejection.provider_id == selection.selected_provider)
            || selection.rejected_providers.iter().any(|rejection| {
                matches!(
                    &rejection.reasons,
                    PlanProviderRejectReason::StorageIncompatible { resource_ids }
                        if resource_ids.is_empty()
                            || resource_ids.windows(2).any(|pair| pair[0] >= pair[1])
                )
            })
        {
            return Err(invalid_plan(
                "provider rejection evidence is duplicate, non-canonical, or rejects the selected provider",
            ));
        }
        match (
            selection.requested_provider.as_ref(),
            selection.selection_reason,
        ) {
            (None, ProviderSelectionReason::DeterministicCompatible) => {}
            (Some(requested), ProviderSelectionReason::PreferredCompatible)
                if requested == &selection.selected_provider => {}
            (Some(requested), ProviderSelectionReason::FallbackFromPreferred)
                if requested != &selection.selected_provider
                    && selection
                        .rejected_providers
                        .iter()
                        .any(|rejection| &rejection.provider_id == requested) => {}
            _ => return Err(invalid_plan(
                "provider selection reason is inconsistent with preference and rejection evidence",
            )),
        }
        Ok(())
    }

    pub(super) fn build_memory_plan(
        family: &PreparedModelFamily,
        device_capacity_bytes: u64,
        policy_capacity_bytes: u64,
        reserve_bytes: u64,
        maximum_active_sequences: u32,
        maximum_scheduled_tokens: u64,
        nodes: &[PlanNode],
        selected_resource_profiles: &BTreeMap<ResourceId, DynamicStorageProfile>,
        reusable_execution_policy: Option<&ReusableExecutionPolicy>,
    ) -> Result<MemoryPlan, VNextError> {
        validate_scheduled_token_ceiling(maximum_scheduled_tokens)?;
        let program_inputs = family
            .program()
            .inputs()
            .iter()
            .cloned()
            .collect::<BTreeSet<_>>();
        let program_outputs = family
            .program()
            .outputs()
            .iter()
            .cloned()
            .collect::<BTreeSet<_>>();
        let state_initializations = family
            .program()
            .states()
            .iter()
            .map(|state| (state.value_id.clone(), state.initialization))
            .collect::<BTreeMap<_, _>>();
        let mut values = BTreeMap::<ResourceId, ValueAllocationAccumulator>::new();
        let mut static_allocations = Vec::new();
        let mut dynamic_descriptors = Vec::new();
        let workspace_layout_fingerprint = workspace_storage_layout_fingerprint()?;
        for node in nodes {
            let value_alignment = node.provider_resources.value_alignment_bytes;
            for binding in &node.values {
                let logical_layout_fingerprint =
                    tensor_storage_layout_fingerprint(binding.tensor().layout())?;
                for component in binding.storage().components() {
                    if component.offset_bytes() % value_alignment != 0 {
                        return Err(invalid_plan(format!(
                            "resource `{}` offset is not aligned for provider `{}`",
                            component.resource_id(),
                            node.provider_resources.provider_id
                        )));
                    }
                    let end = component
                        .offset_bytes()
                        .checked_add(component.length_bytes())
                        .ok_or_else(|| invalid_plan("resource byte range overflows u64"))?;
                    let token_projection = node
                        .work
                        .token_projection(binding.role(), binding.ordinal())
                        .map(|projection| {
                            if component.offset_bytes() != 0
                                || component.length_bytes() % projection.canonical_extent() != 0
                            {
                                return Err(invalid_plan(format!(
                                    "token-scaled resource `{}` is not one exact canonical tensor range",
                                    component.resource_id()
                                )));
                            }
                            Ok((
                                component.length_bytes() / projection.canonical_extent(),
                                projection.canonical_extent(),
                            ))
                        })
                        .transpose()?;
                    let demand = Self::value_resource_demand(
                        family,
                        binding.value_id(),
                        binding.usage(),
                        end,
                        token_projection,
                        maximum_scheduled_tokens,
                        &program_inputs,
                        &program_outputs,
                    )?;
                    let initialization = state_initializations
                        .get(binding.value_id())
                        .copied()
                        .unwrap_or(StateInitialization::None);
                    values
                        .entry(component.resource_id().clone())
                        .and_modify(|allocation| {
                            allocation.merge_result =
                                allocation.merge_result.take().and_then(|_| {
                                    allocation.merge(
                                        end,
                                        value_alignment,
                                        binding.usage(),
                                        component.element_type(),
                                        demand,
                                        initialization,
                                        logical_layout_fingerprint.clone(),
                                    )
                                });
                        })
                        .or_insert_with(|| ValueAllocationAccumulator {
                            end_bytes: end,
                            alignment_bytes: value_alignment,
                            usage: binding.usage(),
                            element_type: component.element_type(),
                            demand,
                            initialization,
                            logical_layout_fingerprints: BTreeSet::from([
                                logical_layout_fingerprint.clone(),
                            ]),
                            merge_result: Some(()),
                        });
                }
            }
            if let Some(workspace) = &node.provider_resources.scratch {
                let resource_id = node.scratch_resource.clone().ok_or_else(|| {
                    invalid_plan(format!(
                        "node `{}` scratch base identity is missing",
                        node.id
                    ))
                })?;
                if workspace.scope != ProviderWorkspaceScope::Invocation {
                    return Err(invalid_plan(format!(
                        "node `{}` scratch workspace is not invocation scoped",
                        node.id
                    )));
                }
                let storage = DynamicStorageContract::new(
                    *selected_resource_profiles
                        .get(&resource_id)
                        .ok_or_else(|| {
                            invalid_plan(format!(
                                "scratch resource `{resource_id}` has no selected storage profile"
                            ))
                        })?,
                    workspace_layout_fingerprint.clone(),
                )?;
                dynamic_descriptors.push(DynamicResourceDescriptor::new(
                    resource_id,
                    workspace
                        .size_formula
                        .bind_runtime_limits(maximum_active_sequences, maximum_scheduled_tokens)?,
                    workspace.alignment_bytes,
                    BufferUsage::Scratch,
                    ElementType::U8,
                    AllocationLifetime::Invocation,
                    AllocationKind::Scratch {
                        node_id: node.id.clone(),
                    },
                    storage,
                    StateInitialization::None,
                    maximum_active_sequences,
                )?);
            } else if node.scratch_resource.is_some() {
                return Err(invalid_plan(format!(
                    "node `{}` has scratch resources without a provider estimate",
                    node.id
                )));
            }
            if let Some(workspace) = &node.provider_resources.binding {
                let resource_id = node.binding_resource.clone().ok_or_else(|| {
                    invalid_plan(format!(
                        "node `{}` binding workspace base identity is missing",
                        node.id
                    ))
                })?;
                if workspace.scope != ProviderWorkspaceScope::Invocation {
                    return Err(invalid_plan(format!(
                        "node `{}` binding workspace is not invocation scoped",
                        node.id
                    )));
                }
                let storage = DynamicStorageContract::new(
                    *selected_resource_profiles
                        .get(&resource_id)
                        .ok_or_else(|| {
                            invalid_plan(format!(
                                "binding resource `{resource_id}` has no selected storage profile"
                            ))
                        })?,
                    workspace_layout_fingerprint.clone(),
                )?;
                dynamic_descriptors.push(DynamicResourceDescriptor::new(
                    resource_id,
                    workspace
                        .size_formula
                        .bind_runtime_limits(maximum_active_sequences, maximum_scheduled_tokens)?,
                    workspace.alignment_bytes,
                    BufferUsage::Binding,
                    ElementType::U8,
                    AllocationLifetime::Invocation,
                    AllocationKind::Binding {
                        node_id: node.id.clone(),
                    },
                    storage,
                    StateInitialization::None,
                    maximum_active_sequences,
                )?);
            } else if node.binding_resource.is_some() {
                return Err(invalid_plan(format!(
                    "node `{}` has binding resources without a provider estimate",
                    node.id
                )));
            }
            if let Some(workspace) = &node.provider_resources.persistent {
                let resource_id = node.persistent_resource.clone().ok_or_else(|| {
                    invalid_plan(format!(
                        "node `{}` persistent base identity is missing",
                        node.id
                    ))
                })?;
                match workspace.scope {
                    ProviderWorkspaceScope::Plan => {
                        let bytes = workspace.fixed_bytes().ok_or_else(|| {
                            invalid_plan(format!(
                                "node `{}` plan workspace does not have a fixed formula",
                                node.id
                            ))
                        })?;
                        let storage = DynamicStorageContract::new(
                            *selected_resource_profiles
                                .get(&resource_id)
                                .ok_or_else(|| {
                                    invalid_plan(format!(
                                    "plan workspace `{resource_id}` has no selected storage profile"
                                ))
                                })?,
                            workspace_layout_fingerprint.clone(),
                        )?;
                        static_allocations.push(ResourceAllocation::new(
                            resource_id,
                            bytes,
                            workspace.alignment_bytes,
                            BufferUsage::Persistent,
                            ElementType::U8,
                            AllocationKind::Persistent {
                                node_id: node.id.clone(),
                            },
                            storage,
                        )?);
                    }
                    scope @ (ProviderWorkspaceScope::Request
                    | ProviderWorkspaceScope::Sequence
                    | ProviderWorkspaceScope::Step) => {
                        let lifetime = match scope {
                            ProviderWorkspaceScope::Request => AllocationLifetime::Request,
                            ProviderWorkspaceScope::Sequence => AllocationLifetime::Sequence,
                            ProviderWorkspaceScope::Step => AllocationLifetime::Step,
                            ProviderWorkspaceScope::Plan | ProviderWorkspaceScope::Invocation => {
                                unreachable!()
                            }
                        };
                        let storage = DynamicStorageContract::new(
                            *selected_resource_profiles.get(&resource_id).ok_or_else(|| {
                                invalid_plan(format!(
                                    "persistent resource `{resource_id}` has no selected storage profile"
                                ))
                            })?,
                            workspace_layout_fingerprint.clone(),
                        )?;
                        dynamic_descriptors.push(DynamicResourceDescriptor::new(
                            resource_id,
                            workspace.size_formula.bind_runtime_limits(
                                maximum_active_sequences,
                                maximum_scheduled_tokens,
                            )?,
                            workspace.alignment_bytes,
                            BufferUsage::Persistent,
                            ElementType::U8,
                            lifetime,
                            AllocationKind::Persistent {
                                node_id: node.id.clone(),
                            },
                            storage,
                            StateInitialization::None,
                            maximum_active_sequences,
                        )?);
                    }
                    ProviderWorkspaceScope::Invocation => {
                        return Err(invalid_plan(format!(
                            "node `{}` persistent workspace cannot be invocation scoped",
                            node.id
                        )));
                    }
                }
            } else if node.persistent_resource.is_some() {
                return Err(invalid_plan(format!(
                    "node `{}` has persistent resources without a provider estimate",
                    node.id
                )));
            }
        }
        for (resource_id, accumulator) in values {
            accumulator.merge_result.ok_or_else(|| {
                invalid_plan(format!(
                    "resource `{resource_id}` has conflicting usage, dtype, lifetime, or demand"
                ))
            })?;
            let logical_layout_fingerprint = canonical_fingerprint(
                &accumulator.logical_layout_fingerprints,
                "fingerprint dynamic resource tensor layout classes",
            )?;
            match accumulator.demand {
                ValueResourceDemand::PlanStatic => {
                    let storage = DynamicStorageContract::new(
                        static_contiguous_storage_profile()?,
                        logical_layout_fingerprint,
                    )?;
                    static_allocations.push(ResourceAllocation::new(
                        resource_id,
                        accumulator.end_bytes,
                        accumulator.alignment_bytes,
                        accumulator.usage,
                        accumulator.element_type,
                        AllocationKind::Value,
                        storage,
                    )?);
                }
                demand => {
                    let storage = DynamicStorageContract::new(
                        *selected_resource_profiles.get(&resource_id).ok_or_else(|| {
                            invalid_plan(format!(
                                "dynamic value resource `{resource_id}` has no selected storage profile"
                            ))
                        })?,
                        logical_layout_fingerprint,
                    )?;
                    dynamic_descriptors.push(DynamicResourceDescriptor::new(
                        resource_id,
                        demand.dynamic_demand(accumulator.end_bytes)?,
                        accumulator.alignment_bytes,
                        accumulator.usage,
                        accumulator.element_type,
                        demand.lifetime().ok_or_else(|| {
                            invalid_plan("dynamic value demand lost its scoped lifetime")
                        })?,
                        AllocationKind::Value,
                        storage,
                        accumulator.initialization,
                        maximum_active_sequences,
                    )?);
                }
            }
        }
        MemoryPlan::from_core(
            device_capacity_bytes,
            policy_capacity_bytes,
            reserve_bytes,
            maximum_active_sequences,
            static_allocations,
            dynamic_descriptors,
            nodes,
            reusable_execution_policy,
        )
    }

    pub(super) fn value_resource_demand(
        family: &PreparedModelFamily,
        value_id: &ProgramValueId,
        usage: BufferUsage,
        minimum_bytes: u64,
        token_projection: Option<(u64, u64)>,
        maximum_scheduled_tokens: u64,
        program_inputs: &BTreeSet<ProgramValueId>,
        program_outputs: &BTreeSet<ProgramValueId>,
    ) -> Result<ValueResourceDemand, VNextError> {
        if family
            .program()
            .weights()
            .iter()
            .any(|weight| &weight.value_id == value_id)
        {
            return Ok(ValueResourceDemand::PlanStatic);
        }
        let state = family
            .program()
            .states()
            .iter()
            .find(|state| &state.value_id == value_id);
        let Some(state) = state else {
            if usage != BufferUsage::Activations {
                return Err(invalid_plan(format!(
                    "non-state value `{value_id}` is not backed by activation memory"
                )));
            }
            let lifetime =
                if program_inputs.contains(value_id) || program_outputs.contains(value_id) {
                    AllocationLifetime::Request
                } else {
                    AllocationLifetime::Step
                };
            if let Some((bytes_per_token, canonical_tokens)) = token_projection {
                if bytes_per_token == 0 || canonical_tokens == 0 {
                    return Err(invalid_plan(
                        "token-scaled activation has zero bytes or canonical tokens",
                    ));
                }
                let maximum_tokens = Self::activation_token_capacity(
                    lifetime,
                    canonical_tokens,
                    maximum_scheduled_tokens,
                )?;
                return Ok(ValueResourceDemand::TokenScaled {
                    lifetime,
                    bytes_per_token,
                    maximum_tokens,
                });
            }
            return Ok(ValueResourceDemand::Fixed { lifetime });
        };
        let lifetime = match state.lifetime {
            StateLifetime::Request => AllocationLifetime::Request,
            StateLifetime::Sequence => AllocationLifetime::Sequence,
            StateLifetime::Step => AllocationLifetime::Step,
        };
        state.capacity_demand.validate(state.tensor.byte_len()?)?;
        match state.capacity_demand {
            StateCapacityDemand::FixedPerScope => Ok(ValueResourceDemand::Fixed { lifetime }),
            StateCapacityDemand::TokenScaled {
                bytes_per_token,
                maximum_tokens,
            } => {
                if bytes_per_token < minimum_bytes {
                    return Err(invalid_plan(
                        "token-scaled state demand is smaller than its resolved resource range",
                    ));
                }
                Ok(ValueResourceDemand::TokenScaled {
                    lifetime,
                    bytes_per_token,
                    maximum_tokens,
                })
            }
        }
    }

    pub(super) fn activation_token_capacity(
        lifetime: AllocationLifetime,
        canonical_tokens: u64,
        maximum_scheduled_tokens: u64,
    ) -> Result<u64, VNextError> {
        if canonical_tokens == 0 {
            return Err(invalid_plan(
                "token-scaled activation has zero canonical tokens",
            ));
        }
        if lifetime == AllocationLifetime::Request {
            Ok(canonical_tokens)
        } else {
            validate_scheduled_token_ceiling(maximum_scheduled_tokens)?;
            Ok(maximum_scheduled_tokens)
        }
    }

    pub(super) fn plan_id_for_hash(hash: &PlanHash) -> Result<PlanId, VNextError> {
        PlanId::new(format!("plan/sha256/{}", hash.as_str()))
    }

    pub(super) fn validate_internal(&self) -> Result<(), VNextError> {
        if self.payload.schema != EXECUTION_PLAN_SCHEMA {
            return Err(VNextError::UnsupportedPlanSchema {
                expected_major: EXECUTION_PLAN_SCHEMA.major,
                expected_minor: EXECUTION_PLAN_SCHEMA.minor,
                actual_major: self.payload.schema.major,
                actual_minor: self.payload.schema.minor,
            });
        }
        let computed = PlanHash::new(canonical_fingerprint(
            &PlanHashMaterial::from(&self.payload),
            "validate execution plan hash",
        )?)?;
        if computed != self.plan_hash {
            return Err(VNextError::PlanHashMismatch {
                expected: computed.to_string(),
                actual: self.plan_hash.to_string(),
            });
        }
        if self.payload.plan_id != Self::plan_id_for_hash(&computed)? {
            return Err(invalid_plan(
                "plan id is not derived from the semantic plan hash",
            ));
        }
        if self.payload.nodes.is_empty()
            || !is_canonical_sha256(&self.payload.prepared_family_fingerprint)
            || !is_canonical_sha256(&self.payload.program_fingerprint)
            || !is_canonical_sha256(&self.payload.capability_catalog_fingerprint)
            || !is_canonical_sha256(&self.payload.device_runtime_implementation_fingerprint)
            || !is_canonical_sha256(&self.payload.policy_fingerprint)
            || self.payload.maximum_scheduled_tokens == 0
        {
            return Err(invalid_plan("plan provenance or node set is invalid"));
        }
        self.payload.memory.validate()?;
        let dynamic_capacity_bytes = self
            .payload
            .memory
            .usable_capacity_bytes
            .checked_sub(self.payload.memory.static_bytes)
            .ok_or_else(|| invalid_plan("static memory exceeds usable capacity"))?;
        let base_pools = MemoryPlan::derive_dynamic_pools(
            &self.payload.memory.dynamic_descriptors,
            &self.payload.nodes,
            dynamic_capacity_bytes,
        )?;
        let expected_reusable_execution = self
            .payload
            .memory
            .reusable_execution
            .as_ref()
            .map(|actual| {
                MemoryPlan::derive_reusable_execution(
                    &actual.policy()?,
                    self.payload.nodes.len(),
                    &self.payload.memory.dynamic_descriptors,
                    &base_pools,
                )
            })
            .transpose()?;
        if self.payload.memory.reusable_execution != expected_reusable_execution {
            return Err(invalid_plan(
                "reusable execution budgets are not derived from plan resources",
            ));
        }
        let reusable_workspace_ceilings = expected_reusable_execution
            .as_ref()
            .map(ReusableExecutionMemoryPlan::pool_workspace_ceilings)
            .transpose()?
            .unwrap_or_default();
        let expected_pools = MemoryPlan::derive_dynamic_pools_with_reusable(
            &self.payload.memory.dynamic_descriptors,
            &self.payload.nodes,
            dynamic_capacity_bytes,
            &reusable_workspace_ceilings,
        )?;
        if self.payload.memory.dynamic_pools != expected_pools {
            return Err(invalid_plan(
                "memory pools or invocation reuse are not derived from plan dependencies",
            ));
        }
        let static_allocations = self
            .payload
            .memory
            .static_allocations
            .iter()
            .map(|allocation| (allocation.resource_id.clone(), allocation))
            .collect::<BTreeMap<_, _>>();
        let dynamic_descriptors = self
            .payload
            .memory
            .dynamic_descriptors
            .iter()
            .map(|descriptor| (descriptor.base_resource_id.clone(), descriptor))
            .collect::<BTreeMap<_, _>>();
        let mut seen_nodes = BTreeSet::new();
        let mut canonical_values = BTreeMap::new();
        for node in &self.payload.nodes {
            node.provider_resources.validate_shape()?;
            Self::validate_provider_selection_evidence(&node.selection)?;
            Self::validate_node_work_contract(node)?;
            if !seen_nodes.insert(node.id.clone())
                || !is_canonical_sha256(&node.provider_implementation_fingerprint)
                || node
                    .dependencies
                    .iter()
                    .any(|dependency| dependency == &node.id || !seen_nodes.contains(dependency))
                || node.dependencies.windows(2).any(|pair| pair[0] >= pair[1])
                || node
                    .state_effects
                    .windows(2)
                    .any(|pair| pair[0].state_id >= pair[1].state_id)
                || node.resources.iter().collect::<BTreeSet<_>>().len() != node.resources.len()
                || node.resources.iter().any(|resource| {
                    !static_allocations.contains_key(resource)
                        && !dynamic_descriptors.contains_key(resource)
                })
                || node.provider_resources.provider_id != node.selection.selected_provider
            {
                return Err(invalid_plan(format!(
                    "node `{}` identity, dependency, or resource closure is invalid",
                    node.id
                )));
            }
            let expected_resources = node
                .values
                .iter()
                .flat_map(|binding| binding.storage().components())
                .map(|component| component.resource_id().clone())
                .chain(node.scratch_resource.iter().cloned())
                .chain(node.binding_resource.iter().cloned())
                .chain(node.persistent_resource.iter().cloned())
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect::<Vec<_>>();
            if node.resources != expected_resources {
                return Err(invalid_plan(format!(
                    "node `{}` resource closure is not canonical",
                    node.id
                )));
            }
            for effect in &node.state_effects {
                if !matches!(
                    effect.lifetime,
                    AllocationLifetime::Request
                        | AllocationLifetime::Sequence
                        | AllocationLifetime::Step
                ) || effect.resource_ids.is_empty()
                    || effect
                        .resource_ids
                        .windows(2)
                        .any(|pair| pair[0] >= pair[1])
                {
                    return Err(invalid_plan(format!(
                        "node `{}` state effect has an invalid lifetime or resource closure",
                        node.id
                    )));
                }
                let matching = node
                    .values
                    .iter()
                    .filter(|binding| binding.value_id() == &effect.state_value_id)
                    .collect::<Vec<_>>();
                let expected_effect_resources = matching
                    .iter()
                    .flat_map(|binding| binding.storage().components())
                    .map(|component| component.resource_id().clone())
                    .collect::<BTreeSet<_>>()
                    .into_iter()
                    .collect::<Vec<_>>();
                let reads = matching.iter().any(|binding| {
                    matches!(
                        binding.access(),
                        TensorAccess::Read | TensorAccess::ReadWrite
                    )
                });
                let writes = matching.iter().any(|binding| {
                    matches!(
                        binding.access(),
                        TensorAccess::Write | TensorAccess::ReadWrite
                    )
                });
                let expected_access = match (reads, writes) {
                    (true, false) => Some(TensorAccess::Read),
                    (false, true) => Some(TensorAccess::Write),
                    (true, true) => Some(TensorAccess::ReadWrite),
                    (false, false) => None,
                };
                if effect.resource_ids != expected_effect_resources
                    || expected_access != Some(effect.access)
                    || effect.resource_ids.iter().any(|resource_id| {
                        dynamic_descriptors
                            .get(resource_id)
                            .is_none_or(|descriptor| descriptor.lifetime != effect.lifetime)
                    })
                {
                    return Err(invalid_plan(format!(
                        "node `{}` state effect is not derived from its typed bindings",
                        node.id
                    )));
                }
            }
            if node.scratch_resource.is_some() != node.provider_resources.scratch.is_some()
                || node.binding_resource.is_some() != node.provider_resources.binding.is_some()
                || node.persistent_resource.is_some()
                    != node.provider_resources.persistent.is_some()
            {
                return Err(invalid_plan(format!(
                    "node `{}` workspace base identity presence differs from its provider estimate",
                    node.id
                )));
            }
            if let Some(resource_id) = &node.scratch_resource {
                let descriptor = dynamic_descriptors.get(resource_id).ok_or_else(|| {
                    invalid_plan(format!("node `{}` scratch descriptor is missing", node.id))
                })?;
                let workspace = node.provider_resources.scratch.as_ref().ok_or_else(|| {
                    invalid_plan(format!("node `{}` scratch estimate is missing", node.id))
                })?;
                if descriptor.demand
                    != workspace.size_formula.bind_runtime_limits(
                        self.payload.memory.maximum_active_sequences,
                        self.payload.maximum_scheduled_tokens,
                    )?
                    || descriptor.alignment_bytes != workspace.alignment_bytes
                    || descriptor.usage != BufferUsage::Scratch
                    || descriptor.lifetime != AllocationLifetime::Invocation
                    || descriptor.theoretical_maximum_instances
                        != self.payload.memory.maximum_active_sequences
                    || descriptor.kind
                        != (AllocationKind::Scratch {
                            node_id: node.id.clone(),
                        })
                {
                    return Err(invalid_plan(format!(
                        "node `{}` scratch descriptor differs from its provider estimate",
                        node.id
                    )));
                }
            }
            if let Some(resource_id) = &node.binding_resource {
                let descriptor = dynamic_descriptors.get(resource_id).ok_or_else(|| {
                    invalid_plan(format!("node `{}` binding descriptor is missing", node.id))
                })?;
                let workspace = node.provider_resources.binding.as_ref().ok_or_else(|| {
                    invalid_plan(format!("node `{}` binding estimate is missing", node.id))
                })?;
                if descriptor.demand
                    != workspace.size_formula.bind_runtime_limits(
                        self.payload.memory.maximum_active_sequences,
                        self.payload.maximum_scheduled_tokens,
                    )?
                    || descriptor.alignment_bytes != workspace.alignment_bytes
                    || descriptor.usage != BufferUsage::Binding
                    || descriptor.lifetime != AllocationLifetime::Invocation
                    || descriptor.theoretical_maximum_instances
                        != self.payload.memory.maximum_active_sequences
                    || descriptor.kind
                        != (AllocationKind::Binding {
                            node_id: node.id.clone(),
                        })
                {
                    return Err(invalid_plan(format!(
                        "node `{}` binding descriptor differs from its provider estimate",
                        node.id
                    )));
                }
            }
            if let Some(resource_id) = &node.persistent_resource {
                let workspace = node.provider_resources.persistent.as_ref().ok_or_else(|| {
                    invalid_plan(format!("node `{}` persistent estimate is missing", node.id))
                })?;
                match workspace.scope {
                    ProviderWorkspaceScope::Plan => {
                        let allocation = static_allocations.get(resource_id).ok_or_else(|| {
                            invalid_plan(format!(
                                "node `{}` plan-static persistent allocation is missing",
                                node.id
                            ))
                        })?;
                        if Some(allocation.per_instance_bytes) != workspace.fixed_bytes()
                            || allocation.alignment_bytes != workspace.alignment_bytes
                            || allocation.usage != BufferUsage::Persistent
                            || !workspace.storage.accepts(allocation.storage.profile())
                            || allocation.storage.logical_layout_fingerprint()
                                != workspace_storage_layout_fingerprint()?
                            || allocation.kind
                                != (AllocationKind::Persistent {
                                    node_id: node.id.clone(),
                                })
                        {
                            return Err(invalid_plan(format!(
                                "node `{}` plan-static persistent allocation differs from its provider estimate",
                                node.id
                            )));
                        }
                    }
                    scope @ (ProviderWorkspaceScope::Request
                    | ProviderWorkspaceScope::Sequence
                    | ProviderWorkspaceScope::Step) => {
                        let expected_lifetime = match scope {
                            ProviderWorkspaceScope::Request => AllocationLifetime::Request,
                            ProviderWorkspaceScope::Sequence => AllocationLifetime::Sequence,
                            ProviderWorkspaceScope::Step => AllocationLifetime::Step,
                            ProviderWorkspaceScope::Plan | ProviderWorkspaceScope::Invocation => {
                                unreachable!()
                            }
                        };
                        let descriptor = dynamic_descriptors.get(resource_id).ok_or_else(|| {
                            invalid_plan(format!(
                                "node `{}` dynamic persistent descriptor is missing",
                                node.id
                            ))
                        })?;
                        if descriptor.demand
                            != workspace.size_formula.bind_runtime_limits(
                                self.payload.memory.maximum_active_sequences,
                                self.payload.maximum_scheduled_tokens,
                            )?
                            || descriptor.alignment_bytes != workspace.alignment_bytes
                            || descriptor.usage != BufferUsage::Persistent
                            || descriptor.lifetime != expected_lifetime
                            || descriptor.theoretical_maximum_instances
                                != self.payload.memory.maximum_active_sequences
                            || descriptor.kind
                                != (AllocationKind::Persistent {
                                    node_id: node.id.clone(),
                                })
                        {
                            return Err(invalid_plan(format!(
                                "node `{}` dynamic persistent descriptor differs from its provider estimate",
                                node.id
                            )));
                        }
                    }
                    ProviderWorkspaceScope::Invocation => {
                        return Err(invalid_plan(format!(
                            "node `{}` persistent workspace cannot be invocation scoped",
                            node.id
                        )));
                    }
                }
            }
            for binding in &node.values {
                Self::validate_cross_node_value(binding, &mut canonical_values)?;
            }
        }
        Self::validate_global_storage_aliasing(&canonical_values, &self.payload.nodes)?;
        Ok(())
    }

    pub fn payload(&self) -> &ExecutionPlanPayload {
        &self.payload
    }

    pub fn plan_hash(&self) -> &PlanHash {
        &self.plan_hash
    }

    pub(crate) fn operation_registry_authority(&self) -> &OperationRegistryAuthority {
        &self.operation_registry_authority
    }

    pub fn to_json(&self) -> Result<Vec<u8>, VNextError> {
        serde_json::to_vec(self).map_err(|error| VNextError::Serialization {
            context: "serialize execution plan",
            message: error.to_string(),
        })
    }

    pub fn decode_untrusted(bytes: &[u8]) -> Result<UnvalidatedExecutionPlan, VNextError> {
        if bytes.len() > MAX_EXECUTION_PLAN_WIRE_BYTES {
            return Err(VNextError::Serialization {
                context: "decode untrusted execution plan",
                message: format!(
                    "execution plan wire size {} exceeds limit {}",
                    bytes.len(),
                    MAX_EXECUTION_PLAN_WIRE_BYTES
                ),
            });
        }
        serde_json::from_slice::<UnvalidatedExecutionPlanWire>(bytes)
            .map(UnvalidatedExecutionPlan::from)
            .map_err(|error| VNextError::Serialization {
                context: "decode untrusted execution plan",
                message: error.to_string(),
            })
    }

    pub fn from_json_validated<P: RuntimePolicy>(
        bytes: &[u8],
        family: &PreparedModelFamily,
        capabilities: &CapabilityCatalog,
        policy: &P,
        node_resolutions: Vec<PlanNodeResolution>,
    ) -> Result<Self, VNextError> {
        Self::decode_untrusted(bytes)?.revalidate(family, capabilities, policy, node_resolutions)
    }

    pub fn validate_against<P: RuntimePolicy>(
        &self,
        family: &PreparedModelFamily,
        capabilities: &CapabilityCatalog,
        policy: &P,
        node_resolutions: &[PlanNodeResolution],
    ) -> Result<(), VNextError> {
        let rebuilt = ExecutionPlan::build(PlanBuildRequest::new(
            family,
            capabilities,
            policy,
            node_resolutions.to_vec(),
        )?)?;
        if rebuilt.operation_registry_authority != self.operation_registry_authority {
            return Err(invalid_plan(
                "execution plan belongs to a different operation runtime registry",
            ));
        }
        if &rebuilt != self {
            return Err(invalid_plan(
                "execution plan is not identical to its semantic rebuild",
            ));
        }
        Ok(())
    }
}
