use super::*;
use crate::execution_cost::{
    CanonicalWaveCostBuilder, CoreReadbackRoute, CostCommandPath, CostPhysicalCommand,
    CostProviderIdentity, MAX_COST_COMMANDS, MAX_COST_ROWS,
};
use crate::vnext::*;
use ExecutionCostRouteUnknown as U;

#[cfg(test)]
mod tests;

pub struct EagerCoreWaveCostQuery<'a> {
    pub rows: &'a [OperationCostWorkRow],
    pub participant_indices: &'a [usize],
    pub uploads: &'a [EagerCoreInputUpload<'a>],
    pub readbacks: &'a [EagerCoreReadback<'a>],
    pub attempt_staged_readbacks: bool,
    /// Exact result of the executor's first-fit workspace bucket selection.
    /// None means that this wave uses the ordinary transient workspace path.
    pub reusable_bucket: Option<&'a ReusableExecutionBucketId>,
    /// Product input whose upload may be omitted only with exact retained-slot
    /// residency evidence in this view and its per-witness state.
    pub token_mask_input: Option<EagerCoreTokenMaskInput<'a>>,
}

pub(super) fn poll(budget: &mut dyn ResourcePlanningBudget) -> Result<(), U> {
    if budget.has_budget() {
        Ok(())
    } else {
        Err(U::BudgetExhausted)
    }
}

/// Appends the entire core/provider route in the same order as actual eager
/// dispatch. The actual encoder remains the sole physical authority. Success
/// requires a complete pure provider declaration and resident resource fit.
pub fn append_complete_eager_cost_route<R: DeviceRuntime>(
    runtime: &R,
    providers: &BoundOperationProviderSet<R>,
    resolved: &dyn ExecutablePlanView,
    resources: &Arc<PlanRuntimeResources<R>>,
    view: &ExecutionCostRouteView,
    state: &ExecutionCostRouteState,
    query: &EagerCoreWaveCostQuery<'_>,
    canonical: &mut CanonicalWaveCostBuilder,
    budget: &mut dyn ResourcePlanningBudget,
) -> Result<ExecutionCostRouteState, U> {
    poll(budget)?;
    if !Arc::ptr_eq(&view.fence, &state.fence) {
        return Err(U::StaleView);
    }
    if resources.planning_plan_hash() != resolved.execution_plan().plan_hash()
        || !resources.planning_runtime_matches(runtime)
    {
        return Err(U::StaleView);
    }
    if query.rows.is_empty()
        || query.rows.len() > MAX_COST_ROWS
        || query.rows.len() != query.participant_indices.len()
        || query.uploads.len() > MAX_COST_COMMANDS
        || query.readbacks.len() > MAX_COST_ROWS
    {
        return Err(U::Capacity);
    }
    let capability = runtime
        .cost_core_execution_capabilities()
        .ok_or(U::Unsupported)?;
    let eager_graph_state = graph_policy_is_eager(
        runtime.cost_graph_capture_capability(),
        view.graph_stream_state(),
    );
    if !capability.single_transfer_commands
        || resolved
            .execution_plan()
            .payload()
            .memory()
            .reusable_execution()
            .is_some_and(|plan| plan.program_policy().is_some())
        || !eager_graph_state
    {
        return Err(U::ExecutionPolicy);
    }
    let mut projection_rows = Vec::new();
    projection_rows
        .try_reserve_exact(query.rows.len())
        .map_err(|_| U::Capacity)?;
    let mut total_tokens = 0_u64;
    let mut zeros = 0_u32;
    let mut next = state.clone();
    let mut previous_authority = None;
    for (row, &index) in query.rows.iter().zip(query.participant_indices) {
        poll(budget)?;
        let authority = view.participant_authority(index).ok_or(U::InvalidInput)?;
        if previous_authority.is_some_and(|previous| previous >= authority) {
            return Err(U::InvalidInput);
        }
        previous_authority = Some(authority);
        let frontier = next.frontiers.get_mut(index).ok_or(U::InvalidInput)?;
        if *frontier != row.offset {
            return Err(U::StaleView);
        }
        let end = row
            .offset
            .checked_add(row.count.get())
            .ok_or(U::InvalidInput)?;
        if end > row.full_input_tokens.get() {
            return Err(U::InvalidInput);
        }
        *frontier = end;
        total_tokens = total_tokens
            .checked_add(row.count.get())
            .ok_or(U::InvalidInput)?;
        if !next.initialized[index] {
            zeros = zeros
                .checked_add(
                    view.resources.participants()[index]
                        .pending_zero_commands()
                        .ok_or(U::InitializationState)?,
                )
                .ok_or(U::Capacity)?;
            next.initialized[index] = true;
        }
        // New zero-initialized growth requires its own physical-piece proof.
        // Fixed recurrent state is supported; token-scaled zero state stays
        // Unknown until its newly allocated extents are projected explicitly.
        let covered = state
            .resources
            .covered_tokens(index)
            .ok_or(U::InvalidInput)?;
        if end > covered {
            for descriptor in resolved
                .execution_plan()
                .payload()
                .memory()
                .dynamic_descriptors()
            {
                poll(budget)?;
                if descriptor.initialization() != StateInitialization::Zero {
                    continue;
                }
                let before = descriptor
                    .evaluate_request_bytes_for_shape(DynamicResourceShape::from_validated(
                        1, covered, 0,
                    ))
                    .map_err(|_| U::InitializationState)?;
                let after = descriptor
                    .evaluate_request_bytes_for_shape(DynamicResourceShape::from_validated(
                        1, end, 0,
                    ))
                    .map_err(|_| U::InitializationState)?;
                if before != after {
                    return Err(U::InitializationState);
                }
            }
        }
        projection_rows.push(ResourcePlanningRow {
            participant_index: index,
            start_token: row.offset,
            token_count: row.count.get(),
        });
    }
    let projection = match resources.project_resource_wave_with_bucket(
        &view.resources,
        &state.resources,
        &projection_rows,
        query.reusable_bucket,
        budget,
    ) {
        ResourcePlanningAvailability::Known(projection) => projection,
        ResourcePlanningAvailability::Unknown(reason) => return Err(U::Resource(reason)),
    };
    let selected_step = projection.step_slot;
    next.resources = projection.state;
    let mut command_index = 0_u32;
    for _ in 0..zeros {
        poll(budget)?;
        append_transfer(
            canonical,
            &mut command_index,
            capability.zero_native_operation,
            DeviceCommandPhase::Initialization,
            None,
            0,
            0,
        )?;
    }
    let shape = DynamicResourceShape::from_validated(query.rows.len() as u32, total_tokens, 0);
    for (index, node) in resolved
        .execution_plan()
        .payload()
        .nodes()
        .iter()
        .enumerate()
    {
        poll(budget)?;
        let Some(requirement) = node.provider_resources().scratch() else {
            continue;
        };
        if requirement.reuse_policy() == ProviderWorkspaceReusePolicy::Preserve {
            return Err(U::CoreLayout);
        }
        if requirement.reuse_policy() != ProviderWorkspaceReusePolicy::ZeroBeforeUse {
            continue;
        }
        let resource_id = node.scratch_resource().ok_or(U::CoreLayout)?;
        let descriptor = resolved
            .execution_plan()
            .payload()
            .memory()
            .dynamic_descriptors()
            .iter()
            .find(|descriptor| descriptor.base_resource_id() == resource_id)
            .ok_or(U::CoreLayout)?;
        if descriptor.storage().profile().view() != DynamicStorageView::Contiguous
            || requirement
                .evaluate_shape_bytes(shape)
                .map_err(|_| U::CoreLayout)?
                == 0
        {
            return Err(U::CoreLayout);
        }
        let provider = providers
            .providers()
            .get(index)
            .ok_or(U::ProviderRoute)?
            .descriptor();
        let identity = CostProviderIdentity {
            provider_id: provider.provider_id().as_str(),
            implementation_fingerprint: provider.provider_implementation_fingerprint(),
            operation_fingerprint: provider.operation_fingerprint(),
        };
        append_transfer(
            canonical,
            &mut command_index,
            capability.zero_native_operation,
            DeviceCommandPhase::Initialization,
            Some((index as u32, identity)),
            query.rows.len() as u32,
            total_tokens,
        )?;
    }
    let mut input_commands =
        super::uploads::input_commands(resolved, query.rows, query.uploads, budget)?;
    next.last_token_mask_uploads = None;
    if let Some(mask) = query.token_mask_input {
        if mask.contents.len() != query.rows.len() {
            return Err(U::InvalidInput);
        }
        let mut seen = vec![false; query.rows.len()];
        for upload in query.uploads {
            poll(budget)?;
            if upload.node_id == mask.node_id && upload.input_ordinal == mask.input_ordinal {
                let seen = seen
                    .get_mut(upload.participant_index)
                    .ok_or(U::InvalidInput)?;
                if *seen
                    || upload.logical_offset_bytes != 0
                    || upload.layout.element_type() != ElementType::U8
                    || upload.layout.byte_len().map_err(|_| U::InvalidInput)?
                        != mask.vocabulary_size
                {
                    return Err(U::InvalidInput);
                }
                *seen = true;
            }
        }
        if seen.iter().any(|seen| !seen) {
            return Err(U::InvalidInput);
        }
        let uploads = next
            .token_masks
            .as_mut()
            .ok_or(U::OutputBranch)?
            .project_contents(
                selected_step.as_ref(),
                mask.vocabulary_size,
                mask.contents,
                budget,
            )?;
        if uploads.iter().any(|upload| !upload) {
            let mut filtered = Vec::new();
            filtered
                .try_reserve_exact(query.uploads.len())
                .map_err(|_| U::Capacity)?;
            for upload in query.uploads {
                poll(budget)?;
                if upload.node_id != mask.node_id
                    || upload.input_ordinal != mask.input_ordinal
                    || uploads[upload.participant_index]
                {
                    filtered.push(*upload);
                }
            }
            input_commands =
                super::uploads::input_commands(resolved, query.rows, &filtered, budget)?;
        }
        next.last_token_mask_uploads = Some(uploads);
    }
    for _ in 0..input_commands {
        poll(budget)?;
        append_transfer(
            canonical,
            &mut command_index,
            capability.upload_native_operation,
            DeviceCommandPhase::DynamicBinding,
            None,
            0,
            0,
        )?;
    }
    let route = providers
        .eager_cost_route_with_ranges(
            resolved,
            query.rows,
            projection.physical_ranges.as_ref(),
            &mut || {
                if budget.has_budget() {
                    Ok(())
                } else {
                    Err(VNextError::InvalidExecutionPlan {
                        reason: "future cost route budget exhausted".to_owned(),
                    })
                }
            },
        )
        .map_err(|_| {
            if budget.has_budget() {
                U::ProviderRoute
            } else {
                U::BudgetExhausted
            }
        })?
        .ok_or(U::ProviderRoute)?;
    poll(budget)?;
    let binding_layout = query
        .reusable_bucket
        .and_then(|bucket| resources.planning_program_binding_layout(bucket));
    let binding_nodes = program_binding_nodes(
        resolved.execution_plan().payload().nodes(),
        query.reusable_bucket,
        binding_layout,
        budget,
    )?;
    // A backend that coalesces bindings must project the actual provider write
    // spans through its real arena layout and shared transfer-layout algorithm.
    let appended = if binding_nodes.is_empty() || capability.preserves_program_bindings {
        route.append_canonical_with_program_bindings(canonical, command_index, &binding_nodes)
    } else {
        let patches = route
            .program_binding_patches(&binding_nodes)
            .map_err(|_| U::ProviderRoute)?;
        let merged = runtime
            .cost_coalesced_program_binding(
                binding_layout.ok_or(U::CoreLayout)?,
                &patches,
                &mut || {
                    if budget.has_budget() {
                        Ok(())
                    } else {
                        Err(VNextError::InvalidExecutionPlan {
                            reason: "future binding projection budget exhausted".to_owned(),
                        })
                    }
                },
            )
            .ok_or(U::CoreLayout)?
            .map_err(|_| {
                if budget.has_budget() {
                    U::CoreLayout
                } else {
                    U::BudgetExhausted
                }
            })?;
        route.append_canonical_with_coalesced_program_binding(
            canonical,
            command_index,
            &binding_nodes,
            &merged,
        )
    };
    command_index = appended.map_err(|error| match error {
        crate::execution_cost::CanonicalCostError::Capacity => U::Capacity,
        _ => U::ProviderRoute,
    })?;
    if command_index as usize > MAX_COST_COMMANDS {
        return Err(U::Capacity);
    }
    let readback_bytes =
        super::uploads::readback_bytes(resolved, query.rows, query.readbacks, budget)?;
    let readback = append_readback_route(
        capability,
        query.attempt_staged_readbacks,
        query.readbacks.len(),
        readback_bytes,
        view.readback_available_bytes,
        canonical,
        &mut command_index,
        budget,
    )?;
    canonical
        .core_readback_route(readback)
        .map_err(|_| U::InvalidInput)?;
    poll(budget)?;
    Ok(next)
}

#[allow(clippy::too_many_arguments)]
fn append_readback_route(
    capability: DeviceCoreCostCapabilities,
    attempt_staged: bool,
    count: usize,
    bytes: u64,
    available_bytes: u64,
    canonical: &mut CanonicalWaveCostBuilder,
    command_index: &mut u32,
    budget: &mut dyn ResourcePlanningBudget,
) -> Result<CoreReadbackRoute, U> {
    let readback = if count == 0 {
        CoreReadbackRoute::NoReadback
    } else if attempt_staged {
        if capability.staged_host_readback_without_commands
            == capability.staged_host_readback_native_operation.is_some()
        {
            // Exactly one of the supported staging mechanisms must be known.
            return Err(U::ReadbackState);
        }
        if bytes <= available_bytes {
            CoreReadbackRoute::SubmissionStaged
        } else if capability.fallback_readback == CoreReadbackRoute::HostSynchronized {
            CoreReadbackRoute::SubmissionFallbackSynchronized
        } else {
            return Err(U::ReadbackState);
        }
    } else {
        capability.fallback_readback
    };
    if readback == CoreReadbackRoute::Unknown {
        return Err(U::ReadbackState);
    }
    if readback == CoreReadbackRoute::SubmissionStaged {
        if let Some(operation) = capability.staged_host_readback_native_operation {
            // readback_bytes proved contiguous Step storage and a valid single
            // participant range. Actual staging emits one physical piece each.
            for _ in 0..count {
                poll(budget)?;
                append_transfer(
                    canonical,
                    command_index,
                    operation,
                    DeviceCommandPhase::ResultBinding,
                    None,
                    0,
                    0,
                )?;
            }
        }
    }
    Ok(readback)
}

fn graph_policy_is_eager(
    capability: DeviceCostGraphCaptureCapability,
    stream: Option<DeviceCostGraphStreamState>,
) -> bool {
    match capability {
        DeviceCostGraphCaptureCapability::Unsupported => true,
        DeviceCostGraphCaptureCapability::Supported => {
            stream.is_some_and(|state| state.is_unconfigured_empty())
        }
        DeviceCostGraphCaptureCapability::Unknown => false,
    }
}

fn program_binding_nodes(
    nodes: &[PlanNode],
    bucket: Option<&ReusableExecutionBucketId>,
    layout: Option<&ProgramBindingLayout>,
    budget: &mut dyn ResourcePlanningBudget,
) -> Result<Vec<usize>, U> {
    let Some(bucket) = bucket else {
        // Actual invocation creation has no program-binding handle here;
        // validate_program_binding_patch requires zero program patches.
        return if layout.is_none() {
            Ok(Vec::new())
        } else {
            Err(U::CoreLayout)
        };
    };
    if nodes.len() > MAX_COST_COMMANDS {
        return Err(U::Capacity);
    }
    let count = nodes
        .iter()
        .filter(|node| node.binding_resource().is_some())
        .count();
    if count == 0 {
        return if layout.is_none() {
            Ok(Vec::new())
        } else {
            Err(U::CoreLayout)
        };
    }
    let layout = layout.ok_or(U::CoreLayout)?;
    if layout.reusable_execution_bucket_id() != bucket || layout.slots().len() != count {
        return Err(U::CoreLayout);
    }
    let mut result = Vec::new();
    result.try_reserve_exact(count).map_err(|_| U::Capacity)?;
    for (index, node) in nodes.iter().enumerate() {
        poll(budget)?;
        let Some(resource) = node.binding_resource() else {
            continue;
        };
        let slot = layout.slot_for_node(index).ok_or(U::CoreLayout)?;
        if slot.node_id() != node.id() || slot.resource_id() != resource {
            return Err(U::CoreLayout);
        }
        result.push(index);
    }
    Ok(result)
}

fn append_transfer<'a>(
    canonical: &mut CanonicalWaveCostBuilder,
    index: &mut u32,
    operation: &'static str,
    phase: DeviceCommandPhase,
    provider: Option<(u32, CostProviderIdentity<'a>)>,
    participants: u32,
    tokens: u64,
) -> Result<(), U> {
    if *index as usize >= MAX_COST_COMMANDS {
        return Err(U::Capacity);
    }
    canonical
        .physical_command(CostPhysicalCommand {
            native_op_id: operation,
            command_index: *index,
            node_index: provider.map(|value| value.0),
            command_phase: phase,
            provider: provider.map(|value| value.1),
            path: CostCommandPath::Eager,
            participant_start: 0,
            participant_count: participants,
            token_count: tokens,
            batching_form: if provider.is_some() {
                DeviceBatchingForm::Packed.as_str()
            } else {
                DeviceBatchingForm::Scalar.as_str()
            },
            compute_dispatch_count: 0,
            transfer_command_count: 1,
            reusable_graph_node_count: None,
        })
        .map_err(|_| U::InvalidInput)?;
    *index = index.checked_add(1).ok_or(U::Capacity)?;
    Ok(())
}
