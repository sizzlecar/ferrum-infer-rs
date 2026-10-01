//! Opt-in eager CPU fixture projection through the real core/provider algebra.
//! Its 4-element tensor program is not a measured model or a qualified sample.
use super::*;
use vnext::ExecutionCostRouteUnknown as U;

/// The controlled native CPU program has real full-logits and greedy-token
/// fills. Each projected product uses the same recipe as native submission.
/// Bind that uncertainty to the original recipe, using the production checked
/// constructor. Unsupported product policies retain the default Unknown path.
pub(super) fn project_with_host_content(
    executor: &ControlledExecutor,
    view: &vnext::ExecutionCostRouteView,
    state: &vnext::ExecutionCostRouteState,
    query: &vnext::FutureWaveCostQuery<'_>,
    host: &vnext::FutureHostPendingQueryV2<'_>,
    budget: &mut dyn vnext::ResourcePlanningBudget,
) -> vnext::ExecutionCostRouteAvailability<vnext::ExecutionCostRouteForecastV2> {
    use ferrum_interfaces::execution_cost::{HostContentForecastV2, HostPendingSetV2};
    let result = (|| {
        if !executor.project_structured_cpu_fill.load(Ordering::Acquire)
            || !executor
                .native_structured_submission
                .load(Ordering::Acquire)
            || !view.structured_capture_enabled()
            || host.eligible_rows.len() > query.rows.len()
        {
            return Err(U::Unsupported);
        }
        let mut positions = Vec::with_capacity(host.eligible_rows.len());
        for row in host.eligible_rows {
            if !budget.has_budget() {
                return Err(U::BudgetExhausted);
            }
            if positions
                .last()
                .is_some_and(|prior| *prior >= row.physical_position)
                || !cpu_fill_supports_policy(row.clean_policy)
                || !matches!(
                    query
                        .rows
                        .get(row.physical_position as usize)
                        .map(|r| r.output),
                    Some(vnext::FutureCostOutput::Decode {
                        policy: ferrum_interfaces::model_executor::LogitsReturnPolicy::FullLogits
                    })
                )
            {
                return Err(U::OutputBranch);
            }
            positions.push(row.physical_position);
        }
        let projection = inner(executor, view, state, query, budget)?;
        let recipe = projection
            .statistical_evidence
            .as_ref()
            .ok_or(U::OutputBranch)?
            .structured_capture()
            .ok_or(U::OutputBranch)?
            .map_err(|_| U::OutputBranch)?;
        let pending = HostPendingSetV2::new_installed(
            &projection.shape,
            recipe,
            &positions,
            host.constraint,
            None,
        )
        .map_err(|_| U::OutputBranch)?;
        if !budget.has_budget() {
            return Err(U::BudgetExhausted);
        }
        Ok(vnext::ExecutionCostRouteForecastV2 {
            projection,
            host_content: HostContentForecastV2::Unresolved(pending),
        })
    })();
    match result {
        Ok(value) => vnext::ExecutionCostRouteAvailability::Known(value),
        Err(reason) => {
            *executor.cost_route_unknown.lock() = Some(reason);
            eprintln!("controlled CPU host forecast unavailable: {reason:?}");
            vnext::ExecutionCostRouteAvailability::Unknown(reason)
        }
    }
}

fn cpu_fill_supports_policy(
    policy: &ferrum_interfaces::model_executor::LogitsReturnPolicy,
) -> bool {
    use ferrum_interfaces::model_executor::LogitsReturnPolicy;
    match policy {
        LogitsReturnPolicy::FullLogits => true,
        LogitsReturnPolicy::GreedyArgmax {
            token_mask,
            repetition_penalty: None,
        } => {
            // ControlledCpuFill actually emits token 6 for its greedy product;
            // it has no mask upload or repetition processor. Only accept a
            // policy whose mask permits that real fixed result.
            token_mask.as_ref().is_none_or(|mask| {
                mask.valid_token_mask
                    .get(6)
                    .is_some_and(|valid| *valid != 0)
            })
        }
        _ => false,
    }
}

pub(super) fn project(
    executor: &ControlledExecutor,
    view: &vnext::ExecutionCostRouteView,
    state: &vnext::ExecutionCostRouteState,
    query: &vnext::FutureWaveCostQuery<'_>,
    budget: &mut dyn vnext::ResourcePlanningBudget,
) -> vnext::ExecutionCostRouteAvailability<vnext::ExecutionCostRouteProjection> {
    let result = inner(executor, view, state, query, budget);
    match result {
        Ok(value) => vnext::ExecutionCostRouteAvailability::Known(value),
        Err(reason) => {
            *executor.cost_route_unknown.lock() = Some(reason);
            eprintln!("controlled fixture future projection unavailable: {reason:?}");
            vnext::ExecutionCostRouteAvailability::Unknown(reason)
        }
    }
}

fn inner(
    executor: &ControlledExecutor,
    view: &vnext::ExecutionCostRouteView,
    state: &vnext::ExecutionCostRouteState,
    query: &vnext::FutureWaveCostQuery<'_>,
    budget: &mut dyn vnext::ResourcePlanningBudget,
) -> std::result::Result<vnext::ExecutionCostRouteProjection, U> {
    if let Some(reason) = executor
        .projection_readiness_fault
        .lock()
        .as_ref()
        .and_then(|fault| fault(query.rows.len(), executor.physical.load(Ordering::Acquire)))
    {
        return Err(reason);
    }
    let f = executor.evidence.fixture.as_ref().unwrap();
    let cpu_fill = executor.project_structured_cpu_fill.load(Ordering::Acquire);
    if !f.provider_trace.lock().unwrap().cost_route_statistics
        || (!cpu_fill && query.kind != ActualWaveKind::Prefill)
    {
        return Err(U::Unsupported);
    }
    if query.rows.is_empty() || !budget.has_budget() {
        return Err(U::InvalidInput);
    }
    if executor.single_row_prefill_only.load(Ordering::Acquire)
        && query.kind == ActualWaveKind::Prefill
        && query.rows.len() > 1
    {
        return Err(U::Unsupported);
    }
    let plan = f.resolved.execution_plan();
    let first = plan.payload().nodes().first().ok_or(U::InvalidInput)?;
    let last = plan.payload().nodes().last().ok_or(U::InvalidInput)?;
    let input = first
        .values()
        .iter()
        .find(|v| v.role() == vnext::ResolvedValueRole::Input && v.ordinal() == 0)
        .ok_or(U::CoreLayout)?;
    let output = last
        .values()
        .iter()
        .find(|v| v.role() == vnext::ResolvedValueRole::Output && v.ordinal() == 0)
        .ok_or(U::CoreLayout)?;
    let [input] = input.storage().components() else {
        return Err(U::CoreLayout);
    };
    let [output] = output.storage().components() else {
        return Err(U::CoreLayout);
    };
    let layout = |c: &vnext::ResolvedStorageComponent| {
        vnext::HostTransferLayout::new(
            c.element_type(),
            c.length_bytes() / c.element_type().size_bytes(),
        )
        .map_err(|_| U::CoreLayout)
    };
    let input_layout = layout(input)?;
    let output_layout = layout(output)?;
    let mut work = Vec::new();
    let mut indices = Vec::new();
    let mut uploads = Vec::new();
    let mut readbacks = Vec::new();
    for (position, row) in query.rows.iter().enumerate() {
        if !budget.has_budget() {
            return Err(U::BudgetExhausted);
        }
        let (offset, count, total, final_logits) = match (row.work, row.output) {
            (
                ActualRowWork::Prefill {
                    offset,
                    count,
                    total_prompt_tokens,
                },
                vnext::FutureCostOutput::Prefill { final_logits },
            ) => {
                if offset.checked_add(count).is_none_or(|end| {
                    end > total_prompt_tokens || final_logits != (end == total_prompt_tokens)
                }) {
                    return Err(U::InvalidInput);
                }
                (offset, count, total_prompt_tokens, final_logits)
            }
            (ActualRowWork::Decode { kv_tokens }, vnext::FutureCostOutput::Decode { policy })
                if cpu_fill && cpu_fill_supports_policy(policy) =>
            {
                (
                    kv_tokens,
                    1,
                    kv_tokens.checked_add(1).ok_or(U::InvalidInput)?,
                    true,
                )
            }
            _ => return Err(U::Unsupported),
        };
        work.push(vnext::OperationCostWorkRow {
            offset: u64::from(offset),
            count: NonZeroU64::new(u64::from(count)).ok_or(U::InvalidInput)?,
            full_input_tokens: NonZeroU64::new(u64::from(total)).ok_or(U::InvalidInput)?,
        });
        indices.push(row.participant_index);
        uploads.push(vnext::EagerCoreInputUpload {
            node_id: first.id(),
            input_ordinal: 0,
            participant_index: position,
            logical_offset_bytes: 0,
            layout: input_layout,
        });
        if final_logits {
            readbacks.push(vnext::EagerCoreReadback {
                node_id: last.id(),
                resource_id: output.resource_id(),
                participant_index: position,
                logical_offset_bytes: 0,
                layout: output_layout,
            });
        }
    }
    let providers = f
        .registry
        .bind_plan(&f.resolved)
        .map_err(|_| U::ProviderRoute)?;
    let mut canonical =
        CanonicalWaveCostBuilder::new_with_structured_statistics(0, CostProductOutput::FullLogits);
    let next = vnext::append_complete_eager_cost_route(
        f.runtime.as_ref(),
        &providers,
        &f.resolved,
        &f.plan_resources,
        view,
        state,
        &vnext::EagerCoreWaveCostQuery {
            rows: &work,
            participant_indices: &indices,
            uploads: &uploads,
            readbacks: &readbacks,
            attempt_staged_readbacks: false,
            reusable_bucket: None,
            token_mask_input: None,
        },
        &mut canonical,
        budget,
    ).map_err(|reason| {
        if cpu_fill {
            eprintln!("controlled CPU original resource projection failed: {reason:?}; rows={work:#?}; resources={:#?}", view.resource_view());
        }
        reason
    })?;
    if cpu_fill {
        // The core result above is the original conservative resource proof.
        // The controlled executor executes its installed CPU fill, not that
        // resource fixture's tensor provider, so use its one shared real cost
        // recipe rather than claiming the core dry-run costs were executed.
        let provider_identities = super::structured::provider_identities(providers.providers());
        let layout = executor.cpu_fill_layout(query.rows.iter().filter_map(|row| match row.work {
            ActualRowWork::Decode { kv_tokens } => Some(kv_tokens as usize),
            _ => None,
        }));
        canonical = super::structured::command_builder_with_layout(
            query.rows.len(),
            work.iter().map(|row| row.count.get()).sum(),
            query.rows.iter().any(|row| match row.output {
                vnext::FutureCostOutput::Prefill { .. } => true,
                vnext::FutureCostOutput::Decode { policy } => policy.requires_full_logits(),
                _ => false,
            }),
            if executor
                .native_structured_submission
                .load(Ordering::Acquire)
            {
                Some(provider_identities.as_slice())
            } else {
                None
            },
            layout,
        );
    }
    for row in query.rows {
        let output = match row.output {
            vnext::FutureCostOutput::Prefill { final_logits } => {
                CostRowOutput::Prefill { final_logits }
            }
            vnext::FutureCostOutput::Decode { policy }
                if cpu_fill && cpu_fill_supports_policy(policy) =>
            {
                CostRowOutput::Decode {
                    requires_full_logits: policy.requires_full_logits(),
                    repetition_tokens: 0,
                    repetition_penalty_bits: 1_f32.to_bits(),
                }
            }
            _ => return Err(U::Unsupported),
        };
        canonical
            .row(CanonicalCostRow {
                work: row.work,
                host_policy_signature: row.host_policy_signature,
                host_features: row.host_features,
                mask_upload_required: false,
                output,
            })
            .map_err(|_| U::InvalidInput)?;
    }
    let shape = canonical
        .finish_with_captured_structure(
            query.kind,
            ActualWavePath::PlanRuntime,
            next.projected_graph_state(),
            ActualWaveRowOrder::Ordered,
            super::structured::recurrent_state_bytes(executor, query.rows.len()),
        )
        .map_err(|_| U::InvalidInput)?;
    if let Err(reason) = &shape.statistical {
        eprintln!("controlled fixture statistical projection unavailable: {reason:?}");
    }
    Ok(vnext::ExecutionCostRouteProjection {
        shape: shape.exact,
        statistical_evidence: shape.statistical.ok(),
        state: next,
    })
}
