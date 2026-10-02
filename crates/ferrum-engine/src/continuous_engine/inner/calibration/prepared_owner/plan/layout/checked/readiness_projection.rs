//! An uninterrupted initial projection can complete the inventory. Only
//! positions survive a preparation action; its final capture uses fresh owners.
use super::*;

pub(super) async fn prepare(
    session: &mut CalibrationSession,
    input: &PreparedProbeInputs,
    group: &[Case],
    cases: &[Case],
    budget: &mut ProbeExecutionBudget,
    maximum_retained_bytes: usize,
    attempts: &mut readiness::Attempts,
) -> Result<Option<CheckedCaseInventory>> {
    let mut first_target = 0;
    let mut retain_complete = true;
    loop {
        let projection_remaining = input
            .settings
            .maximum_offered_waves
            .get()
            .checked_sub(budget.preflight_charge().projection_attempts)
            .filter(|n| *n > 0)
            .ok_or_else(|| error("readiness pure projection budget exhausted"))?;
        let reserved = readiness::initial_inventory_admissions(group)?;
        budget.claim_input_projection_requests(reserved)?;
        let can_act = |index: usize, reason: &GeometryProjectionUnknown| {
            group
                .get(index)
                .is_some_and(|case| readiness::can_act(reason, case, cases, input, attempts))
        };
        let mut charge = ProbePreflightCharge::default();
        let progress = inventory::probe_readiness(
            session,
            group,
            &input.templates,
            &input.prompts,
            &input.pair,
            input.population.population_policy(),
            &input.required_geometry.sequence_tokens,
            InventoryLimits {
                route_population: input.population.route_population,
                deadline: budget.deadline(),
                maximum_admitted_requests: reserved,
                maximum_projection_attempts: projection_remaining,
                maximum_retained_bytes,
                maximum_route_states: 256,
                prefill_chunk: input.chunk,
                prefill_row_ceiling: input.prefill_row_ceiling,
            },
            inventory::ReadinessCursor {
                first_target,
                retain_complete,
            },
            &mut charge,
            &can_act,
        )
        .await;
        budget.record_geometry(charge)?;
        tracing::info!(
            template = group.first().map(|case| case.template),
            first_target,
            succeeded = progress.is_ok(),
            ?charge,
            "Automatic short readiness projection completed"
        );
        let progress = match progress? {
            inventory::ReadinessCapture::Complete(inventory) => return Ok(Some(inventory)),
            inventory::ReadinessCapture::Progress(progress) => progress,
        };
        // Once the initial traversal stopped, no later view may promote its
        // successful prefix into complete input authority, even at target 0.
        retain_complete = false;
        if let Some(gap) = &progress.gap {
            if let Some(action) =
                readiness::next_action(std::slice::from_ref(gap), group, cases, input, attempts)?
            {
                execute(session, input, action, budget).await?;
                // Recheck the original failed target on the next fresh capture.
                // Earlier targets are only deferred to final validation, never
                // considered Known under this new resource generation.
                first_target = progress
                    .next_target
                    .checked_sub(1)
                    .ok_or_else(|| error("readiness missing failed target"))?;
                continue;
            }
        }
        first_target = progress.next_target;
        if first_target == progress.total_targets {
            return Ok(None);
        }
    }
}

async fn execute(
    session: &mut CalibrationSession,
    input: &PreparedProbeInputs,
    action: readiness::Action,
    budget: &mut ProbeExecutionBudget,
) -> Result<()> {
    if let readiness::Action::Resources(readiness) = &action {
        return resource_readiness::prepare(session, input, readiness, budget);
    }
    let readiness::Action::Execute(readiness) = action else {
        unreachable!()
    };
    let (requests, settings) = inventory::readiness_requests_with_row_ceiling(
        &readiness,
        &input.templates,
        input.chunk,
        input.prefill_row_ceiling,
    )?;
    budget.reserve_readiness_waves(
        readiness
            .waves_with_row_ceiling(
                input.prompts[readiness.template],
                input.chunk.get() as usize,
                input.prefill_row_ceiling,
            )?
            .1,
    )?;
    let summary = session
        .run_readiness_probe_cohort(requests, settings, budget)
        .await?;
    tracing::info!(template = readiness.template, rows = readiness.width,
        prompt_tokens = input.prompts[readiness.template], route = ?readiness.route,
        output_limit = readiness.maximum_output.get(), attempted_waves = summary.wave_attempts,
        reconciled_waves = summary.reconciled_waves,
        "Automatic route readiness completed outside numerical population");
    Ok(())
}
