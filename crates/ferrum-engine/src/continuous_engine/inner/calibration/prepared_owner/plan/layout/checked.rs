//! Live route inventory and input-only selection after reference retirement.
use super::super::super::super::cohort_driver::ProbeExecutionBudget;
use super::*;
use crate::continuous_engine::inner::calibration::geometry_projection::GeometryProjectionUnknown;
use ferrum_interfaces::vnext::{ExecutionCostRouteUnknown, ResourcePlanningUnknown};
use inventory::{CheckedCaseInventory, InventoryGapReason, InventoryLimits};
use populations::{CaseOpportunity, CasePopulation};
use tokio::time::Instant;

mod cursor;
mod readiness;
mod readiness_projection;
mod resource_readiness;
mod universe;
pub(in crate::continuous_engine::inner::calibration) use cursor::CheckedInputCursor;
#[cfg(test)]
mod tests;

fn prepare_cases(
    input: &PreparedProbeInputs,
) -> Result<(Vec<Case>, usize, Vec<PreparedPrefixUnavailable>, usize)> {
    let widths: Vec<_> = (1..=input.required_geometry.probe_maximum_rows).collect();
    let vocabulary = usize::try_from(
        input
            .population
            .nonnegative_envelope
            .as_ref()
            .ok_or_else(|| error("probe physical domain absent"))?
            .workload_domain
            .limits()
            .output_vocabulary_elements
            .get(),
    )
    .map_err(|_| error("probe vocabulary does not fit host"))?;
    let (mut cases, skipped, unavailable) = cases(
        &input.templates[..input.base_template_count],
        &input.settings,
        &input.outputs[..input.base_template_count],
        &widths,
        &input.pair,
        input.reset,
        vocabulary,
        &input.original_template_indices[..input.base_template_count],
        input
            .population
            .maximum_retained_numeric_bytes
            .checked_sub(
                input
                    .retained_payload_bytes()
                    .ok_or_else(|| error("checked input capacity overflow"))?,
            )
            .ok_or_else(|| error("checked input capacity exhausted"))?,
    )?;
    append_continuation_prefill_cases_with_row_ceiling(
        &mut cases,
        &input.prompts,
        input.chunk.get() as usize,
        input.prefill_row_ceiling,
        input
            .population
            .maximum_retained_numeric_bytes
            .checked_sub(
                input
                    .retained_payload_bytes()
                    .ok_or_else(|| error("continuation prefill input capacity overflow"))?,
            )
            .and_then(|remaining| {
                remaining.checked_sub(
                    unavailable
                        .capacity()
                        .checked_mul(std::mem::size_of::<PreparedPrefixUnavailable>())?,
                )
            })
            .and_then(|remaining| {
                remaining.checked_sub(
                    widths
                        .capacity()
                        .checked_mul(std::mem::size_of::<usize>())?,
                )
            })
            .ok_or_else(|| error("continuation prefill input capacity exhausted"))?,
    )?;
    let candidate_case_budget = input
        .population
        .maximum_retained_numeric_bytes
        .checked_sub(
            input
                .retained_payload_bytes()
                .ok_or_else(|| error("prefill candidate inputs overflow"))?,
        )
        .and_then(|n| {
            n.checked_sub(
                unavailable
                    .capacity()
                    .checked_mul(std::mem::size_of::<PreparedPrefixUnavailable>())?,
            )
        })
        .and_then(|n| {
            n.checked_sub(
                widths
                    .capacity()
                    .checked_mul(std::mem::size_of::<usize>())?,
            )
        })
        .ok_or_else(|| error("prefill candidate retained capacity exhausted"))?;
    append_scheduler_prefill_cases(&mut cases, input, candidate_case_budget)?;
    work::bind_cases(input, &mut cases)?;
    let retained_inputs = input
        .retained_payload_bytes()
        .and_then(|n| n.checked_add(cases.capacity().checked_mul(std::mem::size_of::<Case>())?))
        .and_then(|n| {
            n.checked_add(
                unavailable
                    .capacity()
                    .checked_mul(std::mem::size_of::<PreparedPrefixUnavailable>())?,
            )
        })
        .and_then(|n| {
            n.checked_add(
                widths
                    .capacity()
                    .checked_mul(std::mem::size_of::<usize>())?,
            )
        })
        .ok_or_else(|| error("checked input capacity overflow"))?;
    let inventory_limit = input
        .population
        .maximum_retained_numeric_bytes
        .checked_sub(retained_inputs)
        .filter(|n| *n > 0)
        .ok_or_else(|| error("checked input capacity exhausted"))?;
    Ok((cases, skipped, unavailable, inventory_limit))
}

pub(in crate::continuous_engine::inner::calibration) async fn build(
    session: &mut CalibrationSession,
    input: PreparedProbeInputs,
    budget: &mut ProbeExecutionBudget,
) -> Result<PreparedProbePlan> {
    let (cases, skipped, unavailable, inventory_limit) = prepare_cases(&input)?;
    let inventory = Box::pin(collect_ready(
        session,
        &input,
        &cases,
        budget,
        inventory_limit,
    ))
    .await?;
    freeze_inventory(
        session,
        input,
        cases,
        skipped,
        unavailable,
        inventory,
        budget,
        inventory_limit,
        false,
        None,
        None,
        None,
    )?
    .ok_or_else(|| error("checked automatic input selection has no executable population"))
}

#[allow(clippy::too_many_arguments)]
fn freeze_inventory(
    session: &CalibrationSession,
    mut input: PreparedProbeInputs,
    cases: Vec<Case>,
    skipped: usize,
    unavailable: Vec<PreparedPrefixUnavailable>,
    mut inventory: CheckedCaseInventory,
    budget: &mut ProbeExecutionBudget,
    inventory_limit: usize,
    allow_empty: bool,
    changed: Option<&[populations::CheckedPopulationKey]>,
    selected_priority: Option<u8>,
    mut retained_seed: Option<&mut Option<ferrum_scheduler::implementations::continuous::cost_model::structured_v2::DeclaredAlgorithmUniverseV1>>,
) -> Result<Option<PreparedProbePlan>> {
    let seed_bytes = universe::freeze_seed(
        &inventory,
        &input.population,
        inventory_limit,
        retained_seed.as_deref_mut(),
    )?;
    // The global discovery seed outlives this selection, but it is not the
    // numerical source's algorithm universe. Preserve raw checked families and
    // their complete endpoint/phase obligations; uncollected families stay gaps.
    input.external_retained_bytes = input
        .external_retained_bytes
        .checked_add(seed_bytes)
        .ok_or_else(|| error("cold algorithm seed external capacity overflow"))?;
    let inventory_limit = inventory_limit
        .checked_sub(seed_bytes)
        .ok_or_else(|| error("frozen algorithm seed exceeds retained capacity"))?;

    let requests_remaining = budget.selection_requests_remaining();
    let waves_remaining = budget.selection_attempts_remaining();
    let offer_rows_remaining = input
        .limits
        .max_samples
        .get()
        .min(input.limits.max_total_shape_rows.get());
    let seed = retained_seed.as_deref().and_then(|slot| slot.as_ref());
    let with_composition = if let Some(seed) = seed {
        selection::composition_authorized(
            &inventory.opportunities,
            &inventory.inputs,
            requests_remaining,
            &input.population,
            true,
            inventory_limit
                .checked_sub(
                    inventory
                        .retained_payload_bytes()
                        .ok_or_else(|| error("combination inventory capacity overflow"))?,
                )
                .unwrap_or(0),
            seed,
        )?
    } else {
        false
    };
    // Optional source scope needs original checked recipes only during input
    // selection. Capacity denial keeps the old raw path executable.
    if !with_composition {
        inventory.release_original_inputs();
    }
    let geometry_work = budget.input_geometry_work(input.input_geometry_visit_limit)?;
    let mut selection = selection::select_with_capacity_and_trajectories(
        &cases,
        &inventory.opportunities,
        &inventory.inputs,
        &input.prompts,
        input.chunk.get() as usize,
        input.prefill_row_ceiling,
        &input.population,
        selection::SelectionCapacity {
            requests: requests_remaining,
            execution_actions: waves_remaining,
            declared_offer_rows: offer_rows_remaining,
        },
        inventory_limit
            .checked_sub(
                inventory
                    .retained_payload_bytes()
                    .ok_or_else(|| error("checked inventory capacity overflow"))?,
            )
            .ok_or_else(|| error("checked selection capacity exhausted"))?,
        changed,
        selected_priority,
        geometry_work,
        Some(input.maximum_retained_sources),
        seed.filter(|_| with_composition),
        with_composition.then_some(&inventory),
    )?;
    inventory.release_original_inputs();
    if seed.is_some_and(|seed| seed.algorithm_count() >= 2) && !with_composition {
        selection.gaps.push(selection::SelectionGap {
            population: None,
            reason: selection::SelectionGapReason::CombinationCapacity,
        });
    }
    budget.require_selection_time()?;
    let selection_bytes = selection
        .retained_payload_bytes()
        .ok_or_else(|| error("checked selection capacity overflow"))?;
    if inventory
        .retained_payload_bytes()
        .and_then(|n| n.checked_add(selection_bytes))
        .is_none_or(|n| n > inventory_limit)
    {
        return Err(error("checked selection exceeds shared retained capacity"));
    }
    tracing::info!(candidate_cases = cases.len(),
        checked_cases = inventory.opportunities.iter().filter(|o| matches!(o.population, CasePopulation::Unique(_))).count(),
        inventory_gaps = inventory.gaps.len(),
        populations = selection.populations.len(),
        scheduled_populations = selection.populations.iter().filter(|p| p.scheduled).count(),
        source_batches = selection.batches.iter().filter(|b| b.scheduled).count(),
        planned_requests = selection.requests,
        planned_waves = selection.serial_wave_upper_bound,
        selection_gaps = selection.gaps.len(),
        input_geometry = ?selection.input_geometry,
        preflight = ?budget.preflight_charge(),
        "Automatic checked input selection before numerical collection");
    // Cold, bounded source-level diagnostics retain the reason an installed
    // policy was not scheduled without dumping the input inventory or recipes.
    for (batch_index, batch) in selection.batches.iter().enumerate().take(64) {
        let configured = batch
            .representative_case_indices
            .iter()
            .any(|&i| cases[i].preset == SloAutomaticCostProbeSamplingPresetV1::Configured);
        let greedy_length = batch
            .representative_case_indices
            .iter()
            .any(|&i| cases[i].preset == SloAutomaticCostProbeSamplingPresetV1::GreedyLength);
        let mut matching = selection.gaps.iter().filter(|gap| {
            batch
                .population_indices
                .iter()
                .any(|&i| gap.population.as_ref() == Some(&selection.populations[i].key))
        });
        let reasons: [Option<&selection::SelectionGapReason>; 3] =
            std::array::from_fn(|_| matching.next().map(|gap| &gap.reason));
        tracing::info!(
            batch_index,
            scheduled = batch.scheduled,
            populations = batch.population_indices.len(),
            combination_algorithms = batch
                .algorithm_universe
                .as_ref()
                .map(|universe| universe.algorithm_count()),
            representatives = batch.representative_case_indices.len(),
            configured,
            greedy_length,
            cycles = ?batch.periodic_cycles(),
            occurrences = batch.execution_case_count()?,
            requests = batch.requests,
            serial_waves = batch.serial_wave_upper_bound,
            serial_token_work = batch.serial_token_work,
            maximum_anchor_span = ?batch.periodic_anchor_span(),
            ?reasons,
            "Automatic input source budget and policy coverage"
        );
        if batch.scheduled {
            for &case_index in &batch.representative_case_indices {
                let case = &cases[case_index];
                tracing::info!(
                    batch_index,
                    case_index,
                    rendered_template = case.template,
                    original_template = input.original_template_indices[case.template],
                    output = ?input.templates[case.template].output(),
                    preset = ?case.preset,
                    route = ?case.route,
                    target_product = ?case.product,
                    rows = case.width,
                    maximum_output = case.maximum_output.get(),
                    "Automatic scheduled source original representative"
                );
            }
        }
        #[cfg(test)]
        eprintln!("automatic input source: index={batch_index} scheduled={} populations={} representatives={} configured={configured} greedy_length={greedy_length} cycles={:?} requests={} serial_waves={} token_work={} anchor_span={:?} reasons={reasons:?}",
            batch.scheduled, batch.population_indices.len(), batch.representative_case_indices.len(),
            batch.periodic_cycles(), batch.requests, batch.serial_wave_upper_bound,
            batch.serial_token_work, batch.periodic_anchor_span());
    }
    if selection.batches.len() > 64 {
        tracing::info!(
            omitted = selection.batches.len() - 64,
            "Remaining automatic input source diagnostics retained in frozen manifest"
        );
    }
    if selection.execution_case_indices.is_empty() {
        #[cfg(test)]
        {
            // Source8 has not started here. Retain the original typed reasons
            // even when no tracing subscriber is installed by a CPU test.
            // This cold diagnostic neither adds members nor changes selection.
            eprintln!("automatic empty input selection: candidates={} checked={} guaranteed={} input_facts={} requests_remaining={requests_remaining} waves_remaining={waves_remaining} populations={} batches={} inventory_gaps={} selection_gaps={} skipped={} unavailable={unavailable:?}",
                cases.len(),
                inventory.opportunities.iter().filter(|o| matches!(o.population, CasePopulation::Unique(_))).count(),
                inventory.opportunities.iter().filter(|o| o.minimum_fresh_members > 0).count(),
                inventory.inputs.iter().map(Vec::len).sum::<usize>(),
                selection.populations.len(), selection.batches.len(), inventory.gaps.len(), selection.gaps.len(), skipped);
            for gap in inventory.gaps.iter().take(32) {
                let case = &cases[gap.case_index];
                eprintln!("automatic inventory gap: case={} width={} route={:?} preset={:?} release={} output={} reason={:?}",
                    gap.case_index, case.width, case.route, case.preset, case.release_generated, case.maximum_output, gap.reason);
            }
            for gap in selection.gaps.iter().take(16) {
                eprintln!("automatic empty selection gap: reason={:?}", gap.reason);
            }
        }
        if allow_empty {
            tracing::info!(
                candidate_cases = cases.len(),
                inventory_gaps = inventory.gaps.len(),
                selection_gaps = selection.gaps.len(),
                "Automatic input unit has no executable population; full obligations remain declared"
            );
            return Ok(None);
        }
        return Err(error(format!(
            "checked automatic input selection has no executable population: candidates={}, checked={}, requests_remaining={}, waves_remaining={}, first_projection_gap={:?}, selection_reasons={:?}",
            cases.len(), inventory.opportunities.iter().filter(|o| matches!(o.population, CasePopulation::Unique(_))).count(),
            budget.requests_remaining(), budget.attempts_remaining(),
            inventory.gaps.iter().find(|gap| matches!(gap.reason, InventoryGapReason::Projection(_))),
            selection.gaps.iter().map(|gap| &gap.reason).take(8).collect::<Vec<_>>()
        )));
    }
    // The source8 pass labels describe declared cohort identity. Actual
    // numerical phases remain the original per-population block state machine.
    let count = selection.execution_case_indices.len();
    let mut order: [Vec<usize>; 3] = std::array::from_fn(|_| Vec::new());
    for (ordinal, &index) in selection.execution_case_indices.iter().enumerate() {
        order[(ordinal * 3 / count).min(2)].push(index);
    }
    let mut selected_widths = Vec::new();
    for &index in &selection.execution_case_indices {
        if !selected_widths.contains(&cases[index].width) {
            selected_widths.push(cases[index].width);
        }
    }
    selected_widths.sort_unstable();
    let source_input_allowance = inventory_limit
        .checked_sub(
            inventory
                .retained_payload_bytes()
                .ok_or_else(|| error("source input inventory capacity overflow"))?,
        )
        .and_then(|n| n.checked_sub(selection.retained_payload_bytes()?))
        .and_then(|n| {
            n.checked_sub(
                selected_widths
                    .capacity()
                    .checked_mul(std::mem::size_of::<usize>())?,
            )
        })
        .and_then(|n| {
            order.iter().try_fold(n, |remaining, pass| {
                remaining.checked_sub(pass.capacity().checked_mul(std::mem::size_of::<usize>())?)
            })
        })
        .ok_or_else(|| error("source inputs have no shared retained allowance"))?;
    let source_inputs = source_inputs::freeze_sources(
        &cases,
        &inventory.opportunities,
        &input.prompts,
        &selection,
        input.chunk,
        input.prefill_row_ceiling,
        source_input_allowance,
    )?;
    drop(inventory);
    let plan = freeze(
        input,
        FrozenCases {
            cases,
            order,
            widths: selected_widths,
            skipped,
            unavailable,
            input_opportunities: None,
            checked_selection: Some(selection),
            source_inputs,
            preflight_charge: budget.preflight_charge(),
        },
    )?;
    session.validate_declared_prefix_plan(
        &plan.declaration.cohort_plan,
        &plan.declaration.prefix_plan,
    )?;
    Ok(Some(plan))
}

fn readiness_missing(reason: &InventoryGapReason) -> bool {
    matches!(
        reason,
        InventoryGapReason::Projection(GeometryProjectionUnknown::Route(
            ExecutionCostRouteUnknown::OnDemandResidentProgram
                | ExecutionCostRouteUnknown::Resource(
                    ResourcePlanningUnknown::UnmaterializedCapacity
                )
        ))
    )
}

async fn collect_ready(
    session: &mut CalibrationSession,
    input: &PreparedProbeInputs,
    cases: &[Case],
    budget: &mut ProbeExecutionBudget,
    maximum_bytes: usize,
) -> Result<CheckedCaseInventory> {
    let base = cases
        .len()
        .checked_mul(
            std::mem::size_of::<CaseOpportunity>()
                + std::mem::size_of::<Vec<selection::CheckedInputFacts>>()
                + std::mem::size_of::<Vec<usize>>()
                + std::mem::size_of::<inventory::InventoryGap>()
                + std::mem::size_of::<bool>()
                + 2 * std::mem::size_of::<usize>()
                + std::mem::size_of::<Case>(),
        )
        .filter(|n| *n < maximum_bytes)
        .ok_or_else(|| error("checked inventory base capacity"))?;
    let mut readiness_attempts = readiness::Attempts::new(
        cases
            .len()
            .checked_mul(3)
            .ok_or_else(|| error("readiness route capacity overflow"))?,
        maximum_bytes - base,
    )?;
    let base = base
        .checked_add(readiness_attempts.retained_payload_bytes()?)
        .filter(|n| *n < maximum_bytes)
        .ok_or_else(|| error("checked inventory readiness capacity"))?;
    let mut out = CheckedCaseInventory {
        opportunities: (0..cases.len())
            .map(|_| CaseOpportunity {
                population: CasePopulation::Unknown {
                    known_alternatives: Vec::new(),
                },
                minimum_fresh_members: 0,
            })
            .collect(),
        inputs: (0..cases.len()).map(|_| Vec::new()).collect(),
        original_inputs: Vec::new(),
        algorithm_inputs: Vec::new(),
        algorithm_case_inputs: (0..cases.len()).map(|_| Vec::new()).collect(),
        gaps: Vec::with_capacity(cases.len()),
        charge: ProbePreflightCharge::default(),
    };
    let mut done = Vec::with_capacity(cases.len());
    let mut prefiltered_cases = 0usize;
    for (case_index, case) in cases.iter().enumerate() {
        let skipped = !readiness::needs_inventory(case);
        done.push(skipped);
        if skipped {
            // Ordinary warm-prefill execution has no guaranteed fresh member.
            // Keep the original case and its explicit gap, without admitting
            // roots that cannot contribute to checked selection.
            prefiltered_cases += 1;
            out.gaps.push(inventory::InventoryGap {
                case_index,
                reason: InventoryGapReason::WarmResidencyUnproven,
            });
        }
    }
    let initial_inventory_admissions = readiness::initial_inventory_admissions(cases)?;
    tracing::info!(
        candidate_cases = cases.len(),
        prefiltered_cases,
        initial_inventory_admissions,
        "Automatic checked inventory before live admissions"
    );
    for first in 0..cases.len() {
        if done[first] {
            continue;
        }
        if Instant::now() >= budget.deadline() {
            return Err(error("checked inventory shared deadline expired"));
        }
        let indices: Vec<_> = (first..cases.len())
            .filter(|&i| !done[i] && inventory::same_group(&cases[first], &cases[i]))
            .collect();
        let group: Vec<_> = indices.iter().map(|&i| cases[i].clone()).collect();
        let remaining_bytes = maximum_bytes
            .checked_sub(
                out.retained_payload_bytes()
                    .and_then(|n| n.checked_add(base))
                    .ok_or_else(|| error("checked inventory retained overflow"))?,
            )
            .filter(|n| *n > 0)
            .ok_or_else(|| error("checked inventory retained exhausted"))?;
        let prepared = readiness_projection::prepare(
            session,
            input,
            &group,
            cases,
            budget,
            remaining_bytes,
            &mut readiness_attempts,
        )
        .await?;
        // An uninterrupted first traversal already captured every input.
        // Any preparation action still requires a fresh complete inventory.
        let mut captured = match prepared {
            Some(inventory) => inventory,
            None => capture(session, input, &group, budget, remaining_bytes).await?,
        };
        for gap in &captured.gaps {
            tracing::debug!(case = indices[gap.case_index], rows = cases[indices[gap.case_index]].width,
                reason = ?gap.reason, "Automatic checked input remains unavailable");
        }
        let external = base
            .checked_add(
                captured
                    .retained_payload_bytes()
                    .ok_or_else(|| error("algorithm capture retained overflow"))?,
            )
            .ok_or_else(|| error("algorithm capture retained overflow"))?;
        out.merge_algorithms(&captured, &indices, maximum_bytes, external)?;
        out.merge_originals(&mut captured, maximum_bytes, base)?;
        drop(captured.algorithm_inputs);
        drop(captured.algorithm_case_inputs);
        for ((&global, opportunity), facts) in indices
            .iter()
            .zip(captured.opportunities.into_iter())
            .zip(captured.inputs.into_iter())
        {
            done[global] = true;
            out.opportunities[global] = opportunity;
            out.inputs[global] = facts;
        }
        out.gaps.extend(captured.gaps.into_iter().map(|mut gap| {
            gap.case_index = indices[gap.case_index];
            gap
        }));
        if out
            .retained_payload_bytes()
            .is_none_or(|n| n > maximum_bytes)
        {
            return Err(error("checked inventory retained capacity exhausted"));
        }
    }
    out.charge = budget.preflight_charge();
    Ok(out)
}

async fn capture(
    session: &mut CalibrationSession,
    input: &PreparedProbeInputs,
    cases: &[Case],
    budget: &mut ProbeExecutionBudget,
    maximum_retained_bytes: usize,
) -> Result<CheckedCaseInventory> {
    let projection_remaining = input
        .settings
        .maximum_offered_waves
        .get()
        .checked_sub(budget.preflight_charge().projection_attempts)
        .filter(|n| *n > 0)
        .ok_or_else(|| error("pure projection budget exhausted"))?;
    // The grouped capture may fail after creating only some owners. Reserve
    // all of its original owner slots before admission and never refund them.
    let reserved = readiness::initial_inventory_admissions(cases)?;
    budget.claim_input_projection_requests(reserved)?;
    let mut charge = ProbePreflightCharge::default();
    let report = inventory::collect_charged(
        session,
        cases,
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
        &mut charge,
    )
    .await;
    budget.record_geometry(charge)?;
    tracing::info!(
        template = cases.first().map(|case| case.template),
        stage = "complete_inventory",
        succeeded = report.is_ok(),
        ?charge,
        "Automatic checked inventory projection completed"
    );
    report
}
