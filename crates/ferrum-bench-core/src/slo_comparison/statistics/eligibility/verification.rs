use super::*;

pub(super) fn validate_binding<'a>(
    contract: &'a FrozenComparisonContract,
    plan: &FrozenEligibilityPlan,
    pilot_contract: &FrozenComparisonContract,
    source: &str,
) -> Result<&'a FrozenPairedBootstrap, EligibilityFailure> {
    let configuration = contract
        .uncertainty
        .as_ref()
        .and_then(|method| method.paired_bootstrap.as_ref())
        .ok_or_else(|| failure("eligibility needs the implemented frozen bootstrap method"))?;
    let primary: Vec<_> = contract
        .cells
        .iter()
        .filter(|cell| cell.scope == CellScope::Primary)
        .collect();
    planning::validate(plan, configuration, primary.len())?;
    configuration::check_work(pilot_contract.pairs.len(), 1, primary.len())?;
    let plan_digest = json_digest(plan)?;
    if configuration
        .eligibility_plan_sha256
        .as_deref()
        .is_none_or(|hash| !same_digest(hash, &plan_digest))
        || !valid_digest(source)
        || !same_digest(source, &configuration.declared_design.pilot_source_sha256)
        || plan.frozen_unix_ns > pilot_contract.frozen_unix_ns
        || pilot_contract.frozen_unix_ns >= contract.frozen_unix_ns
        || pilot_contract.uncertainty.is_some()
    {
        return Err(failure(
            "eligibility plan/source binding or independent pilot chronology is invalid",
        ));
    }
    if 1.0
        / f64::from(
            configuration
                .declared_design
                .minimum_measured_requests_per_arm,
        )
        > plan.maximum_request_rank_step
        || 1.0 / configuration.declared_design.minimum_visible_gaps_per_arm as f64
            > plan.maximum_visible_gap_rank_step
    {
        return Err(failure(
            "final declared P99 support is below the frozen empirical rank resolution",
        ));
    }
    let pilot_primary: Vec<_> = pilot_contract
        .cells
        .iter()
        .filter(|cell| cell.scope == CellScope::Primary)
        .collect();
    let dataset = |contract: &FrozenComparisonContract| {
        let mut dataset = contract.dataset.clone();
        dataset.source_path.clear();
        dataset.repeats.clear();
        dataset
    };
    if pilot_primary != primary
        || pilot_contract.baseline != contract.baseline
        || pilot_contract.candidate != contract.baseline
        || pilot_contract.shared != contract.shared
        || pilot_contract.capacity != contract.capacity
        || pilot_contract.slo != contract.slo
        || pilot_contract.memory != contract.memory
        || pilot_contract.sampling != contract.sampling
        || pilot_contract.http_connection_mode != contract.http_connection_mode
        || dataset(pilot_contract) != dataset(contract)
        || pilot_contract
            .pairs
            .iter()
            .any(|pair| pair.selection.samples != contract.pairs[0].selection.samples)
        || contract
            .pairs
            .iter()
            .any(|pair| pair.selection.samples != contract.pairs[0].selection.samples)
    {
        return Err(failure("pilot must use the full primary family, identical baseline A/A identity, fixed configuration and ordered workload"));
    }
    Ok(configuration)
}

pub(super) fn run_key(source: &ObservedArmSource) -> Option<(String, String, u32)> {
    Some((
        source.benchmark_run_id.clone()?,
        source.cell_id.clone()?,
        source.report_repeat_index,
    ))
}

pub(super) fn verify_pilot(
    contract: &FrozenComparisonContract,
    plan: &FrozenEligibilityPlan,
    pilot_contract: &FrozenComparisonContract,
    pilot: &SloComparisonReport,
    configuration: &FrozenPairedBootstrap,
    source: &str,
) -> Result<VerifiedInferenceEligibility, EligibilityFailure> {
    let primary: Vec<_> = pilot
        .cells
        .iter()
        .filter(|cell| cell.scope == CellScope::Primary)
        .collect();
    let mut pilot_configuration = configuration.clone();
    pilot_configuration.declared_design.planned_pairs = pilot_contract.pairs.len() as u32;
    let mut issues = design::issues(pilot_contract, &primary, &pilot_configuration);
    if !pilot.issues.is_empty() {
        issues.push("pilot contains duplicate or unfrozen cells".into());
    }
    if !issues.is_empty() {
        return Err(EligibilityFailure { issues });
    }
    let mut runs = BTreeSet::new();
    let mut finished = 0;
    for cell in &primary {
        for pair in &cell.pairs {
            for source in [&pair.baseline_source, &pair.candidate_source]
                .into_iter()
                .flatten()
            {
                let Some(key) = run_key(source) else {
                    return Err(failure("pilot acquisition has no original run identity"));
                };
                if !runs.insert(key) {
                    return Err(failure(
                        "pilot reuses one acquisition as multiple independent arms/blocks",
                    ));
                }
                if source.offered_requests == 0
                    || source.observed_visible_gaps == 0
                    || 1.0 / source.offered_requests as f64 > plan.maximum_request_rank_step
                    || 1.0 / source.observed_visible_gaps as f64
                        > plan.maximum_visible_gap_rank_step
                {
                    return Err(failure(format!(
                        "pilot C{} lacks declared empirical P99 rank resolution",
                        cell.concurrency
                    )));
                }
                finished = finished.max(source.execution.measurement_ended_unix_ns);
            }
            for memory in [&pair.baseline_memory, &pair.candidate_memory]
                .into_iter()
                .flatten()
            {
                finished = finished
                    .max(memory.device_allocation.ended_unix_ns)
                    .max(memory.os_footprint.ended_unix_ns)
                    .max(memory.maximum_rss.ended_unix_ns);
            }
            let baseline = pair
                .baseline_source
                .as_ref()
                .expect("validated complete pilot");
            let candidate = pair
                .candidate_source
                .as_ref()
                .expect("validated complete pilot");
            let (first, second) = if baseline.execution.measurement_started_unix_ns
                < candidate.execution.measurement_started_unix_ns
            {
                (baseline, candidate)
            } else {
                (candidate, baseline)
            };
            if second
                .execution
                .measurement_started_unix_ns
                .saturating_sub(first.execution.measurement_ended_unix_ns)
                > plan.maximum_within_pair_idle_ns
            {
                return Err(failure(
                    "pilot exceeds frozen within-pair idle/perturbation boundary",
                ));
            }
        }
    }
    if finished > configuration.declared_design.pilot_finished_unix_ns
        || configuration.declared_design.pilot_finished_unix_ns >= contract.frozen_unix_ns
    {
        return Err(failure(
            "raw pilot completion exceeds its declared pre-freeze boundary",
        ));
    }
    let diagnostics = planning::diagnostics(
        &primary,
        plan,
        configuration.declared_design.first_pair_order,
    )?;
    if diagnostics
        .iter()
        .any(|diagnostic| !diagnostic.within_declared_limits)
    {
        return Err(failure(
            "raw pilot order, trend or lag diagnostic exceeds the frozen tolerance",
        ));
    }
    let forecasts = planning::forecasts(plan, configuration, &primary)?;
    let selected_pairs = forecasts.iter().find(|forecast| forecast.all_precision_targets_met)
        .map(|forecast| forecast.planned_pairs)
        .ok_or_else(|| failure("no frozen allocation meets pilot-based precision planning; do not reuse candidate results to alter the plan"))?;
    if selected_pairs as usize != contract.pairs.len() {
        return Err(failure(
            "final pair allocation does not match the first qualifying frozen pilot-planning point",
        ));
    }
    let measurements: Vec<_> = primary
        .iter()
        .map(|cell| (cell.concurrency, &cell.paired_measurements_sha256))
        .collect();
    Ok(VerifiedInferenceEligibility {
        report: InferenceEligibilityReport {
            schema_version: 1,
            final_contract_sha256: json_digest(contract)?,
            plan_sha256: json_digest(plan)?,
            plan: plan.clone(),
            pilot_contract_sha256: json_digest(pilot_contract)?,
            pilot_source_sha256: source.into(),
            pilot_measurements_sha256: json_digest(&measurements)?,
            assumptions: plan.assumptions,
            pilot_finished_unix_ns: configuration.declared_design.pilot_finished_unix_ns,
            selected_pairs,
            planning_resamples: plan.planning_resamples,
            planning_seed: plan.seed,
            diagnostics, forecasts,
            scope: "Eligibility verifies raw A/A pilot identity, complete fixed workload, empirical rank resolution, deterministic precision planning and preregistered order/trend/lag diagnostics. It does not verify real-world IID or bootstrap population coverage. Independent exchangeable complete paired blocks and approximate percentile-bootstrap validity remain explicit assumptions; final measured precision and all simultaneous effect bounds must still pass.".into(),
        },
        pilot_runs: runs,
    })
}
