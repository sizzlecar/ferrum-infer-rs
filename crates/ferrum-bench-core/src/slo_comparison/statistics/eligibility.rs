//! Raw-pilot eligibility and a conditional approximate-inference decision.
//!
//! Planning assesses empirical precision, not population coverage. A/A
//! diagnostics do not assume E[A/B] == 1. Eligibility cannot establish IID;
//! independent/exchangeable blocks remain explicit experimental assumptions.

use super::*;

mod planning;
mod types;
mod verification;
pub use types::*;

fn failure(message: impl Into<String>) -> EligibilityFailure {
    EligibilityFailure {
        issues: vec![message.into()],
    }
}

/// Recompute the complete A/A pilot from original collector records. The
/// source digest identifies acquisition bytes (computed by the artifact loader)
/// and is not trusted as evidence of performance or execution by itself.
pub fn verify_inference_eligibility(
    contract: &FrozenComparisonContract,
    plan: &FrozenEligibilityPlan,
    pilot_contract: &FrozenComparisonContract,
    pilot_inputs: &[ComparisonCellInput<'_>],
    pilot_source_sha256: &str,
) -> Result<VerifiedInferenceEligibility, EligibilityFailure> {
    contract.validate()?;
    pilot_contract.validate()?;
    let configuration =
        verification::validate_binding(contract, plan, pilot_contract, pilot_source_sha256)?;
    let pilot = compare_slo_reports(pilot_contract, pilot_inputs)?;
    verification::verify_pilot(
        contract,
        plan,
        pilot_contract,
        &pilot,
        configuration,
        pilot_source_sha256,
    )
}

/// A ProofPass is a contract decision under explicitly declared assumptions
/// and approximate simultaneous bootstrap bounds, never an unconditional
/// finite-sample guarantee. A supplied report or bool cannot unlock this path.
pub fn compare_with_eligibility(
    contract: &FrozenComparisonContract,
    inputs: &[ComparisonCellInput<'_>],
    eligibility: &VerifiedInferenceEligibility,
) -> Result<SloComparisonReport, ComparisonError> {
    if json_digest(contract)? != eligibility.report.final_contract_sha256 {
        return Err(ComparisonError(
            "eligibility belongs to a different frozen contract".into(),
        ));
    }
    let mut report = compare_slo_reports(contract, inputs)?;
    let mut acquisition_issues = Vec::new();
    for cell in &report.cells {
        for pair in &cell.pairs {
            for source in [&pair.baseline_source, &pair.candidate_source]
                .into_iter()
                .flatten()
            {
                if verification::run_key(source)
                    .is_some_and(|key| eligibility.pilot_runs.contains(&key))
                    || source.execution.measurement_started_unix_ns
                        <= eligibility.report.pilot_finished_unix_ns
                {
                    return Err(ComparisonError(
                        "main experiment reuses pilot evidence or precedes pilot completion".into(),
                    ));
                }
            }
            if cell.scope == CellScope::Primary {
                if let (Some(baseline), Some(candidate)) =
                    (&pair.baseline_source, &pair.candidate_source)
                {
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
                        > eligibility.report.plan.maximum_within_pair_idle_ns
                    {
                        acquisition_issues.push(format!(
                            "C{} / {} exceeds frozen main-experiment within-pair idle boundary",
                            cell.concurrency, pair.pair_id
                        ));
                    }
                }
            }
        }
    }
    report.inference_eligibility = Some(eligibility.report.clone());
    let Some(inference) = report.computed_bootstrap.as_mut() else {
        return Err(ComparisonError(
            "eligible comparison lacks its frozen bootstrap method".into(),
        ));
    };
    inference.calibration = BootstrapCalibrationStatus::EligibleUnderDeclaredAssumptions;
    inference
        .issues
        .retain(|issue| issue != UNVERIFIED_CALIBRATION_ISSUE);
    let eligible_primary: Vec<_> = report
        .cells
        .iter()
        .filter(|cell| cell.scope == CellScope::Primary)
        .collect();
    let configuration = contract
        .uncertainty
        .as_ref()
        .and_then(|method| method.paired_bootstrap.as_ref())
        .expect("eligibility validates method");
    let complete = report.issues.is_empty()
        && acquisition_issues.is_empty()
        && design::issues(contract, &eligible_primary, configuration).is_empty()
        && inference.cells.len() == eligible_primary.len()
        && inference.cells.iter().all(|cell| {
            cell.metrics.len() == ComparisonMetric::ALL.len()
                && cell.metrics.values().all(|metric| {
                    metric.precision_target_met
                        && metric.strict_limit_cleared
                        && !metric.degenerate_empirical_distribution
                        && metric.one_sided_bound.is_some()
                })
        });
    inference.issues.extend(acquisition_issues);
    if complete && report.status == ComparisonStatus::Inconclusive {
        report.status = ComparisonStatus::ProofPass;
        inference.status = ComparisonStatus::ProofPass;
        for cell in &mut report.cells {
            if cell.scope == CellScope::Primary {
                cell.status = ComparisonStatus::ProofPass;
            }
        }
    }
    for cell in &mut report.cells {
        cell.issues.retain(|issue| issue != UNVERIFIED_CELL_ISSUE);
    }
    report.uncertainty_scope = "Contract decision under declared independent, exchangeable complete paired-block assumptions using approximate simultaneous percentile-bootstrap bounds. Raw pilot verifies planning, support and acquisition diagnostics, not real-world IID or population coverage. Scope is the frozen fixed ShareGPT workload; no semantic-quality, other-model or other-hardware claim.".into();
    Ok(report)
}

#[cfg(test)]
pub(in crate::slo_comparison) mod tests;
