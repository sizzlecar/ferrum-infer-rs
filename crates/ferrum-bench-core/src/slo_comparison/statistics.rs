//! Reproducible paired-cluster CI computation, with calibration kept explicit.
//!
//! Empirical resampling follows Efron (1979), doi:10.1214/aos/1176344552.
//! Percentile coverage is approximate; Hall (1988), doi:10.1214/aos/1176350933,
//! explains coverage limitations. Bonferroni covers the full frozen primary
//! family, not a post-hoc subset. Neither a supplied interval nor a pilot digest
//! verifies small-sample/P99 adequacy. Raw-pilot eligibility is a separate,
//! explicit path to a conditional decision, never a population-coverage proof.

use super::*;

mod compute;
mod configuration;
mod design;
pub(super) mod eligibility;
mod types;
pub use eligibility::{
    compare_with_eligibility, verify_inference_eligibility, EligibilityFailure,
    FrozenEligibilityPlan, InferenceAssumptions, InferenceEligibilityReport,
    PairAllocationForecast, PilotMetricDiagnostic, PlanningMetricForecast,
    VerifiedInferenceEligibility,
};
pub use types::*;

const METHOD_ID: &str = "paired-cluster-percentile-bonferroni-v1";
const ANALYSIS_UNIT: &str = "complete_paired_repetition_across_primary_cells";
const RANDOM_GENERATOR: &str = "sha256-counter-le64-rejection-v1";
pub(super) const UNVERIFIED_CELL_ISSUE: &str = "built-in cluster intervals are reported separately; independent eligibility remains unverified";
const UNVERIFIED_CALIBRATION_ISSUE: &str = "raw independent pilot eligibility is unverified; declared support and computed intervals alone cannot establish ProofPass";

pub(super) fn validate_method(
    contract: &FrozenComparisonContract,
    method: &FrozenStatisticalMethod,
) -> Result<(), ComparisonError> {
    configuration::validate_method(contract, method)
}

pub(super) fn evaluate(
    contract: &FrozenComparisonContract,
    cells: &[CellComparison],
    report_issues: &[String],
) -> Result<Option<ComputedBootstrapInference>, ComparisonError> {
    let Some(method) = contract.uncertainty.as_ref() else {
        return Ok(None);
    };
    let Some(configuration) = method.paired_bootstrap.as_ref() else {
        return Ok(None);
    };
    let primary: Vec<_> = cells
        .iter()
        .filter(|cell| cell.scope == CellScope::Primary)
        .collect();
    let family_size = primary.len() * ComparisonMetric::ALL.len();
    let tail =
        (configuration.family_alpha - configuration.monte_carlo_error_budget) / family_size as f64;
    let rank = compute::tail_rank(
        configuration.resamples as usize,
        tail,
        configuration.monte_carlo_error_budget / family_size as f64,
    );
    let measurements: Vec<_> = primary
        .iter()
        .map(|cell| (&cell.concurrency, &cell.paired_measurements_sha256))
        .collect();
    let mut result = ComputedBootstrapInference {
        method_id: method.method_id.clone(),
        analysis_unit: method.analysis_unit.clone(),
        estimand: configuration.estimand,
        configuration_sha256: method.configuration_sha256.clone(),
        paired_measurements_sha256: json_digest(&measurements)?,
        random_generator: RANDOM_GENERATOR.into(),
        seed: configuration.seed,
        resamples: configuration.resamples,
        family_size,
        family_alpha: configuration.family_alpha,
        monte_carlo_error_budget: configuration.monte_carlo_error_budget,
        per_comparison_tail_probability: tail,
        conservative_tail_rank: rank,
        calibration: BootstrapCalibrationStatus::Unverified,
        status: ComparisonStatus::Inconclusive,
        cells: Vec::new(),
        issues: design::issues(contract, &primary, configuration),
        scope: "Approximate one-sided percentile bootstrap bounds for arithmetic mean paired ratios of complete fixed-workload runs. Requests and visible gaps within a run are not treated as IID. Raw independent-pilot eligibility verifies planning, support and acquisition diagnostics; independent exchangeable blocks and approximate population coverage remain explicit assumptions that finite pilot data cannot prove. Hashes/order checks are not physical attestation. No inference to the entire ShareGPT population, semantic output quality, or other hardware/configurations.".into(),
    };
    if !report_issues.is_empty() {
        result.issues.push("comparison includes duplicate/unfrozen cells; no family member may be silently selected away".into());
    }
    if rank.is_none() {
        result.issues.push("resample tail resolution is insufficient for the frozen family and Monte Carlo error budget".into());
    }
    let eligible = result.issues.is_empty();
    // This computation-only path never substitutes a narrow interval for the
    // separate verifier's raw independent pilot evidence.
    result.issues.push(UNVERIFIED_CALIBRATION_ISSUE.into());
    if !eligible {
        return Ok(Some(result));
    }

    let mut columns = Vec::with_capacity(family_size);
    for cell in &primary {
        for metric in ComparisonMetric::ALL {
            let column: Option<Vec<_>> = cell
                .pairs
                .iter()
                .map(|pair| {
                    pair.ratios
                        .get(&metric)
                        .map(|ratio| ratio.candidate_over_baseline)
                })
                .collect();
            let Some(column) = column else {
                result
                    .issues
                    .push("missing ratio in the full primary matrix".into());
                return Ok(Some(result));
            };
            columns.push(column);
        }
    }
    let mut distributions = compute::resample_means(
        &columns,
        configuration.resamples as usize,
        configuration.seed,
    )?;
    let rank = rank.expect("eligible simulation has a conservative rank");
    let mut column_index = 0;
    for cell in primary {
        let mut metrics = BTreeMap::new();
        for metric in ComparisonMetric::ALL {
            let column = &columns[column_index];
            let distribution = &mut distributions[column_index];
            let degenerate = column.iter().all(|value| *value == column[0]);
            let point = mean(column);
            let direction = if metric.is_throughput() {
                BootstrapBoundDirection::Lower
            } else {
                BootstrapBoundDirection::Upper
            };
            let bound = if degenerate {
                result.issues.push(format!("C{} / {metric:?} has no observed between-pair variation; a zero-width bootstrap interval is not evidence of certainty", cell.concurrency));
                None
            } else {
                distribution.sort_unstable_by(f64::total_cmp);
                Some(
                    distribution[if metric.is_throughput() {
                        rank - 1
                    } else {
                        distribution.len() - rank
                    }],
                )
            };
            let relative_width = bound
                .map(|bound| {
                    if metric.is_throughput() {
                        point / bound - 1.0
                    } else {
                        bound / point - 1.0
                    }
                })
                .filter(|width| width.is_finite())
                .map(|width| width.max(0.0));
            let maximum_width = configuration.maximum_relative_bound_width[&metric];
            let strict_limit_cleared = bound.is_some_and(|bound| {
                if metric.is_throughput() {
                    bound > contract.ratio_limits[&metric]
                } else {
                    bound < contract.ratio_limits[&metric]
                }
            });
            metrics.insert(
                metric,
                BootstrapMetricBound {
                    point_estimate: point,
                    direction,
                    one_sided_bound: bound,
                    relative_bound_width: relative_width,
                    declared_maximum_relative_width: maximum_width,
                    precision_target_met: relative_width
                        .is_some_and(|width| width <= maximum_width),
                    strict_limit_cleared,
                    degenerate_empirical_distribution: degenerate,
                },
            );
            column_index += 1;
        }
        result.cells.push(BootstrapPrimaryCell {
            concurrency: cell.concurrency,
            pairs: cell.pairs.len(),
            metrics,
        });
    }
    Ok(Some(result))
}

#[cfg(test)]
mod tests;
