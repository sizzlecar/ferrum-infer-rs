//! Frozen, paired comparisons of real ShareGPT client-visible SLO reports.
//!
//! Adapters supply the existing sidecar's `legacy_benchmark` and each
//! `repeats[i].evaluation`; this module does not introduce another collector.
//! Hardware/build/memory evidence is bound by an external acquisition adapter.
//! A matching digest is an identity check, not proof of how bytes were measured.

use serde::Serialize;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

pub mod artifact;
mod contract;
mod evidence;
mod render;
mod statistics;
mod types;

use evidence::{observed_source, validate_arm};
pub use render::MarkdownLanguage;
pub use statistics::{
    compare_with_eligibility, verify_inference_eligibility, BootstrapBoundDirection,
    BootstrapCalibrationStatus, BootstrapEstimand, BootstrapMetricBound, BootstrapPrimaryCell,
    ComputedBootstrapInference, DeclaredBootstrapDesign, EligibilityFailure, FrozenEligibilityPlan,
    FrozenPairedBootstrap, InferenceAssumptions, InferenceEligibilityReport,
    PairAllocationForecast, PairedArmOrder, PilotMetricDiagnostic, PlanningMetricForecast,
    VerifiedInferenceEligibility,
};
pub use types::*;

fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn json_digest(value: &impl Serialize) -> Result<String, ComparisonError> {
    serde_json::to_vec(value)
        .map(|bytes| digest(&bytes))
        .map_err(|e| ComparisonError(e.to_string()))
}
fn valid_digest(value: &str) -> bool {
    let raw = value.strip_prefix("sha256:").unwrap_or(value);
    raw.len() == 64 && raw.bytes().all(|b| b.is_ascii_hexdigit())
}
fn same_digest(a: &str, b: &str) -> bool {
    a.strip_prefix("sha256:")
        .unwrap_or(a)
        .eq_ignore_ascii_case(b.strip_prefix("sha256:").unwrap_or(b))
}

fn combine_evidence(statuses: impl IntoIterator<Item = ComparisonStatus>) -> ComparisonStatus {
    let mut unknown = false;
    for status in statuses {
        if status == ComparisonStatus::Failed {
            return status;
        }
        unknown |= status == ComparisonStatus::Unknown;
    }
    if unknown {
        ComparisonStatus::Unknown
    } else {
        ComparisonStatus::ObservedPass
    }
}

fn satisfies(metric: ComparisonMetric, ratio: f64, limit: f64) -> bool {
    if metric.is_throughput() {
        ratio >= limit
    } else {
        ratio <= limit
    }
}

fn compare_pair(
    contract: &FrozenComparisonContract,
    concurrency: u32,
    frozen: &FrozenPair,
    input: &PairedRepeatInput<'_>,
    seen: &mut BTreeSet<(bool, String, String, u32)>,
) -> PairedRepeatComparison {
    let baseline = validate_arm(contract, concurrency, frozen, &input.baseline, false, seen);
    let candidate = validate_arm(contract, concurrency, frozen, &input.candidate, true, seen);
    let baseline_source = observed_source(&input.baseline, &baseline);
    let candidate_source = observed_source(&input.candidate, &candidate);
    let mut issues: Vec<String> = baseline
        .issues
        .iter()
        .map(|s| format!("baseline: {s}"))
        .chain(candidate.issues.iter().map(|s| format!("candidate: {s}")))
        .collect();
    let mut evidence_status = combine_evidence([baseline.status, candidate.status]);
    let b = input.baseline.execution;
    let c = input.candidate.execution;
    if b.measurement_started_unix_ns < c.measurement_ended_unix_ns
        && c.measurement_started_unix_ns < b.measurement_ended_unix_ns
    {
        issues.push("same-machine paired measurement windows overlap".into());
    }
    if baseline.server_inputs != candidate.server_inputs || baseline.outputs != candidate.outputs {
        issues.push("paired requests have different actual prompt/output usage lengths".into());
    }
    if evidence_status != ComparisonStatus::Failed && !issues.is_empty() {
        evidence_status = ComparisonStatus::Unknown;
    }
    let mut ratios = BTreeMap::new();
    for metric in ComparisonMetric::ALL {
        if let (Some(&baseline), Some(&candidate)) = (
            baseline.metrics.get(&metric),
            candidate.metrics.get(&metric),
        ) {
            let ratio = candidate / baseline;
            if ratio.is_finite() && ratio > 0.0 {
                ratios.insert(
                    metric,
                    RepeatRatio {
                        baseline,
                        candidate,
                        candidate_over_baseline: ratio,
                    },
                );
            }
        }
    }
    if ratios.len() != ComparisonMetric::ALL.len() && evidence_status != ComparisonStatus::Failed {
        evidence_status = ComparisonStatus::Unknown;
        issues.push("one or more paired ratios are missing, zero or non-finite".into());
    }
    let status = if evidence_status != ComparisonStatus::ObservedPass {
        evidence_status
    } else if ratios.iter().any(|(&metric, value)| {
        !satisfies(
            metric,
            value.candidate_over_baseline,
            contract.ratio_limits[&metric],
        )
    }) {
        ComparisonStatus::Failed
    } else {
        ComparisonStatus::ObservedPass
    };
    PairedRepeatComparison {
        pair_id: input.pair_id.clone(),
        ratios,
        baseline_memory: input.baseline.memory.cloned(),
        candidate_memory: input.candidate.memory.cloned(),
        baseline_source: Some(baseline_source),
        candidate_source: Some(candidate_source),
        issues,
        evidence_status,
        status,
    }
}

fn absent_pair(pair_id: &str) -> PairedRepeatComparison {
    PairedRepeatComparison {
        pair_id: pair_id.into(),
        ratios: BTreeMap::new(),
        baseline_memory: None,
        candidate_memory: None,
        baseline_source: None,
        candidate_source: None,
        issues: vec!["missing frozen paired repetition".into()],
        evidence_status: ComparisonStatus::Unknown,
        status: ComparisonStatus::Unknown,
    }
}

// Divide before summing to avoid overflowing a finite arithmetic mean.
fn mean(values: &[f64]) -> f64 {
    values.iter().map(|v| *v / values.len() as f64).sum()
}

/// Evaluates all frozen cells and all declared pairs. Observed point estimates
/// use the arithmetic mean of paired ratios. No repetition count or threshold
/// is hardcoded, and no confidence interval is inferred from a point estimate.
/// In this version valid descriptive success is always Inconclusive for proof.
pub fn compare_slo_reports(
    contract: &FrozenComparisonContract,
    inputs: &[ComparisonCellInput<'_>],
) -> Result<SloComparisonReport, ComparisonError> {
    contract.validate()?;
    let mut report_issues = Vec::new();
    let mut input_cells = BTreeMap::new();
    for input in inputs {
        if !contract
            .cells
            .iter()
            .any(|cell| cell.concurrency == input.concurrency)
        {
            report_issues.push(format!("unfrozen cell C{} supplied", input.concurrency));
        }
        if input_cells.insert(input.concurrency, input).is_some() {
            report_issues.push(format!("duplicate cell C{} supplied", input.concurrency));
        }
    }
    let mut seen = BTreeSet::new();
    let mut cells = Vec::new();
    for frozen_cell in &contract.cells {
        let input = input_cells.get(&frozen_cell.concurrency).copied();
        let mut issues = Vec::new();
        let mut input_pairs = BTreeMap::new();
        if let Some(input) = input {
            for pair in &input.pairs {
                if !contract
                    .pairs
                    .iter()
                    .any(|frozen| frozen.pair_id == pair.pair_id)
                {
                    issues.push(format!("unfrozen pair {} supplied", pair.pair_id));
                }
                if input_pairs.insert(&pair.pair_id, pair).is_some() {
                    issues.push(format!("duplicate pair {} supplied", pair.pair_id));
                }
            }
        } else {
            issues.push("missing frozen cell".into());
        }
        let pairs: Vec<_> = contract
            .pairs
            .iter()
            .map(|pair| {
                input_pairs
                    .get(&pair.pair_id)
                    .map(|input| {
                        compare_pair(contract, frozen_cell.concurrency, pair, input, &mut seen)
                    })
                    .unwrap_or_else(|| absent_pair(&pair.pair_id))
            })
            .collect();
        let mut evidence_status = combine_evidence(pairs.iter().map(|pair| pair.evidence_status));
        if evidence_status != ComparisonStatus::Failed && !issues.is_empty() {
            evidence_status = ComparisonStatus::Unknown;
        }
        let mut metrics = BTreeMap::new();
        for metric in ComparisonMetric::ALL {
            let ratios: Vec<_> = pairs
                .iter()
                .filter_map(|pair| {
                    pair.ratios
                        .get(&metric)
                        .map(|value| value.candidate_over_baseline)
                })
                .collect();
            let complete = ratios.len() == contract.pairs.len();
            let average = complete
                .then(|| mean(&ratios))
                .filter(|value| value.is_finite() && *value > 0.0);
            let range = complete.then(|| {
                (
                    ratios.iter().copied().fold(f64::INFINITY, f64::min),
                    ratios.iter().copied().fold(0.0, f64::max),
                )
            });
            let limit = contract.ratio_limits[&metric];
            let status = if evidence_status != ComparisonStatus::ObservedPass {
                evidence_status
            } else {
                match average {
                    Some(value) if satisfies(metric, value, limit) => {
                        ComparisonStatus::ObservedPass
                    }
                    Some(_) => ComparisonStatus::Failed,
                    None => ComparisonStatus::Unknown,
                }
            };
            metrics.insert(
                metric,
                MetricComparison {
                    limit,
                    mean_paired_ratio: average,
                    observed_ratio_range: range,
                    status,
                },
            );
        }
        let descriptive_status = combine_evidence(metrics.values().map(|value| value.status));
        let paired_measurements_sha256 = json_digest(&pairs)?;
        let statistical_evidence = input.and_then(|input| input.statistical_evidence).cloned();
        if let Some(evidence) = &statistical_evidence {
            if !evidence.confidence_level.is_finite()
                || evidence
                    .ratio_intervals
                    .values()
                    .any(|&(lo, hi)| !lo.is_finite() || !hi.is_finite())
            {
                return Err(ComparisonError(
                    "statistical evidence contains non-finite values".into(),
                ));
            }
            if contract.uncertainty.as_ref() != Some(&evidence.method)
                || !same_digest(
                    &evidence.paired_measurements_sha256,
                    &paired_measurements_sha256,
                )
                || !valid_digest(&evidence.source_artifact_sha256)
                || !evidence.confidence_level.is_finite()
                || evidence.confidence_level <= 0.0
                || evidence.confidence_level >= 1.0
                || evidence.ratio_intervals.len() != ComparisonMetric::ALL.len()
                || evidence
                    .ratio_intervals
                    .values()
                    .any(|&(lo, hi)| !lo.is_finite() || !hi.is_finite() || lo <= 0.0 || hi < lo)
            {
                issues.push("supplied statistical evidence has mismatched method, measurements or invalid intervals".into());
            }
        }
        issues.push(if contract.uncertainty.is_none() { "no inferential method was frozen; paired means and ranges are descriptive" }
            else if contract.uncertainty.as_ref().is_some_and(|method| method.paired_bootstrap.is_some()) {
                statistics::UNVERIFIED_CELL_ISSUE
            } else { "inferential method verification is not implemented; supplied intervals cannot establish proof" }.into());
        let status = match descriptive_status {
            ComparisonStatus::Failed | ComparisonStatus::Unknown => descriptive_status,
            _ => ComparisonStatus::Inconclusive,
        };
        cells.push(CellComparison {
            concurrency: frozen_cell.concurrency,
            scope: frozen_cell.scope,
            pairs,
            metrics,
            issues,
            status,
            statistical_evidence,
            paired_measurements_sha256,
        });
    }
    let primary = combine_evidence(
        cells
            .iter()
            .filter(|cell| cell.scope == CellScope::Primary)
            .map(|cell| cell.status),
    );
    let status = if primary == ComparisonStatus::Failed {
        primary
    } else if primary == ComparisonStatus::Unknown || !report_issues.is_empty() {
        ComparisonStatus::Unknown
    } else {
        ComparisonStatus::Inconclusive
    };
    let computed_bootstrap = statistics::evaluate(contract, &cells, &report_issues)?;
    Ok(SloComparisonReport { schema_version: 1, frozen_contract_sha256: json_digest(contract)?, contract: contract.clone(),
        cells, issues: report_issues, status,
        uncertainty_scope: "Arithmetic mean of paired ratios; observed min/max are not confidence bounds. Optional paired-cluster bootstrap intervals have unverified independent-pilot calibration and cannot establish ProofPass. Output validity covers collector protocol/empty-output checks, not semantic task accuracy. Identity manifests require acquisition-layer verification.".into(),
        computed_bootstrap, inference_eligibility: None, inference_eligibility_failure: None })
}

#[cfg(test)]
mod tests;
