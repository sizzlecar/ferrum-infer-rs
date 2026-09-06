//! Replay the concrete legacy-Metal submission probe. CI artifact/candidate
//! provenance is checked by the caller; this document does not attest its own
//! origin, any quantized operator, or model performance.
use ferrum_bench_core::release_regression::{
    numerics::{
        compare_outputs, compare_submission_segments, NumericalMetrics, RawOutput, SubmissionPhase,
        SUBMISSION_PHASES,
    },
    submission::{submission_scope, SubmissionConfig, PROBE_EXECUTION_PATH, SUBMISSION_CHECK_ID},
    Backend, ObligationScope,
};
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Document {
    schema_version: u32,
    config: ProbeConfig,
    output_shape: [usize; 2],
    expected_output_elements: usize,
    configured_precision: Precision,
    executed_precision: Option<Precision>,
    execution_path: String,
    #[serde(rename = "coverage")]
    _coverage: String,
    submission_phases: Option<Vec<SubmissionPhase>>,
    submission_metrics: Option<Vec<NumericalMetrics>>,
    started_at: String,
    completed: bool,
    finished_at: Option<String>,
    execution_elapsed_ms: Option<f64>,
    result: Option<ProbeResult>,
}
#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct ProbeConfig {
    require_backend: Backend,
    op: String,
    tokens: usize,
    intermediate: usize,
    k: usize,
    seed: u64,
    max_nmse: f64,
}
impl ProbeConfig {
    fn fixture(&self) -> SubmissionConfig {
        SubmissionConfig {
            tokens: self.tokens,
            intermediate: self.intermediate,
            k: self.k,
            seed: self.seed,
            max_nmse: self.max_nmse,
        }
    }
}
#[derive(Debug, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Precision {
    reference_storage: String,
    backend_input_storage: String,
    backend_output_storage: String,
    kernel_entrypoint: String,
}
#[derive(Debug, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
enum Status {
    Passed,
    Failed,
    NotRun,
}
#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct ProbeResult {
    schema_version: u32,
    op: String,
    backend: Backend,
    seed: u64,
    tolerance: Option<f64>,
    tolerance_f64_bits: u64,
    status: Status,
    reason: Option<String>,
    reference: Option<RawOutput>,
    actual: Option<RawOutput>,
    metrics: Option<NumericalMetrics>,
}

#[derive(Debug)]
pub(super) struct VerifiedSubmission {
    config: SubmissionConfig,
    metrics: Vec<NumericalMetrics>,
    elapsed_ms: f64,
}
impl VerifiedSubmission {
    pub(super) fn checker_id(&self) -> &'static str {
        SUBMISSION_CHECK_ID
    }
    pub(super) fn scope(&self) -> ObligationScope {
        submission_scope()
    }
    pub(super) fn config(&self) -> &SubmissionConfig {
        &self.config
    }
    pub(super) fn metrics(&self) -> &[NumericalMetrics] {
        &self.metrics
    }
    pub(super) fn elapsed_ms(&self) -> f64 {
        self.elapsed_ms
    }
}

fn close(actual: f64, expected: f64) -> bool {
    actual.is_finite()
        && expected.is_finite()
        && (actual - expected).abs() <= 8.0 * f64::EPSILON * actual.abs().max(expected.abs())
}
fn metrics_match(actual: &NumericalMetrics, expected: &NumericalMetrics) -> bool {
    actual.element_count == expected.element_count
        && actual.uses_absolute_mse == expected.uses_absolute_mse
        && close(actual.nmse, expected.nmse)
        && close(actual.max_abs, expected.max_abs)
        && close(actual.reference_mse, expected.reference_mse)
}

pub(super) fn verify(
    bytes: &[u8],
    expected: &SubmissionConfig,
) -> Result<VerifiedSubmission, String> {
    let segment_len = expected.segment_len()?;
    let document: Document =
        serde_json::from_slice(bytes).map_err(|error| format!("submission report: {error}"))?;
    let expected_elements = segment_len
        .checked_mul(SUBMISSION_PHASES.len())
        .ok_or("submission output overflow")?;
    if document.schema_version != 1
        || !document.completed
        || document.started_at.trim().is_empty()
        || document
            .finished_at
            .as_deref()
            .is_none_or(|value| value.trim().is_empty())
    {
        return Err("submission report did not complete".into());
    }
    let elapsed_ms = document
        .execution_elapsed_ms
        .filter(|n| n.is_finite() && *n >= 0.0)
        .ok_or("submission report lacks a valid execution duration")?;
    if document.config.require_backend != Backend::Metal
        || document.config.op != "metal-context"
        || document.config.fixture() != *expected
        || document.execution_path != PROBE_EXECUTION_PATH
    {
        return Err(
            "submission backend, operation, execution path or registered inputs mismatch".into(),
        );
    }
    let precision = Precision {
        reference_storage: "f32".into(),
        backend_input_storage: "f32".into(),
        backend_output_storage: "f32".into(),
        kernel_entrypoint: expected.kernel_entrypoint().into(),
    };
    if document.configured_precision != precision
        || document.executed_precision.as_ref() != Some(&precision)
    {
        return Err("submission did not execute the registered Metal F32 path".into());
    }
    if document.output_shape != [SUBMISSION_PHASES.len(), segment_len]
        || document.expected_output_elements != expected_elements
        || document.submission_phases.as_deref() != Some(SUBMISSION_PHASES.as_slice())
    {
        return Err("submission phase semantics or output shape mismatch".into());
    }
    let result = document.result.ok_or("submission result is missing")?;
    if result.schema_version != 1
        || result.status != Status::Passed
        || result.reason.is_some()
        || result.op != "metal_context"
        || result.backend != Backend::Metal
        || result.seed != expected.seed
        || result.tolerance != Some(expected.max_nmse)
        || result.tolerance_f64_bits != expected.max_nmse.to_bits()
    {
        return Err(
            "submission failed, was not executed, or changed its numerical contract".into(),
        );
    }
    let reference = result
        .reference
        .ok_or("submission CPU reference is missing")?
        .to_f32();
    let actual = result
        .actual
        .ok_or("submission Metal output is missing")?
        .to_f32();
    let overall = compare_outputs(&reference, &actual, expected.max_nmse)
        .map_err(|error| error.to_string())?;
    let metrics = compare_submission_segments(segment_len, &reference, &actual, expected.max_nmse)?;
    if result
        .metrics
        .as_ref()
        .is_none_or(|reported| !metrics_match(reported, &overall))
    {
        return Err("submission aggregate metrics disagree with raw output".into());
    }
    let reported = document
        .submission_metrics
        .ok_or("submission segment metrics are missing")?;
    if reported.len() != metrics.len()
        || reported
            .iter()
            .zip(&metrics)
            .any(|(a, b)| !metrics_match(a, b))
    {
        return Err("submission segment metrics disagree with raw output".into());
    }
    Ok(VerifiedSubmission {
        config: expected.clone(),
        metrics,
        elapsed_ms,
    })
}

#[cfg(test)]
#[path = "submission_tests.rs"]
mod tests;
