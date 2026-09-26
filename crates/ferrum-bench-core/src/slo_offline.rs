//! Re-evaluate retained benchmark request evidence without inventing a sidecar.
//!
//! Last-visible time is recovered only from an observed first SSE text update
//! and a complete sequence of visible gaps. Legacy terminal TPOT is never used.
//! Applying a threshold here does not prove that it was declared before a run.

mod files;
mod reconstruct;
#[cfg(test)]
mod tests;

pub use files::{read_bounded, write_new};

use crate::slo::{evaluate_slo, RequestSloEvidence, SloEvaluationConfig, SloEvaluationReport};
use crate::{BenchRepeatMetrics, BenchReport, BenchmarkRequestCorrelation, Scenario};
use ferrum_types::SloClientVisibleConfig;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Resource limits, not statistical eligibility or workload requirements.
pub const MAX_REPORT_BYTES: usize = 256 * 1024 * 1024;
pub const MAX_CONFIG_BYTES: usize = 1024 * 1024;
const MAX_CELLS: usize = 256;
const MAX_REPEATS: u32 = 4096;
const MAX_REQUESTS: u64 = 262_144;
const MAX_GAPS: usize = 16_777_216;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OfflineSloError(pub String);

impl std::fmt::Display for OfflineSloError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for OfflineSloError {}
fn error(message: impl Into<String>) -> OfflineSloError {
    OfflineSloError(message.into())
}
fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

#[derive(Debug, Serialize)]
pub struct OfflineSloReport {
    pub schema_version: u32,
    pub source: &'static str,
    pub evidence_boundary: &'static str,
    pub threshold_declaration: &'static str,
    pub report_sha256: String,
    pub client_config_sha256: String,
    pub cells: Vec<OfflineSloCell>,
}

#[derive(Debug, Serialize)]
pub struct OfflineSloCell {
    pub input_cell_index: usize,
    pub benchmark_run_id: Option<String>,
    pub cell_id: Option<String>,
    pub model: String,
    pub backend: String,
    pub scenario: Scenario,
    pub concurrency: Option<u32>,
    pub request_rate: Option<f64>,
    pub repeats: Vec<OfflineSloRepeat>,
}

#[derive(Debug, Serialize)]
pub struct OfflineSloRepeat {
    /// Zero-based, unlike the legacy repeat summary's one-based `repeat`.
    pub repeat_index: u32,
    pub legacy_summary: Option<BenchRepeatMetrics>,
    /// Same input row order as evaluation.request_evidence. Missing rows are
    /// appended as Unknown; they are never silently removed from the denominator.
    pub reconstruction: Vec<RequestReconstruction>,
    pub evaluation: Option<SloEvaluationReport>,
    /// Preserved when a missing duration prevents throughput evaluation.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub unevaluated_request_evidence: Option<Vec<RequestSloEvidence>>,
    pub evaluation_unavailable: Option<&'static str>,
}

#[derive(Debug, Serialize)]
pub struct RequestReconstruction {
    pub correlation: Option<BenchmarkRequestCorrelation>,
    pub server_request_id: Option<String>,
    pub issues: Vec<ReconstructionIssue>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReconstructionIssue {
    MissingRequestRecord,
    MissingTiming,
    MissingVisibleCounts,
    MissingObservedFirst,
    IncompleteVisibleGaps,
    NonSseTiming,
    MissingUsage,
    FailedAdmissionUnknown,
}

/// Reads either one canonical BenchReport or an array of cells. Input hashes
/// cover the original bytes, not a reserialized approximation. No wall clock,
/// arrival schedule, rejection status or transport acceptance is manufactured.
pub fn reevaluate_json(
    report_bytes: &[u8],
    client_config_bytes: &[u8],
) -> Result<OfflineSloReport, OfflineSloError> {
    if report_bytes.len() > MAX_REPORT_BYTES || client_config_bytes.len() > MAX_CONFIG_BYTES {
        return Err(error("offline SLO input exceeds its byte limit"));
    }
    let client: SloClientVisibleConfig = serde_json::from_slice(client_config_bytes)
        .map_err(|e| error(format!("client-visible SLO config: {e}")))?;
    let config = SloEvaluationConfig::from_client_visible(&client).map_err(|e| error(e.0))?;
    let reports: Vec<BenchReport> = match report_bytes.iter().find(|b| !b.is_ascii_whitespace()) {
        Some(b'[') => serde_json::from_slice(report_bytes),
        _ => serde_json::from_slice(report_bytes).map(|report| vec![report]),
    }
    .map_err(|e| error(format!("benchmark report: {e}")))?;
    if reports.is_empty() || reports.len() > MAX_CELLS {
        return Err(error("offline SLO requires 1..=256 cells"));
    }
    let mut offered = 0_u64;
    let mut gaps = 0_usize;
    for report in &reports {
        if report.n_repeats == 0 || report.n_repeats > MAX_REPEATS || report.n_requests_per_run == 0
        {
            return Err(error("invalid or excessive repeat/request count"));
        }
        offered = offered
            .checked_add(u64::from(report.n_repeats) * u64::from(report.n_requests_per_run))
            .ok_or_else(|| error("offered count overflow"))?;
        if offered > MAX_REQUESTS {
            return Err(error("offline SLO request limit exceeded"));
        }
        for row in report.request_records.iter().flatten().flatten() {
            gaps = gaps
                .checked_add(row.timing.as_ref().map_or(0, |t| t.raw_event_gaps_ms.len()))
                .ok_or_else(|| error("gap count overflow"))?;
            if gaps > MAX_GAPS {
                return Err(error("offline SLO gap limit exceeded"));
            }
        }
    }
    let cells = reports
        .iter()
        .enumerate()
        .map(|(index, report)| {
            reconstruct::cell(index, report, &config)
                .map_err(|e| error(format!("cell {index}: {e}")))
        })
        .collect::<Result<_, _>>()?;
    Ok(OfflineSloReport {
        schema_version: 1,
        source: "offline_reconstructed",
        evidence_boundary: "Retained request-local SSE observations only; last-visible = observed first + complete visible gaps. Not an original live sidecar. No recovered arrivals, wall-clock anchors, HTTP status or time promises. Failed admission remains Unknown. Hashes establish source bytes, not measurement authenticity or output quality.",
        threshold_declaration: "posthoc_application; prior declaration not verified; descriptive evaluation is not predeclared absolute-SLO acceptance or statistical comparison proof",
        report_sha256: digest(report_bytes),
        client_config_sha256: digest(client_config_bytes),
        cells,
    })
}
