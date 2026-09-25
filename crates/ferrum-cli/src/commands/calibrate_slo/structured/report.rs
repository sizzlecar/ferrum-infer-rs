use super::*;
use ferrum_engine::continuous_engine::{
    StructuredCalibrationArtifact, StructuredCalibrationProgress, StructuredCapturePhase,
    StructuredPhaseFreezeReceipt,
};
use ferrum_scheduler::implementations::continuous::cost_profile::structured_v9::StructuredProfileExportReceiptV9;
use serde::Serialize;

#[derive(Default, Serialize)]
pub(crate) struct StructuredReport {
    pub schema_version: u32,
    pub scope: &'static str,
    pub progress: Option<StructuredCalibrationProgress>,
    pub freezes: Vec<StructuredPhaseFreezeReceipt>,
    pub source: Option<SourceReceipt>,
    pub exported_profile: Option<StructuredProfileExportReceiptV9>,
    pub finalization_error: Option<String>,
}
impl StructuredReport {
    pub(super) fn new() -> Self {
        Self {
            schema_version: 1,
            scope: "one predeclared structured ordinary-decode domain; complete independent fit/residual/qualification populations; qualification is a position-transfer challenge, not a p99 guarantee or an independent fourth heldout; no serving SLO or full horizon claim",
            ..Self::default()
        }
    }
}

#[derive(Serialize)]
pub(crate) struct SourceReceipt {
    pub source_path: PathBuf,
    pub source_sha256: [u8; 32],
    pub source_bytes: u64,
    pub phase: StructuredCapturePhase,
    pub offered_waves: u64,
    pub scope_members: u64,
    pub scope_failures: u64,
    pub failure: Option<String>,
    pub numerical_model_present: bool,
}
impl From<&StructuredCalibrationArtifact> for SourceReceipt {
    fn from(value: &StructuredCalibrationArtifact) -> Self {
        Self {
            source_path: value.source_path.clone(),
            source_sha256: value.source_sha256,
            source_bytes: value.source_bytes,
            phase: value.phase,
            offered_waves: value.offered_waves,
            scope_members: value.scope_members,
            scope_failures: value.scope_failures,
            failure: value.failure.clone(),
            numerical_model_present: value.model.is_some(),
        }
    }
}

/// A finished source may be an intentionally retained failed capture. Its
/// existence or a footer is never enough to authorize publication. A second
/// independent check in the schema-9 exporter replays the original source.
pub(super) fn require_exportable(source: &SourceReceipt, completed: bool) -> Result<()> {
    if !completed
        || source.phase != StructuredCapturePhase::Qualified
        || !source.numerical_model_present
        || source.scope_failures != 0
        || source.failure.is_some()
    {
        return Err(FerrumError::config(
            "structured calibration did not qualify its complete live population; source retained, no profile exported",
        ));
    }
    Ok(())
}
