use super::*;
use ferrum_engine::continuous_engine::{
    StructuredCalibrationArtifactV2, StructuredCalibrationProgress, StructuredCapturePhase,
    StructuredPhaseFreezeReceipt,
};
use ferrum_scheduler::implementations::continuous::{
    cost_model::structured_v2::{StructuredCoverageReportV2, StructuredScopeV2},
    cost_profile::StructuredProfileExportReceiptV10,
};
use serde::Serialize;
#[derive(Serialize)]
pub(crate) struct StructuredReportV2 {
    pub schema_version: u32,
    pub scope_note: &'static str,
    pub declared_scope: StructuredScopeV2,
    pub progress: Option<StructuredCalibrationProgress>,
    pub phase_coverage: Vec<PhaseCoverageV2>,
    pub freezes: Vec<StructuredPhaseFreezeReceipt>,
    pub source: Option<SourceReceiptV2>,
    pub exported_profile: Option<StructuredProfileExportReceiptV10>,
    pub finalization_error: Option<String>,
}
impl StructuredReportV2 {
    pub(super) fn new(scope: StructuredScopeV2) -> Self {
        Self {schema_version:1,scope_note:"one predeclared owner and finite numeric windows; original complete request cohorts/FIFO/three phase clocks; declared/observed/missing pending and Length coverage is an empirical challenge, not p99 or serving SLO proof; no unsupported future owner or joint branch is authorized",
            declared_scope:scope,progress:None,phase_coverage:Vec::new(),freezes:Vec::new(),source:None,exported_profile:None,finalization_error:None}
    }
}
#[derive(Serialize)]
pub(crate) struct PhaseCoverageV2 {
    pub phase: StructuredCapturePhase,
    pub coverage: StructuredCoverageReportV2,
}
#[derive(Serialize)]
pub(crate) struct SourceReceiptV2 {
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
impl From<&StructuredCalibrationArtifactV2> for SourceReceiptV2 {
    fn from(v: &StructuredCalibrationArtifactV2) -> Self {
        Self {
            source_path: v.source_path.clone(),
            source_sha256: v.source_sha256,
            source_bytes: v.source_bytes,
            phase: v.phase,
            offered_waves: v.offered_waves,
            scope_members: v.scope_members,
            scope_failures: v.scope_failures,
            failure: v.failure.clone(),
            numerical_model_present: v.model.is_some(),
        }
    }
}
pub(super) fn require_exportable(source: &SourceReceiptV2, completed: bool) -> Result<()> {
    if !completed
        || source.phase != StructuredCapturePhase::Qualified
        || !source.numerical_model_present
        || source.scope_failures != 0
        || source.failure.is_some()
    {
        return Err(FerrumError::config("V2 complete live population did not qualify; original source retained, no profile exported"));
    }
    Ok(())
}
