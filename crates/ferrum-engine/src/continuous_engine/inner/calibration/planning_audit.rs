//! Explicit bounded structural diagnostics. No action here admits, publishes,
//! executes, settles a wave, creates a source member or advances a clock.
use super::*;
use ferrum_scheduler::implementations::continuous::cost_model::structured_v2::StructuredQueryDemandV2;
use serde::{Deserialize, Serialize};
use std::num::{NonZeroU32, NonZeroU64, NonZeroUsize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RequiredFutureAuditPlanV2 {
    pub paths: Vec<RequiredFutureAuditPathV2>,
    pub limits: RequiredFutureAuditLimitsV2,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RequiredFutureAuditPathV2 {
    pub waves: Vec<Vec<RequiredFutureAuditRowV2>>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RequiredFutureAuditRowV2 {
    /// Index into the complete session-bound frontier table, not a resource ID.
    pub frontier_index: usize,
    pub action: RequiredFutureAuditActionV2,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind", deny_unknown_fields)]
pub enum RequiredFutureAuditActionV2 {
    Decode,
    Prefill { count: NonZeroU32 },
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RequiredFutureAuditLimitsV2 {
    /// Independent diagnostic allowance, never an execution/planning permit.
    /// One deadline includes capture and every path; no renewal between waves.
    pub budget_ms: NonZeroU64,
    pub maximum_queries: NonZeroUsize,
    pub maximum_coordinates: NonZeroUsize,
}
impl RequiredFutureAuditPlanV2 {
    /// Validate pure declaration bounds; this does not validate live owners or
    /// grant any snapshot, model, projection or execution authority.
    pub fn validate(&self, frontiers: usize) -> Result<()> {
        if frontiers == 0
            || frontiers > 256
            || self.paths.is_empty()
            || self.paths.len() > 8
            || self.limits.budget_ms.get() > 30_000
            || self.limits.maximum_queries.get() > 1024
            || self.limits.maximum_coordinates.get() > 65_536
            || self.paths.iter().any(|p| {
                p.waves.is_empty()
                    || p.waves.len() > 16
                    || p.waves.iter().any(|w| {
                        w.is_empty()
                            || w.len() > 256
                            || w.iter().enumerate().any(|(i, row)| {
                                row.frontier_index >= frontiers
                                    || w[..i]
                                        .iter()
                                        .any(|prior| prior.frontier_index == row.frontier_index)
                            })
                    })
            })
        {
            return Err(FerrumError::invalid_request(
                "invalid bounded future-owner audit plan",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RequiredFutureAuditFailureV2 {
    Snapshot {
        reason: &'static str,
        route_reason: Option<String>,
    },
    Planning {
        reason: String,
    },
    Route {
        reason: String,
    },
    UnsupportedHostOrProjection,
    MissingStructuredEvidence,
    Limit {
        resource: &'static str,
    },
}
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RequiredFutureAuditCostV2 {
    KnownAtRead {
        local_now_ns: u64,
        planning_ns: u64,
        valid_for_ns: u64,
    },
    Unknown {
        local_now_ns: Option<u64>,
        reason: String,
    },
    NotQueried {
        reason: &'static str,
    },
}
#[derive(Debug, Clone, Serialize)]
pub struct RequiredFutureAuditQueryV2 {
    pub path_index: usize,
    pub wave_index: usize,
    pub alternative_index: usize,
    pub exact_first_wave: bool,
    pub demand: Option<StructuredQueryDemandV2>,
    pub input_unknown: Option<String>,
    pub cost: RequiredFutureAuditCostV2,
}
#[derive(Debug, Clone, Serialize)]
pub struct RequiredFutureAuditPathReportV2 {
    pub declared_waves: usize,
    pub projected_waves: usize,
    pub stopped: Option<RequiredFutureAuditFailureV2>,
}
#[derive(Debug, Clone, Serialize)]
pub struct RequiredFutureAuditFrontierV2 {
    pub request_id: RequestId,
    pub owner_incarnation: u64,
    pub work_generation: u64,
    pub generated_tokens: usize,
    pub kv_tokens: usize,
    pub prefill_progress: Option<(usize, usize)>,
    pub input: CalibrationRequestEvidence,
}
#[derive(Debug, Clone, Serialize)]
pub struct RequiredFutureAuditReportV2 {
    pub schema_version: u32,
    /// Always structural diagnostics, never a witness or dispatch authorization.
    pub purpose: &'static str,
    pub snapshot_generation: Option<u64>,
    pub observed_at_ns: Option<u64>,
    pub model_version: Option<u64>,
    pub frontier_table: Vec<RequiredFutureAuditFrontierV2>,
    /// Ordered model/numerical/device-runtime/execution-config fingerprint.
    pub execution_fingerprint: Option<[[u8; 32]; 4]>,
    pub paths: Vec<RequiredFutureAuditPathReportV2>,
    pub queries: Vec<RequiredFutureAuditQueryV2>,
    pub retained_coordinates: usize,
    pub unavailable: Option<RequiredFutureAuditFailureV2>,
    pub truncated: bool,
    pub completed_all_declared_paths: bool,
}
impl RequiredFutureAuditReportV2 {
    pub(in crate::continuous_engine::inner) fn new(plan: &RequiredFutureAuditPlanV2) -> Self {
        Self {
            schema_version: 1,
            purpose: "required_structure_only_not_feasibility",
            snapshot_generation: None,
            observed_at_ns: None,
            model_version: None,
            frontier_table: Vec::new(),
            execution_fingerprint: None,
            paths: plan
                .paths
                .iter()
                .map(|p| RequiredFutureAuditPathReportV2 {
                    declared_waves: p.waves.len(),
                    projected_waves: 0,
                    stopped: None,
                })
                .collect(),
            queries: Vec::new(),
            retained_coordinates: 0,
            unavailable: None,
            truncated: false,
            completed_all_declared_paths: false,
        }
    }
}

impl CalibrationSession {
    /// Call before a real cohort wave using the full `frontiers()` table.
    /// A narrow genuine V2 seed plus the ordinary reference/snapshot contract
    /// is currently required; missing prerequisites are explicit in the report.
    /// No submitted-wave/collector state or request/output counter is changed.
    pub async fn audit_required_owners_v2(
        &self,
        frontiers: &[CalibrationFrontier],
        plan: &RequiredFutureAuditPlanV2,
    ) -> Result<RequiredFutureAuditReportV2> {
        plan.validate(frontiers.len())?;
        if self.pending.is_some()
            || self.indeterminate
            || frontiers
                .iter()
                .any(|f| !Arc::ptr_eq(&f.session, &self.identity))
        {
            return Err(FerrumError::invalid_request(
                "future-owner audit requires this session's settled frontier",
            ));
        }
        let inner = &self.engine.inner;
        let _iteration = inner
            .iteration_lock
            .try_lock()
            .map_err(|_| FerrumError::resource_exhausted("calibration iteration is busy"))?;
        if inner.slo_controller.lock().has_pending_calibration_work() {
            return Err(FerrumError::invalid_request(
                "future-owner audit cannot bypass pending physical work",
            ));
        }
        inner.audit_required_owners_v2(frontiers, plan)
    }
}
