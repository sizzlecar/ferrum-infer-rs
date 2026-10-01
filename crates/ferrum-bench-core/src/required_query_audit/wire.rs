use serde::Deserialize;
use serde_json::Value;
use std::collections::BTreeMap;

pub(super) const SCHEMA: &str = "ferrum.planner-required-query-observation.v1";
#[derive(Deserialize)]
pub(super) struct Envelope {
    pub schema: String,
    pub ordinal: u64,
    pub retained_ordinal: u64,
    pub transaction: u64,
    #[serde(flatten)]
    pub event: Event,
}
#[derive(Deserialize)]
#[serde(tag = "event", content = "data", rename_all = "snake_case")]
pub(super) enum Event {
    RunIdentity(Value),
    TransactionBegin,
    Snapshot {
        cost_model_version: u64,
        requests: Vec<Value>,
    },
    ControllerCheckpoint(Checkpoint),
    AttemptBegin {
        attempt: u64,
        phase: String,
        depth: usize,
        requests: Vec<Value>,
        work: Vec<Value>,
    },
    QueryConstructed {
        attempt: u64,
        alternative: usize,
        input_unknown: Option<String>,
        demand_error: Option<String>,
    },
    QueryLookup {
        attempt: u64,
        alternative: usize,
        outcome: LookupOutcome,
    },
    AttemptEnd {
        attempt: u64,
        constructed: usize,
        queried: usize,
        not_queried_start: usize,
        not_queried_end: usize,
        reason: String,
    },
    ReplayBegin {
        replay: u64,
        waves: usize,
    },
    ReplayEnd {
        replay: u64,
        reason: String,
    },
    SelectedReplay {
        replay: u64,
    },
    TransactionEnd(End),
}
#[derive(Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub(super) enum LookupOutcome {
    Known { cost: Cost },
    StructuredUnknown { reason: String },
    ModelUnavailable,
    ClockMappingFailed,
}
#[derive(Deserialize)]
pub(super) struct Cost {
    pub model_version: u64,
    pub typical_ns: u64,
    pub planning_ns: u64,
    pub valid_for_ns: u64,
}
#[derive(Clone, Deserialize)]
pub(super) struct Checkpoint {
    pub stage: String,
    pub edge: String,
    pub elapsed_ns: Option<u64>,
    pub hard_budget_ns: u64,
    pub optional_deadline_elapsed_ns: Option<u64>,
    pub completion_preparation_ns: Option<u64>,
    pub publication_reserve_ns: Option<u64>,
}
#[derive(Deserialize)]
pub(super) struct End {
    pub audit_available: bool,
    pub outcome: String,
    pub decision: String,
    pub reason: String,
    pub planning_wall_ns: u64,
    pub budget_ns: u64,
    pub planner_exhausted: bool,
    pub hard_exhausted: bool,
    pub search: SearchStats,
}
#[derive(Deserialize)]
pub(super) struct SearchStats {
    pub enumeration_attempts: u64,
    pub expanded_candidates: u64,
    pub generated_candidates: u64,
    pub candidate_truncations: u64,
    pub resource_unknown_candidates: u64,
    pub cost_unknown_candidates: u64,
    pub shape_unknown_candidates: u64,
    pub measured_replay_work_ns: u64,
    pub replay_reserve_ns: u64,
    pub replay_reserve_stops: u64,
    pub search_soft_stops: u64,
}
#[derive(Deserialize)]
pub(super) struct Header {
    pub schema: String,
    pub event: String,
    pub run_id: String,
    pub authority: String,
}
#[derive(Deserialize)]
pub(super) struct Footer {
    pub schema: String,
    pub event: String,
    pub recording_complete: bool,
    pub requires_successful_close: bool,
    pub statistics: FooterStats,
}
#[derive(Deserialize)]
pub(super) struct FooterStats {
    pub offered: u64,
    pub accepted: u64,
    pub written: u64,
    pub lost: BTreeMap<String, u64>,
    pub first_lost_ordinal: u64,
    pub last_lost_ordinal: u64,
    pub transactions: u64,
    pub active_transactions: u64,
    pub abandoned_transactions: u64,
    pub counter_exhausted: bool,
    pub writer_failed: bool,
    pub first_error: Option<String>,
}
