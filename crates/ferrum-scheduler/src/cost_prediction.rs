//! Immutable, instance-bound service-cost observations. These values neither
//! authorize execution nor certify client-visible latency percentiles.

use ferrum_execution::execution_attempt::{AttemptPrefixRestore, AttemptWorkKind};
use ferrum_execution::model_executor::PrefillShapePolicy;
use ferrum_execution::token_execution::ExecutionRange;
use ferrum_interfaces::BatchPlan;
use ferrum_types::SchedulerSloConfig;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::watch;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CostWork {
    pub kind: AttemptWorkKind,
    pub requested: ExecutionRange,
    pub context_tokens: usize,
    pub prefix_restore: AttemptPrefixRestore,
    /// Unknown internal reuse is not equivalent to zero reuse.
    pub internally_reused_tokens: Option<usize>,
}

#[derive(Debug, Clone, Copy)]
pub struct CostQuery<'a> {
    pub work: &'a [CostWork],
    pub work_is_exact: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CostUnknownReason {
    NoCoverage,
    Stale,
    InexactWork,
    UnsupportedDomain,
    QueryLimit,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CostPrediction {
    Known {
        /// Includes the estimator's declared engineering safety margin once.
        upper_service_time: Duration,
        samples: usize,
        observed_at: Instant,
        expires_at: Instant,
    },
    Unknown(CostUnknownReason),
}

/// Implementations must perform bounded read-only queries, never fitting,
/// serialization, worker waits, or acquiring an execution/scheduler lock.
pub trait CostSnapshot: Send + Sync {
    fn revision(&self) -> u64;
    fn predict(&self, query: &CostQuery<'_>, now: Instant) -> CostPrediction;
}

/// An assembly instance owns one source. Subscribe before loading; publish a
/// complete immutable snapshot before notifying its revision.
pub trait CostSnapshotSource: Send + Sync {
    fn load(&self) -> Arc<dyn CostSnapshot>;
    fn subscribe(&self) -> watch::Receiver<u64>;
}

/// One Engine planning turn, including all asynchronous admission subrounds.
/// Time is reread after awaits, but the snapshot is never replaced mid-turn.
pub struct TimePlanningRound {
    pub(crate) snapshot: Arc<dyn CostSnapshot>,
    pub(crate) config: SchedulerSloConfig,
    pub(crate) prefill_shape_policy: PrefillShapePolicy,
    pub(crate) retry_at: Option<Instant>,
    pub(crate) time_blocked: bool,
    pub(crate) statistics: PlanningStatistics,
    pub(crate) selected: Vec<CostWork>,
    pub(crate) selected_exact: bool,
    pub(crate) selected_over_limit: bool,
}

impl TimePlanningRound {
    pub fn new(
        snapshot: Arc<dyn CostSnapshot>,
        config: SchedulerSloConfig,
        prefill_shape_policy: PrefillShapePolicy,
    ) -> Self {
        Self {
            snapshot,
            config,
            prefill_shape_policy,
            retry_at: None,
            time_blocked: false,
            statistics: PlanningStatistics::default(),
            selected: Vec::new(),
            selected_exact: true,
            selected_over_limit: false,
        }
    }

    pub fn revision(&self) -> u64 {
        self.snapshot.revision()
    }
    pub fn retry_at(&self) -> Option<Instant> {
        self.retry_at
    }
    pub fn time_blocked(&self) -> bool {
        self.time_blocked
    }
    /// Whether the complete selected batch was represented by exact projected
    /// work. An incomplete or over-limit query cannot authorize range checks.
    pub fn selection_is_exact(&self, selected_count: usize) -> bool {
        selected_count > 0
            && self.selected_exact
            && !self.selected_over_limit
            && self.selected.len() == selected_count
    }
}

#[derive(Debug, Clone, Copy, Default, serde::Serialize)]
pub(crate) struct PlanningStatistics {
    pub known_queries: u64,
    pub unknown_queries: u64,
    pub held_candidates: u64,
    pub rejected_candidates: u64,
    pub shrunk_prefills: u64,
    pub static_fallbacks: u64,
    pub budget_misses: u64,
}

#[derive(Debug)]
pub struct BatchSelection {
    pub batch: Option<BatchPlan>,
    /// Only a future, finite coverage wait; never an allocator epoch.
    pub retry_at: Option<Instant>,
    pub time_blocked: bool,
}
