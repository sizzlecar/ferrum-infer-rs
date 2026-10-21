//! Immutable, instance-bound service-cost observations. These values neither
//! authorize execution nor certify client-visible latency percentiles.

use ferrum_execution::execution_attempt::{AttemptPrefixRestore, AttemptWorkKind};
use ferrum_execution::model_executor::PrefillShapePolicy;
use ferrum_execution::token_execution::ExecutionRange;
use ferrum_interfaces::request_observation::{
    CompletionForecast, CompletionQuery, RequestStoppingObservation, StoppingForecast,
    StoppingQuery,
};
use ferrum_interfaces::BatchPlan;
use ferrum_types::{CostQueryLimits, RequestId, SchedulerSloConfig};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::watch;

mod unsupported_counts;
pub(crate) use unsupported_counts::JointUnsupportedCounts;

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
    pub execution: CostExecution,
}

impl CostQuery<'_> {
    pub fn projected_bytes(participants: usize) -> Option<usize> {
        participants
            .checked_mul(std::mem::size_of::<CostWork>())?
            .checked_add(std::mem::size_of::<Self>())
    }

    pub fn fits(participants: usize, limits: CostQueryLimits) -> bool {
        participants <= limits.max_participants
            && Self::projected_bytes(participants)
                .is_some_and(|bytes| bytes <= limits.max_query_bytes)
    }
}

/// Execution grouping is declared by Engine from the real invocation path.
/// `SingleInvocation` may contain both phases; it makes no kernel-fusion claim.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CostExecution {
    SingleInvocation,
    /// All prefill work completes before an independent decode invocation.
    SequentialPrefillThenDecode,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CostUnknownReason {
    NoCoverage,
    Stale,
    InexactWork,
    UnsupportedDomain,
    QueryLimit,
    PlanningBudget,
}

/// Interpretation of the risk cost, never a client-visible SLO certificate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CostRisk {
    /// A window maximum and an engineering margin, not a fitted quantile.
    EngineeringMargin,
    CalibratedQuantile {
        quantile: f64,
        approximation: CostRiskApproximation,
    },
    /// One conditional member of a complete empirical scenario distribution;
    /// this value by itself carries no quantile or coverage probability.
    ConditionalScenario { dependence: CostScenarioDependence },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CostRiskApproximation {
    /// Empirical residual calibration within the declared support domain;
    /// finite samples do not establish conditional or multistep coverage.
    EmpiricalConditional,
    /// Sequential group risk was combined under an explicit union bound.
    /// The underlying empirical group estimates remain approximations.
    SequentialUnionBound,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CostSupport {
    ExactObservedWork,
    ValidatedFeatureDomain,
}

/// Cost distributions are empirical sensitivity scenarios, not joint coverage
/// certificates. A shared rank explicitly assumes rank dependence across the
/// invocation groups in one trajectory; it is not an independence assumption.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CostScenarioDependence {
    SharedResidualRank,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CostScenario {
    pub rank: usize,
    pub weight: f64,
    pub service_time: Duration,
}

#[derive(Debug, Clone, PartialEq)]
pub enum CostScenarioPrediction {
    Known {
        central_service_time: Duration,
        /// Complete declared empirical mass, with consecutive ranks starting
        /// at zero. Partial support must return Unknown, not renormalize away
        /// missing tails. Same-rank coupling also requires identical weights.
        scenarios: Vec<CostScenario>,
        dependence: CostScenarioDependence,
        support: CostSupport,
        samples: usize,
        observed_at: Instant,
        expires_at: Instant,
    },
    Unknown(CostUnknownReason),
}

impl CostScenarioPrediction {
    pub fn validated(self, now: Instant, max_scenarios: usize) -> Self {
        let Self::Known {
            central_service_time,
            ref scenarios,
            samples,
            observed_at,
            expires_at,
            ..
        } = self
        else {
            return self;
        };
        if expires_at <= now {
            return Self::Unknown(CostUnknownReason::Stale);
        }
        if scenarios.len() > max_scenarios {
            return Self::Unknown(CostUnknownReason::QueryLimit);
        }
        if central_service_time.is_zero()
            || samples == 0
            || observed_at > now
            || expires_at <= observed_at
            || scenarios.is_empty()
            || scenarios.iter().enumerate().any(|(rank, scenario)| {
                scenario.rank != rank
                    || scenario.service_time.is_zero()
                    || !scenario.weight.is_finite()
                    || scenario.weight <= 0.0
                    || scenario.weight > 1.0
            })
            || (scenarios
                .iter()
                .map(|scenario| scenario.weight)
                .sum::<f64>()
                - 1.0)
                .abs()
                > 1e-9
        {
            return Self::Unknown(CostUnknownReason::UnsupportedDomain);
        }
        self
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CostPrediction {
    Known {
        /// Expected/central service cost used for action benefit estimates;
        /// never substitute a tail estimate for this value.
        central_service_time: Duration,
        /// Final risk cost after exactly one declared calibration/margin.
        upper_service_time: Duration,
        risk: CostRisk,
        support: CostSupport,
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
    fn query_limits(&self) -> CostQueryLimits;
    fn predict(&self, query: &CostQuery<'_>, now: Instant) -> CostPrediction;
    /// A bounded read-only empirical distribution. The caller supplies its
    /// resource limit before allocation. A snapshot cannot synthesize missing
    /// tail support or use these rank differences as paired benefit evidence.
    fn cost_scenarios(
        &self,
        _query: &CostQuery<'_>,
        _now: Instant,
        _max_scenarios: usize,
    ) -> CostScenarioPrediction {
        CostScenarioPrediction::Unknown(CostUnknownReason::NoCoverage)
    }
    fn stopping_forecast(&self, _query: &StoppingQuery<'_>) -> StoppingForecast {
        StoppingForecast::unknown()
    }
    fn completion_forecast(&self, _query: &CompletionQuery<'_>) -> CompletionForecast {
        CompletionForecast::unknown()
    }
}

/// An assembly instance owns one source. Subscribe before loading; publish a
/// complete immutable snapshot before notifying its revision.
pub trait CostSnapshotSource: Send + Sync {
    fn load(&self) -> Arc<dyn CostSnapshot>;
    fn subscribe(&self) -> watch::Receiver<u64>;
}

/// Historical engine output-channel facts. Successful nonempty text enqueue
/// is a delivery proxy before HTTP/protocol projection, not client SSE timing.
/// First/last text offsets and age are measured from original ingress.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RequestClockFacts {
    pub captured_at: Instant,
    pub age: Duration,
    pub first_text: Option<Duration>,
    pub last_text: Option<Duration>,
    pub max_text_interval: Option<Duration>,
    pub nonempty_text_events: u64,
    pub violating_text_intervals: u64,
    pub itl_threshold: Option<Duration>,
    pub generated_tokens: usize,
    pub streaming: bool,
}

impl RequestClockFacts {
    fn is_consistent(self) -> bool {
        match (self.nonempty_text_events, self.first_text, self.last_text) {
            (0, None, None) => {
                self.max_text_interval.is_none() && self.violating_text_intervals == 0
            }
            (count, Some(first), Some(last))
                if self.streaming && first <= last && last <= self.age =>
            {
                (count == 1
                    && first == last
                    && self.max_text_interval.is_none()
                    && self.violating_text_intervals == 0)
                    || (count > 1
                        && self
                            .max_text_interval
                            .is_some_and(|interval| interval <= last - first)
                        && self.violating_text_intervals < count
                        && (self.itl_threshold.is_some() || self.violating_text_intervals == 0))
            }
            _ => false,
        }
    }
}

/// Read-only paged KV owner snapshot. This grants no reservations or reuse
/// credit; hypothetical plans still require actual runtime resource admission.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlanningCapacity {
    pub block_size: usize,
    pub total_blocks: usize,
    pub free_blocks: usize,
    /// Whether this same snapshot covers other persistent admission demand.
    pub admission_coverage: ferrum_execution::model_executor::PagedAdmissionCoverage,
}

/// One Engine planning turn, including all asynchronous admission subrounds.
/// Time is reread after awaits, but the snapshot is never replaced mid-turn.
pub struct TimePlanningRound {
    pub(crate) snapshot: Arc<dyn CostSnapshot>,
    pub(crate) config: SchedulerSloConfig,
    pub(crate) prefill_shape_policy: PrefillShapePolicy,
    pub(crate) execution: CostExecution,
    pub(crate) query_limits: CostQueryLimits,
    pub(crate) retry_at: Option<Instant>,
    pub(crate) time_blocked: bool,
    pub(crate) statistics: PlanningStatistics,
    pub(crate) selected: Vec<CostWork>,
    pub(crate) selected_exact: bool,
    pub(crate) execution_already_projected: bool,
    pub(crate) selected_over_limit: bool,
    pub(crate) query_count: usize,
    pub(crate) prediction_elapsed: Duration,
    request_facts: Option<Vec<(RequestId, RequestStoppingObservation)>>,
    pub(crate) request_fact_bytes: usize,
    pub(crate) admission_gate:
        Option<crate::implementations::continuous::admission_gate::FrozenAdmissionGate>,
    /// Demand frozen with the latest capacity observation; promotions do not
    /// replenish reusable sequence metadata within that observation.
    pub(crate) capacity_initial_new_sequences: Option<usize>,
    request_clocks: Option<Vec<(RequestId, RequestClockFacts)>>,
    capacity: Option<PlanningCapacity>,
    resident_capacity: Option<ferrum_execution::model_executor::ResidentCohortCapacitySnapshot>,
}

impl TimePlanningRound {
    pub fn new(
        snapshot: Arc<dyn CostSnapshot>,
        config: SchedulerSloConfig,
        prefill_shape_policy: PrefillShapePolicy,
        execution: CostExecution,
    ) -> Self {
        let supported =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| snapshot.query_limits()))
                .unwrap_or(CostQueryLimits {
                    max_participants: 0,
                    max_query_bytes: 0,
                });
        Self {
            snapshot,
            config,
            prefill_shape_policy,
            execution,
            query_limits: config.query.intersect(supported),
            retry_at: None,
            time_blocked: false,
            statistics: PlanningStatistics::default(),
            selected: Vec::new(),
            selected_exact: true,
            execution_already_projected: true,
            selected_over_limit: false,
            query_count: 0,
            prediction_elapsed: Duration::ZERO,
            request_facts: None,
            request_fact_bytes: 0,
            admission_gate: None,
            capacity_initial_new_sequences: None,
            request_clocks: None,
            capacity: None,
            resident_capacity: None,
        }
    }

    pub fn revision(&self) -> u64 {
        self.snapshot.revision()
    }

    pub fn max_request_participants(&self) -> usize {
        self.config.planning.max_participants
    }

    /// Include owner fact/resource projection in the same comparison budget.
    pub fn charge_projection_time(&mut self, elapsed: Duration) {
        self.prediction_elapsed = self.prediction_elapsed.saturating_add(elapsed);
    }

    /// Invalidate a previous cohort before attempting a fresh bounded capture.
    /// A missing member or failed projection must not leave old facts usable.
    /// The immutable predictor and the round's consumed budgets are unchanged.
    pub fn clear_request_facts(&mut self) {
        self.request_facts = None;
        self.admission_gate = None;
        self.request_clocks = None;
        self.request_fact_bytes = 0;
        self.resident_capacity = None;
    }
    /// Freeze the complete request cohort. Missing, duplicate or over-budget
    /// members invalidate the whole projection rather than authorizing a prefix.
    pub fn set_request_facts(
        &mut self,
        facts: impl IntoIterator<Item = (RequestId, RequestStoppingObservation)>,
    ) -> bool {
        self.clear_request_facts();
        let mut retained = Vec::new();
        let mut bytes = 0usize;
        for (request_id, fact) in facts {
            if retained.len() >= self.config.planning.max_participants
                || !fact.is_consistent()
                || retained
                    .iter()
                    .any(|(id, other): &(RequestId, RequestStoppingObservation)| {
                        *id == request_id || other.lifecycle == fact.lifecycle
                    })
            {
                return false;
            }
            let Some(next_bytes) = bytes
                .checked_add(std::mem::size_of::<(RequestId, RequestStoppingObservation)>())
                .and_then(|size| size.checked_add(fact.rules.identity().len()))
                .filter(|size| *size <= self.config.planning.max_request_fact_bytes)
            else {
                return false;
            };
            if retained.try_reserve_exact(1).is_err() {
                return false;
            }
            bytes = next_bytes;
            retained.push((request_id, fact));
        }
        self.request_fact_bytes = bytes;
        self.request_facts = Some(retained);
        true
    }

    pub fn request_facts(&self) -> Option<&[(RequestId, RequestStoppingObservation)]> {
        self.request_facts.as_deref()
    }

    /// Clock members must match the already-frozen stopping cohort exactly.
    /// Both projections share one retained-byte budget; incomplete projection
    /// cannot authorize a subset with favorable historical latency.
    pub fn set_request_clocks(
        &mut self,
        clocks: impl IntoIterator<Item = (RequestId, RequestClockFacts)>,
    ) -> bool {
        self.admission_gate = None;
        self.request_clocks = None;
        let Some(facts) = &self.request_facts else {
            return false;
        };
        let mut retained = Vec::new();
        let mut bytes = self.request_fact_bytes;
        for (id, clock) in clocks {
            if retained.len() >= facts.len()
                || !clock.is_consistent()
                || !facts.iter().any(|(request_id, _)| *request_id == id)
                || retained.iter().any(|(request_id, _)| *request_id == id)
            {
                return false;
            }
            let Some(next_bytes) = bytes
                .checked_add(std::mem::size_of::<(RequestId, RequestClockFacts)>())
                .filter(|size| *size <= self.config.planning.max_request_fact_bytes)
            else {
                return false;
            };
            if retained.try_reserve_exact(1).is_err() {
                return false;
            }
            bytes = next_bytes;
            retained.push((id, clock));
        }
        if retained.len() != facts.len() {
            return false;
        }
        self.request_clocks = Some(retained);
        true
    }

    pub fn request_clocks(&self) -> Option<&[(RequestId, RequestClockFacts)]> {
        self.request_clocks.as_deref()
    }

    pub fn set_capacity(&mut self, capacity: Option<PlanningCapacity>) {
        self.admission_gate = None;
        self.capacity_initial_new_sequences = None;
        self.resident_capacity = None;
        self.capacity = capacity.filter(|value| {
            value.block_size > 0
                && value.total_blocks > 0
                && value.free_blocks <= value.total_blocks
        });
    }

    pub fn capacity(&self) -> Option<PlanningCapacity> {
        self.capacity
    }

    /// Bind an executor's complete resident-cohort proof to these exact facts.
    /// This cannot authorize new admission and is invalidated on recapture.
    pub fn set_resident_capacity(
        &mut self,
        capacity: Option<ferrum_execution::model_executor::ResidentCohortCapacitySnapshot>,
    ) {
        self.admission_gate = None;
        self.resident_capacity = capacity.filter(|proof| {
            self.request_facts.as_ref().is_some_and(|facts| {
                !facts.is_empty()
                    && proof.participants == facts.len()
                    && proof.max_batch_sequences >= facts.len()
            })
        });
    }

    /// Current complete-cohort observation, invalidated by fact/resource refresh.
    pub fn resident_capacity(
        &self,
    ) -> Option<ferrum_execution::model_executor::ResidentCohortCapacitySnapshot> {
        self.resident_capacity
    }

    pub fn stopping_forecast(&mut self, query: &StoppingQuery<'_>) -> StoppingForecast {
        if self.query_count >= self.config.planning.max_queries
            || self.prediction_elapsed >= Duration::from_micros(self.config.planning.max_elapsed_us)
        {
            return StoppingForecast::unknown();
        }
        self.query_count += 1;
        let started = Instant::now();
        let forecast = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.snapshot.stopping_forecast(query)
        }))
        .unwrap_or_else(|_| StoppingForecast::unknown());
        self.prediction_elapsed += started.elapsed();
        if self.prediction_elapsed > Duration::from_micros(self.config.planning.max_elapsed_us) {
            StoppingForecast::unknown()
        } else {
            forecast
        }
    }
    pub fn completion_forecast(&mut self, query: &CompletionQuery<'_>) -> CompletionForecast {
        if self.query_count >= self.config.planning.max_queries
            || self.prediction_elapsed >= Duration::from_micros(self.config.planning.max_elapsed_us)
        {
            return CompletionForecast::unknown();
        }
        self.query_count += 1;
        let started = Instant::now();
        let forecast = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.snapshot.completion_forecast(query)
        }))
        .unwrap_or_else(|_| CompletionForecast::unknown());
        self.prediction_elapsed += started.elapsed();
        if self.prediction_elapsed > Duration::from_micros(self.config.planning.max_elapsed_us) {
            CompletionForecast::unknown()
        } else {
            forecast
        }
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
        self.execution_already_projected && self.cost_selection_is_exact(selected_count)
    }

    pub(crate) fn cost_selection_is_exact(&self, selected_count: usize) -> bool {
        selected_count > 0
            && self.selected_exact
            && !self.selected_over_limit
            && self.selected.len() == selected_count
    }
}

#[derive(Debug, Clone, Copy, Default, serde::Serialize)]
pub(crate) struct PlanningStatistics {
    /// Actual immutable cost-distribution calls in completed comparisons.
    pub joint_cost_snapshot_queries: u64,
    /// Reuses of exact cost distributions, distinct from model calls.
    pub joint_cost_cache_hits: u64,
    /// Maximum retained cache bytes of one completed comparison.
    pub joint_cost_cache_peak_bytes: u64,
    pub joint_comparisons: u64,
    pub joint_interventions: u64,
    pub joint_unknown_resources: u64,
    pub joint_unresolved: u64,
    /// Returned comparison failures only; not inferred causes or request counts.
    pub joint_unsupported: JointUnsupportedCounts,
    /// A supported alternative could not be applied by the admission gate.
    pub joint_admission_action_required: u64,
    pub joint_known_infeasible: u64,
    pub known_queries: u64,
    pub unknown_queries: u64,
    pub held_candidates: u64,
    pub rejected_candidates: u64,
    pub shrunk_prefills: u64,
    pub static_fallbacks: u64,
    pub budget_misses: u64,
}

impl PlanningStatistics {
    pub(crate) fn record_joint_unsupported(&mut self, reason: crate::joint_policy::Unsupported) {
        self.joint_unsupported.record(reason);
        let aggregate = if reason.is_resource_unknown() {
            &mut self.joint_unknown_resources
        } else {
            &mut self.joint_unresolved
        };
        *aggregate = aggregate.saturating_add(1);
    }
}

#[derive(Debug)]
pub struct BatchSelection {
    pub batch: Option<BatchPlan>,
    /// Only a future, finite coverage wait; never an allocator epoch.
    pub retry_at: Option<Instant>,
    pub time_blocked: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cost_scenarios_require_complete_bounded_rank_mass_and_current_support() {
        let now = Instant::now();
        let prediction = CostScenarioPrediction::Known {
            central_service_time: Duration::from_millis(10),
            scenarios: vec![
                CostScenario {
                    rank: 0,
                    weight: 0.5,
                    service_time: Duration::from_millis(8),
                },
                CostScenario {
                    rank: 1,
                    weight: 0.5,
                    service_time: Duration::from_millis(12),
                },
            ],
            dependence: CostScenarioDependence::SharedResidualRank,
            support: CostSupport::ValidatedFeatureDomain,
            samples: 20,
            observed_at: now,
            expires_at: now + Duration::from_secs(1),
        };
        assert_eq!(prediction.clone().validated(now, 2), prediction);
        assert_eq!(
            prediction.clone().validated(now, 1),
            CostScenarioPrediction::Unknown(CostUnknownReason::QueryLimit)
        );
        assert_eq!(
            prediction
                .clone()
                .validated(now + Duration::from_secs(1), 2),
            CostScenarioPrediction::Unknown(CostUnknownReason::Stale)
        );
        for invalid_rank in [false, true] {
            let mut invalid = prediction.clone();
            if let CostScenarioPrediction::Known { scenarios, .. } = &mut invalid {
                if invalid_rank {
                    scenarios[1].rank = 0;
                } else {
                    scenarios.pop();
                }
            }
            assert_eq!(
                invalid.validated(now, 2),
                CostScenarioPrediction::Unknown(CostUnknownReason::UnsupportedDomain)
            );
        }
    }

    struct UnknownSnapshot;
    impl CostSnapshot for UnknownSnapshot {
        fn revision(&self) -> u64 {
            7
        }
        fn query_limits(&self) -> CostQueryLimits {
            CostQueryLimits::default()
        }
        fn predict(&self, _: &CostQuery<'_>, _: Instant) -> CostPrediction {
            CostPrediction::Unknown(CostUnknownReason::NoCoverage)
        }
    }
    fn request_fact(admission: u64) -> (RequestId, RequestStoppingObservation) {
        use ferrum_interfaces::request_observation::*;
        (
            RequestId::new(),
            RequestStoppingObservation {
                admitted_at: Instant::now(),
                lifecycle: RequestLifecycleId {
                    producer: 1,
                    admission,
                },
                revision: 1,
                observed_at: Instant::now(),
                rules: Arc::new(
                    EffectiveStoppingRules::new(b"eos".to_vec(), StoppingMechanism::Observed, 64)
                        .unwrap(),
                ),
                cap: 100,
                generated_tokens: 0,
                verified_survival_through: 0,
                terminal: None,
                completion: None,
            },
        )
    }

    #[test]
    fn planning_request_facts_reject_whole_over_budget_or_duplicate_cohorts() {
        let mut config: SchedulerSloConfig = "ttft:1000,tpot:100,itl:100".parse().unwrap();
        config.planning.max_participants = 1;
        let mut round = TimePlanningRound::new(
            Arc::new(UnknownSnapshot),
            config,
            PrefillShapePolicy::Identity,
            CostExecution::SingleInvocation,
        );
        assert!(round.request_facts().is_none());
        assert!(round.set_request_facts([request_fact(1)]));
        assert_eq!(round.request_facts().unwrap().len(), 1);
        assert!(!round.set_request_facts([request_fact(1), request_fact(2)]));
        assert!(round.request_facts().is_none());
        round.config.planning.max_participants = 2;
        let same = request_fact(3);
        assert!(!round.set_request_facts([same.clone(), same]));
        assert!(round.request_facts().is_none());
        round.config.planning.max_request_fact_bytes =
            std::mem::size_of::<(RequestId, RequestStoppingObservation)>();
        assert!(!round.set_request_facts([request_fact(4)]));
        assert!(
            round.request_facts().is_none(),
            "full rule identity bytes must fit independently"
        );
    }

    #[test]
    fn neutral_query_extent_bounds_its_own_representation() {
        let limits = CostQueryLimits {
            max_participants: 2,
            max_query_bytes: CostQuery::projected_bytes(2).unwrap(),
        };
        assert!(CostQuery::fits(2, limits));
        assert!(!CostQuery::fits(3, limits));
        assert!(!CostQuery::fits(
            2,
            CostQueryLimits {
                max_query_bytes: limits.max_query_bytes - 1,
                ..limits
            }
        ));
        assert!(CostQuery::projected_bytes(usize::MAX).is_none());
        let supported = limits.intersect(CostQueryLimits {
            max_participants: 1,
            max_query_bytes: limits.max_query_bytes + 1,
        });
        assert_eq!(supported.max_participants, 1);
        assert_eq!(supported.max_query_bytes, limits.max_query_bytes);
    }

    #[test]
    fn clock_projection_requires_complete_membership_and_shared_byte_budget() {
        let config: SchedulerSloConfig = "ttft:1000,tpot:100,itl:100".parse().unwrap();
        let mut round = TimePlanningRound::new(
            Arc::new(UnknownSnapshot),
            config,
            PrefillShapePolicy::Identity,
            CostExecution::SingleInvocation,
        );
        let fact = request_fact(1);
        let id = fact.0.clone();
        let clock = RequestClockFacts {
            captured_at: Instant::now(),
            age: Duration::from_millis(20),
            first_text: Some(Duration::from_millis(5)),
            last_text: Some(Duration::from_millis(15)),
            max_text_interval: Some(Duration::from_millis(10)),
            nonempty_text_events: 2,
            violating_text_intervals: 1,
            itl_threshold: Some(Duration::from_millis(9)),
            generated_tokens: 3,
            streaming: true,
        };
        assert!(!round.set_request_clocks([(id.clone(), clock)]));
        assert!(round.set_request_facts([fact.clone()]));
        assert!(round.set_request_clocks([(id.clone(), clock)]));
        assert_eq!(
            round.request_clocks().unwrap()[0]
                .1
                .violating_text_intervals,
            1
        );
        assert!(!round.set_request_clocks([]));
        assert!(round.request_clocks().is_none());
        assert!(!round.set_request_clocks([(RequestId::new(), clock)]));
        assert!(!round.set_request_clocks([(id.clone(), clock), (id.clone(), clock)]));
        let bytes =
            round.request_fact_bytes + std::mem::size_of::<(RequestId, RequestClockFacts)>();
        round.config.planning.max_request_fact_bytes = bytes - 1;
        assert!(!round.set_request_clocks([(id.clone(), clock)]));
        round.config.planning.max_request_fact_bytes = bytes;
        assert!(round.set_request_clocks([(id, clock)]));
        assert!(round.set_request_facts([fact]));
        assert!(
            round.request_clocks().is_none(),
            "a changed stopping cohort invalidates old clocks"
        );
        for capacity in [
            PlanningCapacity {
                block_size: 0,
                total_blocks: 2,
                free_blocks: 1,
                admission_coverage:
                    ferrum_execution::model_executor::PagedAdmissionCoverage::Unknown,
            },
            PlanningCapacity {
                block_size: 16,
                total_blocks: 2,
                free_blocks: 3,
                admission_coverage:
                    ferrum_execution::model_executor::PagedAdmissionCoverage::Unknown,
            },
        ] {
            round.set_capacity(Some(capacity));
            assert!(round.capacity().is_none());
        }
        let capacity = PlanningCapacity {
            block_size: 16,
            total_blocks: 2,
            free_blocks: 1,
            admission_coverage: ferrum_execution::model_executor::PagedAdmissionCoverage::Unknown,
        };
        round.set_capacity(Some(capacity));
        assert_eq!(round.capacity(), Some(capacity));
    }
}
