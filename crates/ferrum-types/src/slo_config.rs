//! Typed bounds and fallback behavior for optional execution-cost feedback.

use serde::{Deserialize, Serialize};

/// Resource limits for one read-only cost query, independent of observation
/// summary storage. Byte accounting is defined by the neutral query contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct CostQueryLimits {
    pub max_participants: usize,
    pub max_query_bytes: usize,
}

impl Default for CostQueryLimits {
    fn default() -> Self {
        Self {
            max_participants: 256,
            max_query_bytes: 128 * 1024,
        }
    }
}

impl CostQueryLimits {
    pub fn validate(self) -> Result<(), String> {
        if self.max_participants == 0 || self.max_query_bytes == 0 {
            return Err("SLO query participant and byte limits must be positive".into());
        }
        Ok(())
    }

    /// The caller and immutable snapshot must both support the query.
    pub fn intersect(self, other: Self) -> Self {
        Self {
            max_participants: self.max_participants.min(other.max_participants),
            max_query_bytes: self.max_query_bytes.min(other.max_query_bytes),
        }
    }
}

/// Whole-round planning resource guards. These defaults bound host work; they
/// are not latency SLOs, model quality thresholds, or performance acceptance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SloPlanningBudget {
    /// Optional immutable cost-distribution memoization. Zero disables it.
    pub max_cost_cache_bytes: usize,
    pub max_queries: usize,
    /// Maximum empirical cost ranks retained per query; a resource guard,
    /// never a statistical precision or performance acceptance requirement.
    pub max_scenarios: usize,
    pub max_candidates: usize,
    pub max_participants: usize,
    /// Cumulative synchronous prediction time; asynchronous admission waits do
    /// not consume this budget. A query already running cannot be preempted.
    pub max_elapsed_us: u64,
    /// Conservative retained request facts, including complete rule identities.
    /// Separate from execution summaries and single cost-query projection.
    pub max_request_fact_bytes: usize,
}

impl Default for SloPlanningBudget {
    fn default() -> Self {
        Self {
            max_cost_cache_bytes: 2 * 1024 * 1024,
            max_queries: 4096,
            max_scenarios: 32,
            max_candidates: 256,
            max_participants: 256,
            max_elapsed_us: 10_000,
            max_request_fact_bytes: 1024 * 1024,
        }
    }
}

impl SloPlanningBudget {
    pub fn validate(self) -> Result<(), String> {
        if self.max_queries == 0
            || self.max_scenarios == 0
            || self.max_candidates == 0
            || self.max_participants == 0
            || self.max_elapsed_us == 0
            || self.max_request_fact_bytes == 0
        {
            return Err("SLO planning resource limits must be positive".into());
        }
        Ok(())
    }
}

/// Joint request-level objectives alongside the three millisecond budgets.
/// These are configurable service targets, separate from the cost model's
/// calibration quantile and from any benchmark acceptance tolerances.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct JointSloTargets {
    pub ttft_quantile: f64,
    pub tpot_quantile: f64,
    /// Visible non-empty SSE text-event intervals, not device token timings.
    pub itl_quantile: f64,
    /// All arrivals remain in the denominator, including rejections/errors.
    pub min_completion_rate: f64,
    /// Measured from original ingress; replanning cannot renew this deadline.
    pub request_deadline_ms: u64,
    /// Minimum arrival-cohort age for empirical product completion evidence.
    /// This sampling maturity does not replace the request deadline.
    pub completion_cohort_maturity_ms: u64,
    /// Known infeasibility is distinct from missing cost/stopping coverage.
    pub known_infeasible_admission: KnownInfeasibleAdmissionPolicy,
}

impl Default for JointSloTargets {
    fn default() -> Self {
        Self {
            ttft_quantile: 0.95,
            tpot_quantile: 0.95,
            itl_quantile: 0.95,
            min_completion_rate: 1.0,
            request_deadline_ms: 120_000,
            completion_cohort_maturity_ms: 30_000,
            known_infeasible_admission: KnownInfeasibleAdmissionPolicy::Reject,
        }
    }
}

impl JointSloTargets {
    pub fn validate(self) -> Result<(), String> {
        for (name, value) in [
            ("ttft_quantile", self.ttft_quantile),
            ("tpot_quantile", self.tpot_quantile),
            ("itl_quantile", self.itl_quantile),
            ("min_completion_rate", self.min_completion_rate),
        ] {
            if !value.is_finite() || value <= 0.0 || value > 1.0 {
                return Err(format!("SLO {name} must be finite and in (0, 1]"));
            }
        }
        if self.request_deadline_ms == 0 {
            return Err("SLO request_deadline_ms must be positive".into());
        }
        if self.completion_cohort_maturity_ms == 0 {
            return Err("SLO completion_cohort_maturity_ms must be positive".into());
        }
        self.known_infeasible_admission.validate()?;
        Ok(())
    }
}

/// Applies only before a new request receives execution admission. Already
/// admitted work continues; rejecting it would not repair an observed SLO miss.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case", deny_unknown_fields)]
pub enum KnownInfeasibleAdmissionPolicy {
    #[default]
    Reject,
    /// Wait from original ingress, then reject. Replanning never renews it.
    Wait { max_wait_ms: u64 },
}

impl KnownInfeasibleAdmissionPolicy {
    pub fn validate(self) -> Result<(), String> {
        if matches!(self, Self::Wait { max_wait_ms: 0 }) {
            return Err("known infeasible admission max_wait_ms must be positive".into());
        }
        Ok(())
    }
}

/// Product lifecycle mode. An omitted product SLO config is also off.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SloMode {
    Off,
    Observe,
    /// Preserve the historical behavior of an explicitly enabled SLO config.
    #[default]
    LearnAndControl,
}

/// Concrete implementation selection belongs to product assembly, never the
/// execution observer or scheduler query contract.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PredictorFamily {
    #[default]
    Auto,
    Physical,
    QuantileGbdt,
    ReferenceDiagnostic,
}

/// Policy comparison is independent of the selected prediction model family.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SloStrategy {
    #[default]
    Joint,
    ReferenceStepDiagnostic,
}

/// Label-independent selection of actual execution observations for fitting.
/// Validation/calibration remain the fixed most-recent chronological suffix.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case", deny_unknown_fields)]
pub enum TrainingHistoryPolicy {
    #[default]
    Recent,
    WidthStratified {
        samples_per_width: usize,
    },
}

impl TrainingHistoryPolicy {
    pub fn validate(self) -> Result<(), String> {
        match self {
            Self::Recent => Ok(()),
            Self::WidthStratified {
                samples_per_width: 1..=2,
            } => Ok(()),
            Self::WidthStratified { .. } => {
                Err("SLO width history requires 1 or 2 samples per width".into())
            }
        }
    }
}

/// Product configuration retains the existing flat `[scheduler.slo]` wire
/// shape. Only `policy` is projected into the engine/scheduler configuration.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SloConfig {
    pub policy: crate::SchedulerSloConfig,
    pub mode: SloMode,
    pub predictor: PredictorFamily,
    pub training_history: TrainingHistoryPolicy,
    pub reference: ReferenceEstimatorConfig,
    // Legacy summary bounds are converted by assembly, which knows the
    // execution-summary layout. They never become neutral query byte bounds.
    legacy_query_observer_limits: Option<(usize, usize)>,
}

impl SloConfig {
    pub fn new(policy: crate::SchedulerSloConfig) -> Self {
        Self {
            policy,
            mode: SloMode::LearnAndControl,
            predictor: PredictorFamily::Auto,
            training_history: TrainingHistoryPolicy::Recent,
            reference: ReferenceEstimatorConfig::default(),
            legacy_query_observer_limits: None,
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        self.policy.validate()?;
        self.training_history.validate()?;
        if self.predictor == PredictorFamily::ReferenceDiagnostic
            && self.training_history != TrainingHistoryPolicy::Recent
        {
            return Err("SLO width history requires a learned predictor".into());
        }
        self.reference.validate()
    }

    pub fn scheduler_policy(&self) -> Option<crate::SchedulerSloConfig> {
        (self.mode == SloMode::LearnAndControl).then_some(self.policy)
    }

    /// A legacy config without explicit `query` used observation projection
    /// bounds for queries. Assembly must convert these to a participant bound
    /// using the actual execution-summary representation before enabling it.
    pub fn legacy_query_observer_limits(&self) -> Option<(usize, usize)> {
        self.legacy_query_observer_limits
    }

    /// Store assembly's representation-aware migration so subsequent config
    /// serialization contains neutral limits, with no legacy coupling.
    pub fn resolve_query_limits(&mut self, limits: CostQueryLimits) {
        self.policy.query = limits;
        self.legacy_query_observer_limits = None;
    }
}

impl Serialize for SloConfig {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let migrated = self.legacy_query_observer_limits.is_none();
        let mut state = serializer.serialize_struct("SloConfig", 12 + usize::from(migrated))?;
        state.serialize_field("ttft_ms", &self.policy.ttft_ms)?;
        state.serialize_field("tpot_ms", &self.policy.tpot_ms)?;
        state.serialize_field("itl_ms", &self.policy.itl_ms)?;
        if migrated {
            state.serialize_field("query", &self.policy.query)?;
        }
        state.serialize_field("planning", &self.policy.planning)?;
        state.serialize_field("joint", &self.policy.joint)?;
        state.serialize_field("strategy", &self.policy.strategy)?;
        state.serialize_field("unknown_cost", &self.policy.unknown_cost)?;
        state.serialize_field("unknown_stopping", &self.policy.unknown_stopping)?;
        state.serialize_field("mode", &self.mode)?;
        state.serialize_field("predictor", &self.predictor)?;
        state.serialize_field("training_history", &self.training_history)?;
        state.serialize_field("reference", &self.reference)?;
        state.end()
    }
}

impl std::str::FromStr for SloConfig {
    type Err = String;

    fn from_str(raw: &str) -> Result<Self, Self::Err> {
        raw.parse().map(Self::new)
    }
}

impl<'de> Deserialize<'de> for SloConfig {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Wire {
            ttft_ms: f64,
            tpot_ms: f64,
            itl_ms: f64,
            #[serde(default)]
            query: Option<CostQueryLimits>,
            #[serde(default)]
            planning: SloPlanningBudget,
            #[serde(default)]
            joint: JointSloTargets,
            #[serde(default)]
            strategy: Option<SloStrategy>,
            #[serde(default)]
            unknown_cost: UnknownCostPolicy,
            #[serde(default)]
            unknown_stopping: UnknownCostPolicy,
            #[serde(default)]
            mode: SloMode,
            #[serde(default)]
            predictor: Option<PredictorFamily>,
            #[serde(default)]
            training_history: TrainingHistoryPolicy,
            #[serde(default)]
            reference: Option<ReferenceEstimatorConfig>,
        }
        let wire = Wire::deserialize(deserializer)?;
        let reference = wire.reference.unwrap_or_default();
        let legacy_query_observer_limits = (wire.reference.is_some() && wire.query.is_none())
            .then_some((reference.max_participants, reference.max_summary_bytes));
        let config = Self {
            policy: crate::SchedulerSloConfig {
                ttft_ms: wire.ttft_ms,
                tpot_ms: wire.tpot_ms,
                itl_ms: wire.itl_ms,
                query: wire.query.unwrap_or_default(),
                planning: wire.planning,
                joint: wire.joint,
                strategy: wire.strategy.unwrap_or(if wire.reference.is_some() {
                    SloStrategy::ReferenceStepDiagnostic
                } else {
                    SloStrategy::Joint
                }),
                unknown_cost: wire.unknown_cost,
                unknown_stopping: wire.unknown_stopping,
            },
            mode: wire.mode,
            predictor: wire.predictor.unwrap_or(if wire.reference.is_some() {
                PredictorFamily::ReferenceDiagnostic
            } else {
                PredictorFamily::Auto
            }),
            training_history: wire.training_history,
            reference,
            legacy_query_observer_limits,
        };
        config.validate().map_err(serde::de::Error::custom)?;
        Ok(config)
    }
}

/// Bounds for one instance's reference estimator, not a device or client SLO.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ReferenceEstimatorConfig {
    pub queue_capacity: usize,
    pub max_queue_bytes: usize,
    pub max_participants: usize,
    pub max_summary_bytes: usize,
    pub max_domains: usize,
    pub window_samples: usize,
    pub min_samples: usize,
    pub snapshot_ttl_ms: u64,
    /// Fraction added to the observed window maximum. This is an engineering
    /// margin, not a statistical confidence or latency percentile guarantee.
    pub safety_margin: f64,
}

impl Default for ReferenceEstimatorConfig {
    fn default() -> Self {
        Self {
            queue_capacity: 256,
            max_queue_bytes: 4 * 1024 * 1024,
            max_participants: 256,
            max_summary_bytes: 128 * 1024,
            max_domains: 256,
            window_samples: 32,
            min_samples: 2,
            snapshot_ttl_ms: 30_000,
            safety_margin: 0.25,
        }
    }
}

impl ReferenceEstimatorConfig {
    pub fn validate(&self) -> Result<(), String> {
        for (name, value) in [
            ("queue_capacity", self.queue_capacity),
            ("max_queue_bytes", self.max_queue_bytes),
            ("max_participants", self.max_participants),
            ("max_summary_bytes", self.max_summary_bytes),
            ("max_domains", self.max_domains),
            ("window_samples", self.window_samples),
            ("min_samples", self.min_samples),
        ] {
            if value == 0 {
                return Err(format!("SLO reference {name} must be positive"));
            }
        }
        if self.max_queue_bytes < self.max_summary_bytes {
            return Err("SLO reference queue must fit one permitted summary".into());
        }
        if self.min_samples > self.window_samples {
            return Err("SLO reference min_samples exceeds the bounded sample window".into());
        }
        if self.snapshot_ttl_ms == 0 {
            return Err("SLO reference snapshot_ttl_ms must be positive".into());
        }
        if !self.safety_margin.is_finite() || self.safety_margin < 0.0 {
            return Err("SLO reference safety_margin must be finite and nonnegative".into());
        }
        // Reject overflowing declared storage bounds before constructing any
        // queue or history. Actual summary-size validation belongs to the
        // estimator, which knows the execution contract's representation.
        self.max_domains
            .checked_mul(self.window_samples)
            .and_then(|samples| samples.checked_mul(std::mem::size_of::<std::time::Duration>()))
            .ok_or("SLO reference sample storage bound overflows")?;
        Ok(())
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CostFallback {
    #[default]
    StaticFallback,
    Reject,
}

/// Unknown/expired predictions cannot silently become reliable zero costs.
/// A bounded wait ends at the original request's ingress plus `max_wait_ms`;
/// replanning must not renew the deadline.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case", deny_unknown_fields)]
pub enum UnknownCostPolicy {
    #[default]
    StaticFallback,
    Reject,
    Wait {
        max_wait_ms: u64,
        #[serde(default)]
        on_timeout: CostFallback,
    },
}

impl UnknownCostPolicy {
    pub fn validate(&self) -> Result<(), String> {
        if matches!(self, Self::Wait { max_wait_ms: 0, .. }) {
            return Err("SLO unknown-cost wait must have a positive finite deadline".into());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn training_history_policy_is_typed_bounded_and_defaults_to_recent() {
        let mut config: SloConfig = "ttft:200,tpot:15,itl:50".parse().unwrap();
        assert_eq!(config.training_history, TrainingHistoryPolicy::Recent);
        config.training_history = TrainingHistoryPolicy::WidthStratified {
            samples_per_width: 2,
        };
        let encoded = serde_json::to_value(config).unwrap();
        assert_eq!(encoded["training_history"]["mode"], "width_stratified");
        assert_eq!(
            serde_json::from_value::<SloConfig>(encoded.clone()).unwrap(),
            config
        );
        let mut legacy = encoded.clone();
        legacy.as_object_mut().unwrap().remove("training_history");
        assert_eq!(
            serde_json::from_value::<SloConfig>(legacy)
                .unwrap()
                .training_history,
            TrainingHistoryPolicy::Recent
        );
        for count in [0, 3] {
            let mut invalid = encoded.clone();
            invalid["training_history"]["samples_per_width"] = count.into();
            assert!(serde_json::from_value::<SloConfig>(invalid).is_err());
        }
        config.predictor = PredictorFamily::ReferenceDiagnostic;
        assert!(config.validate().is_err());
    }

    #[test]
    fn known_infeasible_admission_wait_is_explicit_and_bounded() {
        let mut config: SloConfig = "ttft:200,tpot:15,itl:50".parse().unwrap();
        config.policy.joint.known_infeasible_admission =
            KnownInfeasibleAdmissionPolicy::Wait { max_wait_ms: 25 };
        config.policy.planning.max_cost_cache_bytes = 0;
        config.validate().unwrap();
        let restored: SloConfig =
            serde_json::from_str(&serde_json::to_string(&config).unwrap()).unwrap();
        assert_eq!(restored, config);
        config.policy.joint.known_infeasible_admission =
            KnownInfeasibleAdmissionPolicy::Wait { max_wait_ms: 0 };
        assert!(config.validate().is_err());
    }

    #[test]
    fn product_modes_preserve_unknown_policy_without_enabling_control() {
        for mode in [SloMode::Off, SloMode::Observe, SloMode::LearnAndControl] {
            let mut config: SloConfig = "ttft:200,tpot:15,itl:50".parse().unwrap();
            config.mode = mode;
            config.policy.unknown_cost = UnknownCostPolicy::Wait {
                max_wait_ms: 25,
                on_timeout: CostFallback::Reject,
            };
            config.validate().unwrap();
            assert_eq!(
                config.scheduler_policy(),
                (mode == SloMode::LearnAndControl).then_some(config.policy)
            );
            let restored: SloConfig =
                serde_json::from_str(&serde_json::to_string(&config).unwrap()).unwrap();
            assert_eq!(restored, config);
        }
    }

    #[test]
    fn legacy_reference_keeps_declared_policy_and_projection_migration() {
        let original: SloConfig = serde_json::from_value(serde_json::json!({
            "ttft_ms": 200, "tpot_ms": 15, "itl_ms": 50,
            "reference": { "max_participants": 17, "max_summary_bytes": 4096 },
            "unknown_cost": { "mode": "wait", "max_wait_ms": 37, "on_timeout": "reject" }
        }))
        .unwrap();
        assert_eq!(original.predictor, PredictorFamily::ReferenceDiagnostic);
        assert_eq!(
            original.policy.strategy,
            SloStrategy::ReferenceStepDiagnostic
        );
        assert_eq!(original.legacy_query_observer_limits(), Some((17, 4096)));
        assert_eq!(
            original.policy.unknown_cost,
            UnknownCostPolicy::Wait {
                max_wait_ms: 37,
                on_timeout: CostFallback::Reject
            }
        );
        // Saving before assembly must not accidentally drop the pending byte
        // projection migration and widen old reject/wait support boundaries.
        let restored: SloConfig =
            serde_json::from_str(&serde_json::to_string(&original).unwrap()).unwrap();
        assert_eq!(restored, original);
        let mut migrated = original;
        let limits = CostQueryLimits {
            max_participants: 7,
            max_query_bytes: 2048,
        };
        migrated.resolve_query_limits(limits);
        let restored: SloConfig =
            serde_json::from_str(&serde_json::to_string(&migrated).unwrap()).unwrap();
        assert_eq!(restored, migrated);
        assert_eq!(restored.policy.query, limits);
        assert_eq!(restored.legacy_query_observer_limits(), None);
    }

    #[test]
    fn neutral_scheduler_has_no_estimator_configuration() {
        let invalid = serde_json::json!({
            "ttft_ms": 200, "tpot_ms": 15, "itl_ms": 50,
            "reference": { "max_participants": 17 }
        });
        assert!(serde_json::from_value::<crate::SchedulerSloConfig>(invalid).is_err());
        let explicit: SloConfig = serde_json::from_value(serde_json::json!({
            "ttft_ms": 200, "tpot_ms": 15, "itl_ms": 50,
            "reference": { "max_participants": 17 },
            "query": { "max_participants": 3, "max_query_bytes": 1024 },
            "predictor": "physical", "strategy": "joint"
        }))
        .unwrap();
        assert_eq!(explicit.legacy_query_observer_limits(), None);
        assert_eq!(explicit.predictor, PredictorFamily::Physical);
        assert_eq!(explicit.policy.strategy, SloStrategy::Joint);
        assert_eq!(explicit.policy.query.max_participants, 3);
    }

    #[test]
    fn resource_budgets_reject_unusable_configuration() {
        for invalid in [
            CostQueryLimits {
                max_participants: 0,
                ..Default::default()
            },
            CostQueryLimits {
                max_query_bytes: 0,
                ..Default::default()
            },
        ] {
            assert!(invalid.validate().is_err());
        }
        for invalid in [
            SloPlanningBudget {
                max_queries: 0,
                ..Default::default()
            },
            SloPlanningBudget {
                max_candidates: 0,
                ..Default::default()
            },
            SloPlanningBudget {
                max_participants: 0,
                ..Default::default()
            },
            SloPlanningBudget {
                max_elapsed_us: 0,
                ..Default::default()
            },
            SloPlanningBudget {
                max_request_fact_bytes: 0,
                ..Default::default()
            },
        ] {
            assert!(invalid.validate().is_err());
        }
    }

    #[test]
    fn joint_targets_validate_metric_quantiles_completion_and_ingress_deadline() {
        JointSloTargets::default().validate().unwrap();
        JointSloTargets {
            ttft_quantile: 1.0,
            tpot_quantile: 1.0,
            itl_quantile: 1.0,
            ..Default::default()
        }
        .validate()
        .unwrap();
        for invalid in [
            JointSloTargets {
                ttft_quantile: 0.0,
                ..Default::default()
            },
            JointSloTargets {
                tpot_quantile: 1.001,
                ..Default::default()
            },
            JointSloTargets {
                itl_quantile: f64::NAN,
                ..Default::default()
            },
            JointSloTargets {
                min_completion_rate: 0.0,
                ..Default::default()
            },
            JointSloTargets {
                min_completion_rate: f64::INFINITY,
                ..Default::default()
            },
            JointSloTargets {
                request_deadline_ms: 0,
                ..Default::default()
            },
            JointSloTargets {
                completion_cohort_maturity_ms: 0,
                ..Default::default()
            },
        ] {
            assert!(invalid.validate().is_err());
        }
    }

    #[test]
    fn stopping_unknown_policy_does_not_override_explicit_cost_policy() {
        let mut config: SloConfig = serde_json::from_value(serde_json::json!({
            "ttft_ms": 200, "tpot_ms": 15, "itl_ms": 50,
            "unknown_cost": { "mode": "reject" },
            "unknown_stopping": { "mode": "wait", "max_wait_ms": 25, "on_timeout": "reject" },
            "joint": { "ttft_quantile": 0.99, "min_completion_rate": 0.999, "request_deadline_ms": 7000 }
        })).unwrap();
        assert_eq!(config.policy.unknown_cost, UnknownCostPolicy::Reject);
        assert_eq!(
            config.policy.unknown_stopping,
            UnknownCostPolicy::Wait {
                max_wait_ms: 25,
                on_timeout: CostFallback::Reject
            }
        );
        assert_eq!(config.policy.joint.ttft_quantile, 0.99);
        assert_eq!(config.policy.joint.min_completion_rate, 0.999);
        assert_eq!(config.policy.joint.request_deadline_ms, 7000);
        config.policy.unknown_stopping = UnknownCostPolicy::Wait {
            max_wait_ms: 0,
            on_timeout: CostFallback::StaticFallback,
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn reference_bounds_reject_unusable_and_overflowing_storage() {
        let defaults = ReferenceEstimatorConfig::default();
        defaults.validate().unwrap();
        for invalid in [
            ReferenceEstimatorConfig {
                queue_capacity: 0,
                ..defaults
            },
            ReferenceEstimatorConfig {
                max_queue_bytes: defaults.max_summary_bytes - 1,
                ..defaults
            },
            ReferenceEstimatorConfig {
                min_samples: defaults.window_samples + 1,
                ..defaults
            },
            ReferenceEstimatorConfig {
                max_domains: usize::MAX,
                ..defaults
            },
            ReferenceEstimatorConfig {
                snapshot_ttl_ms: 0,
                ..defaults
            },
            ReferenceEstimatorConfig {
                safety_margin: f64::NAN,
                ..defaults
            },
            ReferenceEstimatorConfig {
                safety_margin: -0.1,
                ..defaults
            },
        ] {
            assert!(invalid.validate().is_err(), "{invalid:?}");
        }
    }

    #[test]
    fn unknown_wait_has_an_explicit_nonrenewable_timeout_policy() {
        let wait: UnknownCostPolicy =
            serde_json::from_str(r#"{"mode":"wait","max_wait_ms":25,"on_timeout":"reject"}"#)
                .unwrap();
        assert_eq!(
            wait,
            UnknownCostPolicy::Wait {
                max_wait_ms: 25,
                on_timeout: CostFallback::Reject
            }
        );
        wait.validate().unwrap();
        assert!(UnknownCostPolicy::Wait {
            max_wait_ms: 0,
            on_timeout: CostFallback::StaticFallback
        }
        .validate()
        .is_err());
        assert!(serde_json::from_str::<UnknownCostPolicy>(r#"{"mode":"wait"}"#).is_err());
    }
}
