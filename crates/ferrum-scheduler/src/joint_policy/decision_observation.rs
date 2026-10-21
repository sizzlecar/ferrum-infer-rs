//! Same-decision facts, without additional prediction or retained trajectories.
use super::{JointFeasibility, PolicyChoice, ProbabilityInterval, QueueRisk};
use ferrum_types::SchedulerSloConfig;
use serde::Serialize;
use std::time::Duration;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MetricClass {
    Feasible,
    Infeasible,
    Unresolved,
    Undefined,
}

impl MetricClass {
    fn classify(interval: Option<ProbabilityInterval>, limit: f64) -> Self {
        match interval {
            None => Self::Undefined,
            Some(value) if value.lower > limit => Self::Infeasible,
            Some(value) if value.upper > limit => Self::Unresolved,
            Some(_) => Self::Feasible,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct MetricClasses {
    pub ttft: MetricClass,
    pub tpot: MetricClass,
    pub itl: MetricClass,
    pub completion_failure: MetricClass,
}

impl MetricClasses {
    pub(super) fn from_risk(risk: QueueRisk, config: &SchedulerSloConfig) -> Self {
        Self {
            ttft: MetricClass::classify(Some(risk.ttft), 1.0 - config.joint.ttft_quantile),
            tpot: MetricClass::classify(risk.tpot, 1.0 - config.joint.tpot_quantile),
            itl: MetricClass::classify(risk.itl, 1.0 - config.joint.itl_quantile),
            completion_failure: MetricClass::classify(
                Some(risk.completion_failure),
                1.0 - config.joint.min_completion_rate,
            ),
        }
    }
}

#[derive(Debug, Clone, Copy, Default, Serialize)]
pub struct FeasibilityCounts {
    pub feasible: u64,
    pub infeasible: u64,
    pub unresolved: u64,
}

impl FeasibilityCounts {
    pub(crate) fn record(&mut self, value: JointFeasibility) {
        let count = match value {
            JointFeasibility::Feasible => &mut self.feasible,
            JointFeasibility::Infeasible => &mut self.infeasible,
            JointFeasibility::Unresolved => &mut self.unresolved,
        };
        *count = count.saturating_add(1);
    }
    pub(crate) fn add(&mut self, other: Self) {
        self.feasible = self.feasible.saturating_add(other.feasible);
        self.infeasible = self.infeasible.saturating_add(other.infeasible);
        self.unresolved = self.unresolved.saturating_add(other.unresolved);
    }
}

#[derive(Debug, Clone, Copy, Default, Serialize)]
pub struct MetricCounts {
    pub feasible: u64,
    pub infeasible: u64,
    pub unresolved: u64,
    pub undefined: u64,
}

impl MetricCounts {
    fn record(&mut self, value: MetricClass) {
        let count = match value {
            MetricClass::Feasible => &mut self.feasible,
            MetricClass::Infeasible => &mut self.infeasible,
            MetricClass::Unresolved => &mut self.unresolved,
            MetricClass::Undefined => &mut self.undefined,
        };
        *count = count.saturating_add(1);
    }
    fn add(&mut self, other: Self) {
        self.feasible = self.feasible.saturating_add(other.feasible);
        self.infeasible = self.infeasible.saturating_add(other.infeasible);
        self.unresolved = self.unresolved.saturating_add(other.unresolved);
        self.undefined = self.undefined.saturating_add(other.undefined);
    }
}

#[derive(Debug, Clone, Copy, Default, Serialize)]
pub struct MetricClassCounts {
    pub ttft: MetricCounts,
    pub tpot: MetricCounts,
    pub itl: MetricCounts,
    pub completion_failure: MetricCounts,
}

impl MetricClassCounts {
    pub(crate) fn record(&mut self, value: MetricClasses) {
        self.ttft.record(value.ttft);
        self.tpot.record(value.tpot);
        self.itl.record(value.itl);
        self.completion_failure.record(value.completion_failure);
    }
    pub(crate) fn add(&mut self, other: Self) {
        self.ttft.add(other.ttft);
        self.tpot.add(other.tpot);
        self.itl.add(other.itl);
        self.completion_failure.add(other.completion_failure);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SelectionGate {
    RiskInfeasible,
    RiskUnresolved,
    /// Missing complete endpoint evidence for a partial-static comparison, or
    /// an unproved common-window bound when static is not known infeasible.
    PartialWindowUnproven,
    TargetTimeUnavailable,
    StaticFeasibleWindowDebt,
    StaticTimeOrPairedEvidenceUnavailable,
    CandidateNotFaster,
    GainNotAboveEvidenceMargin,
    StaticUnresolved,
    Eligible,
}

#[derive(Debug, Clone, Copy, Default, Serialize)]
pub struct SelectionGateCounts {
    pub risk_infeasible: u64,
    pub risk_unresolved: u64,
    pub partial_window_unproven: u64,
    pub target_time_unavailable: u64,
    pub static_feasible_window_debt: u64,
    pub static_time_or_paired_evidence_unavailable: u64,
    pub candidate_not_faster: u64,
    pub gain_not_above_evidence_margin: u64,
    pub static_unresolved: u64,
    pub eligible: u64,
}

impl SelectionGateCounts {
    fn record(&mut self, gate: SelectionGate) {
        let count = match gate {
            SelectionGate::RiskInfeasible => &mut self.risk_infeasible,
            SelectionGate::RiskUnresolved => &mut self.risk_unresolved,
            SelectionGate::PartialWindowUnproven => &mut self.partial_window_unproven,
            SelectionGate::TargetTimeUnavailable => &mut self.target_time_unavailable,
            SelectionGate::StaticFeasibleWindowDebt => &mut self.static_feasible_window_debt,
            SelectionGate::StaticTimeOrPairedEvidenceUnavailable => {
                &mut self.static_time_or_paired_evidence_unavailable
            }
            SelectionGate::CandidateNotFaster => &mut self.candidate_not_faster,
            SelectionGate::GainNotAboveEvidenceMargin => &mut self.gain_not_above_evidence_margin,
            SelectionGate::StaticUnresolved => &mut self.static_unresolved,
            SelectionGate::Eligible => &mut self.eligible,
        };
        *count = count.saturating_add(1);
    }
    fn add(&mut self, other: Self) {
        macro_rules! add {
            ($($field:ident),+ $(,)?) => {$(
                self.$field = self.$field.saturating_add(other.$field);
            )+};
        }
        add!(
            risk_infeasible,
            risk_unresolved,
            partial_window_unproven,
            target_time_unavailable,
            static_feasible_window_debt,
            static_time_or_paired_evidence_unavailable,
            candidate_not_faster,
            gain_not_above_evidence_margin,
            static_unresolved,
            eligible
        );
    }
}

#[derive(Debug, Clone, Copy, Default, Serialize)]
pub struct CandidateDecisionCounts {
    pub joint: FeasibilityCounts,
    pub metrics: MetricClassCounts,
    /// One first reached selection gate per alternative, not all possible causes.
    pub gates: SelectionGateCounts,
}

impl CandidateDecisionCounts {
    pub(super) fn record(
        &mut self,
        joint: JointFeasibility,
        metrics: MetricClasses,
        gate: SelectionGate,
    ) {
        self.joint.record(joint);
        self.metrics.record(metrics);
        self.gates.record(gate);
    }
    pub(crate) fn add(&mut self, other: Self) {
        self.joint.add(other.joint);
        self.metrics.add(other.metrics);
        self.gates.add(other.gates);
    }
}

#[derive(Debug, Clone, Copy, Default, Serialize)]
pub struct ProductSuccessEvidence {
    /// Unavailable support is also uncertain: these are overlapping counts.
    pub unavailable_requests: usize,
    pub uncertain_requests: usize,
    pub success_lower_sum: f64,
    pub success_upper_sum: f64,
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct RiskDenominators {
    /// TTFT and completion use this equal-request denominator.
    pub requests: usize,
    pub tpot_known_eligible: f64,
    pub tpot_unknown_eligible: f64,
    /// Uncertain outcome within the known eligible denominator.
    pub tpot_unknown_outcome: f64,
    pub tpot_undefined: f64,
    pub itl_known_events: f64,
    pub itl_unknown_events: f64,
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct ActionDecisionEvidence {
    pub joint: JointFeasibility,
    pub metrics: MetricClasses,
    pub risk: QueueRisk,
    pub denominators: RiskDenominators,
    pub target_time: Option<Duration>,
    pub known_target_time_mass: f64,
    pub unsupported_trajectory_mass: f64,
    /// Weighted requests whose endpoint completion was not known.
    pub unfinished_request_mass: f64,
    pub window_debt: bool,
    pub partial_window_unproven: bool,
    pub first_gate: Option<SelectionGate>,
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct UnresolvedDecisionWitness {
    /// Same aggregate; never a join to the owner's separately retained query.
    pub revision: Option<u64>,
    pub alternatives_total: usize,
    pub static_raw: JointFeasibility,
    pub static_effective: JointFeasibility,
    pub partial_static_ranks: usize,
    pub endpoint_unresolved_mass: f64,
    pub failure_limits: [f64; 4],
    pub product_success: ProductSuccessEvidence,
    pub static_action: ActionDecisionEvidence,
    /// Retain details for the first two alternatives. Every alternative is
    /// evaluated and counted; any omitted details are explicit.
    pub alternatives: [Option<ActionDecisionEvidence>; 2],
    pub details_omitted: usize,
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct DecisionObservation {
    pub choice: PolicyChoice,
    pub alternatives_total: usize,
    pub static_raw: JointFeasibility,
    pub static_effective: JointFeasibility,
    pub static_metrics: MetricClasses,
    pub candidates: CandidateDecisionCounts,
    pub unresolved: Option<UnresolvedDecisionWitness>,
}
