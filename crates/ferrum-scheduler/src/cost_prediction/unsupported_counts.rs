//! Bounded counts of the reasons actually returned by joint comparisons.
//! A missing count is not proof of support; no unvisited cause is inferred.
use super::CostUnknownReason;
use crate::joint_policy::Unsupported;
use serde::{ser::SerializeMap, Serialize, Serializer};

#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "snake_case")]
#[repr(usize)]
enum Reason {
    EmptyCohort,
    ParticipantBudget,
    CandidateBudget,
    QueryBudget,
    TimeBudget,
    WaitingResources,
    UnknownKvCapacity,
    PressureActive,
    ColdFillFirst,
    PrefixRendezvous,
    DecodeBackpressure,
    MissingCapacity,
    InsufficientCapacity,
    AdmissionCapacity,
    PendingRequestState,
    InvalidClock,
    NonStreamingClock,
    HistoricalTextCoalescing,
    StoppingCoverage,
    FeasibilityUnresolved,
    InvalidState,
    InvalidPlan,
    NoProgress,
    BoundedHorizonExhausted,
    Arithmetic,
    CostNoCoverage,
    CostStale,
    CostInexactWork,
    CostUnsupportedDomain,
    CostQueryLimit,
    CostPlanningBudget,
}

impl Reason {
    const ALL: [Self; 31] = [
        Self::EmptyCohort,
        Self::ParticipantBudget,
        Self::CandidateBudget,
        Self::QueryBudget,
        Self::TimeBudget,
        Self::WaitingResources,
        Self::UnknownKvCapacity,
        Self::PressureActive,
        Self::ColdFillFirst,
        Self::PrefixRendezvous,
        Self::DecodeBackpressure,
        Self::MissingCapacity,
        Self::InsufficientCapacity,
        Self::AdmissionCapacity,
        Self::PendingRequestState,
        Self::InvalidClock,
        Self::NonStreamingClock,
        Self::HistoricalTextCoalescing,
        Self::StoppingCoverage,
        Self::FeasibilityUnresolved,
        Self::InvalidState,
        Self::InvalidPlan,
        Self::NoProgress,
        Self::BoundedHorizonExhausted,
        Self::Arithmetic,
        Self::CostNoCoverage,
        Self::CostStale,
        Self::CostInexactWork,
        Self::CostUnsupportedDomain,
        Self::CostQueryLimit,
        Self::CostPlanningBudget,
    ];

    fn from_unsupported(reason: Unsupported) -> Self {
        match reason {
            Unsupported::EmptyCohort => Self::EmptyCohort,
            Unsupported::ParticipantBudget => Self::ParticipantBudget,
            Unsupported::CandidateBudget => Self::CandidateBudget,
            Unsupported::QueryBudget => Self::QueryBudget,
            Unsupported::TimeBudget => Self::TimeBudget,
            Unsupported::WaitingResources => Self::WaitingResources,
            Unsupported::UnknownKvCapacity => Self::UnknownKvCapacity,
            Unsupported::PressureActive => Self::PressureActive,
            Unsupported::ColdFillFirst => Self::ColdFillFirst,
            Unsupported::PrefixRendezvous => Self::PrefixRendezvous,
            Unsupported::DecodeBackpressure => Self::DecodeBackpressure,
            Unsupported::MissingCapacity => Self::MissingCapacity,
            Unsupported::InsufficientCapacity => Self::InsufficientCapacity,
            Unsupported::AdmissionCapacity => Self::AdmissionCapacity,
            Unsupported::PendingRequestState => Self::PendingRequestState,
            Unsupported::InvalidClock => Self::InvalidClock,
            Unsupported::NonStreamingClock => Self::NonStreamingClock,
            Unsupported::HistoricalTextCoalescing => Self::HistoricalTextCoalescing,
            Unsupported::StoppingCoverage => Self::StoppingCoverage,
            Unsupported::FeasibilityUnresolved => Self::FeasibilityUnresolved,
            Unsupported::InvalidState => Self::InvalidState,
            Unsupported::InvalidPlan => Self::InvalidPlan,
            Unsupported::NoProgress => Self::NoProgress,
            Unsupported::BoundedHorizonExhausted => Self::BoundedHorizonExhausted,
            Unsupported::Arithmetic => Self::Arithmetic,
            Unsupported::Cost(CostUnknownReason::NoCoverage) => Self::CostNoCoverage,
            Unsupported::Cost(CostUnknownReason::Stale) => Self::CostStale,
            Unsupported::Cost(CostUnknownReason::InexactWork) => Self::CostInexactWork,
            Unsupported::Cost(
                CostUnknownReason::UnsupportedDomain | CostUnknownReason::UnsupportedWithDetail(_),
            ) => Self::CostUnsupportedDomain,
            Unsupported::Cost(CostUnknownReason::QueryLimit) => Self::CostQueryLimit,
            Unsupported::Cost(CostUnknownReason::PlanningBudget) => Self::CostPlanningBudget,
        }
    }
}

/// Fixed storage and constant-time recording. Serialization exposes stable
/// named counters, including zeroes, without building an intermediate map.
#[derive(Debug, Clone, Copy)]
pub(crate) struct JointUnsupportedCounts {
    values: [u64; Reason::ALL.len()],
}

impl Default for JointUnsupportedCounts {
    fn default() -> Self {
        Self {
            values: [0; Reason::ALL.len()],
        }
    }
}

impl JointUnsupportedCounts {
    pub(crate) fn record(&mut self, reason: Unsupported) {
        let count = &mut self.values[Reason::from_unsupported(reason) as usize];
        *count = count.saturating_add(1);
    }

    pub(crate) fn add(&mut self, other: Self) {
        for (count, additional) in self.values.iter_mut().zip(other.values) {
            *count = count.saturating_add(additional);
        }
    }
}

impl Serialize for JointUnsupportedCounts {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut map = serializer.serialize_map(Some(Reason::ALL.len()))?;
        for reason in Reason::ALL {
            map.serialize_entry(&reason, &self.values[reason as usize])?;
        }
        map.end()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cost_prediction::PlanningStatistics;

    #[test]
    fn returned_joint_reasons_preserve_named_counts_and_aggregate_partition() {
        let cases = [
            (Unsupported::EmptyCohort, "empty_cohort"),
            (Unsupported::ParticipantBudget, "participant_budget"),
            (Unsupported::CandidateBudget, "candidate_budget"),
            (Unsupported::QueryBudget, "query_budget"),
            (Unsupported::TimeBudget, "time_budget"),
            (Unsupported::WaitingResources, "waiting_resources"),
            (Unsupported::UnknownKvCapacity, "unknown_kv_capacity"),
            (Unsupported::PressureActive, "pressure_active"),
            (Unsupported::ColdFillFirst, "cold_fill_first"),
            (Unsupported::PrefixRendezvous, "prefix_rendezvous"),
            (Unsupported::DecodeBackpressure, "decode_backpressure"),
            (Unsupported::MissingCapacity, "missing_capacity"),
            (Unsupported::InsufficientCapacity, "insufficient_capacity"),
            (Unsupported::AdmissionCapacity, "admission_capacity"),
            (Unsupported::PendingRequestState, "pending_request_state"),
            (Unsupported::InvalidClock, "invalid_clock"),
            (Unsupported::NonStreamingClock, "non_streaming_clock"),
            (
                Unsupported::HistoricalTextCoalescing,
                "historical_text_coalescing",
            ),
            (Unsupported::StoppingCoverage, "stopping_coverage"),
            (Unsupported::FeasibilityUnresolved, "feasibility_unresolved"),
            (Unsupported::InvalidState, "invalid_state"),
            (Unsupported::InvalidPlan, "invalid_plan"),
            (Unsupported::NoProgress, "no_progress"),
            (
                Unsupported::BoundedHorizonExhausted,
                "bounded_horizon_exhausted",
            ),
            (Unsupported::Arithmetic, "arithmetic"),
            (
                Unsupported::Cost(CostUnknownReason::NoCoverage),
                "cost_no_coverage",
            ),
            (Unsupported::Cost(CostUnknownReason::Stale), "cost_stale"),
            (
                Unsupported::Cost(CostUnknownReason::InexactWork),
                "cost_inexact_work",
            ),
            (
                Unsupported::Cost(CostUnknownReason::UnsupportedDomain),
                "cost_unsupported_domain",
            ),
            (
                Unsupported::Cost(CostUnknownReason::QueryLimit),
                "cost_query_limit",
            ),
            (
                Unsupported::Cost(CostUnknownReason::PlanningBudget),
                "cost_planning_budget",
            ),
        ];
        let mut combined = JointUnsupportedCounts::default();
        let mut expected_resources = 0;
        let mut expected_unresolved = 0;
        for (index, (reason, key)) in cases.iter().copied().enumerate() {
            let mut round = PlanningStatistics::default();
            for _ in 0..=index {
                round.record_joint_unsupported(reason);
            }
            let count = (index + 1) as u64;
            if reason.is_resource_unknown() {
                expected_resources += count;
                assert_eq!(round.joint_unknown_resources, count);
                assert_eq!(round.joint_unresolved, 0);
            } else {
                expected_unresolved += count;
                assert_eq!(round.joint_unresolved, count);
                assert_eq!(round.joint_unknown_resources, 0);
            }
            assert_eq!(
                serde_json::to_value(round.joint_unsupported).unwrap()[key],
                count
            );
            combined.add(round.joint_unsupported);
        }
        let serialized = serde_json::to_value(combined).unwrap();
        for (index, (_, key)) in cases.iter().enumerate() {
            assert_eq!(serialized[*key], (index + 1) as u64);
        }
        assert_eq!(
            serialized
                .as_object()
                .unwrap()
                .values()
                .map(|value| value.as_u64().unwrap())
                .sum::<u64>(),
            expected_resources + expected_unresolved
        );
    }

    #[test]
    fn joint_reason_recording_and_aggregation_saturate() {
        let mut counts = JointUnsupportedCounts::default();
        counts.values[Reason::ColdFillFirst as usize] = u64::MAX;
        counts.record(Unsupported::ColdFillFirst);
        counts.add(counts);
        assert_eq!(
            serde_json::to_value(counts).unwrap()["cold_fill_first"],
            u64::MAX
        );
        assert_eq!(serde_json::to_value(counts).unwrap()["pressure_active"], 0);
    }

    #[test]
    fn refined_reasons_keep_previous_resource_partition() {
        for reason in [
            Unsupported::ColdFillFirst,
            Unsupported::PressureActive,
            Unsupported::PrefixRendezvous,
            Unsupported::DecodeBackpressure,
            Unsupported::MissingCapacity,
            Unsupported::InsufficientCapacity,
            Unsupported::AdmissionCapacity,
        ] {
            assert!(reason.is_resource_unknown());
        }
        for reason in [
            Unsupported::PendingRequestState,
            Unsupported::InvalidClock,
            Unsupported::NonStreamingClock,
            Unsupported::HistoricalTextCoalescing,
            Unsupported::FeasibilityUnresolved,
            Unsupported::StoppingCoverage,
            Unsupported::BoundedHorizonExhausted,
        ] {
            assert!(!reason.is_resource_unknown());
        }
    }
}
