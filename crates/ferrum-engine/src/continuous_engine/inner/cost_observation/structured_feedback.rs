//! Complete actual V2 waves against the immutable pre-drain snapshot. This is
//! retrospective drift feedback, not a submitted prediction or calibration.
use super::profile::EngineCostSnapshot;
use super::selected_feedback::{Comparison, FeedbackObservation as Observation};
use super::*;
use ferrum_scheduler::implementations::continuous::cost_model::structured_v2::{
    StructuredQueryV2, StructuredUnknownV2 as Unknown,
};

pub(super) fn evaluate(
    entry: &CostEvidenceEntry,
    previous: Option<&EngineCostSnapshot>,
    clock: &dyn CostObservationClock,
) -> Observation {
    if let CostEvidenceEntry::Training { sample, .. } = entry {
        use ferrum_scheduler::implementations::continuous::cost_model::WaveObservationOutcome as O;
        match sample.outcome {
            O::NotSubmitted | O::Deferred => return Observation::NotSubmitted,
            O::FailedAfterSubmit | O::PartiallyCompleted { .. } => {
                return Observation::FailedOrPartial
            }
            O::Completed => {}
        }
    }
    let stages = match entry {
        CostEvidenceEntry::Training { stages, .. } => stages.as_ref(),
        CostEvidenceEntry::StagesOnly { stages, .. } => Some(stages),
    };
    let Some(stages) = stages else {
        return Observation::Uncomparable;
    };
    if stages.completeness == HostStageCompleteness::Failed {
        return Observation::FailedOrPartial;
    }
    let (input, actual) = match trainer::structured_v2::structured_serving_observation_v2(stages) {
        Ok(value) => value,
        Err(reason) => {
            query_metrics::record_structured_v2_retrospective::<()>(&Err(reason));
            return Observation::Uncomparable;
        }
    };
    let Some(previous) = previous else {
        return Observation::Uncomparable;
    };
    if previous.fingerprint() != &actual.fingerprint {
        return Observation::InvalidIdentity;
    }
    let Some(now) = clock.now_ns().filter(|now| *now >= actual.observed_at_ns) else {
        return Observation::InvalidIdentity;
    };
    let query = StructuredQueryV2::exact(input);
    let value = previous.audit_structured_query_v2(&query, now);
    query_metrics::record_structured_v2_retrospective(&value);
    let value = match value {
        Ok(value) => value,
        Err(Unknown::WrongFingerprint | Unknown::Clock) => return Observation::InvalidIdentity,
        // Missing owner/support, expired samples or a revoked epoch never
        // become zero error and cannot acquire a margin or new eligibility.
        Err(_) => return Observation::Uncomparable,
    };
    let Some(base) = value
        .planning_ns
        .checked_sub(previous.feedback_margin(query.domain_signature()))
    else {
        return Observation::InvalidIdentity;
    };
    tracing::trace!(target: "ferrum::structured_serving_feedback", call_id=actual.call_id,
        owner_domain=?query.domain_signature(), model_version=value.model_version,
        observed_at_ns=actual.observed_at_ns, consumed_at_ns=now,
        planning_ns=value.planning_ns, actual_ns=actual.wall_ns,
        "retrospective complete V2 wave against immutable pre-drain model; not pre-submit or client SLO");
    Observation::Compared(Comparison {
        family: *query.domain_signature(),
        base_planning_ns: base,
        actual_ns: actual.wall_ns,
        observed_at_ns: actual.observed_at_ns,
        consumed_at_ns: now,
    })
}
