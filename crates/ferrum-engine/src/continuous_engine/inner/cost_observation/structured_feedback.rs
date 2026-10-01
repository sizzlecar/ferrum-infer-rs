//! Complete actual V2 waves against the immutable pre-drain snapshot. This is
//! retrospective drift feedback, not a submitted prediction or calibration.
use super::profile::EngineCostSnapshot;
use super::selected_feedback::{Comparison, FeedbackObservation as Observation};
use super::*;
use ferrum_scheduler::implementations::continuous::cost_model::structured_v2::StructuredQueryFailureV2 as QueryFailure;
use ferrum_scheduler::implementations::continuous::cost_model::structured_v2::StructuredUnknownV2 as Unknown;

/// Fixed-size context from the original evaluation. Diagnostics never perform
/// another projection or lookup, and cannot change feedback classification.
#[derive(Clone, Copy)]
pub(super) enum Failure {
    MissingStages,
    MissingSnapshot,
    Fingerprint,
    Clock,
    RouteProof,
    Projection(super::resolved::ProjectionError),
    FeedbackScope(Unknown),
    Lookup(Unknown),
    MissingSource,
    MarginUnderflow,
}
impl std::fmt::Debug for Failure {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Projection(reason) => formatter.debug_tuple("Projection").field(reason).finish(),
            Self::FeedbackScope(reason) => formatter
                .debug_tuple("FeedbackScope")
                .field(reason)
                .finish(),
            Self::Lookup(reason) => formatter.debug_tuple("Lookup").field(reason).finish(),
            Self::MissingStages => formatter.write_str("MissingStages"),
            Self::MissingSnapshot => formatter.write_str("MissingSnapshot"),
            Self::Fingerprint => formatter.write_str("Fingerprint"),
            Self::Clock => formatter.write_str("Clock"),
            Self::RouteProof => formatter.write_str("RouteProof"),
            Self::MissingSource => formatter.write_str("MissingSource"),
            Self::MarginUnderflow => formatter.write_str("MarginUnderflow"),
        }
    }
}

pub(super) fn evaluate(
    entry: &CostEvidenceEntry,
    previous: Option<&EngineCostSnapshot>,
    clock: &dyn CostObservationClock,
) -> Observation {
    evaluate_projected(entry, previous, clock, None, &mut None)
}

pub(super) fn evaluate_resolved(
    resolved: &super::resolved::ResolvedCostEntry,
    previous: Option<&EngineCostSnapshot>,
    clock: &dyn CostObservationClock,
    failure: &mut Option<Failure>,
) -> Observation {
    evaluate_projected(resolved.entry(), previous, clock, Some(resolved), failure)
}

/// Only the live resolver can attach this complete same-call preparation
/// receipt. Its numerical training rejection is not ordinary model drift.
pub(super) fn evaluate_preparation(
    resolved: &super::resolved::ResolvedCostEntry,
    previous: Option<&EngineCostSnapshot>,
    clock: &dyn CostObservationClock,
    failure: &mut Option<Failure>,
) -> Option<Observation> {
    let (fingerprint, observed_at_ns) = resolved.preparation_feedback()?;
    let Some(previous) = previous else {
        *failure = Some(Failure::MissingSnapshot);
        return Some(Observation::Uncomparable);
    };
    if previous.fingerprint() != fingerprint {
        *failure = Some(Failure::Fingerprint);
        return Some(Observation::InvalidIdentity);
    }
    Some(match clock.now_ns().filter(|now| *now >= observed_at_ns) {
        Some(consumed_at_ns) => Observation::OutsidePreparation {
            observed_at_ns,
            consumed_at_ns,
        },
        None => {
            *failure = Some(Failure::Clock);
            Observation::InvalidIdentity
        }
    })
}

/// A private original selector remains valid between calibration blocks.
/// No ticket is created, no source membership is counted and no numerical cost
/// is inferred for the excluded execution route.
pub(super) fn evaluate_outside(
    entry: &CostEvidenceEntry,
    previous: Option<&EngineCostSnapshot>,
    clock: &dyn CostObservationClock,
    failure: &mut Option<Failure>,
) -> Option<Observation> {
    let stages = match entry {
        CostEvidenceEntry::Training { stages, .. } => stages.as_deref(),
        CostEvidenceEntry::StagesOnly { stages, .. } => Some(stages.as_ref()),
    }?;
    let proof = stages.route_evidence.as_ref().filter(|p| p.is_outside())?;
    let Some(previous) = previous else {
        *failure = Some(Failure::MissingSnapshot);
        return Some(Observation::Uncomparable);
    };
    let Some(observed_at_ns) = proof.feedback_observed_at(stages, previous.fingerprint()) else {
        *failure = Some(Failure::RouteProof);
        return Some(Observation::Uncomparable);
    };
    Some(match clock.now_ns().filter(|now| *now >= observed_at_ns) {
        Some(consumed_at_ns) => Observation::OutsideRoute {
            observed_at_ns,
            consumed_at_ns,
        },
        None => {
            *failure = Some(Failure::Clock);
            Observation::InvalidIdentity
        }
    })
}

fn evaluate_projected(
    entry: &CostEvidenceEntry,
    previous: Option<&EngineCostSnapshot>,
    clock: &dyn CostObservationClock,
    resolved: Option<&super::resolved::ResolvedCostEntry>,
    failure: &mut Option<Failure>,
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
        *failure = Some(Failure::MissingStages);
        return Observation::Uncomparable;
    };
    if stages.completeness == HostStageCompleteness::Failed {
        return Observation::FailedOrPartial;
    }
    let fallback = resolved
        .is_none()
        .then(|| super::resolved::ResolvedCostEntry::project(entry));
    let projected = match resolved {
        Some(resolved) => resolved.structured(),
        None => fallback
            .as_ref()
            .expect("projection created")
            .as_ref()
            .map_err(|e| *e),
    };
    let projected = match projected {
        Ok(value) => value,
        Err(reason) => {
            *failure = Some(Failure::Projection(reason));
            let reason = match reason {
                super::resolved::ProjectionError::Settlement(_) => Unknown::InvalidSample,
                super::resolved::ProjectionError::MissingRecipe => Unknown::MissingEvidence,
                super::resolved::ProjectionError::Numeric(reason) => reason,
            };
            query_metrics::record_structured_v2_retrospective::<()>(&Err(reason));
            return Observation::Uncomparable;
        }
    };
    if let Err(reason) = projected.feedback_scope {
        *failure = Some(Failure::FeedbackScope(reason));
        query_metrics::record_structured_v2_retrospective::<()>(&Err(reason));
        return Observation::Uncomparable;
    }
    let actual = &projected.actual;
    let query = &projected.query;
    let Some(previous) = previous else {
        *failure = Some(Failure::MissingSnapshot);
        return Observation::Uncomparable;
    };
    if previous.fingerprint() != &actual.fingerprint {
        *failure = Some(Failure::Fingerprint);
        return Observation::InvalidIdentity;
    }
    let Some(now) = clock.now_ns().filter(|now| *now >= actual.observed_at_ns) else {
        *failure = Some(Failure::Clock);
        return Observation::InvalidIdentity;
    };
    let value = previous.audit_structured_query_v2_detailed(&query, now);
    query_metrics::record_structured_v2_retrospective(
        &value
            .as_ref()
            .map(|_| ())
            .map_err(|failure| failure.reason()),
    );
    let value = match value {
        Ok(value) => value,
        Err(QueryFailure::Invalid(Unknown::Stale)) => {
            // TTL is not a model error. Still validate the complete original
            // numerical input: the prediction's time gate ran before that
            // validation and must not conceal damaged evidence.
            return match previous.expired_structured_feedback_domain(query, now) {
                Ok(domain) => Observation::ExpiredModel {
                    model_epoch: previous.model_version(),
                    domain,
                    observed_at_ns: actual.observed_at_ns,
                    consumed_at_ns: now,
                },
                Err(error) => {
                    let reason = error.reason();
                    *failure = Some(Failure::Lookup(reason));
                    if matches!(reason, Unknown::WrongFingerprint | Unknown::Clock) {
                        Observation::InvalidIdentity
                    } else {
                        Observation::Uncomparable
                    }
                }
            };
        }
        Err(QueryFailure::Invalid(reason @ (Unknown::WrongFingerprint | Unknown::Clock))) => {
            *failure = Some(Failure::Lookup(reason));
            return Observation::InvalidIdentity;
        }
        Err(QueryFailure::Invalid(Unknown::WrongDomain))
            if previous.undeclared_structured_owner(&query) =>
        {
            return Observation::OutsideCatalog {
                observed_at_ns: actual.observed_at_ns,
                consumed_at_ns: now,
            };
        }
        Err(QueryFailure::OutsideSupport(_) | QueryFailure::OutsidePredictionRange(_)) => {
            // The immutable query has already checked epoch, fingerprint,
            // clock and complete input evidence. No prediction was issued
            // for this input, so it cannot be an error observation for the
            // smaller independently qualified region.
            // The predictor classified this at the original support/range
            // boundary, after identity and input validation. No blanket
            // exemption for Numerical, Capacity or qualification errors.
            return Observation::OutsideSupport {
                observed_at_ns: actual.observed_at_ns,
                consumed_at_ns: now,
            };
        }
        // Missing evidence and a revoked epoch still fail.
        Err(QueryFailure::Invalid(reason)) => {
            *failure = Some(Failure::Lookup(reason));
            return Observation::Uncomparable;
        }
    };
    let Some(source) = previous.prospective_source(query) else {
        *failure = Some(Failure::MissingSource);
        return Observation::InvalidIdentity;
    };
    let Some(base) = value
        .planning_ns
        .checked_sub(previous.feedback_margin(&source.domain))
    else {
        *failure = Some(Failure::MarginUnderflow);
        return Observation::InvalidIdentity;
    };
    tracing::trace!(target: "ferrum::structured_serving_feedback", call_id=actual.call_id,
        owner_domain=?source.domain, model_version=value.model_version,
        observed_at_ns=actual.observed_at_ns, consumed_at_ns=now,
        planning_ns=value.planning_ns, actual_ns=actual.wall_ns,
        "retrospective complete V2 wave against immutable pre-drain model; not pre-submit or client SLO");
    Observation::Compared(Comparison {
        family: source.domain,
        base_planning_ns: base,
        actual_ns: actual.wall_ns,
        observed_at_ns: actual.observed_at_ns,
        consumed_at_ns: now,
    })
}
