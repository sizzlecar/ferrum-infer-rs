//! Observational only. This path never observes into a trainer or changes an
//! imported artifact, support, residual, validity, TTL or generation.
use super::super::audit::{
    SelectedNotCompleted, SelectedServingComparison, SelectedServingEvaluation,
};
use super::*;
use ferrum_scheduler::implementations::continuous::cost_model as model;
use model::statistical::model::ModelUnknown;

pub(super) fn evaluate(
    entry: &CostEvidenceEntry,
    accepted_ordinal: u64,
    previous: Option<&EngineCostSnapshot>,
    clock: &dyn CostObservationClock,
) -> SelectedServingEvaluation {
    use SelectedServingEvaluation as E;
    if let CostEvidenceEntry::Training { sample, .. } = entry {
        use model::WaveObservationOutcome as O;
        let skip = match sample.outcome {
            O::Completed => None,
            O::NotSubmitted => Some(SelectedNotCompleted::NotSubmitted),
            O::Deferred => Some(SelectedNotCompleted::Deferred),
            O::FailedAfterSubmit => Some(SelectedNotCompleted::FailedAfterSubmit),
            O::PartiallyCompleted { .. } => Some(SelectedNotCompleted::PartiallyCompleted),
        };
        if let Some(reason) = skip {
            return E::NotCompleted(reason);
        }
    }
    // No invented calibration source digest/partition for ordinary serving.
    let actual = match host_content::statistical::complete_observation(entry) {
        Ok(value) => value,
        Err(reason) => return E::InvalidActual(reason),
    };
    let terminal = match entry {
        CostEvidenceEntry::Training { stages, .. } => stages.as_deref(),
        CostEvidenceEntry::StagesOnly { stages, .. } => Some(stages.as_ref()),
    }
    .is_some_and(|stages| stages.rows.iter().any(|row| row.terminal.is_some()));
    let prediction = previous.map(|model| {
        if accepted_ordinal == 0 {
            return Err(ModelUnknown::WrongSource);
        }
        if model.fingerprint() != &actual.fingerprint {
            return Err(ModelUnknown::WrongFingerprint);
        }
        // Read AFTER validation, at the actual lookup. Receipt time is retained
        // for correlation and cannot make a queued, expired model look fresh.
        let now = clock
            .now_ns()
            .filter(|now| *now >= actual.observed_at_ns)
            .ok_or(ModelUnknown::Clock)?;
        let value = model.predict_selected_wave(&actual.exact, &actual.selected, now)?;
        let compared = SelectedServingComparison {
            accepted_ordinal,
            call_id: actual.call_id,
            family_signature: *actual.selected.family_signature(),
            terminal,
            observed_at_ns: actual.observed_at_ns,
            consumed_at_ns: now,
            model_version: model.model_version(),
            planning_ns: value.planning_ns,
            actual_ns: actual.wall_ns,
            underestimate_ns: actual.wall_ns.saturating_sub(value.planning_ns),
        };
        tracing::trace!(target: "ferrum::selected_serving_audit", ?compared,
            "retrospective complete selected wave comparison; not pre-submit or visible SLO");
        Ok(compared)
    });
    if let Some(Err(reason)) = &prediction {
        tracing::trace!(target: "ferrum::selected_serving_audit", accepted_ordinal,
            call_id = actual.call_id, ?reason,
            "retrospective selected wave unavailable at consumption");
    }
    E::Complete {
        terminal,
        prediction,
    }
}
