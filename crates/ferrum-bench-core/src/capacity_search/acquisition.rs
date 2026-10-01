//! Acquisition integrity is separate from whether the declared SLO passed.
//! Missing telemetry may remain Unknown; lost work/protocol cannot be reused.
use super::types::*;
use crate::slo::RequestOutcome;

pub(super) fn classify(
    contract: &CapacityContract,
    planned: &PlannedCapacityRun,
    evidence: &CapacityRunEvidence,
    assessment: &CapacityRunAssessment,
) -> CapacityAcquisitionDisposition {
    use CapacityAcquisitionDisposition as D;
    if evidence.warmup.errored != 0
        || evidence.warmup.quality.request_error_count() > 0
        || evidence.quality.iter().any(|q| q.request_error_count() > 0)
    {
        return D::ProtocolFailure;
    }
    if evidence.server_queue_attempts.iter().any(|a| {
        !a.request_started_seconds.is_finite()
            || !a.response_completed_seconds.is_finite()
            || a.response_completed_seconds < a.request_started_seconds
            || match &a.observation {
                Ok(q) => q.validate().is_err(),
                Err(
                    ServerQueueFailure::Malformed
                    | ServerQueueFailure::ResponseTooLarge
                    | ServerQueueFailure::Runtime(_),
                ) => true,
                Err(_) => false,
            }
    }) || evidence
        .server_queue_attempts
        .windows(2)
        .any(|w| w[1].request_started_seconds < w[0].response_completed_seconds)
        || evidence.queue.iter().any(|q| {
            !q.at_seconds.is_finite()
                || q.at_seconds < 0.0
                || !q.oldest_request_age_ms.is_finite()
                || q.oldest_request_age_ms < 0.0
        })
    {
        return D::ProtocolFailure;
    }
    let mut previous_server = None;
    for current in evidence
        .server_queue_attempts
        .iter()
        .filter_map(|a| a.observation.as_ref().ok())
    {
        if previous_server.is_some_and(|previous: &ferrum_types::ExecutorQueueObservation| {
            current.engine_instance != previous.engine_instance
                || current.observed_at_ns <= previous.observed_at_ns
        }) {
            return D::ProtocolFailure;
        }
        previous_server = Some(current);
    }
    let count = planned.scheduled_arrival_ms.len();
    if evidence.requests.len() != count
        || evidence.request_records.len() != count
        || evidence.quality.len() != count
        || evidence.arrivals.len() != count
    {
        return D::IncompleteEvidence;
    }
    if evidence
        .requests
        .iter()
        .any(|r| r.outcome != RequestOutcome::Completed)
    {
        return D::ExecutionFailure;
    }
    if assessment.evaluation.is_none() {
        return D::ProtocolFailure;
    }
    if !assessment.workload_completed {
        return D::IncompleteEvidence;
    }
    for (index, arrival) in evidence.arrivals.iter().enumerate() {
        let Some((scheduled, (dispatched, (started, backlog)))) = arrival.scheduled_arrival_ms.zip(
            arrival.dispatched_ms.zip(
                arrival
                    .request_started_ms
                    .zip(arrival.client_dispatch_backlog),
            ),
        ) else {
            return D::IncompleteEvidence;
        };
        if [scheduled, dispatched, started]
            .iter()
            .any(|n| !n.is_finite() || *n < 0.0)
            || scheduled != planned.scheduled_arrival_ms[index]
            || dispatched < scheduled
            || started < dispatched
            || backlog
                != planned
                    .scheduled_arrival_ms
                    .partition_point(|t| *t <= dispatched)
                    .saturating_sub(index) as u64
        {
            return D::IncompleteEvidence;
        }
    }
    if evidence.send_window_seconds != planned.send_seconds
        || evidence.measured_duration_seconds
            > (evidence.run_ended_unix_ns - evidence.run_started_unix_ns) as f64 / 1e9
    {
        return D::IncompleteEvidence;
    }
    if !evidence.measured_duration_seconds.is_finite()
        || evidence.measured_duration_seconds < planned.send_seconds
        || evidence.measured_duration_seconds - planned.send_seconds
            > contract.window.maximum_drain_seconds
    {
        return D::Undrained;
    }
    let idle = match evidence.queue_observation_source {
        QueueObservationSource::ClientScheduledLifecycle | QueueObservationSource::ServerQueue => {
            evidence.queue.last().is_some_and(|q| {
                q.at_seconds >= evidence.measured_duration_seconds
                    && q.waiting_requests == 0
                    && q.active_requests == 0
            })
        }
        QueueObservationSource::ServerAdmissionV1 => {
            evidence.server_queue_attempts.last().is_some_and(|a| {
                a.request_started_seconds >= evidence.measured_duration_seconds
                    && a.observation.as_ref().is_ok_and(|q| {
                        q.validate().is_ok()
                            && q.waiting_requests == 0
                            && q.preempted_requests == 0
                            && q.active_prefill_sequences == 0
                            && q.active_decode_sequences == 0
                    })
            })
        }
    };
    if !idle {
        return D::Undrained;
    }
    D::Complete
}
