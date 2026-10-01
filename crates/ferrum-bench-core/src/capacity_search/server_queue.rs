//! Evaluate only original server admission observations; HTTP clock brackets
//! establish window membership, server monotonic differences establish slopes.
use super::{
    evaluate::{missing, slope},
    types::*,
};

pub(super) fn evaluate(
    contract: &CapacityContract,
    evidence: &CapacityRunEvidence,
    result: &mut CapacityRunAssessment,
    failed: &mut bool,
) {
    let attempts = &evidence.server_queue_attempts;
    let window = &contract.window;
    if !evidence.queue_capture_complete
        || attempts.len() > contract.maximum_queue_samples_per_run
        || !evidence.queue.is_empty()
    {
        missing(result, "server admission capture incomplete, over bound, or mixed with a different queue source");
    }
    if attempts.iter().any(|a| {
        !a.request_started_seconds.is_finite()
            || !a.response_completed_seconds.is_finite()
            || a.response_completed_seconds < a.request_started_seconds
    }) || attempts
        .windows(2)
        .any(|a| a[1].request_started_seconds < a[0].response_completed_seconds)
    {
        missing(
            result,
            "server admission HTTP intervals are malformed or overlap",
        );
        return;
    }
    let mut good = Vec::new();
    let mut errors = 0usize;
    for attempt in attempts {
        match &attempt.observation {
            Ok(observation) => {
                if observation.validate().is_err() {
                    missing(result, "invalid original server admission value");
                    return;
                }
                good.push((attempt, observation));
            }
            Err(error) => {
                errors += 1;
                if matches!(error, ServerQueueFailure::Runtime(message) if message.len() > 512) {
                    missing(
                        result,
                        "server admission error exceeds bounded diagnostic size",
                    );
                }
            }
        }
    }
    if errors > 0 {
        // Failed attempts remain raw evidence. They cannot bridge a coverage
        // gap; redundant successful reads may still satisfy the frozen bound.
        result.issues.push(format!(
            "server admission samples unavailable/failed: {errors}"
        ));
    }
    let (Some((first_attempt, first)), Some((last_attempt, last))) = (good.first(), good.last())
    else {
        missing(
            result,
            "server admission unavailable (N/A); no queue age was inferred",
        );
        return;
    };
    if first_attempt.response_completed_seconds > window.observe_from_seconds
        || last_attempt.request_started_seconds < evidence.measured_duration_seconds
        || last_attempt.response_completed_seconds > evidence.measured_duration_seconds + window.maximum_queue_sample_gap_seconds
        || good.windows(2).any(|pair| {
            let (a, left) = pair[0];
            let (b, right) = pair[1];
            left.engine_instance != right.engine_instance
                || right.observed_at_ns <= left.observed_at_ns
                // The worst possible observation gap is bracketed without
                // equating either host's Instant epoch or wall time.
                || b.response_completed_seconds - a.request_started_seconds > window.maximum_queue_sample_gap_seconds
                || (right.observed_at_ns - left.observed_at_ns) as f64 / 1e9 > window.maximum_queue_sample_gap_seconds
        }) {
        missing(result, "server admission epoch/clock or conservative observation-window coverage is incomplete");
        return;
    }
    let points: Vec<_> = good
        .iter()
        // Only observations certainly inside the sustained send window enter
        // slope estimation. Boundary-straddling reads remain in raw evidence.
        .filter(|(a, _)| {
            a.request_started_seconds >= window.observe_from_seconds
                && a.response_completed_seconds <= window.send_seconds
        })
        .map(|(_, q)| ServiceQueueSample {
            at_seconds: (q.observed_at_ns - first.observed_at_ns) as f64 / 1e9,
            waiting_requests: u64::from(q.waiting_requests) + u64::from(q.preempted_requests),
            active_requests: u64::from(q.active_prefill_sequences)
                + u64::from(q.active_decode_sequences),
            oldest_request_age_ms: q.oldest_unfinished_ingress_age_ns.unwrap_or(0) as f64 / 1e6,
        })
        .collect();
    let points = points.iter().collect::<Vec<_>>();
    result.unfinished_requests_slope_per_second =
        slope(&points, |s| (s.waiting_requests + s.active_requests) as f64);
    result.oldest_age_slope_ms_per_second = slope(&points, |s| s.oldest_request_age_ms);
    let waiting: Vec<_> = good
        .iter()
        .filter(|(a, _)| {
            a.request_started_seconds >= window.observe_from_seconds
                && a.response_completed_seconds <= window.send_seconds
        })
        .map(|(_, q)| ServiceQueueSample {
            at_seconds: (q.observed_at_ns - first.observed_at_ns) as f64 / 1e9,
            waiting_requests: u64::from(q.waiting_requests),
            active_requests: 0,
            oldest_request_age_ms: q.oldest_waiting_ingress_age_ns.unwrap_or(0) as f64 / 1e6,
        })
        .collect();
    let waiting = waiting.iter().collect::<Vec<_>>();
    result.waiting_requests_slope_per_second = slope(&waiting, |s| s.waiting_requests as f64);
    result.oldest_waiting_ingress_age_slope_ms_per_second =
        slope(&waiting, |s| s.oldest_request_age_ms);
    if result.unfinished_requests_slope_per_second.is_none()
        || result.oldest_age_slope_ms_per_second.is_none()
    {
        missing(
            result,
            "insufficient server observations wholly inside the send window",
        );
    }
    if result
        .unfinished_requests_slope_per_second
        .is_some_and(|v| v > window.maximum_unfinished_requests_slope_per_second)
        || result
            .oldest_age_slope_ms_per_second
            .is_some_and(|v| v > window.maximum_oldest_age_slope_ms_per_second)
    {
        *failed = true;
        result.issues.push(
            "original server unfinished queue or ingress age grows beyond frozen tolerance".into(),
        );
    }
    if last.waiting_requests != 0
        || last.active_prefill_sequences != 0
        || last.active_decode_sequences != 0
        || last.preempted_requests != 0
    {
        *failed = true;
        result
            .issues
            .push("original server scheduler members remain after client drain".into());
    }
}
