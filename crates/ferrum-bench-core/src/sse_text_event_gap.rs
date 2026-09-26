//! Client-observed gaps between consecutive nonempty SSE text updates.
//!
//! These samples are independent of strict token ITL: usage mismatches and
//! transport coalescing remain visible in the counts and do not erase stalls.
//! Content and reasoning text both count as updates. No event is invented for
//! an empty, role-only or finish-only frame, or for a usage token.

use crate::{
    checked_scalar_stats, repeat_percentiles, BenchRepeatMetrics, ItlEvidenceSource, MetricSet,
    RepeatPercentiles, RequestRecord,
};
use serde::{Deserialize, Serialize};

/// Audit counts for one repeat, or summed over every repeat in a report.
/// Diagnostic categories may overlap. Event/transport/usage diagnostics cover
/// every SSE request, including failures; contributors are successful requests.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SseTextEventGapEvidence {
    pub requests: u64,
    pub successful_requests: u64,
    pub failed_requests: u64,
    pub observed_text_events: u64,
    /// Declared SSE interval counts; `compute_metrics` also validates that each
    /// count equals the retained request's raw interval sample count.
    pub observed_intervals: u64,
    pub contributing_requests: u64,
    pub contributing_intervals: u64,
    pub successful_requests_without_sse_evidence: u64,
    pub fewer_than_two_events_requests: u64,
    pub interval_count_mismatch_requests: u64,
    pub transport_coalesced_requests: u64,
    pub transport_coalesced_output_chunks: u64,
    pub event_usage_mismatch_requests: u64,
    pub missing_usage_requests: u64,
    pub failed_requests_with_observed_intervals: u64,
}

impl SseTextEventGapEvidence {
    fn add_assign(&mut self, other: &Self) {
        for (destination, count) in [
            (&mut self.requests, other.requests),
            (&mut self.successful_requests, other.successful_requests),
            (&mut self.failed_requests, other.failed_requests),
            (&mut self.observed_text_events, other.observed_text_events),
            (&mut self.observed_intervals, other.observed_intervals),
            (&mut self.contributing_requests, other.contributing_requests),
            (
                &mut self.contributing_intervals,
                other.contributing_intervals,
            ),
            (
                &mut self.successful_requests_without_sse_evidence,
                other.successful_requests_without_sse_evidence,
            ),
            (
                &mut self.fewer_than_two_events_requests,
                other.fewer_than_two_events_requests,
            ),
            (
                &mut self.interval_count_mismatch_requests,
                other.interval_count_mismatch_requests,
            ),
            (
                &mut self.transport_coalesced_requests,
                other.transport_coalesced_requests,
            ),
            (
                &mut self.transport_coalesced_output_chunks,
                other.transport_coalesced_output_chunks,
            ),
            (
                &mut self.event_usage_mismatch_requests,
                other.event_usage_mismatch_requests,
            ),
            (
                &mut self.missing_usage_requests,
                other.missing_usage_requests,
            ),
            (
                &mut self.failed_requests_with_observed_intervals,
                other.failed_requests_with_observed_intervals,
            ),
        ] {
            *destination = destination
                .checked_add(count)
                .expect("SSE text event gap count overflow");
        }
    }
}

pub(crate) fn summarize_requests(
    records: &[RequestRecord],
) -> (Option<RepeatPercentiles>, Option<SseTextEventGapEvidence>) {
    if !records
        .iter()
        .any(|record| record.itl_evidence.source == ItlEvidenceSource::SseDeltaEvents)
    {
        return (None, None);
    }
    let mut evidence = SseTextEventGapEvidence {
        requests: u64::try_from(records.len()).expect("SSE request count overflow"),
        ..Default::default()
    };
    let mut samples = Vec::new();
    let mut complete_successful_timing = true;
    for record in records {
        if record.success {
            evidence.successful_requests += 1;
        } else {
            evidence.failed_requests += 1;
        }
        let timing = &record.itl_evidence;
        if timing.source != ItlEvidenceSource::SseDeltaEvents {
            if record.success {
                evidence.successful_requests_without_sse_evidence += 1;
                complete_successful_timing = false;
            }
            continue;
        }
        evidence.observed_text_events += u64::from(timing.output_events);
        evidence.observed_intervals += u64::from(timing.observed_intervals);
        evidence.fewer_than_two_events_requests += u64::from(timing.output_events < 2);
        evidence.transport_coalesced_output_chunks +=
            u64::from(timing.transport_coalesced_output_chunks);
        evidence.transport_coalesced_requests +=
            u64::from(timing.transport_coalesced_output_chunks > 0);
        evidence.missing_usage_requests += u64::from(timing.usage_output_tokens.is_none());
        evidence.event_usage_mismatch_requests += u64::from(
            timing
                .usage_output_tokens
                .is_some_and(|usage| usage != timing.output_events),
        );
        let complete = timing.observed_intervals == timing.output_events.saturating_sub(1)
            && usize::try_from(timing.observed_intervals).ok() == Some(record.itl_ms.len());
        evidence.interval_count_mismatch_requests += u64::from(!complete);
        if !record.success {
            evidence.failed_requests_with_observed_intervals +=
                u64::from(!record.itl_ms.is_empty());
            continue;
        }
        if !complete {
            // Never improve the visible-gap result by dropping an incomplete
            // successful request while keeping only the other requests.
            complete_successful_timing = false;
            continue;
        }
        if timing.output_events < 2 {
            continue;
        }
        evidence.contributing_requests += 1;
        evidence.contributing_intervals += u64::from(timing.observed_intervals);
        samples.extend_from_slice(&record.itl_ms);
    }
    let metric =
        (complete_successful_timing && !samples.is_empty()).then(|| repeat_percentiles(&samples));
    (metric, Some(evidence))
}

pub(crate) fn summarize_repeats(
    repeats: &[BenchRepeatMetrics],
) -> (Option<MetricSet>, Option<SseTextEventGapEvidence>) {
    if repeats.is_empty()
        || repeats
            .iter()
            .any(|row| row.sse_text_event_gap_evidence.is_none())
    {
        return (None, None);
    }
    let mut evidence = SseTextEventGapEvidence::default();
    for row in repeats {
        evidence.add_assign(
            row.sse_text_event_gap_evidence
                .as_ref()
                .expect("evidence checked"),
        );
    }
    // Do not silently drop repeats without any intervals, or substitute zero.
    let Some(metrics) = repeats
        .iter()
        .map(|row| row.sse_text_event_gap_ms)
        .collect::<Option<Vec<_>>>()
    else {
        return (None, Some(evidence));
    };
    let values =
        |field: fn(&RepeatPercentiles) -> f64| metrics.iter().map(field).collect::<Vec<_>>();
    let metric = MetricSet {
        p50: checked_scalar_stats(&values(|m| m.p50), "SSE text event gap p50"),
        p75: checked_scalar_stats(&values(|m| m.p75), "SSE text event gap p75"),
        p95: checked_scalar_stats(&values(|m| m.p95), "SSE text event gap p95"),
        p99: checked_scalar_stats(&values(|m| m.p99), "SSE text event gap p99"),
    };
    (Some(metric), Some(evidence))
}

#[cfg(test)]
mod tests;
