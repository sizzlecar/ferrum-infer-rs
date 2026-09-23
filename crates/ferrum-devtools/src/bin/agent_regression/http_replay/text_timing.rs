//! Text-event latency remains separate from agent tool progress and usage tokens.
use super::{json, RequestRecord, Value};
use ferrum_bench_core::percentile;
use serde::Serialize;

#[derive(Serialize)]
pub(super) struct TextTiming {
    pub timestamps_valid: bool,
    pub event_count: usize,
    pub transport_coalesced_chunks: u64,
    pub usage_event_count_mismatch: Option<bool>,
    pub ttft_ms: Option<f64>,
    pub last_visible_tpot_ms: Option<f64>,
    pub tpot_unavailable_reason: Option<&'static str>,
    pub itl_ms: Vec<f64>,
}

pub(super) fn measure(record: &RequestRecord) -> Option<TextTiming> {
    let text = record.visible_text.as_ref()?;
    let times = &text.timestamps_ns;
    let valid = times
        .iter()
        .all(|&at| at >= record.submitted_ns && record.ended_ns.is_none_or(|end| at <= end))
        && times.windows(2).all(|pair| pair[0] <= pair[1]);
    let usage = record
        .usage
        .as_ref()
        .and_then(|usage| usage["completion_tokens"].as_u64());
    let ttft_ms = valid
        .then(|| {
            times
                .first()
                .map(|&at| (at - record.submitted_ns) as f64 / 1e6)
        })
        .flatten();
    let single_choice = text.single_choice_requested == Some(true) && !text.multi_choice_observed;
    let last_visible_tpot_ms = (valid && single_choice)
        .then(|| {
            let tokens = usage.filter(|&tokens| tokens > 1)?;
            Some((*times.last()? - *times.first()?) as f64 / 1e6 / (tokens - 1) as f64)
        })
        .flatten();
    Some(TextTiming {
        timestamps_valid: valid,
        event_count: times.len(),
        transport_coalesced_chunks: text.transport_coalesced_chunks,
        usage_event_count_mismatch: usage.map(|tokens| tokens != times.len() as u64),
        ttft_ms,
        last_visible_tpot_ms,
        tpot_unavailable_reason: if !valid {
            Some("invalid_timestamps")
        } else if !single_choice {
            Some("single_choice_usage_not_established")
        } else if usage.is_none() {
            Some("usage_output_tokens_missing")
        } else if usage.is_some_and(|tokens| tokens < 2) {
            Some("fewer_than_two_usage_tokens")
        } else if times.is_empty() {
            Some("visible_text_missing")
        } else {
            None
        },
        itl_ms: if valid {
            times
                .windows(2)
                .map(|pair| (pair[1] - pair[0]) as f64 / 1e6)
                .collect()
        } else {
            Vec::new()
        },
    })
}

fn distribution(values: &[f64]) -> Value {
    json!({
        "sample_count": values.len(),
        "p50_ms": (!values.is_empty()).then(|| percentile(values, 0.5)),
        "p99_ms": (!values.is_empty()).then(|| percentile(values, 0.99))
    })
}

pub(super) fn summarize(records: &[RequestRecord]) -> Value {
    let mut ttft = Vec::new();
    let mut tpot = Vec::new();
    let mut itl = Vec::new();
    let mut completed = 0;
    let mut missing = 0;
    let mut invalid = 0;
    let mut coalesced = 0;
    let mut mismatches = 0;
    let mut failed_intervals = 0;
    for record in records {
        let complete = record.http_status == Some(200) && record.saw_done && record.error.is_none();
        completed += usize::from(complete);
        let Some(text) = measure(record) else {
            missing += 1;
            continue;
        };
        invalid += usize::from(!text.timestamps_valid);
        coalesced += usize::from(text.transport_coalesced_chunks > 0);
        mismatches += usize::from(text.usage_event_count_mismatch == Some(true));
        // Match the serving SLO's AllOfferedObservedGaps population. A stream
        // failure cannot erase the visible stalls observed before it failed.
        if text.timestamps_valid {
            failed_intervals += if complete { 0 } else { text.itl_ms.len() };
            itl.extend(text.itl_ms.iter().copied());
        }
        if complete && text.timestamps_valid {
            ttft.extend(text.ttft_ms);
            tpot.extend(text.last_visible_tpot_ms);
        }
    }
    json!({
        "measurement": "first_choice_nonempty_content_reasoning_or_reasoning_content_sse_text_events",
        "ttft_tpot_population": "transport_completed_requests_with_valid_text_timestamps",
        "itl_population": "all_offered_observed_text_gaps_with_valid_timestamps",
        "request_count": records.len(), "completed_request_count": completed,
        "failed_request_count": records.len() - completed,
        "missing_evidence_request_count": missing, "invalid_timestamp_request_count": invalid,
        "transport_coalesced_request_count": coalesced,
        "usage_event_mismatch_request_count": mismatches,
        "failed_or_pending_itl_intervals": failed_intervals,
        "ttft": distribution(&ttft), "last_visible_tpot": distribution(&tpot),
        "itl": distribution(&itl), "strict_single_token_timing_measured": false,
        "slo_compliance": "not_evaluated"
    })
}
