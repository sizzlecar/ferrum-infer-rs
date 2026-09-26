use crate::{
    compute_metrics, BenchReport, Env, OutputTokenCountSource, QualityIssueCounts,
    RequestItlEvidence, RequestRecord, RunRecord, Scenario, Slo, WarmupSummary,
};

fn sse(
    success: bool,
    events: u32,
    usage: Option<u32>,
    gaps: &[f64],
    coalesced: u32,
) -> RequestRecord {
    RequestRecord {
        benchmark_correlation: None,
        server_request_id: None,
        success,
        ttft_ms: 10.0,
        e2e_ms: 100.0 + gaps.iter().sum::<f64>(),
        input_tokens: 8,
        server_input_tokens: None,
        output_tokens: usage.unwrap_or(events),
        output_token_count_source: if usage.is_some() {
            OutputTokenCountSource::Usage
        } else {
            OutputTokenCountSource::StreamChunks
        },
        itl_evidence: RequestItlEvidence::sse(success, events, usage, gaps.len() as u32, coalesced),
        quality_issues: QualityIssueCounts::default(),
        itl_ms: gaps.to_vec(),
    }
}

fn report(repeats: Vec<Vec<RequestRecord>>) -> BenchReport {
    let requests = repeats[0].len() as u32;
    compute_metrics(
        "test-model".into(),
        "test-http".into(),
        Scenario::ClosedLoop,
        Some(requests),
        None,
        8,
        0,
        0,
        Slo::unbounded(),
        repeats
            .into_iter()
            .map(|records| RunRecord {
                records,
                expected_requests: requests,
                duration_s: 100.0,
                warmup: WarmupSummary::default(),
            })
            .collect(),
        Env::default(),
    )
}

#[test]
fn usage_mismatch_retains_visible_stall_without_qualifying_strict_itl() {
    let report = report(vec![vec![
        sse(true, 4, Some(9), &[10.0, 5000.0, 20.0], 0),
        sse(true, 2, Some(2), &[2.0], 0),
    ]]);
    let metric = report.sse_text_event_gap_ms.as_ref().unwrap();
    assert_eq!(metric.p50.mean, 15.0);
    assert!((metric.p99.mean - 4850.6).abs() < 1e-9);
    let evidence = report.sse_text_event_gap_evidence.as_ref().unwrap();
    assert_eq!(evidence.contributing_requests, 2);
    assert_eq!(evidence.contributing_intervals, 4);
    assert_eq!(evidence.event_usage_mismatch_requests, 1);
    assert!(!report.has_complete_itl_evidence());
    assert_eq!(report.itl_ms.p99.mean, 0.0);
    assert_eq!(
        report.repeat_metrics[0]
            .itl_eligibility_counts
            .event_usage_mismatch,
        1
    );
}

#[test]
fn coalescing_and_missing_usage_preserve_observed_zero_and_long_gaps() {
    let report = report(vec![vec![
        sse(true, 4, Some(4), &[0.0, 0.0, 8000.0], 2),
        sse(true, 2, None, &[0.0], 0),
    ]]);
    let metric = report.sse_text_event_gap_ms.as_ref().unwrap();
    assert_eq!(metric.p50.mean, 0.0);
    assert!((metric.p99.mean - 7760.0).abs() < 1e-9);
    let evidence = report.sse_text_event_gap_evidence.as_ref().unwrap();
    assert_eq!(evidence.transport_coalesced_requests, 1);
    assert_eq!(evidence.transport_coalesced_output_chunks, 2);
    assert_eq!(evidence.missing_usage_requests, 1);
    assert_eq!(evidence.observed_text_events, 6);
    assert_eq!(evidence.observed_intervals, 4);
    assert_eq!(evidence.contributing_intervals, 4);
    assert!(!report.has_complete_itl_evidence());
    assert_eq!(report.itl_ms.p99.mean, 0.0);
}

#[test]
fn partial_failed_stream_and_short_requests_remain_in_diagnostics() {
    let mut failed = sse(false, 2, Some(2), &[9000.0], 1);
    failed.quality_issues.malformed_stream = 1;
    failed.quality_issues.missing_done = 1;
    let report = report(vec![vec![
        failed,
        sse(true, 0, Some(0), &[], 0),
        sse(true, 1, Some(1), &[], 0),
        sse(true, 2, Some(2), &[25.0], 0),
    ]]);
    assert_eq!(
        report.sse_text_event_gap_ms.as_ref().unwrap().p99.mean,
        25.0
    );
    let evidence = report.sse_text_event_gap_evidence.as_ref().unwrap();
    assert_eq!(evidence.requests, 4);
    assert_eq!(evidence.successful_requests, 3);
    assert_eq!(evidence.failed_requests, 1);
    assert_eq!(evidence.failed_requests_with_observed_intervals, 1);
    assert_eq!(evidence.fewer_than_two_events_requests, 2);
    assert_eq!(evidence.observed_intervals, 2);
    assert_eq!(evidence.contributing_intervals, 1);
    assert_eq!(evidence.transport_coalesced_requests, 1);
    assert_eq!(report.repeat_metrics[0].quality_issues.malformed_stream, 1);
    assert_eq!(report.repeat_metrics[0].quality_issues.missing_done, 1);
}

#[test]
fn collected_streams_without_intervals_are_unavailable_instead_of_zero() {
    for record in [
        sse(true, 0, Some(0), &[], 0),
        sse(true, 1, Some(1), &[], 0),
        sse(false, 2, Some(2), &[5000.0], 0),
    ] {
        let report = report(vec![vec![record]]);
        assert!(report.sse_text_event_gap_ms.is_none());
        assert!(report.repeat_metrics[0].sse_text_event_gap_ms.is_none());
        let evidence = report.sse_text_event_gap_evidence.unwrap();
        assert_eq!(evidence.contributing_intervals, 0);
        assert_eq!(evidence.requests, 1);
    }
}

#[test]
fn incomplete_successful_timing_invalidates_metric_without_selective_dropping() {
    let incomplete = sse(true, 4, Some(5), &[1000.0, 5000.0], 0);
    let report = report(vec![vec![sse(true, 2, Some(2), &[1.0], 0), incomplete]]);
    assert!(report.sse_text_event_gap_ms.is_none());
    let evidence = report.sse_text_event_gap_evidence.unwrap();
    assert_eq!(evidence.interval_count_mismatch_requests, 1);
    assert_eq!(evidence.observed_intervals, 3);
    assert_eq!(evidence.contributing_intervals, 1);
    assert_eq!(evidence.successful_requests, 2);
}

#[test]
fn engine_timing_is_not_relabelled_as_sse_and_mixed_successes_are_incomplete() {
    let mut engine = sse(true, 2, Some(2), &[1000.0], 0);
    engine.itl_evidence = RequestItlEvidence::engine(true, 2, 1);
    let engine_only = report(vec![vec![engine.clone()]]);
    assert!(engine_only.has_complete_itl_evidence());
    assert_eq!(engine_only.itl_ms.p99.mean, 1000.0);
    assert!(engine_only.sse_text_event_gap_ms.is_none());
    assert!(engine_only.sse_text_event_gap_evidence.is_none());
    let mixed = report(vec![vec![engine, sse(true, 2, Some(2), &[1.0], 0)]]);
    assert!(mixed.sse_text_event_gap_ms.is_none());
    let evidence = mixed.sse_text_event_gap_evidence.unwrap();
    assert_eq!(evidence.successful_requests_without_sse_evidence, 1);
    assert_eq!(evidence.successful_requests, 2);
}

#[test]
fn repeat_percentiles_are_aggregated_without_dropping_an_empty_repeat() {
    let complete = report(vec![
        vec![sse(true, 2, Some(2), &[10.0], 0)],
        vec![sse(true, 3, Some(3), &[20.0, 40.0], 0)],
    ]);
    let metric = complete.sse_text_event_gap_ms.as_ref().unwrap();
    assert_eq!(metric.p50.mean, 20.0);
    assert!((metric.p99.mean - 24.9).abs() < 1e-9);
    assert_eq!(
        complete
            .sse_text_event_gap_evidence
            .unwrap()
            .contributing_intervals,
        3
    );
    let incomplete = report(vec![
        vec![sse(true, 2, Some(2), &[10.0], 0)],
        vec![sse(true, 1, Some(1), &[], 0)],
    ]);
    assert!(incomplete.sse_text_event_gap_ms.is_none());
    assert!(incomplete.repeat_metrics[0].sse_text_event_gap_ms.is_some());
    assert!(incomplete.repeat_metrics[1].sse_text_event_gap_ms.is_none());
    let evidence = incomplete.sse_text_event_gap_evidence.unwrap();
    assert_eq!(evidence.requests, 2);
    assert_eq!(evidence.contributing_intervals, 1);
    assert_eq!(evidence.fewer_than_two_events_requests, 1);
}

#[test]
fn old_json_does_not_backfill_visible_gap_from_strict_itl() {
    let current = report(vec![vec![sse(true, 2, Some(2), &[42.0], 0)]]);
    let mut json = serde_json::to_value(&current).unwrap();
    let restored: BenchReport = serde_json::from_value(json.clone()).unwrap();
    assert_eq!(restored.sse_text_event_gap_ms.unwrap().p99.mean, 42.0);
    assert_eq!(
        restored.sse_text_event_gap_evidence,
        current.sse_text_event_gap_evidence
    );
    for field in ["sse_text_event_gap_ms", "sse_text_event_gap_evidence"] {
        json.as_object_mut().unwrap().remove(field);
        for repeat in json["repeat_metrics"].as_array_mut().unwrap() {
            repeat.as_object_mut().unwrap().remove(field);
        }
    }
    let old: BenchReport = serde_json::from_value(json).unwrap();
    assert!(old.has_complete_itl_evidence());
    assert_eq!(old.itl_ms.p99.mean, 42.0);
    assert!(old.sse_text_event_gap_ms.is_none());
    assert!(old.sse_text_event_gap_evidence.is_none());
    assert!(old.repeat_metrics[0].sse_text_event_gap_ms.is_none());
    assert!(old.repeat_metrics[0].sse_text_event_gap_evidence.is_none());
}
