use super::*;

fn sse(
    success: bool,
    ttft_ms: f64,
    e2e_ms: f64,
    events: u32,
    usage: Option<u32>,
    gaps: &[f64],
) -> RequestRecord {
    RequestRecord {
        benchmark_correlation: None,
        server_request_id: None,
        success,
        ttft_ms,
        e2e_ms,
        input_tokens: 8,
        server_input_tokens: Some(20),
        output_tokens: usage.unwrap_or(events),
        output_token_count_source: if usage.is_some() {
            OutputTokenCountSource::Usage
        } else {
            OutputTokenCountSource::StreamChunks
        },
        itl_evidence: RequestItlEvidence::sse(success, events, usage, gaps.len() as u32, 0),
        quality_issues: QualityIssueCounts::default(),
        itl_ms: gaps.to_vec(),
    }
}

fn report(repeats: Vec<Vec<RequestRecord>>) -> BenchReport {
    let expected_requests = repeats[0].len() as u32;
    let runs = repeats
        .into_iter()
        .enumerate()
        .map(|(repeat, mut records)| {
            for (index, record) in records.iter_mut().enumerate() {
                record.benchmark_correlation = Some(
                    BenchmarkRequestCorrelation::new(
                        "timing-audit".into(),
                        "cell-c4".into(),
                        repeat as u32,
                        BenchmarkPhase::Measured,
                        index as u32,
                    )
                    .unwrap(),
                );
                record.server_request_id = Some(format!("server-{repeat}-{index}"));
            }
            RunRecord {
                records,
                expected_requests,
                duration_s: 100.0,
                warmup: WarmupSummary::default(),
            }
        })
        .collect();
    compute_metrics(
        "model".into(),
        "http".into(),
        Scenario::ClosedLoop,
        Some(4),
        None,
        8,
        0,
        0,
        Slo::unbounded(),
        runs,
        Env::default(),
    )
}

// Reconstruct from serialized request evidence, without invoking the metric
// builder or its percentile helper. This checks that no required observations
// were lost, reordered or filtered before reaching the report.
fn assert_percentiles(mut samples: Vec<f64>, actual: RepeatPercentiles) {
    assert!(!samples.is_empty());
    samples.sort_by(f64::total_cmp);
    for (q, reported) in [
        (0.50, actual.p50),
        (0.75, actual.p75),
        (0.95, actual.p95),
        (0.99, actual.p99),
    ] {
        let position = q * (samples.len() - 1) as f64;
        let lower = position.floor() as usize;
        let upper = position.ceil() as usize;
        let expected =
            samples[lower] + (position - lower as f64) * (samples[upper] - samples[lower]);
        assert!(
            (reported - expected).abs() < 1e-9,
            "{reported} != {expected}"
        );
    }
}

#[test]
fn serialized_request_timings_reconstruct_repeat_metrics_without_dropping_stalls() {
    let repeats = [1.0, 2.0]
        .into_iter()
        .map(|scale| {
            let mut mismatch = sse(
                true,
                41.0 * scale,
                5141.0 * scale,
                4,
                Some(9),
                &[0.0, 5000.0 * scale, 2.0 * scale],
            );
            mismatch.itl_evidence.transport_coalesced_output_chunks = 1;
            vec![
                sse(
                    true,
                    11.0 * scale,
                    51.0 * scale,
                    3,
                    Some(3),
                    &[5.0 * scale, 25.0 * scale],
                ),
                mismatch,
                sse(true, 7.0 * scale, 9.0 * scale, 1, Some(1), &[]),
                sse(false, 10000.0, 30000.0, 2, Some(4), &[10000.0]),
                sse(false, 0.0, 0.0, 0, None, &[]),
            ]
        })
        .collect();
    let original = report(repeats);
    let serialized = serde_json::to_string(&original).unwrap();
    let restored: BenchReport = serde_json::from_str(&serialized).unwrap();
    assert_eq!(restored.request_records, original.request_records);
    let records = restored.request_records.as_ref().unwrap();
    let tokens = restored.output_tokens_per_request.as_ref().unwrap();
    let events = restored.itl_evidence_per_request.as_ref().unwrap();
    for (repeat, requests) in records.iter().enumerate() {
        let mut ttft = Vec::new();
        let mut e2e = Vec::new();
        let mut tpot = Vec::new();
        let mut gaps = Vec::new();
        for (index, request) in requests.iter().enumerate() {
            assert_eq!(request.correlation.repeat_index as usize, repeat);
            assert_eq!(request.correlation.request_index as usize, index);
            assert_eq!(
                request.server_request_id,
                Some(format!("server-{repeat}-{index}"))
            );
            let timing = request.timing.as_ref().unwrap();
            assert_eq!(timing.event_source, events[repeat][index].source);
            assert_eq!(
                timing.raw_event_gaps_ms.len(),
                events[repeat][index].observed_intervals as usize
            );
            if !timing.success {
                continue;
            }
            ttft.push(timing.reported_ttft_ms);
            e2e.push(timing.reported_e2e_ms);
            if tokens[repeat][index] >= 2 {
                tpot.push(
                    (timing.reported_e2e_ms - timing.reported_ttft_ms)
                        / f64::from(tokens[repeat][index] - 1),
                );
            }
            assert_eq!(timing.event_source, ItlEvidenceSource::SseDeltaEvents);
            assert_eq!(
                events[repeat][index].observed_intervals,
                events[repeat][index].output_events.saturating_sub(1)
            );
            gaps.extend_from_slice(&timing.raw_event_gaps_ms);
        }
        let summary = &restored.repeat_metrics[repeat];
        assert_percentiles(ttft, summary.ttft_ms);
        assert_percentiles(e2e, summary.e2e_ms);
        assert_percentiles(tpot, summary.tpot_ms);
        assert_percentiles(gaps, summary.sse_text_event_gap_ms.unwrap());
        assert_eq!(summary.completed_requests, 3);
        assert_eq!(summary.errored_requests, 2);
        assert_eq!(
            summary
                .sse_text_event_gap_evidence
                .as_ref()
                .unwrap()
                .contributing_intervals,
            5
        );
        let mismatch = requests[1].timing.as_ref().unwrap();
        assert_eq!(
            mismatch.raw_event_gaps_ms,
            [0.0, 5000.0 * (repeat + 1) as f64, 2.0 * (repeat + 1) as f64]
        );
        assert_eq!(
            events[repeat][1].eligibility,
            ItlEligibility::EventUsageMismatch
        );
        assert_eq!(
            requests[3].timing.as_ref().unwrap().raw_event_gaps_ms,
            [10000.0]
        );
        assert_eq!(
            requests[3].timing.as_ref().unwrap().observed_first_output,
            Some(true)
        );
        assert_eq!(
            requests[4].timing.as_ref().unwrap().observed_first_output,
            Some(false)
        );
    }
    assert!(!restored.has_complete_itl_evidence());
}

#[test]
fn reported_fallbacks_do_not_claim_observed_first_output() {
    let mut unknown = sse(false, 0.0, 0.0, 0, None, &[]);
    unknown.itl_evidence = RequestItlEvidence::default();
    // Match bench-serve's join_failed_record: the task did not return its
    // collector, so SSE + zero events is a placeholder, not evidence that
    // no text had arrived before the panic.
    let mut join_failed = sse(false, 0.0, 0.0, 0, None, &[]);
    join_failed.output_token_count_source = OutputTokenCountSource::None;
    join_failed.quality_issues.panic = 1;
    let result = report(vec![vec![
        sse(false, 0.0, 125.0, 0, None, &[]),
        join_failed,
        sse(true, 60.0, 60.0, 0, Some(2), &[]),
        unknown,
    ]]);
    let requests = &result.request_records.as_ref().unwrap()[0];
    for (request, observed) in requests[..3].iter().zip([Some(false), None, Some(false)]) {
        let timing = request.timing.as_ref().unwrap();
        assert_eq!(timing.observed_first_output, observed);
        assert!(timing.raw_event_gaps_ms.is_empty());
    }
    assert_eq!(requests[0].timing.as_ref().unwrap().reported_e2e_ms, 125.0);
    assert_eq!(requests[1].timing.as_ref().unwrap().reported_e2e_ms, 0.0);
    assert_eq!(requests[1].timing.as_ref().unwrap().reported_ttft_ms, 0.0);
    assert!(!requests[1].timing.as_ref().unwrap().success);
    assert_eq!(
        requests[1].timing.as_ref().unwrap().event_source,
        ItlEvidenceSource::SseDeltaEvents
    );
    assert_eq!(result.repeat_metrics[0].quality_issues.panic, 1);
    assert_eq!(requests[2].timing.as_ref().unwrap().reported_ttft_ms, 60.0);
    assert_eq!(
        requests[3].timing.as_ref().unwrap().observed_first_output,
        None
    );
    // Existing success/fallback aggregation is preserved, not silently fixed.
    assert_eq!(result.repeat_metrics[0].ttft_ms.p99, 60.0);
    assert!(result.sse_text_event_gap_ms.is_none());
}

#[test]
fn raw_engine_and_incomplete_sse_intervals_keep_their_source_and_availability() {
    let mut engine = sse(true, 10.0, 1010.0, 2, Some(2), &[1000.0]);
    engine.itl_evidence = RequestItlEvidence::engine(true, 2, 1);
    let result = report(vec![vec![engine]]);
    let timing = result.request_records.as_ref().unwrap()[0][0]
        .timing
        .as_ref()
        .unwrap();
    assert_eq!(timing.event_source, ItlEvidenceSource::EngineTokenEvents);
    assert_eq!(timing.observed_first_output, Some(true));
    assert_eq!(timing.raw_event_gaps_ms, [1000.0]);
    assert!(result.sse_text_event_gap_ms.is_none());

    let incomplete = report(vec![vec![sse(
        true,
        10.0,
        6010.0,
        4,
        Some(4),
        &[1000.0, 5000.0],
    )]]);
    assert_eq!(
        incomplete.request_records.as_ref().unwrap()[0][0]
            .timing
            .as_ref()
            .unwrap()
            .raw_event_gaps_ms,
        [1000.0, 5000.0]
    );
    assert!(incomplete.sse_text_event_gap_ms.is_none());
    assert_eq!(
        incomplete
            .sse_text_event_gap_evidence
            .as_ref()
            .unwrap()
            .interval_count_mismatch_requests,
        1
    );
}

#[test]
fn old_report_json_keeps_missing_request_timing_absent() {
    let current = report(vec![vec![sse(true, 10.0, 52.0, 2, Some(2), &[42.0])]]);
    let mut json = serde_json::to_value(&current).unwrap();
    json["request_records"][0][0]
        .as_object_mut()
        .unwrap()
        .remove("timing");
    let legacy: BenchReport = serde_json::from_value(json).unwrap();
    assert!(legacy.request_records.as_ref().unwrap()[0][0]
        .timing
        .is_none());
    assert_eq!(legacy.repeat_metrics, current.repeat_metrics);
    let saved = serde_json::to_value(&legacy).unwrap();
    assert!(saved["request_records"][0][0].get("timing").is_none());
    assert_eq!(
        saved["request_records"][0][0]["server_request_id"],
        "server-0-0"
    );
}
