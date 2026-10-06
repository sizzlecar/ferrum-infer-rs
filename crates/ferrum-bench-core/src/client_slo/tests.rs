use super::*;
use crate::{QualityIssueCounts, RequestItlEvidence, WarmupSummary};

fn request(ttft: f64, gaps: &[f64], usage: Option<u32>, coalesced: u32) -> RequestRecord {
    let events = gaps.len() as u32 + 1;
    RequestRecord {
        benchmark_correlation: None,
        server_request_id: None,
        success: true,
        ttft_ms: ttft,
        e2e_ms: ttft + gaps.iter().sum::<f64>() + 1000.0,
        input_tokens: 4,
        server_input_tokens: Some(4),
        output_tokens: usage.unwrap_or(events),
        output_token_count_source: if usage.is_some() {
            OutputTokenCountSource::Usage
        } else {
            OutputTokenCountSource::StreamChunks
        },
        itl_evidence: RequestItlEvidence::sse(true, events, usage, gaps.len() as u32, coalesced),
        quality_issues: QualityIssueCounts::default(),
        itl_ms: gaps.to_vec(),
    }
}

fn run(records: Vec<RequestRecord>) -> RunRecord {
    RunRecord {
        expected_requests: records.len() as u32,
        records,
        duration_s: 10.0,
        warmup: WarmupSummary::default(),
    }
}

fn config() -> ClientSloConfig {
    "ttft:100,tpot:100,itl:100".parse().unwrap()
}

#[test]
fn parser_requires_each_unique_positive_finite_metric() {
    for raw in [
        "ttft:1,tpot:2",
        "ttft:1,tpot:2,itl:3,itl:4",
        "ttft:NaN,tpot:2,itl:3",
        "ttft:0,tpot:2,itl:3",
        "ttft:1,tpot:2,e2e:3",
    ] {
        assert!(raw.parse::<ClientSloConfig>().is_err(), "{raw}");
    }
    assert_eq!(
        "ttft:100 tpot:100 itl:100"
            .parse::<ClientSloConfig>()
            .unwrap(),
        config()
    );
}

#[test]
fn last_visible_tpot_excludes_terminal_delay_and_uses_usage_not_event_count() {
    let record = request(5.0, &[0.0, 30.0], Some(7), 1);
    assert!(record.tpot_ms().unwrap() > 100.0);
    let timing = RequestTiming::from_record(&record);
    assert_eq!(timing.first_visible_ms, Some(5.0));
    assert_eq!(timing.last_visible_ms, Some(35.0));
    assert_eq!(timing.tpot_ms(), Some(5.0));
    let report = ClientSloReport::evaluate(config(), &[run(vec![record])]).unwrap();
    assert_eq!(report.status, ClientSloStatus::Pass);
    assert_eq!(
        report.repeats[0]
            .text_timing
            .as_ref()
            .unwrap()
            .event_usage_mismatch_requests,
        1
    );
    assert_eq!(
        report.repeats[0]
            .text_timing
            .as_ref()
            .unwrap()
            .transport_coalesced_requests,
        1
    );
    assert_eq!(
        report.repeats[0].successful_output_throughput_tps,
        Some(0.7)
    );
}

#[test]
fn coalesced_long_stall_and_failed_stream_are_never_dropped() {
    let mut failed = request(5.0, &[0.0, 5000.0], Some(8), 1);
    failed.success = false;
    let report = ClientSloReport::evaluate(
        config(),
        &[run(vec![request(1.0, &[1.0], Some(2), 0), failed])],
    )
    .unwrap();
    let row = &report.repeats[0];
    assert_eq!(row.status, ClientSloStatus::Fail);
    assert!(row.visible_sse_itl_ms.unwrap().p99 > 4000.0);
    assert_eq!(row.errors, 1);
    // Successful tokens only, divided by the entire wave including failed work.
    assert_eq!(row.successful_output_throughput_tps, Some(0.2));
    assert_eq!(row.text_timing.as_ref().unwrap().contributing_intervals, 3);
}

#[test]
fn missing_usage_gap_or_request_evidence_cannot_pass() {
    let mut missing_gap = request(5.0, &[2.0], Some(3), 0);
    missing_gap.itl_evidence.output_events = 3;
    let mut missing_request = run(vec![request(5.0, &[2.0], Some(2), 0)]);
    missing_request.expected_requests = 2;
    for run in [
        run(vec![request(5.0, &[2.0], None, 0)]),
        run(vec![missing_gap]),
        missing_request,
        run(vec![request(5.0, &[], Some(1), 0)]),
    ] {
        assert_eq!(
            ClientSloReport::evaluate(config(), &[run]).unwrap().status,
            ClientSloStatus::Unknown
        );
    }
}

#[test]
fn percentiles_are_evaluated_per_repeat_without_request_joint_gate() {
    let mut records = vec![request(1.0, &[1.0], Some(2), 0); 100];
    records[0] = request(200.0, &[1.0], Some(2), 0);
    records[1] = request(1.0, &[200.0], Some(2), 0);
    // Two distinct requests fail individual limits, while every P99 is below 100.
    let passing = run(records);
    assert_eq!(
        ClientSloReport::evaluate(config(), &[passing.clone()])
            .unwrap()
            .status,
        ClientSloStatus::Pass
    );
    let failing = run(vec![request(150.0, &[1.0], Some(2), 0)]);
    let report = ClientSloReport::evaluate(config(), &[passing, failing]).unwrap();
    assert_eq!(report.repeats[0].status, ClientSloStatus::Pass);
    assert_eq!(report.repeats[1].status, ClientSloStatus::Fail);
    assert_eq!(report.status, ClientSloStatus::Fail);
}

#[test]
fn even_one_admission_rejection_fails_independently_of_percentiles() {
    let mut rejected = request(1.0, &[], None, 0);
    rejected.success = false;
    rejected.quality_issues.http_rejected = 1;
    let report = ClientSloReport::evaluate(config(), &[run(vec![rejected])]).unwrap();
    assert_eq!(report.status, ClientSloStatus::Fail);
    assert_eq!(report.repeats[0].rejected_requests, 1);
}

#[test]
fn telescope_preserves_submillisecond_gaps() {
    let gaps = vec![0.001; 4096];
    let timing = RequestTiming::from_record(&request(5432.1, &gaps, Some(4097), 0));
    assert!((timing.last_visible_ms.unwrap() - 5436.196).abs() < 1e-9);
    assert!((timing.tpot_ms().unwrap() - 0.001).abs() < 1e-12);
}
