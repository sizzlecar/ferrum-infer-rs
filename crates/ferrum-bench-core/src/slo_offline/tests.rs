use super::*;
use crate::slo::{SloStatus, TpotBoundary};
use crate::{
    compute_metrics, BenchmarkPhase, BenchmarkRequestCorrelation, Env, ItlEligibility,
    OutputTokenCountSource, QualityIssueCounts, RequestItlEvidence, RequestRecord, RunRecord, Slo,
    WarmupSummary,
};

fn config() -> Vec<u8> {
    serde_json::to_vec(&SloClientVisibleConfig {
        latency: ferrum_types::SloLatencyBudgets {
            ttft_ms: 100.try_into().unwrap(),
            tpot_ms: 20.try_into().unwrap(),
            itl_ms: 20.try_into().unwrap(),
        },
        attainment: Default::default(),
    })
    .unwrap()
}
fn request(
    index: u32,
    success: bool,
    events: u32,
    usage: Option<u32>,
    gaps: &[f64],
) -> RequestRecord {
    RequestRecord {
        benchmark_correlation: Some(
            BenchmarkRequestCorrelation::new(
                "run".into(),
                "cell".into(),
                0,
                BenchmarkPhase::Measured,
                index,
            )
            .unwrap(),
        ),
        server_request_id: Some(format!("request-{index}")),
        success,
        ttft_ms: 10.0,
        e2e_ms: 10_000.0,
        input_tokens: 8,
        server_input_tokens: Some(10),
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
fn report(records: Vec<RequestRecord>) -> BenchReport {
    let expected_requests = records.len() as u32;
    let mut report = compute_metrics(
        "fixture-model".into(),
        "http".into(),
        Scenario::ClosedLoop,
        Some(2),
        None,
        8,
        0,
        0,
        Slo::unbounded(),
        vec![RunRecord {
            records,
            expected_requests,
            duration_s: 2.0,
            warmup: WarmupSummary::default(),
        }],
        Env::default(),
    );
    report.benchmark_run_id = Some("run".into());
    report.cell_id = Some("cell".into());
    report
}
fn evaluate(report: &BenchReport) -> OfflineSloReport {
    reevaluate_json(&serde_json::to_vec(report).unwrap(), &config()).unwrap()
}

#[test]
fn raw_collector_projection_recovers_visible_span_in_canonical_report_order() {
    // bench-serve awaits its handles in request order; compute_metrics requires
    // correlation indices to match these aligned report positions. The large
    // terminal delay ensures old terminal TPOT cannot satisfy these values.
    let original = report(vec![
        request(0, true, 3, Some(3), &[2.0, 8.0]),
        request(1, true, 2, Some(6), &[30.0]),
    ]);
    let result = evaluate(&original);
    let repeat = &result.cells[0].repeats[0];
    let evaluated = repeat.evaluation.as_ref().unwrap();
    assert_eq!(
        evaluated.config.tpot_boundary,
        TpotBoundary::LastVisibleOutput
    );
    assert_eq!(
        repeat.reconstruction[0]
            .correlation
            .as_ref()
            .unwrap()
            .request_index,
        0
    );
    assert_eq!(evaluated.request_evidence[0].last_visible_ms, Some(20.0));
    assert_eq!(evaluated.requests[0].tpot.observed_ms, Some(5.0));
    assert_eq!(evaluated.requests[1].tpot.observed_ms, Some(6.0));
    assert_eq!(evaluated.successful_output.tokens_per_second, Some(4.5));
    assert_eq!(evaluated.visible_evidence.event_usage_mismatch_requests, 1);
    assert_eq!(
        evaluated.request_evidence[1]
            .strict_token_evidence
            .eligibility,
        ItlEligibility::EventUsageMismatch
    );
    assert!(original.repeat_metrics[0].tpot_ms.p50 > 1000.0);
}

#[test]
fn missing_gap_retains_stall_but_cannot_invent_last_visible_or_pass() {
    let original = report(vec![request(0, true, 4, Some(4), &[1.0, 700.0])]);
    let result = evaluate(&original);
    let evaluated = result.cells[0].repeats[0].evaluation.as_ref().unwrap();
    assert_eq!(evaluated.request_evidence[0].last_visible_ms, None);
    assert_eq!(evaluated.requests[0].tpot.status, SloStatus::Unknown);
    assert_eq!(evaluated.pooled_visible_itl.sample_count, 2);
    assert!(
        evaluated
            .pooled_visible_itl
            .observed_percentiles_ms
            .unwrap()
            .p99
            > 690.0
    );
    assert_eq!(evaluated.pooled_visible_itl.status, SloStatus::Unknown);
    assert_ne!(evaluated.latency_and_outcome_status, SloStatus::Pass);
}

#[test]
fn single_event_is_real_last_visible_but_no_interval_percentile() {
    let original = report(vec![request(0, true, 1, Some(1), &[])]);
    let result = evaluate(&original);
    let evaluated = result.cells[0].repeats[0].evaluation.as_ref().unwrap();
    assert_eq!(evaluated.request_evidence[0].last_visible_ms, Some(10.0));
    assert_eq!(evaluated.tpot.status, SloStatus::NotApplicable);
    assert_eq!(
        evaluated.pooled_visible_itl.status,
        SloStatus::NotApplicable
    );
    assert_eq!(evaluated.pooled_visible_itl.observed_percentiles_ms, None);
    assert_eq!(evaluated.tpot.observed_percentiles_ms, None);
}

#[test]
fn usage_event_mismatch_is_diagnostic_but_two_usage_claims_must_agree() {
    let mut original = report(vec![request(0, true, 2, Some(5), &[8.0])]);
    let result = evaluate(&original);
    let evaluated = result.cells[0].repeats[0].evaluation.as_ref().unwrap();
    assert_eq!(evaluated.requests[0].tpot.observed_ms, Some(2.0));
    assert_eq!(evaluated.visible_evidence.event_usage_mismatch_requests, 1);
    original.output_tokens_per_request.as_mut().unwrap()[0][0] = 2;
    assert!(
        reevaluate_json(&serde_json::to_vec(&original).unwrap(), &config())
            .unwrap_err()
            .0
            .contains("conflicting aligned usage")
    );
}

#[test]
fn failed_long_gap_is_not_a_success_survivor_or_assumed_accepted() {
    let original = report(vec![
        request(0, true, 2, Some(2), &[1.0]),
        request(1, false, 2, Some(2), &[5000.0]),
    ]);
    let result = evaluate(&original);
    let evaluated = result.cells[0].repeats[0].evaluation.as_ref().unwrap();
    assert_eq!(evaluated.outcomes.failed, 1);
    assert_eq!(evaluated.admissions.unknown, 1);
    assert_eq!(evaluated.visible_evidence.failed_or_pending_intervals, 1);
    assert!(
        evaluated
            .pooled_visible_itl
            .observed_percentiles_ms
            .unwrap()
            .p99
            > 4900.0
    );
    assert_eq!(evaluated.raw_output.tokens_per_second, Some(2.0));
    assert_eq!(evaluated.successful_output.tokens_per_second, Some(1.0));
}

#[test]
fn lost_collector_or_usage_stays_unknown_and_offered_count_is_preserved() {
    let mut original = report(vec![
        request(0, true, 2, Some(2), &[1.0]),
        request(1, true, 2, None, &[2.0]),
    ]);
    original.request_records.as_mut().unwrap()[0][0].timing = None;
    let result = evaluate(&original);
    let evaluated = result.cells[0].repeats[0].evaluation.as_ref().unwrap();
    assert_eq!(evaluated.outcomes.offered, 2);
    assert_eq!(evaluated.outcomes.pending, 1);
    assert_eq!(evaluated.requests[1].tpot.status, SloStatus::Unknown);
    assert_eq!(evaluated.successful_output.tokens_per_second, None);
    original.request_records = None;
    let result = evaluate(&original);
    let evaluated = result.cells[0].repeats[0].evaluation.as_ref().unwrap();
    assert_eq!(evaluated.outcomes.pending, 2);
    assert_eq!(evaluated.latency_and_outcome_status, SloStatus::Unknown);
}

#[test]
fn source_hashes_array_cells_and_missing_duration_are_explicit() {
    let mut original = report(vec![request(0, true, 2, Some(2), &[1.0])]);
    original.repeat_metrics.clear();
    let bytes = serde_json::to_vec(&vec![original.clone(), original]).unwrap();
    let result = reevaluate_json(&bytes, &config()).unwrap();
    assert_eq!(result.cells.len(), 2);
    assert_eq!(result.report_sha256, digest(&bytes));
    assert_eq!(result.client_config_sha256, digest(&config()));
    assert_eq!(result.source, "offline_reconstructed");
    let repeat = &result.cells[0].repeats[0];
    assert!(repeat.evaluation.is_none());
    assert!(repeat.evaluation_unavailable.is_some());
    assert_eq!(
        repeat.unevaluated_request_evidence.as_ref().unwrap()[0].last_visible_ms,
        Some(11.0)
    );
}

#[test]
fn alignment_and_duplicate_identity_fail_instead_of_dropping_rows() {
    let mut original = report(vec![
        request(0, true, 2, Some(2), &[1.0]),
        request(1, true, 2, Some(2), &[2.0]),
    ]);
    original.request_records.as_mut().unwrap()[0][1]
        .correlation
        .request_index = 0;
    assert!(reevaluate_json(&serde_json::to_vec(&original).unwrap(), &config()).is_err());
    original.request_records.as_mut().unwrap()[0][1]
        .correlation
        .request_index = 1;
    original.itl_evidence_per_request.as_mut().unwrap()[0].pop();
    assert!(reevaluate_json(&serde_json::to_vec(&original).unwrap(), &config()).is_err());
}

#[test]
fn bounded_regular_inputs_and_create_new_output_preserve_existing_artifacts() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("report.json");
    let original = report(vec![request(0, true, 2, Some(2), &[1.0])]);
    let bytes = serde_json::to_vec(&original).unwrap();
    std::fs::write(&path, &bytes).unwrap();
    assert_eq!(read_bounded(&path, bytes.len()).unwrap(), bytes);
    assert!(read_bounded(&path, bytes.len() - 1).is_err());
    assert!(write_new(&path, &evaluate(&original)).is_err());
    assert_eq!(std::fs::read(&path).unwrap(), bytes);
    let alias = dir.path().join("alias.json");
    std::fs::hard_link(&path, &alias).unwrap();
    assert!(write_new(&alias, &evaluate(&original)).is_err());
    write_new(&dir.path().join("new.json"), &evaluate(&original)).unwrap();
}
