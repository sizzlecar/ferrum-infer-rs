use super::*;
use crate::{ItlEligibility, QualityIssueCounts};

fn config() -> SloEvaluationConfig {
    SloEvaluationConfig {
        ttft_ms: 100.0,
        tpot_ms: 20.0,
        visible_itl_ms: 20.0,
        attainment: SloAttainmentTargets {
            min_accepted_joint_attainment: 0.99,
            min_offered_joint_attainment: Some(0.99),
            max_reject_rate: 0.01,
            ..Default::default()
        },
        tpot_boundary: TpotBoundary::LastVisibleOutput,
    }
}

fn text_request(first_ms: f64, gaps: &[f64]) -> RequestSloEvidence {
    let events = gaps.len() as u32 + 1;
    let last_ms = first_ms + gaps.iter().sum::<f64>();
    RequestSloEvidence {
        outcome: RequestOutcome::Completed,
        admission: AdmissionEvidence::Accepted,
        first_visible_ms: Some(first_ms),
        last_visible_ms: Some(last_ms),
        terminal_ms: Some(last_ms + 1.0),
        usage_output_tokens: Some(events),
        raw_output_count: RawOutputCount {
            count: events,
            source: RawOutputCountSource::Usage,
        },
        visible_text: Some(VisibleTextEvidence {
            output_events: events,
            gaps_ms: gaps.to_vec(),
            transport_coalesced_output_chunks: 0,
        }),
        strict_token_evidence: RequestItlEvidence::sse(true, events, Some(events), events - 1, 0),
    }
}

fn legacy(record: &RequestSloEvidence) -> RequestRecord {
    RequestRecord {
        benchmark_correlation: None,
        server_request_id: None,
        success: record.outcome == RequestOutcome::Completed,
        ttft_ms: record.first_visible_ms.unwrap_or_default(),
        e2e_ms: record.terminal_ms.unwrap_or_default(),
        input_tokens: 4,
        server_input_tokens: Some(8),
        output_tokens: record.raw_output_count.count,
        output_token_count_source: OutputTokenCountSource::Usage,
        itl_evidence: record.strict_token_evidence.clone(),
        quality_issues: QualityIssueCounts::default(),
        itl_ms: record.visible_text.as_ref().unwrap().gaps_ms.clone(),
    }
}

#[test]
fn terminal_and_last_visible_tpot_are_explicit_and_legacy_is_unchanged() {
    let mut evidence = text_request(10.0, &[10.0, 10.0]);
    evidence.terminal_ms = Some(110.0);
    let old = legacy(&evidence);
    assert_eq!(old.tpot_ms(), Some(50.0));
    let mut cfg = config();
    let visible = evaluate_slo(&cfg, &[evidence.clone()], 1.0).unwrap();
    assert_eq!(visible.requests[0].tpot.observed_ms, Some(10.0));
    assert_eq!(visible.latency_and_outcome_status, SloStatus::Pass);
    cfg.tpot_boundary = TpotBoundary::LegacyTerminal;
    let terminal = evaluate_slo(&cfg, &[evidence], 1.0).unwrap();
    assert_eq!(terminal.requests[0].tpot.observed_ms, old.tpot_ms());
    assert_eq!(terminal.requests[0].joint, SloStatus::Fail);

    cfg.tpot_boundary = TpotBoundary::LastVisibleOutput;
    let adapted = RequestSloEvidence::from_legacy_record(&old, None);
    assert_eq!(adapted.last_visible_ms, None);
    let unknown = evaluate_slo(&cfg, &[adapted], 1.0).unwrap();
    assert_eq!(unknown.requests[0].tpot.status, SloStatus::Unknown);
    assert_eq!(
        unknown.requests[0].tpot.issue,
        Some(EvidenceIssue::MissingLastVisibleOutput)
    );
    assert_eq!(unknown.slo_output.tokens_per_second, None);
    assert_eq!(old.tpot_ms(), Some(50.0));
}

#[test]
fn pooled_itl_does_not_replace_request_max_gap_in_joint_attainment() {
    // One long response dominates the pooled distribution. The short response
    // still contributes one failed request to the joint denominator.
    let long = text_request(1.0, &vec![1.0; 1000]);
    let stalled = text_request(1.0, &[200.0]);
    let report = evaluate_slo(&config(), &[long, stalled], 20.0).unwrap();
    assert_eq!(report.pooled_visible_itl.observed_percentile_ms, Some(1.0));
    assert_eq!(report.pooled_visible_itl.status, SloStatus::Pass);
    assert_eq!(report.request_max_visible_itl.sample_count, 2);
    assert_eq!(report.request_max_visible_itl.status, SloStatus::Fail);
    assert_eq!(report.offered_joint.pass, 1);
    assert_eq!(report.offered_joint.fail, 1);
    assert_eq!(report.offered_joint_attainment_lower_bound, 0.5);
    assert_eq!(report.offered_joint_attainment_status, SloStatus::Fail);
}

#[test]
fn failed_and_pending_streams_keep_observed_stalls_in_the_primary_itl_distribution() {
    let good = text_request(1.0, &[1.0]);
    let mut stalled = text_request(1.0, &[1_000.0]);
    stalled.outcome = RequestOutcome::Failed;
    let report = evaluate_slo(&config(), &[good.clone(), stalled.clone()], 2.0).unwrap();
    assert_eq!(
        report.pooled_visible_itl.population,
        LatencyPopulation::AllOfferedObservedGaps
    );
    assert_eq!(report.pooled_visible_itl.sample_count, 2);
    assert!(report.pooled_visible_itl.observed_percentile_ms.unwrap() > 900.0);
    assert_eq!(report.pooled_visible_itl.status, SloStatus::Fail);
    assert_eq!(report.requests[1].joint, SloStatus::Fail);
    assert_eq!(report.ttft.population, LatencyPopulation::CompletedRequests);
    assert_eq!(
        report.request_max_visible_itl.population,
        LatencyPopulation::CompletedRequests
    );
    assert_eq!(report.request_max_visible_itl.sample_count, 1);

    stalled.outcome = RequestOutcome::Pending;
    stalled.terminal_ms = None;
    let pending = evaluate_slo(&config(), &[good, stalled], 2.0).unwrap();
    assert_eq!(pending.pooled_visible_itl.sample_count, 2);
    assert!(pending.pooled_visible_itl.observed_percentile_ms.unwrap() > 900.0);
    assert_eq!(pending.pooled_visible_itl.request_evidence.unknown, 1);
    assert_eq!(pending.pooled_visible_itl.status, SloStatus::Unknown);
    assert_ne!(pending.requests[1].joint, SloStatus::Pass);
}

#[test]
fn mismatch_and_coalescing_do_not_discard_visible_stalls() {
    let mut record = text_request(2.0, &[0.0, 700.0, 0.0]);
    record.usage_output_tokens = Some(9);
    record.raw_output_count.count = 9;
    record
        .visible_text
        .as_mut()
        .unwrap()
        .transport_coalesced_output_chunks = 2;
    record.strict_token_evidence = RequestItlEvidence::sse(true, 4, Some(9), 3, 2);
    assert_eq!(
        record.strict_token_evidence.eligibility,
        ItlEligibility::EventUsageMismatch
    );
    let report = evaluate_slo(&config(), &[record.clone()], 2.0).unwrap();
    assert_eq!(report.pooled_visible_itl.sample_count, 3);
    assert_eq!(
        report.requests[0].request_max_visible_gap.observed_ms,
        Some(700.0)
    );
    assert_eq!(report.requests[0].joint, SloStatus::Fail);
    assert_eq!(report.visible_evidence.event_usage_mismatch_requests, 1);
    assert_eq!(report.visible_evidence.transport_coalesced_requests, 1);
    assert_eq!(report.request_evidence, [record]);
}

#[test]
fn single_token_is_not_applicable_and_has_no_fake_zero_percentile() {
    let report = evaluate_slo(&config(), &[text_request(3.0, &[])], 1.0).unwrap();
    assert_eq!(report.requests[0].tpot.status, SloStatus::NotApplicable);
    assert_eq!(
        report.requests[0].request_max_visible_gap.status,
        SloStatus::NotApplicable
    );
    assert_eq!(report.tpot.observed_percentiles_ms, None);
    assert_eq!(report.pooled_visible_itl.observed_percentile_ms, None);
    assert_eq!(report.pooled_visible_itl.status, SloStatus::NotApplicable);
    assert_eq!(report.requests[0].joint, SloStatus::Pass);
    assert_eq!(report.slo_output.tokens_per_second, Some(1.0));
}

#[test]
fn failed_rejected_and_pending_requests_remain_in_the_offered_denominator() {
    let good = text_request(1.0, &[1.0]);
    let mut failed = good.clone();
    failed.outcome = RequestOutcome::Failed;
    let mut rejected = good.clone();
    rejected.outcome = RequestOutcome::Rejected;
    rejected.admission = AdmissionEvidence::Rejected;
    let mut pending = good.clone();
    pending.outcome = RequestOutcome::Pending;
    pending.terminal_ms = None;
    let report = evaluate_slo(&config(), &[good, failed, rejected, pending], 2.0).unwrap();
    assert_eq!(
        report.outcomes,
        OutcomeCounts {
            offered: 4,
            completed: 1,
            failed: 1,
            rejected: 1,
            pending: 1,
        }
    );
    assert_eq!(report.offered_joint.pass, 1);
    assert_eq!(report.offered_joint.fail, 2);
    assert_eq!(report.offered_joint.unknown, 1);
    assert_eq!(report.offered_joint_attainment_lower_bound, 0.25);
    assert_eq!(
        report.accepted_joint_attainment_lower_bound,
        Some(1.0 / 3.0)
    );
    assert_eq!(report.admissions.accepted, 3);
    assert_eq!(report.success_rate_lower_bound, 0.25);
    assert_eq!(report.reject_rate, 0.25);
    assert_eq!(report.ttft.sample_count, 1);
    assert_eq!(report.visible_evidence.failed_or_pending_intervals, 3);
    assert_eq!(report.confirmed_request_goodput_rps, 0.5);
    assert_eq!(report.slo_output.known_tokens, 2);
    assert_eq!(report.slo_output.unknown_eligibility_requests, 1);
    assert_eq!(report.slo_output.tokens_per_second, None);
    assert_eq!(report.latency_and_outcome_status, SloStatus::Fail);
}

#[test]
fn raw_successful_and_slo_output_rates_have_distinct_token_sets() {
    let good = text_request(1.0, &[1.0, 1.0]);
    let late = text_request(101.0, &[1.0]);
    let mut failed = text_request(1.0, &[1.0, 1.0, 1.0]);
    failed.outcome = RequestOutcome::Failed;
    let report = evaluate_slo(&config(), &[good, late, failed], 2.0).unwrap();
    assert_eq!(report.raw_output.tokens_per_second, Some(4.5));
    assert_eq!(report.successful_output.tokens_per_second, Some(2.5));
    assert_eq!(report.slo_output.tokens_per_second, Some(1.5));
    assert_eq!(report.raw_counts_by_source[&RawOutputCountSource::Usage], 9);
}

#[test]
fn event_counts_do_not_become_usage_throughput_when_usage_is_missing() {
    let mut record = text_request(1.0, &[1.0, 1.0]);
    record.usage_output_tokens = None;
    record.raw_output_count.source = RawOutputCountSource::TextEvents;
    record.strict_token_evidence = RequestItlEvidence::sse(true, 3, None, 2, 0);
    let report = evaluate_slo(&config(), &[record], 1.0).unwrap();
    assert_eq!(report.pooled_visible_itl.status, SloStatus::Pass);
    assert_eq!(report.requests[0].tpot.status, SloStatus::Unknown);
    assert_eq!(report.requests[0].joint, SloStatus::Unknown);
    assert_eq!(report.raw_output.known_tokens, 0);
    assert_eq!(report.raw_output.tokens_per_second, None);
    assert_eq!(
        report.raw_counts_by_source[&RawOutputCountSource::TextEvents],
        3
    );
    assert_eq!(report.successful_output.unknown_count_requests, 1);
    assert_eq!(report.successful_output.tokens_per_second, None);
    assert_eq!(report.slo_output.unknown_eligibility_requests, 1);
    assert_eq!(report.slo_output.tokens_per_second, None);
}

#[test]
fn incomplete_visible_evidence_keeps_raw_stalls_and_cannot_pass() {
    let mut incomplete = text_request(1.0, &[10.0, 10.0]);
    incomplete.visible_text.as_mut().unwrap().output_events = 5;
    let report = evaluate_slo(&config(), &[incomplete.clone()], 1.0).unwrap();
    assert_eq!(
        report.requests[0].request_max_visible_gap.issue,
        Some(EvidenceIssue::IncompleteVisibleIntervals)
    );
    assert_eq!(report.pooled_visible_itl.sample_count, 2);
    assert_eq!(report.pooled_visible_itl.status, SloStatus::Unknown);
    assert_eq!(report.request_max_visible_itl.observed_percentiles_ms, None);
    assert_eq!(report.request_evidence, [incomplete]);
    assert_eq!(report.offered_joint.pass, 0);
}

#[test]
fn completed_zero_output_and_unobserved_first_text_are_not_successes() {
    let mut no_tokens = text_request(1.0, &[]);
    no_tokens.usage_output_tokens = Some(0);
    no_tokens.raw_output_count.count = 0;
    no_tokens.first_visible_ms = None;
    no_tokens.last_visible_ms = None;
    no_tokens.visible_text.as_mut().unwrap().output_events = 0;
    let mut no_text_evidence = text_request(1.0, &[1.0]);
    no_text_evidence.first_visible_ms = None;
    no_text_evidence.last_visible_ms = None;
    no_text_evidence.visible_text = None;
    let report = evaluate_slo(&config(), &[no_tokens, no_text_evidence], 1.0).unwrap();
    assert_eq!(report.requests[0].task_success, SloStatus::Fail);
    assert_eq!(report.requests[1].task_success, SloStatus::Unknown);
    assert_eq!(report.task_success.pass, 0);
    assert_eq!(report.offered_joint.pass, 0);
    assert_eq!(report.ttft.status, SloStatus::Unknown);
    assert_eq!(report.ttft.observed_percentiles_ms, None);
}

#[test]
fn no_completed_requests_never_receive_zero_latency_pass() {
    let mut failed = text_request(2.0, &[900.0]);
    failed.outcome = RequestOutcome::Failed;
    let report = evaluate_slo(&config(), &[failed], 1.0).unwrap();
    for metric in [&report.ttft, &report.tpot, &report.request_max_visible_itl] {
        assert_eq!(metric.status, SloStatus::Unknown);
        assert_eq!(metric.observed_percentiles_ms, None);
        assert_eq!(metric.sample_count, 0);
    }
    assert_eq!(report.pooled_visible_itl.status, SloStatus::Fail);
    assert_eq!(
        report.pooled_visible_itl.observed_percentile_ms,
        Some(900.0)
    );
    assert_eq!(report.pooled_visible_itl.sample_count, 1);
    assert_eq!(report.latency_and_outcome_status, SloStatus::Fail);
    assert_eq!(report.successful_output.tokens_per_second, Some(0.0));
}

#[test]
fn serde_roundtrip_preserves_versioned_evidence_and_legacy_schema() {
    let report = evaluate_slo(&config(), &[text_request(1.0, &[0.0, 800.0])], 2.0).unwrap();
    let encoded = serde_json::to_string(&report).unwrap();
    let decoded: SloEvaluationReport = serde_json::from_str(&encoded).unwrap();
    assert_eq!(decoded, report);
    assert_eq!(decoded.schema_version, 1);
    let old: crate::Slo =
        serde_json::from_str(r#"{"ttft_p99_ms":500.0,"tpot_p99_ms":50.0,"e2e_p99_ms":30000.0}"#)
            .unwrap();
    assert_eq!(old.e2e_p99_ms, 30000.0);
    let old_json = serde_json::to_value(old).unwrap();
    assert_eq!(old_json.as_object().unwrap().len(), 3);
    assert!(old_json.get("visible_itl_ms").is_none());
}

#[test]
fn configuration_and_numeric_evidence_must_be_finite_and_in_range() {
    let baseline = config();
    for value in [f64::NAN, f64::INFINITY, -1.0, 0.0] {
        let mut cfg = baseline.clone();
        cfg.ttft_ms = value;
        assert!(cfg.validate().is_err());
        cfg = baseline.clone();
        cfg.tpot_ms = value;
        assert!(cfg.validate().is_err());
        cfg = baseline.clone();
        cfg.visible_itl_ms = value;
        assert!(cfg.validate().is_err());
    }
    for value in [f64::NAN, f64::INFINITY, -0.1, 1.1] {
        let mut cfg = baseline.clone();
        cfg.attainment.min_accepted_joint_attainment = value;
        assert!(cfg.validate().is_err());
        cfg = baseline.clone();
        cfg.attainment.min_offered_joint_attainment = Some(value);
        assert!(cfg.validate().is_err());
        cfg = baseline.clone();
        cfg.attainment.max_error_rate = value;
        assert!(cfg.validate().is_err());
        cfg = baseline.clone();
        cfg.attainment.max_reject_rate = value;
        assert!(cfg.validate().is_err());
    }
    for value in [f64::NAN, f64::INFINITY, -0.1, 0.0, 100.1] {
        let mut cfg = baseline.clone();
        cfg.attainment.ttft_percentile = value;
        assert!(cfg.validate().is_err());
        cfg = baseline.clone();
        cfg.attainment.tpot_percentile = value;
        assert!(cfg.validate().is_err());
        cfg = baseline.clone();
        cfg.attainment.itl_percentile = value;
        assert!(cfg.validate().is_err());
    }
    for value in [f64::NAN, f64::INFINITY, -1.0, 0.0] {
        assert!(evaluate_slo(&baseline, &[text_request(1.0, &[1.0])], value).is_err());
    }
    assert!(evaluate_slo(&baseline, &[], 1.0).is_err());
    let mut record = text_request(1.0, &[1.0]);
    record.visible_text.as_mut().unwrap().gaps_ms[0] = f64::NAN;
    assert!(evaluate_slo(&baseline, &[record], 1.0).is_err());
    let mut record = text_request(1.0, &[1.0]);
    record.last_visible_ms = Some(0.5);
    assert!(evaluate_slo(&baseline, &[record], 1.0).is_err());
    let mut record = text_request(1.0, &[1.0]);
    record.raw_output_count.count = 3;
    assert!(evaluate_slo(&baseline, &[record], 1.0).is_err());
}

#[test]
fn an_unknown_joint_cannot_pass_even_when_the_known_lower_bound_meets_target() {
    let good = text_request(1.0, &[1.0]);
    let mut unknown = good.clone();
    unknown.last_visible_ms = None;
    let mut cfg = config();
    cfg.attainment.min_offered_joint_attainment = Some(0.5);
    cfg.attainment.min_accepted_joint_attainment = 0.5;
    let report = evaluate_slo(&cfg, &[good, unknown], 1.0).unwrap();
    assert_eq!(report.offered_joint_attainment_status, SloStatus::Unknown);
    assert_eq!(report.accepted_joint_attainment_status, SloStatus::Unknown);
    assert_eq!(report.offered_joint.pass, 1);
    assert_eq!(report.offered_joint.unknown, 1);
    assert_eq!(report.offered_joint_attainment_lower_bound, 0.5);
    assert_eq!(report.tpot.status, SloStatus::Unknown);
    assert_eq!(report.latency_and_outcome_status, SloStatus::Unknown);
}

#[test]
fn accepted_joint_requires_explicit_admission_evidence() {
    let accepted = text_request(1.0, &[1.0]);
    let mut unknown = accepted.clone();
    unknown.admission = AdmissionEvidence::Unknown;
    let report = evaluate_slo(&config(), &[accepted, unknown], 1.0).unwrap();
    assert_eq!(report.admissions.accepted, 1);
    assert_eq!(report.admissions.unknown, 1);
    assert_eq!(report.offered_joint_attainment_lower_bound, 1.0);
    assert_eq!(report.offered_joint_attainment_status, SloStatus::Pass);
    assert_eq!(report.accepted_joint_attainment_lower_bound, Some(1.0));
    assert_eq!(report.accepted_joint_attainment_status, SloStatus::Unknown);
    assert_eq!(report.latency_and_outcome_status, SloStatus::Unknown);

    let old = legacy(&text_request(1.0, &[1.0]));
    let adapted = RequestSloEvidence::from_legacy_record(&old, Some(2.0));
    assert_eq!(adapted.admission, AdmissionEvidence::Accepted);
    let report = evaluate_slo(&config(), &[adapted], 1.0).unwrap();
    assert_eq!(report.accepted_joint_attainment_lower_bound, Some(1.0));
    assert_eq!(report.accepted_joint_attainment_status, SloStatus::Pass);
    let mut failed = old;
    failed.success = false;
    assert_eq!(
        RequestSloEvidence::from_legacy_record(&failed, Some(2.0)).admission,
        AdmissionEvidence::Unknown
    );
}

#[test]
fn client_contract_reuses_percentile_units_scopes_and_attainment_targets() {
    use ferrum_types::SloLatencyBudgets;
    use std::num::NonZeroU64;

    let client = SloClientVisibleConfig {
        latency: SloLatencyBudgets {
            ttft_ms: NonZeroU64::new(100).unwrap(),
            tpot_ms: NonZeroU64::new(1000).unwrap(),
            itl_ms: NonZeroU64::new(20).unwrap(),
        },
        attainment: SloAttainmentTargets {
            ttft_percentile: 50.0,
            tpot_percentile: 75.0,
            itl_percentile: 99.0,
            min_accepted_joint_attainment: 0.5,
            min_offered_joint_attainment: Some(0.5),
            ..Default::default()
        },
    };
    let mut cfg = SloEvaluationConfig::from_client_visible(&client).unwrap();
    assert_eq!(cfg.attainment, client.attainment);
    assert_eq!(cfg.tpot_boundary, TpotBoundary::LastVisibleOutput);
    let records = [
        text_request(1.0, &vec![1.0; 1000]),
        text_request(50.0, &[200.0]),
    ];
    let pooled = evaluate_slo(&cfg, &records, 2.0).unwrap();
    assert_eq!(pooled.ttft.evaluated_percentile, 50.0);
    assert_eq!(pooled.ttft.observed_percentile_ms, Some(25.5));
    assert_eq!(pooled.tpot.evaluated_percentile, 75.0);
    assert_eq!(pooled.tpot.observed_percentile_ms, Some(150.25));
    assert_eq!(pooled.latency_and_outcome_status, SloStatus::Pass);
    cfg.attainment.itl_percentile_scope = SloItlPercentileScope::RequestMaximumGap;
    let request_max = evaluate_slo(&cfg, &records, 2.0).unwrap();
    assert_eq!(request_max.pooled_visible_itl.status, SloStatus::Pass);
    assert_eq!(request_max.request_max_visible_itl.status, SloStatus::Fail);
    assert_eq!(request_max.latency_and_outcome_status, SloStatus::Fail);
}

#[test]
fn accepted_error_limit_is_independent_from_joint_and_rejection_limits() {
    let good = text_request(1.0, &[1.0]);
    let mut failed = good.clone();
    failed.outcome = RequestOutcome::Failed;
    let mut cfg = config();
    cfg.attainment.min_accepted_joint_attainment = 0.5;
    cfg.attainment.min_offered_joint_attainment = Some(0.5);
    cfg.attainment.max_error_rate = 0.25;
    let report = evaluate_slo(&cfg, &[good, failed], 1.0).unwrap();
    assert_eq!(report.accepted_joint_attainment_status, SloStatus::Pass);
    assert_eq!(report.offered_joint_attainment_status, SloStatus::Pass);
    assert_eq!(report.accepted_error_rate_lower_bound, Some(0.5));
    assert_eq!(report.accepted_error_rate_status, SloStatus::Fail);
    assert_eq!(report.reject_rate_status, SloStatus::Pass);
    assert_eq!(report.latency_and_outcome_status, SloStatus::Fail);
}
