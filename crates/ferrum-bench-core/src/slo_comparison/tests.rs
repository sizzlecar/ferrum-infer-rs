use super::*;
use crate::{
    compute_metrics,
    dataset::{
        ShareGptCounts, ShareGptDatasetEvidence, ShareGptFilter, ShareGptSample, ShareGptSelection,
    },
    env::HttpRequestSampling,
    slo::{
        evaluate_slo, AdmissionEvidence, RequestOutcome, RequestSloEvidence, SloEvaluationConfig,
        SloEvaluationReport, SloStatus, TpotBoundary,
    },
    BenchReport, BenchmarkPhase, BenchmarkRequestCorrelation, Env, OutputTokenCountSource,
    QualityIssueCounts, RequestItlEvidence, RequestRecord, RunRecord, Scenario, Slo, WarmupSummary,
};
use ferrum_types::{RuntimeConfigSource, SloAttainmentTargets};

fn hash(label: &str) -> String {
    digest(label.as_bytes())
}

pub(super) fn contract(pair_count: u32) -> FrozenComparisonContract {
    let selection = |pair: u32| {
        let samples: Vec<_> = (0..3)
            .map(|request_index| ShareGptSample {
                source_record_index: u64::from(request_index),
                original_id: None,
                phase: BenchmarkPhase::Measured,
                request_index,
                prompt_sha256: hash(&format!("prompt-{request_index}")),
                assistant_sha256: hash(&format!("assistant-{request_index}")),
                input_tokens: 4,
                reference_output_tokens: 3,
                requested_output_tokens: 3,
            })
            .collect();
        ShareGptSelection {
            repeat_index: pair,
            rng_seed: 17 + u64::from(pair),
            selection_sha256: json_digest(&samples).unwrap(),
            samples,
        }
    };
    let server = |name: &str| FrozenServerIdentity {
        implementation: name.into(),
        backend: "test-backend".into(),
        request_model_alias: "test-model".into(),
        binary_sha256: hash(name),
        effective_configuration_sha256: hash(&format!("{name}-config")),
        numerical_policy: "declared arithmetic".into(),
        intentional_differences: Vec::new(),
    };
    let memory = SampledMemoryPolicy {
        window: "pre-load through shutdown".into(),
        interval_ms: 100,
        max_sample_gap_ns: 200_000_000,
    };
    FrozenComparisonContract {
        schema_version: 1,
        frozen_unix_ns: 1,
        cells: vec![FrozenCell {
            concurrency: 2,
            scope: CellScope::Primary,
        }],
        pairs: (0..pair_count)
            .map(|pair| FrozenPair {
                pair_id: format!("pair-{pair}"),
                selection: selection(pair),
            })
            .collect(),
        shared: SharedExecutionIdentity {
            hardware_fingerprint_sha256: hash("machine"),
            hardware_label: "fixture host".into(),
            model_content_sha256: hash("weights"),
            weight_precision: "Q4".into(),
            kv_precision: "F16".into(),
            tokenizer_sha256: hash("tokenizer"),
            chat_template_sha256: hash("template"),
            client_binary_sha256: hash("client"),
            client_slo_config_sha256: hash("client-slo-config"),
        },
        baseline: server("baseline"),
        candidate: server("candidate"),
        capacity: FixedServerCapacity {
            slots: 8,
            context_tokens_per_request: 64,
            batch_tokens: 32,
        },
        dataset: ShareGptDatasetEvidence {
            dataset: "sharegpt".into(),
            source_path: "/fixture/sharegpt.json".into(),
            source_sha256: hash("dataset"),
            source_format: "json_array".into(),
            tokenizer_sha256: hash("tokenizer"),
            filter: ShareGptFilter {
                min_input_tokens: 1,
                max_input_tokens: Some(16),
                min_output_tokens: 1,
                max_output_tokens: None,
                max_total_tokens: Some(32),
                chat_template_reserve_tokens: 2,
                fixed_output_tokens: None,
            },
            counts: ShareGptCounts {
                records: 3,
                eligible: 3,
                ..Default::default()
            },
            prompt_seed: 17,
            sampling: "without replacement".into(),
            ignore_eos: true,
            enable_thinking: Some(false),
            repeats: Vec::new(),
        },
        http_connection_mode: "fresh".into(),
        sampling: HttpRequestSampling::default(),
        memory: FrozenMemoryPolicy {
            device_allocation: memory.clone(),
            os_footprint: memory.clone().into(),
            maximum_rss_window: memory.window,
        },
        slo: SloEvaluationConfig {
            ttft_ms: 500.0,
            tpot_ms: 100.0,
            visible_itl_ms: 100.0,
            attainment: SloAttainmentTargets {
                min_accepted_joint_attainment: 1.0,
                min_offered_joint_attainment: Some(1.0),
                max_reject_rate: 0.0,
                max_error_rate: 0.0,
                ..Default::default()
            },
            tpot_boundary: TpotBoundary::LastVisibleOutput,
        },
        ratio_limits: ComparisonMetric::ALL
            .into_iter()
            .map(|metric| (metric, if metric.is_throughput() { 0.8 } else { 0.75 }))
            .collect(),
        uncertainty: None,
    }
}

#[derive(Clone)]
pub(super) struct OwnedArm {
    pub(super) report: BenchReport,
    pub(super) evaluation: SloEvaluationReport,
    pub(super) execution: ArmExecutionEvidence,
    pub(super) memory: PeakMemoryEvidence,
}
impl OwnedArm {
    fn input(&self) -> ArmRepeatInput<'_> {
        ArmRepeatInput {
            legacy_benchmark: &self.report,
            report_repeat_index: 0,
            evaluation: &self.evaluation,
            execution: &self.execution,
            memory: Some(&self.memory),
        }
    }
}

pub(super) fn arm(
    contract: &FrozenComparisonContract,
    pair: usize,
    candidate: bool,
    scale: f64,
) -> OwnedArm {
    let server = if candidate {
        &contract.candidate
    } else {
        &contract.baseline
    };
    let selection = &contract.pairs[pair].selection;
    let run_id = format!(
        "{}-{pair}",
        if candidate { "candidate" } else { "baseline" }
    );
    let records: Vec<_> = selection
        .samples
        .iter()
        .map(|sample| {
            let ttft = (40.0 + f64::from(sample.request_index)) * scale;
            let gaps = vec![10.0 * scale, 20.0 * scale];
            RequestRecord {
                benchmark_correlation: Some(
                    BenchmarkRequestCorrelation::new(
                        run_id.clone(),
                        "cell-2".into(),
                        0,
                        BenchmarkPhase::Measured,
                        sample.request_index,
                    )
                    .unwrap(),
                ),
                server_request_id: None,
                success: true,
                ttft_ms: ttft,
                e2e_ms: ttft + gaps.iter().sum::<f64>() + 200.0,
                input_tokens: sample.input_tokens,
                server_input_tokens: Some(sample.input_tokens + 2),
                output_tokens: sample.requested_output_tokens,
                output_token_count_source: OutputTokenCountSource::Usage,
                itl_evidence: RequestItlEvidence::sse(true, 3, Some(3), 2, 0),
                quality_issues: QualityIssueCounts::default(),
                itl_ms: gaps,
            }
        })
        .collect();
    let evidence: Vec<_> = records
        .iter()
        .map(|record| {
            RequestSloEvidence::from_legacy_record(
                record,
                Some(record.ttft_ms + record.itl_ms.iter().sum::<f64>()),
            )
        })
        .collect();
    let duration_s = if candidate { 1.9 } else { 2.0 };
    let evaluation = evaluate_slo(&contract.slo, &evidence, duration_s).unwrap();
    let mut env = Env {
        http_connection_mode: Some(contract.http_connection_mode.clone()),
        http_request_sampling: Some(contract.sampling),
        ..Default::default()
    };
    env.runtime_config.upsert(
        "bench_slo_client_config_sha256",
        contract.shared.client_slo_config_sha256.clone(),
        RuntimeConfigSource::ConfigFile,
    );
    let mut report = compute_metrics(
        server.request_model_alias.clone(),
        server.backend.clone(),
        Scenario::ClosedLoop,
        Some(2),
        None,
        4,
        3,
        0,
        Slo::unbounded(),
        vec![RunRecord {
            records,
            expected_requests: 3,
            duration_s,
            warmup: WarmupSummary::default(),
        }],
        env,
    );
    let mut dataset = contract.dataset.clone();
    let mut local_selection = selection.clone();
    local_selection.repeat_index = 0;
    dataset.repeats = vec![local_selection];
    report.dataset_evidence = Some(dataset);
    // bench_serve supplements the generic aggregator with client tokenization
    // and aggregate source evidence before placing it in the SLO sidecar.
    report.actual_input_tokens_per_request = Some(vec![selection
        .samples
        .iter()
        .map(|sample| sample.input_tokens)
        .collect()]);
    report.output_token_count_source = Some("usage".into());
    let start =
        10_000_000_000 + pair as u64 * 10_000_000_000 + u64::from(candidate) * 4_000_000_000;
    let execution = ArmExecutionEvidence {
        shared: contract.shared.clone(),
        server: server.clone(),
        capacity: contract.capacity.clone(),
        source_manifest_sha256: hash(&run_id),
        measurement_started_unix_ns: start,
        measurement_ended_unix_ns: start + 3_000_000_000,
    };
    let peak = |measurement, bytes| MemoryPeakEvidence {
        measurement,
        peak_bytes: Some(bytes),
        source_sha256: hash("memory"),
        complete: true,
        error_count: 0,
        started_unix_ns: start - 1,
        ended_unix_ns: start + 3_000_000_001,
        window: contract.memory.maximum_rss_window.clone(),
        sample_count: Some(32),
        interval_ms: Some(100),
        max_sample_gap_ns: Some(100_000_000),
    };
    OwnedArm {
        report,
        evaluation,
        execution,
        memory: PeakMemoryEvidence {
            device_allocation: peak(MemoryMeasurement::SampledDeviceAllocation, 1_000_000_000),
            os_footprint: peak(MemoryMeasurement::SampledOsPhysicalFootprint, 2_000_000_000),
            maximum_rss: peak(MemoryMeasurement::ProcessMaximumRss, 3_000_000_000),
        },
    }
}

struct Fixture {
    contract: FrozenComparisonContract,
    pairs: Vec<(OwnedArm, OwnedArm)>,
}
impl Fixture {
    fn new(pair_count: u32) -> Self {
        let contract = contract(pair_count);
        let pairs = (0..pair_count as usize)
            .map(|pair| {
                (
                    arm(&contract, pair, false, 1.0),
                    arm(&contract, pair, true, 0.5),
                )
            })
            .collect();
        Self { contract, pairs }
    }
    fn cell(&self) -> ComparisonCellInput<'_> {
        ComparisonCellInput {
            concurrency: 2,
            pairs: self
                .pairs
                .iter()
                .enumerate()
                .map(|(i, (baseline, candidate))| PairedRepeatInput {
                    pair_id: self.contract.pairs[i].pair_id.clone(),
                    baseline: baseline.input(),
                    candidate: candidate.input(),
                })
                .collect(),
            statistical_evidence: None,
        }
    }
    fn compare(&self) -> SloComparisonReport {
        compare_slo_reports(&self.contract, &[self.cell()]).unwrap()
    }
}

fn diagnostics(report: &SloComparisonReport) -> String {
    let mut output = format!(
        "report status={:?}, issues={:?}",
        report.status, report.issues
    );
    for cell in &report.cells {
        output.push_str(&format!(
            "\nC{} {:?}: status={:?}, issues={:?}",
            cell.concurrency, cell.scope, cell.status, cell.issues
        ));
        for pair in &cell.pairs {
            output.push_str(&format!(
                "\n  pair {}: evidence={:?}, comparison={:?}, issues={:?}",
                pair.pair_id, pair.evidence_status, pair.status, pair.issues
            ));
        }
        for (metric, result) in &cell.metrics {
            output.push_str(&format!("\n  {metric:?}: {result:?}"));
        }
    }
    output
}

fn assert_report_status(report: &SloComparisonReport, expected: ComparisonStatus) {
    assert_eq!(report.status, expected, "{}", diagnostics(report));
}

#[test]
fn real_collector_projection_keeps_last_visible_and_legacy_terminal_distinct() {
    let fixture = Fixture::new(2);
    let report = fixture.compare();
    let pair = &report.cells[0].pairs[0];
    assert!(pair.issues.is_empty(), "{}", diagnostics(&report));
    assert_report_status(&report, ComparisonStatus::Inconclusive);
    assert_eq!(
        pair.evidence_status,
        ComparisonStatus::ObservedPass,
        "{}",
        diagnostics(&report)
    );
    assert_eq!(fixture.pairs[0].0.report.repeat_metrics[0].repeat, 1);
    assert_eq!(fixture.pairs[0].0.input().report_repeat_index, 0);
    assert_eq!(pair.ratios[&ComparisonMetric::TpotP50].baseline, 15.0);
    assert_eq!(pair.ratios[&ComparisonMetric::TpotP50].candidate, 7.5);
    assert!(fixture.pairs[0].0.report.repeat_metrics[0].tpot_ms.p50 > 100.0);
    assert_eq!(
        report.cells[0].pairs[1]
            .baseline_source
            .as_ref()
            .unwrap()
            .report_repeat_index,
        0
    );
    assert_eq!(report.contract.pairs[1].selection.repeat_index, 1);
}

#[test]
fn arbitrary_declared_pair_counts_have_no_invented_proof_gate() {
    for count in [1, 2, 4] {
        let fixture = Fixture::new(count);
        let report = fixture.compare();
        assert_report_status(&report, ComparisonStatus::Inconclusive);
        assert_eq!(report.cells[0].pairs.len(), count as usize);
        assert!(
            report.cells[0]
                .metrics
                .values()
                .all(|metric| metric.status == ComparisonStatus::ObservedPass),
            "{}",
            diagnostics(&report)
        );
    }
}

#[test]
fn ratio_aggregation_is_mean_of_pairs_and_limits_are_external() {
    let mut fixture = Fixture::new(2);
    fixture.pairs[1] = (
        arm(&fixture.contract, 1, false, 2.0),
        arm(&fixture.contract, 1, true, 1.5),
    );
    let report = fixture.compare();
    let metric = &report.cells[0].metrics[&ComparisonMetric::TtftP50];
    assert_eq!(metric.mean_paired_ratio, Some(0.625));
    assert_eq!(metric.observed_ratio_range, Some((0.5, 0.75)));
    fixture
        .contract
        .ratio_limits
        .insert(ComparisonMetric::TtftP50, 0.6);
    assert_report_status(&fixture.compare(), ComparisonStatus::Failed);
    fixture
        .contract
        .ratio_limits
        .insert(ComparisonMetric::TtftP50, 0.7);
    fixture
        .contract
        .ratio_limits
        .insert(ComparisonMetric::SuccessfulUsageOutputTps, 2.0);
    assert_report_status(&fixture.compare(), ComparisonStatus::Failed);
}

#[test]
fn survivor_only_cells_and_pairs_remain_visible_and_unknown() {
    let mut fixture = Fixture::new(2);
    fixture.pairs.pop();
    fixture.contract.cells.push(FrozenCell {
        concurrency: 4,
        scope: CellScope::Primary,
    });
    let report = fixture.compare();
    assert_report_status(&report, ComparisonStatus::Unknown);
    assert_eq!(report.cells.len(), 2);
    assert_eq!(report.cells[0].pairs.len(), 2);
    assert_eq!(report.cells[0].pairs[1].status, ComparisonStatus::Unknown);
    assert_eq!(report.cells[1].status, ComparisonStatus::Unknown);
    let markdown = report.to_markdown(MarkdownLanguage::English);
    assert!(markdown.contains("| Primary | 4 |"));
    assert!(markdown.contains("missing frozen cell"));
}

#[test]
fn repeated_identity_is_not_an_independent_pair() {
    let mut fixture = Fixture::new(2);
    fixture.contract.pairs[1].selection = fixture.contract.pairs[0].selection.clone();
    fixture.pairs[1] = fixture.pairs[0].clone();
    let report = fixture.compare();
    assert_report_status(&report, ComparisonStatus::Unknown);
    assert!(report.cells[0].pairs[1]
        .issues
        .iter()
        .any(|issue| issue.contains("reused")));
}

#[test]
fn identity_capacity_selection_and_policy_mismatches_are_unknown() {
    let changes: [fn(&mut Fixture); 5] = [
        |f| f.pairs[0].1.execution.shared.model_content_sha256 = hash("other-model"),
        |f| f.pairs[0].1.execution.capacity.slots += 1,
        |f| {
            f.pairs[0]
                .1
                .report
                .dataset_evidence
                .as_mut()
                .unwrap()
                .repeats[0]
                .samples
                .swap(0, 1)
        },
        |f| {
            f.pairs[0]
                .1
                .report
                .dataset_evidence
                .as_mut()
                .unwrap()
                .ignore_eos = false
        },
        |f| f.pairs[0].1.execution.measurement_started_unix_ns = 0,
    ];
    for change in changes {
        let mut fixture = Fixture::new(1);
        change(&mut fixture);
        assert_report_status(&fixture.compare(), ComparisonStatus::Unknown);
    }
}

#[test]
fn original_raw_evidence_overrides_a_fabricated_pass_summary() {
    let mut fixture = Fixture::new(1);
    fixture.pairs[0].1.evaluation.request_evidence[0].outcome = RequestOutcome::Failed;
    assert_eq!(
        fixture.pairs[0].1.evaluation.latency_and_outcome_status,
        SloStatus::Pass
    );
    assert_report_status(&fixture.compare(), ComparisonStatus::Failed);
    fixture.pairs[0].1.evaluation.request_evidence[0].outcome = RequestOutcome::Rejected;
    fixture.pairs[0].1.evaluation.request_evidence[0].admission = AdmissionEvidence::Rejected;
    assert_report_status(&fixture.compare(), ComparisonStatus::Failed);
}

#[test]
fn unknown_usage_pending_and_unknown_admission_never_pass() {
    let changes: [fn(&mut RequestSloEvidence); 3] = [
        |r| r.usage_output_tokens = None,
        |r| {
            r.outcome = RequestOutcome::Pending;
            r.terminal_ms = None;
        },
        |r| r.admission = AdmissionEvidence::Unknown,
    ];
    for change in changes {
        let mut fixture = Fixture::new(1);
        change(&mut fixture.pairs[0].1.evaluation.request_evidence[0]);
        assert_report_status(&fixture.compare(), ComparisonStatus::Unknown);
    }
}

#[test]
fn missing_memory_columns_or_sampler_stalls_are_not_zero_memory() {
    let mut fixture = Fixture::new(1);
    fixture.pairs[0].1.memory.os_footprint.peak_bytes = None;
    assert_report_status(&fixture.compare(), ComparisonStatus::Unknown);
    assert!(fixture
        .compare()
        .to_markdown(MarkdownLanguage::English)
        .contains("| Unknown |"));
    fixture.pairs[0].1.memory.os_footprint.peak_bytes = Some(2_000_000_000);
    fixture.pairs[0]
        .1
        .memory
        .device_allocation
        .max_sample_gap_ns = Some(900_000_000);
    assert_report_status(&fixture.compare(), ComparisonStatus::Unknown);
}

#[test]
fn one_sample_cannot_cover_a_nonzero_memory_window() {
    let mut fixture = Fixture::new(1);
    fixture.pairs[0].1.memory.device_allocation.sample_count = Some(1);
    assert_report_status(&fixture.compare(), ComparisonStatus::Unknown);
    let mut input = fixture.cell();
    input.pairs[0].candidate.memory = None;
    assert_eq!(
        compare_slo_reports(&fixture.contract, &[input])
            .unwrap()
            .status,
        ComparisonStatus::Unknown
    );
}

#[test]
fn zero_latency_distributions_cannot_establish_relative_improvement() {
    let mut fixture = Fixture::new(1);
    let baseline = &mut fixture.pairs[0].0;
    for (raw, record) in baseline
        .evaluation
        .request_evidence
        .iter_mut()
        .zip(baseline.report.request_records.as_mut().unwrap()[0].iter_mut())
    {
        raw.visible_text.as_mut().unwrap().gaps_ms.fill(0.0);
        raw.last_visible_ms = raw.first_visible_ms;
        record.timing.as_mut().unwrap().raw_event_gaps_ms.fill(0.0);
    }
    let report = fixture.compare();
    assert_report_status(&report, ComparisonStatus::Unknown);
    assert_eq!(
        report.cells[0].metrics[&ComparisonMetric::VisibleItlP50].mean_paired_ratio,
        None
    );
}

#[test]
fn duplicate_cell_cannot_hide_a_frozen_cell() {
    let fixture = Fixture::new(1);
    let report = compare_slo_reports(&fixture.contract, &[fixture.cell(), fixture.cell()]).unwrap();
    assert_report_status(&report, ComparisonStatus::Unknown);
    assert!(report
        .issues
        .iter()
        .any(|issue| issue.contains("duplicate cell")));
}

#[test]
fn coalesced_sse_diagnostic_does_not_discard_visible_stalls() {
    let mut fixture = Fixture::new(1);
    let candidate = &mut fixture.pairs[0].1;
    let record = &mut candidate.evaluation.request_evidence[0];
    record
        .visible_text
        .as_mut()
        .unwrap()
        .transport_coalesced_output_chunks = 1;
    record.strict_token_evidence = RequestItlEvidence::sse(true, 3, Some(3), 2, 1);
    candidate.report.itl_evidence_per_request.as_mut().unwrap()[0][0] =
        record.strict_token_evidence.clone();
    let report = fixture.compare();
    assert_eq!(
        report.cells[0].pairs[0].evidence_status,
        ComparisonStatus::ObservedPass,
        "{}",
        diagnostics(&report)
    );
    assert_eq!(
        report.cells[0].pairs[0].ratios[&ComparisonMetric::VisibleItlP99].candidate,
        10.0
    );
}

#[test]
fn baseline_absolute_slo_failure_can_be_compared_but_candidate_failure_cannot_win() {
    let mut fixture = Fixture::new(1);
    fixture.contract.slo.ttft_ms = 30.0;
    fixture.pairs[0] = (
        arm(&fixture.contract, 0, false, 1.0),
        arm(&fixture.contract, 0, true, 0.5),
    );
    assert_eq!(
        fixture.pairs[0].0.evaluation.latency_and_outcome_status,
        SloStatus::Fail
    );
    assert_report_status(&fixture.compare(), ComparisonStatus::Inconclusive);
    fixture.contract.slo.ttft_ms = 10.0;
    fixture.pairs[0] = (
        arm(&fixture.contract, 0, false, 1.0),
        arm(&fixture.contract, 0, true, 0.5),
    );
    assert_report_status(&fixture.compare(), ComparisonStatus::Failed);
}

#[test]
fn baseline_and_candidate_must_measure_the_same_usage_work() {
    let mut fixture = Fixture::new(1);
    fixture.pairs[0]
        .1
        .report
        .server_input_tokens_per_request
        .as_mut()
        .unwrap()[0][0] = Some(7);
    fixture.pairs[0].1.report.repeat_metrics[0].server_input_tokens = Some(19);
    let report = fixture.compare();
    assert_report_status(&report, ComparisonStatus::Unknown);
    assert!(report.cells[0].pairs[0]
        .issues
        .iter()
        .any(|issue| issue.contains("different actual prompt/output usage")));
}

#[test]
fn raw_legacy_gaps_and_new_last_visible_are_cross_checked() {
    let mut fixture = Fixture::new(1);
    fixture.pairs[0].1.evaluation.request_evidence[0].last_visible_ms = Some(30.0);
    assert_report_status(&fixture.compare(), ComparisonStatus::Unknown);
    fixture.pairs[0].1 = arm(&fixture.contract, 0, true, 0.5);
    fixture.pairs[0].1.report.request_records.as_mut().unwrap()[0][0]
        .timing
        .as_mut()
        .unwrap()
        .raw_event_gaps_ms[0] = 1.0;
    assert_report_status(&fixture.compare(), ComparisonStatus::Unknown);
}

#[test]
fn supplying_tight_intervals_cannot_bypass_unimplemented_inference() {
    let mut fixture = Fixture::new(1);
    let method = FrozenStatisticalMethod {
        method_id: "external-method".into(),
        analysis_unit: "paired-repeat".into(),
        configuration_sha256: hash("method"),
        paired_bootstrap: None,
    };
    fixture.contract.uncertainty = Some(method.clone());
    let descriptive = fixture.compare();
    let evidence = StatisticalEvidence {
        method,
        paired_measurements_sha256: descriptive.cells[0].paired_measurements_sha256.clone(),
        source_artifact_sha256: hash("statistics"),
        confidence_level: 0.95,
        ratio_intervals: ComparisonMetric::ALL
            .into_iter()
            .map(|metric| {
                (
                    metric,
                    if metric.is_throughput() {
                        (1.0, 1.1)
                    } else {
                        (0.4, 0.6)
                    },
                )
            })
            .collect(),
    };
    let mut input = fixture.cell();
    input.statistical_evidence = Some(&evidence);
    let report = compare_slo_reports(&fixture.contract, &[input]).unwrap();
    assert_report_status(&report, ComparisonStatus::Inconclusive);
    assert!(report.cells[0]
        .issues
        .iter()
        .any(|issue| issue.contains("verification is not implemented")));
}

#[test]
fn contract_rejects_nonfinite_zero_limits_and_invalid_lengths() {
    for invalid in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        let mut contract = contract(1);
        contract
            .ratio_limits
            .insert(ComparisonMetric::TpotP99, invalid);
        assert!(contract.validate().is_err());
    }
    let mut contract = contract(1);
    contract.dataset.filter.min_output_tokens = 4;
    assert!(contract.validate().is_err());
    contract.dataset.filter.min_output_tokens = 1;
    contract.slo.tpot_boundary = TpotBoundary::LegacyTerminal;
    assert!(contract.validate().is_err());
}

#[test]
fn tables_share_numerical_rows_separate_memory_and_escape_labels() {
    let mut fixture = Fixture::new(1);
    fixture.contract.candidate.implementation = "candidate|<unsafe>\nlabel".into();
    fixture.pairs[0].1 = arm(&fixture.contract, 0, true, 0.5);
    // The collector correlation ID is separately constrained; use its original
    // valid identity while retaining a display label that must be escaped.
    let report = fixture.compare();
    let english = report.to_markdown(MarkdownLanguage::English);
    let chinese = report.to_markdown(MarkdownLanguage::Chinese);
    let numeric_rows = |text: &str| {
        text.lines()
            .filter(|line| line.starts_with("| Primary |"))
            .map(str::to_owned)
            .collect::<Vec<_>>()
    };
    assert_eq!(numeric_rows(&english), numeric_rows(&chinese));
    assert!(english.contains("candidate&#124;&lt;unsafe&gt; label"));
    assert!(english.contains("Device allocation GiB"));
    assert!(english.contains("OS footprint GiB"));
    assert!(english.contains("RSS GiB"));
    assert!(english.contains("Observed range (not CI)"));
}

#[test]
fn generic_aggregator_without_client_supplements_remains_unknown() {
    let mut fixture = Fixture::new(1);
    fixture.pairs[0].1.report.actual_input_tokens_per_request = None;
    fixture.pairs[0].1.report.output_token_count_source = None;
    let report = fixture.compare();
    assert_report_status(&report, ComparisonStatus::Unknown);
    assert!(report.cells[0].pairs[0]
        .issues
        .iter()
        .any(|issue| issue.contains("aligned legacy per-request evidence")));
    assert!(report.cells[0].pairs[0]
        .issues
        .iter()
        .any(|issue| issue.contains("complete usage-token evidence")));
}

#[test]
fn legacy_display_repeat_is_one_based_but_request_indices_remain_zero_based() {
    let mut fixture = Fixture::new(1);
    assert_report_status(&fixture.compare(), ComparisonStatus::Inconclusive);
    fixture.pairs[0].1.report.repeat_metrics[0].repeat = 0;
    let report = fixture.compare();
    assert_report_status(&report, ComparisonStatus::Unknown);
    assert!(
        report.cells[0].pairs[0]
            .issues
            .iter()
            .any(|issue| issue.contains("legacy one-based repeat=0")),
        "{}",
        diagnostics(&report)
    );
    fixture.pairs[0].1.report.repeat_metrics[0].repeat = 1;
    fixture.pairs[0].1.report.request_records.as_mut().unwrap()[0][0]
        .correlation
        .repeat_index = 1;
    assert_report_status(&fixture.compare(), ComparisonStatus::Unknown);
}

#[test]
fn rendered_arm_slo_and_memory_coverage_are_independent_of_comparison_status() {
    let mut fixture = Fixture::new(1);
    fixture.contract.slo.ttft_ms = 30.0;
    fixture.pairs[0] = (
        arm(&fixture.contract, 0, false, 1.0),
        arm(&fixture.contract, 0, true, 0.5),
    );
    let report = fixture.compare();
    let pair = &report.cells[0].pairs[0];
    assert_eq!(
        pair.baseline_source.as_ref().unwrap().absolute_slo_status,
        SloStatus::Fail
    );
    assert_eq!(
        pair.candidate_source.as_ref().unwrap().absolute_slo_status,
        SloStatus::Pass
    );
    assert_report_status(&report, ComparisonStatus::Inconclusive);
    let english = report.to_markdown(MarkdownLanguage::English);
    assert!(english.contains("Arm absolute SLO"));
    assert!(english.contains("Comparison status"));
    assert!(english
        .lines()
        .any(|line| line.contains("| baseline |") && line.ends_with("| Fail | Inconclusive |")));
    assert!(english
        .lines()
        .any(|line| line.contains("| candidate |") && line.ends_with("| Pass | Inconclusive |")));

    fixture.pairs[0].1.memory.device_allocation.complete = false;
    let report = fixture.compare();
    assert_report_status(&report, ComparisonStatus::Unknown);
    let english = report.to_markdown(MarkdownLanguage::English);
    let candidate = english
        .lines()
        .find(|line| line.contains("| candidate |"))
        .unwrap();
    assert!(candidate.contains("Unknown (unverified max"));
    assert!(candidate.contains("coverage 0/1"));
    // Footprint and RSS retain their own complete coverage despite the device
    // sampler being incomplete; the raw device peak remains in the JSON.
    assert_eq!(candidate.matches("unverified max").count(), 1);
    assert_eq!(
        report.cells[0].pairs[0]
            .candidate_memory
            .as_ref()
            .unwrap()
            .device_allocation
            .peak_bytes,
        Some(1_000_000_000)
    );
}
