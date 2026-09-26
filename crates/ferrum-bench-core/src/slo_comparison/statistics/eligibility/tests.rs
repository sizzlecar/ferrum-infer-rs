use super::*;
use crate::slo_comparison::tests::{arm, contract, OwnedArm};
use crate::{
    compute_metrics,
    slo::{evaluate_slo, RequestSloEvidence},
    BenchmarkPhase, BenchmarkRequestCorrelation, OutputTokenCountSource, QualityIssueCounts,
    RequestItlEvidence, RequestRecord, RunRecord, Scenario, Slo, WarmupSummary,
};

fn expand_samples(contract: &mut FrozenComparisonContract) {
    for pair in &mut contract.pairs {
        let template = pair.selection.samples[0].clone();
        pair.selection.samples = (0..128)
            .map(|index| {
                let mut sample = template.clone();
                sample.source_record_index = u64::from(index);
                sample.request_index = index;
                sample.prompt_sha256 = digest(format!("prompt-{index}").as_bytes());
                sample.assistant_sha256 = digest(format!("assistant-{index}").as_bytes());
                sample
            })
            .collect();
        pair.selection.selection_sha256 = json_digest(&pair.selection.samples).unwrap();
    }
    contract.dataset.counts.records = 128;
    contract.dataset.counts.eligible = 128;
}

fn raw_arm(
    contract: &FrozenComparisonContract,
    pair: usize,
    candidate: bool,
    pilot: bool,
    latency: f64,
    tps_ratio: f64,
) -> OwnedArm {
    // The shared legacy fixture has exactly three requests. Use it only for
    // execution/environment metadata; construct the full raw run below.
    let mut skeleton = contract.clone();
    for pair in &mut skeleton.pairs {
        pair.selection.samples.truncate(3);
        pair.selection.selection_sha256 = json_digest(&pair.selection.samples).unwrap();
    }
    let mut owned = arm(&skeleton, pair, candidate, latency);
    let run_id = format!(
        "{}-{}-{pair}",
        if pilot { "pilot" } else { "main" },
        if candidate { "candidate" } else { "baseline" }
    );
    let selection = &contract.pairs[pair].selection;
    let records: Vec<_> = selection
        .samples
        .iter()
        .map(|sample| {
            let ttft = (40.0 + f64::from(sample.request_index)) * latency;
            let gaps = vec![10.0 * latency, 20.0 * latency];
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
    let duration = 2.0 / tps_ratio;
    owned.evaluation = evaluate_slo(&contract.slo, &evidence, duration).unwrap();
    let server = if candidate {
        &contract.candidate
    } else {
        &contract.baseline
    };
    let mut dataset = contract.dataset.clone();
    let mut local_selection = selection.clone();
    local_selection.repeat_index = 0;
    dataset.repeats = vec![local_selection];
    let tokens = vec![selection
        .samples
        .iter()
        .map(|sample| sample.input_tokens)
        .collect()];
    owned.report = compute_metrics(
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
            expected_requests: selection.samples.len() as u32,
            duration_s: duration,
            warmup: WarmupSummary::default(),
        }],
        owned.report.env.clone(),
    );
    owned.report.dataset_evidence = Some(dataset);
    owned.report.actual_input_tokens_per_request = Some(tokens);
    owned.report.output_token_count_source = Some("usage".into());
    let baseline_first = pair % 2 == 0;
    let second = candidate == baseline_first;
    let start = if pilot {
        10_000_000_000
    } else {
        100_000_000_000
    } + pair as u64 * 10_000_000_000
        + u64::from(second) * 4_000_000_000;
    owned.execution.measurement_started_unix_ns = start;
    owned.execution.measurement_ended_unix_ns = start + 3_000_000_000;
    owned.execution.source_manifest_sha256 = digest(run_id.as_bytes());
    for memory in [
        &mut owned.memory.device_allocation,
        &mut owned.memory.os_footprint,
        &mut owned.memory.maximum_rss,
    ] {
        memory.started_unix_ns = start - 1;
        memory.ended_unix_ns = start + 3_000_000_001;
    }
    owned
}

fn input(arm: &OwnedArm) -> ArmRepeatInput<'_> {
    ArmRepeatInput {
        legacy_benchmark: &arm.report,
        report_repeat_index: 0,
        evaluation: &arm.evaluation,
        execution: &arm.execution,
        memory: Some(&arm.memory),
    }
}
fn cell<'a>(
    contract: &FrozenComparisonContract,
    arms: &'a [(OwnedArm, OwnedArm)],
) -> ComparisonCellInput<'a> {
    ComparisonCellInput {
        concurrency: 2,
        pairs: arms
            .iter()
            .enumerate()
            .map(|(index, (b, c))| PairedRepeatInput {
                pair_id: contract.pairs[index].pair_id.clone(),
                baseline: input(b),
                candidate: input(c),
            })
            .collect(),
        statistical_evidence: None,
    }
}

pub(in crate::slo_comparison) struct Fixture {
    pub(in crate::slo_comparison) contract: FrozenComparisonContract,
    pub(in crate::slo_comparison) pilot_contract: FrozenComparisonContract,
    pub(in crate::slo_comparison) plan: FrozenEligibilityPlan,
    pub(in crate::slo_comparison) source: String,
    pub(in crate::slo_comparison) pilot: Vec<(OwnedArm, OwnedArm)>,
    pub(in crate::slo_comparison) main: Vec<(OwnedArm, OwnedArm)>,
}
impl Fixture {
    pub(in crate::slo_comparison) fn new() -> Self {
        let mut contract = contract(3);
        expand_samples(&mut contract);
        contract.frozen_unix_ns = 50_000_000_000;
        let mut pilot_contract = contract.clone();
        pilot_contract.frozen_unix_ns = 2;
        pilot_contract.candidate = pilot_contract.baseline.clone();
        let plan = FrozenEligibilityPlan {
            schema_version: 1,
            frozen_unix_ns: 1,
            assumptions: InferenceAssumptions::IndependentExchangeablePairedBlocksFixedWorkload,
            planning_pair_counts: vec![3, 6],
            planning_resamples: 4000,
            seed: 17,
            maximum_request_rank_step: 0.01,
            maximum_visible_gap_rank_step: 0.01,
            maximum_order_log_ratio_shift: 0.1,
            maximum_time_trend_log_ratio_shift: 0.1,
            maximum_absolute_lag_one_correlation: 0.95,
            maximum_within_pair_idle_ns: 2_000_000_000,
            independent_block_protocol: "fixture independent restart and warmup protocol".into(),
        };
        let source = digest(b"original-pilot-manifest");
        contract.uncertainty = Some(
            FrozenStatisticalMethod::paired_cluster_bootstrap(FrozenPairedBootstrap {
                schema_version: 1,
                estimand: BootstrapEstimand::ArithmeticMeanPairedRatio,
                seed: 61,
                resamples: 4000,
                family_alpha: 0.05,
                monte_carlo_error_budget: 0.001,
                maximum_relative_bound_width: ComparisonMetric::ALL
                    .into_iter()
                    .map(|m| (m, 0.25))
                    .collect(),
                eligibility_plan_sha256: Some(plan.configuration_sha256().unwrap()),
                declared_design: DeclaredBootstrapDesign {
                    pilot_source_sha256: source.clone(),
                    pilot_finished_unix_ns: 40_000_000_000,
                    planned_pairs: 3,
                    minimum_measured_requests_per_arm: 100,
                    minimum_gap_bearing_requests_per_arm: 100,
                    minimum_visible_gaps_per_arm: 100,
                    first_pair_order: PairedArmOrder::BaselineFirst,
                    independent_block_protocol: plan.independent_block_protocol.clone(),
                },
            })
            .unwrap(),
        );
        let pilot = [0.98, 1.01, 1.0]
            .iter()
            .enumerate()
            .map(|(i, scale)| {
                (
                    raw_arm(&pilot_contract, i, false, true, 1.0, 1.0),
                    raw_arm(&pilot_contract, i, true, true, *scale, [0.99, 1.01, 1.0][i]),
                )
            })
            .collect();
        let main = [0.45, 0.55, 0.65]
            .iter()
            .enumerate()
            .map(|(i, scale)| {
                (
                    raw_arm(&contract, i, false, false, 1.0, 1.0),
                    raw_arm(&contract, i, true, false, *scale, [1.1, 1.2, 1.3][i]),
                )
            })
            .collect();
        Self {
            contract,
            pilot_contract,
            plan,
            source,
            pilot,
            main,
        }
    }
    fn verify(&self) -> Result<VerifiedInferenceEligibility, EligibilityFailure> {
        verify_inference_eligibility(
            &self.contract,
            &self.plan,
            &self.pilot_contract,
            &[cell(&self.pilot_contract, &self.pilot)],
            &self.source,
        )
    }
    fn compare(&self, token: &VerifiedInferenceEligibility) -> SloComparisonReport {
        compare_with_eligibility(&self.contract, &[cell(&self.contract, &self.main)], token)
            .unwrap()
    }
    fn bind_plan(&mut self) {
        let mut config = self
            .contract
            .uncertainty
            .as_ref()
            .unwrap()
            .paired_bootstrap
            .clone()
            .unwrap();
        config.eligibility_plan_sha256 = Some(self.plan.configuration_sha256().unwrap());
        self.contract.uncertainty =
            Some(FrozenStatisticalMethod::paired_cluster_bootstrap(config).unwrap());
    }
}

#[test]
fn raw_pilot_to_frozen_planning_to_conditional_proof_is_reachable() {
    let fixture = Fixture::new();
    let without =
        compare_slo_reports(&fixture.contract, &[cell(&fixture.contract, &fixture.main)]).unwrap();
    assert_eq!(
        without.status,
        ComparisonStatus::Inconclusive,
        "{without:#?}"
    );
    let token = fixture.verify().unwrap();
    assert_eq!(token.report.selected_pairs, 3);
    assert!(token.report.forecasts[0].all_precision_targets_met);
    assert_eq!(token.report, fixture.verify().unwrap().report);
    let report = fixture.compare(&token);
    assert_eq!(report.status, ComparisonStatus::ProofPass, "{report:#?}");
    assert_eq!(
        report.computed_bootstrap.unwrap().calibration,
        BootstrapCalibrationStatus::EligibleUnderDeclaredAssumptions
    );
    assert!(report.inference_eligibility.is_some());
}

#[test]
fn eligibility_does_not_transfer_contract_or_substitute_pilot_forecast_for_actual_precision() {
    let mut fixture = Fixture::new();
    let token = fixture.verify().unwrap();
    fixture
        .contract
        .ratio_limits
        .insert(ComparisonMetric::TtftP50, 0.74);
    assert!(compare_with_eligibility(
        &fixture.contract,
        &[cell(&fixture.contract, &fixture.main)],
        &token
    )
    .is_err());
    let mut fixture = Fixture::new();
    let token = fixture.verify().unwrap();
    for (i, scale) in [0.01, 0.5, 1.6].into_iter().enumerate() {
        fixture.main[i].1 = raw_arm(&fixture.contract, i, true, false, scale, [1.1, 1.2, 1.3][i]);
    }
    let report = fixture.compare(&token);
    assert_eq!(report.status, ComparisonStatus::Inconclusive, "{report:#?}");
    assert!(
        !report.computed_bootstrap.unwrap().cells[0].metrics[&ComparisonMetric::TtftP99]
            .precision_target_met
    );
}

#[test]
fn pilot_missing_pair_wrong_source_order_and_postfreeze_data_cannot_issue_token() {
    let mut fixture = Fixture::new();
    fixture.pilot.pop();
    assert!(fixture.verify().is_err());
    let mut fixture = Fixture::new();
    fixture.source = digest(b"other source");
    assert!(fixture.verify().is_err());
    let mut fixture = Fixture::new();
    fixture.pilot[1].0.execution.measurement_started_unix_ns = 20_000_000_000;
    assert!(fixture.verify().is_err());
    let mut fixture = Fixture::new();
    fixture.pilot[2].0.memory.maximum_rss.ended_unix_ns = fixture.contract.frozen_unix_ns + 1;
    assert!(fixture.verify().is_err());
}

#[test]
fn pilot_rank_support_degeneracy_and_unresolved_simulation_tail_remain_insufficient() {
    let mut fixture = Fixture::new();
    fixture.plan.maximum_request_rank_step = 0.001;
    fixture.bind_plan();
    assert!(fixture
        .verify()
        .unwrap_err()
        .to_string()
        .contains("rank resolution"));
    let mut fixture = Fixture::new();
    for index in 0..fixture.pilot.len() {
        fixture.pilot[index].1 = raw_arm(&fixture.pilot_contract, index, true, true, 1.0, 1.0);
    }
    assert!(fixture
        .verify()
        .unwrap_err()
        .to_string()
        .contains("degenerate"));
    let mut fixture = Fixture::new();
    fixture.plan.planning_resamples = 32;
    fixture.bind_plan();
    assert!(fixture
        .verify()
        .unwrap_err()
        .to_string()
        .contains("tail budget"));
}

#[test]
fn planning_work_is_bounded_across_the_entire_grid_before_resampling() {
    let mut fixture = Fixture::new();
    fixture.plan.planning_pair_counts = vec![400, 401, 402];
    fixture.plan.planning_resamples = 20_000;
    fixture.bind_plan();
    assert!(fixture
        .verify()
        .unwrap_err()
        .to_string()
        .contains("cumulative"));
}

#[test]
fn aa_directional_ratios_are_reported_without_inventing_a_unit_mean_null() {
    let mut fixture = Fixture::new();
    for (index, scale) in [2.0, 0.5, 1.0].into_iter().enumerate() {
        fixture.pilot[index].1 = raw_arm(
            &fixture.pilot_contract,
            index,
            true,
            true,
            scale,
            [0.99, 1.01, 1.0][index],
        );
    }
    let pilot = compare_slo_reports(
        &fixture.pilot_contract,
        &[cell(&fixture.pilot_contract, &fixture.pilot)],
    )
    .unwrap();
    let diagnostics = planning::diagnostics(
        &[&pilot.cells[0]],
        &fixture.plan,
        PairedArmOrder::BaselineFirst,
    )
    .unwrap();
    let ttft = diagnostics
        .iter()
        .find(|d| d.metric == ComparisonMetric::TtftP99)
        .unwrap();
    assert!((ttft.mean_forward_ratio - 7.0 / 6.0).abs() < 1e-12);
    assert!((ttft.mean_reverse_ratio - 7.0 / 6.0).abs() < 1e-12);
    assert!(!ttft.within_declared_limits);
}

#[test]
fn caller_success_fields_and_mutated_plans_cannot_create_eligibility() {
    let mut fixture = Fixture::new();
    fixture.plan.maximum_order_log_ratio_shift *= 2.0;
    assert!(fixture.verify().is_err());
    let mut wire = serde_json::to_value(&fixture.plan).unwrap();
    wire["verified"] = serde_json::json!(true);
    assert!(serde_json::from_value::<FrozenEligibilityPlan>(wire).is_err());
}
