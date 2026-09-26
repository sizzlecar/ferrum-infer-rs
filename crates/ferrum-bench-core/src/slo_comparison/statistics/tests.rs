use super::super::tests::{arm, contract, OwnedArm};
use super::*;

fn configuration(pairs: u32) -> FrozenPairedBootstrap {
    FrozenPairedBootstrap {
        schema_version: 1,
        estimand: BootstrapEstimand::ArithmeticMeanPairedRatio,
        seed: 61,
        resamples: 4000,
        family_alpha: 0.05,
        monte_carlo_error_budget: 0.001,
        eligibility_plan_sha256: None,
        maximum_relative_bound_width: ComparisonMetric::ALL
            .into_iter()
            .map(|metric| (metric, 0.25))
            .collect(),
        declared_design: DeclaredBootstrapDesign {
            pilot_source_sha256: digest(b"independent-fixture-pilot"),
            pilot_finished_unix_ns: 1,
            planned_pairs: pairs,
            // These fixture counts are declarations, never sufficient to make
            // these deliberately tiny synthetic examples statistical proof.
            minimum_measured_requests_per_arm: 3,
            minimum_gap_bearing_requests_per_arm: 3,
            minimum_visible_gaps_per_arm: 6,
            first_pair_order: PairedArmOrder::BaselineFirst,
            independent_block_protocol: "fixture declaration, raw pilot not verified".into(),
        },
    }
}

fn shift(arm: &mut OwnedArm, start: u64) {
    arm.execution.measurement_started_unix_ns = start;
    arm.execution.measurement_ended_unix_ns = start + 3_000_000_000;
    for memory in [
        &mut arm.memory.device_allocation,
        &mut arm.memory.os_footprint,
        &mut arm.memory.maximum_rss,
    ] {
        memory.started_unix_ns = start - 1;
        memory.ended_unix_ns = start + 3_000_000_001;
    }
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

struct InferenceFixture {
    contract: FrozenComparisonContract,
    pairs: Vec<(OwnedArm, OwnedArm)>,
}

impl InferenceFixture {
    fn new(scales: &[f64]) -> Self {
        let mut contract = contract(scales.len() as u32);
        contract.frozen_unix_ns = 2;
        contract.uncertainty = Some(
            FrozenStatisticalMethod::paired_cluster_bootstrap(configuration(scales.len() as u32))
                .unwrap(),
        );
        let pairs = scales
            .iter()
            .enumerate()
            .map(|(index, scale)| {
                let mut baseline = arm(&contract, index, false, 1.0);
                let mut candidate = arm(&contract, index, true, *scale);
                if index % 2 == 1 {
                    let b = baseline.execution.measurement_started_unix_ns;
                    let c = candidate.execution.measurement_started_unix_ns;
                    shift(&mut baseline, c);
                    shift(&mut candidate, b);
                }
                (baseline, candidate)
            })
            .collect();
        Self { contract, pairs }
    }

    fn compare(&self) -> Result<SloComparisonReport, ComparisonError> {
        compare_slo_reports(
            &self.contract,
            &[ComparisonCellInput {
                concurrency: 2,
                pairs: self
                    .pairs
                    .iter()
                    .enumerate()
                    .map(|(index, (baseline, candidate))| PairedRepeatInput {
                        pair_id: self.contract.pairs[index].pair_id.clone(),
                        baseline: input(baseline),
                        candidate: input(candidate),
                    })
                    .collect(),
                statistical_evidence: None,
            }],
        )
    }

    fn change_configuration(&mut self, change: impl FnOnce(&mut FrozenPairedBootstrap)) {
        let mut configuration = self
            .contract
            .uncertainty
            .as_ref()
            .unwrap()
            .paired_bootstrap
            .clone()
            .unwrap();
        change(&mut configuration);
        self.contract.uncertainty =
            Some(FrozenStatisticalMethod::paired_cluster_bootstrap(configuration).unwrap());
    }
}

#[test]
fn bootstrap_recomputes_real_adapter_ratios_but_cannot_certify_unverified_pilot() {
    let fixture = InferenceFixture::new(&[0.45, 0.55, 0.65]);
    let report = fixture.compare().unwrap();
    assert_eq!(
        report.status,
        ComparisonStatus::Inconclusive,
        "{:?}",
        report.cells[0].pairs
    );
    let inference = report.computed_bootstrap.unwrap();
    assert_eq!(
        inference.calibration,
        BootstrapCalibrationStatus::Unverified
    );
    assert_eq!(inference.status, ComparisonStatus::Inconclusive);
    assert_eq!(inference.family_size, 7);
    assert_eq!(inference.cells.len(), 1, "{:?}", inference.issues);
    let bound = &inference.cells[0].metrics[&ComparisonMetric::TtftP99];
    assert!(bound.strict_limit_cleared, "{bound:?}");
    assert!(bound.precision_target_met);
    assert!((bound.point_estimate - 0.55).abs() < 1e-12);
    assert!(inference
        .issues
        .iter()
        .any(|issue| issue.contains("eligibility")));
}

#[test]
fn bootstrap_same_seed_is_reproducible_and_correlated_cells_share_draws() {
    let column = vec![1.0, 2.0, 4.0, 8.0];
    let mut columns = vec![column; 14]; // Two complete primary metric vectors.
    for value in &mut columns[7] {
        *value *= 2.0;
    }
    let first = compute::resample_means(&columns, 128, 7).unwrap();
    assert_eq!(first, compute::resample_means(&columns, 128, 7).unwrap());
    assert_ne!(first, compute::resample_means(&columns, 128, 8).unwrap());
    for index in 0..128 {
        assert_eq!(first[0][index], first[6][index]);
        assert_eq!(first[7][index], 2.0 * first[0][index]);
        assert_eq!(first[13][index], first[0][index]);
    }
}

#[test]
fn bootstrap_rejects_zero_negative_nonfinite_missing_and_unequal_ratio_columns() {
    for invalid in [0.0, -1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let mut columns = vec![vec![0.5, 0.6]; 7];
        columns[1][0] = invalid;
        assert!(compute::resample_means(&columns, 4, 9).is_err());
    }
    assert!(compute::resample_means(&[], 4, 9).is_err());
    assert!(compute::resample_means(&[vec![0.5, 0.6]], 4, 9).is_err());
    let mut columns = vec![vec![0.5, 0.6]; 7];
    columns[3].pop();
    assert!(compute::resample_means(&columns, 4, 9).is_err());
}

#[test]
fn bootstrap_tail_rank_respects_simulation_error_and_family_size() {
    assert_eq!(compute::tail_rank(64, 0.01, 0.001), None);
    let rank = compute::tail_rank(20_000, 0.01, 0.001).unwrap();
    let mu = 20_000.0 * 0.01;
    assert!((-(mu - rank as f64).powi(2) / (2.0 * mu)).exp() <= 0.001);
    assert!(compute::tail_rank(20_000, 0.005, 0.0005).unwrap() < rank);
    let mut fixture = InferenceFixture::new(&[0.45, 0.55, 0.65]);
    fixture.change_configuration(|config| config.resamples = 32);
    let report = fixture.compare().unwrap();
    let inference = report.computed_bootstrap.unwrap();
    assert_eq!(report.status, ComparisonStatus::Inconclusive);
    assert!(inference.conservative_tail_rank.is_none());
    assert!(inference.cells.is_empty());
    assert!(inference
        .issues
        .iter()
        .any(|issue| issue.contains("tail resolution")));
}

#[test]
fn bootstrap_declared_p99_support_and_single_cluster_do_not_manufacture_intervals() {
    let mut fixture = InferenceFixture::new(&[0.45, 0.55, 0.65]);
    fixture.change_configuration(|config| {
        config.declared_design.minimum_measured_requests_per_arm = 4
    });
    let inference = fixture.compare().unwrap().computed_bootstrap.unwrap();
    assert!(inference.cells.is_empty());
    assert!(inference.issues.iter().any(|issue| issue.contains("P99")));
    let single = InferenceFixture::new(&[0.5]).compare().unwrap();
    assert_eq!(single.status, ComparisonStatus::Inconclusive);
    assert!(single
        .computed_bootstrap
        .unwrap()
        .issues
        .iter()
        .any(|issue| issue.contains("single paired cluster")));
}

#[test]
fn bootstrap_cannot_drop_missing_primary_cells_or_inconvenient_pairs() {
    let mut fixture = InferenceFixture::new(&[0.45, 0.55, 0.65]);
    fixture.contract.cells.push(FrozenCell {
        concurrency: 4,
        scope: CellScope::Primary,
    });
    let report = fixture.compare().unwrap();
    assert_eq!(report.status, ComparisonStatus::Unknown);
    let inference = report.computed_bootstrap.unwrap();
    assert_eq!(inference.family_size, 14);
    assert!(inference.cells.is_empty());
    let mut fixture = InferenceFixture::new(&[0.45, 0.55, 0.65]);
    fixture.pairs.pop();
    let report = fixture.compare().unwrap();
    assert_eq!(report.status, ComparisonStatus::Unknown);
    assert!(report.computed_bootstrap.unwrap().cells.is_empty());
}

#[test]
fn bootstrap_duplicate_pair_cannot_choose_a_convenient_replacement() {
    let fixture = InferenceFixture::new(&[0.45, 0.55, 0.65]);
    let mut pairs: Vec<_> = fixture
        .pairs
        .iter()
        .enumerate()
        .map(|(index, (baseline, candidate))| PairedRepeatInput {
            pair_id: fixture.contract.pairs[index].pair_id.clone(),
            baseline: input(baseline),
            candidate: input(candidate),
        })
        .collect();
    pairs.push(PairedRepeatInput {
        pair_id: fixture.contract.pairs[0].pair_id.clone(),
        baseline: input(&fixture.pairs[0].0),
        candidate: input(&fixture.pairs[0].1),
    });
    let report = compare_slo_reports(
        &fixture.contract,
        &[ComparisonCellInput {
            concurrency: 2,
            pairs,
            statistical_evidence: None,
        }],
    )
    .unwrap();
    assert_eq!(report.cells[0].status, ComparisonStatus::Unknown);
    assert!(report.computed_bootstrap.unwrap().cells.is_empty());
}

#[test]
fn bootstrap_unfavorable_pair_is_preserved_even_when_mean_clears_limit() {
    let report = InferenceFixture::new(&[0.45, 0.55, 1.2]).compare().unwrap();
    let inference = report.computed_bootstrap.unwrap();
    assert_eq!(inference.cells.len(), 1, "{:?}", inference.issues);
    let bound = &inference.cells[0].metrics[&ComparisonMetric::TtftP99];
    assert!(bound.point_estimate < report.contract.ratio_limits[&ComparisonMetric::TtftP99]);
    assert!(!bound.strict_limit_cleared, "{bound:?}");
    assert_eq!(inference.cells[0].pairs, 3);
}

#[test]
fn bootstrap_boundary_equality_and_constant_repeats_are_not_certainty() {
    let mut fixture = InferenceFixture::new(&[0.45, 0.55, 0.65]);
    let first = fixture.compare().unwrap();
    let bound = first.computed_bootstrap.as_ref().unwrap().cells[0].metrics
        [&ComparisonMetric::TtftP99]
        .one_sided_bound
        .unwrap();
    fixture
        .contract
        .ratio_limits
        .insert(ComparisonMetric::TtftP99, bound);
    let inference = fixture.compare().unwrap().computed_bootstrap.unwrap();
    assert!(!inference.cells[0].metrics[&ComparisonMetric::TtftP99].strict_limit_cleared);
    let constant = &inference.cells[0].metrics[&ComparisonMetric::SuccessfulUsageOutputTps];
    assert!(constant.degenerate_empirical_distribution);
    assert_eq!(constant.one_sided_bound, None);
    assert!(!constant.precision_target_met);
}

#[test]
fn bootstrap_wrong_analysis_unit_digest_and_postfreeze_pilot_are_rejected() {
    let mut fixture = InferenceFixture::new(&[0.45, 0.55, 0.65]);
    fixture.contract.uncertainty.as_mut().unwrap().analysis_unit = "iid_visible_gaps".into();
    assert!(fixture.compare().is_err());
    let mut fixture = InferenceFixture::new(&[0.45, 0.55, 0.65]);
    fixture
        .contract
        .uncertainty
        .as_mut()
        .unwrap()
        .paired_bootstrap
        .as_mut()
        .unwrap()
        .seed += 1;
    assert!(fixture.compare().is_err());
    let mut fixture = InferenceFixture::new(&[0.45, 0.55, 0.65]);
    fixture.change_configuration(|config| config.declared_design.pilot_finished_unix_ns = 3);
    assert!(fixture.compare().is_err());
}

#[test]
fn bootstrap_arm_order_and_overlapping_paired_blocks_are_not_independence() {
    let mut fixture = InferenceFixture::new(&[0.45, 0.55, 0.65]);
    fixture.change_configuration(|config| {
        config.declared_design.first_pair_order = PairedArmOrder::CandidateFirst
    });
    let inference = fixture.compare().unwrap().computed_bootstrap.unwrap();
    assert!(inference.cells.is_empty());
    assert!(inference
        .issues
        .iter()
        .any(|issue| issue.contains("arm order")));
    let mut fixture = InferenceFixture::new(&[0.45, 0.55, 0.65]);
    shift(&mut fixture.pairs[1].0, 14_000_000_000);
    shift(&mut fixture.pairs[1].1, 10_000_000_000);
    let inference = fixture.compare().unwrap().computed_bootstrap.unwrap();
    assert!(inference.cells.is_empty());
    assert!(inference
        .issues
        .iter()
        .any(|issue| issue.contains("overlap")));
}

#[test]
fn bootstrap_configuration_and_resource_limits_are_checked_before_computation() {
    for bad in [f64::NAN, f64::INFINITY, -0.1, 0.0, 1.0] {
        let mut config = configuration(3);
        config.family_alpha = bad;
        assert!(FrozenStatisticalMethod::paired_cluster_bootstrap(config).is_err());
    }
    let mut config = configuration(4096);
    config.resamples = 1_000_000;
    assert!(FrozenStatisticalMethod::paired_cluster_bootstrap(config).is_err());
    let mut config = configuration(3);
    config
        .maximum_relative_bound_width
        .remove(&ComparisonMetric::TtftP99);
    assert!(FrozenStatisticalMethod::paired_cluster_bootstrap(config).is_err());
    let mut wire = serde_json::to_value(configuration(3)).unwrap();
    wire["declared_design"]["calibrated"] = serde_json::json!(true);
    assert!(serde_json::from_value::<FrozenPairedBootstrap>(wire).is_err());
}
