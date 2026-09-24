use super::*;
use ferrum_interfaces::{execution_cost::*, vnext::DeviceCommandPhase};
use std::num::NonZeroU64;

pub(crate) fn fingerprint() -> ExecutionFingerprint {
    ExecutionFingerprint {
        model_weights: [1; 32],
        numerical_policy: [2; 32],
        device_runtime: [3; 32],
        execution_config: [4; 32],
    }
}
pub(crate) fn settings() -> WholeWaveSettingsV1 {
    WholeWaveSettingsV1 {
        drift_margin_ns: 5,
        max_wave_ns: NonZeroU64::new(1_000_000).unwrap(),
        max_sample_age_ns: NonZeroU64::new(10_000).unwrap(),
        ..Default::default()
    }
}
pub(crate) fn partition() -> CalibrationPartitionV1 {
    CalibrationPartitionV1 {
        source_sha256: [7; 32],
        protocol_sha256: [8; 32],
        fit_through_ordinal: 8,
        residual_through_ordinal: 16,
    }
}
pub(crate) fn sample(ordinal: u64, count: u32, wall: u64) -> WholeWaveObservationV1 {
    sample_route(ordinal, count, wall, "selected.mma", false)
}
pub(crate) fn sample_route(
    ordinal: u64,
    count: u32,
    wall: u64,
    entry: &str,
    final_prefill: bool,
) -> WholeWaveObservationV1 {
    let mut command = SelectedCommandCostBuilderV1::new(count.into());
    command
        .kernel(
            SelectedAlgorithmClassV1::new(entry, 1, [5; 32], [6; 32]).unwrap(),
            KernelNumericWorkV1 {
                logical_units: u64::from(count) * 64,
                padded_units: u64::from(count.div_ceil(8)) * 8 * 64,
                inner_units_per_logical_unit: 256,
                grid: [count.div_ceil(8), 1, 1],
                scratch_bytes: 8192,
                staged_weight_bytes: 0,
            },
        )
        .unwrap();
    let evidence = command.finish().unwrap();
    let mut builder = CanonicalWaveCostBuilder::new(0, CostProductOutput::FullLogits);
    builder
        .physical_command(CostPhysicalCommand {
            native_op_id: entry,
            command_index: 0,
            node_index: None,
            command_phase: DeviceCommandPhase::Compute,
            provider: None,
            path: CostCommandPath::Eager,
            participant_start: 0,
            participant_count: 1,
            token_count: count.into(),
            batching_form: "packed",
            compute_dispatch_count: 1,
            transfer_command_count: 0,
            reusable_graph_node_count: None,
            statistical_evidence: Some(&evidence),
        })
        .unwrap();
    builder
        .core_readback_route(if final_prefill {
            CoreReadbackRoute::HostSynchronized
        } else {
            CoreReadbackRoute::NoReadback
        })
        .unwrap();
    builder
        .row(CanonicalCostRow {
            work: ActualRowWork::Prefill {
                offset: 0,
                count,
                total_prompt_tokens: if final_prefill { count } else { 64 },
            },
            host_policy_signature: [1; 32],
            mask_upload_required: false,
            output: CostRowOutput::Prefill {
                final_logits: final_prefill,
            },
            host_features: Some(HostCostFeaturesV1 {
                policy: HostCostPolicyV2 {
                    empirical_content_domain: Some(HostContentDomainV1::PlainTextGreedyV1),
                    categorical_signature: [2; 32],
                    decoder_text_bytes_per_token: 8,
                    decoder_scratch_bytes_per_token: 4,
                    raw_token_bytes_bound: 4,
                },
                state: HostCostStateV1 {
                    generated_tokens_before: 0,
                    maximum_output_tokens: 32,
                    sampling_history_tokens: 0,
                    sampling_history_scope: CostSamplingHistoryScope::FullGeneration,
                    pending_decoded_utf8: false,
                    completion_state_signature: satisfied_completion_cost_signature(),
                },
            }),
        })
        .unwrap();
    let canonical = builder
        .finish_with_statistics(
            ActualWaveKind::Prefill,
            ActualWavePath::PlanRuntime,
            ActualWaveGraphState::Disabled,
            ActualWaveRowOrder::Ordered,
            64,
        )
        .unwrap();
    WholeWaveObservationV1 {
        source_sha256: partition().source_sha256,
        accepted_ordinal: ordinal,
        call_id: ordinal,
        fingerprint: fingerprint(),
        exact: canonical.exact,
        selected: canonical.statistical.unwrap(),
        boundary: CostBoundary::PreparationToHostSettledV1,
        outcome: WaveObservationOutcome::Completed,
        observed_at_ns: 100 + ordinal,
        wall_ns: wall,
    }
}
pub(crate) fn populations() -> (Vec<WholeWaveObservationV1>, Vec<WholeWaveObservationV1>) {
    (
        (1..=8)
            .map(|i| sample(i, if i % 2 == 0 { 16 } else { 8 }, 100))
            .collect(),
        (9..=16)
            .map(|i| sample(i, if i % 2 == 0 { 16 } else { 8 }, 120))
            .collect(),
    )
}
pub(crate) fn fitted() -> FittedWholeWaveModelV1 {
    FittedWholeWaveModelV1::fit(
        fingerprint(),
        settings(),
        partition(),
        &populations().0,
        120,
    )
    .unwrap()
}
fn model() -> WholeWaveModelV1 {
    fitted().calibrate(&populations().1, 120).unwrap()
}
#[test]
fn independent_residual_calibrates_whole_wave_and_heldout_cannot_retrain() {
    let model = model();
    let heldout = sample(17, 12, 130);
    let prediction = model
        .predict(&fingerprint(), &heldout.exact, &heldout.selected, 120)
        .unwrap();
    assert_eq!(
        (prediction.fit_samples, prediction.residual_samples),
        (8, 8)
    );
    assert!(prediction.fitted_ns >= 100 && prediction.fitted_ns <= 101);
    assert_eq!(prediction.planning_ns, 125);
    assert_eq!(
        model
            .evaluate_heldout(&heldout, 120)
            .unwrap()
            .underestimate_ns,
        Some(5)
    );
    let mut extreme = heldout.clone();
    extreme.wall_ns = 999_999;
    assert!(
        model
            .evaluate_heldout(&extreme, 120)
            .unwrap()
            .underestimate_ns
            .unwrap()
            > 900_000
    );
    assert_eq!(
        model
            .predict(&fingerprint(), &heldout.exact, &heldout.selected, 120)
            .unwrap(),
        prediction
    );
}
#[test]
fn each_population_needs_eight_and_phase_source_cannot_leak() {
    let (fit, res) = populations();
    assert!(matches!(
        FittedWholeWaveModelV1::fit(fingerprint(), settings(), partition(), &fit[..7], 120),
        Err(ModelUnknown::InsufficientFit)
    ));
    assert!(matches!(
        fitted().calibrate(&res[..7], 120),
        Err(ModelUnknown::InsufficientResidual)
    ));
    assert!(matches!(
        fitted().calibrate(&fit, 120),
        Err(ModelUnknown::PhaseLeakage)
    ));
    assert_eq!(
        model().evaluate_heldout(&res[0], 120),
        Err(ModelUnknown::PhaseLeakage)
    );
    let mut bad = fit;
    bad[0].source_sha256 = [9; 32];
    assert!(matches!(
        FittedWholeWaveModelV1::fit(fingerprint(), settings(), partition(), &bad, 120),
        Err(ModelUnknown::WrongSource)
    ));
}
#[test]
fn support_unknown_is_not_a_lowered_cost_and_ttl_keeps_aging() {
    let model = model();
    let outside = sample(17, 24, 140);
    assert_eq!(
        model.predict(&fingerprint(), &outside.exact, &outside.selected, 120),
        Err(ModelUnknown::JointSupport)
    );
    let query = sample(17, 12, 120);
    let known = model
        .predict(&fingerprint(), &query.exact, &query.selected, 120)
        .unwrap();
    assert_eq!(known.valid_until_ns, 10101);
    assert!(model
        .predict(
            &fingerprint(),
            &query.exact,
            &query.selected,
            known.valid_until_ns
        )
        .is_ok());
    assert_eq!(
        model.predict(
            &fingerprint(),
            &query.exact,
            &query.selected,
            known.valid_until_ns + 1
        ),
        Err(ModelUnknown::Stale)
    );
    assert_eq!(
        model.predict(&fingerprint(), &query.exact, &query.selected, 119),
        Err(ModelUnknown::Clock)
    );
}
#[test]
fn failed_or_wrong_boundary_samples_and_capacity_are_not_training() {
    let (mut fit, _) = populations();
    fit[0].outcome = WaveObservationOutcome::FailedAfterSubmit;
    assert!(matches!(
        FittedWholeWaveModelV1::fit(fingerprint(), settings(), partition(), &fit, 120),
        Err(ModelUnknown::InvalidSample)
    ));
    fit[0].outcome = WaveObservationOutcome::Completed;
    fit[0].boundary = CostBoundary::PreparationToCommit;
    assert!(matches!(
        FittedWholeWaveModelV1::fit(fingerprint(), settings(), partition(), &fit, 120),
        Err(ModelUnknown::InvalidSample)
    ));
    let mut s = settings();
    s.max_retained_samples = std::num::NonZeroUsize::new(8).unwrap();
    let f =
        FittedWholeWaveModelV1::fit(fingerprint(), s, partition(), &populations().0, 120).unwrap();
    assert!(matches!(
        f.calibrate(&populations().1, 120),
        Err(ModelUnknown::Capacity)
    ));
}
#[test]
fn joint_support_does_not_combine_independent_coordinate_maxima() {
    let mut a = [0; support::AXES];
    let mut b = a;
    let mut q = a;
    a[0] = 8;
    a[1] = 2;
    b[0] = 2;
    b[1] = 8;
    q[0] = 8;
    q[1] = 8;
    let support = Support::new([a, b].into_iter()).unwrap();
    assert!(support.contains(&a));
    assert!(!support.contains(&q));
}
#[test]
fn final_and_algorithm_changes_cannot_borrow_another_family() {
    let model = model();
    for q in [
        sample_route(17, 12, 120, "different.selected.kernel", false),
        sample_route(17, 12, 120, "selected.mma", true),
    ] {
        assert_eq!(
            model.predict(&fingerprint(), &q.exact, &q.selected, 120),
            Err(ModelUnknown::FamilyMissing)
        );
    }
}
#[test]
fn q99_is_an_empirical_quantile_not_a_maximum_or_heldout_guarantee() {
    let mut s = settings();
    s.max_samples_per_bucket = std::num::NonZeroUsize::new(256).unwrap();
    let p = CalibrationPartitionV1 {
        residual_through_ordinal: 136,
        ..partition()
    };
    let residual: Vec<_> = (9..=136)
        .map(|i| {
            sample(
                i,
                if i % 2 == 0 { 16 } else { 8 },
                if i == 136 { 1000 } else { 120 },
            )
        })
        .collect();
    let model = FittedWholeWaveModelV1::fit(fingerprint(), s, p, &populations().0, 500)
        .unwrap()
        .calibrate(&residual, 500)
        .unwrap();
    let q = sample(137, 12, 1000);
    let evaluation = model.evaluate_heldout(&q, 500).unwrap();
    assert_eq!(evaluation.prediction.unwrap().planning_ns, 125);
    assert_eq!(evaluation.underestimate_ns, Some(875));
}

pub(crate) mod independent_rows;
