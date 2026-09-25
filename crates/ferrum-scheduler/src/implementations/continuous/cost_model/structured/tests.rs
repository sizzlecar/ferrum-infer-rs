use super::*;
use ferrum_interfaces::execution_cost::{
    HostContentDomainV1, HostCostPolicyV2, HostRowRoleV2, HostTerminalExpectationV1,
    StructuredHostRowV1,
};

fn fingerprint() -> ExecutionFingerprint {
    ExecutionFingerprint {
        model_weights: [1; 32],
        numerical_policy: [2; 32],
        device_runtime: [3; 32],
        execution_config: [4; 32],
    }
}
fn settings() -> StructuredSettingsV1 {
    StructuredSettingsV1 {
        static_margin_ns: 0,
        max_sample_age_ns: 10_000,
        ..Default::default()
    }
}
fn partition() -> StructuredPartitionV1 {
    StructuredPartitionV1 {
        source: [7; 32],
        protocol: [8; 32],
        fit_through: 16,
        residual_through: 32,
        qualification_through: 41,
    }
}
fn host_rows(rows: usize, terminal: Option<usize>) -> Vec<StructuredHostRowV1> {
    (0..rows)
        .map(|p| StructuredHostRowV1 {
            physical_position: p as u32,
            role: HostRowRoleV2::Decode,
            installed_policy: HostCostPolicyV2 {
                empirical_content_domain: Some(HostContentDomainV1::PlainTextGreedyV1),
                categorical_signature: [9; 32],
                decoder_text_bytes_per_token: 4,
                decoder_scratch_bytes_per_token: 4,
                raw_token_bytes_bound: 4,
            },
            no_generated_history: false,
            pending_decoded_utf8: false,
            initial_prefill: false,
            final_prefill: false,
            mask_upload_required: false,
            decode_requires_full_logits: Some(true),
            repetition_penalty_bits: Some(1.0_f32.to_bits()),
            terminal_expectation: if terminal == Some(p) {
                HostTerminalExpectationV1::LengthBoundary
            } else {
                HostTerminalExpectationV1::TokenMayTerminate
            },
        })
        .collect()
}
fn input(work: u64, terminal: Option<usize>) -> StructuredInputV1 {
    let tail = input::terminal_basis(terminal);
    let mut basis = vec![1.0, work as f64, (work * 2) as f64]; // Deliberate exact collinearity.
    basis.extend(tail.map(|n| n as f64));
    let mut support = vec![work, work * 2];
    support.extend(tail);
    StructuredInputV1 {
        domain: [11; 32],
        scope: StructuredScopeV1::OrdinaryDecodeSingleLength { rows: 8 },
        basis,
        support,
        physical_host_rows: host_rows(8, terminal),
        terminal_position: terminal,
    }
}
fn target(work: u64, terminal: Option<usize>) -> u64 {
    let [count, p, p2] = input::terminal_basis(terminal);
    1000 + 2 * work + 30 * count + 4 * p + 3 * p2
}
fn observation(
    ordinal: u64,
    work: u64,
    terminal: Option<usize>,
    extra: u64,
) -> StructuredNumericObservationV1 {
    StructuredNumericObservationV1 {
        source: [7; 32],
        protocol: [8; 32],
        ordinal,
        call_id: ordinal,
        fingerprint: fingerprint(),
        input: input(work, terminal),
        boundary: CostBoundary::PreparationToHostSettledV1,
        outcome: WaveObservationOutcome::Completed,
        observed_at_ns: ordinal,
        wall_ns: target(work, terminal) + extra,
    }
}
fn fit_samples() -> Vec<StructuredNumericObservationV1> {
    (0..16)
        .map(|i| {
            observation(
                i + 1,
                [64, 80, 96, 128][(i / 4) as usize],
                [None, Some(0), Some(3), Some(7)][(i % 4) as usize],
                0,
            )
        })
        .collect()
}
fn residual_samples() -> Vec<StructuredNumericObservationV1> {
    (0..16)
        .map(|i| {
            observation(
                i + 17,
                [64, 80, 96, 128][(i / 4) as usize],
                [None, Some(0), Some(2), Some(7)][(i % 4) as usize],
                20,
            )
        })
        .collect()
}
fn qualification_samples() -> Vec<StructuredNumericObservationV1> {
    (0..9)
        .map(|i| {
            observation(
                i + 33,
                96,
                if i == 0 { None } else { Some(i as usize - 1) },
                15,
            )
        })
        .collect()
}
fn calibrated() -> CalibratedStructuredModelV1 {
    FittedStructuredModelV1::fit(fingerprint(), settings(), partition(), &fit_samples(), 16)
        .unwrap()
        .calibrate(&residual_samples(), 32)
        .unwrap()
}

#[test]
fn shares_one_fit_across_positions_without_per_bitmap_minimum() {
    let model = calibrated().qualify(&qualification_samples(), 41).unwrap();
    assert_eq!(model.qualification_samples, 9);
    for position in 0..8 {
        let prediction = model
            .predict(&fingerprint(), &input(96, Some(position)), 41)
            .unwrap();
        assert_eq!(prediction.identified_rank, 5); // six columns, one exact dependency.
        assert_eq!(prediction.fit_samples, 16);
        assert_eq!(prediction.residual_samples, 16);
        assert!((19..=20).contains(&prediction.residual_ns));
        assert!(prediction.planning_ns >= target(96, Some(position)) + 15);
        assert!(prediction.fitted_ns.abs_diff(target(96, Some(position))) <= 1);
    }
    // Fit has no position 1/2/4/5/6 and only four observations per seen pattern;
    // independent heldout challenges the whole declared position scope anyway.
    assert!(fit_samples()
        .iter()
        .all(|s| s.input.terminal_position != Some(5)));
}

#[test]
fn one_failed_position_disables_the_whole_scope() {
    let mut heldout = qualification_samples();
    heldout[6].wall_ns += 100;
    assert!(matches!(
        calibrated().qualify(&heldout, 41),
        Err(StructuredUnknown::QualificationUnderestimate)
    ));
}

#[test]
fn missing_or_filtered_position_is_not_qualified() {
    let mut heldout = qualification_samples();
    heldout[8].input = input(96, Some(6));
    heldout[8].wall_ns = target(96, Some(6));
    assert!(matches!(
        calibrated().qualify(&heldout, 41),
        Err(StructuredUnknown::QualificationCoverage)
    ));
    let mut omitted = qualification_samples();
    omitted.remove(4);
    assert!(matches!(
        calibrated().qualify(&omitted, 41),
        Err(StructuredUnknown::IncompletePhasePopulation)
    ));
}

#[test]
fn qualification_is_not_used_to_expand_support_or_refit_residual() {
    let mut heldout = qualification_samples();
    heldout[8] = observation(41, 160, Some(7), 0);
    assert!(matches!(
        calibrated().qualify(&heldout, 41),
        Err(StructuredUnknown::JointSupport)
    ));
}

#[test]
fn query_direction_must_be_identified_even_with_coordinate_support() {
    let model = calibrated().qualify(&qualification_samples(), 41).unwrap();
    let mut query = input(96, Some(2));
    query.basis[2] = 193.0;
    query.support[1] = 193; // within extrema, breaks learned equality.
    assert!(matches!(
        model.predict(&fingerprint(), &query, 41),
        Err(StructuredUnknown::UnidentifiedDirection)
    ));
}

#[test]
fn support_does_not_create_coordinatewise_maximum() {
    let support = JointSupport::new([[10, 100], [100, 10]].iter().map(|p| p.as_slice())).unwrap();
    assert!(support.contains(&[10, 90]));
    assert!(!support.contains(&[90, 90]));
    assert!(!support.contains(&[9, 10]));
    assert!(!support.contains(&[10, 100, 0]));
}

#[test]
fn numerical_near_dependence_is_not_silently_discarded() {
    let mut samples = fit_samples();
    for (i, sample) in samples.iter_mut().enumerate() {
        sample.input.basis[2] += if i == 0 { 0.000001 } else { 0.0 };
    }
    assert!(matches!(
        RowSpaceFit::fit(&samples, &settings()),
        Err(StructuredUnknown::IllConditioned)
    ));
}

#[test]
fn fit_requires_redundancy_beyond_identified_rank() {
    let mut samples = fit_samples();
    samples.truncate(8);
    assert!(matches!(
        RowSpaceFit::fit(&samples, &settings()),
        Err(StructuredUnknown::InsufficientRedundancy)
    ));
}

#[test]
fn ttl_uses_oldest_fit_sample_and_qualification_does_not_refresh_it() {
    let model = calibrated().qualify(&qualification_samples(), 41).unwrap();
    let query = input(96, Some(4));
    assert_eq!(
        model
            .predict(&fingerprint(), &query, 41)
            .unwrap()
            .valid_until_ns,
        10001
    );
    assert!(model.predict(&fingerprint(), &query, 10001).is_ok());
    assert!(matches!(
        model.predict(&fingerprint(), &query, 10002),
        Err(StructuredUnknown::Stale)
    ));
    assert!(matches!(
        model.predict(&fingerprint(), &query, 40),
        Err(StructuredUnknown::Clock)
    ));
}

#[test]
fn phases_reject_duplicate_calls_and_time_travel() {
    let fitted =
        FittedStructuredModelV1::fit(fingerprint(), settings(), partition(), &fit_samples(), 16)
            .unwrap();
    let mut residual = residual_samples();
    residual[0].call_id = 1;
    assert!(matches!(
        fitted.calibrate(&residual, 32),
        Err(StructuredUnknown::DuplicateRecord)
    ));
    let fitted =
        FittedStructuredModelV1::fit(fingerprint(), settings(), partition(), &fit_samples(), 16)
            .unwrap();
    let mut residual = residual_samples();
    residual[0].ordinal = 16;
    assert!(matches!(
        fitted.calibrate(&residual, 32),
        Err(StructuredUnknown::PhaseLeakage)
    ));
    let mut fit = fit_samples();
    fit[3].observed_at_ns = 1;
    assert!(matches!(
        FittedStructuredModelV1::fit(fingerprint(), settings(), partition(), &fit, 16),
        Err(StructuredUnknown::Clock)
    ));
}

#[test]
fn sources_fingerprints_and_complete_host_boundary_are_checked() {
    for scenario in 0..3 {
        let mut samples = fit_samples();
        let expected = match scenario {
            0 => {
                samples[0].source = [33; 32];
                StructuredUnknown::WrongSource
            }
            1 => {
                samples[0].fingerprint.device_runtime = [33; 32];
                StructuredUnknown::WrongFingerprint
            }
            _ => {
                samples[0].boundary = CostBoundary::PreparationToCommit;
                StructuredUnknown::InvalidSample
            }
        };
        assert!(
            matches!(FittedStructuredModelV1::fit(fingerprint(),settings(),partition(),&samples,16),Err(reason) if reason == expected)
        );
    }
}

#[test]
fn unqualified_host_modes_do_not_fall_back_to_terminal_count() {
    for scenario in 0..6 {
        let mut rows = host_rows(8, Some(2));
        match scenario {
            0 => rows[3].terminal_expectation = HostTerminalExpectationV1::LengthBoundary,
            1 => rows[3].pending_decoded_utf8 = true,
            2 => rows[3].mask_upload_required = true,
            3 => rows[3].role = HostRowRoleV2::Prefill,
            4 => rows[3].no_generated_history = true,
            _ => rows[3].installed_policy.categorical_signature = [22; 32],
        }
        assert!(matches!(
            input::ordinary_decode_position(&rows),
            Err(StructuredUnknown::UnsupportedScope)
        ));
    }
}

#[test]
fn raw_position_mutations_cannot_leave_numeric_evidence_unchanged() {
    let mut query = input(96, Some(2));
    query.physical_host_rows.swap(2, 3);
    assert!(query.validate(&settings()).is_err());
    let mut query = input(96, Some(2));
    query.terminal_position = Some(3);
    assert!(query.validate(&settings()).is_err());
}

#[test]
fn query_never_uses_other_device_domain_or_fingerprint() {
    let model = calibrated().qualify(&qualification_samples(), 41).unwrap();
    let mut query = input(96, Some(2));
    query.domain = [44; 32];
    assert!(matches!(
        model.predict(&fingerprint(), &query, 41),
        Err(StructuredUnknown::WrongDomain)
    ));
    let mut changed = fingerprint();
    changed.numerical_policy = [44; 32];
    assert!(matches!(
        model.predict(&changed, &input(96, Some(2)), 41),
        Err(StructuredUnknown::WrongFingerprint)
    ));
}

#[test]
fn population_floor_and_finite_feature_limits_cannot_be_lowered() {
    let mut invalid = settings();
    invalid.min_phase_samples = 7;
    assert!(matches!(
        invalid.validate(),
        Err(StructuredUnknown::InvalidSettings)
    ));
    let mut query = input(96, Some(2));
    query.basis[1] = f64::NAN;
    assert!(matches!(
        query.validate(&settings()),
        Err(StructuredUnknown::InvalidInput)
    ));
    let mut too_wide = settings();
    too_wide.max_axes = 4;
    assert!(matches!(
        input(96, Some(2)).validate(&too_wide),
        Err(StructuredUnknown::Capacity)
    ));
}

#[test]
fn every_phase_requires_its_entire_declared_population() {
    let mut fit = fit_samples();
    fit[3].wall_ns += 1_000;
    fit.remove(3); // Dropping the slow fit point cannot redefine the population.
    assert!(matches!(
        FittedStructuredModelV1::fit(fingerprint(), settings(), partition(), &fit, 16),
        Err(StructuredUnknown::IncompletePhasePopulation)
    ));
    let fitted =
        FittedStructuredModelV1::fit(fingerprint(), settings(), partition(), &fit_samples(), 16)
            .unwrap();
    let mut residual = residual_samples();
    residual[7].wall_ns += 1_000;
    residual.remove(7); // Nor may the empirical residual tail be filtered.
    assert!(matches!(
        fitted.calibrate(&residual, 32),
        Err(StructuredUnknown::IncompletePhasePopulation)
    ));
    let mut fit = fit_samples();
    fit[3].ordinal = fit[2].ordinal;
    assert!(matches!(
        FittedStructuredModelV1::fit(fingerprint(), settings(), partition(), &fit, 16),
        Err(StructuredUnknown::DuplicateRecord)
    ));
}

#[test]
fn every_phase_binds_the_original_capture_protocol() {
    let mut fit = fit_samples();
    fit[0].protocol = [99; 32];
    assert!(matches!(
        FittedStructuredModelV1::fit(fingerprint(), settings(), partition(), &fit, 16),
        Err(StructuredUnknown::WrongProtocol)
    ));
    let fitted =
        FittedStructuredModelV1::fit(fingerprint(), settings(), partition(), &fit_samples(), 16)
            .unwrap();
    let mut residual = residual_samples();
    residual[0].protocol = [99; 32];
    assert!(matches!(
        fitted.calibrate(&residual, 32),
        Err(StructuredUnknown::WrongProtocol)
    ));
    let mut heldout = qualification_samples();
    heldout[0].protocol = [99; 32];
    assert!(matches!(
        calibrated().qualify(&heldout, 41),
        Err(StructuredUnknown::WrongProtocol)
    ));
}

#[test]
fn later_phases_cannot_backfill_measurements_from_before_the_freeze() {
    let fitted =
        FittedStructuredModelV1::fit(fingerprint(), settings(), partition(), &fit_samples(), 100)
            .unwrap();
    assert!(matches!(
        fitted.calibrate(&residual_samples(), 200),
        Err(StructuredUnknown::Clock)
    ));
    let fitted =
        FittedStructuredModelV1::fit(fingerprint(), settings(), partition(), &fit_samples(), 16)
            .unwrap();
    let calibrated = fitted.calibrate(&residual_samples(), 40).unwrap();
    assert!(matches!(
        calibrated.qualify(&qualification_samples(), 100),
        Err(StructuredUnknown::Clock)
    ));
    // Equality is valid in a monotonic clock with finite resolution.
    let fitted =
        FittedStructuredModelV1::fit(fingerprint(), settings(), partition(), &fit_samples(), 16)
            .unwrap();
    let mut residual = residual_samples();
    residual[0].observed_at_ns = 16;
    let calibrated = fitted.calibrate(&residual, 32).unwrap();
    let mut heldout = qualification_samples();
    heldout[0].observed_at_ns = 32;
    assert!(calibrated.qualify(&heldout, 41).is_ok());
}

#[test]
fn redundancy_settings_cannot_overflow_or_exceed_the_possible_population() {
    for redundancy in [usize::MAX, 256] {
        let mut invalid = settings();
        invalid.min_fit_redundancy = redundancy;
        assert!(matches!(
            FittedStructuredModelV1::fit(fingerprint(), invalid, partition(), &fit_samples(), 16),
            Err(StructuredUnknown::InvalidSettings)
        ));
    }
    let mut invalid = settings();
    invalid.min_fit_redundancy = usize::MAX;
    assert!(matches!(
        RowSpaceFit::fit(&fit_samples(), &invalid),
        Err(StructuredUnknown::InvalidSettings)
    ));
    let mut boundary = settings();
    boundary.min_fit_redundancy = 11; // 16 samples minus the identified rank 5.
    assert!(FittedStructuredModelV1::fit(
        fingerprint(),
        boundary.clone(),
        partition(),
        &fit_samples(),
        16
    )
    .is_ok());
    boundary.min_fit_redundancy = 12;
    assert!(matches!(
        FittedStructuredModelV1::fit(fingerprint(), boundary, partition(), &fit_samples(), 16),
        Err(StructuredUnknown::InsufficientRedundancy)
    ));
}
