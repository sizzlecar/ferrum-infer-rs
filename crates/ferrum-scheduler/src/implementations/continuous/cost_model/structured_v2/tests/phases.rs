//! Numerical phase tests reuse actual typed canonical producer inputs. Timings
//! are synthetic, explicitly not a calibration-session or GPU performance test.
use super::*;
fn fp() -> ExecutionFingerprint {
    ExecutionFingerprint {
        model_weights: [1; 32],
        numerical_policy: [2; 32],
        device_runtime: [3; 32],
        execution_config: [4; 32],
    }
}
fn settings() -> StructuredSettingsV2 {
    StructuredSettingsV2 {
        max_sample_age_ns: 10_000,
        static_margin_ns: 20,
        ..Default::default()
    }
}
fn contract() -> StructuredSourceContractV2 {
    StructuredSourceContractV2 {
        capture_identity: [10; 32],
        protocol: [11; 32],
        membership_rule: [12; 32],
        cohort_manifest: [13; 32],
        phase_members: [16; 3],
    }
}
fn phase(phase: StructuredPhaseV2) -> Vec<StructuredNumericObservationV2> {
    let source = contract();
    let offset = match phase {
        StructuredPhaseV2::Fit => 0,
        StructuredPhaseV2::Residual => 16,
        StructuredPhaseV2::Qualification => 32,
    };
    (0..16)
        .map(|i| {
            let terminal = [9, 0, 1, 2][i / 4];
            let input = project(&wave(
                terminal,
                true,
                8,
                "fixture.first",
                [i & 1 != 0, i & 2 != 0],
            ))
            .unwrap();
            let n = (offset + i + 1) as u64;
            let wall_ns = 1000
                + input.pending_positions.len() as u64 * 13
                + input.length_positions.len() as u64 * 7;
            StructuredObservationV2 {
                source: source.capture_identity,
                protocol: source.protocol,
                ordinal: n * 3,
                membership: StructuredMemberBindingV2 {
                    rule_signature: source.membership_rule,
                    offered_ordinal: n * 4,
                    member_ordinal: n,
                    phase,
                },
                call_id: n + 100,
                fingerprint: fp(),
                input,
                boundary: CostBoundary::PreparationToHostSettledV1,
                outcome: WaveObservationOutcome::Completed,
                observed_at_ns: n * 10,
                wall_ns,
            }
        })
        .collect()
}
fn scope(samples: &[StructuredNumericObservationV2]) -> StructuredScopeV2 {
    StructuredScopeV2 {
        owner: samples[0].input.owner().clone(),
        coverage: StructuredCoverageV2 {
            pending_eligible_positions: vec![0, 1],
            authorized_pending_constraints: vec![
                HostPendingConstraintV2::AnySubset,
                HostPendingConstraintV2::NonEmptySubset,
            ],
            pending_counts: vec![0, 1, 2],
            length_counts: vec![0, 1, 2],
            pending_positions: vec![0, 1],
            length_positions: vec![0, 1],
            joint_counts: (0..3).flat_map(|p| (0..3).map(move |l| (p, l))).collect(),
        },
    }
}
fn fitted() -> FittedStructuredModelV2 {
    let fit = phase(StructuredPhaseV2::Fit);
    FittedStructuredModelV2::fit(fp(), settings(), scope(&fit), contract(), &fit, 170).unwrap()
}
fn calibrated() -> CalibratedStructuredModelV2 {
    fitted()
        .calibrate(&phase(StructuredPhaseV2::Residual), 330)
        .unwrap()
}
fn qualified() -> QualifiedStructuredModelV2 {
    calibrated()
        .qualify(&phase(StructuredPhaseV2::Qualification), 490)
        .unwrap()
}
#[test]
fn structured_v2_three_complete_phases_freeze_one_model_and_future_envelope() {
    let model = qualified();
    let actual = phase(StructuredPhaseV2::Qualification);
    let input = actual[0].input.clone();
    let query = StructuredQueryV2 {
        input,
        pending: Some(PendingQuery {
            eligible: vec![0, 1],
            constraint: HostPendingConstraintV2::AnySubset,
        }),
    };
    let before = model.parameters_signature();
    let p = model.predict_query(&fp(), &query, 500).unwrap();
    assert!((1000..=1001).contains(&p.fitted_lower_ns));
    assert!((1026..=1027).contains(&p.fitted_upper_ns));
    assert_eq!(p.valid_until_ns, 10010);
    assert_eq!(
        (
            p.fit_samples,
            p.residual_samples,
            model.qualification_samples
        ),
        (16, 16, 16)
    );
    assert_eq!(before, model.parameters_signature());
    assert_eq!(
        model.source_contract().cohort_manifest,
        contract().cohort_manifest
    );
    assert_eq!(model.owner(), &scope(&actual).owner);
    assert_eq!(model.domain_signature(), query.domain_signature());
    assert!(matches!(
        model.predict_query(&fp(), &query, 10011),
        Err(StructuredUnknown::Stale)
    ));
    assert!(matches!(
        model.predict_query(&fp(), &query, 489),
        Err(StructuredUnknown::Clock)
    ));
    let mut wrong = fp();
    wrong.device_runtime = [99; 32];
    assert!(matches!(
        model.predict_query(&wrong, &query, 500),
        Err(StructuredUnknown::WrongFingerprint)
    ));
}
#[test]
fn structured_v2_cannot_omit_slow_member_in_any_phase() {
    let mut fit = phase(StructuredPhaseV2::Fit);
    fit[7].wall_ns += 5000;
    let s = scope(&fit);
    fit.remove(7);
    assert!(matches!(
        FittedStructuredModelV2::fit(fp(), settings(), s, contract(), &fit, 170),
        Err(StructuredUnknown::IncompletePhasePopulation)
    ));
    let mut residual = phase(StructuredPhaseV2::Residual);
    residual[7].wall_ns += 5000;
    residual.remove(7);
    assert!(matches!(
        fitted().calibrate(&residual, 330),
        Err(StructuredUnknown::IncompletePhasePopulation)
    ));
    let mut qualification = phase(StructuredPhaseV2::Qualification);
    qualification.remove(7);
    assert!(matches!(
        calibrated().qualify(&qualification, 490),
        Err(StructuredUnknown::IncompletePhasePopulation)
    ));
}
#[test]
fn structured_v2_source_protocol_member_and_fifo_are_independently_bound() {
    let mut samples = phase(StructuredPhaseV2::Residual);
    samples[1].protocol = [99; 32];
    assert!(matches!(
        fitted().calibrate(&samples, 330),
        Err(StructuredUnknown::WrongProtocol)
    ));
    samples = phase(StructuredPhaseV2::Residual);
    samples[1].source = [99; 32];
    assert!(matches!(
        fitted().calibrate(&samples, 330),
        Err(StructuredUnknown::WrongSource)
    ));
    samples = phase(StructuredPhaseV2::Residual);
    samples[1].membership.member_ordinal = samples[0].membership.member_ordinal;
    assert!(matches!(
        fitted().calibrate(&samples, 330),
        Err(StructuredUnknown::DuplicateRecord)
    ));
    samples = phase(StructuredPhaseV2::Residual);
    samples[1].ordinal = samples[0].ordinal;
    assert!(matches!(
        fitted().calibrate(&samples, 330),
        Err(StructuredUnknown::DuplicateRecord)
    ));
    samples = phase(StructuredPhaseV2::Residual);
    samples[1].membership.offered_ordinal = samples[0].membership.offered_ordinal;
    assert!(matches!(
        fitted().calibrate(&samples, 330),
        Err(StructuredUnknown::DuplicateRecord)
    ));
    samples = phase(StructuredPhaseV2::Residual);
    samples[1].call_id = 101;
    assert!(matches!(
        fitted().calibrate(&samples, 330),
        Err(StructuredUnknown::DuplicateRecord)
    ));
}
#[test]
fn structured_v2_later_phases_cannot_backfill_before_original_freeze() {
    let mut residual = phase(StructuredPhaseV2::Residual);
    residual[0].observed_at_ns = 169;
    assert!(matches!(
        fitted().calibrate(&residual, 330),
        Err(StructuredUnknown::Clock)
    ));
    let mut qual = phase(StructuredPhaseV2::Qualification);
    qual[0].observed_at_ns = 329;
    assert!(matches!(
        calibrated().qualify(&qual, 490),
        Err(StructuredUnknown::Clock)
    ));
    assert!(fitted()
        .calibrate(&phase(StructuredPhaseV2::Residual), 330)
        .is_ok()); // first sample at exact fit freeze 170
    assert!(matches!(
        calibrated().qualify(&phase(StructuredPhaseV2::Qualification), 10011),
        Err(StructuredUnknown::Stale)
    ));
}
#[test]
fn structured_v2_underestimate_fails_qualification_without_refit() {
    let mut samples = phase(StructuredPhaseV2::Qualification);
    samples[0].wall_ns += 100;
    assert!(matches!(
        calibrated().qualify(&samples, 490),
        Err(StructuredUnknown::QualificationUnderestimate)
    ));
    let mut samples = phase(StructuredPhaseV2::Residual);
    samples[0].boundary = CostBoundary::PreparationToCommit;
    assert!(matches!(
        fitted().calibrate(&samples, 330),
        Err(StructuredUnknown::InvalidSample)
    ));
}
#[test]
fn structured_v2_authorization_requires_actual_joint_and_position_challenges() {
    let samples = phase(StructuredPhaseV2::Qualification);
    let mut s = scope(&samples);
    let query = StructuredQueryV2 {
        input: samples[0].input.clone(),
        pending: Some(PendingQuery {
            eligible: vec![0, 1],
            constraint: HostPendingConstraintV2::AnySubset,
        }),
    };
    s.coverage.joint_counts.retain(|pair| *pair != (1, 0));
    assert!(matches!(
        s.authorize(&query),
        Err(StructuredUnknown::QualificationCoverage)
    ));
    s = scope(&samples);
    s.coverage.pending_counts = vec![0, 1];
    assert!(matches!(
        s.authorize(&query),
        Err(StructuredUnknown::QualificationCoverage)
    ));
    s = scope(&samples);
    s.coverage.length_positions.clear();
    assert!(matches!(
        s.authorize(&StructuredQueryV2::exact(samples[4].input.clone())),
        Err(StructuredUnknown::QualificationCoverage)
    ));
    let partial = &samples[..12];
    let report = scope(&samples).coverage_report(partial).unwrap();
    assert_eq!(report.missing.length_counts, [2]);
    assert!(report.missing.joint_counts.contains(&(2, 2)));
    assert!(!report.complete());
    assert!(scope(&samples)
        .coverage_report(&samples)
        .unwrap()
        .complete());
}
#[test]
fn structured_v2_conditional_empty_greedy_set_is_a_legal_singleton() {
    // Forecast production separately verifies the actual Greedy physical route.
    // This exercises the numerical singleton contract without minting a receipt.
    let samples = phase(StructuredPhaseV2::Qualification);
    let model = qualified();
    let query = StructuredQueryV2 {
        input: samples[0].input.clone(),
        pending: Some(PendingQuery {
            eligible: vec![],
            constraint: HostPendingConstraintV2::AnySubset,
        }),
    };
    let p = model.predict_query(&fp(), &query, 500).unwrap();
    assert_eq!(p.fitted_lower_ns, p.fitted_upper_ns);
    assert!(query.pending.is_some());
    let mut s = scope(&samples);
    s.coverage.pending_eligible_positions.clear();
    s.coverage.authorized_pending_constraints = vec![HostPendingConstraintV2::AnySubset];
    let report = s.coverage_report(&samples[..1]).unwrap();
    assert!(report.empty_challenge_seen && report.full_challenge_seen);
    assert!(!report.missing_intermediate_challenge);
}
#[test]
fn structured_v2_support_rejects_overflow_before_consuming_unbounded_iterator() {
    let point = [1u64];
    let mut count = 0;
    let result = JointSupport::new(std::iter::repeat_with(|| {
        count += 1;
        point.as_slice()
    }));
    assert!(matches!(result, Err(StructuredUnknown::Capacity)));
    assert_eq!(count, 4097);
    let oversized = vec![1; 4097];
    assert!(matches!(
        JointSupport::new(std::iter::once(oversized.as_slice())),
        Err(StructuredUnknown::Capacity)
    ));
}
