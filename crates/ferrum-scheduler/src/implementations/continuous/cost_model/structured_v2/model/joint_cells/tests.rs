use super::super::super::nonnegative::{FitSample, NonNegativeFit};
use super::super::physical_envelope::envelope;
use super::*;
use ferrum_interfaces::{execution_cost::*, vnext::DeviceCommandPhase};
#[path = "../physical_envelope/numerical_family_tests/fixture.rs"]
mod fixture;

fn contract() -> NonNegativeEnvelopeContractV1 {
    NonNegativeEnvelopeContractV1 {
        algorithm_universe: None,
        planning_estimator: NonNegativePlanningEstimatorV1::IdentifiedFitJointCellsV1,
        workload_domain: fixture::domain(),
        settings: Default::default(),
        challenge: WorkAxisAndBranchChallengesV1::WorkAxesAndPlainTextBranchesV1,
        template_policy: StructuredCostTemplatePolicyV1::OrderedV1,
        population_policy: StructuredPopulationPolicyV1::HomogeneousOrdinaryDecodeV1,
    }
}

#[test]
fn joint_bank_matches_complete_cells_and_preserves_zero_and_dimension_boundaries() {
    let a = JointCellKey::of(&[128, 1], &[1]);
    let b = JointCellKey::of(&[1, 128], &[1]);
    let seen = std::collections::BTreeSet::from([a.clone(), b]);
    assert!(!seen.contains(&JointCellKey::of(&[128, 128], &[1])));
    assert_eq!(a, JointCellKey::of(&[255, 1], &[1]));
    assert_ne!(a, JointCellKey::of(&[256, 1], &[1]));
    assert_ne!(JointCellKey::of(&[0], &[1]), JointCellKey::of(&[1], &[1]));
    assert_ne!(
        JointCellKey::of(&[1, 1], &[1]),
        JointCellKey::of(&[1], &[1, 1])
    );
}

#[test]
fn joint_bank_completion_outcome_cannot_choose_a_cell_or_its_work_forecast() {
    let mut wave = fixture::Wave::default();
    wave.policy.empirical_content_domain = Some(HostContentDomainV1::PlainTextInstalledV2(
        PlainTextPolicyCapabilityV2 {
            sampling: PlainTextSamplingRouteV2::Greedy {
                repetition_penalty: false,
            },
            model_eos: true,
            user_stop: true,
        },
    ));
    let original = fixture::checked(&wave.build())
        .unwrap()
        .original_input()
        .clone();
    let continued = original.clone().with_settled_terminal_causes(&[]).unwrap();
    let terminal = original
        .clone()
        .with_settled_terminal_causes(&[(0, ferrum_types::FinishReason::EOS)])
        .unwrap();
    assert_ne!(
        continued.joint_support_coordinates(),
        terminal.joint_support_coordinates()
    );
    assert_eq!(prospective_input(&continued).unwrap(), original);
    assert_eq!(prospective_input(&terminal).unwrap(), original);
    assert_eq!(
        input_key(&continued, &contract()).unwrap(),
        input_key(&terminal, &contract()).unwrap()
    );
}

#[test]
fn joint_bank_stops_on_first_complete_input_count_without_reading_labels() {
    let (_, mut samples) = fixture::population(StructuredPhaseV2::Residual, false);
    let settings = StructuredSettingsV2::default();
    let schedule = OwnerBlockScheduleV1::new_with_input_readiness(
        24,
        [24; 3],
        [8; 3],
        OwnerInputReadinessV1::new([4; 3], 32_000_000).unwrap(),
    )
    .unwrap();
    // Three cells each have seven members: the family total cannot stand in
    // for the independently declared per-cell minimum.
    assert_eq!(
        assess_inputs(
            &schedule,
            &contract(),
            StructuredPhaseV2::Residual,
            24,
            &samples[..21],
            &settings,
            &mut 0
        )
        .unwrap(),
        OwnerInputReadinessDecisionV1::Wait
    );
    let ready = assess_inputs(
        &schedule,
        &contract(),
        StructuredPhaseV2::Residual,
        24,
        &samples,
        &settings,
        &mut 0,
    )
    .unwrap();
    assert_eq!(ready, OwnerInputReadinessDecisionV1::Freeze);
    for (i, s) in samples.iter_mut().enumerate() {
        s.wall_ns = if i % 2 == 0 { 0 } else { u64::MAX };
    }
    assert_eq!(
        assess_inputs(
            &schedule,
            &contract(),
            StructuredPhaseV2::Residual,
            24,
            &samples,
            &settings,
            &mut 0
        )
        .unwrap(),
        ready
    );
    let mut visits = schedule
        .input_readiness
        .as_ref()
        .unwrap()
        .maximum_geometry_visits
        / 2;
    assert_eq!(
        assess_inputs(
            &schedule,
            &contract(),
            StructuredPhaseV2::Residual,
            24,
            &samples,
            &settings,
            &mut visits
        )
        .unwrap(),
        OwnerInputReadinessDecisionV1::Exhausted(OwnerInputReadinessGapV1::GeometryWorkBudget)
    );
    let mut bounded = settings;
    bounded.max_axes = 1;
    assert!(groups(&samples, &contract(), &bounded).is_err());
}

fn phase_samples(phase: StructuredPhaseV2) -> Vec<StructuredNumericObservationV2> {
    let (_, mut samples) = fixture::population(phase, false);
    for s in &mut samples {
        s.membership.member_ordinal = s.ordinal;
        s.ordinal += 24;
        s.membership.offered_ordinal += 24;
        s.observed_at_ns += 240;
        s.input = s
            .input
            .clone()
            .with_cost_template_policy(StructuredCostTemplatePolicyV1::InstalledAlgorithmSetV1)
            .unwrap();
    }
    samples
}

fn phase_close(
    phase: StructuredPhaseV2,
    samples: &[StructuredNumericObservationV2],
    previous: Option<[u8; 32]>,
) -> StructuredOwnerPhaseCloseV1 {
    let i = phase.index() as u64;
    StructuredOwnerPhaseCloseV1::new(
        phase,
        StructuredOwnerPhaseBoundaryV1 {
            first_block: i + 2,
            last_block: i + 2,
            first_offered: (i + 1) * 24 + 1,
            last_offered: (i + 2) * 24,
            opening_fifo_cutoff: (i + 1) * 24,
            closing_fifo_cutoff: (i + 2) * 24,
            opened_at_ns: (i + 1) * 240,
            frozen_at_ns: (i + 2) * 240,
        },
        [20 + i as u8; 32],
        previous,
        samples,
    )
}

fn candidate_fit() -> FittedStructuredModelV2 {
    let samples = phase_samples(StructuredPhaseV2::Fit);
    let settings = StructuredSettingsV2 {
        max_sample_age_ns: 10_000,
        static_margin_ns: 100,
        ..Default::default()
    };
    let mut numerical = contract();
    numerical.template_policy = StructuredCostTemplatePolicyV1::InstalledAlgorithmSetV1;
    let mut schedule = OwnerBlockScheduleV1::new(24, [24; 3], [8; 3]).unwrap();
    schedule.prediction_validity = Some(OwnerPredictionValidityPolicyV1::OriginalSampleAgeV1);
    let contract = StructuredOwnerPhaseContractV1 {
        capture_identity: [10; 32],
        protocol: [11; 32],
        membership_rule: [12; 32],
        declaration_sha256: [13; 32],
        owner_attempt_id: 1,
        schedule,
        discovery_block: 1,
        discovery_offered_cutoff: 24,
        discovery_fifo_cutoff: 24,
        discovery_closed_at_ns: 240,
        expires_at_ns: 10_240,
        domain_policy: StructuredServiceDomainPolicyV1::NonNegativePhysicalEnvelopeV1,
        nonnegative_envelope: Some(numerical),
        input_target: None,
    };
    let scope = StructuredScopeV2 {
        owner: samples[0].input.owner().clone(),
        numerical_family: Some(samples[0].input.numerical_family_key().unwrap()),
        coverage: StructuredCoverageV2 {
            pending_eligible_positions: vec![],
            authorized_pending_constraints: vec![],
            pending_counts: vec![0],
            length_counts: vec![0],
            pending_positions: vec![],
            length_positions: vec![],
            joint_counts: vec![(0, 0)],
        },
    };
    FittedStructuredModelV2::fit_owner_blocks(
        samples[0].fingerprint.clone(),
        settings,
        scope,
        contract,
        phase_close(StructuredPhaseV2::Fit, &samples, None),
        &samples,
    )
    .unwrap()
}

#[test]
fn joint_bank_failed_q_cell_stays_unknown_without_refit_margin_or_expiry_renewal() {
    let fitted = candidate_fit();
    let fit_certificate = fitted.nonnegative_fit_certificate().unwrap().clone();
    let expiry = fitted.state.expires_at_ns;
    let residual = phase_samples(StructuredPhaseV2::Residual);
    let rclose = phase_close(
        StructuredPhaseV2::Residual,
        &residual,
        Some(fitted.parameters_signature()),
    );
    let calibrated = fitted.calibrate_owner_blocks(rclose, &residual).unwrap();
    let r_signature = calibrated.parameters_signature();
    let mut qualification = phase_samples(StructuredPhaseV2::Qualification);
    for s in &mut qualification {
        if s.input.owner().rows == 2 {
            s.wall_ns += 100_000;
        }
    }
    let qclose = phase_close(
        StructuredPhaseV2::Qualification,
        &qualification,
        Some(r_signature),
    );
    let model = calibrated
        .qualify_owner_blocks(qclose, &qualification)
        .unwrap();
    assert_eq!(model.nonnegative_fit_certificate(), Some(&fit_certificate));
    assert_eq!(
        model.owner_block_phase_signatures().unwrap()[1],
        r_signature
    );
    assert_eq!(model.calibrated.fitted.state.expires_at_ns, expiry);
    for sample in &qualification {
        let query = StructuredQueryV2::exact(sample.input.clone());
        let result = model.predict_query(&sample.fingerprint, &query, 961);
        if sample.input.owner().rows == 2 {
            assert!(result.is_err());
            assert_eq!(model.catalog_input_membership(&query).unwrap(), Some(false));
        } else {
            let prediction = result.unwrap();
            assert!(prediction.planning_ns >= sample.wall_ns);
            assert_eq!(prediction.valid_until_ns, expiry);
        }
        assert!(model
            .predict_query(&sample.fingerprint, &query, 959)
            .is_err());
        assert!(model
            .predict_query(&sample.fingerprint, &query, expiry + 1)
            .is_err());
    }
}

#[test]
fn joint_bank_all_q_failures_terminate_the_candidate_at_the_original_boundary() {
    let fit = candidate_fit();
    let residual = phase_samples(StructuredPhaseV2::Residual);
    let rclose = phase_close(
        StructuredPhaseV2::Residual,
        &residual,
        Some(fit.parameters_signature()),
    );
    let calibrated = fit.calibrate_owner_blocks(rclose, &residual).unwrap();
    let mut qualification = phase_samples(StructuredPhaseV2::Qualification);
    for s in &mut qualification {
        s.wall_ns += 100_000;
    }
    let qclose = phase_close(
        StructuredPhaseV2::Qualification,
        &qualification,
        Some(calibrated.parameters_signature()),
    );
    assert!(matches!(
        calibrated.qualify_owner_blocks(qclose, &qualification),
        Err(StructuredUnknown::QualificationUnderestimate)
    ));
}
