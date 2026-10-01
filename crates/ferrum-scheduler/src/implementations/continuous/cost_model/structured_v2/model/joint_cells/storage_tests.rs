//! Storage capacity and startup replacement of independently qualified banks.
//! Synthetic numerical phases do not grant live receipt or hardware authority.
use super::super::super::nonnegative::{FitSample, NonNegativeFit};
use super::super::physical_envelope::envelope;
use super::*;
use ferrum_interfaces::{execution_cost::*, vnext::DeviceCommandPhase};

#[path = "../physical_envelope/numerical_family_tests/fixture.rs"]
mod fixture;

fn samples(phase: StructuredPhaseV2, widths: &[u32]) -> Vec<StructuredNumericObservationV2> {
    let (_, originals) = fixture::population(phase, false);
    originals
        .iter()
        .enumerate()
        .map(|(i, original)| {
            let mut sample = original.clone();
            let input = originals
                .iter()
                .find(|s| s.input.owner().rows == widths[i % widths.len()])
                .unwrap();
            sample.input = input
                .input
                .clone()
                .with_cost_template_policy(StructuredCostTemplatePolicyV1::InstalledAlgorithmSetV1)
                .unwrap();
            sample.wall_ns = input.wall_ns;
            sample.membership.member_ordinal = sample.ordinal;
            sample.ordinal += 24;
            sample.membership.offered_ordinal += 24;
            sample.observed_at_ns += 240;
            sample
        })
        .collect()
}

fn close(
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

fn qualified(
    widths: &[u32],
    failed_width: Option<u32>,
    strategy: NonNegativePlanningEstimatorV1,
) -> QualifiedStructuredModelV2 {
    let fit = samples(StructuredPhaseV2::Fit, &[2, 4, 8]);
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
        nonnegative_envelope: Some(NonNegativeEnvelopeContractV1 {
            algorithm_universe: None,
            planning_estimator: strategy,
            workload_domain: fixture::domain(),
            settings: Default::default(),
            challenge: WorkAxisAndBranchChallengesV1::WorkAxesAndPlainTextBranchesV1,
            template_policy: StructuredCostTemplatePolicyV1::InstalledAlgorithmSetV1,
            population_policy: StructuredPopulationPolicyV1::HomogeneousOrdinaryDecodeV1,
        }),
        input_target: None,
    };
    let scope = StructuredScopeV2 {
        owner: fit[0].input.owner().clone(),
        numerical_family: Some(fit[0].input.numerical_family_key().unwrap()),
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
    let fitted = FittedStructuredModelV2::fit_owner_blocks(
        fit[0].fingerprint.clone(),
        StructuredSettingsV2 {
            max_sample_age_ns: 10_000,
            static_margin_ns: 100,
            ..Default::default()
        },
        scope,
        contract,
        close(StructuredPhaseV2::Fit, &fit, None),
        &fit,
    )
    .unwrap();
    let residual = samples(StructuredPhaseV2::Residual, widths);
    let rclose = close(
        StructuredPhaseV2::Residual,
        &residual,
        Some(fitted.parameters_signature()),
    );
    let calibrated = fitted.calibrate_owner_blocks(rclose, &residual).unwrap();
    let mut qualification = samples(StructuredPhaseV2::Qualification, widths);
    for sample in &mut qualification {
        if Some(sample.input.owner().rows) == failed_width {
            sample.wall_ns += 100_000;
        }
    }
    let qclose = close(
        StructuredPhaseV2::Qualification,
        &qualification,
        Some(calibrated.parameters_signature()),
    );
    calibrated
        .qualify_owner_blocks(qclose, &qualification)
        .unwrap()
}

fn joint(widths: &[u32]) -> QualifiedStructuredModelV2 {
    qualified(
        widths,
        None,
        NonNegativePlanningEstimatorV1::IdentifiedFitJointCellsV1,
    )
}

#[test]
fn joint_bank_public_input_key_uses_checked_contract_and_measurable_storage() {
    let model = joint(&[2]);
    let contract = &model
        .calibrated
        .fitted
        .numerical
        .physical()
        .unwrap()
        .contract;
    let input = &samples(StructuredPhaseV2::Qualification, &[2])[0].input;
    let key = contract.joint_input_cell(input).unwrap();
    assert_eq!(key, input_key(input, contract).unwrap());
    assert_eq!(
        key.retained_payload_bytes(),
        Some(std::mem::size_of::<StructuredJointCellKeyV1>() + key.retained().unwrap())
    );
    let mut changed = contract.clone();
    changed.planning_estimator = NonNegativePlanningEstimatorV1::FittedResidualV1;
    assert!(matches!(
        changed.joint_input_cell(input),
        Err(StructuredUnknown::WrongProtocol)
    ));
    let mut changed = contract.clone();
    changed.algorithm_universe =
        Some(DeclaredAlgorithmUniverseV1::from_inputs([input], 4096).unwrap());
    assert!(matches!(
        changed.joint_input_cell(input),
        Err(StructuredUnknown::WrongDomain)
    ));
}

#[test]
fn joint_bank_single_cell_counts_all_owned_capacity_in_published_model() {
    let mut model = joint(&[2]);
    let sample = &samples(StructuredPhaseV2::Qualification, &[2])[0];
    let query = StructuredQueryV2::exact(sample.input.clone());
    let before_prediction = model
        .predict_query(&sample.fingerprint, &query, 961)
        .unwrap();
    let before_signature = model.parameters_signature();
    let before_model = model.retained_payload_bytes().unwrap();
    let bank = model.calibrated.joint_bank.as_mut().unwrap();
    assert_eq!(bank.cells.len(), 1);
    let slot_size = std::mem::size_of::<(JointCellKey, CellMargin)>();
    let old_capacity = bank.cells.capacity();
    let expected = old_capacity * slot_size + bank.cells[0].0.retained().unwrap();
    assert_eq!(bank.retained(), Some(expected));
    bank.cells.reserve_exact(3);
    let extra = (bank.cells.capacity() - old_capacity) * slot_size;
    assert!(extra > 0);
    assert_eq!(bank.retained(), Some(expected + extra));
    assert_eq!(model.retained_payload_bytes(), Some(before_model + extra));
    assert_eq!(model.parameters_signature(), before_signature);
    let after_prediction = model
        .predict_query(&sample.fingerprint, &query, 961)
        .unwrap();
    assert_eq!(after_prediction.planning_ns, before_prediction.planning_ns);
    assert_eq!(
        after_prediction.valid_until_ns,
        before_prediction.valid_until_ns
    );
}

#[test]
fn joint_bank_startup_replacement_requires_every_old_qualified_key() {
    let old = joint(&[2]);
    let superset = joint(&[2, 4, 8]);
    let missing = joint(&[4, 8]);
    assert!(old.preserves_physical_input_coverage(&old));
    assert!(superset.preserves_physical_input_coverage(&old));
    assert!(!old.preserves_physical_input_coverage(&superset));
    assert!(!missing.preserves_physical_input_coverage(&old));
    let failed = qualified(
        &[2, 4, 8],
        Some(2),
        NonNegativePlanningEstimatorV1::IdentifiedFitJointCellsV1,
    );
    assert_eq!(
        failed.calibrated.joint_bank.as_ref().unwrap().cells.len(),
        3
    );
    assert!(!failed.preserves_physical_input_coverage(&old));
    // Failed old keys are unavailable already and do not prevent preservation
    // of the remaining qualified subset.
    assert!(missing.preserves_physical_input_coverage(&failed));
    for sample in samples(StructuredPhaseV2::Qualification, &[2]) {
        let query = StructuredQueryV2::exact(sample.input.clone());
        assert_eq!(
            superset.catalog_input_membership(&query).unwrap(),
            Some(true)
        );
        assert_eq!(
            missing.catalog_input_membership(&query).unwrap(),
            Some(false)
        );
        assert_eq!(
            failed.catalog_input_membership(&query).unwrap(),
            Some(false)
        );
    }
}

#[test]
fn joint_bank_replacement_does_not_equate_other_strategies_or_coordinate_contracts() {
    let model = joint(&[2, 4, 8]);
    let legacy = qualified(
        &[2, 4, 8],
        None,
        NonNegativePlanningEstimatorV1::FittedResidualV1,
    );
    assert!(!model.preserves_physical_input_coverage(&legacy));
    assert!(!legacy.preserves_physical_input_coverage(&model));

    let mut changed = joint(&[2, 4, 8]);
    changed
        .calibrated
        .fitted
        .numerical
        .physical_mut()
        .unwrap()
        .contract
        .template_policy = StructuredCostTemplatePolicyV1::OrderedV1;
    assert!(!changed.preserves_physical_input_coverage(&model));
    assert!(!model.preserves_physical_input_coverage(&changed));

    let mut changed = joint(&[2, 4, 8]);
    let input = &samples(StructuredPhaseV2::Fit, &[2])[0].input;
    changed
        .calibrated
        .fitted
        .numerical
        .physical_mut()
        .unwrap()
        .contract
        .algorithm_universe =
        Some(DeclaredAlgorithmUniverseV1::from_inputs([input], 4096).unwrap());
    assert!(!changed.preserves_physical_input_coverage(&model));
    assert!(!model.preserves_physical_input_coverage(&changed));

    let mut changed = joint(&[2, 4, 8]);
    changed.calibrated.fitted.exemplar.support.push(0);
    assert!(!changed.preserves_physical_input_coverage(&model));
}
