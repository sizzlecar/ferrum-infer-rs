//! Typed numerical fixtures; clocks and durations are synthetic, not evidence
//! of live collection or backend performance.
use super::*;

mod prediction_validity;

pub(super) fn block_contract() -> StructuredOwnerPhaseContractV1 {
    let c = contract();
    StructuredOwnerPhaseContractV1 {
        capture_identity: c.capture_identity,
        protocol: c.protocol,
        membership_rule: c.membership_rule,
        declaration_sha256: [81; 32],
        owner_attempt_id: 4,
        schedule: OwnerBlockScheduleV1::new(32, [32; 3], [8; 3]).unwrap(),
        discovery_block: 1,
        discovery_offered_cutoff: 32,
        discovery_fifo_cutoff: 96,
        discovery_closed_at_ns: 320,
        expires_at_ns: 10_000,
        domain_policy: StructuredServiceDomainPolicyV1::FrozenFitSupportV1,
        nonnegative_envelope: None,
        input_target: None,
    }
}

pub(super) fn block_samples(
    p: StructuredPhaseV2,
    first_ticket: u64,
    count: usize,
    member_before: usize,
) -> Vec<StructuredNumericObservationV2> {
    let originals = phase(p);
    (0..count)
        .map(|i| {
            let mut sample = originals[i % originals.len()].clone();
            let ticket = first_ticket + i as u64;
            sample.ordinal = ticket * 3;
            sample.membership.offered_ordinal = ticket;
            sample.membership.member_ordinal = (member_before + i + 1) as u64;
            sample.call_id = ticket + 100;
            sample.observed_at_ns = ticket * 10;
            sample
        })
        .collect()
}

pub(super) fn block_close(
    p: StructuredPhaseV2,
    first_block: u64,
    last_block: u64,
    previous: Option<[u8; 32]>,
    rows: &[StructuredNumericObservationV2],
) -> StructuredOwnerPhaseCloseV1 {
    StructuredOwnerPhaseCloseV1::new(
        p,
        StructuredOwnerPhaseBoundaryV1 {
            first_block,
            last_block,
            first_offered: (first_block - 1) * 32 + 1,
            last_offered: last_block * 32,
            opening_fifo_cutoff: (first_block - 1) * 32 * 3,
            closing_fifo_cutoff: last_block * 32 * 3,
            opened_at_ns: (first_block - 1) * 32 * 10,
            frozen_at_ns: last_block * 32 * 10,
        },
        [p.index() as u8 + 91; 32],
        previous,
        rows,
    )
}

fn block_fitted() -> FittedStructuredModelV2 {
    let rows = block_samples(StructuredPhaseV2::Fit, 33, 16, 0);
    FittedStructuredModelV2::fit_owner_blocks(
        fp(),
        settings(),
        scope(&rows),
        block_contract(),
        block_close(StructuredPhaseV2::Fit, 2, 2, None, &rows),
        &rows,
    )
    .unwrap()
}

fn block_calibrated() -> CalibratedStructuredModelV2 {
    let model = block_fitted();
    let rows = block_samples(StructuredPhaseV2::Residual, 65, 16, 16);
    let close = block_close(
        StructuredPhaseV2::Residual,
        3,
        3,
        Some(model.parameters_signature()),
        &rows,
    );
    model.calibrate_owner_blocks(close, &rows).unwrap()
}

#[test]
fn owner_blocks_schedule_accounts_for_the_whole_last_block() {
    let schedule = OwnerBlockScheduleV1::new(256, [256; 3], [8; 3]).unwrap();
    assert_eq!(schedule.maximum_phase_members, [263; 3]);
    let s = StructuredSettingsV2 {
        max_phase_samples: 263,
        ..settings()
    };
    schedule.validate(&s).unwrap();
    assert!(!schedule.is_ready(StructuredPhaseV2::Fit, 256, 7).unwrap());
    assert!(schedule.is_ready(StructuredPhaseV2::Fit, 512, 263).unwrap());
    assert!(matches!(
        schedule.is_ready(StructuredPhaseV2::Fit, 511, 8),
        Err(StructuredUnknown::IncompletePhasePopulation)
    ));
    assert!(matches!(
        schedule.validate(&StructuredSettingsV2 {
            max_phase_samples: 256,
            ..s
        }),
        Err(StructuredUnknown::Capacity)
    ));
    assert_eq!(
        OwnerBlockScheduleV1::new(256, [513; 3], [8; 3])
            .unwrap()
            .maximum_phase_members,
        [768; 3]
    );
    assert!(OwnerBlockScheduleV1::new(0, [256; 3], [8; 3]).is_err());
    assert!(OwnerBlockScheduleV1::new(usize::MAX, [usize::MAX; 3], [8; 3]).is_err());
}

#[test]
fn owner_blocks_rowspace_keeps_original_tickets_phase_signatures_and_ttl() {
    let fitted = block_fitted();
    let fit_signature = fitted.parameters_signature();
    let residual = block_samples(StructuredPhaseV2::Residual, 65, 16, 16);
    assert_eq!(
        fitted.service_input_membership(&residual[0].input).unwrap(),
        StructuredServiceInputMembershipV1::Eligible
    );
    let calibrated = fitted
        .calibrate_owner_blocks(
            block_close(
                StructuredPhaseV2::Residual,
                3,
                3,
                Some(fit_signature),
                &residual,
            ),
            &residual,
        )
        .unwrap();
    let residual_signature = calibrated.parameters_signature();
    let heldout = block_samples(StructuredPhaseV2::Qualification, 97, 16, 32);
    assert_eq!(
        calibrated
            .service_input_membership(&heldout[0].input)
            .unwrap(),
        StructuredServiceInputMembershipV1::Eligible
    );
    let qualified = calibrated
        .qualify_owner_blocks(
            block_close(
                StructuredPhaseV2::Qualification,
                4,
                4,
                Some(residual_signature),
                &heldout,
            ),
            &heldout,
        )
        .unwrap();
    assert_eq!(qualified.source_contract().phase_members, [16, 16, 16]);
    assert_eq!(qualified.owner_block_contract(), Some(&block_contract()));
    assert_eq!(qualified.service_window_contract(), None);
    assert_eq!(
        qualified.owner_block_phase_signatures(),
        Some([
            fit_signature,
            residual_signature,
            qualified.parameters_signature()
        ])
    );
    let query = StructuredQueryV2::exact(heldout[0].input.clone());
    let prediction = qualified.predict_query(&fp(), &query, 1300).unwrap();
    assert_eq!(prediction.valid_until_ns, 10_000);
    assert!(matches!(
        qualified.predict_query(&fp(), &query, 10_001),
        Err(StructuredUnknown::Stale)
    ));
    assert!(qualified.retained_payload_bytes().unwrap() > std::mem::size_of_val(&qualified));
}

#[test]
fn owner_blocks_allow_input_shortage_but_reject_delaying_a_ready_phase() {
    let mut rows = block_samples(StructuredPhaseV2::Fit, 33, 7, 0);
    rows.extend(block_samples(StructuredPhaseV2::Fit, 65, 16, 7));
    let fit = || {
        FittedStructuredModelV2::fit_owner_blocks(
            fp(),
            settings(),
            scope(&rows),
            block_contract(),
            block_close(StructuredPhaseV2::Fit, 2, 3, None, &rows),
            &rows,
        )
    };
    assert!(fit().is_ok());

    let mut delayed = block_samples(StructuredPhaseV2::Fit, 33, 8, 0);
    delayed.extend(block_samples(StructuredPhaseV2::Fit, 65, 16, 8));
    assert!(matches!(
        FittedStructuredModelV2::fit_owner_blocks(
            fp(),
            settings(),
            scope(&delayed),
            block_contract(),
            block_close(StructuredPhaseV2::Fit, 2, 3, None, &delayed),
            &delayed,
        ),
        Err(StructuredUnknown::PhaseLeakage)
    ));
}

#[test]
fn owner_blocks_complete_last_block_keeps_263_members_and_original_denominator() {
    let mut c = block_contract();
    c.schedule = OwnerBlockScheduleV1::new(256, [256; 3], [8; 3]).unwrap();
    c.discovery_offered_cutoff = 256;
    c.discovery_fifo_cutoff = 768;
    c.discovery_closed_at_ns = 2560;
    let s = StructuredSettingsV2 {
        max_phase_samples: 263,
        ..settings()
    };
    let mut rows = block_samples(StructuredPhaseV2::Fit, 257, 7, 0);
    rows.extend(block_samples(StructuredPhaseV2::Fit, 513, 256, 7));
    let close = StructuredOwnerPhaseCloseV1::new(
        StructuredPhaseV2::Fit,
        StructuredOwnerPhaseBoundaryV1 {
            first_block: 2,
            last_block: 3,
            first_offered: 257,
            last_offered: 768,
            opening_fifo_cutoff: 768,
            closing_fifo_cutoff: 2304,
            opened_at_ns: 2560,
            frozen_at_ns: 7680,
        },
        [92; 32],
        None,
        &rows,
    );
    let model =
        FittedStructuredModelV2::fit_owner_blocks(fp(), s, scope(&rows), c, close.clone(), &rows)
            .unwrap();
    assert_eq!(close.member_count, 263);
    assert_ne!(model.parameters_signature(), [0; 32]);
    assert_eq!(
        close.boundary.last_offered - close.boundary.first_offered + 1,
        512
    );
    let mut shortened = rows;
    shortened.pop();
    assert!(matches!(
        FittedStructuredModelV2::fit_owner_blocks(
            fp(),
            settings(),
            scope(&shortened),
            block_contract(),
            close,
            &shortened,
        ),
        Err(StructuredUnknown::IncompletePhasePopulation)
    ));
}

#[test]
fn owner_blocks_reject_wrong_phase_frozen_model_and_original_population_tampering() {
    let rows = block_samples(StructuredPhaseV2::Residual, 65, 16, 16);
    let signature = block_fitted().parameters_signature();
    let original = block_close(StructuredPhaseV2::Residual, 3, 3, Some(signature), &rows);
    for case in 0..6 {
        let mut close = original.clone();
        let mut samples = rows.clone();
        match case {
            0 => close.previous_parameters_sha256 = Some([99; 32]),
            1 => close.phase = StructuredPhaseV2::Qualification,
            2 => close.boundary.first_block += 1,
            3 => close.boundary.first_offered -= 1,
            4 => samples[0].ordinal = close.boundary.opening_fifo_cutoff,
            5 => samples[0].observed_at_ns = close.boundary.opened_at_ns - 1,
            _ => unreachable!(),
        }
        assert!(block_fitted()
            .calibrate_owner_blocks(close, &samples)
            .is_err());
    }
    let mut samples = rows.clone();
    samples[0].membership.offered_ordinal += 1;
    assert!(matches!(
        block_fitted().calibrate_owner_blocks(original, &samples),
        Err(StructuredUnknown::IncompletePhasePopulation)
    ));
    assert!(matches!(
        block_fitted().calibrate(&rows, 960),
        Err(StructuredUnknown::WrongProtocol)
    ));
    assert!(matches!(
        super::fitted().calibrate_owner_blocks(
            block_close(StructuredPhaseV2::Residual, 3, 3, Some(signature), &rows),
            &rows,
        ),
        Err(StructuredUnknown::WrongProtocol)
    ));
}

#[test]
fn owner_blocks_underestimate_is_terminal_and_cannot_be_diluted_by_later_blocks() {
    let rows = block_samples(StructuredPhaseV2::Qualification, 97, 16, 32);
    let calibrated = block_calibrated();
    let signature = calibrated.parameters_signature();
    let model = calibrated
        .qualify_owner_blocks(
            block_close(
                StructuredPhaseV2::Qualification,
                4,
                4,
                Some(signature),
                &rows,
            ),
            &rows,
        )
        .unwrap();
    let bound = model
        .predict_query(
            &fp(),
            &StructuredQueryV2::exact(rows[0].input.clone()),
            1300,
        )
        .unwrap()
        .planning_ns;
    let mut slow = rows;
    slow[0].wall_ns = bound + 1;
    assert!(matches!(
        block_calibrated().qualify_owner_blocks(
            block_close(
                StructuredPhaseV2::Qualification,
                4,
                4,
                Some(signature),
                &slow
            ),
            &slow,
        ),
        Err(StructuredUnknown::QualificationUnderestimate)
    ));
    slow.extend(block_samples(StructuredPhaseV2::Qualification, 129, 16, 48));
    assert!(matches!(
        block_calibrated().qualify_owner_blocks(
            block_close(
                StructuredPhaseV2::Qualification,
                4,
                5,
                Some(signature),
                &slow
            ),
            &slow,
        ),
        Err(StructuredUnknown::PhaseLeakage)
    ));
}

#[test]
fn owner_blocks_declaration_and_attempt_change_the_frozen_identity() {
    let rows = block_samples(StructuredPhaseV2::Fit, 33, 16, 0);
    let original = block_fitted().parameters_signature();
    for case in 0..3 {
        let mut c = block_contract();
        match case {
            0 => c.owner_attempt_id += 1,
            1 => c.declaration_sha256[0] ^= 1,
            2 => c.expires_at_ns -= 1,
            _ => unreachable!(),
        }
        let model = FittedStructuredModelV2::fit_owner_blocks(
            fp(),
            settings(),
            scope(&rows),
            c,
            block_close(StructuredPhaseV2::Fit, 2, 2, None, &rows),
            &rows,
        )
        .unwrap();
        assert_ne!(model.parameters_signature(), original);
    }
}

#[test]
fn owner_blocks_do_not_reset_expiry_or_accept_invalid_member_identity() {
    let rows = block_samples(StructuredPhaseV2::Residual, 65, 16, 16);
    let signature = block_fitted().parameters_signature();
    let mut expired = block_close(StructuredPhaseV2::Residual, 3, 3, Some(signature), &rows);
    expired.boundary.frozen_at_ns = 10_001;
    assert!(matches!(
        block_fitted().calibrate_owner_blocks(expired, &rows),
        Err(StructuredUnknown::Stale)
    ));
    for case in 0..3 {
        let mut changed = rows.clone();
        match case {
            0 => changed[1].call_id = changed[0].call_id,
            1 => changed[1].membership.member_ordinal = changed[0].membership.member_ordinal,
            2 => changed[1].fingerprint.model_weights[0] ^= 1,
            _ => unreachable!(),
        }
        let close = block_close(StructuredPhaseV2::Residual, 3, 3, Some(signature), &changed);
        assert!(block_fitted()
            .calibrate_owner_blocks(close, &changed)
            .is_err());
    }
}
