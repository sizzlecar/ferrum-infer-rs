use super::*;

fn qualify_with(
    policy: Option<OwnerPredictionValidityPolicyV1>,
    deadline: u64,
) -> Result<QualifiedStructuredModelV2> {
    let mut contract = block_contract();
    contract.schedule.prediction_validity = policy;
    contract.expires_at_ns = deadline;
    let fit = block_samples(StructuredPhaseV2::Fit, 33, 16, 0);
    let fitted = FittedStructuredModelV2::fit_owner_blocks(
        fp(),
        settings(),
        scope(&fit),
        contract,
        block_close(StructuredPhaseV2::Fit, 2, 2, None, &fit),
        &fit,
    )?;
    let residual = block_samples(StructuredPhaseV2::Residual, 65, 16, 16);
    let residual_close = block_close(
        StructuredPhaseV2::Residual,
        3,
        3,
        Some(fitted.parameters_signature()),
        &residual,
    );
    let calibrated = fitted.calibrate_owner_blocks(residual_close, &residual)?;
    let qualification = block_samples(StructuredPhaseV2::Qualification, 97, 16, 32);
    let qualification_close = block_close(
        StructuredPhaseV2::Qualification,
        4,
        4,
        Some(calibrated.parameters_signature()),
        &qualification,
    );
    calibrated.qualify_owner_blocks(qualification_close, &qualification)
}

#[test]
fn owner_prediction_validity_separates_collection_from_original_member_age() {
    let deadline = 1290; // Qualification closes at1280; a sample costs at least1000ns.
    let old = qualify_with(None, deadline).unwrap();
    let new = qualify_with(
        Some(OwnerPredictionValidityPolicyV1::OriginalSampleAgeV1),
        deadline,
    )
    .unwrap();
    let query = StructuredQueryV2::exact(
        block_samples(StructuredPhaseV2::Qualification, 97, 16, 32)[0]
            .input
            .clone(),
    );
    let old_value = old.predict_query(&fp(), &query, 1281).unwrap();
    let new_value = new.predict_query(&fp(), &query, 1281).unwrap();
    assert_eq!(old_value.planning_ns, new_value.planning_ns);
    assert_eq!(old_value.valid_until_ns, deadline);
    assert!(old_value.planning_ns > deadline - 1281);
    let first_sample = block_samples(StructuredPhaseV2::Fit, 33, 16, 0)[0].observed_at_ns;
    let expiry = first_sample + settings().max_sample_age_ns;
    assert_eq!(new_value.valid_until_ns, expiry);
    assert!(matches!(
        old.predict_query(&fp(), &query, deadline + 1),
        Err(StructuredUnknown::Stale)
    ));
    assert!(new.predict_query(&fp(), &query, deadline + 1).is_ok());
    assert!(new.predict_query(&fp(), &query, expiry).is_ok());
    assert!(matches!(
        new.predict_query(&fp(), &query, expiry + 1),
        Err(StructuredUnknown::Stale)
    ));
    assert_ne!(old.parameters_signature(), new.parameters_signature());
    assert_eq!(new.source_contract().phase_members, [16; 3]);
}

#[test]
fn owner_prediction_validity_never_extends_fit_residual_or_qualification_collection() {
    for deadline in [639, 959, 1279] {
        assert!(
            matches!(
                qualify_with(
                    Some(OwnerPredictionValidityPolicyV1::OriginalSampleAgeV1),
                    deadline
                ),
                Err(StructuredUnknown::Stale)
            ),
            "phase completed beyond original deadline={deadline}"
        );
    }
}

#[test]
fn owner_prediction_validity_absent_schedule_keeps_original_wire_shape() {
    let old = OwnerBlockScheduleV1::new(32, [32; 3], [8; 3]).unwrap();
    let original = r#"{"block_offered":32,"phase_min_offered":[32,32,32],"min_members":[8,8,8],"maximum_phase_members":[39,39,39]}"#;
    assert_eq!(serde_json::to_string(&old).unwrap(), original);
    let decoded: OwnerBlockScheduleV1 = serde_json::from_str(original).unwrap();
    assert_eq!(decoded.prediction_validity, None);
    assert_eq!(serde_json::to_string(&decoded).unwrap(), original);
}
