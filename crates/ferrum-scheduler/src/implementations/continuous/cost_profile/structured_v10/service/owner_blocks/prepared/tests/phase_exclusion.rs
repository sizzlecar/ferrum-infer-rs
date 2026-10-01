use super::*;

#[test]
fn source8_rejects_runtime_first_offer_fifo_frontier_policy() {
    use crate::implementations::continuous::cost_model::structured_v2::OwnerOpeningFrontierPolicyV1;
    let mut h = header();
    assert!(h.declaration.population.schedule.opening_frontier.is_none());
    h.declaration.population.schedule.opening_frontier =
        Some(OwnerOpeningFrontierPolicyV1::FirstOfferFifoV1);
    assert!(
        StructuredPreparedOwnerBlockHeaderV8::new(
            h.capture_identity,
            h.generation,
            h.fingerprint,
            h.producer,
            h.opening,
            h.declaration,
            h.maximum_file_bytes,
        )
        .is_err(),
        "source8 cannot borrow an undeclared runtime population boundary"
    );
}

#[test]
fn source8_cohort_crosses_owner_phase_without_reusing_its_remaining_numeric_sample() {
    // Each cohort has three real calls; ten-offer blocks deliberately close
    // midway through cohorts. No sample is removed or given a virtual ordinal.
    let (bytes, c, _) = collector::collected_with(10, [20, 10, 10]);
    let audit = c.prepared_audit();
    assert_eq!(
        audit.cohort_phase_policy,
        StructuredPreparedCohortPhasePolicyV8::FirstEligibleOwnerPhaseV1
    );
    assert_eq!(
        (audit.population.offered, audit.preparation_attempts),
        (120, 40)
    );
    assert!(audit.excluded_cohort_phase_attempts > 0);
    assert_eq!(
        audit
            .phase_exclusions
            .iter()
            .map(|v| v.excluded_original_offers)
            .sum::<u64>(),
        audit.excluded_cohort_phase_attempts
    );
    assert!(audit.population.owners.iter().any(|v| v.qualified));
    // Re-run the original journal through the same explicit membership rule.
    // The source7 implementation does not accept this new protocol.
    assert!(replay_structured_source_v7(&bytes, &CostProfileLoadLimits::default()).is_err());
    let mut lines = bytes.split_inclusive(|v| *v == b'\n');
    let h: StructuredPreparedOwnerBlockHeaderV8 =
        serde_json::from_slice(lines.next().unwrap()).unwrap();
    let mut replay =
        StructuredPreparedOwnerBlockCollectorV8::new(h, CostProfileLoadLimits::default()).unwrap();
    for line in lines {
        let record: StructuredPreparedOwnerBlockRecordV8 = serde_json::from_slice(line).unwrap();
        replay.push(&record).unwrap();
    }
    let replayed = replay.prepared_audit();
    assert_eq!(
        replayed.excluded_cohort_phase_attempts,
        audit.excluded_cohort_phase_attempts
    );
    assert_eq!(
        record_bytes_v7(&replayed.phase_exclusions).unwrap(),
        record_bytes_v7(&audit.phase_exclusions).unwrap()
    );
    assert_eq!(
        record_bytes_v7(&replayed.population.owners).unwrap(),
        record_bytes_v7(&audit.population.owners).unwrap()
    );
    assert_eq!(replayed.population.offered, audit.population.offered);

    assert_eq!(replay.source_receipt(), c.source_receipt());
    assert!(replay_structured_source_v8(&bytes, &CostProfileLoadLimits::default()).is_ok());
}

#[test]
fn source8_cohort_membership_policy_is_declared_and_cannot_be_omitted_or_relabelled() {
    let h = header();
    let original = serde_json::to_value(&h).unwrap();
    assert_eq!(
        original["cohort_phase_policy"],
        "first_eligible_owner_phase_v1"
    );
    for value in [None, Some(serde_json::json!("all_owner_phases"))] {
        let mut changed = original.clone();
        match value {
            Some(v) => {
                changed["cohort_phase_policy"] = v;
            }
            None => {
                changed
                    .as_object_mut()
                    .unwrap()
                    .remove("cohort_phase_policy");
            }
        }
        assert!(serde_json::from_value::<StructuredPreparedOwnerBlockHeaderV8>(changed).is_err());
    }
}

#[test]
fn source8_family_phase_policy_must_match_population_even_with_a_recomputed_signature() {
    let h = numerical_family::declared(8, [1; 3]);
    h.validate().unwrap();
    assert_eq!(
        h.cohort_phase_policy(),
        StructuredPreparedCohortPhasePolicyV8::FirstEligibleNumericalFamilyPhaseV1
    );
    let service = StructuredServiceHeaderV7::new(
        h.capture_identity,
        h.generation,
        h.fingerprint.clone(),
        h.producer.clone(),
        h.opening,
        h.declaration.population.clone(),
        h.maximum_file_bytes,
    )
    .unwrap();
    service.validate().unwrap();
    assert_eq!(
        service.declaration.population_policy(),
        h.declaration.population.population_policy()
    );
    assert_ne!(service.protocol, h.protocol);
    // Re-sign a structurally valid header with the wrong cohort policy. The
    // declared population and policy must match independently of its checksum.
    let mut value = serde_json::to_value(&h).unwrap();
    value["cohort_phase_policy"] = serde_json::json!("first_eligible_owner_phase_v1");
    let mut changed: StructuredPreparedOwnerBlockHeaderV8 = serde_json::from_value(value).unwrap();
    let mut digest = Sha256::new();
    digest.update(PREPARED_OWNER_BLOCK_SOURCE_PROTOCOL_V8.as_bytes());
    digest.update([0]);
    digest.update(MODEL_REVISION_V2.as_bytes());
    digest.update(changed.declaration_sha256);
    digest.update(record_bytes_v7(&changed.fingerprint).unwrap());
    digest.update(changed.maximum_file_bytes.to_le_bytes());
    digest.update(record_bytes_v7(&changed.cohort_phase_policy()).unwrap());
    digest.update(record_bytes_v7(&changed.tail_policy()).unwrap());
    changed.protocol = digest.finalize().into();
    assert!(changed.validate().is_err());

    let exact = header();
    assert_eq!(
        exact.cohort_phase_policy(),
        StructuredPreparedCohortPhasePolicyV8::FirstEligibleOwnerPhaseV1
    );
    let encoded = serde_json::to_value(&exact).unwrap();
    assert!(encoded["declaration"]["population"]["nonnegative_envelope"]
        .get("population_policy")
        .is_none());
}
