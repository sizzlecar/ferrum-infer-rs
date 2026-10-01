use super::*;
use crate::implementations::continuous::cost_model::structured_v2::OwnerPredictionValidityPolicyV1;

#[test]
fn source7_selected_domains_preserve_full_source_after_mixed_child_expiry() {
    let boot = domain(1, 10);
    let mut h = bound_header(boot.clone());
    let identity = ExecutorCostIdentity {
        schema_version: EXECUTOR_COST_IDENTITY_SCHEMA,
        model_weights: h.fingerprint.model_weights,
        numerical_policy: h.fingerprint.numerical_policy,
        device_runtime: h.fingerprint.device_runtime,
        execution_config: h.fingerprint.execution_config,
    };
    h.declaration
        .nonnegative_envelope
        .as_mut()
        .unwrap()
        .workload_domain = CostWorkloadDomainV1::new_vnext(
        &identity,
        CostWorkloadLimitsV1 {
            maximum_rows: NonZeroU32::new(2).unwrap(),
            maximum_context_tokens: NonZeroU32::new(128).unwrap(),
            maximum_scheduled_tokens_per_wave: NonZeroU64::new(128).unwrap(),
            output_vocabulary_elements: NonZeroU64::new(16).unwrap(),
            repetition_slot_capacity: 4,
            fixed_state_bytes_per_row: 0,
        },
    )
    .unwrap();
    h.declaration
        .nonnegative_envelope
        .as_mut()
        .unwrap()
        .population_policy = StructuredPopulationPolicyV1::ExactOwnerV1;
    h.declaration.schedule.prediction_validity =
        Some(OwnerPredictionValidityPolicyV1::OriginalSampleAgeV1);
    let h = StructuredServiceHeaderV7::new_with_monotonic_domain(
        h.capture_identity,
        h.generation,
        h.fingerprint,
        h.producer,
        h.opening,
        h.declaration,
        h.maximum_file_bytes,
        boot.clone(),
    )
    .unwrap();
    let mut bytes = record_bytes_v7(&h).unwrap();
    let mut collector =
        StructuredServiceCollectorV7::new(h, CostProfileLoadLimits::default()).unwrap();
    let mut closing = paired(1);
    for block in 1..=8u64 {
        let first = (block - 1) * 8 + 1;
        append(
            &mut bytes,
            &collector
                .open_block(first * 2000 - 1, (first - 1) * 3)
                .unwrap(),
        );
        for ticket in first..first + 8 {
            let rows = if block <= 4 { 1 } else { 2 };
            let names: Vec<_> = (0..rows)
                .map(|row| format!("expiry-{ticket}-{row}"))
                .collect();
            let ids: Vec<_> = names.iter().map(String::as_str).collect();
            let prepared = old::prepared_batch_route_policy_bounds_and_future(
                &ids,
                2,
                1,
                None,
                ferrum_interfaces::execution_cost::ActualWaveGraphState::Disabled,
                Some([44; 32]),
                64,
                &vec![4; rows],
                None,
            )
            .0;
            let stages = old::stages(&old::header(), &prepared, ticket, 1000);
            let record = StructuredServiceRecordV7::Completed {
                wave: StructuredServiceWaveV7::from_diagnostic(
                    ticket,
                    ticket * 2000,
                    ticket * 3,
                    serde_json::to_value(stages).unwrap(),
                    None,
                )
                .unwrap(),
            };
            collector.push(&record).unwrap();
            append(&mut bytes, &record);
        }
        closing = paired((first + 7) * 2000 + 1101);
        append(&mut bytes, &collector.close_block(closing).unwrap());
    }
    assert_eq!(collector.qualified_children(), 2, "{:?}", collector.audit());
    let (record, checkpoint) = collector.checkpoint(closing).unwrap();
    append(&mut bytes, &record);
    let limits = CostProfileLoadLimits::default();
    let original = checkpoint
        .activate_same_boot_memory(closing, &boot, &limits)
        .unwrap();
    let oldest = original
        .children
        .iter()
        .min_by_key(|child| closing.monotonic_ns - child.provenance().oldest_imported_age_ns)
        .unwrap();
    let expires = closing.monotonic_ns - oldest.provenance().oldest_imported_age_ns
        + oldest.runtime_limits().1;
    let at = paired(expires + 1);
    let selected: Vec<_> = original
        .children
        .iter()
        .filter(|child| child.is_current_local(at.monotonic_ns).is_ok())
        .map(|child| *child.domain_signature())
        .collect();
    assert_eq!(selected.len(), 1);
    let files = Files::new(&bytes);
    assert!(export_structured_profile_v14_same_boot(
        &files.source,
        Sha256::digest(&bytes).into(),
        bytes.len() as u64,
        &files.profile,
        &boot,
        at,
        &limits,
    )
    .is_err());
    export_structured_profile_v14_same_boot_selected(
        &files.source,
        Sha256::digest(&bytes).into(),
        bytes.len() as u64,
        &files.profile,
        &boot,
        at,
        &limits,
        &selected,
    )
    .unwrap();
    let loaded = load_structured_profile_v14_same_boot_selected(
        &files.profile,
        &old::fingerprint(),
        &limits,
        &boot,
        at,
        &selected,
    )
    .unwrap();
    assert_eq!(loaded.children.len(), 1);
    assert_eq!(loaded.offered_attempts, original.offered_attempts);
    assert_eq!(loaded.total_shape_rows, original.total_shape_rows);
    assert_eq!(loaded.source_bytes, original.source_bytes);
    assert_eq!(loaded.source_sha256, original.source_sha256);
    let surviving = original
        .children
        .iter()
        .find(|child| child.domain_signature() == loaded.children[0].domain_signature())
        .unwrap();
    assert_eq!(
        loaded.children[0].parameters_signature(),
        surviving.parameters_signature()
    );
    assert_eq!(
        loaded.children[0].provenance().clock,
        surviving.provenance().clock
    );
    assert!(loaded.children[0].is_current_local(at.monotonic_ns).is_ok());
    assert!(
        load_structured_profile_v14_same_boot_selected(
            &files.profile,
            &old::fingerprint(),
            &limits,
            &boot,
            at,
            &[*oldest.domain_signature()],
        )
        .is_err(),
        "selecting the expired child cannot renew its age"
    );
    for invalid in [vec![], vec![[0; 32]], vec![selected[0], selected[0]]] {
        assert!(load_structured_profile_v14_same_boot_selected(
            &files.profile,
            &old::fingerprint(),
            &limits,
            &boot,
            at,
            &invalid,
        )
        .is_err());
    }
    let mut short = limits.clone();
    short.max_samples =
        std::num::NonZeroUsize::new(original.offered_attempts as usize - 1).unwrap();
    assert!(
        load_structured_profile_v14_same_boot_selected(
            &files.profile,
            &old::fingerprint(),
            &short,
            &boot,
            at,
            &selected,
        )
        .is_err(),
        "omitted children cannot reduce original replay work"
    );
    let mut metadata: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&files.profile).unwrap()).unwrap();
    let omitted = metadata["children"]
        .as_array_mut()
        .unwrap()
        .iter_mut()
        .find(|child| {
            serde_json::from_value::<[u8; 32]>(child["domain_signature"].clone()).unwrap()
                == *oldest.domain_signature()
        })
        .unwrap();
    omitted["parameters_sha256"][0] = serde_json::json!(255);
    std::fs::write(&files.profile, serde_json::to_vec(&metadata).unwrap()).unwrap();
    assert!(
        load_structured_profile_v14_same_boot_selected(
            &files.profile,
            &old::fingerprint(),
            &limits,
            &boot,
            at,
            &selected,
        )
        .is_err(),
        "even omitted child metadata must match original replay"
    );
}
