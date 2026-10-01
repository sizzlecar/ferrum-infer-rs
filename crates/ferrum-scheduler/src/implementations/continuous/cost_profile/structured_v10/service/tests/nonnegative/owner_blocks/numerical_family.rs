//! Source7 retains its original wave/block membership when widths share a
//! numerical family. Consecutive request work is not a source8 cohort lock.
use super::*;
use crate::implementations::continuous::cost_model::structured_v2::{
    StructuredPopulationPolicyV1, StructuredUnknownV2,
};
use crate::implementations::continuous::cost_profile::structured_v10::wire::Prepared;
use ferrum_interfaces::execution_cost::ActualWaveGraphState;

mod subprocess;

const POLICY: [u8; 32] = [44; 32];

fn family_header() -> StructuredServiceHeaderV7 {
    let mut h = block_header();
    let identity = ExecutorCostIdentity {
        schema_version: EXECUTOR_COST_IDENTITY_SCHEMA,
        model_weights: h.fingerprint.model_weights,
        numerical_policy: h.fingerprint.numerical_policy,
        device_runtime: h.fingerprint.device_runtime,
        execution_config: h.fingerprint.execution_config,
    };
    let envelope = h.declaration.nonnegative_envelope.as_mut().unwrap();
    envelope.population_policy = StructuredPopulationPolicyV1::HomogeneousOrdinaryDecodeV1;
    envelope.workload_domain = CostWorkloadDomainV1::new_vnext(
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
    StructuredServiceHeaderV7::new(
        h.capture_identity,
        h.generation,
        h.fingerprint,
        h.producer,
        h.opening,
        h.declaration,
        h.maximum_file_bytes,
    )
    .unwrap()
}

fn prepared(ticket: u64) -> Prepared {
    let group = (ticket - 1) / 3;
    let generated = (ticket - 1) % 3 + 1;
    let a = format!("source7-family-{group}-a");
    let b = format!("source7-family-{group}-b");
    let (ids, maxima) = if generated == 3 {
        (vec![b.as_str()], vec![4])
    } else {
        (vec![a.as_str(), b.as_str()], vec![3, 4])
    };
    old::prepared_batch_route_policy_bounds_and_future(
        &ids,
        generated + 1,
        generated,
        None,
        ActualWaveGraphState::Disabled,
        Some(POLICY),
        64 + (generated - 1) as u32,
        &maxima,
        None,
    )
    .0
}

fn completed(ticket: u64, prepared_ticket: u64, call: u64) -> StructuredServiceRecordV7 {
    let mut stages = old::stages(&old::header(), &prepared(prepared_ticket), ticket, 1_000);
    if call != ticket {
        // Keep valid current timestamps and settlement binding so the negative
        // case specifically challenges capture-wide call uniqueness.
        stages.call_id = call;
        let binding = observation::stage_binding(&stages, None).unwrap();
        let settled = stages
            .structured_evidence
            .as_mut()
            .unwrap()
            .as_mut()
            .unwrap();
        settled.call_id = call;
        settled.stage_binding = binding;
    }
    StructuredServiceRecordV7::Completed {
        wave: StructuredServiceWaveV7::from_diagnostic(
            ticket,
            ticket * 2_000,
            ticket * 3,
            serde_json::to_value(stages).unwrap(),
            None,
        )
        .unwrap(),
    }
}

fn collected_family() -> (
    Vec<u8>,
    StructuredServiceCollectorV7,
    StructuredServiceCheckpointV7,
    StructuredServiceClockV7,
) {
    let h = family_header();
    let mut bytes = record_bytes_v7(&h).unwrap();
    let mut c = StructuredServiceCollectorV7::new(h, CostProfileLoadLimits::default()).unwrap();
    let mut closing = paired(1);
    for block in 1..=4 {
        let first = (block - 1) * 8 + 1;
        append(
            &mut bytes,
            &c.open_block(first * 2_000 - 1, (first - 1) * 3).unwrap(),
        );
        for ticket in first..first + 8 {
            let record = completed(ticket, ticket, ticket);
            c.push(&record).unwrap();
            append(&mut bytes, &record);
        }
        closing = paired((first + 7) * 2_000 + 1_101);
        let record = c.close_block(closing).unwrap();
        let StructuredServiceRecordV7::BlockClose {
            discoveries,
            freezes,
            ..
        } = &record
        else {
            unreachable!()
        };
        if block == 1 {
            assert!(freezes.is_empty());
            assert_eq!(discoveries.len(), 1);
            let scope = &discoveries[0].scope;
            assert_eq!(scope.owner.rows, 2);
            assert!(scope.numerical_family.is_some());
        } else {
            assert!(discoveries.is_empty());
            assert_eq!(freezes.len(), 1);
            let freeze = &freezes[0];
            assert_eq!(freeze.failure, None, "{freeze:?}");
            assert_eq!(freeze.close.member_count, 8);
            assert_eq!(
                (freeze.domain.owner_offered, freeze.domain.eligible),
                (8, 8)
            );
            assert_eq!(freeze.close.phase, phase_at((block - 2) as usize));
            assert_eq!(freeze.close.boundary.first_offered, first);
            assert_eq!(freeze.close.boundary.last_offered, first + 7);
            assert_eq!(freeze.close.boundary.first_block, block);
            assert_eq!(freeze.close.boundary.last_block, block);
        }
        append(&mut bytes, &record);
    }
    assert_eq!(c.offered(), 32);
    assert_eq!(c.audit().owners.len(), 1);
    assert_eq!(c.qualified_children(), 1);
    let (record, checkpoint) = c.checkpoint(closing).unwrap();
    append(&mut bytes, &record);
    assert_eq!(
        c.source_receipt(),
        (bytes.len() as u64, Sha256::digest(&bytes).into())
    );
    (bytes, c, checkpoint, closing)
}

fn future(
    domain: &CostWorkloadDomainV1,
    rows: usize,
    graph: ActualWaveGraphState,
) -> StructuredQueryV2 {
    let names: Vec<_> = (0..rows)
        .map(|i| format!("source7-independent-future-{i}"))
        .collect();
    let ids: Vec<_> = names.iter().map(String::as_str).collect();
    old::prepared_batch_route_policy_bounds_and_future(
        &ids,
        2,
        1,
        None,
        graph,
        Some(POLICY),
        64,
        &vec![4; rows],
        Some(domain),
    )
    .3
    .unwrap()
}

#[test]
fn source7_family_mixed_width_memory_replay_profile14_and_future_queries_agree() {
    let (bytes, collector, checkpoint, _) = collected_family();
    let limits = CostProfileLoadLimits::default();
    let now = paired(70_000);
    let cutoff = checkpoint.source_receipt();
    assert_eq!(cutoff, collector.source_receipt());
    let memory = checkpoint
        .activate_same_process_memory(now, &limits)
        .unwrap();
    let replay = replay_structured_source_v7(&bytes, &limits)
        .unwrap()
        .activate_same_process_memory(now, &limits)
        .unwrap();
    let files = Files::new(&bytes);
    let receipt = export_structured_profile_v14(
        &files.source,
        Sha256::digest(&bytes).into(),
        cutoff.0,
        &files.profile,
        0,
        &limits,
    )
    .unwrap();
    assert_eq!(receipt.schema_version, 14);
    let imported = load_structured_profile_v14(
        &files.profile,
        &old::fingerprint(),
        &limits,
        ProfileLoadClock {
            wall_unix_ns: Some(now.wall_unix_ns),
            wall_max_error_ns: Some(0),
            monotonic_now_ns: now.monotonic_ns,
        },
    )
    .unwrap();
    for catalog in [&memory, &replay, &imported] {
        assert_eq!(catalog.children.len(), 1);
        assert_eq!(
            (catalog.offered_attempts, catalog.total_shape_rows),
            (32, 54)
        );
        assert_eq!(catalog.source_sha256, cutoff.1);
        let child = &catalog.children[0];
        assert_eq!(child.owner().rows, 2);
        assert!(child.numerical_family_key().is_some());
        assert_eq!(
            child.provenance().phases.each_ref().map(|p| p.members),
            [8; 3]
        );
        assert!(child.same_population(&memory.children[0]));
        assert_eq!(
            child.parameters_signature(),
            memory.children[0].parameters_signature()
        );
    }
    let domain = family_header()
        .declaration
        .nonnegative_envelope
        .unwrap()
        .workload_domain;
    let a = future(&domain, 1, ActualWaveGraphState::Disabled);
    let b = future(&domain, 2, ActualWaveGraphState::Disabled);
    assert_ne!(a.owner(), b.owner());
    assert_eq!(
        a.input().numerical_family_key().unwrap(),
        b.input().numerical_family_key().unwrap()
    );
    for query in [&a, &b] {
        let expected = memory.children[0]
            .predict_query_local(&old::fingerprint(), query, now.monotonic_ns)
            .unwrap();
        for catalog in [&replay, &imported] {
            let actual = catalog.children[0]
                .predict_query_local(&old::fingerprint(), query, now.monotonic_ns)
                .unwrap();
            assert_eq!(
                (actual.planning_ns, actual.valid_until_ns),
                (expected.planning_ns, expected.valid_until_ns)
            );
        }
    }
    let other_route = future(&domain, 2, ActualWaveGraphState::ConfiguredEager);
    assert!(matches!(
        memory.children[0].predict_query_local(&old::fingerprint(), &other_route, now.monotonic_ns),
        Err(StructuredUnknownV2::WrongDomain)
    ));

    // File metadata cannot change the independently replayed family authority.
    let mut altered: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&files.profile).unwrap()).unwrap();
    altered["children"][0]["numerical_family"]["route"][4] = serde_json::json!(7);
    std::fs::write(&files.profile, serde_json::to_vec(&altered).unwrap()).unwrap();
    assert!(load_structured_profile_v14(
        &files.profile,
        &old::fingerprint(),
        &limits,
        ProfileLoadClock {
            wall_unix_ns: Some(now.wall_unix_ns),
            wall_max_error_ns: Some(0),
            monotonic_now_ns: now.monotonic_ns,
        },
    )
    .is_err());
}

#[test]
fn source7_family_request_may_advance_across_phases_but_original_waves_cannot_repeat() {
    // These are the same retained requests on opposite sides of Fit/Residual.
    // The latter wave performs new work and has its own original call/ticket.
    let before = prepared(16);
    let after = prepared(17);
    assert_eq!(before.rows.len(), 2);
    assert_eq!(after.rows.len(), 2);
    for (a, b) in before.rows.iter().zip(&after.rows) {
        assert_eq!(a.request_id, b.request_id);
        assert_eq!(a.owner_incarnation, b.owner_incarnation);
        assert_eq!(a.work_generation + 1, b.work_generation);
        assert_eq!(a.frontier.generated_before + 1, b.frontier.generated_before);
    }
    let (bytes, _, _, _) = collected_family();
    let limits = CostProfileLoadLimits::default();
    assert!(replay_structured_source_v7(&bytes, &limits).is_ok());
    for mutation in 0..6 {
        let mut changed = false;
        let mut modified = Vec::new();
        for (index, line) in bytes.split_inclusive(|b| *b == b'\n').enumerate() {
            if index == 0 {
                modified.extend_from_slice(line);
                continue;
            }
            let mut record: StructuredServiceRecordV7 = serde_json::from_slice(line).unwrap();
            match &mut record {
                StructuredServiceRecordV7::Completed { wave } if wave.ticket == 17 => {
                    match mutation {
                        0 => record = completed(17, 17, 16), // Duplicate original call.
                        1 => record = completed(17, 16, 17), // Duplicate request frontier.
                        2 => {
                            changed = true;
                            continue;
                        } // Missing original offer.
                        _ => {}
                    }
                    changed |= mutation <= 2;
                }
                StructuredServiceRecordV7::BlockOpen {
                    block: 3,
                    assignments,
                    ..
                } if mutation == 3 => {
                    assignments[0].phase = StructuredPhaseV2::Fit;
                    changed = true;
                }
                StructuredServiceRecordV7::BlockClose {
                    block: 2, freezes, ..
                } if mutation == 4 => {
                    freezes.clear();
                    changed = true;
                }
                StructuredServiceRecordV7::BlockClose {
                    block: 3, freezes, ..
                } if mutation == 5 => {
                    freezes[0].close.boundary.first_offered -= 1;
                    changed = true;
                }
                _ => {}
            }
            append(&mut modified, &record);
        }
        assert!(changed);
        assert!(
            replay_structured_source_v7(&modified, &limits).is_err(),
            "mutation {mutation}"
        );
    }
}

#[test]
fn source7_population_policy_is_signed_and_exact_default_stays_omitted() {
    let family = family_header();
    let mut declaration = family.declaration.clone();
    declaration
        .nonnegative_envelope
        .as_mut()
        .unwrap()
        .population_policy = StructuredPopulationPolicyV1::ExactOwnerV1;
    let exact = StructuredServiceHeaderV7::new(
        family.capture_identity,
        family.generation,
        family.fingerprint.clone(),
        family.producer.clone(),
        family.opening,
        declaration.clone(),
        family.maximum_file_bytes,
    )
    .unwrap();
    assert_ne!(family.declaration_sha256, exact.declaration_sha256);
    assert_ne!(family.protocol, exact.protocol);
    let encoded = serde_json::to_value(&exact).unwrap();
    assert!(encoded["declaration"]["nonnegative_envelope"]
        .get("population_policy")
        .is_none());
    assert_eq!(
        record_bytes_v7(&exact).unwrap(),
        record_bytes_v7(&serde_json::from_value::<StructuredServiceHeaderV7>(encoded).unwrap())
            .unwrap()
    );

    // Changing only the claimed grouping must not retain the old signature.
    let mut relabelled = family;
    relabelled.declaration = declaration;
    assert!(
        StructuredServiceCollectorV7::new(relabelled, CostProfileLoadLimits::default()).is_err()
    );
}
