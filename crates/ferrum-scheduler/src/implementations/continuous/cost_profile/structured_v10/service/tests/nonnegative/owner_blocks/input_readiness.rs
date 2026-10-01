//! Canonical producer, real source7 block protocol, independent replay.
use super::*;
use crate::implementations::continuous::cost_model::structured_v2::OwnerInputReadinessV1;
use ferrum_interfaces::execution_cost::ActualWaveGraphState;

fn ready_header(visits: u64) -> StructuredServiceHeaderV7 {
    let mut h = block_header();
    h.declaration.settings.max_phase_samples = h.declaration.settings.max_phase_samples.max(16);
    h.declaration.schedule = OwnerBlockScheduleV1::new_with_input_readiness(
        8,
        [8; 3],
        [8; 3],
        OwnerInputReadinessV1::new([2; 3], visits).unwrap(),
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

fn input_wave(ticket: u64, length: bool) -> StructuredServiceRecordV7 {
    // A different original request per offer: no invented continuation or
    // dropped frontier. Maximum output changes the actual canonical Length.
    let (p, _, _) = old::prepared_route_policy_bounds(
        &format!("input-readiness-{ticket}"),
        2,
        1,
        None,
        ActualWaveGraphState::Disabled,
        None,
        64,
        if length { 2 } else { 3 },
    );
    let stages = old::stages(&old::header(), &p, ticket, 1_000);
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

fn next_block(
    c: &mut StructuredServiceCollectorV7,
    bytes: &mut Vec<u8>,
    block: u64,
    mixed: bool,
) -> StructuredServiceRecordV7 {
    let first = (block - 1) * 8 + 1;
    append(
        bytes,
        &c.open_block(first * 2_000 - 1, (first - 1) * 3).unwrap(),
    );
    for ticket in first..first + 8 {
        let r = input_wave(ticket, mixed && ticket % 2 == 0);
        c.push(&r).unwrap();
        append(bytes, &r);
    }
    let r = c.close_block(paired((first + 7) * 2_000 + 1_101)).unwrap();
    append(bytes, &r);
    r
}

#[test]
fn source7_input_readiness_waits_for_declared_length_in_each_independent_phase_and_replays() {
    let h = ready_header(32_000_000);
    let mut bytes = record_bytes_v7(&h).unwrap();
    let mut c = StructuredServiceCollectorV7::new(h, CostProfileLoadLimits::default()).unwrap();
    for block in 1..=7 {
        let r = next_block(&mut c, &mut bytes, block, block % 2 == 1);
        let StructuredServiceRecordV7::BlockClose {
            freezes,
            discoveries,
            ..
        } = r
        else {
            unreachable!()
        };
        if block == 1 {
            assert_eq!(discoveries.len(), 1);
            assert!(discoveries[0].contract.input_target.is_some());
        } else if block % 2 == 0 {
            assert!(
                freezes.is_empty(),
                "count minimum alone cannot close this population"
            );
            assert_eq!(c.audit().owners[0].eligible, 8);
        } else {
            assert_eq!(freezes.len(), 1);
            assert_eq!(freezes[0].failure, None, "{:?}", freezes[0]);
            assert_eq!(freezes[0].close.member_count, 16);
            assert_eq!(freezes[0].close.boundary.first_block, block - 1);
            assert_eq!(freezes[0].close.boundary.last_block, block);
        }
    }
    assert_eq!(c.qualified_children(), 1);
    let closing = paired(56 * 2_000 + 1_101);
    let (r, original) = c.checkpoint(closing).unwrap();
    append(&mut bytes, &r);
    let limits = CostProfileLoadLimits::default();
    let replayed = replay_structured_source_v7(&bytes, &limits).unwrap();
    let a = original
        .activate_same_process_memory(paired(120_000), &limits)
        .unwrap();
    let b = replayed
        .activate_same_process_memory(paired(120_000), &limits)
        .unwrap();
    assert_eq!(
        a.children[0].parameters_signature(),
        b.children[0].parameters_signature()
    );
    assert_eq!(
        a.children[0]
            .provenance()
            .phases
            .each_ref()
            .map(|p| p.members),
        [16; 3]
    );
    assert_eq!(
        b.children[0]
            .provenance()
            .phases
            .each_ref()
            .map(|p| p.members),
        [16; 3]
    );

    // The original producer inputs, not a claimed target in the close DTO,
    // determine the target. Reject at that close before later hash receipts.
    let mut forged = Vec::new();
    let mut changed = false;
    for line in bytes.split_inclusive(|&b| b == b'\n') {
        let mut value: serde_json::Value = serde_json::from_slice(line).unwrap();
        if !changed {
            if let Some(target) = value
                .get_mut("discoveries")
                .and_then(|d| d.get_mut(0))
                .and_then(|d| d.get_mut("contract"))
                .and_then(|c| c.get_mut("input_target"))
            {
                target["branches"][1] = serde_json::json!(1);
                changed = true;
            }
        }
        forged.extend(record_bytes_v7(&value).unwrap());
    }
    assert!(changed);
    let error = match replay_structured_source_v7(&forged, &limits) {
        Err(error) => error,
        Ok(_) => panic!("forged discovery target acquired authority"),
    };
    assert!(
        error
            .to_string()
            .contains("original block/freeze/certificate"),
        "{error}"
    );
}

#[test]
fn source7_input_readiness_missing_branch_and_geometry_budget_end_without_publication() {
    for (visits, last, reason) in [
        (32_000_000, 3, "MissingInputCoverage"),
        (1, 2, "GeometryWorkBudget"),
    ] {
        let h = ready_header(visits);
        let mut bytes = record_bytes_v7(&h).unwrap();
        let mut c = StructuredServiceCollectorV7::new(h, CostProfileLoadLimits::default()).unwrap();
        next_block(&mut c, &mut bytes, 1, true);
        for block in 2..=last {
            next_block(&mut c, &mut bytes, block, false);
        }
        let audit = c.audit();
        assert_eq!(c.qualified_children(), 0);
        assert_eq!(
            audit.owners[0].eligible, 0,
            "failed phase releases retained samples"
        );
        assert!(
            audit.owners[0].failure.as_deref().unwrap().contains(reason),
            "{:?}",
            audit.owners[0]
        );
        let closing = paired(last * 8 * 2_000 + 1_101);
        let (_, checkpoint) = c.checkpoint(closing).unwrap();
        assert_eq!(checkpoint.qualified_children(), 0);
        assert!(checkpoint
            .activate_same_process_memory(closing, &CostProfileLoadLimits::default())
            .is_err());
    }
}

#[test]
fn source7_input_readiness_none_preserves_legacy_schedule_wire() {
    let schedule = OwnerBlockScheduleV1::new(8, [8; 3], [8; 3]).unwrap();
    let expected = serde_json::json!({"block_offered":8,"phase_min_offered":[8,8,8],"min_members":[8,8,8],"maximum_phase_members":[15,15,15]});
    assert_eq!(serde_json::to_value(&schedule).unwrap(), expected);
    let decoded: OwnerBlockScheduleV1 = serde_json::from_value(expected).unwrap();
    assert_eq!(decoded, schedule);
    assert!(
        !serde_json::to_value(block_header()).unwrap()["declaration"]["schedule"]
            .as_object()
            .unwrap()
            .contains_key("input_readiness")
    );
}

#[test]
fn source7_input_readiness_v2_preserves_original_phase_receipts_and_replays() {
    let mut h = ready_header(32_000_000);
    h.declaration.schedule.input_readiness =
        Some(OwnerInputReadinessV1::new_cached_residual_v2([2; 3], 32_000_000).unwrap());
    let h = StructuredServiceHeaderV7::new(
        h.capture_identity,
        h.generation,
        h.fingerprint,
        h.producer,
        h.opening,
        h.declaration,
        h.maximum_file_bytes,
    )
    .unwrap();
    let mut bytes = record_bytes_v7(&h).unwrap();
    let mut c = StructuredServiceCollectorV7::new(h, CostProfileLoadLimits::default()).unwrap();
    for block in 1..=7 {
        let record = next_block(&mut c, &mut bytes, block, block % 2 == 1);
        let StructuredServiceRecordV7::BlockClose { freezes, .. } = record else {
            unreachable!()
        };
        if block == 1 || block % 2 == 0 {
            assert!(
                freezes.is_empty(),
                "counts cannot bypass the late Length input"
            );
        } else {
            assert_eq!(freezes.len(), 1);
            assert_eq!(freezes[0].failure, None);
            assert_eq!(freezes[0].close.member_count, 16);
            assert_eq!(freezes[0].close.boundary.first_block, block - 1);
            assert_eq!(freezes[0].close.boundary.last_block, block);
        }
    }
    let limits = CostProfileLoadLimits::default();
    let (record, original) = c.checkpoint(paired(56 * 2_000 + 1_101)).unwrap();
    append(&mut bytes, &record);
    let replayed = replay_structured_source_v7(&bytes, &limits).unwrap();
    let original = original
        .activate_same_process_memory(paired(120_000), &limits)
        .unwrap();
    let replayed = replayed
        .activate_same_process_memory(paired(120_000), &limits)
        .unwrap();
    assert_eq!(original.children.len(), 1);
    assert_eq!(replayed.children.len(), 1);
    assert_eq!(
        original.children[0].parameters_signature(),
        replayed.children[0].parameters_signature()
    );
    assert_eq!(
        original.children[0].provenance().phases,
        replayed.children[0].provenance().phases
    );
    assert_eq!(
        original.children[0].provenance().source_sha256,
        replayed.children[0].provenance().source_sha256
    );
    assert_eq!(
        original.children[0]
            .provenance()
            .phases
            .each_ref()
            .map(|p| p.members),
        [16; 3]
    );
    // Revision participates in the original declaration. Relabelling V2 as V1
    // cannot import a source with different work and earliest-close semantics.
    let mut forged = Vec::new();
    for (index, line) in bytes.split_inclusive(|&b| b == b'\n').enumerate() {
        if index == 0 {
            let mut header: serde_json::Value = serde_json::from_slice(line).unwrap();
            header["declaration"]["schedule"]["input_readiness"]["revision"] =
                serde_json::json!("work_axes_and_branches_v1");
            forged.extend(record_bytes_v7(&header).unwrap());
        } else {
            forged.extend_from_slice(line);
        }
    }
    assert!(replay_structured_source_v7(&forged, &limits).is_err());
}
