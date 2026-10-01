use super::*;

fn records(
    bytes: &[u8],
) -> (
    StructuredPreparedOwnerBlockHeaderV8,
    Vec<StructuredPreparedOwnerBlockRecordV8>,
) {
    let mut lines = bytes.split_inclusive(|v| *v == b'\n');
    let header = serde_json::from_slice(lines.next().unwrap()).unwrap();
    let records = lines.map(|l| serde_json::from_slice(l).unwrap()).collect();
    (header, records)
}
fn before(
    h: &StructuredPreparedOwnerBlockHeaderV8,
    records: &[StructuredPreparedOwnerBlockRecordV8],
) -> StructuredPreparedOwnerBlockCollectorV8 {
    let mut c =
        StructuredPreparedOwnerBlockCollectorV8::new(h.clone(), CostProfileLoadLimits::default())
            .unwrap();
    for r in records {
        c.push(r).unwrap();
    }
    c
}

#[test]
fn source8_partial_tail_retains_actual_population_and_prior_qualified_models() {
    let (bytes, c, checkpoint) = collector::collected_with(24, [16, 8, 9]);
    let audit = c.prepared_audit();
    let tail = audit.sealed_partial_tail.unwrap();
    assert_eq!((tail.offered, tail.tail_offered, tail.block), (99, 3, 5));
    assert_eq!(tail.accepted_fifo_cutoff, c.last_fifo());
    assert_eq!(audit.population.offered, 99);
    assert_eq!(audit.preparation_attempts, 33);
    let replay = replay_structured_source_v8(&bytes, &CostProfileLoadLimits::default()).unwrap();
    assert_eq!(replay.source_receipt(), checkpoint.source_receipt());
    assert_eq!(replay.qualified_children(), checkpoint.qualified_children());
    let now = StructuredServiceClockV7 {
        monotonic_ns: tail.closing.monotonic_ns + 100,
        wall_unix_ns: tail.closing.wall_unix_ns + 100,
    };
    let memory = checkpoint
        .activate_same_process_memory(now, &CostProfileLoadLimits::default())
        .unwrap();
    assert!(!memory.children.is_empty());
    assert_eq!(memory.offered_attempts, 99);
    // Each numerical phase was frozen at a prior complete original block.
    for child in memory.children {
        assert!(child
            .provenance()
            .phases
            .iter()
            .all(|p| p.accepted_fifo_cutoff <= 96));
    }
    assert!(replay_structured_source_v7(&bytes, &CostProfileLoadLimits::default()).is_err());
}

#[test]
fn source8_partial_tail_cannot_complete_qualification_or_become_a_later_block() {
    let (bytes, mut c, checkpoint) = collector::collected_with_qualification(24, [16, 8, 7], false);
    assert_eq!(c.qualified_children(), 0);
    assert_eq!(checkpoint.qualified_children(), 0);
    let audit = c.prepared_audit();
    assert_eq!(audit.sealed_partial_tail.as_ref().unwrap().tail_offered, 21);
    assert!(audit
        .population
        .owners
        .iter()
        .any(|o| { o.phase == Some(StructuredPhaseV2::Qualification) && o.eligible >= 8 }));
    let replay = replay_structured_source_v8(&bytes, &CostProfileLoadLimits::default()).unwrap();
    assert_eq!(replay.qualified_children(), 0);
    let tail = audit.sealed_partial_tail.unwrap();
    assert!(c
        .open_block(tail.closing.monotonic_ns + 1, tail.accepted_fifo_cutoff)
        .is_err());
}

#[test]
fn source8_partial_tail_rejects_unfinished_cohort_preparation_and_terminal() {
    let (bytes, _, _) = collector::collected_with(24, [16, 8, 9]);
    let (h, r) = records(&bytes);
    let closing = r
        .iter()
        .find_map(|r| match r {
            StructuredPreparedOwnerBlockRecordV8::Tail(
                StructuredPreparedTailRecordV8::PartialTailClosed { tail },
            ) => Some(tail.closing),
            _ => None,
        })
        .unwrap();
    for kind in ["preparation_completed", "request_completed", "cohort_end"] {
        let index = r
            .iter()
            .rposition(|r| serde_json::to_value(r).unwrap()["kind"] == kind)
            .unwrap();
        let mut c = before(&h, &r[..index]);
        assert!(
            c.seal_complete_cohorts_with_partial_tail(closing).is_err(),
            "{kind}"
        );
        assert!(c.audit().poisoned);
    }
}

#[test]
fn source8_partial_tail_rejects_forged_counts_fifo_hash_clock_or_full_block() {
    let (bytes, _, _) = collector::collected_with(24, [16, 8, 9]);
    let (h, r) = records(&bytes);
    let index = r
        .iter()
        .position(|r| matches!(r, StructuredPreparedOwnerBlockRecordV8::Tail(_)))
        .unwrap();
    for field in [
        "offered",
        "tail_offered",
        "accepted_fifo_cutoff",
        "source_prefix_bytes",
    ] {
        let mut value = serde_json::to_value(&r[index]).unwrap();
        value["tail"][field] = serde_json::json!(value["tail"][field].as_u64().unwrap() + 1);
        let bad = serde_json::from_value(value).unwrap();
        let mut c = before(&h, &r[..index]);
        assert!(c.push(&bad).is_err(), "{field}");
        assert!(c.audit().poisoned);
    }
    let mut value = serde_json::to_value(&r[index]).unwrap();
    value["tail"]["source_prefix_sha256"][0] = serde_json::json!(0);
    let original = serde_json::to_value(&r[index]).unwrap();
    value["tail"]["source_prefix_sha256"][0] = serde_json::json!(
        (original["tail"]["source_prefix_sha256"][0]
            .as_u64()
            .unwrap()
            + 1)
            % 256
    );
    assert!(before(&h, &r[..index])
        .push(&serde_json::from_value(value).unwrap())
        .is_err());
    let mut value = serde_json::to_value(&r[index]).unwrap();
    value["tail"]["closing"]["monotonic_ns"] =
        serde_json::json!(h.opening.monotonic_ns + h.declaration.population.maximum_window_ns + 1);
    assert!(before(&h, &r[..index])
        .push(&serde_json::from_value(value).unwrap())
        .is_err());

    let (full, _, _) = collector::collected();
    let (h, r) = records(&full);
    let index = r
        .iter()
        .rposition(|r| {
            matches!(
                r,
                StructuredPreparedOwnerBlockRecordV8::Population(
                    StructuredServiceRecordV7::BlockClose { .. }
                )
            )
        })
        .unwrap();
    let StructuredPreparedOwnerBlockRecordV8::Population(StructuredServiceRecordV7::BlockClose {
        closing,
        ..
    }) = &r[index]
    else {
        unreachable!()
    };
    assert!(before(&h, &r[..index])
        .seal_complete_cohorts_with_partial_tail(*closing)
        .is_err());
}

#[test]
fn source8_tail_policy_cannot_be_omitted_from_original_header() {
    let mut value = serde_json::to_value(header()).unwrap();
    assert_eq!(value["tail_policy"], "complete_cohorts_audit_only_v1");
    value.as_object_mut().unwrap().remove("tail_policy");
    assert!(serde_json::from_value::<StructuredPreparedOwnerBlockHeaderV8>(value).is_err());
}
