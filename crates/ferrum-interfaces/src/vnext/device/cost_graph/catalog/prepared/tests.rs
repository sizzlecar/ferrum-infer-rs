use super::*;
use crate::vnext::*;

fn source(lane: ExecutionLaneId) -> DeviceCostGraphCatalogSource {
    DeviceCostGraphCatalogSource::new(1, 7, &"b".repeat(64), lane).unwrap()
}
fn state(programs: u64) -> DeviceCostGraphStreamState {
    DeviceCostGraphStreamState::new(
        DeviceCostGraphConfiguration::OnDemand,
        programs,
        programs,
        0,
    )
    .unwrap()
}
fn program(lane: ExecutionLaneId, slot: u64) -> DeviceReusableExecutionProgram {
    let bucket = ReusableExecutionBucketSpec::new(
        ReusableExecutionClassId::new("prepared.catalog").unwrap(),
        ReusableExecutionCapacity::new(1, 1, 1).unwrap(),
    )
    .unwrap();
    let id = DeviceReusableExecutionProgramId::new(
        serde_json::from_value(serde_json::json!("a".repeat(64))).unwrap(),
        "b".repeat(64),
        lane,
        bucket.bucket_id().clone(),
        "c".repeat(64),
        "d".repeat(64),
        slot,
        1,
        1,
        1,
    )
    .unwrap();
    let capture = DeviceReusableExecutionCapture::new(id, 3, vec![1], vec![]).unwrap();
    DeviceReusableExecutionProgram::new(
        &capture,
        vec![
            DeviceReusableExecutionSegment::new(0, 0, 1, 1).unwrap(),
            DeviceReusableExecutionSegment::new(1, 2, 3, 1).unwrap(),
        ],
        vec![],
        vec![],
    )
    .unwrap()
}
fn row(node: u32) -> DeviceReplayedLogicalCommandAttribution {
    DeviceReplayedLogicalCommandAttribution::new(
        0,
        node,
        DeviceNativeOperationId::new("test.prepared.catalog").unwrap(),
        DeviceBatchingForm::Scalar,
        1,
        1,
        1,
        0,
        1,
    )
    .unwrap()
}
fn build(
    source: &DeviceCostGraphCatalogSource,
    budget: &Arc<DeviceObservationTemplateBudget>,
    programs: &[DeviceReusableExecutionProgram],
) -> Arc<DevicePreparedCostGraphCatalog> {
    let mut builder = source
        .begin(state(programs.len() as u64), budget, &mut || Ok(()))
        .unwrap();
    for program in programs {
        builder.push_program(program, &mut || Ok(())).unwrap();
        for segment in program.segments() {
            builder
                .push_uploaded_segment(
                    segment,
                    &"e".repeat(64),
                    &[row(segment.start_node_index())],
                    &mut || Ok(()),
                )
                .unwrap();
        }
    }
    builder.finish(&mut || Ok(())).unwrap()
}
fn query_limits() -> DeviceCostGraphCatalogLimits {
    DeviceCostGraphCatalogLimits::new(1, 3, 2).unwrap()
}

fn capacity_bytes(error: DevicePreparedCostGraphCatalogBuildError) -> usize {
    match error {
        DevicePreparedCostGraphCatalogBuildError::Capacity {
            required_exclusive_peak_bytes,
        } => required_exclusive_peak_bytes,
        other => panic!("expected typed quota refusal, got {other:?}"),
    }
}

#[test]
fn prepared_catalog_capacity_receipt_survives_external_release_before_rollback() {
    let lane = ExecutionLaneId::mint().unwrap();
    let source = source(lane);
    let p = program(lane, 1);
    let probe = DeviceObservationTemplateBudget::new(1 << 20).unwrap();
    let mut builder = source.begin(state(1), &probe, &mut || Ok(())).unwrap();
    let initial = probe.retained_payload_bytes();
    builder.push_program(&p, &mut || Ok(())).unwrap();
    let required = probe.peak_retained_payload_bytes();
    drop(builder);
    assert_eq!(probe.retained_payload_bytes(), 0);
    assert!(required > initial);

    let budget = DeviceObservationTemplateBudget::new(required).unwrap();
    let mut builder = source.begin(state(1), &budget, &mut || Ok(())).unwrap();
    let competitor = budget.reserve(1).unwrap();
    let receipt = capacity_bytes(builder.push_program(&p, &mut || Ok(())).unwrap_err());
    assert_eq!(
        receipt, required,
        "other owners must not inflate exclusive demand"
    );
    // The external owner releases after quota refusal but before this failed
    // builder rolls back. A retained-after-failure gate misses this recovery.
    drop(competitor);
    assert_eq!(budget.retained_payload_bytes(), initial);
    drop(builder);
    assert!(budget.maximum_bytes() - budget.retained_payload_bytes() >= receipt);

    // A newly competing lease may still defeat the retry. The typed lower
    // bound is never treated as permission to exceed the real shared quota.
    let mut retry = source.begin(state(1), &budget, &mut || Ok(())).unwrap();
    let competitor = budget.reserve(1).unwrap();
    assert_eq!(
        capacity_bytes(retry.push_program(&p, &mut || Ok(())).unwrap_err()),
        receipt
    );
    drop(retry);
    drop(competitor);
    let mut retry = source.begin(state(1), &budget, &mut || Ok(())).unwrap();
    retry.push_program(&p, &mut || Ok(())).unwrap();
    let root = retry.finish(&mut || Ok(())).unwrap();
    assert!(root.is_current(&source, state(1)));
    drop(root);
    assert_eq!(budget.retained_payload_bytes(), 0);
}

#[test]
fn prepared_catalog_capacity_receipt_does_not_count_rollback_as_new_capacity() {
    let lane = ExecutionLaneId::mint().unwrap();
    let source = source(lane);
    let p = program(lane, 1);
    let probe = DeviceObservationTemplateBudget::new(1 << 20).unwrap();
    let mut builder = source.begin(state(1), &probe, &mut || Ok(())).unwrap();
    builder.push_program(&p, &mut || Ok(())).unwrap();
    let required = probe.peak_retained_payload_bytes();
    drop(builder);
    let budget = DeviceObservationTemplateBudget::new(required - 1).unwrap();
    let mut builder = source.begin(state(1), &budget, &mut || Ok(())).unwrap();
    let receipt = capacity_bytes(builder.push_program(&p, &mut || Ok(())).unwrap_err());
    assert_eq!(receipt, required);
    drop(builder);
    assert_eq!(budget.retained_payload_bytes(), 0);
    assert!(budget.maximum_bytes() - budget.retained_payload_bytes() < receipt);

    let minimal = DeviceObservationTemplateBudget::new(1).unwrap();
    let error = match source.begin(state(1), &minimal, &mut || Ok(())) {
        Ok(_) => panic!("a one-byte budget cannot hold the catalog header"),
        Err(error) => error,
    };
    assert!(capacity_bytes(error) > minimal.maximum_bytes());
    assert_eq!(minimal.retained_payload_bytes(), 0);
    assert!(matches!(
        source.begin(state(1), &probe, &mut || Err(invalid("cancelled"))),
        Err(DevicePreparedCostGraphCatalogBuildError::Rejected(_))
    ));
}

#[test]
fn prepared_catalog_full_id_lookup_preserves_canonical_order_without_query_inventory_limits() {
    let lane = ExecutionLaneId::mint().unwrap();
    let source = source(lane);
    let budget = DeviceObservationTemplateBudget::new(1 << 20).unwrap();
    let programs: Vec<_> = (1..=19).rev().map(|slot| program(lane, slot)).collect();
    let root = build(&source, &budget, &programs);
    // The complete inventory exceeds a one-program query's node/work limits.
    assert!(!root.catalog().fits(query_limits()));
    assert!(root.is_current(&source, state(19)));
    for slot in 1..=19 {
        let expected = program(lane, slot);
        let found = root
            .lookup(expected.program_id(), query_limits(), &mut || Ok(()))
            .unwrap()
            .unwrap();
        assert_eq!(found.program(), &expected);
        assert_eq!(found.uploaded_segments().len(), 2);
    }
    assert!(root
        .catalog()
        .programs()
        .windows(2)
        .all(|p| p[0].program().program_id() < p[1].program().program_id()));
    let absent = program(lane, 20);
    assert!(root
        .lookup(absent.program_id(), query_limits(), &mut || Ok(()))
        .unwrap()
        .is_none());
}

#[test]
fn prepared_catalog_empty_absence_is_bound_to_the_real_lane_and_private_owner() {
    let lane = ExecutionLaneId::mint().unwrap();
    let source = source(lane);
    let budget = DeviceObservationTemplateBudget::new(1 << 20).unwrap();
    let root = build(&source, &budget, &[]);
    assert!(root.is_current(&source, state(0)));
    assert!(source.matches_owner(1, 7, &"b".repeat(64)));
    assert!(!source.matches_owner(2, 7, &"b".repeat(64)));
    assert!(!source.matches_owner(1, 8, &"b".repeat(64)));
    assert!(root
        .lookup(
            program(lane, 1).program_id(),
            query_limits(),
            &mut || Ok(())
        )
        .unwrap()
        .is_none());
    let foreign = ExecutionLaneId::mint().unwrap();
    assert!(!root.matches_runtime_lane(&"b".repeat(64), foreign));
    assert!(root
        .lookup(
            program(foreign, 1).program_id(),
            query_limits(),
            &mut || Ok(())
        )
        .is_err());
    // Even equal public owner numbers cannot recreate a private source token.
    let recycled = DeviceCostGraphCatalogSource::new(1, 7, &"b".repeat(64), lane).unwrap();
    assert!(!root.is_current(&recycled, state(0)));
}

#[test]
fn prepared_catalog_mutation_invalidates_equal_numeric_roots_and_history_keeps_its_lease() {
    let lane = ExecutionLaneId::mint().unwrap();
    let mut source = source(lane);
    let budget = DeviceObservationTemplateBudget::new(1 << 20).unwrap();
    let p = program(lane, 1);
    let old = build(&source, &budget, &[p.clone()]);
    let old_bytes = budget.retained_payload_bytes();
    let held_history = Arc::clone(&old);
    source.invalidate();
    assert!(!old.is_current(&source, state(1)));
    let fresh = build(&source, &budget, &[p]);
    assert_eq!(old.catalog(), fresh.catalog());
    assert!(!old.same_generation(&fresh));
    assert!(fresh.is_current(&source, state(1)));
    assert!(budget.retained_payload_bytes() > old_bytes);
    drop(old);
    assert!(budget.retained_payload_bytes() > old_bytes);
    drop(fresh);
    assert_eq!(budget.retained_payload_bytes(), old_bytes);
    drop(held_history);
    assert_eq!(budget.retained_payload_bytes(), 0);
}

#[test]
fn prepared_catalog_partial_and_evicted_descriptors_are_preserved_without_replay_claims() {
    let lane = ExecutionLaneId::mint().unwrap();
    let source = source(lane);
    let budget = DeviceObservationTemplateBudget::new(1 << 20).unwrap();
    let p = program(lane, 1);
    let capture =
        DeviceReusableExecutionCapture::new(p.program_id().clone(), 3, vec![1], vec![]).unwrap();
    let evicted = DeviceReusableExecutionProgram::new(
        &capture,
        vec![],
        vec![],
        vec![
            DeviceReusableExecutionProgramGap::new(
                0,
                DeviceReusableExecutionProgramGapReason::Evicted,
            ),
            DeviceReusableExecutionProgramGap::new(
                2,
                DeviceReusableExecutionProgramGapReason::Evicted,
            ),
        ],
    )
    .unwrap();
    let root = build(&source, &budget, &[evicted.clone()]);
    let selected = root
        .lookup(evicted.program_id(), query_limits(), &mut || Ok(()))
        .unwrap()
        .unwrap();
    assert_eq!(selected.program(), &evicted);
    assert!(selected.uploaded_segments().is_empty());
    assert!(!selected.program().is_determinism_ready());
}

#[test]
fn prepared_catalog_shared_cpu_budget_is_reserved_before_copy_and_released_on_failure() {
    let lane = ExecutionLaneId::mint().unwrap();
    let source = source(lane);
    let budget = DeviceObservationTemplateBudget::new(1 << 20).unwrap();
    let p = program(lane, 1);
    let root = build(&source, &budget, &[p.clone()]);
    let retained = budget.retained_payload_bytes();
    let peak = budget.peak_retained_payload_bytes();
    assert!(
        peak > retained,
        "finish must release temporary construction allowance"
    );
    drop(root);
    assert_eq!(budget.retained_payload_bytes(), 0);

    let tight = DeviceObservationTemplateBudget::new(peak).unwrap();
    let first = build(&source, &tight, &[p.clone()]);
    let retained = tight.retained_payload_bytes();
    let ordinary_template_reservation = tight.reserve(1);
    // Both APIs charge one real ledger; a second full root cannot silently
    // allocate a second independent budget while the first remains captured.
    let attempt = (|| {
        let mut second = source.begin(state(1), &tight, &mut || Ok(()))?;
        second.push_program(&p, &mut || Ok(()))?;
        for segment in p.segments() {
            second.push_uploaded_segment(
                segment,
                &"e".repeat(64),
                &[row(segment.start_node_index())],
                &mut || Ok(()),
            )?;
        }
        second.finish(&mut || Ok(()))
    })();
    assert!(attempt.is_err());
    drop(ordinary_template_reservation);
    assert_eq!(tight.retained_payload_bytes(), retained);
    drop(first);
    assert_eq!(tight.retained_payload_bytes(), 0);
}

#[test]
fn prepared_catalog_rejects_foreign_duplicate_incomplete_or_cancelled_builds() {
    let lane = ExecutionLaneId::mint().unwrap();
    let source = source(lane);
    let budget = DeviceObservationTemplateBudget::new(1 << 20).unwrap();
    let p = program(lane, 1);
    let foreign = program(ExecutionLaneId::mint().unwrap(), 1);
    let mut builder = source.begin(state(1), &budget, &mut || Ok(())).unwrap();
    assert!(builder.push_program(&foreign, &mut || Ok(())).is_err());
    assert!(builder.finish(&mut || Ok(())).is_err());
    assert_eq!(budget.retained_payload_bytes(), 0);
    let mut builder = source.begin(state(2), &budget, &mut || Ok(())).unwrap();
    builder.push_program(&p, &mut || Ok(())).unwrap();
    builder.push_program(&p, &mut || Ok(())).unwrap();
    assert!(builder.finish(&mut || Ok(())).is_err());
    assert_eq!(budget.retained_payload_bytes(), 0);
    let builder = source.begin(state(1), &budget, &mut || Ok(())).unwrap();
    assert!(builder.finish(&mut || Ok(())).is_err());
    assert_eq!(budget.retained_payload_bytes(), 0);
    let mut builder = source.begin(state(1), &budget, &mut || Ok(())).unwrap();
    assert!(builder
        .push_program(&p, &mut || Err(invalid("cancelled")))
        .is_err());
    assert!(builder.finish(&mut || Ok(())).is_err());
    assert_eq!(budget.retained_payload_bytes(), 0);
}

#[test]
fn prepared_catalog_lookup_enforces_selected_work_and_cancellation_on_hits_and_absence() {
    let lane = ExecutionLaneId::mint().unwrap();
    let source = source(lane);
    let budget = DeviceObservationTemplateBudget::new(1 << 20).unwrap();
    let p = program(lane, 1);
    let root = build(&source, &budget, &[p.clone()]);
    assert!(root
        .lookup(
            p.program_id(),
            DeviceCostGraphCatalogLimits::new(1, 2, 2).unwrap(),
            &mut || Ok(())
        )
        .is_err());
    assert!(root
        .lookup(
            p.program_id(),
            DeviceCostGraphCatalogLimits::new(1, 3, 1).unwrap(),
            &mut || Ok(())
        )
        .is_err());
    for id in [
        p.program_id().clone(),
        program(lane, 2).program_id().clone(),
    ] {
        let mut calls = 0;
        assert!(root
            .lookup(&id, query_limits(), &mut || {
                calls += 1;
                if calls >= 3 {
                    Err(invalid("expired original query budget"))
                } else {
                    Ok(())
                }
            })
            .is_err());
    }
}
