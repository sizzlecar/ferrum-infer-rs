use super::*;
use crate::vnext::*;

fn program(lane: ExecutionLaneId, slot: u64) -> DeviceReusableExecutionProgram {
    let bucket = ReusableExecutionBucketSpec::new(
        ReusableExecutionClassId::new("catalog.test").unwrap(),
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
fn row(node: u32, tokens: u64) -> DeviceReplayedLogicalCommandAttribution {
    DeviceReplayedLogicalCommandAttribution::new(
        0,
        node,
        DeviceNativeOperationId::new("test.catalog.kernel").unwrap(),
        DeviceBatchingForm::Scalar,
        1,
        tokens,
        1,
        0,
        1,
    )
    .unwrap()
}
fn limits() -> DeviceCostGraphCatalogLimits {
    DeviceCostGraphCatalogLimits::new(4, 16, 16).unwrap()
}
fn state(programs: u64) -> DeviceCostGraphStreamState {
    DeviceCostGraphStreamState::new(DeviceCostGraphConfiguration::OnDemand, 4, programs, 0).unwrap()
}
fn inventory(
    programs: &[DeviceReusableExecutionProgram],
    tokens: u64,
    fp: char,
) -> DeviceCostGraphCatalog {
    let mut b = DeviceCostGraphCatalogBuilder::new(state(programs.len() as u64), limits()).unwrap();
    for p in programs {
        b.push_program(p, &mut || Ok(())).unwrap();
        b.push_uploaded_segment(
            &p.segments()[1],
            &fp.to_string().repeat(64),
            &[row(2, tokens)],
            &mut || Ok(()),
        )
        .unwrap();
    }
    b.finish(&mut || Ok(())).unwrap()
}

#[test]
fn graph_catalog_canonical_order_and_uploaded_subset_are_explicit() {
    let lane = ExecutionLaneId::mint().unwrap();
    let a = program(lane, 1);
    let b = program(lane, 2);
    let x = inventory(&[a.clone(), b.clone()], 1, 'e');
    let y = inventory(&[b, a], 1, 'e');
    assert_eq!(x, y);
    assert_eq!(x.programs()[0].program().segments().len(), 2);
    assert_eq!(x.programs()[0].uploaded_segments().len(), 1);
    assert_eq!(
        x.programs()[0].uploaded_segments()[0].segment().ordinal(),
        1
    );
    assert!(x.fits(limits()));
    assert!(!x.fits(DeviceCostGraphCatalogLimits::new(1, 16, 16).unwrap()));
}

#[test]
fn graph_catalog_identity_includes_logical_work_and_executable_not_just_counts() {
    let p = program(ExecutionLaneId::mint().unwrap(), 1);
    let original = inventory(&[p.clone()], 1, 'e');
    let changed_work = inventory(&[p.clone()], 2, 'e');
    let changed_executable = inventory(&[p], 1, 'f');
    assert_eq!(original.stream_state(), changed_work.stream_state());
    assert_ne!(original, changed_work);
    assert_ne!(original, changed_executable);
}

#[test]
fn graph_catalog_rejects_duplicate_missing_program_and_wrong_or_repeated_segment() {
    let p = program(ExecutionLaneId::mint().unwrap(), 1);
    let mut duplicate = DeviceCostGraphCatalogBuilder::new(state(2), limits()).unwrap();
    duplicate.push_program(&p, &mut || Ok(())).unwrap();
    assert!(duplicate.push_program(&p, &mut || Ok(())).is_err());
    assert!(duplicate.finish(&mut || Ok(())).is_err());
    let mut missing = DeviceCostGraphCatalogBuilder::new(state(2), limits()).unwrap();
    missing.push_program(&p, &mut || Ok(())).unwrap();
    assert!(missing.finish(&mut || Ok(())).is_err());
    for bad in [
        DeviceReusableExecutionSegment::new(0, 0, 2, 2).unwrap(),
        p.segments()[0].clone(),
    ] {
        let mut b = DeviceCostGraphCatalogBuilder::new(state(1), limits()).unwrap();
        b.push_program(&p, &mut || Ok(())).unwrap();
        b.push_uploaded_segment(&p.segments()[0], &"e".repeat(64), &[row(0, 1)], &mut || {
            Ok(())
        })
        .unwrap();
        assert!(b
            .push_uploaded_segment(&bad, &"e".repeat(64), &[row(0, 1)], &mut || Ok(()))
            .is_err());
        assert!(b.finish(&mut || Ok(())).is_err());
    }
}

#[test]
fn graph_catalog_rejects_wrong_logical_node_and_budget_or_capacity_truncation() {
    let p = program(ExecutionLaneId::mint().unwrap(), 1);
    let mut bad = DeviceCostGraphCatalogBuilder::new(state(1), limits()).unwrap();
    bad.push_program(&p, &mut || Ok(())).unwrap();
    assert!(bad
        .push_uploaded_segment(&p.segments()[0], &"e".repeat(64), &[row(2, 1)], &mut || Ok(
            ()
        ))
        .is_err());
    assert!(bad.finish(&mut || Ok(())).is_err());
    let mut capacity = DeviceCostGraphCatalogBuilder::new(
        state(1),
        DeviceCostGraphCatalogLimits::new(1, 2, 1).unwrap(),
    )
    .unwrap();
    assert!(capacity.push_program(&p, &mut || Ok(())).is_err());
    let mut interrupted = DeviceCostGraphCatalogBuilder::new(state(1), limits()).unwrap();
    let mut calls = 0;
    assert!(interrupted
        .push_program(&p, &mut || {
            calls += 1;
            if calls == 3 {
                Err(invalid("test budget"))
            } else {
                Ok(())
            }
        })
        .is_err());
    assert_eq!(calls, 3);
    assert!(interrupted.finish(&mut || Ok(())).is_err());
    let mut command_limit = DeviceCostGraphCatalogBuilder::new(
        state(1),
        DeviceCostGraphCatalogLimits::new(1, 3, 1).unwrap(),
    )
    .unwrap();
    command_limit.push_program(&p, &mut || Ok(())).unwrap();
    command_limit
        .push_uploaded_segment(&p.segments()[0], &"e".repeat(64), &[row(0, 1)], &mut || {
            Ok(())
        })
        .unwrap();
    assert!(command_limit
        .push_uploaded_segment(&p.segments()[1], &"e".repeat(64), &[row(2, 1)], &mut || Ok(
            ()
        ))
        .is_err());
    assert!(command_limit.finish(&mut || Ok(())).is_err());
}

#[test]
fn graph_catalog_unconfigured_empty_is_distinct_from_missing_support() {
    let s = DeviceCostGraphStreamState::new(DeviceCostGraphConfiguration::Unconfigured, 0, 0, 0)
        .unwrap();
    let c = DeviceCostGraphCatalogBuilder::new(s, limits())
        .unwrap()
        .finish(&mut || Ok(()))
        .unwrap();
    assert!(c.programs().is_empty());
    assert!(c.stream_state().is_unconfigured_empty());
    assert!(DeviceCostGraphCatalogLimits::new(0, 1, 1).is_err());
}
