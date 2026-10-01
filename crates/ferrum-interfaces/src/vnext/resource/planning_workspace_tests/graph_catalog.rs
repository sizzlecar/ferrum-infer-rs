//! Real numeric resource capture with a contract runtime's sealed inventory.
//! This is not evidence of CUDA replay or provider matching.
use super::*;
use crate::vnext::*;
#[path = "graph_catalog/completion_capture.rs"]
mod completion_capture;

fn legacy_catalog(snapshot: &DeviceCostGraphCatalogSnapshot) -> &Arc<DeviceCostGraphCatalog> {
    match snapshot {
        DeviceCostGraphCatalogSnapshot::Legacy(catalog) => catalog,
        DeviceCostGraphCatalogSnapshot::Prepared(_) => panic!("legacy runtime changed protocol"),
    }
}

fn catalog(lane: ExecutionLaneId, tokens: u64) -> DeviceCostGraphCatalog {
    let bucket = ReusableExecutionBucketSpec::new(
        ReusableExecutionClassId::new("planning.catalog").unwrap(),
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
        1,
        1,
        1,
        1,
    )
    .unwrap();
    let capture = DeviceReusableExecutionCapture::new(id, 1, vec![], vec![]).unwrap();
    let segment = DeviceReusableExecutionSegment::new(0, 0, 1, 1).unwrap();
    let p = DeviceReusableExecutionProgram::new(&capture, vec![segment.clone()], vec![], vec![])
        .unwrap();
    let logical = DeviceReplayedLogicalCommandAttribution::new(
        0,
        0,
        DeviceNativeOperationId::new("test.catalog").unwrap(),
        DeviceBatchingForm::Scalar,
        1,
        tokens,
        1,
        0,
        1,
    )
    .unwrap();
    let mut b = DeviceCostGraphCatalogBuilder::new(
        DeviceCostGraphStreamState::new(DeviceCostGraphConfiguration::OnDemand, 1, 1, 0).unwrap(),
        DeviceCostGraphCatalogLimits::new(1, 1, 1).unwrap(),
    )
    .unwrap();
    b.push_program(&p, &mut || Ok(())).unwrap();
    b.push_uploaded_segment(&segment, &"e".repeat(64), &[logical], &mut || Ok(()))
        .unwrap();
    b.finish(&mut || Ok(())).unwrap()
}
fn route(
    root: &Arc<PlanRuntimeResources<TestRuntime>>,
    session: &SequenceSession<TestRuntime>,
    lane: &ExecutionLane<TestRuntime>,
) -> ExecutionCostRouteView {
    let until = std::time::Instant::now() + std::time::Duration::from_secs(5);
    loop {
        match root.execution_cost_route_view(
            &[session],
            &[0],
            lane,
            ResourcePlanningLimits::default(),
            &mut || true,
        ) {
            ExecutionCostRouteAvailability::Known(view) => return view,
            ExecutionCostRouteAvailability::Unknown(ExecutionCostRouteUnknown::Resource(
                ResourcePlanningUnknown::ReadUnavailable(_),
            )) if std::time::Instant::now() < until => std::thread::yield_now(),
            other => panic!("graph route snapshot: {other:?}"),
        }
    }
}

#[test]
fn planning_graph_catalog_capture_holds_lane_and_pins_no_resource_or_executable() {
    let (h, _, lane) = setup(256);
    let sequence = admitted_sequence_with_ceiling(&h.root, "catalog", 4);
    let session = sequence.open_session().unwrap();
    let unsupported = route(&h.root, &session, &lane);
    assert!(unsupported.graph_catalog().is_none());
    assert!(unsupported.graph_stream_state().is_none());
    *h.runtime.cost_graph_catalog.lock().unwrap() = Some(catalog(lane.id(), 1));
    let probe_count = Arc::new(AtomicU64::new(0));
    let count = Arc::clone(&probe_count);
    let weak_lane = Arc::downgrade(&lane);
    *h.runtime.cost_graph_probe.lock().unwrap() = Some(Box::new(move || {
        let lane = weak_lane.upgrade().unwrap();
        assert!(matches!(
            lane.try_with_resource_planning_lane(|_, _| Ok(())),
            Err(ResourcePlanningUnknown::ReadUnavailable(
                ResourcePlanningReadStage::ExecutionLane
            ))
        ));
        count.fetch_add(1, Ordering::Relaxed);
    }));
    let allocations = h.runtime.allocate_calls();
    let epoch = lane.reusable_execution_epoch();
    let first = route(&h.root, &session, &lane);
    let same = route(&h.root, &session, &lane);
    assert!(first.same_live_evidence(&same));
    assert_eq!(first.graph_catalog().unwrap(), &catalog(lane.id(), 1));
    assert_eq!(first.resource_view().lane_id(), Some(lane.id()));
    assert_eq!(epoch, lane.reusable_execution_epoch());
    assert!(probe_count.load(Ordering::Relaxed) >= 2);
    assert_eq!(allocations, h.runtime.allocate_calls());
    *h.runtime.cost_graph_catalog.lock().unwrap() = Some(catalog(lane.id(), 2));
    let changed = route(&h.root, &session, &lane);
    assert_eq!(first.graph_stream_state(), changed.graph_stream_state());
    assert!(first
        .resource_view()
        .same_live_evidence(changed.resource_view()));
    assert!(
        !first.same_live_evidence(&changed),
        "equal counts cannot conceal changed logical rows"
    );
    *h.runtime.cost_graph_probe.lock().unwrap() = None;
    // Core's real quiescent trim advances the lane epoch. Even a subsequent
    // identical numeric inventory cannot authenticate the pre-trim snapshot.
    h.runtime
        .set_reusable_catalog_lifetime(ReusableExecutionCatalogLifetime::OnDemandBounded);
    h.runtime.set_reusable_resident_executables(1);
    assert!(lane.trim_reusable_executables_if_quiescent().unwrap());
    let after_trim = route(&h.root, &session, &lane);
    assert_eq!(changed.graph_catalog(), after_trim.graph_catalog());
    assert!(lane.reusable_execution_epoch() > epoch);
    assert!(!changed.same_live_evidence(&after_trim));
    *h.runtime.cost_graph_catalog.lock().unwrap() = None;
    session.try_abort_if_quiescent().unwrap();
    drop(session);
    drop(sequence);
    drop(lane);
    close_dynamic_test_root(h.root);
    assert_eq!(first.graph_catalog().unwrap().programs().len(), 1);
}

#[test]
fn planning_graph_catalog_rejects_state_lane_and_budget_mismatch() {
    let (h, _, lane) = setup(256);
    let sequence = admitted_sequence_with_ceiling(&h.root, "catalog-reject", 4);
    let session = sequence.open_session().unwrap();
    let capture = |budget: &mut dyn ResourcePlanningBudget| {
        h.root.execution_cost_route_view(
            &[&session],
            &[0],
            &lane,
            ResourcePlanningLimits::default(),
            budget,
        )
    };
    *h.runtime.cost_graph_catalog.lock().unwrap() =
        Some(catalog(ExecutionLaneId::mint().unwrap(), 1));
    assert!(matches!(
        capture(&mut || true),
        ExecutionCostRouteAvailability::Unknown(ExecutionCostRouteUnknown::Resource(
            ResourcePlanningUnknown::StaleIdentity
        ))
    ));
    *h.runtime.cost_graph_catalog.lock().unwrap() = Some(catalog(lane.id(), 1));
    *h.runtime.cost_graph_state_override.lock().unwrap() = Some(
        DeviceCostGraphStreamState::new(DeviceCostGraphConfiguration::OnDemand, 2, 1, 0).unwrap(),
    );
    assert!(matches!(
        capture(&mut || true),
        ExecutionCostRouteAvailability::Unknown(ExecutionCostRouteUnknown::Resource(
            ResourcePlanningUnknown::StaleIdentity
        ))
    ));
    *h.runtime.cost_graph_state_override.lock().unwrap() = None;
    let mut polls = 0;
    assert!(matches!(
        capture(&mut || {
            polls += 1;
            polls < 3
        }),
        ExecutionCostRouteAvailability::Unknown(ExecutionCostRouteUnknown::Resource(
            ResourcePlanningUnknown::BudgetExhausted
        ))
    ));
    assert_eq!(
        polls, 3,
        "runtime budget interruption is not erased as unsupported"
    );
    // Existing resource-only capture never queries the graph catalog.
    *h.runtime.cost_graph_probe.lock().unwrap() =
        Some(Box::new(|| panic!("unexpected graph query")));
    let _ = lane_view(&h.root, &session, &lane);
    *h.runtime.cost_graph_probe.lock().unwrap() = None;
    *h.runtime.cost_graph_catalog.lock().unwrap() = None;
    session.try_abort_if_quiescent().unwrap();
    drop(session);
    drop(sequence);
    drop(lane);
    close_dynamic_test_root(h.root);
}

#[test]
fn planning_shared_graph_catalog_keeps_arc_and_fresh_capture_identity() {
    let (h, _, lane) = setup(256);
    let sequence = admitted_sequence_with_ceiling(&h.root, "shared-catalog", 4);
    let session = sequence.open_session().unwrap();
    let original = Arc::new(catalog(lane.id(), 1));
    *h.runtime.cost_graph_shared_catalog.lock().unwrap() = Some(Arc::clone(&original));
    let allocations = h.runtime.allocate_calls();
    let first = route(&h.root, &session, &lane);
    let second = route(&h.root, &session, &lane);
    assert!(Arc::ptr_eq(
        legacy_catalog(first.graph_catalog.as_ref().unwrap()),
        &original
    ));
    assert!(Arc::ptr_eq(
        legacy_catalog(second.graph_catalog.as_ref().unwrap()),
        &original
    ));
    assert!(first.same_live_evidence(&second));
    assert_eq!(allocations, h.runtime.allocate_calls());

    let changed = Arc::new(catalog(lane.id(), 2));
    *h.runtime.cost_graph_shared_catalog.lock().unwrap() = Some(Arc::clone(&changed));
    let next = route(&h.root, &session, &lane);
    assert!(Arc::ptr_eq(
        legacy_catalog(next.graph_catalog.as_ref().unwrap()),
        &changed
    ));
    assert!(!first.same_live_evidence(&next));
    assert_eq!(first.graph_catalog().unwrap(), original.as_ref());

    *h.runtime.cost_graph_shared_catalog.lock().unwrap() = None;
    session.try_abort_if_quiescent().unwrap();
    drop(session);
    drop(sequence);
    drop(lane);
    close_dynamic_test_root(h.root);
    assert_eq!(first.graph_catalog().unwrap().programs().len(), 1);
}

#[test]
fn planning_shared_graph_catalog_does_not_cache_state_lane_or_budget_permission() {
    let (h, _, lane) = setup(256);
    let valid = Arc::new(catalog(lane.id(), 1));
    *h.runtime.cost_graph_shared_catalog.lock().unwrap() = Some(Arc::clone(&valid));
    let capture = |budget: &mut dyn ResourcePlanningBudget| {
        lane.try_with_cost_planning_lane(
            ResourcePlanningLimits::default(),
            budget,
            |_, _, catalog, _| Ok(catalog),
        )
    };
    assert!(Arc::ptr_eq(
        legacy_catalog(&capture(&mut || true).unwrap().unwrap()),
        &valid
    ));
    *h.runtime.cost_graph_state_override.lock().unwrap() = Some(
        DeviceCostGraphStreamState::new(DeviceCostGraphConfiguration::OnDemand, 2, 1, 0).unwrap(),
    );
    assert!(matches!(
        capture(&mut || true),
        Err(ResourcePlanningUnknown::StaleIdentity)
    ));
    *h.runtime.cost_graph_state_override.lock().unwrap() = None;
    *h.runtime.cost_graph_shared_catalog.lock().unwrap() =
        Some(Arc::new(catalog(ExecutionLaneId::mint().unwrap(), 1)));
    assert!(matches!(
        capture(&mut || true),
        Err(ResourcePlanningUnknown::StaleIdentity)
    ));
    // A cached catalog was built under larger limits; a later capture must
    // apply its own smaller bound even when the backend returns the same Arc.
    let original_program = &valid.programs()[0];
    let declaration = DeviceReusableExecutionCapture::new(
        original_program.program().program_id().clone(),
        2,
        vec![],
        vec![],
    )
    .unwrap();
    let larger_program = DeviceReusableExecutionProgram::new(
        &declaration,
        original_program.program().segments().to_vec(),
        vec![],
        vec![DeviceReusableExecutionProgramGap::new(
            1,
            DeviceReusableExecutionProgramGapReason::MissingComputeCommand,
        )],
    )
    .unwrap();
    let mut builder = DeviceCostGraphCatalogBuilder::new(
        valid.stream_state(),
        DeviceCostGraphCatalogLimits::new(1, 2, 1).unwrap(),
    )
    .unwrap();
    builder
        .push_program(&larger_program, &mut || Ok(()))
        .unwrap();
    let segment = &original_program.uploaded_segments()[0];
    builder
        .push_uploaded_segment(
            segment.segment(),
            segment.reusable_executable_fingerprint(),
            segment.logical_commands(),
            &mut || Ok(()),
        )
        .unwrap();
    *h.runtime.cost_graph_shared_catalog.lock().unwrap() =
        Some(Arc::new(builder.finish(&mut || Ok(())).unwrap()));
    let smaller = ResourcePlanningLimits {
        maximum_free_extents: 1,
        ..ResourcePlanningLimits::default()
    };
    assert!(matches!(
        lane.try_with_cost_planning_lane(smaller, &mut || true, |_, _, _, _| Ok(())),
        Err(ResourcePlanningUnknown::LimitExceeded)
    ));
    *h.runtime.cost_graph_shared_catalog.lock().unwrap() = Some(valid);

    let mut polls = 0;
    assert!(matches!(
        capture(&mut || {
            polls += 1;
            polls < 2
        }),
        Err(ResourcePlanningUnknown::BudgetExhausted)
    ));
    assert_eq!(polls, 2);
    *h.runtime.cost_graph_shared_catalog.lock().unwrap() = None;
    drop(lane);
    close_dynamic_test_root(h.root);
}
