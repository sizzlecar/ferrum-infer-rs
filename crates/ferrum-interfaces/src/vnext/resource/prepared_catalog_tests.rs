//! CPU contract runtime exercising the actual lane creation/capture brackets.
use super::*;
use crate::vnext::*;

fn runtime() -> Arc<TestRuntime> {
    Arc::new(TestRuntime::new(
        DeviceId::new("prepared.catalog.lane").unwrap(),
        1024 * 1024,
        BTreeSet::from([linear_profile()]),
    ))
}

#[test]
fn prepared_catalog_lane_creation_binds_before_empty_capture_and_never_calls_legacy_producer() {
    let runtime = runtime();
    runtime
        .prepared_catalog_enabled
        .store(true, Ordering::Release);
    *runtime.cost_graph_probe.lock().unwrap() = Some(Box::new(|| {
        panic!("prepared capture invoked legacy inventory")
    }));
    let lane = ExecutionLane::create(Arc::clone(&runtime)).unwrap();
    let capture = || {
        lane.try_with_cost_planning_lane(
            ResourcePlanningLimits::default(),
            &mut || true,
            |_, graph, catalog, _| {
                let Some(DeviceCostGraphCatalogSnapshot::Prepared(root)) = catalog else {
                    panic!("bound lane did not expose a complete prepared root");
                };
                assert_eq!(graph, Some(root.catalog().stream_state()));
                assert!(root.matches_runtime_lane(
                    &runtime.descriptor.runtime_implementation_fingerprint,
                    lane.id()
                ));
                assert!(root.catalog().programs().is_empty());
                Ok(root)
            },
        )
        .unwrap()
    };
    let first = capture();
    let second = capture();
    assert!(Arc::ptr_eq(&first, &second));
    assert!(first.same_generation(&second));
    let retained = runtime.catalog_budget.retained_payload_bytes();
    drop(lane);
    assert_eq!(runtime.stream_drops.load(Ordering::Acquire), 1);
    assert_eq!(runtime.catalog_budget.retained_payload_bytes(), retained);
    drop(first);
    drop(second);
    assert_eq!(runtime.catalog_budget.retained_payload_bytes(), 0);
}

#[test]
fn prepared_catalog_raw_stream_is_unprepared_and_foreign_rebind_revokes_current_root() {
    let runtime = runtime();
    runtime
        .prepared_catalog_enabled
        .store(true, Ordering::Release);
    let mut stream = runtime.create_stream().unwrap();
    assert!(matches!(
        runtime
            .cost_prepared_reusable_graph_catalog(&stream, &mut || Ok(()))
            .unwrap(),
        DevicePreparedCostGraphCatalogAvailability::Unprepared
    ));
    let lane = ExecutionLaneId::mint().unwrap();
    runtime
        .bind_cost_graph_catalog_lane(&mut stream, lane)
        .unwrap();
    let root = Arc::clone(stream.prepared_catalog.as_ref().unwrap());
    runtime
        .bind_cost_graph_catalog_lane(&mut stream, lane)
        .unwrap();
    assert!(Arc::ptr_eq(
        &root,
        stream.prepared_catalog.as_ref().unwrap()
    ));
    let foreign = ExecutionLaneId::mint().unwrap();
    assert!(runtime
        .bind_cost_graph_catalog_lane(&mut stream, foreign)
        .is_err());
    assert!(!root.is_current(
        stream.catalog_source.as_ref().unwrap(),
        root.catalog().stream_state()
    ));
    assert!(matches!(
        runtime
            .cost_prepared_reusable_graph_catalog(&stream, &mut || Ok(()))
            .unwrap(),
        DevicePreparedCostGraphCatalogAvailability::Unprepared
    ));
    assert_eq!(stream.catalog_source.as_ref().unwrap().lane_id(), lane);
}

#[test]
fn prepared_catalog_bind_failure_drops_the_new_stream_and_legacy_runtime_stays_compatible() {
    let runtime = runtime();
    runtime
        .prepared_catalog_enabled
        .store(true, Ordering::Release);
    runtime
        .prepared_catalog_bind_fails
        .store(true, Ordering::Release);
    assert!(matches!(
        ExecutionLane::create(Arc::clone(&runtime)),
        Err(ExecutionLaneCreationError::Device(_))
    ));
    assert_eq!(runtime.stream_drops.load(Ordering::Acquire), 1);
    assert_eq!(runtime.catalog_budget.retained_payload_bytes(), 0);
    runtime
        .prepared_catalog_enabled
        .store(false, Ordering::Release);
    let lane = ExecutionLane::create(Arc::clone(&runtime)).unwrap();
    lane.try_with_cost_planning_lane(
        ResourcePlanningLimits::default(),
        &mut || true,
        |_, graph, catalog, _| {
            assert!(graph.is_none());
            assert!(catalog.is_none());
            Ok(())
        },
    )
    .unwrap();
}
