use super::*;

#[test]
fn reusable_preparation_preserves_plan_catalog_lifetime() {
    for plan in [
        DeviceReusableExecutionPlan::new(2).unwrap(),
        DeviceReusableExecutionPlan::on_demand(2).unwrap(),
    ] {
        for receipt in [
            DeviceReusableExecutionPreparation::preparing(plan),
            DeviceReusableExecutionPreparation::preparing_with_progress(plan, 1, 0, 1, 1, 0)
                .unwrap(),
            DeviceReusableExecutionPreparation::ready(plan, 1, 0, 1, 1, 0).unwrap(),
        ] {
            assert_eq!(receipt.catalog_lifetime(), plan.catalog_lifetime());
            let wire = serde_json::to_value(receipt).unwrap();
            match plan.catalog_lifetime() {
                ReusableExecutionCatalogLifetime::StartupSealed => {
                    assert!(wire.get("catalog_lifetime").is_none());
                }
                ReusableExecutionCatalogLifetime::OnDemandBounded => {
                    assert_eq!(wire["catalog_lifetime"], "on_demand_bounded");
                }
            }
        }
    }
}

#[test]
fn idle_lane_slot_reclaim_trims_on_demand_resident_catalog_and_advances_epoch() {
    let (harness, lane) = idle_reusable_step_slot_harness();
    harness
        .runtime
        .set_reusable_catalog_lifetime(ReusableExecutionCatalogLifetime::OnDemandBounded);
    harness.runtime.set_reusable_resident_executables(1);
    let epoch_before = lane.reusable_execution_epoch();

    assert!(harness
        .root
        .dynamic_pools
        .try_reclaim_one_idle_lane_slot()
        .unwrap());
    assert_eq!(harness.runtime.reusable_trim_calls(), 1);
    assert_eq!(lane.reusable_execution_epoch(), epoch_before + 1);
    let status = harness.root.maintenance_controller.status().unwrap();
    assert_eq!(
        status.pools()[0]
            .live_occupancy()
            .lane_stable()
            .total()
            .claim_count(),
        0
    );

    // An empty second trim cannot invalidate another generation of programs.
    assert!(lane.trim_reusable_executables_if_quiescent().unwrap());
    assert_eq!(lane.reusable_execution_epoch(), epoch_before + 1);
    drop(lane);
    close_dynamic_test_root(harness.root);
}

#[test]
fn on_demand_trim_cannot_reclaim_externally_retained_addresses() {
    let (harness, lane, external_pin) = idle_reusable_step_slot_harness_with_pin();
    harness
        .runtime
        .set_reusable_catalog_lifetime(ReusableExecutionCatalogLifetime::OnDemandBounded);
    harness.runtime.set_reusable_resident_executables(1);
    let epoch_before = lane.reusable_execution_epoch();
    let status_before = harness.root.maintenance_controller.status().unwrap();

    assert!(!harness
        .root
        .dynamic_pools
        .try_reclaim_one_idle_lane_slot()
        .unwrap());
    assert_eq!(harness.runtime.reusable_trim_calls(), 1);
    assert_eq!(lane.reusable_execution_epoch(), epoch_before + 1);
    assert_eq!(
        harness.root.maintenance_controller.status().unwrap(),
        status_before
    );

    drop(external_pin);
    assert!(harness
        .root
        .dynamic_pools
        .try_reclaim_one_idle_lane_slot()
        .unwrap());
    assert_eq!(lane.reusable_execution_epoch(), epoch_before + 1);
    drop(lane);
    close_dynamic_test_root(harness.root);
}

#[test]
fn failed_on_demand_trim_closes_lane_without_reclaim_or_epoch_change() {
    let (harness, lane) = idle_reusable_step_slot_harness();
    harness
        .runtime
        .set_reusable_catalog_lifetime(ReusableExecutionCatalogLifetime::OnDemandBounded);
    harness.runtime.set_reusable_resident_executables(1);
    harness
        .runtime
        .reusable_trim_fails
        .store(true, Ordering::Release);
    let epoch_before = lane.reusable_execution_epoch();
    let status_before = harness.root.maintenance_controller.status().unwrap();

    assert!(harness
        .root
        .dynamic_pools
        .try_reclaim_one_idle_lane_slot()
        .is_err());
    assert_eq!(harness.runtime.reusable_trim_calls(), 1);
    assert!(lane.is_fail_closed());
    assert_eq!(lane.reusable_execution_epoch(), epoch_before);
    assert_eq!(
        harness.root.maintenance_controller.status().unwrap(),
        status_before
    );
    assert!(lane.trim_reusable_executables_if_quiescent().is_err());
    assert_eq!(harness.runtime.reusable_trim_calls(), 1);

    drop(lane);
    close_dynamic_test_root(harness.root);
}
