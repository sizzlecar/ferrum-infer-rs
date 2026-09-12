use super::restore_initialization_tests::{checkpoint_fixture, RestoreHarness};
use super::*;
use crate::vnext::{
    CheckpointCapacityMaintenanceOutcome, CheckpointCapacityMaintenanceSkipReason,
    CheckpointCapacityPolicy,
};

fn isolated_spec() -> checkpoint_fixture::Spec {
    checkpoint_fixture::Spec {
        device_id: Some(
            DeviceId::new(format!(
                "device.checkpoint-maintenance-{}",
                NEXT_TEST_DEVICE.fetch_add(1, Ordering::Relaxed)
            ))
            .unwrap(),
        ),
        ..Default::default()
    }
}

fn fixed_spec(cap: Option<u64>) -> checkpoint_fixture::Spec {
    let mut spec = isolated_spec();
    spec.states.remove(0);
    spec.locations.remove(0);
    spec.layouts.remove(0);
    spec.checkpoint_capacity = cap.map(|bytes| CheckpointCapacityPolicy::new(bytes).unwrap());
    spec
}

fn allocate_checkpoint(harness: &RestoreHarness) -> Arc<CheckpointBackingOwner<TestRuntime>> {
    let requests = harness
        .fixture
        .plan
        .checkpoint_byte_plan(1)
        .unwrap()
        .backing_requests()
        .unwrap();
    match harness
        .root
        .trusted_runtime_binding()
        .unwrap()
        .try_allocate_checkpoint_backing(&requests)
        .unwrap()
    {
        CheckpointBackingAllocationDecision::Allocated(owner) => owner,
        _ => panic!("maintained checkpoint must fit"),
    }
}

fn maintain(harness: &RestoreHarness) -> Result<CheckpointCapacityMaintenanceOutcome, VNextError> {
    let plan = &harness.fixture.plan;
    harness
        .root
        .trusted_runtime_binding()
        .unwrap()
        .prepare_checkpoint_capacity_maintenance(plan, &plan.checkpoint_byte_plan(1).unwrap())?
        .try_maintain()
}

#[test]
fn checkpoint_maintenance_grows_full_three_sequence_pool_without_new_slots_or_fee() {
    let harness = RestoreHarness::with_initial_sequences(fixed_spec(Some(1024)), Some(3));
    let second = harness.new_session("maintenance-second");
    let third = harness.new_session("maintenance-third");
    let before = harness.root.dynamic_pool_status().unwrap();
    assert_eq!(before.maximum_active_sequences(), 3);
    assert!(before
        .pools()
        .iter()
        .filter(|pool| pool.contract().minimum_sequence_bytes() > 0)
        .all(|pool| pool.free_bytes() == 0));
    let binding = harness.root.trusted_runtime_binding().unwrap();
    let requests = harness
        .fixture
        .plan
        .checkpoint_byte_plan(1)
        .unwrap()
        .backing_requests()
        .unwrap();
    match binding.try_allocate_checkpoint_backing(&requests).unwrap() {
        CheckpointBackingAllocationDecision::Deferred(deferred) => {
            assert_eq!(deferred.action(), DeferredAction::WaitForRelease)
        }
        _ => panic!("resident but occupied state is an availability deferral"),
    }
    let ledger_before = binding.logical_admission().snapshot().unwrap();
    let CheckpointCapacityMaintenanceOutcome::Ready(receipt) = maintain(&harness).unwrap() else {
        panic!("available device budget must grow")
    };
    assert!(!receipt.growths().is_empty());
    assert!(receipt.rebalance().is_none());
    let after = harness.root.dynamic_pool_status().unwrap();
    assert_eq!(after.maximum_active_sequences(), 3);
    let ledger_after = binding.logical_admission().snapshot().unwrap();
    assert_eq!(
        ledger_before.active_sequences(),
        ledger_after.active_sequences()
    );
    assert_eq!(
        binding
            .logical_admission()
            .checkpoint_retained_bytes()
            .unwrap(),
        0
    );
    let owner = allocate_checkpoint(&harness);
    assert_eq!(
        binding
            .logical_admission()
            .checkpoint_retained_bytes()
            .unwrap(),
        owner.extent_bytes()
    );
    assert_eq!(
        after.budget_claimed_bytes(),
        harness
            .root
            .dynamic_pool_status()
            .unwrap()
            .budget_claimed_bytes()
    );
    drop(owner);
    assert_eq!(
        binding
            .logical_admission()
            .checkpoint_retained_bytes()
            .unwrap(),
        0
    );
    let allocations = harness.runtime.allocate_calls();
    let CheckpointCapacityMaintenanceOutcome::Ready(noop) = maintain(&harness).unwrap() else {
        panic!("released extents suffice")
    };
    assert!(noop.growths().is_empty());
    assert_eq!(harness.runtime.allocate_calls(), allocations);
    second.try_abort_if_quiescent().unwrap();
    third.try_abort_if_quiescent().unwrap();
    drop((second, third, binding));
    harness.close();
}

#[test]
fn checkpoint_maintenance_disabled_and_undersized_cap_do_not_allocate() {
    for cap in [None, Some(1)] {
        let harness = RestoreHarness::with_initial_sequences(fixed_spec(cap), Some(1));
        let before = harness.root.dynamic_pool_status().unwrap();
        let allocations = harness.runtime.allocate_calls();
        assert!(matches!(
            maintain(&harness).unwrap(),
            CheckpointCapacityMaintenanceOutcome::Skipped(
                CheckpointCapacityMaintenanceSkipReason::Retention(_)
            )
        ));
        assert_eq!(harness.runtime.allocate_calls(), allocations);
        assert_eq!(
            harness
                .root
                .dynamic_pool_status()
                .unwrap()
                .budget_claimed_bytes(),
            before.budget_claimed_bytes()
        );
        harness.close();
    }
}

#[test]
fn checkpoint_maintenance_budget_pressure_skips_without_reclaim_or_wait() {
    let harness = RestoreHarness::with_initial_sequences(fixed_spec(Some(1024)), Some(1));
    let before = harness.root.dynamic_pool_status().unwrap();
    // Real account reservation models other live device owners consuming the
    // same global ceiling; it is not a synthetic allocation-error flag.
    let available =
        before.budget_device_wide_usable_ceiling_bytes() - before.budget_claimed_bytes();
    let reservation =
        DeviceCapacityReservation::reserve(&harness.root.dynamic_pools.budget, available).unwrap();
    let allocations = harness.runtime.allocate_calls();
    assert!(matches!(
        maintain(&harness).unwrap(),
        CheckpointCapacityMaintenanceOutcome::Skipped(
            CheckpointCapacityMaintenanceSkipReason::DeviceCapacity(_)
        )
    ));
    assert_eq!(harness.runtime.allocate_calls(), allocations);
    let after = harness.root.dynamic_pool_status().unwrap();
    assert!(before
        .pools()
        .iter()
        .any(|pool| pool.contract().minimum_sequence_bytes() == 0 && pool.free_bytes() > 0));
    let occupancy = |status: &DynamicPoolMaintenanceStatus| {
        status
            .pools()
            .iter()
            .map(|pool| {
                (
                    pool.pool_id().clone(),
                    pool.resident_bytes(),
                    pool.free_bytes(),
                    pool.pending_growth_bytes(),
                    pool.live_occupancy().clone(),
                )
            })
            .collect::<Vec<_>>()
    };
    assert_eq!(occupancy(&after), occupancy(&before));
    assert_eq!(
        after.budget_claimed_bytes(),
        before.budget_claimed_bytes() + available
    );
    drop(reservation);
    assert!(matches!(
        maintain(&harness).unwrap(),
        CheckpointCapacityMaintenanceOutcome::Ready(_)
    ));
    drop(allocate_checkpoint(&harness));
    harness.close();
}

#[test]
fn checkpoint_maintenance_allocation_failure_rolls_back_budget_and_can_retry() {
    let mut spec = isolated_spec();
    spec.checkpoint_capacity = Some(CheckpointCapacityPolicy::new(1024).unwrap());
    spec.states[1].tensor.element_type = ElementType::F32;
    let harness = RestoreHarness::with_initial_sequences(spec, Some(1));
    let before = harness.root.dynamic_pool_status().unwrap();
    assert!(before.pools().len() > 1);
    harness
        .runtime
        .fail_on_call(harness.runtime.allocate_calls() + 2);
    assert!(maintain(&harness).is_err());
    let after = harness.root.dynamic_pool_status().unwrap();
    assert_eq!(after.budget_claimed_bytes(), before.budget_claimed_bytes());
    assert_eq!(
        after
            .pools()
            .iter()
            .map(|p| p.resident_bytes())
            .collect::<Vec<_>>(),
        before
            .pools()
            .iter()
            .map(|p| p.resident_bytes())
            .collect::<Vec<_>>()
    );
    assert!(after.pools().iter().all(|p| p.pending_growth_bytes() == 0));
    harness.runtime.fail_on_call(0);
    assert!(matches!(
        maintain(&harness).unwrap(),
        CheckpointCapacityMaintenanceOutcome::Ready(_)
    ));
    drop(allocate_checkpoint(&harness));
    harness.close();
}

#[test]
fn checkpoint_maintenance_rejects_foreign_plan_and_drop_does_no_work() {
    let harness = RestoreHarness::with_initial_sequences(fixed_spec(Some(1024)), Some(1));
    let foreign = checkpoint_fixture::Fixture::build(checkpoint_fixture::Spec::default()).unwrap();
    let binding = harness.root.trusted_runtime_binding().unwrap();
    assert!(binding
        .prepare_checkpoint_capacity_maintenance(
            &foreign.plan,
            &foreign.plan.checkpoint_byte_plan(1).unwrap()
        )
        .is_err());
    let allocations = harness.runtime.allocate_calls();
    let owner = binding
        .prepare_checkpoint_capacity_maintenance(
            &harness.fixture.plan,
            &harness.fixture.plan.checkpoint_byte_plan(1).unwrap(),
        )
        .unwrap();
    drop(owner);
    assert_eq!(harness.runtime.allocate_calls(), allocations);
    assert_eq!(
        binding
            .logical_admission()
            .checkpoint_retained_bytes()
            .unwrap(),
        0
    );
    drop(binding);
    harness.close();
}

#[test]
fn checkpoint_maintenance_repairs_contiguous_fragmentation_from_real_byte_plans() {
    let mut spec = isolated_spec();
    spec.states.truncate(1);
    spec.locations.truncate(1);
    spec.layouts.truncate(1);
    spec.checkpoint_capacity = Some(CheckpointCapacityPolicy::new(4096).unwrap());
    // One source plus four compact entries occupies the initial five minima.
    let harness = RestoreHarness::with_initial_sequences(spec, Some(5));
    let first = allocate_checkpoint(&harness);
    let second = allocate_checkpoint(&harness);
    let third = allocate_checkpoint(&harness);
    let fourth = allocate_checkpoint(&harness);
    let small_bytes = first.extent_bytes();
    assert_eq!(
        harness
            .root
            .dynamic_pool_status()
            .unwrap()
            .pools()
            .iter()
            .find(|pool| pool.contract().minimum_sequence_bytes() > 0)
            .unwrap()
            .free_bytes(),
        0
    );
    drop((first, third));
    let boundary = small_bytes / 4 + 1;
    let plan = &harness.fixture.plan;
    let bytes = plan.checkpoint_byte_plan(boundary).unwrap();
    let binding = harness.root.trusted_runtime_binding().unwrap();
    let requests = bytes.backing_requests().unwrap();
    assert!(matches!(
        binding.try_allocate_checkpoint_backing(&requests).unwrap(),
        CheckpointBackingAllocationDecision::BackingDeferred(_)
    ));
    let before = harness.root.dynamic_pool_status().unwrap();
    let CheckpointCapacityMaintenanceOutcome::Ready(receipt) = binding
        .prepare_checkpoint_capacity_maintenance(plan, &bytes)
        .unwrap()
        .try_maintain()
        .unwrap()
    else {
        panic!("packing requires new contiguous extent")
    };
    assert!(!receipt.growths().is_empty());
    assert!(receipt.rebalance().is_none());
    let owner = match binding.try_allocate_checkpoint_backing(&requests).unwrap() {
        CheckpointBackingAllocationDecision::Allocated(owner) => owner,
        _ => panic!("new contiguous chunk must satisfy the exact demand"),
    };
    assert!(
        harness
            .root
            .dynamic_pool_status()
            .unwrap()
            .budget_claimed_bytes()
            > before.budget_claimed_bytes()
    );
    drop((owner, second, fourth, binding));
    harness.close();
}
