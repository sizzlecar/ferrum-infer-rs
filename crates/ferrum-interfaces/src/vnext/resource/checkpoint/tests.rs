use super::*;
use crate::vnext::CheckpointCapacityClaimDecision;
use std::future::Future;
use std::task::{Context, Poll, Wake, Waker};

fn fixture(profile: DynamicStorageProfile, maximum_bytes: u64) -> (Harness, ResourceId) {
    let catalog = pool_catalog(
        profile,
        AllocationLifetime::Sequence,
        'a',
        1,
        maximum_bytes,
        TestDemand::Fixed,
    );
    let resource_id = catalog.descriptors[0].base_resource_id().clone();
    let runtime = new_runtime(&catalog, maximum_bytes);
    (harness(runtime, catalog, maximum_bytes, false), resource_id)
}

fn requests(
    root: &Arc<PlanRuntimeResources<TestRuntime>>,
    resource: &ResourceId,
    bytes: u64,
) -> CheckpointBackingRequests {
    let binding = root.trusted_runtime_binding().unwrap();
    CheckpointBackingRequests::new(
        binding.plan_hash().clone(),
        vec![CheckpointBackingRequest::new(resource.clone(), bytes).unwrap()],
    )
    .unwrap()
}

fn allocate(
    root: &Arc<PlanRuntimeResources<TestRuntime>>,
    request: &CheckpointBackingRequests,
) -> Arc<CheckpointBackingOwner<TestRuntime>> {
    match root
        .trusted_runtime_binding()
        .unwrap()
        .try_allocate_checkpoint_backing(request)
        .unwrap()
    {
        CheckpointBackingAllocationDecision::Allocated(owner) => owner,
        _ => panic!("resident checkpoint must allocate"),
    }
}

fn initialize(harness: &Harness) {
    harness
        .root
        .maintenance_controller
        .initialize_pools(&harness.pool_ids)
        .unwrap();
}

#[test]
fn checkpoint_compact_extents_charge_alignment_without_charging_the_shared_chunk_twice() {
    let (harness, resource) = fixture(linear_profile(), 64);
    initialize(&harness);
    let request = requests(&harness.root, &resource, 17);
    let first = allocate(&harness.root, &request);
    let second = allocate(&harness.root, &request);
    assert_eq!(first.logical_bytes(), 17);
    assert_eq!(first.extent_bytes(), 32);
    assert_ne!(first.authority(), second.authority());
    let first_evidence = first.backing_evidence().next().unwrap();
    let second_evidence = second.backing_evidence().next().unwrap();
    assert_eq!(
        first_evidence.segments()[0].chunk(),
        second_evidence.segments()[0].chunk()
    );
    assert_ne!(first_evidence.segments(), second_evidence.segments());
    let status = harness.root.dynamic_pool_status().unwrap();
    assert_eq!(status.budget_claimed_bytes(), 64);
    assert_eq!(status.process_claimed_bytes(), 64);
    let occupancy = status.pools()[0].live_occupancy().transient();
    assert_eq!(occupancy.checkpoint().physical_bytes(), 64);
    assert_eq!(occupancy.checkpoint().claim_count(), 2);
    assert_eq!(occupancy.sequence().claim_count(), 0);
    let logical = harness
        .root
        .dynamic_pools
        .logical_admission
        .snapshot()
        .unwrap();
    assert_eq!(logical.active_requests(), 0);
    assert_eq!(logical.active_sequences(), 0);
    assert_eq!(logical.active_checkpoint_claims(), 2);
    assert_eq!(logical.domains()[0].used().get(), 64);
    assert_eq!(first.claims().entries()[0].units().get(), 32);
    let view = first.view(&resource).unwrap();
    assert_eq!(view.size_bytes(), 17);
    assert_eq!(view.capacity_size_bytes(), 32);
    assert_eq!(view.segment_bindings().len(), 1);
    drop(view);
    drop(first);
    let status = harness.root.dynamic_pool_status().unwrap();
    assert_eq!(status.pools()[0].free_bytes(), 32);
    assert_eq!(status.budget_claimed_bytes(), 64);
    drop(second);
    assert_eq!(
        harness.root.dynamic_pool_status().unwrap().pools()[0].free_bytes(),
        64
    );
    close_dynamic_test_root(harness.root);
}

#[test]
fn checkpoint_uses_state_pool_block_quantum_and_rejects_duplicate_base_aliases() {
    let (harness, resource) = fixture(paged_profile(), 64);
    initialize(&harness);
    let request = requests(&harness.root, &resource, 17);
    let owner = allocate(&harness.root, &request);
    assert_eq!(owner.logical_bytes(), 17);
    assert_eq!(owner.extent_bytes(), 64);
    assert_eq!(owner.backing_evidence().count(), 1);
    // Layout must union alias ranges before allocation. Duplicate projections
    // cannot accidentally allocate or charge the same base state twice.
    let hash = harness
        .root
        .trusted_runtime_binding()
        .unwrap()
        .plan_hash()
        .clone();
    assert!(CheckpointBackingRequests::new(
        hash,
        vec![
            CheckpointBackingRequest::new(resource.clone(), 8).unwrap(),
            CheckpointBackingRequest::new(resource, 8).unwrap(),
        ]
    )
    .is_err());
    assert_eq!(
        harness
            .root
            .dynamic_pools
            .logical_admission
            .snapshot()
            .unwrap()
            .active_checkpoint_claims(),
        1
    );
    drop(owner);
    close_dynamic_test_root(harness.root);
}

#[test]
fn checkpoint_can_allocate_with_every_real_sequence_slot_occupied() {
    let (harness, resource) = fixture(linear_profile(), 640);
    initialize(&harness);
    harness
        .root
        .maintenance_controller
        .grow_pool(&harness.pool_ids[0], 576)
        .unwrap();
    let maximum = harness
        .root
        .dynamic_pools
        .logical_admission
        .snapshot()
        .unwrap()
        .maximum_active_sequences();
    let sequences = (0..maximum)
        .map(|index| admitted_sequence(&harness.root, &format!("checkpoint-slot-{index}")))
        .collect::<Vec<_>>();
    let before = harness
        .root
        .dynamic_pools
        .logical_admission
        .snapshot()
        .unwrap();
    let request = requests(&harness.root, &resource, 17);
    let owner = allocate(&harness.root, &request);
    let during = harness
        .root
        .dynamic_pools
        .logical_admission
        .snapshot()
        .unwrap();
    assert_eq!(during.active_sequences(), before.active_sequences());
    assert_eq!(during.active_requests(), before.active_requests());
    assert_eq!(during.active_child_claims(), before.active_child_claims());
    drop(sequences);
    assert_eq!(
        harness
            .root
            .dynamic_pools
            .logical_admission
            .snapshot()
            .unwrap()
            .active_sequences(),
        0
    );
    assert!(owner.view(&resource).is_ok());
    let independent_sequence = admitted_sequence(&harness.root, "checkpoint-independent");
    assert_eq!(owner.extent_bytes(), 32);
    drop(independent_sequence);
    drop(owner);
    close_dynamic_test_root(harness.root);
}

#[test]
fn checkpoint_optional_allocation_defers_without_growing_or_reserving_a_slot() {
    let (harness, resource) = fixture(linear_profile(), 128);
    let request = requests(&harness.root, &resource, 32);
    let allocations = harness.runtime.allocate_calls();
    let binding = harness.root.trusted_runtime_binding().unwrap();
    let deferred = match binding.try_allocate_checkpoint_backing(&request).unwrap() {
        CheckpointBackingAllocationDecision::BackingDeferred(deferred) => deferred,
        _ => panic!("nonresident backing must defer"),
    };
    assert_eq!(deferred.scope(), DynamicBackingClaimScope::Checkpoint);
    assert_eq!(deferred.scope().lifetime(), None);
    assert_eq!(harness.runtime.allocate_calls(), allocations);
    let snapshot = harness
        .root
        .dynamic_pools
        .logical_admission
        .snapshot()
        .unwrap();
    assert_eq!(snapshot.active_checkpoint_claims(), 0);
    assert_eq!(snapshot.active_requests(), 0);
    assert_eq!(snapshot.active_sequences(), 0);
    drop(binding);
    close_dynamic_test_root(harness.root);
}

#[test]
fn checkpoint_logical_rejection_rolls_back_prepared_physical_extents() {
    let (harness, resource) = fixture(linear_profile(), 64);
    initialize(&harness);
    let coordinator = &harness.root.dynamic_pools.logical_admission;
    let domain = harness.root.dynamic_pools.domains[0].domain_id();
    let capacity = CapacityVector::new(vec![
        CapacityEntry::new(domain, CapacityUnits::new(64)).unwrap()
    ])
    .unwrap();
    let demand = AdmissionDemand::from_plan(
        capacity.clone(),
        capacity,
        AdmissionFitPolicy::ImmediateOnly,
        AdmissionPressureAction::WaitForRelease,
    )
    .unwrap();
    // Model a competing prepare/claim/commit transaction with its claim held
    // before physical commit. Our prepared extents must roll back on pressure.
    let competing = match coordinator.try_claim_checkpoint(&demand).unwrap() {
        CheckpointCapacityClaimDecision::Claimed(lease) => lease,
        _ => panic!("logical claim must fit"),
    };
    let request = requests(&harness.root, &resource, 17);
    let binding = harness.root.trusted_runtime_binding().unwrap();
    assert!(matches!(
        binding.try_allocate_checkpoint_backing(&request).unwrap(),
        CheckpointBackingAllocationDecision::Deferred(_)
    ));
    let status = harness.root.dynamic_pool_status().unwrap();
    assert_eq!(status.pools()[0].free_bytes(), 64);
    assert_eq!(status.pools()[0].live_occupancy().total().claim_count(), 0);
    assert_eq!(
        coordinator.snapshot().unwrap().active_checkpoint_claims(),
        1
    );
    drop(competing);
    let owner = allocate(&harness.root, &request);
    drop(owner);
    drop(binding);
    close_dynamic_test_root(harness.root);
}

#[test]
fn checkpoint_rejects_foreign_plan_unknown_resource_and_out_of_bounds_without_side_effects() {
    let (harness, resource) = fixture(linear_profile(), 64);
    initialize(&harness);
    let binding = harness.root.trusted_runtime_binding().unwrap();
    let foreign_hash = serde_json::from_value(json!("2".repeat(64))).unwrap();
    let foreign = CheckpointBackingRequests::new(
        foreign_hash,
        vec![CheckpointBackingRequest::new(resource.clone(), 16).unwrap()],
    )
    .unwrap();
    let unknown = requests(
        &harness.root,
        &ResourceId::new("resource/unknown-checkpoint").unwrap(),
        16,
    );
    let oversized = requests(&harness.root, &resource, 65);
    let before = harness
        .root
        .dynamic_pools
        .logical_admission
        .snapshot()
        .unwrap();
    for request in [foreign, unknown, oversized] {
        assert!(binding.try_allocate_checkpoint_backing(&request).is_err());
        assert_eq!(
            harness
                .root
                .dynamic_pools
                .logical_admission
                .snapshot()
                .unwrap(),
            before
        );
        assert_eq!(
            harness.root.dynamic_pool_status().unwrap().pools()[0].free_bytes(),
            64
        );
    }
    assert!(CheckpointBackingRequest::new(resource, 0).is_err());
    assert!(CheckpointBackingRequests::new(binding.plan_hash().clone(), Vec::new()).is_err());
    drop(binding);
    close_dynamic_test_root(harness.root);
}

#[test]
fn checkpoint_rejects_request_step_and_non_state_sequence_descriptors() {
    for (lifetime, usage) in [
        (AllocationLifetime::Request, "state"),
        (AllocationLifetime::Step, "state"),
        (AllocationLifetime::Sequence, "activations"),
    ] {
        let catalog = pool_catalog_with_options(
            linear_profile(),
            lifetime,
            'b',
            1,
            64,
            TestDemand::Fixed,
            usage,
            false,
            StateInitialization::None,
        );
        let resource = catalog.descriptors[0].base_resource_id().clone();
        let runtime = new_runtime(&catalog, 64);
        let harness = harness(runtime, catalog, 64, false);
        initialize(&harness);
        let request = requests(&harness.root, &resource, 16);
        let binding = harness.root.trusted_runtime_binding().unwrap();
        assert!(binding.try_allocate_checkpoint_backing(&request).is_err());
        assert_eq!(
            harness
                .root
                .dynamic_pools
                .logical_admission
                .snapshot()
                .unwrap()
                .active_checkpoint_claims(),
            0
        );
        assert_eq!(
            harness.root.dynamic_pool_status().unwrap().pools()[0].free_bytes(),
            64
        );
        drop(binding);
        close_dynamic_test_root(harness.root);
    }
}

#[test]
fn checkpoint_segment_retention_pins_all_ownership_through_close_and_last_drop() {
    let (harness, resource) = fixture(linear_profile(), 64);
    initialize(&harness);
    let request = requests(&harness.root, &resource, 17);
    let binding = harness.root.trusted_runtime_binding().unwrap();
    let owner = allocate(&harness.root, &request);
    let authority = owner.authority();
    let view = owner.view(&resource).unwrap();
    let retention = view.segment_bindings()[0].retention();
    assert_eq!(retention.reusable_address_scope(), None);
    drop(view);
    drop(owner);
    let root = match PlanRuntimeResources::close(harness.root) {
        Ok(PlanRuntimeCloseOutcome::Referenced { resources, .. }) => resources,
        _ => panic!("retained checkpoint must keep plan alive"),
    };
    assert!(root.is_closing());
    assert!(binding.try_allocate_checkpoint_backing(&request).is_err());
    assert_eq!(
        authority.coordinator_id(),
        root.dynamic_pools.logical_admission.id()
    );
    let snapshot = root.dynamic_pools.logical_admission.snapshot().unwrap();
    assert_eq!(snapshot.active_checkpoint_claims(), 1);
    assert_eq!(snapshot.domains()[0].used().get(), 32);
    drop(binding);
    drop(retention);
    let snapshot = root.dynamic_pools.logical_admission.snapshot().unwrap();
    assert_eq!(snapshot.active_checkpoint_claims(), 0);
    assert_eq!(snapshot.domains()[0].used().get(), 0);
    assert_eq!(
        root.maintenance_controller.status().unwrap().pools()[0].free_bytes(),
        64
    );
    close_dynamic_test_root(root);
}

struct ReleaseWakeProbe {
    pools: Arc<DynamicPoolSet<TestRuntime>>,
    calls: AtomicU64,
    physical_released: AtomicBool,
}

impl Wake for ReleaseWakeProbe {
    fn wake(self: Arc<Self>) {
        self.wake_by_ref();
    }

    fn wake_by_ref(self: &Arc<Self>) {
        self.calls.fetch_add(1, Ordering::AcqRel);
        let released = self.pools.pools.values().all(|pool| {
            let state = pool.state.lock().unwrap();
            state.live_occupancy.transient().checkpoint().claim_count() == 0
        });
        self.physical_released.store(released, Ordering::Release);
    }
}

#[test]
fn checkpoint_release_returns_physical_extents_before_waking_a_pending_waiter() {
    let (harness, resource) = fixture(linear_profile(), 64);
    initialize(&harness);
    let request = requests(&harness.root, &resource, 64);
    let owner = allocate(&harness.root, &request);
    let pin = Arc::clone(&owner);
    let binding = harness.root.trusted_runtime_binding().unwrap();
    let deferred = match binding.try_allocate_checkpoint_backing(&request).unwrap() {
        CheckpointBackingAllocationDecision::BackingDeferred(deferred) => deferred,
        _ => panic!("full checkpoint pool must defer"),
    };
    let waiter = harness
        .root
        .register_capacity_waiter(deferred.wait_condition())
        .unwrap();
    let probe = Arc::new(ReleaseWakeProbe {
        pools: Arc::clone(&harness.root.dynamic_pools),
        calls: AtomicU64::new(0),
        physical_released: AtomicBool::new(false),
    });
    let waker = Waker::from(Arc::clone(&probe));
    let mut context = Context::from_waker(&waker);
    let mut waiting = Box::pin(waiter.wait_for_change());
    assert!(matches!(waiting.as_mut().poll(&mut context), Poll::Pending));
    drop(owner);
    assert_eq!(probe.calls.load(Ordering::Acquire), 0);
    drop(pin);
    assert!(probe.calls.load(Ordering::Acquire) > 0);
    assert!(probe.physical_released.load(Ordering::Acquire));
    assert!(matches!(
        waiting.as_mut().poll(&mut context),
        Poll::Ready(Ok(_))
    ));
    drop(waiting);
    drop(waker);
    drop(probe);
    drop(binding);
    let replacement = allocate(&harness.root, &request);
    drop(replacement);
    close_dynamic_test_root(harness.root);
}

#[test]
fn checkpoint_unwind_releases_both_ledgers_and_keeps_runtime_alive_until_buffers_drop() {
    let (harness, resource) = fixture(linear_profile(), 64);
    initialize(&harness);
    let request = requests(&harness.root, &resource, 17);
    let dropped_after_backend = harness.runtime.dropped_after_backend_probe();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let owner = allocate(&harness.root, &request);
        let _retention = owner.device_buffer_retention();
        panic!("abandon unpublished checkpoint");
    }));
    assert!(result.is_err());
    assert_eq!(
        harness
            .root
            .dynamic_pools
            .logical_admission
            .snapshot()
            .unwrap()
            .active_checkpoint_claims(),
        0
    );
    assert_eq!(
        harness.root.dynamic_pool_status().unwrap().pools()[0].free_bytes(),
        64
    );
    drop(harness.runtime);
    close_dynamic_test_root(harness.root);
    assert!(!dropped_after_backend.load(Ordering::Acquire));
}
