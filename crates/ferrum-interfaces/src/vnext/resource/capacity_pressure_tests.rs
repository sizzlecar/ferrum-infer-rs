use super::restore_initialization_tests::{checkpoint_fixture, RestoreHarness};
use super::*;
use crate::vnext::{
    CapacityClaimDecision, CheckpointCapacityClaimDecision, CheckpointCapacityPolicy,
};

const CHECKPOINT_PAGE_BYTES: u64 = 65536;
const CHECKPOINT_EXTENDED_TOKENS: usize = 5;

fn checkpoint_harness(initial_sequences: u32) -> RestoreHarness {
    let profile = DynamicStorageProfile::new(
        DynamicStorageAllocator::FixedBlockArena {
            block_bytes: CHECKPOINT_PAGE_BYTES,
        },
        DynamicStorageView::PagedRegions {
            block_bytes: CHECKPOINT_PAGE_BYTES,
        },
    )
    .unwrap();
    let mut spec = checkpoint_fixture::Spec {
        device_id: Some(
            DeviceId::new(format!(
                "device.foreground-pressure-{}",
                NEXT_TEST_DEVICE.fetch_add(1, Ordering::Relaxed)
            ))
            .unwrap(),
        ),
        checkpoint_capacity: Some(
            CheckpointCapacityPolicy::new(8 * CHECKPOINT_PAGE_BYTES).unwrap(),
        ),
        profile,
        port_profile: profile,
        ..Default::default()
    };
    spec.states.truncate(1);
    spec.locations.truncate(1);
    spec.layouts.truncate(1);
    // Use the fixture's declared 64 KiB paged ABI, as in the restore test.
    // Four token positions share the initial page; position five needs one more.
    spec.states[0].tensor.dimensions = vec![16384];
    spec.states[0].capacity_demand = crate::vnext::StateCapacityDemand::TokenScaled {
        bytes_per_token: 16384,
        maximum_tokens: 64,
    };
    let harness = RestoreHarness::with_initial_sequences(spec, Some(initial_sequences));
    let _ = harness.fixture.layout(); // Diagnose unsupported declarations explicitly.
    harness
}

fn checkpoint(h: &RestoreHarness) -> Arc<CheckpointBackingOwner<TestRuntime>> {
    let requests = h
        .fixture
        .plan
        .checkpoint_byte_plan(1)
        .unwrap()
        .backing_requests()
        .unwrap();
    match h
        .root
        .trusted_runtime_binding()
        .unwrap()
        .try_allocate_checkpoint_backing(&requests)
        .unwrap()
    {
        CheckpointBackingAllocationDecision::Allocated(owner) => owner,
        _ => panic!("resident checkpoint must fit"),
    }
}

fn extend(
    session: &Arc<SequenceSession<TestRuntime>>,
    tokens: usize,
) -> SequenceResourceExtensionDecision<TestRuntime> {
    session
        .try_ensure_backing_covers(
            SequenceResourceExtensionRequest::new(
                work(tokens),
                AdmissionPressureAction::WaitForRelease,
            )
            .unwrap(),
        )
        .unwrap()
}

fn extension_pressure(
    session: &Arc<SequenceSession<TestRuntime>>,
    tokens: usize,
) -> AdmissionDeferred {
    // Exercise the logical transaction with the same trusted delta and parent
    // lease used by sequence extension. The public extension prepares physical
    // slices first, so a physically full pool instead returns BackingDeferred.
    let resources = session.resources();
    let plan = &resources.request_resources().plan;
    let committed = resources.backing_snapshot().unwrap();
    let (demand, _) = plan
        .sequence_extension_demand(
            committed.committed_shape(),
            work(tokens).fit_shape(),
            AdmissionPressureAction::WaitForRelease,
        )
        .unwrap();
    match plan
        .logical_admission()
        .try_claim_for_sequence(resources.logical_lease(), &demand)
        .unwrap()
    {
        CapacityClaimDecision::Deferred(deferred) => deferred,
        CapacityClaimDecision::Claimed(_) => panic!("occupied state unexpectedly admitted delta"),
        CapacityClaimDecision::PermanentRejected(rejected) => {
            panic!("valid extension was permanently rejected: {rejected:?}")
        }
    }
}

#[test]
fn foreground_pressure_grows_beside_a_retained_checkpoint_without_evicting() {
    let h = checkpoint_harness(2);
    let owner = checkpoint(&h);
    let before = h.root.dynamic_pool_status().unwrap();
    let binding = h.root.trusted_runtime_binding().unwrap();
    let ledger = binding.logical_admission().snapshot().unwrap();
    let retained = binding
        .logical_admission()
        .checkpoint_retained_bytes()
        .unwrap();
    let deferred = extension_pressure(&h.session, CHECKPOINT_EXTENDED_TOKENS);
    assert_eq!(deferred.action(), DeferredAction::WaitForRelease);
    assert!(deferred
        .blockers()
        .iter()
        .any(|b| b.requested().get() <= b.current_total().get()));
    let receipt = h
        .root
        .try_maintain_for_capacity_pressure(&deferred)
        .unwrap()
        .unwrap();
    assert_eq!(
        receipt
            .growths()
            .iter()
            .map(|g| g.chunk_bytes())
            .sum::<u64>(),
        CHECKPOINT_PAGE_BYTES
    );
    assert!(receipt.rebalance().is_none());
    assert!(receipt.maintenance_boundary().is_none());
    let after = h.root.dynamic_pool_status().unwrap();
    assert_eq!(
        after.budget_claimed_bytes(),
        before.budget_claimed_bytes() + CHECKPOINT_PAGE_BYTES
    );
    assert_eq!(
        binding
            .logical_admission()
            .checkpoint_retained_bytes()
            .unwrap(),
        retained
    );
    assert_eq!(
        binding
            .logical_admission()
            .snapshot()
            .unwrap()
            .active_sequences(),
        ledger.active_sequences()
    );
    assert_eq!(
        after.maximum_active_sequences(),
        before.maximum_active_sequences()
    );
    // The same unchanged demand now has enough free capacity: no second grow.
    assert!(h
        .root
        .try_maintain_for_capacity_pressure(&deferred)
        .unwrap()
        .unwrap()
        .growths()
        .is_empty());
    assert!(matches!(
        extend(&h.session, CHECKPOINT_EXTENDED_TOKENS),
        SequenceResourceExtensionDecision::Extended(_)
    ));
    assert_eq!(owner.extent_bytes(), retained);
    drop((owner, binding));
    h.close();
}

#[test]
fn foreground_pressure_rechecks_release_instead_of_growing_a_stale_shortfall() {
    let h = checkpoint_harness(2);
    let owner = checkpoint(&h);
    let deferred = extension_pressure(&h.session, CHECKPOINT_EXTENDED_TOKENS);
    drop(owner);
    let before = h.root.dynamic_pool_status().unwrap();
    let allocations = h.runtime.allocate_calls();
    for _ in 0..2 {
        assert!(h
            .root
            .try_maintain_for_capacity_pressure(&deferred)
            .unwrap()
            .unwrap()
            .growths()
            .is_empty());
    }
    assert_eq!(h.runtime.allocate_calls(), allocations);
    assert_eq!(h.root.dynamic_pool_status().unwrap(), before);
    assert!(matches!(
        extend(&h.session, CHECKPOINT_EXTENDED_TOKENS),
        SequenceResourceExtensionDecision::Extended(_)
    ));
    h.close();
}

#[test]
fn foreground_pressure_checks_logical_leases_before_physical_prepare() {
    let h = checkpoint_harness(2);
    let binding = h.root.trusted_runtime_binding().unwrap();
    let requests = h
        .fixture
        .plan
        .checkpoint_byte_plan(1)
        .unwrap()
        .backing_requests()
        .unwrap();
    let occupied = checkpoint(&h);
    let CheckpointBackingAllocationDecision::Deferred(checkpoint_demand) =
        binding.try_allocate_checkpoint_backing(&requests).unwrap()
    else {
        panic!("the real byte plan must report its occupied-pool demand")
    };
    drop(occupied);
    let demand = AdmissionDemand::from_plan(
        checkpoint_demand.immediate_requested().clone(),
        checkpoint_demand.fit_requested().clone(),
        AdmissionFitPolicy::ImmediateOnly,
        AdmissionPressureAction::WaitForRelease,
    )
    .unwrap();
    let fee = demand
        .immediate_claim()
        .entries()
        .iter()
        .map(|entry| entry.units().get())
        .sum();
    // Pause the real checkpoint transaction after its logical fee, before its
    // physical prepare. The still-free physical bytes are already spoken for.
    let CheckpointCapacityClaimDecision::Claimed(lease) = binding
        .logical_admission()
        .try_claim_checkpoint(&demand, fee)
        .unwrap()
    else {
        panic!("logical checkpoint claim must fit")
    };
    let before = h.root.dynamic_pool_status().unwrap();
    let state = before
        .pools()
        .iter()
        .find(|p| p.contract().minimum_sequence_bytes() > 0)
        .unwrap();
    assert_eq!(state.free_bytes(), CHECKPOINT_PAGE_BYTES);
    let deferred = extension_pressure(&h.session, CHECKPOINT_EXTENDED_TOKENS);
    let receipt = h
        .root
        .try_maintain_for_capacity_pressure(&deferred)
        .unwrap()
        .unwrap();
    assert_eq!(
        receipt
            .growths()
            .iter()
            .map(|g| g.chunk_bytes())
            .sum::<u64>(),
        CHECKPOINT_PAGE_BYTES
    );
    assert_eq!(
        binding
            .logical_admission()
            .checkpoint_retained_bytes()
            .unwrap(),
        CHECKPOINT_PAGE_BYTES
    );
    assert!(matches!(
        extend(&h.session, CHECKPOINT_EXTENDED_TOKENS),
        SequenceResourceExtensionDecision::Extended(_)
    ));
    drop((lease, binding));
    h.close();
}

#[test]
fn foreground_pressure_budget_failure_preserves_every_pool_and_retained_owner() {
    let h = checkpoint_harness(2);
    let owner = checkpoint(&h);
    let deferred = extension_pressure(&h.session, CHECKPOINT_EXTENDED_TOKENS);
    let status = h.root.dynamic_pool_status().unwrap();
    let remaining =
        status.budget_device_wide_usable_ceiling_bytes() - status.budget_claimed_bytes();
    let reservation =
        DeviceCapacityReservation::reserve(&h.root.dynamic_pools.budget, remaining).unwrap();
    let before = h.root.dynamic_pool_status().unwrap();
    let allocations = h.runtime.allocate_calls();
    assert!(matches!(
        h.root.try_maintain_for_capacity_pressure(&deferred),
        Err(VNextError::DeviceCapacityUnavailable(_))
    ));
    assert_eq!(h.runtime.allocate_calls(), allocations);
    assert_eq!(h.root.dynamic_pool_status().unwrap(), before);
    assert_eq!(
        h.root
            .trusted_runtime_binding()
            .unwrap()
            .logical_admission()
            .checkpoint_retained_bytes()
            .unwrap(),
        owner.extent_bytes()
    );
    drop(reservation);
    assert!(!h
        .root
        .try_maintain_for_capacity_pressure(&deferred)
        .unwrap()
        .unwrap()
        .growths()
        .is_empty());
    drop(owner);
    h.close();
}

#[test]
fn foreground_pressure_pool_ceiling_is_not_bypassed_by_device_headroom() {
    let catalog = pool_catalog(
        paged_profile(),
        AllocationLifetime::Sequence,
        'a',
        1,
        128,
        TestDemand::Tokens,
    );
    let runtime = new_runtime(&catalog, 512);
    let h = harness(runtime, catalog, 512, false);
    h.root
        .maintenance_controller
        .grow_pool(&h.pool_ids[0], 128)
        .unwrap();
    let first = admitted_sequence_with_ceiling(&h.root, "pressure-ceiling-first", 2);
    let second = admitted_sequence(&h.root, "pressure-ceiling-second");
    let session = first.open_session().unwrap();
    let deferred = extension_pressure(&session, 2);
    let before = h.root.dynamic_pool_status().unwrap();
    let allocations = h.runtime.allocate_calls();
    assert!(matches!(
        h.root.try_maintain_for_capacity_pressure(&deferred),
        Err(VNextError::DynamicPoolResidentUnavailable(_))
    ));
    assert_eq!(h.runtime.allocate_calls(), allocations);
    assert_eq!(h.root.dynamic_pool_status().unwrap(), before);
    session.try_abort_if_quiescent().unwrap();
    drop((session, first, second));
}

#[test]
fn foreground_pressure_does_not_grow_for_active_slot_blockers() {
    let h = checkpoint_harness(3);
    let second = h.new_session("pressure-slot-second");
    let third = h.new_session("pressure-slot-third");
    let request = admitted_request(&h.root, "pressure-slot-waiter");
    let admission = SequenceResourceAdmissionRequest::new(
        work(1),
        AdmissionFitPolicy::ImmediateOnly,
        AdmissionPressureAction::WaitForRelease,
    )
    .unwrap();
    // First observation has both a byte shortfall and the slot blocker.
    // After adding one page explicitly, only the slot blocker remains.
    for byte_pressure in [true, false] {
        let SequenceResourceAdmissionDecision::Deferred(deferred) =
            request.try_admit_sequence(admission.clone()).unwrap()
        else {
            panic!("all three execution slots are in use")
        };
        let before = h.root.dynamic_pool_status().unwrap();
        let allocations = h.runtime.allocate_calls();
        assert!(deferred
            .blockers()
            .iter()
            .any(|b| b.kind() == crate::vnext::CapacityShortfallKind::ActiveSequenceCeiling));
        assert!(h
            .root
            .try_maintain_for_capacity_pressure(&deferred)
            .unwrap()
            .is_none());
        assert_eq!(h.runtime.allocate_calls(), allocations);
        assert_eq!(h.root.dynamic_pool_status().unwrap(), before);
        if byte_pressure {
            let pool = before
                .pools()
                .iter()
                .find(|p| p.contract().minimum_sequence_bytes() > 0)
                .unwrap();
            h.root
                .maintenance_controller
                .grow_pool(pool.pool_id(), CHECKPOINT_PAGE_BYTES)
                .unwrap();
        }
    }
    second.try_abort_if_quiescent().unwrap();
    third.try_abort_if_quiescent().unwrap();
    drop((request, second, third));
    h.close();
}

#[test]
fn foreground_pressure_full_input_fit_grows_available_bytes_without_claiming_fit() {
    let catalog = pool_catalog(
        paged_profile(),
        AllocationLifetime::Sequence,
        'a',
        1,
        512,
        TestDemand::Tokens,
    );
    let runtime = new_runtime(&catalog, 512);
    let h = harness(runtime, catalog, 512, false);
    h.root
        .maintenance_controller
        .grow_pool(&h.pool_ids[0], 128)
        .unwrap();
    let first = admitted_sequence(&h.root, "pressure-fit-existing");
    let binding = h.root.trusted_runtime_binding().unwrap();
    let RequestResourceAdmissionDecision::Admitted(request) = binding
        .try_admit_request(
            RequestResourceAdmissionRequest::new(
                work_with_ceiling(1, 4),
                AdmissionFitPolicy::ImmediateOnly,
                AdmissionPressureAction::WaitForRelease,
            )
            .unwrap(),
            RunId::new("run/pressure-fit-new").unwrap(),
            RequestIdentity::new("request/pressure-fit-new").unwrap(),
        )
        .unwrap()
    else {
        panic!("request with the actual four-token ceiling must be admitted")
    };
    let admission = SequenceResourceAdmissionRequest::new(
        chunked_work(4, 0..1),
        AdmissionFitPolicy::FullInputMustFit,
        AdmissionPressureAction::WaitForRelease,
    )
    .unwrap();
    let SequenceResourceAdmissionDecision::Deferred(deferred) =
        request.try_admit_sequence(admission.clone()).unwrap()
    else {
        panic!("full input must fit alongside the existing sequence")
    };
    let receipt = h
        .root
        .try_maintain_for_capacity_pressure(&deferred)
        .unwrap()
        .unwrap();
    assert_eq!(receipt.growths()[0].chunk_bytes(), 192);
    let snapshot = binding.logical_admission().snapshot().unwrap();
    assert_eq!(snapshot.domains()[0].used().get(), 64);
    assert_eq!(snapshot.domains()[0].available().get(), 256);
    let SequenceResourceAdmissionDecision::Admitted(second) =
        request.try_admit_sequence(admission).unwrap()
    else {
        panic!("fit must now succeed, claiming only the immediate chunk")
    };
    assert_eq!(
        binding.logical_admission().snapshot().unwrap().domains()[0]
            .used()
            .get(),
        128
    );
    drop((second, first, request, binding));
}

#[test]
fn foreground_pressure_rejects_a_foreign_coordinator_without_allocation() {
    let source = checkpoint_harness(2);
    let owner = checkpoint(&source);
    let deferred = extension_pressure(&source.session, CHECKPOINT_EXTENDED_TOKENS);
    let target = checkpoint_harness(1);
    let before = target.root.dynamic_pool_status().unwrap();
    let allocations = target.runtime.allocate_calls();
    assert!(target
        .root
        .try_maintain_for_capacity_pressure(&deferred)
        .is_err());
    assert_eq!(target.runtime.allocate_calls(), allocations);
    assert_eq!(target.root.dynamic_pool_status().unwrap(), before);
    drop(owner);
    target.close();
    source.close();
}
