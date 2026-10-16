//! Real admission, reserved allocations and fresh frame ownership. No test
//! claims GPU completion merely by preparing or dropping a resident use.
use super::*;
use crate::vnext::{
    CompletionObservation, CompletionReaper, DynamicStorageContract, MemoryPlan,
    OperationCompletionDisposition,
};

#[path = "completion_tests.rs"]
mod completion_tests;

pub(super) fn setup() -> (
    Harness,
    Arc<ExecutionLane<TestRuntime>>,
    ReusableExecutionBucketSpec,
) {
    setup_with_request_backing(false)
}

fn setup_with_request_backing(
    include_request: bool,
) -> (
    Harness,
    Arc<ExecutionLane<TestRuntime>>,
    ReusableExecutionBucketSpec,
) {
    let step = pool_catalog(
        linear_profile(),
        AllocationLifetime::Step,
        '7',
        1,
        256,
        TestDemand::Tokens,
    );
    let sequence = pool_catalog(
        paged_profile(),
        AllocationLifetime::Sequence,
        '8',
        1,
        256,
        TestDemand::Tokens,
    );
    let (mut memory, bucket) = reusable_step_memory_plan(step.pool_id.clone());
    let mut node = PlanNode::resource_test_node(NodeId::new("node/resident-decode").unwrap());
    let step_pool_id = step.pool_id.clone();
    let mut catalogs = vec![step, sequence];
    if include_request {
        catalogs.push(pool_catalog(
            linear_profile(),
            AllocationLifetime::Request,
            '9',
            1,
            256,
            TestDemand::Tokens,
        ));
        let binding = DynamicResourceDescriptor::resource_test_binding(
            ResourceId::new("resource/resident-binding").unwrap(),
            DynamicResourceDemand::fixed(64).unwrap(),
            16,
            node.id().clone(),
            DynamicStorageContract::resource_test_contract(linear_profile(), "a".repeat(64))
                .unwrap(),
            4,
        )
        .unwrap();
        node = PlanNode::resource_test_node_with_binding(
            node.id().clone(),
            binding.base_resource_id().clone(),
        );
        let binding_pool_id = binding.pool_id().clone();
        let descriptors = vec![binding];
        let pools =
            MemoryPlan::derive_dynamic_pools(&descriptors, std::slice::from_ref(&node), 256)
                .unwrap();
        catalogs.push(PoolCatalog {
            pools,
            descriptors,
            pool_id: binding_pool_id.clone(),
            profile: linear_profile(),
        });
        let mut budgets = vec![
            ReusablePoolWorkspaceBudget::new(step_pool_id, 256, 0).unwrap(),
            ReusablePoolWorkspaceBudget::new(binding_pool_id, 0, 64).unwrap(),
        ];
        budgets.sort_by(|left, right| left.pool_id().cmp(right.pool_id()));
        memory = ReusableExecutionMemoryPlan::new(
            1,
            1,
            vec![ResolvedReusableExecutionBucket::new(bucket.clone(), budgets).unwrap()],
        )
        .unwrap();
    }
    let budget = 256 * catalogs.len() as u64;
    let catalog = combine_catalogs(&catalogs);
    let runtime = new_runtime(&catalog, budget);
    runtime
        .physical_binding_tables_enabled
        .store(true, Ordering::Relaxed);
    let harness = harness_with_nodes_and_reusable(
        runtime,
        catalog,
        budget,
        false,
        Arc::from(vec![node]),
        Some(memory),
    );
    for pool in &harness.pool_ids {
        harness
            .root
            .maintenance_controller
            .grow_pool(pool, 256)
            .unwrap();
    }
    let lane = harness.root.create_execution_lane().unwrap();
    (harness, lane, bucket)
}

pub(super) fn request(
    batch: &ExecutionBatchParticipants<TestRuntime>,
    bucket: &ReusableExecutionBucketSpec,
    token: u32,
) -> StepResourceAdmissionRequest {
    StepResourceAdmissionRequest::new(
        batch
            .bind_work_shape(vec![TokenSpanWork::from_token_ids(&[token], 0..1).unwrap()])
            .unwrap(),
        AdmissionFitPolicy::ImmediateOnly,
        AdmissionPressureAction::WaitForRelease,
    )
    .unwrap()
    .with_reusable_execution_bucket(bucket.bucket_id().clone())
}

pub(super) fn ready(
    reservation: &Arc<ResidentDecodeReservation<TestRuntime>>,
    batch: &ExecutionBatchParticipants<TestRuntime>,
    request: &StepResourceAdmissionRequest,
) -> (
    ResidentDecodeUse<TestRuntime>,
    ResidentFrameRetirement<TestRuntime>,
) {
    match reservation.try_begin_use(batch, request).unwrap() {
        ResidentDecodeDecision::Ready { use_, retirement } => (use_, retirement),
        _ => panic!(
            "current exact physical table and fresh participant must issue a real resident use"
        ),
    }
}

fn close_session(session: &Arc<SequenceSession<TestRuntime>>) {
    session.request_cancel().unwrap();
    session.try_abort().unwrap();
}

#[test]
fn resident_backing_lookup_keeps_current_authority_checks_and_scope_fallbacks() {
    let (harness, lane, bucket) = setup_with_request_backing(true);
    let sequence = admitted_sequence_with_ceiling(&harness.root, "resident-lookup", 4);
    let session = sequence.open_session().unwrap();
    let batch = ExecutionBatchParticipants::new(vec![Arc::clone(&session)]).unwrap();
    let req = request(&batch, &bucket, 3);
    let reservation = batch
        .try_reserve_resident_decode(&req, &lane)
        .unwrap()
        .unwrap();
    let (use_, retirement) = ready(&reservation, &batch, &req);
    let node = &harness.root.dynamic_pools.nodes[0].id().clone();
    let pools = &harness.root.dynamic_pools;
    let snapshot = use_.backing_snapshots().next().unwrap();
    let request = &session.resources().request;
    assert!(!use_.step_backing().backing_slices().is_empty());
    assert!(!use_.claimed_backing().backing_slices().is_empty());
    assert!(!snapshot.backing_slices().is_empty());
    assert!(!request.backing_slices().is_empty());
    for authorities in [
        use_.claimed_backing().backing_slices(),
        use_.step_backing().backing_slices(),
        snapshot.backing_slices(),
        request.backing_slices(),
    ] {
        for authority in authorities {
            let actual = use_.backing_view(node, 0, authority.resource_id()).unwrap();
            let expected = pools.view(authority).unwrap();
            assert_eq!(actual.size_bytes(), expected.size_bytes());
            assert_eq!(actual.capacity_size_bytes(), expected.capacity_size_bytes());
            assert_eq!(actual.usage(), expected.usage());
            assert_eq!(actual.element_type(), expected.element_type());
            assert_eq!(
                actual.segment_bindings().len(),
                expected.segment_bindings().len()
            );
            for (actual, expected) in actual
                .segment_bindings()
                .iter()
                .zip(expected.segment_bindings())
            {
                assert_eq!(actual.segment(), expected.segment());
                assert!(actual.retention_ref().same_owners(expected.retention_ref()));
            }
            drop(actual);
            drop(expected);

            // A lookup hit is not a cached authorization: failures introduced
            // after this frame was issued must still be detected on every view.
            let pool = &pools.pools[authority.evidence().pool_id()];
            pool.state.lock().unwrap().poisoned = true;
            assert!(use_.backing_view(node, 0, authority.resource_id()).is_err());
            pool.state.lock().unwrap().poisoned = false;
            let ordinal = authority.evidence().segments()[0].chunk_ordinal();
            let chunk = pool.state.lock().unwrap().chunks.remove(&ordinal).unwrap();
            assert!(use_.backing_view(node, 0, authority.resource_id()).is_err());
            pool.state.lock().unwrap().chunks.insert(ordinal, chunk);
            assert!(use_.backing_view(node, 0, authority.resource_id()).is_ok());
        }
    }
    let shared = use_.step_backing().backing_slices()[0].resource_id();
    assert!(use_.backing_view(node, 1, shared).is_err());
    assert!(use_
        .backing_view(&NodeId::new("node/unknown").unwrap(), 0, shared)
        .is_err());
    assert!(use_
        .backing_view(node, 0, &ResourceId::new("resource/unknown").unwrap())
        .is_err());
    drop(use_);
    retirement.try_rollback_unsubmitted().unwrap();
    drop(reservation);
    close_session(&session);
    drop(batch);
    drop(session);
    drop(sequence);
    drop(lane);
    close_dynamic_test_root(harness.root);
}

#[test]
fn resident_decode_bootstrap_and_hot_uses_share_allocations_but_not_authority() {
    let (harness, lane, bucket) = setup();
    let sequence = admitted_sequence_with_ceiling(&harness.root, "resident-fresh", 4);
    let session = sequence.open_session().unwrap();
    let batch = ExecutionBatchParticipants::new(vec![Arc::clone(&session)]).unwrap();
    let first = request(&batch, &bucket, 3);
    let reservation = batch
        .try_reserve_resident_decode(&first, &lane)
        .unwrap()
        .unwrap();
    assert!(reservation.is_idle());
    let (step, wave) = match reservation.prepare_bootstrap(&first).unwrap() {
        ResidentBootstrapDecision::Prepared { step, wave } => (step, wave),
        _ => panic!("cold bootstrap must use its reserved slots"),
    };
    assert!(reservation.matches_bootstrap(&wave).unwrap());
    let slot = step.claimed_backing().lane_stable_slot_identity();
    assert!(slot.is_some());
    let weak_step = Arc::downgrade(&step);
    let reaper = CompletionReaper::new();
    let completion = completed_boundary_tests::submit_fixture_wave_through_reaper(
        &harness.root,
        std::slice::from_ref(&session),
        &lane,
        wave,
        &reaper,
    );
    assert!(
        matches!(completion.wait().unwrap(), CompletionObservation::Terminal(receipt)
        if receipt.disposition() == &OperationCompletionDisposition::Succeeded)
    );
    drop(completion);
    assert_eq!(reaper.retained_count(), 0);
    drop(reaper);
    step.try_retire_normal().unwrap();
    assert!(
        weak_step.upgrade().is_none(),
        "hot use must not keep the old Step alive"
    );
    let mut prior_step = None;
    let mut prior_fingerprint = None;
    for token in [3, 9] {
        let req = request(&batch, &bucket, token);
        let (use_, retirement) = ready(&reservation, &batch, &req);
        assert_eq!(use_.work_shape().as_ref(), req.work_shape());
        assert_eq!(use_.lane_stable_slot_identity(), slot);
        assert!(use_.belongs_to_reservation(&reservation));
        assert_eq!(use_.sequence_physical_tables().len(), 1);
        assert!(reservation.try_begin_use(&batch, &req).is_err());
        // Unsubmitted rollback may reuse the frame ordinal, but the fresh
        // Step/attempt authority must never be reused.
        let step = use_.batch_step_id();
        if let Some(old) = prior_step {
            assert_ne!(step, old);
        }
        if let Some(old) = prior_fingerprint {
            assert_ne!(use_.fingerprint(), old);
        }
        prior_step = Some(step);
        prior_fingerprint = Some(use_.fingerprint().to_owned());
        let fresh_owner = use_.retention();
        drop(use_);
        let failure = retirement.try_rollback_unsubmitted().unwrap_err();
        assert!(
            !reservation.is_idle(),
            "retained commands/readbacks still own the fresh frame"
        );
        drop(fresh_owner);
        failure
            .into_retirement()
            .try_rollback_unsubmitted()
            .unwrap();
        assert!(reservation.is_idle());
    }
    drop(reservation);
    close_session(&session);
    drop(batch);
    drop(session);
    drop(sequence);
    drop(lane);
    close_dynamic_test_root(harness.root);
}

#[test]
fn resident_decode_retry_keeps_program_but_clears_attempt_payload() {
    let (harness, lane, bucket) = setup();
    let sequence = admitted_sequence_with_ceiling(&harness.root, "resident-retry", 4);
    let session = sequence.open_session().unwrap();
    let batch = ExecutionBatchParticipants::new(vec![Arc::clone(&session)]).unwrap();
    let req = request(&batch, &bucket, 5);
    let reservation = batch
        .try_reserve_resident_decode(&req, &lane)
        .unwrap()
        .unwrap();
    let (mut use_, retirement) = ready(&reservation, &batch, &req);
    let program = Arc::new(7_u32);
    use_.retain_resident_program(DeviceBufferRetention::plan(Arc::clone(&program)))
        .unwrap();
    use_.retain_resident_program(DeviceBufferRetention::plan(Arc::clone(&program)))
        .unwrap();
    assert!(use_
        .retain_resident_program(DeviceBufferRetention::plan(Arc::new(8_u32)))
        .is_err());
    // The packet owner is a stand-in for an allocation; no bytes or completion
    // are fabricated. Definitely-not-submitted is the only retry reset seam.
    let packet = Arc::new(9_u32);
    let weak_packet = Arc::downgrade(&packet);
    use_.install_terminal_readback_payload(packet);
    let first_attempt = use_.batch_invocation_id();
    use_.begin_dispatch().unwrap();
    use_ = use_.definitely_not_submitted().unwrap();
    assert_ne!(first_attempt, use_.batch_invocation_id());
    assert!(weak_packet.upgrade().is_none());
    assert!(!use_.has_prepared_terminal_readbacks());
    use_.retain_resident_program(DeviceBufferRetention::plan(Arc::clone(&program)))
        .unwrap();
    drop(use_);
    retirement.try_rollback_unsubmitted().unwrap();
    assert_eq!(Arc::strong_count(&program), 1);
    assert!(reservation.is_idle());
    drop(reservation);
    close_session(&session);
    drop(batch);
    drop(session);
    drop(sequence);
    drop(lane);
    close_dynamic_test_root(harness.root);
}

#[test]
fn resident_decode_append_refresh_keeps_slots_and_rejects_stale_authority() {
    let (harness, lane, bucket) = setup();
    let sequence = admitted_sequence_with_ceiling(&harness.root, "resident-current", 4);
    let session = sequence.open_session().unwrap();
    let batch = ExecutionBatchParticipants::new(vec![Arc::clone(&session)]).unwrap();
    let req = request(&batch, &bucket, 5);
    let reservation = batch
        .try_reserve_resident_decode(&req, &lane)
        .unwrap()
        .unwrap();
    let foreign_sequence = admitted_sequence_with_ceiling(&harness.root, "resident-foreign", 4);
    let foreign_session = foreign_sequence.open_session().unwrap();
    let foreign_batch =
        ExecutionBatchParticipants::new(vec![Arc::clone(&foreign_session)]).unwrap();
    let foreign_request = request(&foreign_batch, &bucket, 5);
    assert!(matches!(
        reservation
            .try_begin_use(&foreign_batch, &foreign_request)
            .unwrap(),
        ResidentDecodeDecision::Fallback
    ));
    assert!(reservation.is_idle());
    close_session(&foreign_session);
    drop(foreign_batch);
    drop(foreign_session);
    drop(foreign_sequence);

    let (first, first_retirement) = ready(&reservation, &batch, &req);
    let old_snapshot = Arc::clone(first.backing_snapshots().next().unwrap());
    let sequence_resource = old_snapshot.backing_slices()[0].resource_id().clone();
    let node = NodeId::new("node/resident-decode").unwrap();
    assert_eq!(
        first
            .backing_view(&node, 0, &sequence_resource)
            .unwrap()
            .size_bytes(),
        64
    );
    let old_table = Arc::clone(&first.sequence_physical_tables()[0]);
    let old_table_weak = Arc::downgrade(&old_table);
    let old_backend = Arc::clone(old_table.backend_bindings().unwrap());
    let old_backend_weak = Arc::downgrade(&old_backend);
    // This CPU fixture has a real Step arena; actual Invocation binding slots
    // are covered by the complete CUDA fixture, not by comparing absent slots.
    let step_slot = first.lane_stable_slot_identity();
    assert!(step_slot.is_some());
    let first_step = first.batch_step_id();
    let first_attempt = first.batch_invocation_id();
    assert_eq!(
        old_table
            .rows()
            .next()
            .unwrap()
            .descriptor()
            .view_size_bytes(),
        64
    );
    drop(first);
    first_retirement.try_rollback_unsubmitted().unwrap();

    let snapshot = match session
        .try_ensure_backing_covers(
            SequenceResourceExtensionRequest::new(work(2), AdmissionPressureAction::WaitForRelease)
                .unwrap(),
        )
        .unwrap()
    {
        SequenceResourceExtensionDecision::Extended(snapshot) => snapshot,
        _ => panic!("real larger Sequence physical allocation must be admitted"),
    };
    assert_ne!(snapshot.generation(), old_snapshot.generation());
    let fresh_request = request(&batch, &bucket, 9);
    let (fresh, retirement) = ready(&reservation, &batch, &fresh_request);
    assert_eq!(
        fresh
            .backing_view(&node, 0, &sequence_resource)
            .unwrap()
            .size_bytes(),
        128
    );
    let current_table = Arc::clone(&fresh.sequence_physical_tables()[0]);
    assert!(!Arc::ptr_eq(&old_table, &current_table));
    assert!(current_table.is_append_of(&old_table));
    assert!(!old_table.is_append_of(&current_table));
    assert_eq!(
        current_table
            .rows()
            .next()
            .unwrap()
            .descriptor()
            .view_size_bytes(),
        128
    );
    assert_eq!(
        old_table
            .rows()
            .next()
            .unwrap()
            .descriptor()
            .view_size_bytes(),
        64
    );
    assert!(Arc::ptr_eq(
        fresh.backing_snapshots().next().unwrap(),
        &snapshot
    ));
    assert_eq!(fresh.lane_stable_slot_identity(), step_slot);
    assert_ne!(fresh.batch_step_id(), first_step);
    assert_ne!(fresh.batch_invocation_id(), first_attempt);
    assert_eq!(fresh.work_shape().as_ref(), fresh_request.work_shape());
    drop(fresh);
    retirement.try_rollback_unsubmitted().unwrap();
    assert!(reservation.is_idle());

    // A deliberately stale real snapshot tests the forbidden reverse transition.
    // It is restored before releasing any backing owner; no fabricated table is used.
    session.resources().lock_backing_state().unwrap().current = Arc::clone(&old_snapshot);
    assert!(matches!(
        reservation.try_begin_use(&batch, &fresh_request).unwrap(),
        ResidentDecodeDecision::Fallback
    ));
    assert!(
        reservation.is_idle(),
        "shrink must roll back the unsubmitted frame"
    );
    session.resources().lock_backing_state().unwrap().current = Arc::clone(&snapshot);

    let sequence_pool_id = snapshot.backing_slices()[0].evidence().pool_id();
    let pool = &harness.root.dynamic_pools.pools[sequence_pool_id];
    pool.state.lock().unwrap().poisoned = true;
    assert!(reservation.try_begin_use(&batch, &fresh_request).is_err());
    assert!(reservation.is_idle());
    pool.state.lock().unwrap().poisoned = false;
    let (again, retirement) = ready(&reservation, &batch, &fresh_request);
    assert!(Arc::ptr_eq(
        &again.sequence_physical_tables()[0],
        &current_table
    ));
    assert_eq!(again.lane_stable_slot_identity(), step_slot);
    drop(again);
    retirement.try_rollback_unsubmitted().unwrap();

    drop(old_table);
    drop(old_snapshot);
    assert!(
        old_table_weak.upgrade().is_none(),
        "old backend payload must not retain a catalog cycle"
    );
    assert!(old_backend_weak.upgrade().is_some());
    session.request_cancel().unwrap();
    assert!(reservation.try_begin_use(&batch, &fresh_request).is_err());
    assert!(reservation.is_idle());
    drop(reservation);
    drop(snapshot);
    drop(current_table);
    session.try_abort().unwrap();
    drop(batch);
    drop(session);
    drop(sequence);
    assert_eq!(
        harness
            .root
            .dynamic_pools
            .logical_admission
            .snapshot()
            .unwrap()
            .active_sequences(),
        1,
        "old immutable backend version independently retains its physical/logical owner"
    );
    drop(old_backend);
    assert!(old_backend_weak.upgrade().is_none());
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
    drop(lane);
    close_dynamic_test_root(harness.root);
}

#[test]
fn resident_decode_append_refresh_rejects_same_identity_replacement_and_missing_chunk() {
    let (harness, lane, bucket) = setup();
    let (donor, donor_lane, _) = setup();
    let sequence = admitted_sequence_with_ceiling(&harness.root, "resident-replacement", 4);
    let session = sequence.open_session().unwrap();
    let batch = ExecutionBatchParticipants::new(vec![Arc::clone(&session)]).unwrap();
    let req = request(&batch, &bucket, 5);
    let reservation = batch
        .try_reserve_resident_decode(&req, &lane)
        .unwrap()
        .unwrap();
    let (use_, retirement) = ready(&reservation, &batch, &req);
    let snapshot = Arc::clone(use_.backing_snapshots().next().unwrap());
    drop(use_);
    retirement.try_rollback_unsubmitted().unwrap();
    let evidence = snapshot.backing_slices()[0].evidence();
    let pool = &harness.root.dynamic_pools.pools[evidence.pool_id()];
    let donor_pool = &donor.root.dynamic_pools.pools[evidence.pool_id()];
    let ordinal = evidence.segments()[0].chunk_ordinal();
    let original = pool.state.lock().unwrap().chunks.remove(&ordinal).unwrap();
    assert!(matches!(
        reservation.try_begin_use(&batch, &req).unwrap(),
        ResidentDecodeDecision::Fallback
    ));
    assert!(reservation.is_idle());
    let mut replacement = donor_pool
        .state
        .lock()
        .unwrap()
        .chunks
        .remove(&ordinal)
        .unwrap();
    let donor_identity = replacement.backing.identity.clone();
    let donor_descriptor = replacement.backing.descriptor.clone();
    let backing = Arc::get_mut(&mut replacement.backing).unwrap();
    backing.identity = original.backing.identity.clone();
    backing.descriptor = original.backing.descriptor.clone();
    pool.state
        .lock()
        .unwrap()
        .chunks
        .insert(ordinal, replacement);
    assert!(
        matches!(
            reservation.try_begin_use(&batch, &req).unwrap(),
            ResidentDecodeDecision::Fallback
        ),
        "equal identity/descriptor must not authorize a replacement allocation Arc"
    );
    assert!(reservation.is_idle());
    let mut replacement = pool
        .state
        .lock()
        .unwrap()
        .chunks
        .insert(ordinal, original)
        .unwrap();
    let backing = Arc::get_mut(&mut replacement.backing).unwrap();
    backing.identity = donor_identity;
    backing.descriptor = donor_descriptor;
    donor_pool
        .state
        .lock()
        .unwrap()
        .chunks
        .insert(ordinal, replacement);
    let (current, retirement) = ready(&reservation, &batch, &req);
    drop(current);
    retirement.try_rollback_unsubmitted().unwrap();
    drop(reservation);
    drop(snapshot);
    close_session(&session);
    drop(batch);
    drop(session);
    drop(sequence);
    drop(lane);
    drop(donor_lane);
    close_dynamic_test_root(harness.root);
    close_dynamic_test_root(donor.root);
}

#[test]
fn resident_decode_unknown_retention_and_revoke_never_reopen_capacity_early() {
    let (harness, lane, bucket) = setup();
    let sequence = admitted_sequence_with_ceiling(&harness.root, "resident-unknown", 4);
    let session = sequence.open_session().unwrap();
    let batch = ExecutionBatchParticipants::new(vec![Arc::clone(&session)]).unwrap();
    let req = request(&batch, &bucket, 7);
    let reservation = batch
        .try_reserve_resident_decode(&req, &lane)
        .unwrap()
        .unwrap();
    let (mut use_, retirement) = ready(&reservation, &batch, &req);
    use_.begin_dispatch().unwrap();
    use_.mark_submission_indeterminate();
    let quarantine = use_.retention();
    reservation.revoke();
    assert!(reservation.try_begin_use(&batch, &req).is_err());
    let failure = retirement.try_abort().unwrap_err();
    drop(quarantine);
    drop(use_);
    // Unknown device use cannot be declared an unsubmitted rollback. After
    // the explicit simulated drain drops all owners, abort retires the frame.
    let failure = failure
        .into_retirement()
        .try_rollback_unsubmitted()
        .unwrap_err();
    failure.into_retirement().try_abort().unwrap();
    assert!(
        !reservation.is_idle(),
        "late frame retirement cannot undo revocation"
    );
    session.try_abort().unwrap();
    drop(batch);
    drop(session);
    drop(sequence);
    let held = harness
        .root
        .dynamic_pools
        .logical_admission
        .snapshot()
        .unwrap();
    assert_eq!(held.active_sequences(), 1);
    drop(reservation);
    let released = harness
        .root
        .dynamic_pools
        .logical_admission
        .snapshot()
        .unwrap();
    assert_eq!(released.active_sequences(), 0);
    assert_eq!(released.active_requests(), 0);
    drop(lane);
    close_dynamic_test_root(harness.root);
}
