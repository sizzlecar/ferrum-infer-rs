use super::*;

fn capacity_harness(profile: DynamicStorageProfile, initial_bytes: u64) -> Harness {
    let catalog = pool_catalog(
        profile,
        AllocationLifetime::Sequence,
        'b',
        1,
        1024,
        TestDemand::Tokens,
    );
    let h = harness(new_runtime(&catalog, 1024), catalog, 1024, false);
    h.root
        .maintenance_controller
        .grow_pool(&h.pool_ids[0], initial_bytes)
        .unwrap();
    h
}

#[test]
fn resident_capacity_uses_owned_backing_and_only_actual_free_growth() {
    let h = capacity_harness(paged_profile(), 128);
    let session = admitted_sequence_with_ceiling(&h.root, "resident-a", 4)
        .open_session()
        .unwrap();
    let cohort = [session];
    let before = h.root.dynamic_pool_status().unwrap();
    let allocations = h.runtime.allocate_calls();
    assert!(!SequenceSession::resident_cohort_persistent_capacity(&cohort).unwrap());
    assert_eq!(h.root.dynamic_pool_status().unwrap(), before);
    assert_eq!(h.runtime.allocate_calls(), allocations);
    h.root
        .maintenance_controller
        .grow_pool(&h.pool_ids[0], 128)
        .unwrap();
    let before = h.root.dynamic_pool_status().unwrap();
    let allocations = h.runtime.allocate_calls();
    assert!(SequenceSession::resident_cohort_persistent_capacity(&cohort).unwrap());
    assert_eq!(h.root.dynamic_pool_status().unwrap(), before);
    assert_eq!(h.runtime.allocate_calls(), allocations);
    // Full-cap bytes are 256, but only 192 free bytes are needed: the live
    // sequence already owns its first 64-byte page.
    assert_eq!(before.pools()[0].free_bytes(), 192);
}

#[test]
fn resident_capacity_aggregates_every_member_and_rejects_partial_or_duplicate_cohort() {
    let h = capacity_harness(paged_profile(), 256);
    let first = admitted_sequence_with_ceiling(&h.root, "resident-first", 4)
        .open_session()
        .unwrap();
    let second = admitted_sequence_with_ceiling(&h.root, "resident-second", 4)
        .open_session()
        .unwrap();
    assert!(!SequenceSession::resident_cohort_persistent_capacity(&[Arc::clone(&first)]).unwrap());
    assert!(!SequenceSession::resident_cohort_persistent_capacity(&[
        Arc::clone(&first),
        Arc::clone(&first)
    ])
    .unwrap());
    let cohort = [first, second];
    assert!(!SequenceSession::resident_cohort_persistent_capacity(&cohort).unwrap());
    h.root
        .maintenance_controller
        .grow_pool(&h.pool_ids[0], 256)
        .unwrap();
    assert!(SequenceSession::resident_cohort_persistent_capacity(&cohort).unwrap());
    cohort[1].request_cancel().unwrap();
    assert!(!SequenceSession::resident_cohort_persistent_capacity(&cohort).unwrap());
}

#[test]
fn resident_capacity_accounts_for_all_pools_and_already_owned_recurrent_request_state() {
    let kv = pool_catalog(
        paged_profile(),
        AllocationLifetime::Sequence,
        'a',
        1,
        512,
        TestDemand::Tokens,
    );
    let second = pool_catalog(
        paged_profile(),
        AllocationLifetime::Sequence,
        'b',
        1,
        512,
        TestDemand::Tokens,
    );
    let recurrent = pool_catalog(
        linear_profile(),
        AllocationLifetime::Sequence,
        'c',
        1,
        128,
        TestDemand::Fixed,
    );
    let request = pool_catalog(
        linear_profile(),
        AllocationLifetime::Request,
        'd',
        1,
        128,
        TestDemand::Fixed,
    );
    let first_pool = kv.pool_id.clone();
    let second_pool = second.pool_id.clone();
    let recurrent_pool = recurrent.pool_id.clone();
    let request_pool = request.pool_id.clone();
    let catalog = combine_catalogs(&[kv, second, recurrent, request]);
    let h = harness(new_runtime(&catalog, 1280), catalog, 1280, false);
    for (pool, bytes) in [
        (&first_pool, 256),
        (&second_pool, 128),
        (&recurrent_pool, 64),
        (&request_pool, 64),
    ] {
        h.root
            .maintenance_controller
            .grow_pool(pool, bytes)
            .unwrap();
    }
    let session = admitted_sequence_with_ceiling(&h.root, "all-pools", 4)
        .open_session()
        .unwrap();
    assert!(
        !SequenceSession::resident_cohort_persistent_capacity(&[Arc::clone(&session)]).unwrap()
    );
    h.root
        .maintenance_controller
        .grow_pool(&second_pool, 128)
        .unwrap();
    assert!(SequenceSession::resident_cohort_persistent_capacity(&[Arc::clone(&session)]).unwrap());
    // Fixed state has no spare bytes and needs none: the exact live parent and
    // child already hold it. Poison in any pool still invalidates the proof.
    h.root.dynamic_pools.pools[&recurrent_pool]
        .state
        .lock()
        .unwrap()
        .poisoned = true;
    assert!(!SequenceSession::resident_cohort_persistent_capacity(&[session]).unwrap());
}

#[test]
fn resident_capacity_fails_closed_for_nonpaged_growth_and_foreign_or_contended_owners() {
    let h = capacity_harness(linear_profile(), 512);
    let session = admitted_sequence_with_ceiling(&h.root, "nonpaged", 4)
        .open_session()
        .unwrap();
    assert!(!SequenceSession::resident_cohort_persistent_capacity(&[session]).unwrap());
    let first = capacity_harness(paged_profile(), 256);
    let a = admitted_sequence_with_ceiling(&first.root, "first-plan", 4)
        .open_session()
        .unwrap();
    let second = capacity_harness(paged_profile(), 256);
    let b = admitted_sequence_with_ceiling(&second.root, "second-plan", 4)
        .open_session()
        .unwrap();
    assert!(!SequenceSession::resident_cohort_persistent_capacity(&[Arc::clone(&a), b]).unwrap());
    let pool = &first.root.dynamic_pools.pools[&first.pool_ids[0]];
    let _busy = pool.state.lock().unwrap();
    assert!(!SequenceSession::resident_cohort_persistent_capacity(&[Arc::clone(&a)]).unwrap());
}

#[test]
fn resident_capacity_rejects_an_additional_request_parent_without_a_sequence() {
    let h = capacity_harness(paged_profile(), 512);
    let session = admitted_sequence_with_ceiling(&h.root, "resident-parent", 4)
        .open_session()
        .unwrap();
    assert!(SequenceSession::resident_cohort_persistent_capacity(&[Arc::clone(&session)]).unwrap());
    let extra_parent = admitted_request(&h.root, "unrepresented-parent");
    assert!(
        !SequenceSession::resident_cohort_persistent_capacity(&[Arc::clone(&session)]).unwrap()
    );
    drop(extra_parent);
    assert!(SequenceSession::resident_cohort_persistent_capacity(&[session]).unwrap());
}

#[test]
fn resident_capacity_is_unknown_while_lifecycle_owner_is_busy() {
    let h = capacity_harness(paged_profile(), 256);
    let session = admitted_sequence_with_ceiling(&h.root, "busy-lifecycle", 4)
        .open_session()
        .unwrap();
    let busy = h.root.lifecycle.write().unwrap();
    assert!(
        !SequenceSession::resident_cohort_persistent_capacity(&[Arc::clone(&session)]).unwrap()
    );
    drop(busy);
    // Release proves this local lifecycle gate is available again. The full
    // nonblocking capacity observation can still legitimately be Unknown:
    // try_read_lifecycle also try-locks the process-wide deferred-cleanup
    // registry, which unrelated parallel tests may currently own. This test
    // isolates the manually held lifecycle gate, not every capacity owner.
    assert!(h.root.lifecycle.try_read().is_ok());
}
