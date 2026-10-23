use super::*;

impl Harness {
    /// Other parallel fixtures share the process cleanup registry. Wait only
    /// for its transient observation contention; preserve every actual result.
    pub(super) fn observed_persistent_capacity(
        &self,
        residents: &[(Arc<SequenceSession<TestRuntime>>, u64)],
        ceilings: &[u64],
        slots: usize,
        budget: PersistentCapacityInspectionBudget,
    ) -> Result<(), PersistentCapacityUnknown> {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
        loop {
            let _settled_cleanup = self.root.deferred_cleanup_status();
            let result = self
                .root
                .persistent_capacity_for_cohort(residents, ceilings, slots, budget);
            if result != Err(PersistentCapacityUnknown::Contended) {
                return result;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "original capacity observation remained contended"
            );
            std::thread::yield_now();
        }
    }
}

fn envelope_harness(profile: DynamicStorageProfile, demand: TestDemand, bytes: u64) -> Harness {
    let catalog = pool_catalog(profile, AllocationLifetime::Sequence, 'b', 1, 4096, demand);
    let h = harness(new_runtime(&catalog, 4096), catalog, 4096, false);
    h.root
        .maintenance_controller
        .grow_pool(&h.pool_ids[0], bytes)
        .unwrap();
    h
}

#[test]
fn persistent_envelope_bounds_live_slots_instead_of_allocating_the_waiting_queue() {
    let h = envelope_harness(paged_profile(), TestDemand::Tokens, 2048);
    // Sixteen waiting requests, eight original logical slots. Each full request
    // costs256 bytes: simultaneous top8 fits, sum of all waiting demand does not.
    let ceilings = [4; 16];
    let before = h.root.dynamic_pool_status().unwrap();
    let allocations = h.runtime.allocate_calls();
    assert_eq!(
        h.observed_persistent_capacity(&[], &ceilings, 8, Default::default()),
        Ok(())
    );
    assert_eq!(h.root.dynamic_pool_status().unwrap(), before);
    assert_eq!(h.runtime.allocate_calls(), allocations);
    assert_eq!(
        h.observed_persistent_capacity(&[], &ceilings, 4, Default::default()),
        Err(PersistentCapacityUnknown::PendingOwner)
    );
}

#[test]
fn persistent_envelope_counts_exact_current_owner_and_rejects_unreturned_owner() {
    let h = envelope_harness(paged_profile(), TestDemand::Tokens, 2048);
    let session = admitted_sequence_with_ceiling(&h.root, "cohort-live", 4)
        .open_session()
        .unwrap();
    let retained = session.resources().backing_snapshot().unwrap();
    let ceilings = [4; 16];
    let before = h.root.dynamic_pool_status().unwrap();
    assert_eq!(
        h.observed_persistent_capacity(
            &[(Arc::clone(&session), 4)],
            &ceilings,
            8,
            Default::default()
        ),
        Ok(())
    );
    assert_eq!(h.root.dynamic_pool_status().unwrap(), before);
    drop(session);
    // Last backing generation still retains the original logical slot/parent.
    assert_eq!(
        h.observed_persistent_capacity(&[], &ceilings, 8, Default::default()),
        Err(PersistentCapacityUnknown::PendingOwner)
    );
    drop(retained);
    assert_eq!(
        h.observed_persistent_capacity(&[], &ceilings, 8, Default::default()),
        Ok(())
    );
}

#[test]
fn persistent_envelope_requires_actual_backing_and_every_persistent_pool() {
    let kv = pool_catalog(
        paged_profile(),
        AllocationLifetime::Sequence,
        'a',
        1,
        4096,
        TestDemand::Tokens,
    );
    let fixed = pool_catalog(
        linear_profile(),
        AllocationLifetime::Sequence,
        'b',
        1,
        1024,
        TestDemand::Fixed,
    );
    let kv_id = kv.pool_id.clone();
    let fixed_id = fixed.pool_id.clone();
    let catalog = combine_catalogs(&[kv, fixed]);
    let h = harness(new_runtime(&catalog, 5120), catalog, 5120, false);
    h.root
        .maintenance_controller
        .grow_pool(&kv_id, 2048)
        .unwrap();
    h.root
        .maintenance_controller
        .grow_pool(&fixed_id, 448)
        .unwrap();
    assert_eq!(
        h.observed_persistent_capacity(&[], &[4; 16], 8, Default::default()),
        Err(PersistentCapacityUnknown::InsufficientBacking)
    );
    h.root
        .maintenance_controller
        .grow_pool(&fixed_id, 64)
        .unwrap();
    assert_eq!(
        h.observed_persistent_capacity(&[], &[4; 16], 8, Default::default()),
        Ok(())
    );
}

#[test]
fn persistent_envelope_rejects_variable_or_misaligned_contiguous_packing() {
    let variable = envelope_harness(linear_profile(), TestDemand::Tokens, 2048);
    assert_eq!(
        variable.observed_persistent_capacity(&[], &[4; 8], 8, Default::default()),
        Err(PersistentCapacityUnknown::UnsupportedPacking)
    );
    let fixed = envelope_harness(linear_profile(), TestDemand::Fixed, 528);
    // Enough aggregate bytes but a 16-byte tail is not a complete64-byte cell.
    assert_eq!(
        fixed.observed_persistent_capacity(&[], &[4; 8], 8, Default::default()),
        Err(PersistentCapacityUnknown::UnsupportedPacking)
    );
}

#[test]
fn persistent_envelope_rejects_prepared_foreign_claim_and_recovers_after_rollback() {
    let h = envelope_harness(paged_profile(), TestDemand::Tokens, 2048);
    let pool = &h.root.dynamic_pools.pools[&h.pool_ids[0]];
    let request = evaluated_request(pool, 64);
    let BackingPrepareDecision::Prepared(prepared) =
        h.root.dynamic_pools.prepare_claim(&[request]).unwrap()
    else {
        panic!("existing backing must prepare");
    };
    assert_eq!(
        h.observed_persistent_capacity(&[], &[4; 16], 8, Default::default()),
        Err(PersistentCapacityUnknown::PendingOwner)
    );
    drop(prepared);
    assert_eq!(
        h.observed_persistent_capacity(&[], &[4; 16], 8, Default::default()),
        Ok(())
    );
}

#[test]
fn persistent_envelope_stays_bounded_and_try_only() {
    let h = envelope_harness(paged_profile(), TestDemand::Tokens, 2048);
    let budget = PersistentCapacityInspectionBudget {
        max_scratch_bytes: 1,
        ..Default::default()
    };
    assert_eq!(
        h.observed_persistent_capacity(&[], &[4; 8], 8, budget),
        Err(PersistentCapacityUnknown::InspectionBudget)
    );
    let pool = &h.root.dynamic_pools.pools[&h.pool_ids[0]];
    let busy = pool.state.lock().unwrap();
    assert_eq!(
        h.root
            .persistent_capacity_for_cohort(&[], &[4; 8], 8, Default::default()),
        Err(PersistentCapacityUnknown::Contended)
    );
    drop(busy);
    assert_eq!(
        h.root
            .persistent_capacity_for_cohort(&[], &[4; 65], 8, Default::default()),
        Err(PersistentCapacityUnknown::InvalidQuery)
    );
}
