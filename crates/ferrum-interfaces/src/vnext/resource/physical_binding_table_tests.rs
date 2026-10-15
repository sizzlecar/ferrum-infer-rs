//! Actual Sequence tables: cached physical facts never replace current authority.
use super::*;
use crate::vnext::{DeviceBufferRetention, DisabledSubmissionWaveDispatchTimingSink};

pub(super) struct TestPhysicalBackend {
    // Mirror the backend ownership contract, including physical-before-logical drop.
    _retentions: Vec<DeviceBufferRetention>,
    _keeper: Arc<dyn Send + Sync>,
}

impl TestPhysicalBackend {
    pub(super) fn new(table: &SequencePhysicalBindingTable<TestBuffer>) -> Self {
        Self {
            _retentions: table
                .rows()
                .flat_map(|row| row.physical_regions())
                .map(|region| region.borrowed_buffer_and_physical_range().2.clone())
                .collect(),
            _keeper: Arc::clone(table.lifetime_keeper()),
        }
    }
}

fn table(
    harness: &Harness,
    snapshot: &SequenceBackingSnapshot<TestRuntime>,
) -> Arc<SequencePhysicalBindingTable<TestBuffer>> {
    harness
        .runtime
        .physical_binding_tables_enabled
        .store(true, Ordering::Relaxed);
    snapshot
        .physical_binding_table(
            &harness.root.dynamic_pools,
            harness.runtime.as_ref(),
            &DisabledSubmissionWaveDispatchTimingSink,
        )
        .unwrap()
        .unwrap()
}

fn current(
    harness: &Harness,
    table: &Arc<SequencePhysicalBindingTable<TestBuffer>>,
) -> Result<bool, VNextError> {
    SequencePhysicalBindingTable::current_batch(
        std::slice::from_ref(table),
        &harness.root.dynamic_pools,
        harness.runtime.as_ref(),
    )
}

#[test]
fn physical_binding_table_hot_epoch_keeps_fresh_poison_and_foreign_checks() {
    let (harness, sequence, snapshot) =
        segment_permit_tests::sequence_catalog_permit_fixture("physical-table-hot");
    let (foreign, foreign_sequence, foreign_snapshot) =
        segment_permit_tests::sequence_catalog_permit_fixture("physical-table-foreign");
    let cold = table(&harness, &snapshot);
    assert!(current(&harness, &cold).unwrap());
    let queries = harness.runtime.descriptor_queries.load(Ordering::Relaxed);
    let hot = table(&harness, &snapshot);
    assert!(Arc::ptr_eq(&cold, &hot));
    assert!(current(&harness, &hot).unwrap());
    assert_eq!(
        harness.runtime.descriptor_queries.load(Ordering::Relaxed),
        queries
    );
    assert!(SequencePhysicalBindingTable::current_batch(
        std::slice::from_ref(&cold),
        &foreign.root.dynamic_pools,
        harness.runtime.as_ref()
    )
    .is_err());
    assert!(SequencePhysicalBindingTable::current_batch(
        std::slice::from_ref(&cold),
        &harness.root.dynamic_pools,
        foreign.runtime.as_ref()
    )
    .is_err());
    let pool = &harness.root.dynamic_pools.pools[&harness.pool_ids[0]];
    pool.state.lock().unwrap().poisoned = true;
    assert!(
        current(&harness, &hot).is_err(),
        "equal directory epoch must not bypass poison"
    );
    pool.state.lock().unwrap().poisoned = false;
    let session = sequence.open_session().unwrap();
    let advanced = match session
        .try_ensure_backing_covers(
            SequenceResourceExtensionRequest::new(work(2), AdmissionPressureAction::WaitForRelease)
                .unwrap(),
        )
        .unwrap()
    {
        SequenceResourceExtensionDecision::Extended(snapshot) => snapshot,
        _ => panic!("logical progress must create a current snapshot"),
    };
    assert_ne!(snapshot.generation(), advanced.generation());
    assert!(Arc::ptr_eq(&cold, &table(&harness, &advanced)));
    assert!(current(&harness, &cold).unwrap());
    session.request_cancel().unwrap();
    session.try_abort().unwrap();
    drop(session);
    drop(advanced);
    drop(hot);
    drop(cold);
    drop(snapshot);
    drop(sequence);
    drop(foreign_snapshot);
    drop(foreign_sequence);
    close_dynamic_test_root(harness.root);
    close_dynamic_test_root(foreign.root);
}

#[test]
fn physical_binding_table_replacement_needs_exact_arc_and_generation_after_epoch_change() {
    let (harness, sequence, snapshot) =
        segment_permit_tests::sequence_catalog_permit_fixture("physical-table-replace");
    let (donor, donor_sequence, donor_snapshot) =
        segment_permit_tests::sequence_catalog_permit_fixture("physical-table-donor");
    let cold = table(&harness, &snapshot);
    let pool = &harness.root.dynamic_pools.pools[&harness.pool_ids[0]];
    let donor_pool = &donor.root.dynamic_pools.pools[&donor.pool_ids[0]];
    let ordinal = snapshot.backing_slices()[0].evidence.segments[0].chunk_ordinal();
    let original = pool.state.lock().unwrap().chunks.remove(&ordinal).unwrap();
    assert!(
        !current(&harness, &cold).unwrap(),
        "missing chunk must fall back before encoding"
    );
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
    backing.identity = BackingChunkIdentity::from_parts(
        original.backing.identity.pool_id().clone(),
        ordinal,
        original.backing.identity.generation() + 1,
    )
    .unwrap();
    backing.descriptor = original.backing.descriptor.clone();
    pool.state
        .lock()
        .unwrap()
        .chunks
        .insert(ordinal, replacement);
    assert!(
        !current(&harness, &cold).unwrap(),
        "new generation cannot reuse the old table"
    );
    let mut replacement = pool.state.lock().unwrap().chunks.remove(&ordinal).unwrap();
    Arc::get_mut(&mut replacement.backing).unwrap().identity = original.backing.identity.clone();
    pool.state
        .lock()
        .unwrap()
        .chunks
        .insert(ordinal, replacement);
    assert!(
        !current(&harness, &cold).unwrap(),
        "equal identity is not equal allocation Arc"
    );
    let mut state = pool.state.lock().unwrap();
    let before_replace = state.chunks.epoch();
    let mut replacement = state.chunks.insert(ordinal, original).unwrap();
    assert_ne!(
        state.chunks.epoch(),
        before_replace,
        "direct occupied-ordinal replacement must invalidate the cache"
    );
    drop(state);
    let backing = Arc::get_mut(&mut replacement.backing).unwrap();
    backing.identity = donor_identity;
    backing.descriptor = donor_descriptor;
    donor_pool
        .state
        .lock()
        .unwrap()
        .chunks
        .insert(ordinal, replacement);
    assert!(
        current(&harness, &cold).unwrap(),
        "restored exact current allocation must revalidate"
    );
    drop(cold);
    drop(snapshot);
    drop(sequence);
    drop(donor_snapshot);
    drop(donor_sequence);
    close_dynamic_test_root(harness.root);
    close_dynamic_test_root(donor.root);
}

#[test]
fn physical_binding_table_backend_keeper_outlives_sequence_without_owner_cycle() {
    let (harness, sequence, snapshot) =
        segment_permit_tests::sequence_catalog_permit_fixture("physical-table-keeper");
    let table = table(&harness, &snapshot);
    let table_probe = Arc::downgrade(&table);
    let backend = Arc::clone(table.backend_bindings().unwrap());
    let backend_probe = Arc::downgrade(&backend);
    drop(table);
    drop(snapshot);
    drop(sequence);
    assert!(
        table_probe.upgrade().is_none(),
        "backend must not retain a cycle back to the catalog"
    );
    let retained = harness
        .root
        .dynamic_pools
        .logical_admission
        .snapshot()
        .unwrap();
    assert_eq!(retained.active_sequences(), 1);
    assert_eq!(retained.active_requests(), 1);
    assert!(
        harness
            .root
            .maintenance_controller
            .status()
            .unwrap()
            .pools()[0]
            .live_occupancy()
            .total()
            .physical_bytes()
            > 0
    );
    drop(backend);
    assert!(backend_probe.upgrade().is_none());
    let released = harness
        .root
        .dynamic_pools
        .logical_admission
        .snapshot()
        .unwrap();
    assert_eq!(released.active_sequences(), 0);
    assert_eq!(released.active_requests(), 0);
    assert!(released
        .domains()
        .iter()
        .all(|domain| domain.used().get() == 0));
    assert_eq!(
        harness
            .root
            .maintenance_controller
            .status()
            .unwrap()
            .pools()[0]
            .live_occupancy()
            .total()
            .physical_bytes(),
        0
    );
    assert!(!harness
        .runtime
        .dropped_after_backend
        .load(Ordering::Relaxed));
    close_dynamic_test_root(harness.root);
}
