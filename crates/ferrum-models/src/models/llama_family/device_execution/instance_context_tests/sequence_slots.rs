//! Actual table/pool owners with deterministic original-context retirement.
use super::*;
use crate::common::llm::{ModelKvPreparation, ModelKvReservation, ModelTokenCloseOutcome};

fn prepare(model: &mut LlamaFamilyModel<TrackedCpu>, id: &str) -> ModelKvReservation {
    model
        .prepare_token_kv(ModelKvPreparation {
            cache_id: id,
            capacity_hint: 32,
            prompt: &Arc::from([1u32, 2]),
            adapter: None,
        })
        .unwrap()
}

#[test]
fn physical_sequence_slots_reject_without_allocation_and_reuse_only_after_retirement() {
    let mut model = tracked_model();
    model.runtime_env.metal_paged_kv = Some(true);
    model.runtime_env.prefix_cache = false;
    model.runtime_env.kv_capacity = Some(32);
    model.runtime_env.paged_max_seqs = 2;
    model.runtime_env.kv_max_blocks = Some(8);
    for id in ["first", "peer"] {
        assert!(matches!(
            prepare(&mut model, id),
            ModelKvReservation::Reserved(_)
        ));
    }
    let first_pointers = model.kv_caches["first"]
        .iter()
        .map(|cache| {
            (
                cache.block_table.as_ref().unwrap().as_ptr(),
                cache.context_lens.as_ref().unwrap().as_ptr(),
                cache.k.as_ptr(),
                cache.v.as_ptr(),
            )
        })
        .collect::<Vec<_>>();
    let calls = Arc::clone(&model.execution_context.as_ref().unwrap().calls);
    let before = calls.load(Ordering::SeqCst);
    let ModelKvReservation::SequenceSlotsDeferred(deferral) = prepare(&mut model, "waiting") else {
        panic!("metadata slots must defer independently of six free payload blocks");
    };
    assert_eq!(
        (deferral.capacity_slots(), deferral.occupied_slots()),
        (2, 2)
    );
    assert_eq!(calls.load(Ordering::SeqCst), before);
    assert_eq!(model.kv_caches.len(), 2);
    assert_eq!(
        model
            .paged_block_alloc
            .as_ref()
            .unwrap()
            .lock()
            .unwrap()
            .free_count(),
        6
    );
    assert!(model.kv_free_pool.is_empty());

    // Common legacy construction is guarded too, before any allocation/write.
    let rejected: DeviceFrameResult<()> = model.run_preparation_frame(|model, frame| {
        model.ensure_kv_in_context(&mut frame.context, "legacy", Some(32));
    });
    assert!(matches!(
        rejected,
        DeviceFrameResult::Terminal(Err(FerrumError::ResourceExhausted { .. }))
    ));
    assert_eq!(calls.load(Ordering::SeqCst), before);
    assert_eq!(model.kv_caches.len(), 2);

    // Original graph-bound storage is retained when retirement fails; this is
    // a real model owner boundary, not a fabricated allocator shortfall.
    model.single_graph_cache_id = Some("first".into());
    let blocked = Arc::clone(
        &model
            .execution_context
            .as_ref()
            .unwrap()
            .storage_retirement_blocked,
    );
    blocked.store(true, Ordering::SeqCst);
    assert!(matches!(
        model.close_token_execution("first"),
        ModelTokenCloseOutcome::Held { .. }
    ));
    assert!(matches!(
        prepare(&mut model, "waiting"),
        ModelKvReservation::SequenceSlotsDeferred(_)
    ));
    assert!(model.kv_free_pool.is_empty());
    assert_eq!(calls.load(Ordering::SeqCst), before);
    blocked.store(false, Ordering::SeqCst);
    assert!(matches!(
        model.close_token_execution("first"),
        ModelTokenCloseOutcome::Ended { error: None }
    ));
    assert_eq!(model.kv_free_pool.len(), 1);
    assert!(matches!(
        prepare(&mut model, "waiting"),
        ModelKvReservation::Reserved(_)
    ));
    let replacement_pointers = model.kv_caches["waiting"]
        .iter()
        .map(|cache| {
            (
                cache.block_table.as_ref().unwrap().as_ptr(),
                cache.context_lens.as_ref().unwrap().as_ptr(),
                cache.k.as_ptr(),
                cache.v.as_ptr(),
            )
        })
        .collect::<Vec<_>>();
    assert_eq!(replacement_pointers, first_pointers);
    assert_eq!(model.kv_caches.len(), 2);
    assert!(matches!(
        prepare(&mut model, "peer"),
        ModelKvReservation::Reserved(_)
    ));
    for id in ["waiting", "peer"] {
        assert!(matches!(
            model.close_token_execution(id),
            ModelTokenCloseOutcome::Ended { error: None }
        ));
    }
}

#[test]
fn physical_sequence_slots_do_not_change_contiguous_owner_policy() {
    let mut model = tracked_model();
    model.runtime_env.metal_paged_kv = Some(false);
    model.runtime_env.paged_max_seqs = 1;
    for id in ["first", "second"] {
        assert!(matches!(
            prepare(&mut model, id),
            ModelKvReservation::Reserved(_)
        ));
    }
    assert!(model.paged_pools.is_none());
    assert_eq!(model.kv_caches.len(), 2);
    for id in ["first", "second"] {
        assert!(matches!(
            model.close_token_execution(id),
            ModelTokenCloseOutcome::Ended { error: None }
        ));
    }
}

#[test]
fn physical_sequence_slots_batch_shortfall_precedes_every_new_cache() {
    let mut model = tracked_model();
    model.runtime_env.metal_paged_kv = Some(true);
    model.runtime_env.prefix_cache = false;
    model.runtime_env.kv_capacity = Some(32);
    model.runtime_env.paged_max_seqs = 2;
    model.runtime_env.kv_max_blocks = Some(8);
    assert!(matches!(
        prepare(&mut model, "resident"),
        ModelKvReservation::Reserved(_)
    ));
    let calls = Arc::clone(&model.execution_context.as_ref().unwrap().calls);
    let before = calls.load(Ordering::SeqCst);
    let demands = ["new-a", "new-b", "new-a"].map(|cache_id| {
        ferrum_execution::model_executor::KvSlotRequest {
            cache_id: cache_id.into(),
            target_len: 1,
            admission_target_len: Some(1),
        }
    });
    let ModelKvReservation::SequenceSlotsDeferred(deferred) =
        model.reserve_paged_token_kv_slots(&demands).unwrap()
    else {
        panic!("two distinct new caches cannot consume one physical slot");
    };
    assert_eq!(
        (deferred.occupied_slots(), deferred.required_slots()),
        (1, 2)
    );
    assert_eq!(model.kv_caches.len(), 1);
    assert_eq!(calls.load(Ordering::SeqCst), before);
    assert!(matches!(
        model.close_token_execution("resident"),
        ModelTokenCloseOutcome::Ended { error: None }
    ));
}
