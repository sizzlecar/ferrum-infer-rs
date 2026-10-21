use super::*;
use crate::common::llm::{ModelCapacityResident, ModelKvPreparation, ModelKvReservation};
use crate::test_support::{tiny_llama_model, TinyLlamaConfig};
use ferrum_execution::model_executor::PersistentCapacityInspectionBudget;
use ferrum_kernels::backend::cpu::CpuBackend;
use std::sync::Arc;

fn model() -> LlamaFamilyModel<CpuBackend> {
    let mut model = tiny_llama_model(&TinyLlamaConfig {
        max_seq_len: 64,
        ..Default::default()
    });
    model.runtime_env.metal_paged_kv = Some(true);
    model.runtime_env.prefix_cache = true;
    model.runtime_env.kv_capacity = Some(64);
    model.runtime_env.kv_max_blocks = Some(4);
    model.runtime_env.paged_max_seqs = 2;
    model
}

fn prepare(model: &mut LlamaFamilyModel<CpuBackend>, cache_id: &str, tokens: usize) {
    assert!(matches!(
        model
            .prepare_token_kv(ModelKvPreparation {
                cache_id,
                capacity_hint: 32,
                prompt: &Arc::from(vec![1; tokens]),
                adapter: None,
            })
            .unwrap(),
        ModelKvReservation::Reserved(_)
    ));
}

fn query<'a>(
    residents: &'a [ModelCapacityResident<'a>],
    ceilings: &'a [usize],
) -> ModelPersistentCapacityQuery<'a> {
    ModelPersistentCapacityQuery {
        residents,
        through_tokens: ceilings,
        max_live_sequences: 2,
        max_wave_sequences: 2,
        inspection_budget: PersistentCapacityInspectionBudget::default(),
    }
}

fn resident(cache_id: &str) -> ModelCapacityResident<'_> {
    ModelCapacityResident {
        cache_id,
        consumed_tokens: 0,
        through_tokens: 32,
    }
}

#[test]
fn persistent_capacity_reuses_actual_retired_slots_for_a_larger_waiting_cohort() {
    let mut model = model();
    prepare(&mut model, "resident", 1);
    let residents = [resident("resident")];
    let supported = query(&residents, &[32; 6]);
    assert_eq!(
        model.observe_persistent_capacity(&supported),
        Err(U::InsufficientBacking),
        "configured width and free payload do not create metadata slots"
    );
    prepare(&mut model, "retiring", 1);
    assert_eq!(
        model.observe_persistent_capacity(&supported),
        Err(U::MembershipChanged)
    );
    model.close_token_sequence("retiring").unwrap();
    let before = model.kv_slot_capacity_snapshot().unwrap();
    assert_eq!(model.observe_persistent_capacity(&supported), Ok(()));
    assert_eq!(model.kv_slot_capacity_snapshot().unwrap(), before);
    assert_eq!(model.kv_free_pool.len(), 1);
    let too_large = query(&residents, &[32, 49]);
    assert_eq!(
        model.observe_persistent_capacity(&too_large),
        Err(U::InsufficientBacking),
        "top concurrent demand, not whole-cohort sum, must fit actual payload"
    );
    model.close_token_sequence("resident").unwrap();
    assert_eq!(
        model.observe_persistent_capacity(&query(&[], &[32; 6])),
        Ok(())
    );
}

#[test]
fn persistent_capacity_counts_shared_prefix_refs_and_ignores_soft_free_hashes() {
    let mut model = model();
    prepare(&mut model, "a", 17);
    let shared = model.kv_caches["a"][0].paged_block_indices[0];
    let hash = crate::common::paged_pool::block_hash(0, &[ferrum_types::TokenId::new(1); 16]);
    model
        .paged_block_alloc
        .as_ref()
        .unwrap()
        .lock()
        .unwrap()
        .register_block_hash(shared, hash);
    prepare(&mut model, "b", 17);
    assert_eq!(model.kv_caches["b"][0].paged_block_indices[0], shared);
    let residents = [resident("a"), resident("b")];
    let query = query(&residents, &[32; 5]);
    {
        let mut alloc = model.paged_block_alloc.as_ref().unwrap().lock().unwrap();
        let soft = alloc.allocate().unwrap();
        alloc.register_block_hash(soft, hash.wrapping_add(1));
        alloc.free(&[soft]);
    }
    assert_eq!(model.observe_persistent_capacity(&query), Ok(()));
    model
        .paged_block_alloc
        .as_ref()
        .unwrap()
        .lock()
        .unwrap()
        .acquire(shared);
    assert_eq!(
        model.observe_persistent_capacity(&query),
        Err(U::InvalidOwner),
        "an unattributed pin is not reclaimable capacity"
    );
    model
        .paged_block_alloc
        .as_ref()
        .unwrap()
        .lock()
        .unwrap()
        .free(&[shared]);
    assert_eq!(model.observe_persistent_capacity(&query), Ok(()));
}

#[test]
fn persistent_capacity_checks_payload_tables_layers_and_pending_context() {
    let mut model = model();
    prepare(&mut model, "a", 1);
    prepare(&mut model, "b", 1);
    let residents = [resident("a"), resident("b")];
    let query = query(&residents, &[32, 32]);
    assert_eq!(model.observe_persistent_capacity(&query), Ok(()));
    let last = model.paged_pools.as_mut().unwrap()[0].0.pop().unwrap();
    assert_eq!(
        model.observe_persistent_capacity(&query),
        Err(U::InsufficientBacking)
    );
    model.paged_pools.as_mut().unwrap()[0].0.push(last);
    let table = model.kv_caches.get_mut("a").unwrap()[0]
        .block_table
        .as_mut()
        .unwrap();
    let last = table.pop().unwrap();
    assert_eq!(
        model.observe_persistent_capacity(&query),
        Err(U::InsufficientBacking)
    );
    model.kv_caches.get_mut("a").unwrap()[0]
        .block_table
        .as_mut()
        .unwrap()
        .push(last);
    let prior = model.kv_caches["a"][1].paged_block_indices[0];
    model.kv_caches.get_mut("a").unwrap()[1].paged_block_indices[0] = prior ^ 1;
    assert_eq!(
        model.observe_persistent_capacity(&query),
        Err(U::InvalidOwner)
    );
    model.kv_caches.get_mut("a").unwrap()[1].paged_block_indices[0] = prior;
    let context = model.execution_context.take();
    assert_eq!(
        model.observe_persistent_capacity(&query),
        Err(U::PendingOwner)
    );
    model.execution_context = context;
    assert_eq!(model.observe_persistent_capacity(&query), Ok(()));
}

#[test]
fn persistent_capacity_bounded_inspection_and_contention_fail_closed() {
    let mut model = model();
    prepare(&mut model, "a", 1);
    let residents = [resident("a")];
    let mut query = query(&residents, &[32]);
    let held = model.paged_block_alloc.as_ref().unwrap().lock().unwrap();
    assert_eq!(model.observe_persistent_capacity(&query), Err(U::Contended));
    drop(held);
    query.inspection_budget.max_scratch_bytes = 1;
    assert_eq!(
        model.observe_persistent_capacity(&query),
        Err(U::InspectionBudget)
    );
    query.inspection_budget = PersistentCapacityInspectionBudget::default();
    query.inspection_budget.max_extents = 1;
    assert_eq!(
        model.observe_persistent_capacity(&query),
        Err(U::InspectionBudget)
    );
}

#[test]
fn persistent_capacity_contiguous_and_unallocated_pools_remain_unsupported() {
    let mut model = model();
    assert_eq!(
        model.observe_persistent_capacity(&query(&[], &[32])),
        Err(U::UnsupportedPacking)
    );
    model.runtime_env.metal_paged_kv = Some(false);
    prepare(&mut model, "a", 1);
    assert_eq!(
        model.observe_persistent_capacity(&query(&[resident("a")], &[32])),
        Err(U::UnsupportedPacking)
    );
}
