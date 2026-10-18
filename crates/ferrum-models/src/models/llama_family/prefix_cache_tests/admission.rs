//! Real device prefix admission and chunk publication. No manually registered
//! hashes: every reusable block was published by committed model token work.
use super::*;
use crate::common::llm::{ModelTokenBatchOutcome, ModelTokenCloseOutcome};
use ferrum_types::{FerrumError, TokenId};

fn free<B: MoeLlmBackend>(model: &LlamaFamilyModel<B, KvFp16>) -> usize {
    model
        .paged_block_alloc
        .as_ref()
        .unwrap()
        .lock()
        .unwrap()
        .free_count()
}

fn close<B: MoeLlmBackend>(model: &mut LlamaFamilyModel<B, KvFp16>, id: &str) {
    assert!(matches!(
        model.close_token_execution(id),
        ModelTokenCloseOutcome::Ended { error: None }
    ));
}

fn producer_chunk<B: MoeLlmBackend>(
    model: &mut LlamaFamilyModel<B, KvFp16>,
    id: &str,
    tokens: &[u32],
    position: usize,
    sample: bool,
) -> Option<Vec<f32>> {
    let unified = model.supports_varlen_qkv;
    let outcome = model.execute_token_batch(&[ModelTokenInput {
        cache_id: id,
        work: ModelTokenWork::Prefill { tokens, position },
        logits_policy: &LogitsReturnPolicy::FullLogits,
        sample,
    }]);
    let product = match outcome {
        ModelTokenBatchOutcome::Cohort(ModelTokenOutcome::Terminal(Ok(mut rows))) => {
            assert!(unified);
            assert_eq!(rows.len(), 1);
            rows.pop().unwrap()
        }
        ModelTokenBatchOutcome::Serial(mut rows) => {
            assert!(!unified, "cold producers must retain physical batching");
            assert_eq!(rows.len(), 1);
            match rows.pop().unwrap() {
                ModelTokenOutcome::Terminal(Ok(product)) => product,
                other => panic!("producer did not finish: {other:?}"),
            }
        }
        ModelTokenBatchOutcome::Cohort(other) => {
            panic!("producer batch did not finish: {other:?}")
        }
    };
    match product {
        Some(ExecutorSamplingOutput::FullLogits(logits)) => {
            assert!(sample);
            assert_eq!(logits.len(), model.cfg.vocab_size);
            assert!(logits.iter().all(|value| value.is_finite()));
            Some(logits)
        }
        None => {
            assert!(!sample);
            None
        }
        other => panic!("producer returned an unexpected sampling product: {other:?}"),
    }
}

fn check_frontier<B: MoeLlmBackend>(
    model: &LlamaFamilyModel<B, KvFp16>,
    id: &str,
    length: usize,
    read_u32: &impl Fn(&B::Buffer, usize) -> Vec<u32>,
) {
    for cache in &model.kv_caches[id] {
        assert_eq!(cache.len, length);
        assert_eq!(
            read_u32(cache.context_lens.as_ref().unwrap(), 1),
            [length as u32]
        );
        assert_eq!(
            read_u32(
                cache.block_table.as_ref().unwrap(),
                cache.paged_block_indices.len()
            ),
            cache.paged_block_indices
        );
    }
}

fn live_bits<B: MoeLlmBackend>(
    model: &LlamaFamilyModel<B, KvFp16>,
    id: &str,
) -> Vec<(Vec<u32>, Vec<u32>)> {
    let block_size = LLAMA_PAGED_BLOCK_SIZE;
    let heads = model.cfg.num_kv_heads;
    let dim = model.cfg.head_dim;
    let per_block = heads * block_size * dim;
    model
        .paged_pools
        .as_ref()
        .unwrap()
        .iter()
        .zip(&model.kv_caches[id])
        .map(|((k, v), cache)| {
            let extent =
                (cache.paged_block_indices.iter().copied().max().unwrap() as usize + 1) * per_block;
            // The ordinary FP16 paged kernels use [block, head, token, dim].
            // Capture only initialized tokens, including a partial last block.
            let capture = |buffer| {
                let values = B::to_vec(buffer, extent);
                let mut bits = Vec::new();
                for head in 0..heads {
                    for position in 0..cache.len {
                        let block = cache.paged_block_indices[position / block_size] as usize;
                        let start =
                            block * per_block + (head * block_size + position % block_size) * dim;
                        bits.extend(
                            values[start..start + dim]
                                .iter()
                                .map(|value| value.to_bits()),
                        );
                    }
                }
                bits
            };
            (capture(k), capture(v))
        })
        .collect()
}

fn soft_free_shortfall<B: MoeLlmBackend>(read_u32: &impl Fn(&B::Buffer, usize) -> Vec<u32>) {
    let mut actual = model::<B>(true);
    actual.runtime_env.kv_max_blocks = Some(3);
    actual.runtime_env.paged_max_seqs = 4;
    let prompt: Arc<[u32]> = (1..=33).collect::<Vec<_>>().into();
    let producer: Arc<[u32]> = Arc::from(&prompt[..32]);
    let holder: Arc<[u32]> = Arc::from(&prompt[..17]);
    assert!(matches!(
        actual
            .prepare_token_kv(preparation("producer", &producer))
            .unwrap(),
        ModelKvReservation::Reserved(_)
    ));
    prefill_product(&mut actual, "producer", &producer, 0, true).unwrap();
    let original = actual.kv_caches["producer"][0].paged_block_indices.clone();
    assert_eq!(original, [0, 1]);
    assert!(matches!(
        actual
            .prepare_token_kv(preparation("holder", &holder))
            .unwrap(),
        ModelKvReservation::Reserved(_)
    ));
    prefill_product(&mut actual, "holder", &holder, 0, true).unwrap();
    assert!(actual.kv_caches["holder"]
        .iter()
        .all(|cache| cache.paged_block_indices == [0, 2]));
    close(&mut actual, "producer");
    assert_eq!(free(&actual), 1);
    let holder_bits = live_bits(&actual, "holder");
    let ModelKvReservation::CapacityDeferred(deferred) = actual
        .prepare_token_kv(preparation("waiter", &prompt))
        .unwrap()
    else {
        panic!("reviving the soft-free prefix still leaves no private suffix block");
    };
    assert_eq!(deferred.required_blocks(), 2);
    assert_eq!(deferred.snapshot().free_blocks, 1);
    close(&mut actual, "waiter");
    check_frontier(&actual, "holder", 17, read_u32);
    assert_eq!(live_bits(&actual, "holder"), holder_bits);
    {
        let mut allocator = actual.paged_block_alloc.as_ref().unwrap().lock().unwrap();
        assert_eq!(
            (
                allocator.ref_count(0),
                allocator.ref_count(1),
                allocator.ref_count(2)
            ),
            (1, 0, 1)
        );
        assert_eq!(allocator.free_count(), 1);
        assert_eq!(allocator.hash_table_size(), 2);
        // Observe the automatically published hash through the actual lookup;
        // immediately return its temporary reference to the soft-free state.
        let tokens: Vec<_> = producer.iter().copied().map(TokenId::new).collect();
        let hashes = crate::common::paged_pool::block_hash_chain(&tokens, LLAMA_PAGED_BLOCK_SIZE);
        let revived = allocator
            .try_acquire_by_hash(hashes[1])
            .expect("rollback retains the soft-free hash");
        assert_eq!(revived, original[1]);
        allocator.free(&[revived]);
        assert_eq!(allocator.ref_count(revived), 0);
    }
    close(&mut actual, "holder");
    assert_eq!(free(&actual), 3);
    assert!(matches!(
        actual
            .prepare_token_kv(preparation("retry", &prompt))
            .unwrap(),
        ModelKvReservation::Reserved(_)
    ));
    check_frontier(&actual, "retry", 0, read_u32);
    assert!(actual.kv_caches["retry"]
        .iter()
        .all(|cache| cache.paged_block_indices[..2] == original));
    close(&mut actual, "retry");
    assert_eq!(free(&actual), 3);
}

fn admission_and_chunks<B: MoeLlmBackend>(read_u32: impl Fn(&B::Buffer, usize) -> Vec<u32>) {
    let mut actual = model::<B>(true);
    actual.runtime_env.kv_max_blocks = Some(3);
    actual.runtime_env.paged_max_seqs = 4;
    let mut reference = model::<B>(false);
    let prompt: Arc<[u32]> = (1..=33).collect::<Vec<_>>().into();
    let producer: Arc<[u32]> = Arc::from(&prompt[..32]);
    assert!(matches!(
        actual
            .prepare_token_kv(preparation("owner", &producer))
            .unwrap(),
        ModelKvReservation::Reserved(_)
    ));
    assert!(matches!(
        reference
            .prepare_token_kv(preparation("reference", &prompt))
            .unwrap(),
        ModelKvReservation::Reserved(_)
    ));
    check_frontier(&actual, "owner", 0, &read_u32);
    assert_eq!(free(&actual), 1);
    assert!(producer_chunk(&mut actual, "owner", &producer[..13], 0, false).is_none());
    assert!(producer_chunk(&mut reference, "reference", &prompt[..13], 0, false).is_none());
    assert_eq!(
        actual
            .paged_block_alloc
            .as_ref()
            .unwrap()
            .lock()
            .unwrap()
            .hash_table_size(),
        0
    );
    let owner_logits = producer_chunk(&mut actual, "owner", &producer[13..], 13, true).unwrap();
    let reference_logits =
        producer_chunk(&mut reference, "reference", &prompt[13..32], 13, true).unwrap();
    assert_eq!(owner_logits, reference_logits);
    assert_eq!(
        actual
            .paged_block_alloc
            .as_ref()
            .unwrap()
            .lock()
            .unwrap()
            .hash_table_size(),
        2
    );
    let owner_mapping = actual.kv_caches["owner"][0].paged_block_indices.clone();
    let bits = owner_bits(&actual, "owner");

    // A full immutable prompt reaches preparation before any fresh allocation:
    // two live prefix blocks + one free suffix block must admit 33 tokens.
    assert!(matches!(
        actual
            .prepare_token_kv(preparation("consumer", &prompt))
            .unwrap(),
        ModelKvReservation::Reserved(_)
    ));
    check_frontier(&actual, "consumer", 0, &read_u32);
    assert_eq!(free(&actual), 0);
    for cache in &actual.kv_caches["consumer"] {
        assert_eq!(&cache.paged_block_indices[..2], &owner_mapping);
        assert!(!owner_mapping.contains(&cache.paged_block_indices[2]));
    }
    assert_eq!(actual.prefix_cache_hits, 0);
    assert_eq!(actual.prefix_cache_saved_prefill_tokens, 0);
    let before_refs = {
        let allocator = actual.paged_block_alloc.as_ref().unwrap().lock().unwrap();
        owner_mapping
            .iter()
            .map(|&block| allocator.ref_count(block))
            .collect::<Vec<_>>()
    };
    assert_eq!(before_refs, [2, 2]);

    // Real suffix shortfall must return any newly acquired prefix references
    // before the caller closes its unadmitted cache bookkeeping.
    let ModelKvReservation::CapacityDeferred(deferred) = actual
        .prepare_token_kv(preparation("failed", &prompt))
        .unwrap()
    else {
        panic!("the private suffix cannot fit in the full three-block pool");
    };
    assert_eq!(deferred.required_blocks(), 1);
    assert_eq!(deferred.snapshot().free_blocks, 0);
    assert!(actual.kv_caches.get("failed").is_none_or(|layers| layers
        .iter()
        .all(|cache| cache.len == 0 && cache.paged_block_indices.is_empty())));
    close(&mut actual, "failed");
    assert_eq!(free(&actual), 0);
    {
        let allocator = actual.paged_block_alloc.as_ref().unwrap().lock().unwrap();
        assert_eq!(
            owner_mapping
                .iter()
                .map(|&block| allocator.ref_count(block))
                .collect::<Vec<_>>(),
            before_refs
        );
        assert_eq!(allocator.hash_table_size(), 2);
    }

    // Reject adapter/prefix composition before loading any adapter or taking
    // base-model references. The nonexistent path must never be consulted.
    let adapter = crate::lora::ActiveLoraAdapter {
        name: "unloaded".into(),
        path: "unloaded-prefix-adapter".into(),
    };
    let error = actual
        .prepare_token_kv(ModelKvPreparation {
            adapter: Some(&adapter),
            ..preparation("adapter", &prompt)
        })
        .unwrap_err();
    assert!(matches!(error, FerrumError::Unsupported { .. }));
    assert!(!actual.kv_caches.contains_key("adapter"));
    {
        let allocator = actual.paged_block_alloc.as_ref().unwrap().lock().unwrap();
        assert_eq!(
            owner_mapping
                .iter()
                .map(|&block| allocator.ref_count(block))
                .collect::<Vec<_>>(),
            before_refs
        );
        assert_eq!(allocator.free_count(), 0);
        assert_eq!(allocator.hash_table_size(), 2);
    }
    assert_eq!(owner_bits(&actual, "owner"), bits);

    let mut wrong = prompt[..13].to_vec();
    wrong[0] += 1;
    assert!(matches!(
        actual.execute_token_work(&ModelTokenInput {
            cache_id: "consumer",
            work: ModelTokenWork::Prefill {
                tokens: &wrong,
                position: 0
            },
            logits_policy: &LogitsReturnPolicy::FullLogits,
            sample: false,
        }),
        ModelTokenOutcome::NotSubmitted(_)
    ));
    check_frontier(&actual, "consumer", 0, &read_u32);
    assert_eq!(
        actual
            .paged_block_alloc
            .as_ref()
            .unwrap()
            .lock()
            .unwrap()
            .hash_table_size(),
        2
    );

    // These public legacy entries must not bypass the prepared prompt contract
    // and write position zero of the producer's still-live shared blocks.
    // Validation precedes backend eligibility: an unsupported batch kernel
    // must not hide the invalid write as a generic Unsupported fallback.
    let consumer_mapping = actual.kv_caches["consumer"][0].paged_block_indices.clone();
    let assert_pristine = |model: &LlamaFamilyModel<B, KvFp16>| {
        check_frontier(model, "consumer", 0, &read_u32);
        assert!(model.kv_caches["consumer"]
            .iter()
            .all(|cache| cache.paged_block_indices == consumer_mapping));
        assert_eq!(owner_bits(model, "owner"), bits);
        assert_eq!(model.prefix_cache_hits, 0);
        assert_eq!(model.prefix_cache_saved_prefill_tokens, 0);
        let allocator = model.paged_block_alloc.as_ref().unwrap().lock().unwrap();
        assert_eq!(
            owner_mapping
                .iter()
                .map(|&block| allocator.ref_count(block))
                .collect::<Vec<_>>(),
            before_refs
        );
        assert_eq!(allocator.free_count(), 0);
        assert_eq!(allocator.hash_table_size(), 2);
    };
    let error = actual
        .unified_forward_with_logits_policy(
            &[("consumer".to_owned(), vec![wrong[0]], 0, true)],
            &[LogitsReturnPolicy::FullLogits],
        )
        .expect_err("legacy unified work must validate the prepared shared prompt");
    assert!(matches!(error, FerrumError::RequestValidation { .. }));
    assert_pristine(&actual);
    let error = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        actual.decode_internal("consumer", wrong[0], 0)
    }))
    .expect_err("legacy decode must reject writes into the unconsumed shared prompt");
    assert!(matches!(
        error.downcast_ref::<FerrumError>(),
        Some(FerrumError::RequestValidation { .. })
    ));
    assert_pristine(&actual);

    for (start, end) in [(0, 13), (13, 20)] {
        assert!(
            prefill_product(&mut actual, "consumer", &prompt[start..end], start, false).is_none()
        );
        check_frontier(&actual, "consumer", end, &read_u32);
        assert_eq!(actual.prefix_cache_hits, 1);
        assert_eq!(actual.prefix_cache_saved_prefill_tokens, end as u64);
        assert_eq!(owner_bits(&actual, "owner"), bits);
    }
    let logits = prefill_product(&mut actual, "consumer", &prompt[20..], 20, true).unwrap();
    // Independent cold execution uses the producer's actual 13+19 numerical
    // shapes followed by the private suffix; no shared reference/hash lookup.
    let expected = prefill_product(&mut reference, "reference", &prompt[32..], 32, true).unwrap();
    assert_eq!(logits, expected);
    check_frontier(&actual, "consumer", 33, &read_u32);
    assert_eq!(actual.prefix_cache_saved_prefill_tokens, 32);
    assert_eq!(owner_bits(&actual, "owner"), bits);
    assert_eq!(
        actual.decode_internal("consumer", 70, 33),
        reference.decode_internal("reference", 70, 33)
    );
    close(&mut actual, "consumer");
    assert_eq!(free(&actual), 1);
    assert_eq!(owner_bits(&actual, "owner"), bits);
    close(&mut actual, "owner");
    assert_eq!(free(&actual), 3);

    // Both hashes are soft-free. Lookup must revive them before the private
    // reservation can evict them. Cancel after a fully cached intermediate.
    assert!(matches!(
        actual
            .prepare_token_kv(preparation("revived", &prompt))
            .unwrap(),
        ModelKvReservation::Reserved(_)
    ));
    assert_eq!(
        &actual.kv_caches["revived"][0].paged_block_indices[..2],
        &owner_mapping
    );
    check_frontier(&actual, "revived", 0, &read_u32);
    assert!(prefill_product(&mut actual, "revived", &prompt[..13], 0, false).is_none());
    check_frontier(&actual, "revived", 13, &read_u32);
    close(&mut actual, "revived");
    assert_eq!(free(&actual), 3);
    assert!(actual.kv_caches.is_empty());
    assert_eq!(
        actual
            .paged_block_alloc
            .as_ref()
            .unwrap()
            .lock()
            .unwrap()
            .hash_table_size(),
        2
    );
    close(&mut reference, "reference");
    soft_free_shortfall::<B>(&read_u32);
}

#[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
#[test]
#[ignore = "requires an actual Metal device"]
fn prefix_admission_real_metal_reuses_before_reserving_and_publishes_chunks() {
    use ferrum_kernels::backend::metal::MetalBackend;
    admission_and_chunks::<MetalBackend>(|buffer, count| {
        // Production preparation/execution already returned a terminal proof.
        unsafe { std::slice::from_raw_parts(buffer.raw().contents().cast::<u32>(), count).to_vec() }
    });
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires an actual CUDA device"]
fn prefix_admission_real_cuda_reuses_before_reserving_and_publishes_chunks() {
    use ferrum_kernels::backend::{cuda::CudaBackend, Backend};
    let context = CudaBackend::new_context();
    admission_and_chunks::<CudaBackend>(|buffer, count| {
        context.stream.clone_dtoh(buffer.as_u32()).unwrap()[..count].to_vec()
    });
}
