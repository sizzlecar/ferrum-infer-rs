//! Actual device paged-KV allocation and numerical continuation. These tests
//! use the same real model for storage and computation, unlike the separately
//! labelled CPU allocator/contiguous-model composition.

use super::*;

fn capacity_deferral(outcome: ExecutionBatchOutcome) -> TokenAdmissionCapacityDeferral {
    let ExecutionBatchOutcome::NotSubmitted(batch) = outcome else {
        panic!("the unchanged physical cohort must report zero token submission: {outcome:?}");
    };
    let NoSubmissionReason::Deferred(deferral) = batch.into_reason() else {
        panic!("allocator shortfall must remain typed");
    };
    deferral.as_allocator_capacity().unwrap().clone()
}

fn assert_live<B: MoeLlmBackend>(
    executor: &LlmExecutor,
    model: &Arc<Mutex<LlamaFamilyModel<B>>>,
    frontier: &ExecutionFrontier,
    position: usize,
    blocks: usize,
) {
    // Match production lock order: never wait for the model with registry held.
    let cache_id = {
        let sequences = executor.token_sequences.lock();
        let entry = &sequences.live[frontier.sequence()];
        assert_eq!(&entry.frontier, frontier);
        assert!(!entry.closing);
        entry.cache_id.clone()
    };
    assert!(frontier.sequence().observed_end().is_none());
    let model = model.lock();
    for cache in &model.kv_caches[&cache_id] {
        assert_eq!(cache.len, position);
        assert_eq!(cache.paged_block_indices.len(), blocks);
    }
}

fn free_blocks<B: MoeLlmBackend>(model: &Arc<Mutex<LlamaFamilyModel<B>>>) -> usize {
    model
        .lock()
        .paged_block_alloc
        .as_ref()
        .unwrap()
        .lock()
        .unwrap()
        .free_count()
}

async fn real_capacity_boundary<B: MoeLlmBackend>() {
    let mut reference = real_model::<B>(true);
    let (executor, model, control) = fixture::<B>(true);
    // Before admission creates its pool, set the real physical allocation
    // budget. Each sequence still has a legal two-block table / 32-token limit.
    model.lock().runtime_env.kv_max_blocks = Some(3);
    let prompts: [Vec<TokenId>; 2] = [
        (1..=16).map(TokenId).collect(),
        (17..=32).map(TokenId).collect(),
    ];
    let mut frontiers = Vec::new();
    for prompt in &prompts {
        frontiers.push(admit(&executor, &RequestId::new(), prompt).await);
    }
    assert_eq!(free_blocks(&model), 1);
    let full = LogitsReturnPolicy::FullLogits;
    let prefill: Vec<_> = frontiers
        .iter()
        .map(|frontier| {
            ExecutionWork::new(
                frontier,
                TokenWork::Prefill(PrefillChunk::new(0, 16, 16).unwrap()),
                &full,
            )
            .unwrap()
        })
        .collect();
    let rows = completed(executor.execute_tokens(&prefill).await.unwrap());
    assert_eq!(rows.len(), 2);
    for (index, row) in rows.iter().enumerate() {
        let tokens: Vec<_> = prompts[index].iter().map(|token| token.get()).collect();
        let expected = reference.prefill_internal(&format!("reference-{index}"), &tokens);
        compare_logits(full_logits(row), &expected);
    }
    drop(prefill);
    frontiers = rows
        .into_iter()
        .map(|row| row.next_frontier().clone())
        .collect();
    for frontier in &frontiers {
        assert_live(&executor, &model, frontier, 16, 1);
    }

    let work: Vec<_> = frontiers
        .iter()
        .enumerate()
        .map(|(index, frontier)| {
            ExecutionWork::new(
                frontier,
                TokenWork::Decode {
                    token: TokenId(40 + index as u32),
                    position: 16,
                },
                &full,
            )
            .unwrap()
        })
        .collect();
    let whole = capacity_deferral(executor.execute_tokens(&work).await.unwrap());
    assert_eq!(whole.required_blocks(), 2);
    assert_eq!(whole.snapshot().block_size, 16);
    assert_eq!(whole.snapshot().total_blocks, 3);
    assert_eq!(whole.snapshot().free_blocks, 1);
    assert_eq!(control.batches.load(Ordering::Relaxed), 2);
    assert_eq!(free_blocks(&model), 1);
    for frontier in &frontiers {
        assert_live(&executor, &model, frontier, 16, 1);
    }

    let a = completed(executor.execute_tokens(&work[..1]).await.unwrap()).remove(0);
    let expected_a = reference.decode_internal("reference-0", 40, 16);
    compare_logits(full_logits(&a), &expected_a);
    assert_live(&executor, &model, a.next_frontier(), 17, 2);
    assert_live(&executor, &model, &frontiers[1], 16, 1);
    assert_eq!(free_blocks(&model), 0);

    let b_deferred = capacity_deferral(executor.execute_tokens(&work[1..]).await.unwrap());
    assert_eq!(b_deferred.required_blocks(), 1);
    assert_eq!(b_deferred.snapshot().free_blocks, 0);
    assert_eq!(b_deferred.domain(), whole.domain());
    assert_live(&executor, &model, a.next_frontier(), 17, 2);
    assert_live(&executor, &model, &frontiers[1], 16, 1);

    let candidates = [a.next_frontier().sequence().clone()];
    let observation = executor
        .observe_token_allocator_reclaim(&frontiers[1], b_deferred.domain(), &candidates)
        .unwrap()
        .unwrap();
    assert_eq!(observation.snapshot.free_blocks, 0);
    assert_eq!(observation.reclaimable_blocks, 2);
    assert_eq!(observation.contributing_candidates, vec![0]);
    assert_eq!(
        free_blocks(&model),
        0,
        "observation cannot free or reserve blocks"
    );
    assert!(executor
        .observe_token_allocator_reclaim(
            &frontiers[1],
            b_deferred.domain(),
            &[candidates[0].clone(), candidates[0].clone()],
        )
        .is_err());
    assert!(executor
        .observe_token_allocator_reclaim(
            &frontiers[1],
            b_deferred.domain(),
            &[frontiers[1].sequence().clone()],
        )
        .is_err());

    // Foreign identities/domains come from another actual admitted model,
    // rather than a forged number or a manufactured resource authority.
    let (foreign_executor, foreign_model, _) = fixture::<B>(true);
    let foreign = admit(&foreign_executor, &RequestId::new(), &prompts[0]).await;
    let foreign_domain = foreign_model
        .lock()
        .paged_block_alloc
        .as_ref()
        .unwrap()
        .lock()
        .unwrap()
        .capacity_domain()
        .clone();
    assert_ne!(&foreign_domain, b_deferred.domain());
    assert!(executor
        .observe_token_allocator_reclaim(&frontiers[1], &foreign_domain, &candidates,)
        .unwrap()
        .is_none());
    assert!(executor
        .observe_token_allocator_reclaim(&foreign, b_deferred.domain(), &candidates,)
        .is_err());
    assert!(executor
        .observe_token_allocator_reclaim(
            &frontiers[1],
            b_deferred.domain(),
            &[foreign.sequence().clone()],
        )
        .is_err());
    assert!(
        terminal(foreign_executor.cancel_tokens(foreign.sequence()).unwrap())
            .await
            .error
            .is_none()
    );
    assert!(foreign_model.lock().kv_caches.is_empty());

    let ended_a = terminal(
        executor
            .cancel_tokens(a.next_frontier().sequence())
            .unwrap(),
    )
    .await;
    assert_eq!(ended_a.ended.sequence(), a.next_frontier().sequence());
    assert!(ended_a.error.is_none());
    assert_eq!(free_blocks(&model), 2);
    let after_close = executor
        .observe_token_allocator_reclaim(&frontiers[1], b_deferred.domain(), &candidates)
        .unwrap()
        .unwrap();
    assert_eq!(after_close.snapshot.free_blocks, 2);
    assert_eq!(after_close.reclaimable_blocks, 0);
    assert!(after_close.contributing_candidates.is_empty());

    let b = completed(executor.execute_tokens(&work[1..]).await.unwrap()).remove(0);
    compare_logits(
        full_logits(&b),
        &reference.decode_internal("reference-1", 41, 16),
    );
    assert_live(&executor, &model, b.next_frontier(), 17, 2);
    assert_eq!(free_blocks(&model), 1);
    assert!(a.next_frontier().sequence().observed_end().is_some());
    // One further full-vocabulary continuation detects corrupted/replayed KV,
    // rather than treating a token id or allocation counter as a numeric oracle.
    let next = [ExecutionWork::new(
        b.next_frontier(),
        TokenWork::Decode {
            token: TokenId(42),
            position: 17,
        },
        &full,
    )
    .unwrap()];
    let continued = completed(executor.execute_tokens(&next).await.unwrap()).remove(0);
    compare_logits(
        full_logits(&continued),
        &reference.decode_internal("reference-1", 42, 17),
    );
    assert_live(&executor, &model, continued.next_frontier(), 18, 2);
    assert_eq!(control.batches.load(Ordering::Relaxed), 6);
    let ended_b = terminal(
        executor
            .cancel_tokens(continued.next_frontier().sequence())
            .unwrap(),
    )
    .await;
    assert_eq!(
        ended_b.ended.sequence(),
        continued.next_frontier().sequence()
    );
    assert!(ended_b.error.is_none());
    assert_eq!(free_blocks(&model), 3);
    assert!(model.lock().kv_caches.is_empty());
    assert!(executor.token_sequences.lock().live.is_empty());
    reference.release("reference-0");
    reference.release("reference-1");
}

#[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
#[tokio::test]
#[ignore = "requires an actual Metal device"]
async fn managed_worker_real_metal_allocator_capacity_boundary() {
    real_capacity_boundary::<ferrum_kernels::backend::metal::MetalBackend>().await;
}

#[cfg(feature = "cuda")]
#[tokio::test]
#[ignore = "requires an actual CUDA device"]
async fn managed_worker_real_cuda_allocator_capacity_boundary() {
    real_capacity_boundary::<ferrum_kernels::backend::cuda::CudaBackend>().await;
}
