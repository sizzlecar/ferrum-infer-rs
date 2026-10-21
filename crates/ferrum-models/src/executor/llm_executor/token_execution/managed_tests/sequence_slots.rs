use super::*;

#[tokio::test]
async fn physical_sequence_slots_worker_defers_held_owner_and_reuses_exact_retired_metadata() {
    let (executor, model, control) = fixture::<CpuBackend>(true);
    model.lock().runtime_env.paged_max_seqs = 2;
    let prompt = [TokenId(1), TokenId(2)];
    let first = admit(&executor, &RequestId::new(), &prompt).await;
    let peer = admit(&executor, &RequestId::new(), &prompt).await;
    let cache_id = executor.token_sequences.lock().live[first.sequence()]
        .cache_id
        .clone();
    let metadata = model.lock().kv_caches[&cache_id]
        .iter()
        .map(|cache| {
            (
                cache.block_table.as_ref().unwrap().as_ptr(),
                cache.context_lens.as_ref().unwrap().as_ptr(),
            )
        })
        .collect::<Vec<_>>();
    *control.held_cache.lock() = Some(cache_id.clone());
    let (seen, held) = oneshot::channel();
    *control.held_close_seen.lock() = Some(seen);
    let close = executor.cancel_tokens(first.sequence()).unwrap();
    tokio::time::timeout(Duration::from_secs(5), held)
        .await
        .unwrap()
        .unwrap();
    assert!(first.sequence().observed_end().is_none());
    let waiting = RequestId::new();
    let outcome = executor
        .admit_tokens(ExecutionAdmission {
            context: ExecutorPrefillAdmission::for_startup(&waiting, &prompt, 32),
            adapter: None,
        })
        .await
        .unwrap();
    let TokenAdmissionOutcome::SequenceSlotsDeferred(deferred) = outcome else {
        panic!("held physical metadata must not be replaced by a third slot");
    };
    assert_eq!(
        (
            deferred.capacity_slots(),
            deferred.occupied_slots(),
            deferred.required_slots()
        ),
        (2, 2, 1)
    );
    assert_eq!(executor.token_sequences.lock().live.len(), 2);
    {
        let model = model.lock();
        assert_eq!(model.kv_caches.len(), 2);
        assert!(matches!(
            model.paged_admission_capacity_snapshot().unwrap().coverage,
            ferrum_execution::model_executor::PagedAdmissionCoverage::AllPersistentState {
                max_new_sequences: 0,
                ..
            }
        ));
        assert_eq!(
            model
                .paged_block_alloc
                .as_ref()
                .unwrap()
                .lock()
                .unwrap()
                .free_count(),
            14
        );
    }

    // Retrying the original close wakes the existing worker; no new cleanup
    // lifecycle, timer or physical-release credit is manufactured by the test.
    control.held_cache.lock().take();
    let _same_close = executor.cancel_tokens(first.sequence()).unwrap();
    let ended = terminal(close).await;
    assert_eq!(ended.ended.sequence(), first.sequence());
    assert!(ended
        .error
        .unwrap()
        .to_string()
        .contains("retirement remains held"));
    assert!(matches!(
        model
            .lock()
            .paged_admission_capacity_snapshot()
            .unwrap()
            .coverage,
        ferrum_execution::model_executor::PagedAdmissionCoverage::AllPersistentState {
            max_new_sequences: 1,
            ..
        }
    ));
    let replacement = admit(&executor, &waiting, &prompt).await;
    let replacement_cache = executor.token_sequences.lock().live[replacement.sequence()]
        .cache_id
        .clone();
    let reused = model.lock().kv_caches[&replacement_cache]
        .iter()
        .map(|cache| {
            (
                cache.block_table.as_ref().unwrap().as_ptr(),
                cache.context_lens.as_ref().unwrap().as_ptr(),
            )
        })
        .collect::<Vec<_>>();
    assert_eq!(reused, metadata);
    assert_eq!(model.lock().kv_caches.len(), 2);
    assert_eq!(control.batches.load(Ordering::Relaxed), 0);
    for frontier in [peer, replacement] {
        assert!(
            terminal(executor.cancel_tokens(frontier.sequence()).unwrap())
                .await
                .error
                .is_none()
        );
    }
    assert!(executor.token_sequences.lock().live.is_empty());
}
