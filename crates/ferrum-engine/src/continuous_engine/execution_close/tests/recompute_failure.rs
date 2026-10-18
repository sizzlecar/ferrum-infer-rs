//! Actual scheduler/registry closure with a controlled late observer error.
//! The failing executor retains a real CPU Llama KV owner until the test
//! explicitly closes it; peer scheduling and fence observations are controlled.
use super::*;
use ferrum_models::{test_support::tiny_llama_model, DecoderOnlyLLM};
use ferrum_types::AllocatorCapacityDomain;

fn attach_clients(
    engine: &ContinuousBatchEngine,
    request_id: &RequestId,
) -> (
    tokio::sync::oneshot::Receiver<Result<InferenceResponse>>,
    tokio::sync::mpsc::Receiver<Result<StreamChunk>>,
) {
    let (response, receiver) = tokio::sync::oneshot::channel();
    let (stream, updates) = tokio::sync::mpsc::channel(4);
    let mut sequences = engine.inner.sequences.write();
    let sequence = sequences.get_mut(request_id).unwrap();
    sequence.response_sender = Some(response);
    sequence.stream_sender = Some(stream);
    (receiver, updates)
}

async fn install_controlled_request(
    engine: &ContinuousBatchEngine,
    executor: &CloseProbe,
) -> (RequestId, ExecutorSequence) {
    let mut request = InferenceRequest::new("peer", "tiny-llama-test");
    request
        .metadata
        .insert(PROMPT_TOKENS_METADATA_KEY.into(), serde_json::json!(1));
    let id = request.id.clone();
    engine
        .inner
        .scheduler
        .submit(request.clone())
        .await
        .unwrap();
    let tick = engine
        .inner
        .scheduler
        .prepare_token_admission_observed(
            1,
            None,
            &mut |request| {
                ferrum_scheduler::vnext::AdmissionProbeOutcome::Admitted(
                    ExecutorPrefillAdmissionReceipt {
                        request_id: request.id.clone(),
                    },
                )
            },
            &mut |_| {},
        )
        .unwrap();
    assert_eq!(tick.admitted(), 1);
    engine.inner.scheduler.mark_prefill_complete(&id, 1);
    let identity = executor.issuer.issue_sequence().unwrap();
    let mut state = SequenceState::new(request, vec![TokenId(1)]);
    state.execution_frontier = Some(executor.issuer.frontier(&identity, 1).unwrap());
    state.prefill_complete = true;
    state.prefill_tokens_processed = 1;
    state.generated_tokens.push(TokenId(2));
    state.phase = RequestPhase::Decoding;
    let mut slot = RequestSlotLease::open(&engine.inner, id.clone());
    slot.admit(&engine.inner);
    state.request_slot = Some(slot);
    engine.inner.sequences.write().insert(id.clone(), state);
    (id, identity)
}

async fn assert_client_failed(
    response: tokio::sync::oneshot::Receiver<Result<InferenceResponse>>,
    mut stream: tokio::sync::mpsc::Receiver<Result<StreamChunk>>,
) {
    let error = response.await.unwrap().unwrap_err();
    assert!(error
        .to_string()
        .contains("late terminal observation failed"));
    let error = stream.recv().await.unwrap().unwrap_err();
    assert!(error
        .to_string()
        .contains("late terminal observation failed"));
    assert!(stream.recv().await.is_none());
}

#[tokio::test]
async fn execution_close_late_recompute_failure_ends_products_but_retains_real_cpu_owner() {
    for allocator in [false, true] {
        let (engine, executor, identity, request_id, slot, sender) = fixture(false);
        let cfg = TinyLlamaConfig::default();
        let mut model = tiny_llama_model(&cfg);
        let actual = model.prefill("retained-owner", &[1]);
        let expected = tiny_llama_model(&cfg).prefill("reference", &[1]);
        assert_eq!(actual, expected);
        assert_eq!(actual.len(), cfg.vocab_size);
        *executor.retained_cpu_model.lock() = Some(model);
        let original =
            install_recompute_request(&engine, &executor, &identity, &request_id, slot).await;
        let mut clients = vec![attach_clients(&engine, &request_id)];
        let mut prior_yields = 0;
        let transaction = if allocator {
            // Replace only the not-yet-armed native diagnostic plan. This
            // constructs the same real two-request allocator handoff used by
            // the product pressure path, without claiming CPU paged attention.
            engine
                .inner
                .scheduler
                .abort_execution_capacity_yield(&original, false, 1, None)
                .unwrap();
            let (prior, prior_identity) = install_controlled_request(&engine, &executor).await;
            clients.push(attach_clients(&engine, &prior));
            executor.immediate_ends.lock().insert(prior_identity);
            let (owner, owner_identity) = install_controlled_request(&engine, &executor).await;
            clients.push(attach_clients(&engine, &owner));
            executor.immediate_ends.lock().insert(owner_identity);
            let domain = AllocatorCapacityDomain::new();
            for peer in [&request_id, &prior] {
                let peer_pending = ExecutionCapacityReleaseSnapshot::default()
                    .with_allocator(domain.clone(), peer.clone(), vec![], true)
                    .unwrap();
                assert!(matches!(
                    engine
                        .inner
                        .scheduler
                        .defer_for_allocator_capacity(peer, domain.clone(), 0, &peer_pending,)
                        .unwrap(),
                    ExecutionCapacityAction::Deferred { count: 1 }
                ));
            }
            let first_release = ExecutionCapacityReleaseSnapshot::default()
                .with_allocator(domain.clone(), owner.clone(), vec![prior.clone()], false)
                .unwrap();
            let ExecutionCapacityAction::YieldPlanned { transaction: first } = engine
                .inner
                .scheduler
                .defer_for_allocator_capacity(&owner, domain.clone(), 0, &first_release)
                .unwrap()
            else {
                panic!("the first exact contributor must yield");
            };
            assert_eq!(first.victim_request_id(), &prior);
            assert!(!engine
                .inner
                .execute_capacity_yield(&first, 3, None)
                .await
                .unwrap());
            prior_yields = 1;
            assert_eq!(
                engine.inner.scheduler.trace_phase(&prior),
                Some(RequestPhase::Waiting)
            );
            assert!(engine.inner.sequences.read()[&prior]
                .execution_frontier
                .is_none());
            let tick = engine
                .inner
                .scheduler
                .prepare_token_admission_observed(
                    3,
                    None,
                    &mut |_| panic!("the previous contributor remains held"),
                    &mut |_| {},
                )
                .unwrap();
            assert_eq!(tick.admitted(), 0);
            let release = ExecutionCapacityReleaseSnapshot::default()
                .with_allocator(
                    domain.clone(),
                    owner.clone(),
                    vec![request_id.clone()],
                    false,
                )
                .unwrap();
            let ExecutionCapacityAction::YieldPlanned { transaction } = engine
                .inner
                .scheduler
                .defer_for_allocator_capacity(&owner, domain, 1, &release)
                .unwrap()
            else {
                panic!("allocator owner requires the blocked peer's release");
            };
            assert_eq!(transaction.episode_id(), first.episode_id());
            assert_eq!(transaction.victim_request_id(), &request_id);
            transaction
        } else {
            original
        };
        assert!(!engine
            .inner
            .execute_capacity_yield(&transaction, 2, None)
            .await
            .unwrap());
        assert_eq!(executor.calls.load(Ordering::SeqCst), 1 + prior_yields);
        assert_eq!(
            engine.inner.scheduler.confirmed_capacity_release_epoch(),
            prior_yields as u64
        );
        sender
            .send(Err(FerrumError::backend(
                "late terminal observation failed",
            )))
            .unwrap();
        engine.inner.execution_closes.join_observers().await;
        assert!(identity.observed_end().is_none());
        assert!(engine.inner.drain_execution_closes().is_err());
        assert_eq!(
            engine
                .inner
                .scheduler
                .trace_snapshot()
                .pressure_pending_release_fences,
            1
        );
        // The old entry keeps pending notifications even if this drain's
        // diagnostic error is consumed before the next iteration.
        engine.inner.drain_execution_closes().unwrap();
        engine
            .inner
            .complete_failed_execution_recomputes()
            .await
            .unwrap();
        for (response, stream) in clients {
            assert_client_failed(response, stream).await;
        }
        assert!(engine.inner.sequences.read().is_empty());
        assert_eq!(engine.inner.scheduler.active_count(), 0);
        assert_eq!(engine.inner.scheduler.waiting_count(), 0);
        assert_eq!(engine.inner.execution_closes.unreclaimed_count(), 1);
        assert_eq!(
            engine
                .inner
                .scheduler
                .trace_snapshot()
                .pressure_pending_release_fences,
            1
        );
        // The healthy controlled peer's genuine Ended is the only possible
        // release notification. The failing owner still has no terminal fact.
        let prior_releases = prior_yields as u64 + u64::from(allocator);
        assert_eq!(
            engine.inner.scheduler.confirmed_capacity_release_epoch(),
            prior_releases
        );
        assert_eq!(
            executor.calls.load(Ordering::SeqCst),
            1 + prior_yields + usize::from(allocator)
        );
        {
            let entries = engine.inner.execution_closes.state.lock();
            let entry = &entries.entries[&identity];
            assert!(entry.slot.as_ref().unwrap().armed);
            assert!(matches!(
                entry.purpose,
                ClosePurpose::Finalize { recompute: Some(_) }
            ));
        }
        // Real numerical continuation proves the test's original model KV was
        // retained through the failed observation and product termination.
        {
            let mut owner = executor.retained_cpu_model.lock();
            let owner = owner.as_mut().unwrap();
            let actual = owner.decode("retained-owner", 2, 1);
            let mut reference = tiny_llama_model(&cfg);
            reference.prefill("reference", &[1]);
            let expected = reference.decode("reference", 2, 1);
            assert_eq!(actual, expected);
            assert_eq!(actual.len(), cfg.vocab_size);
            assert!(owner.kv_caches["retained-owner"]
                .iter()
                .all(|cache| cache.len == 2));
            owner.close_token_sequence("retained-owner").unwrap();
            assert!(owner.kv_caches.is_empty());
        }
        executor.issuer.ended(&identity).unwrap();
        engine.inner.drain_execution_closes().unwrap();
        assert_eq!(engine.inner.execution_closes.unreclaimed_count(), 0);
        assert_eq!(
            engine
                .inner
                .scheduler
                .trace_snapshot()
                .pressure_pending_release_fences,
            0
        );
        assert_eq!(
            engine.inner.scheduler.confirmed_capacity_release_epoch(),
            prior_releases + 1
        );
        engine.inner.drain_execution_closes().unwrap();
        assert_eq!(
            engine.inner.scheduler.confirmed_capacity_release_epoch(),
            prior_releases + 1
        );
        assert!(engine.inner.sequences.read().is_empty());
    }
}
