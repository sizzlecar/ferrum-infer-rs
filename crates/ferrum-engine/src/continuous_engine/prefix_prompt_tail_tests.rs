//! Controlled executor protocol checks, not model numerical validation.

use super::*;

#[tokio::test]
async fn prompt_tail_single_prefill_samples_only_after_the_remaining_suffix() {
    for reserve in [None, Some(1), Some(2), Some(4)] {
        let trace_path = resource_trace_temp_path("prompt-tail-planning");
        let mut config = EngineConfig::default();
        config.scheduler.max_running_requests = 1;
        config.runtime.scheduler_trace_jsonl = Some(trace_path.clone());
        let scheduler = Arc::new(ContinuousBatchScheduler::new(config.scheduler.clone()));
        let mut executor = PlanRuntimeChunkedPrefillTestExecutor::new(false);
        executor.prompt_tail_reserve = reserve;
        let observation = executor.observer();
        let executor = ExecutorOwner::new(Box::new(executor));
        let engine = ContinuousBatchEngine::new(
            config,
            Arc::clone(&scheduler),
            Arc::new(ferrum_testkit::MockTokenizer::new(128)),
            Arc::new(ferrum_testkit::MockSampler),
            executor,
        )
        .unwrap();
        let executor = observation;
        let mut request = policy_request();
        request.prompt = "one two three four".to_owned();
        request.sampling_params.max_tokens = 1;
        let request_id = request.id.clone();
        let response = tokio::time::timeout(Duration::from_secs(2), engine.infer(request))
            .await
            .expect("planned tail must finish without a capacity wake")
            .unwrap();
        assert_eq!(response.response().finish_reason, FinishReason::Length);
        assert_eq!(response.response().tokens.len(), 1);
        let expected = match reserve {
            Some(tail) => vec![
                PrefillChunk::new(0, 5 - tail, 5).unwrap(),
                PrefillChunk::new(5 - tail, tail, 5).unwrap(),
            ],
            None => vec![PrefillChunk::new(0, 5, 5).unwrap()],
        };
        assert_eq!(*executor.attempted_chunks.lock().unwrap(), expected);
        assert_eq!(
            executor.capacity_wait_registrations.load(Ordering::Relaxed),
            0
        );
        assert_eq!(
            scheduler
                .trace_snapshot()
                .execution_capacity_blocked_prefill_len,
            0
        );
        assert!(executor.retained.lock().unwrap().is_empty());
        flush_engine_profile_events(&engine);
        let events = read_engine_profile_events(&trace_path);
        let boundaries = events
            .iter()
            .filter(|event| event.phase == "vnext.prefill_prompt_tail_boundary_planned")
            .map(|event| {
                assert_eq!(event.request_id, request_id.to_string());
                (
                    event.shape["tokens_processed"].as_u64().unwrap(),
                    event.shape["scheduled_tokens"].as_u64().unwrap(),
                    event.shape["planned_tokens"].as_u64().unwrap(),
                    event.shape["capture_boundary"].as_u64().unwrap(),
                )
            })
            .collect::<Vec<_>>();
        assert_eq!(
            boundaries,
            reserve
                .map(|tail| vec![(0, 5, (5 - tail) as u64, (5 - tail) as u64)])
                .unwrap_or_default()
        );
        engine.shutdown().await.unwrap();
        std::fs::remove_file(trace_path).unwrap();
    }
}

#[tokio::test]
async fn prompt_tail_time_planning_matches_submitted_ranges_without_capacity_feedback() {
    use super::cost_feedback_tests::Source;
    use crate::continuous_engine::attempt_observation_tests::AttemptRecorder;
    use ferrum_execution::execution_attempt::ExecutionAttemptEligibility;
    use ferrum_scheduler::cost_prediction::CostWork;

    let mut baseline = None;
    for enabled in [false, true] {
        let mut config = EngineConfig::default();
        config.runtime.profile_detail = ObservabilityProfileDetail::Off;
        config.scheduler.max_running_requests = 1;
        if enabled {
            config.scheduler.slo = Some("ttft:1000,tpot:100,itl:100".parse().unwrap());
        }
        let source = Source::new(true);
        let recorder = Arc::new(AttemptRecorder::default());
        let scheduler = Arc::new(ContinuousBatchScheduler::new(config.scheduler.clone()));
        let mut implementation = PlanRuntimeChunkedPrefillTestExecutor::new(false);
        implementation.prompt_tail_reserve = Some(4);
        let observation = implementation.observer();
        let engine = ContinuousBatchEngine::new_observed(
            config,
            scheduler.clone(),
            Arc::new(ferrum_testkit::MockTokenizer::new(128)),
            Arc::new(ferrum_testkit::MockSampler),
            ExecutorOwner::new(Box::new(implementation)),
            None,
            Some(recorder.clone()),
            enabled.then(|| source.clone() as Arc<dyn CostSnapshotSource>),
        )
        .unwrap();
        let mut request = policy_request();
        request.prompt = "one two three four".into();
        request.sampling_params.max_tokens = 1;
        let response = tokio::time::timeout(Duration::from_secs(2), engine.infer(request))
            .await
            .unwrap()
            .unwrap();
        let output = (
            response.response().tokens.clone(),
            response.response().text.clone(),
            response.response().finish_reason.clone(),
        );
        if let Some(baseline) = &baseline {
            assert_eq!(&output, baseline);
        } else {
            baseline = Some(output);
        }
        assert_eq!(
            *observation.attempted_chunks.lock().unwrap(),
            [
                PrefillChunk::new(0, 1, 5).unwrap(),
                PrefillChunk::new(1, 4, 5).unwrap(),
            ],
            "a planned one-token boundary must not impose a one-token capacity ceiling"
        );
        let summaries = recorder.take();
        assert_eq!(summaries.len(), 2);
        let actual = summaries
            .iter()
            .map(|summary| {
                assert_eq!(summary.eligibility, ExecutionAttemptEligibility::Completed);
                assert_eq!(summary.participants.len(), 1);
                let participant = &summary.participants[0];
                CostWork {
                    kind: participant.kind,
                    requested: participant.requested,
                    context_tokens: participant.context_tokens,
                    prefix_restore: participant.prefix_restore,
                    internally_reused_tokens: participant.internally_reused_tokens,
                }
            })
            .collect::<Vec<_>>();
        let queries = source.take_work_queries();
        if enabled {
            assert!(source.load_count() >= 2);
            // Admission candidates need not execute, but every actual selected
            // shape must have been priced, including the unfragmented suffix.
            for work in actual {
                assert!(queries.iter().any(|(_, candidate)| candidate == &[work]));
            }
            for (_, candidate) in queries {
                assert_eq!(candidate.len(), 1);
                let range = candidate[0].requested;
                assert!(
                    (range.start() == 0 && range.end() == 1)
                        || (range.start() == 1 && range.end() == 5)
                );
            }
            assert_eq!(
                serde_json::to_value(scheduler.slo_snapshot().unwrap()).unwrap()["shrunk_prefills"],
                0
            );
        } else {
            assert_eq!(source.load_count(), 0);
            assert!(queries.is_empty());
        }
        assert_eq!(
            observation
                .capacity_wait_registrations
                .load(Ordering::Relaxed),
            0
        );
        engine.shutdown().await.unwrap();
        engine.wait_for_shutdown_completion().await.unwrap();
        assert!(observation.retained.lock().unwrap().is_empty());
        assert!(observation.tokens.0.upgrade().is_none());
    }
}

async fn yield_original_capacity_owner(
    engine: &ContinuousBatchEngine,
    observation: &PlanRuntimeChunkedPrefillObservation,
    id: &RequestId,
    progress_peer: Option<&RequestId>,
) -> ExecutorSequence {
    use ferrum_runtime::CapacityWaitCondition;
    let inner = engine.active_inner().unwrap();
    let original = observation.tokens.frontier_for_request(id).unwrap();
    let mut availability = Vec::new();
    let epochs = inner
        .model_executor
        .write_execution_capacity_snapshot(&mut availability)
        .unwrap()
        .unwrap();
    let condition =
        CapacityWaitCondition::from_observation(epochs.coordinator_id.get(), availability).unwrap();
    let release = inner.execution_capacity_release_snapshot().unwrap();
    let deferral = AdmissionDeferral::new(
        DeferredAction::WaitForRelease,
        AdmissionWakeEpochs::new(
            epochs.coordinator_id,
            epochs.release_epoch,
            epochs.capacity_epoch,
            0,
        ),
        condition,
    );
    if let Some(peer) = progress_peer {
        assert_eq!(
            engine.scheduler.trace_phase(peer),
            Some(RequestPhase::Decoding)
        );
        assert_eq!(
            engine
                .scheduler
                .defer_decode_for_execution_capacity(
                    std::slice::from_ref(peer),
                    deferral.clone(),
                    &release,
                )
                .unwrap(),
            ExecutionCapacityAction::Deferred { count: 1 }
        );
    }
    let action = match engine.scheduler.trace_phase(id).unwrap() {
        RequestPhase::Prefilling => engine
            .scheduler
            .defer_prefill_for_execution_capacity(id, deferral, &release),
        RequestPhase::Decoding => engine.scheduler.defer_decode_for_execution_capacity(
            std::slice::from_ref(id),
            deferral,
            &release,
        ),
        phase => panic!("capacity fixture requires an active phase, got {phase:?}"),
    }
    .unwrap();
    let ExecutionCapacityAction::YieldPlanned { transaction } = action else {
        panic!("the blocked source must yield its actual incarnation, got {action:?}");
    };
    assert_eq!(transaction.victim_request_id(), id);
    assert_eq!(transaction.progress_owner_id(), progress_peer.unwrap_or(id));
    engine
        .scheduler
        .arm_execution_capacity_yield(&transaction)
        .unwrap();
    let ticket = inner
        .begin_execution_recompute(transaction, 1, None)
        .unwrap();
    ticket.initial_result().await.unwrap();
    inner.execution_closes.join_observers().await;
    assert!(original.sequence().observed_end().is_some());
    // This controlled backend advances its original source only after the
    // actual close observer confirms Ended. No unproved release is announced.
    observation.publish_release();
    inner.drain_execution_closes().unwrap();
    if let Some(peer) = progress_peer {
        // A prefill frontier is a peer victim, never a no-progress self
        // recompute. Retire the actual progress owner before readmitting it.
        let peer_sequence = observation.tokens.frontier_for_request(peer).unwrap();
        inner
            .complete_request_for_test(peer, FinishReason::Cancelled)
            .await
            .unwrap();
        inner.output_tasks.join_current().await;
        inner.execution_closes.join_observers().await;
        inner.drain_execution_closes().unwrap();
        assert!(peer_sequence.sequence().observed_end().is_some());
        assert!(observation.tokens.frontier_for_request(peer).is_none());
    }
    assert_eq!(
        engine.scheduler.trace_phase(id),
        Some(RequestPhase::Waiting)
    );
    original.sequence().clone()
}

#[tokio::test]
async fn prompt_tail_capacity_requeue_distinguishes_before_and_after_first_output() {
    use super::cost_feedback_tests::{submit, Source};
    use crate::continuous_engine::attempt_observation_tests::AttemptRecorder;

    let mut config = EngineConfig::default();
    config.runtime.profile_detail = ObservabilityProfileDetail::Off;
    config.scheduler.max_running_requests = 2;
    config.scheduler.prefill_step_chunk = Some(2);
    config.scheduler.slo = Some("ttft:1000,tpot:100,itl:100".parse().unwrap());
    let source = Source::new(true);
    let recorder = Arc::new(AttemptRecorder::default());
    let tokenizer: Arc<dyn Tokenizer + Send + Sync> =
        Arc::new(ferrum_testkit::MockTokenizer::new(128));
    let mut implementation = PlanRuntimeChunkedPrefillTestExecutor::new(false);
    implementation.prompt_tail_reserve = Some(2);
    implementation.releases_sequence_capacity = true;
    let observation = implementation.observer();
    let engine = ContinuousBatchEngine::new_observed(
        config.clone(),
        Arc::new(ContinuousBatchScheduler::new(config.scheduler)),
        tokenizer.clone(),
        Arc::new(ferrum_testkit::MockSampler),
        ExecutorOwner::new(Box::new(implementation)),
        None,
        Some(recorder.clone()),
        Some(source.clone()),
    )
    .unwrap();
    let id = submit(&engine, &tokenizer, "one two three four").await;
    let peer = submit(&engine, &tokenizer, "peer").await;
    let inner = engine.active_inner().unwrap();
    inner.run_iteration().await.unwrap();
    assert_eq!(
        observation
            .tokens
            .frontier_for_request(&id)
            .unwrap()
            .consumed_tokens(),
        2
    );
    assert!(inner
        .sequences
        .get(&id)
        .unwrap()
        .lock()
        .as_ref()
        .unwrap()
        .generated_tokens
        .is_empty());
    assert_eq!(
        inner
            .sequences
            .get(&peer)
            .unwrap()
            .lock()
            .as_ref()
            .unwrap()
            .generated_tokens
            .len(),
        1
    );
    let original = yield_original_capacity_owner(&engine, &observation, &id, Some(&peer)).await;
    {
        let handle = inner.sequences.get(&id).unwrap();
        let state = handle.lock();
        let sequence = state.as_ref().unwrap();
        assert!(sequence.execution_frontier.is_none());
        assert!(sequence.generated_tokens.is_empty());
        assert_eq!(sequence.prefill_tokens_processed, 0);
    }
    drop(source.take_work_queries());
    drop(recorder.take());
    inner.run_iteration().await.unwrap();
    let replacement = observation.tokens.frontier_for_request(&id).unwrap();
    assert_ne!(replacement.sequence(), &original);
    assert_eq!(replacement.consumed_tokens(), 2);
    assert!(replacement.sequence().observed_end().is_none());
    assert!(
        source.take_work_queries().is_empty(),
        "the recompute without resolved scheduler context must remain Unknown"
    );
    let samples = recorder.take();
    assert_eq!(samples.len(), 1);
    assert_eq!(samples[0].participants[0].requested.as_range(), 0..2);
    let statistics = serde_json::to_value(engine.scheduler.slo_snapshot().unwrap()).unwrap();
    assert!(statistics["static_fallbacks"].as_u64().unwrap() > 0);
    inner.run_iteration().await.unwrap();
    let resumed = observation.tokens.frontier_for_request(&id).unwrap();
    assert_eq!(resumed.sequence(), replacement.sequence());
    assert_eq!(resumed.consumed_tokens(), 3);
    assert!(
        source
            .take_work_queries()
            .iter()
            .any(|(_, work)| work.len() == 1 && work[0].requested.as_range() == (2..3)),
        "resolved context after requeue still belongs to the unsampled original prompt"
    );
    inner.run_iteration().await.unwrap();
    let final_frontier = observation.tokens.frontier_for_request(&id).unwrap();
    assert_eq!(final_frontier.sequence(), replacement.sequence());
    assert_eq!(final_frontier.consumed_tokens(), 5);
    assert_eq!(
        observation
            .attempted_chunks
            .lock()
            .unwrap()
            .iter()
            .copied()
            .filter(|chunk| chunk.total_prompt_tokens() == 5)
            .collect::<Vec<_>>(),
        [
            PrefillChunk::new(0, 2, 5).unwrap(),
            PrefillChunk::new(0, 2, 5).unwrap(),
            PrefillChunk::new(2, 1, 5).unwrap(),
            PrefillChunk::new(3, 2, 5).unwrap(),
        ]
    );
    assert_eq!(
        inner
            .sequences
            .get(&id)
            .unwrap()
            .lock()
            .as_ref()
            .unwrap()
            .generated_tokens
            .len(),
        1
    );

    // A real decode commits a second generated token. After this owner's
    // retirement, replaying P+G=7 is not an original prompt-tail candidate.
    inner.run_iteration().await.unwrap();
    assert_eq!(
        inner
            .sequences
            .get(&id)
            .unwrap()
            .lock()
            .as_ref()
            .unwrap()
            .generated_tokens
            .len(),
        2
    );
    let emitted_owner = yield_original_capacity_owner(&engine, &observation, &id, None).await;
    assert_eq!(&emitted_owner, replacement.sequence());
    drop(source.take_work_queries());
    drop(recorder.take());
    for expected in [2, 4, 6] {
        inner.run_iteration().await.unwrap();
        assert_eq!(
            observation
                .tokens
                .frontier_for_request(&id)
                .unwrap()
                .consumed_tokens(),
            expected
        );
    }
    let replay = observation.tokens.frontier_for_request(&id).unwrap();
    assert_ne!(replay.sequence(), &emitted_owner);
    let queries = source.take_work_queries();
    assert!(
        queries.iter().any(|(_, work)| work.len() == 1
            && work[0].context_tokens == 7
            && work[0].requested.as_range() == (4..6)),
        "actual prior output must prevent a spurious prompt-tail cut at token 5"
    );
    inner.run_iteration().await.unwrap();
    inner.output_tasks.join_current().await;
    inner.execution_closes.join_observers().await;
    inner.drain_execution_closes().unwrap();
    assert_eq!(
        observation
            .attempted_chunks
            .lock()
            .unwrap()
            .iter()
            .copied()
            .filter(|chunk| chunk.total_prompt_tokens() == 7)
            .collect::<Vec<_>>(),
        [
            PrefillChunk::new(0, 2, 7).unwrap(),
            PrefillChunk::new(2, 2, 7).unwrap(),
            PrefillChunk::new(4, 2, 7).unwrap(),
            PrefillChunk::new(6, 1, 7).unwrap(),
        ]
    );
    assert!(replay.sequence().observed_end().is_some());
    drop(inner);
    engine.shutdown().await.unwrap();
    engine.wait_for_shutdown_completion().await.unwrap();
    assert!(replacement.sequence().observed_end().is_some());
    assert!(observation.tokens.0.upgrade().is_none());
    assert!(observation.retained.lock().unwrap().is_empty());
}

#[tokio::test]
async fn prompt_tail_batch_prefill_commits_each_boundary_before_sampling() {
    let mut config = EngineConfig::default();
    config.scheduler.max_running_requests = 3;
    let scheduler = Arc::new(ContinuousBatchScheduler::new(config.scheduler.clone()));
    let tokenizer: Arc<dyn Tokenizer + Send + Sync> =
        Arc::new(ferrum_testkit::MockTokenizer::new(128));
    let mut executor = PlanRuntimeChunkedPrefillTestExecutor::new(false);
    executor.prompt_tail_reserve = Some(2);
    let observation = executor.observer();
    let executor = ExecutorOwner::new(Box::new(executor));
    let engine = ContinuousBatchEngine::new(
        config,
        Arc::clone(&scheduler),
        Arc::clone(&tokenizer),
        Arc::new(ferrum_testkit::MockSampler),
        executor,
    )
    .unwrap();
    let executor = observation;
    let mut request_ids = Vec::new();
    for prompt in [
        "one two three four",
        "five six seven eight",
        "nine ten eleven twelve",
    ] {
        let mut request = policy_request();
        request.prompt = prompt.to_owned();
        request.sampling_params.max_tokens = 2;
        let tokens = tokenizer.encode(prompt, true).unwrap();
        request.metadata.insert(
            PROMPT_TOKENS_METADATA_KEY.to_owned(),
            serde_json::json!(tokens.len()),
        );
        scheduler.submit(request.clone()).await.unwrap();
        let mut sequence = SequenceState::new_with_tokenizer_and_model_vocab_size(
            request.clone(),
            tokens.clone(),
            Some(Arc::clone(&tokenizer)),
            Some(128),
        );
        sequence.request_slot = Some(RequestSlotLease::open(
            &engine.active_inner().expect("engine execution is open"),
            request.id.clone(),
        ));
        request_ids.push(request.id.clone());
        engine
            .active_inner()
            .expect("engine execution is open")
            .sequences
            .insert(request.id.clone(), sequence);
        assert!(
            engine
                .active_inner()
                .expect("engine execution is open")
                .probe_executor_prefill_admission_for_test(&request)
                .await
        );
        engine
            .active_inner()
            .expect("engine execution is open")
            .sequences
            .get(&request.id)
            .unwrap()
            .lock()
            .as_mut()
            .unwrap()
            .request_slot
            .as_mut()
            .unwrap()
            .admit(&engine.active_inner().expect("engine execution is open"));
    }

    let first = scheduler
        .next_batch(ferrum_interfaces::BatchHint::simple(3))
        .await
        .unwrap();
    assert_eq!(first.requests.len(), 3);
    let incarnations = request_ids
        .iter()
        .map(|id| {
            let sequences = &engine
                .active_inner()
                .expect("engine execution is open")
                .sequences;
            let frontier = sequences
                .get(id)
                .unwrap()
                .lock()
                .as_ref()
                .unwrap()
                .execution_frontier
                .clone()
                .unwrap();
            assert_eq!(frontier.consumed_tokens(), 0);
            (id.clone(), frontier.sequence().clone())
        })
        .collect::<HashMap<_, _>>();
    engine
        .active_inner()
        .expect("engine execution is open")
        .process_batch(&first)
        .await
        .unwrap();
    assert_eq!(executor.tail_batch_prefill_calls.load(Ordering::Relaxed), 1);
    {
        let sequences = &engine
            .active_inner()
            .expect("engine execution is open")
            .sequences;
        for id in &request_ids {
            let sequence_handle = sequences.get(id).unwrap();
            let sequence_guard = sequence_handle.lock();
            let sequence = sequence_guard.as_ref().unwrap();
            assert_eq!(sequence.prefill_tokens_processed, 3);
            assert!(!sequence.prefill_complete);
            assert!(sequence.generated_tokens.is_empty());
            let frontier = sequence.execution_frontier.as_ref().unwrap();
            assert_eq!(frontier.sequence(), &incarnations[id]);
            assert_eq!(frontier.consumed_tokens(), 3);
        }
    }
    assert_eq!(
        *executor.attempted_chunks.lock().unwrap(),
        vec![PrefillChunk::new(0, 3, 5).unwrap(); 3]
    );

    let last = scheduler
        .next_batch(ferrum_interfaces::BatchHint::simple(3))
        .await
        .unwrap();
    assert_eq!(last.requests.len(), 3);
    engine
        .active_inner()
        .expect("engine execution is open")
        .process_batch(&last)
        .await
        .unwrap();
    assert_eq!(executor.tail_batch_prefill_calls.load(Ordering::Relaxed), 2);
    {
        let sequences = &engine
            .active_inner()
            .expect("engine execution is open")
            .sequences;
        for id in &request_ids {
            let sequence_handle = sequences.get(id).unwrap();
            let sequence_guard = sequence_handle.lock();
            let sequence = sequence_guard.as_ref().unwrap();
            assert!(sequence.prefill_complete);
            assert_eq!(sequence.generated_tokens.len(), 1);
            let frontier = sequence.execution_frontier.as_ref().unwrap();
            assert_eq!(frontier.sequence(), &incarnations[id]);
            assert_eq!(frontier.consumed_tokens(), 5);
        }
    }
    assert_eq!(
        &executor.attempted_chunks.lock().unwrap()[3..],
        &[PrefillChunk::new(3, 2, 5).unwrap(); 3]
    );
    assert!(executor.retained.lock().unwrap().is_empty());
    assert_eq!(
        scheduler
            .trace_snapshot()
            .execution_capacity_blocked_prefill_len,
        0
    );
    for id in &request_ids {
        engine
            .active_inner()
            .expect("engine execution is open")
            .complete_request_for_test(id, FinishReason::Error)
            .await
            .unwrap();
        assert!(incarnations[id].observed_end().is_some());
    }
    engine
        .active_inner()
        .expect("engine execution is open")
        .execution_closes
        .join_observers()
        .await;
    engine
        .active_inner()
        .expect("engine execution is open")
        .drain_execution_closes()
        .unwrap();
    assert_eq!(
        engine
            .active_inner()
            .expect("engine execution is open")
            .execution_closes
            .unreclaimed_count(),
        0
    );
}
