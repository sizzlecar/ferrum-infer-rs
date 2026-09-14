use super::*;

#[tokio::test]
async fn prompt_tail_single_prefill_samples_only_after_the_remaining_suffix() {
    for reserve in [None, Some(1), Some(2)] {
        let trace_path = resource_trace_temp_path("prompt-tail-planning");
        let mut config = EngineConfig::default();
        config.scheduler.max_running_requests = 1;
        config.runtime.scheduler_trace_jsonl = Some(trace_path.clone());
        let scheduler = Arc::new(ContinuousBatchScheduler::new(config.scheduler.clone()));
        let mut executor = PlanRuntimeChunkedPrefillTestExecutor::new(false);
        executor.prompt_tail_reserve = reserve;
        let executor = Arc::new(executor);
        let engine = ContinuousBatchEngine::new_plan_runtime(
            config,
            Arc::clone(&scheduler),
            Arc::new(ferrum_testkit::MockTokenizer::new(128)),
            Arc::new(ferrum_testkit::MockSampler),
            executor.clone(),
            Arc::new(MockTensorFactory),
        )
        .unwrap();
        let mut request = policy_request();
        request.prompt = "one two three four".to_owned();
        request.sampling_params.max_tokens = 1;
        let request_id = request.id.clone();
        let response = tokio::time::timeout(Duration::from_secs(2), engine.infer(request))
            .await
            .expect("planned tail must finish without a capacity wake")
            .unwrap();
        assert_eq!(response.finish_reason, FinishReason::Length);
        assert_eq!(response.tokens.len(), 1);
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
async fn prompt_tail_batch_prefill_commits_each_boundary_before_sampling() {
    let mut config = EngineConfig::default();
    config.scheduler.max_running_requests = 3;
    let scheduler = Arc::new(ContinuousBatchScheduler::new(config.scheduler.clone()));
    let tokenizer: Arc<dyn Tokenizer + Send + Sync> =
        Arc::new(ferrum_testkit::MockTokenizer::new(128));
    let mut executor = PlanRuntimeChunkedPrefillTestExecutor::new(false);
    executor.prompt_tail_reserve = Some(2);
    let executor = Arc::new(executor);
    let engine = ContinuousBatchEngine::new_plan_runtime(
        config,
        Arc::clone(&scheduler),
        Arc::clone(&tokenizer),
        Arc::new(ferrum_testkit::MockSampler),
        executor.clone(),
        Arc::new(MockTensorFactory),
    )
    .unwrap();
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
        let sequence = SequenceState::new_with_tokenizer_and_model_vocab_size(
            request.clone(),
            tokens.clone(),
            Some(Arc::clone(&tokenizer)),
            Some(128),
        );
        executor
            .try_admit_prefill(
                ExecutorPrefillAdmission::for_product_request(
                    &request.id,
                    &tokens,
                    sequence.model_maximum_sequence_tokens(),
                    tokens.len(),
                    0,
                )
                .unwrap(),
            )
            .unwrap();
        request_ids.push(request.id.clone());
        engine.inner.sequences.write().insert(request.id, sequence);
    }

    let first = scheduler
        .next_batch(ferrum_interfaces::BatchHint::simple(3))
        .await
        .unwrap();
    assert_eq!(first.requests.len(), 3);
    engine.inner.process_batch(&first).await.unwrap();
    assert_eq!(executor.tail_batch_prefill_calls.load(Ordering::Relaxed), 1);
    {
        let sequences = engine.inner.sequences.read();
        for id in &request_ids {
            let sequence = &sequences[id];
            assert_eq!(sequence.prefill_tokens_processed, 3);
            assert!(!sequence.prefill_complete);
            assert!(sequence.generated_tokens.is_empty());
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
    engine.inner.process_batch(&last).await.unwrap();
    assert_eq!(executor.tail_batch_prefill_calls.load(Ordering::Relaxed), 2);
    {
        let sequences = engine.inner.sequences.read();
        for id in &request_ids {
            let sequence = &sequences[id];
            assert!(sequence.prefill_complete);
            assert_eq!(sequence.generated_tokens.len(), 1);
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
}
