use super::*;

async fn submit(
    engine: &ContinuousBatchEngine,
    max_tokens: usize,
) -> (RequestId, CreditedOutputSession) {
    let mut request = policy_request();
    request.stream = true;
    request.sampling_params.max_tokens = max_tokens;
    request.evidence_request.capture_prompt_token_ids = true;
    request.evidence_request.capture_engine_token_timing = true;
    let id = request.id.clone();
    let session = engine
        .infer_credited_stream(
            request,
            InferenceRequestContext::capture(),
            Arc::new(OutputProjectionContract::cli_text()),
        )
        .await
        .unwrap();
    (id, session)
}
#[tokio::test]
async fn credited_execution_evidence_keeps_full_commits_and_bounded_stage_prefix_under_lease() {
    let (engine, scheduler, _) = credited_fixture();
    let (id, mut session) = submit(&engine, 2).await;
    credited_ready(&engine, &id).await;
    let original_start = engine.inner.sequences.read()[&id].start_time;
    let prompt = engine.inner.sequences.read()[&id].input_tokens.clone();
    {
        let mut sequences = engine.inner.sequences.write();
        let seq = sequences.get_mut(&id).unwrap();
        for _ in 0..20 {
            seq.record_decode_execution(original_start, original_start);
        }
    }
    for expected in 1..=2 {
        credited_ready(&engine, &id).await;
        engine
            .inner
            .process_batch(&credited_next(&scheduler, 1).await)
            .await
            .unwrap();
        let data = credited_bound(session.frames.next()).await.unwrap();
        assert!(!data.metadata().terminal);
        assert_eq!(data.metadata().generated_tokens, expected);
        drop(data);
    }
    let terminal = credited_bound(session.frames.next()).await.unwrap();
    assert!(terminal.metadata().terminal);
    drop(terminal);
    let completion = credited_bound(session.completion).await.unwrap();
    let OutputCompletion::Succeeded {
        execution_evidence: Some(evidence),
        history: Some(history),
        usage,
        ..
    } = completion.payload()
    else {
        panic!("missing successful credited evidence")
    };
    assert_eq!(evidence.prompt_token_ids, prompt);
    assert_eq!(evidence.output_token_ids, history.tokens);
    assert_eq!(usage.completion_tokens, 2);
    let timing = evidence.engine_token_timing.as_ref().unwrap();
    timing.validate(2).unwrap();
    assert_eq!(timing.decode_stage_intervals.len(), 6);
    assert!(timing.decode_stage_intervals_omitted >= 14);
    assert!(timing.decode_stage_intervals.capacity() <= 6);
    assert!(timing.token_commit_nanos_since_request_start.capacity() <= 2);
    assert!(!engine.inner.sequences.read().contains_key(&id));
    assert!(credited_pool(&engine).snapshot().data_used.projection_bytes > 0);
    drop((completion, session.frames));
    credited_drained(&engine).await;
}
#[tokio::test]
async fn credited_execution_evidence_cancel_releases_only_after_terminal_owner() {
    let (engine, scheduler, executor) = credited_fixture();
    let (id, session) = submit(&engine, 4).await;
    credited_ready(&engine, &id).await;
    engine
        .inner
        .process_batch(&credited_next(&scheduler, 1).await)
        .await
        .unwrap();
    let charged = credited_pool(&engine).snapshot().data_used.projection_bytes;
    assert!(charged > 0);
    drop(session.frames);
    credited_wait(&engine, &id, |s| {
        matches!(s, OutputReadinessState::Closing(_))
    })
    .await;
    assert_eq!(
        credited_pool(&engine).snapshot().data_used.projection_bytes,
        charged
    );
    engine.inner.run_iteration().await.unwrap();
    let completion = credited_bound(session.completion).await.unwrap();
    assert!(matches!(completion.payload(), OutputCompletion::Failed(_)));
    assert!(!engine.inner.sequences.read().contains_key(&id));
    assert_eq!(executor.released_cache_count.load(Ordering::Relaxed), 1);
    assert_eq!(
        credited_pool(&engine).snapshot().data_used.projection_bytes,
        charged
    );
    drop(completion);
    credited_drained(&engine).await;
}
