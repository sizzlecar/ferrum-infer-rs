//! Real engine/scheduler submissions with the credited Chat owner and codec.
use super::*;
use ferrum_types::{
    ApiChatRequest, ApiRequest, ApiStreamOptions, PROMPT_OPENED_REASONING_METADATA_KEY,
};

#[path = "credited_chat_tests/boundary.rs"]
mod boundary;

fn chat_request(max_tokens: usize, opened: bool, usage: bool) -> InferenceRequest {
    let mut request = policy_request();
    request.stream = true;
    request.sampling_params.max_tokens = max_tokens;
    if opened {
        request.prompt = "test <think>".into();
    }
    request
        .metadata
        .insert(PROMPT_OPENED_REASONING_METADATA_KEY.into(), opened.into());
    request.api_request = Some(ApiRequest::Chat(ApiChatRequest {
        messages: Vec::new(),
        tools: Vec::new(),
        tool_choice: None,
        tool_call_protocol: Default::default(),
        legacy_functions: Vec::new(),
        legacy_function_call: None,
        response_format: None,
        stream_options: Some(ApiStreamOptions {
            include_usage: Some(usage),
        }),
    }));
    request
}
fn enable_chat_events(engine: &mut ContinuousBatchEngine) {
    Arc::get_mut(&mut engine.inner)
        .unwrap()
        .config
        .scheduler
        .slo
        .output
        .max_queued_events_per_request = NonZeroUsize::new(4).unwrap();
}
async fn submit_chat(
    engine: &ContinuousBatchEngine,
    tokens: usize,
    opened: bool,
    usage: bool,
) -> (RequestId, CreditedOutputSession) {
    let request = chat_request(tokens, opened, usage);
    let id = request.id.clone();
    let session = engine
        .infer_credited_stream(
            request,
            InferenceRequestContext::capture(),
            Arc::new(OutputProjectionContract::chat_sse(
                id.to_string(),
                "wire-alias".into(),
                usage,
            )),
        )
        .await
        .unwrap();
    (id, session)
}
fn events(frame: &ferrum_interfaces::output_flow::CreditedOutputFrame) -> Vec<serde_json::Value> {
    std::str::from_utf8(frame.wire().payload())
        .unwrap()
        .split("\n\n")
        .filter_map(|event| event.strip_prefix("data: "))
        .filter(|event| *event != "[DONE]")
        .map(|event| serde_json::from_str(event).unwrap())
        .collect()
}

#[tokio::test]
async fn credited_engine_chat_terminal_token_preserves_actual_reasoning_policy_and_usage() {
    for opened in [false, true] {
        let ((mut engine, scheduler, executor), calls) = credited_counted_fixture();
        enable_chat_events(&mut engine);
        let (id, mut session) = submit_chat(&engine, 1, opened, opened).await;
        credited_ready(&engine, &id).await;
        calls.reset();
        engine
            .inner
            .process_batch(&credited_next(&scheduler, 1).await)
            .await
            .unwrap();
        assert_eq!(executor.inner.prefill_count(), 1);
        assert_eq!(executor.batch_decode_calls.load(Ordering::Relaxed), 0);
        assert!(!engine.inner.sequences.read().contains_key(&id));
        let data = credited_bound(session.frames.next()).await.unwrap();
        let channel = if opened { "reasoning" } else { "content" };
        assert_eq!(events(&data)[0]["choices"][0]["delta"][channel], "ok");
        assert_eq!(data.metadata().token, Some(TokenId::new(6)));
        assert_eq!(data.metadata().generated_tokens, 1);
        drop(data);
        let terminal = credited_bound(session.frames.next()).await.unwrap();
        assert!(terminal.metadata().terminal);
        let terminal_events = events(&terminal);
        assert_eq!(terminal_events[0]["choices"][0]["finish_reason"], "length");
        assert!(terminal_events[0]["choices"][0].get("delta").is_some());
        assert_eq!(terminal_events.len(), 1 + usize::from(opened));
        if opened {
            assert_eq!(terminal_events[1]["usage"]["completion_tokens"], 1);
        }
        drop(terminal);
        let completion = credited_bound(session.completion).await.unwrap();
        match completion.payload() {
            OutputCompletion::Succeeded {
                history: Some(history),
                usage,
                reason,
                ..
            } => {
                assert_eq!(history.text, "ok");
                assert_eq!(history.tokens, vec![TokenId::new(6)]);
                assert_eq!(usage.completion_tokens, 1);
                assert_eq!(*reason, FinishReason::Length);
            }
            _ => panic!("Chat terminal lost actual model completion"),
        }
        assert!(credited_pool(&engine).snapshot().data_used.projection_bytes > 0);
        drop((completion, session.frames));
        credited_drained(&engine).await;
        calls.assert_unused();
    }
}

#[tokio::test]
async fn credited_engine_chat_budget_rejection_submits_zero_executor_work() {
    let (mut engine, scheduler, executor) = credited_fixture();
    enable_chat_events(&mut engine);
    Arc::get_mut(&mut engine.inner)
        .unwrap()
        .config
        .scheduler
        .slo
        .output
        .max_projection_bytes_per_request = NonZeroUsize::new(1).unwrap();
    let request = chat_request(2, false, false);
    let result = engine
        .infer_credited_stream(
            request,
            InferenceRequestContext::capture(),
            Arc::new(OutputProjectionContract::chat_sse(
                "id".into(),
                "alias".into(),
                false,
            )),
        )
        .await;
    assert!(result.is_err());
    assert_eq!(executor.inner.prefill_count(), 0);
    assert_eq!(executor.batch_decode_calls.load(Ordering::Relaxed), 0);
    assert_eq!(executor.mixed_calls.load(Ordering::Relaxed), 0);
    assert!(engine.inner.sequences.read().is_empty());
    assert_eq!(scheduler.waiting_count(), 0);
    assert_eq!(credited_pool(&engine).snapshot().retained_accounts, 0);
}

#[tokio::test]
async fn credited_engine_chat_full_consumer_replans_healthy_commit_then_cleans_cancelled_history() {
    let (mut engine, scheduler, executor) = credited_fixture();
    // Both requests omit usage: finish + DONE reserve two events. Leave exactly
    // one data slot so the unread first text frame creates real queue pressure.
    Arc::get_mut(&mut engine.inner)
        .unwrap()
        .config
        .scheduler
        .slo
        .output
        .max_queued_events_per_request = NonZeroUsize::new(3).unwrap();
    let (slow_id, slow) = submit_chat(&engine, 4, false, false).await;
    credited_ready(&engine, &slow_id).await;
    engine
        .inner
        .process_batch(&credited_next(&scheduler, 1).await)
        .await
        .unwrap();
    credited_wait(&engine, &slow_id, |state| {
        matches!(
            state,
            OutputReadinessState::OutputBlocked(
                crate::continuous_engine::output_flow_runtime::OutputBlockReason::WireQueue
            )
        )
    })
    .await;
    let (healthy_id, mut healthy) = submit_chat(&engine, 1, false, false).await;
    credited_ready(&engine, &healthy_id).await;
    let combined = credited_next(&scheduler, 2).await;
    assert_eq!(combined.requests.len(), 2);
    engine.inner.process_batch(&combined).await.unwrap();
    assert_eq!(
        executor.inner.prefill_count(),
        1,
        "blocked cohort must not partially submit"
    );
    assert!(engine.inner.sequences.read()[&healthy_id]
        .generated_tokens
        .is_empty());
    credited_ready(&engine, &healthy_id).await;
    engine.inner.refresh_credited_output_readiness();
    let replanned = credited_next(&scheduler, 2).await;
    assert_eq!(replanned.requests.len(), 1);
    assert_eq!(replanned.requests[0].request.id, healthy_id);
    engine.inner.process_batch(&replanned).await.unwrap();
    assert_eq!(executor.inner.prefill_count(), 2);
    assert_eq!(executor.batch_decode_calls.load(Ordering::Relaxed), 0);
    assert!(!engine.inner.sequences.read().contains_key(&healthy_id));
    let data = credited_bound(healthy.frames.next()).await.unwrap();
    assert_eq!(events(&data)[0]["choices"][0]["delta"]["content"], "ok");
    drop(data);
    drop(credited_bound(healthy.frames.next()).await.unwrap());
    drop(credited_bound(healthy.completion).await.unwrap());
    drop(healthy.frames);
    let projection = credited_pool(&engine).snapshot().data_used.projection_bytes;
    drop(slow.frames);
    credited_wait(&engine, &slow_id, |state| {
        matches!(state, OutputReadinessState::Closing(_))
    })
    .await;
    assert_eq!(
        engine.inner.sequences.read()[&slow_id]
            .generated_tokens
            .len(),
        1
    );
    assert_eq!(
        credited_pool(&engine).snapshot().data_used.projection_bytes,
        projection
    );
    let mut completion = slow.completion;
    assert!(futures::poll!(&mut completion).is_pending());
    engine.inner.run_iteration().await.unwrap();
    assert!(!engine.inner.sequences.read().contains_key(&slow_id));
    assert_eq!(executor.inner.prefill_count(), 2);
    let completion = credited_bound(completion).await.unwrap();
    assert!(matches!(completion.payload(), OutputCompletion::Failed(_)));
    assert_eq!(
        credited_pool(&engine).snapshot().data_used.projection_bytes,
        projection
    );
    drop(completion);
    credited_drained(&engine).await;
}
