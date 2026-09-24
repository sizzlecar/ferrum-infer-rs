use super::*;
use std::num::NonZeroUsize;

fn contract(request: &mut InferenceRequest, chat: bool) -> Arc<OutputProjectionContract> {
    Arc::new(if chat {
        request.metadata.insert(
            ferrum_types::PROMPT_OPENED_REASONING_METADATA_KEY.into(),
            false.into(),
        );
        request.api_request = Some(ApiRequest::Chat(ApiChatRequest {
            messages: Vec::new(),
            tools: Vec::new(),
            tool_choice: None,
            tool_call_protocol: Default::default(),
            legacy_functions: Vec::new(),
            legacy_function_call: None,
            response_format: None,
            stream_options: Some(ApiStreamOptions {
                include_usage: Some(true),
            }),
        }));
        OutputProjectionContract::chat_sse(request.id.to_string(), "alias".into(), true)
    } else {
        OutputProjectionContract::cli_text()
    })
}

async fn drained(engine: &ContinuousBatchEngine) {
    let pool = engine
        .inner
        .output_credit_pool
        .get()
        .unwrap()
        .as_ref()
        .unwrap();
    let mut changed = pool.subscribe();
    bounded(async {
        loop {
            let state = pool.snapshot();
            if state.data_used == Default::default() && state.terminal_held == Default::default() {
                break;
            }
            changed.changed().await.unwrap();
        }
    })
    .await;
}

#[tokio::test]
async fn actual_tokenization_and_rendered_bytes_reject_overload_before_shared_acceptance() {
    for chat in [false, true] {
        for token_limit in [true, false] {
            let mut f = Fixture::new().await;
            let admission = &mut f.config.scheduler.slo.admission;
            admission.max_waiting_prompt_tokens =
                NonZeroUsize::new(if token_limit { 3 } else { 100 }).unwrap();
            admission.max_waiting_prompt_bytes =
                NonZeroUsize::new(if token_limit { 100 } else { 3 }).unwrap();
            let engine = f.build(f.config.clone()).unwrap();
            let mut request = request(&f.config);
            if !token_limit {
                request.prompt = "test".into();
            }
            // A client-supplied zero must not replace the actual four/one
            // token prompt measured by the common engine tokenizer.
            request
                .metadata
                .insert(ferrum_types::PROMPT_TOKENS_METADATA_KEY.into(), 0.into());
            let output = contract(&mut request, chat);
            let id = request.id.clone();
            let result = engine
                .infer_credited_stream(request, InferenceRequestContext::capture(), output)
                .await;
            assert!(matches!(result, Err(FerrumError::ResourceExhausted { .. })));
            assert!(engine.inner.sequences.read().is_empty());
            assert_eq!(engine.inner.scheduler.waiting_count(), 0);
            assert_eq!(engine.inner.scheduler.trace_phase(&id), None);
            assert_eq!(f.executor.entries.load(Ordering::Acquire), 0);
            assert!(!engine.inner.bg_loop_spawned.load(Ordering::Acquire));
            drained(&engine).await;
            bounded(engine.shutdown()).await.unwrap();
        }
    }
}

#[tokio::test]
async fn late_unknown_request_retains_ingress_and_original_output_while_only_new_queue_work_is_limited(
) {
    for chat in [false, true] {
        let mut f = Fixture::new().await;
        f.config.scheduler.slo.admission.max_waiting_requests = NonZeroUsize::new(1).unwrap();
        f.config.scheduler.slo.admission.max_waiting_prompt_tokens = NonZeroUsize::new(2).unwrap();
        f.config.scheduler.slo.admission.max_waiting_prompt_bytes = NonZeroUsize::new(9).unwrap();
        let engine = f.build(f.config.clone()).unwrap();
        // Keep the genuine accepted request in Waiting so this test measures
        // acceptance capacity independently of asynchronous physical admission.
        engine.inner.bg_loop_spawned.store(true, Ordering::Release);
        let ingress = Instant::now() - Duration::from_secs(3600);
        let mut first = request(&f.config);
        first.prompt = "test test".into();
        first.metadata.insert(
            ferrum_types::PROMPT_TOKENS_METADATA_KEY.into(),
            u64::MAX.into(),
        );
        let output = contract(&mut first, chat);
        let id = first.id.clone();
        let maximum = first.sampling_params.max_tokens;
        let session = engine
            .infer_credited_stream(
                first,
                InferenceRequestContext::from_ingress(ingress),
                output,
            )
            .await
            .unwrap();
        {
            let sequences = engine.inner.sequences.read();
            let sequence = &sequences[&id];
            assert_eq!(sequence.input_tokens.len(), 2);
            assert_eq!(sequence.slo.as_ref().unwrap().ingress(), ingress);
            assert_eq!(
                sequence.original_request.sampling_params.max_tokens,
                maximum
            );
            assert!(sequence.generated_tokens.is_empty());
            assert!(sequence.request_slot.is_some());
        }
        assert_eq!(engine.inner.scheduler.waiting_count(), 1);
        let mut second = request(&f.config);
        second.prompt = "test".into();
        let output = contract(&mut second, chat);
        let rejected = second.id.clone();
        assert!(matches!(
            engine
                .infer_credited_stream(second, InferenceRequestContext::capture(), output,)
                .await,
            Err(FerrumError::ResourceExhausted { .. })
        ));
        assert_eq!(engine.inner.scheduler.trace_phase(&rejected), None);
        assert!(engine.inner.sequences.read().contains_key(&id));
        drop(session);
        bounded(engine.shutdown()).await.unwrap();
        assert_eq!(engine.inner.scheduler.waiting_count(), 0);
        assert!(engine.inner.sequences.read().is_empty());
        drained(&engine).await;
    }
}
