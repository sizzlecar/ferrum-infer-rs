use super::*;
use ferrum_output::{InferenceOutput, OutputBudget, OutputSession};
use ferrum_types::{OutputConfig, StreamChunk};

fn budget(limit: usize) -> OutputBudget {
    OutputBudget::new(OutputConfig {
        semantic_bytes_total: limit,
        semantic_bytes_per_session: limit,
        ..OutputConfig::default()
    })
    .unwrap()
}

fn stream(session: &OutputSession, request_id: &RequestId, text: &str) -> RunResponseStream {
    let producer = session.clone();
    let chunk = StreamChunk {
        request_id: request_id.clone(),
        text: text.into(),
        token: Some(TokenId::new(7)),
        finish_reason: Some(FinishReason::Length),
        usage: Some(TokenUsage::new(2, 1)),
        created_at: Utc::now(),
        metadata: HashMap::new(),
        api_response: None,
        execution_evidence: None,
    };
    RunResponseStream::new(
        session.clone(),
        Box::pin(futures::stream::once(async move {
            producer.wrap_chunk(chunk).await
        })),
    )
}

#[tokio::test]
async fn run_semantic_collector_history_and_context_shift_keep_real_owners() {
    for text_output in [false, true] {
        let budget = budget(16_384);
        let session = budget.try_open_session().unwrap();
        let request_id = RequestId(Uuid::new_v4());
        let stream = stream(&session, &request_id, "answer 尾");
        let generation = if text_output {
            collect_run_text_stream(
                stream,
                false,
                RunStreamOutput::new(ModelOutputProtocol::Text, false),
                0,
                &request_id,
                false,
                true,
            )
            .await
            .unwrap()
        } else {
            collect_run_stream(
                stream,
                false,
                true,
                0,
                "session",
                0,
                &request_id.to_string(),
            )
            .await
            .unwrap()
        };
        assert_eq!(budget.usage().queued_bytes, 0);
        assert_eq!(generation.view().raw_text, "answer 尾");
        assert_eq!(generation.view().token_ids, &[TokenId::new(7)]);
        assert_eq!(generation.view().usage.unwrap().completion_tokens, 1);
        let projection = RetainedRunProjection::new(
            generation.session(),
            parse_run_model_output(
                ModelOutputProtocol::Text,
                generation.view().raw_text,
                false,
                Some(FinishReason::Length),
            )
            .unwrap(),
        )
        .unwrap();
        let mut history = vec![
            RunHistoryMessage::new("user", "first"),
            RunHistoryMessage::assistant(
                generation.session(),
                generation.view().raw_text,
                ModelOutputProtocol::Text,
                &projection.parsed,
            )
            .unwrap(),
        ];
        drop(projection);
        drop(generation);
        let retained = session.semantic_usage();
        assert!(
            retained > "answer 尾".len(),
            "raw and prompt are independent retained text"
        );
        assert_eq!(budget.usage().sessions, 1);
        let start = next_context_shift_history_start(&history, 0);
        assert_eq!(start, history.len());
        build_chat_prompt(
            &history[start..],
            "next",
            None,
            "llama",
            None,
            &ChatTemplateOptions::default(),
        )
        .unwrap();
        assert_eq!(
            session.semantic_usage(),
            retained,
            "selection does not remove history"
        );
        build_chat_prompt(
            &history,
            "next",
            None,
            "llama",
            None,
            &ChatTemplateOptions::default(),
        )
        .unwrap();
        assert_eq!(
            session.semantic_usage(),
            retained,
            "temporary prompt copies release after render"
        );
        history.remove(0);
        assert_eq!(
            session.semantic_usage(),
            retained,
            "removing input does not release generated text"
        );
        drop(session);
        assert_eq!(
            budget.usage().sessions,
            1,
            "history itself retains the output session"
        );
        history.clear();
        assert_eq!(budget.usage().semantic_bytes, 0);
        assert_eq!(budget.usage().sessions, 0);
    }
}

#[test]
fn run_semantic_prompt_copy_failure_preserves_history_and_credit() {
    let raw = "cached answer";
    let cost = semantic_cost(&("assistant", raw, raw, None::<&str>)).unwrap();
    let copy = semantic_cost(&("assistant", raw, None::<&str>)).unwrap();
    let budget = budget(cost + copy - 1);
    let session = budget.try_open_session().unwrap();
    let parsed = ParsedReasoningResponse {
        content: raw.into(),
        reasoning: None,
    };
    let history = [
        RunHistoryMessage::assistant(&session, raw, ModelOutputProtocol::Text, &parsed).unwrap(),
    ];
    assert_eq!(session.semantic_usage(), cost);
    let error = build_chat_prompt(
        &history,
        "next",
        None,
        "llama",
        None,
        &ChatTemplateOptions::default(),
    )
    .unwrap_err();
    assert!(matches!(error, FerrumError::ResourceExhausted { .. }));
    assert_eq!(session.semantic_usage(), cost);
    assert_eq!(history[0].raw_content, raw);
    assert_eq!(history[0].prompt.content, raw);
}

#[tokio::test]
async fn run_semantic_collector_quota_error_releases_partial_content() {
    let budget = budget(96);
    let session = budget.try_open_session().unwrap();
    let request_id = RequestId(Uuid::new_v4());
    let output = collect_run_stream(
        stream(&session, &request_id, &"x".repeat(128)),
        false,
        true,
        0,
        "session",
        0,
        &request_id.to_string(),
    )
    .await;
    assert!(matches!(output, Err(FerrumError::ResourceExhausted { .. })));
    assert_eq!(session.semantic_usage(), 0);
    assert_eq!(budget.usage().queued_bytes, 0);
    drop(session);
    assert_eq!(budget.usage().sessions, 0);
}

#[test]
fn run_semantic_nonstream_collector_borrows_original_response() {
    let budget = budget(4096);
    let session = budget.try_open_session().unwrap();
    let response = InferenceResponse {
        request_id: RequestId(Uuid::new_v4()),
        text: "original 尾".into(),
        tokens: vec![TokenId::new(3)],
        finish_reason: FinishReason::Length,
        usage: TokenUsage::new(1, 1),
        latency_ms: 0,
        created_at: Utc::now(),
        metadata: HashMap::new(),
        api_response: None,
        execution_evidence: None,
    };
    let output = InferenceOutput::new(response, session.try_reserve_semantic(0).unwrap()).unwrap();
    let bytes = session.semantic_usage();
    let text = output.response().text.as_ptr();
    let tokens = output.response().tokens.as_ptr();
    let generation = CollectedRunGeneration::from_response(output.clone()).unwrap();
    assert_eq!(session.semantic_usage(), bytes);
    assert_eq!(generation.view().raw_text.as_ptr(), text);
    assert_eq!(generation.view().token_ids.as_ptr(), tokens);
    drop(output);
    drop(session);
    assert_eq!(budget.usage().semantic_bytes, bytes);
    drop(generation);
    assert_eq!(budget.usage().semantic_bytes, 0);
    assert_eq!(budget.usage().sessions, 0);
}

#[test]
fn run_semantic_borrowed_history_hash_preserves_json_bytes() {
    let history = [
        RunHistoryMessage::new("user", "quote \" and 尾\n"),
        RunHistoryMessage::new("assistant", "<think>why</think>answer"),
    ];
    let original: Vec<_> = history
        .iter()
        .map(|message| (&message.prompt.role, &message.raw_content))
        .collect();
    let expected = format!(
        "{:x}",
        Sha256::digest(serde_json::to_vec(&original).unwrap())
    );
    assert_eq!(history_evidence(&history)["sha256"], expected);
}

#[test]
fn run_semantic_retained_history_uses_original_output_session() {
    let budget = OutputBudget::new(OutputConfig {
        max_sessions: 1,
        ..OutputConfig::default()
    })
    .unwrap();
    let session = budget.try_open_session().unwrap();
    let parsed = ParsedReasoningResponse {
        content: "previous answer".into(),
        reasoning: None,
    };
    let mut history = vec![RunHistoryMessage::assistant(
        &session,
        "previous answer",
        ModelOutputProtocol::Text,
        &parsed,
    )
    .unwrap()];
    drop(session);
    assert!(matches!(
        budget.try_open_session(),
        Err(FerrumError::ResourceExhausted { .. })
    ));
    assert_eq!(history[0].raw_content, "previous answer");
    history.clear();
    assert_eq!(budget.usage().semantic_bytes, 0);
    let next = budget.try_open_session().unwrap();
    assert_eq!(next.semantic_usage(), 0);
}
