use super::*;
use ferrum_types::{
    ApiChatRequest, ApiRequest, ApiStreamOptions, ApiToolChoice,
    PROMPT_OPENED_REASONING_METADATA_KEY,
};

fn chat_request(tokens: usize, opened: bool, include_usage: bool) -> InferenceRequest {
    let mut request = InferenceRequest::new("rendered prompt", "wire-model");
    request.stream = true;
    request.sampling_params.max_tokens = tokens;
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
            include_usage: Some(include_usage),
        }),
    }));
    request
}
fn chat_plan(tokens: usize, opened: bool, usage: bool) -> RequestOutputPlan {
    RequestOutputPlan::derive(
        Arc::new(OutputProjectionContract::chat_sse(
            "id\"\n".into(),
            "wire-中".into(),
            usage,
        )),
        &BoundedTokenizer::new(1),
        &chat_request(tokens, opened, usage),
        3,
    )
    .unwrap()
}
fn sse_values(bytes: &[u8]) -> Vec<serde_json::Value> {
    std::str::from_utf8(bytes)
        .unwrap()
        .split("\n\n")
        .filter_map(|event| event.strip_prefix("data: "))
        .filter(|data| *data != "[DONE]")
        .map(|data| serde_json::from_str(data).unwrap())
        .collect()
}
fn channel_text<'a>(delta: ChatOutputDelta<'a>) -> (bool, &'a str) {
    match delta {
        ChatOutputDelta::Reasoning(text) => (true, text),
        ChatOutputDelta::Content(text) => (false, text),
    }
}

#[test]
fn chat_output_actual_serializer_respects_escaped_envelope_and_writer_capacity() {
    let plan = chat_plan(32, false, false);
    let text = "\0\n\"\\中";
    for delta in [
        ChatOutputDelta::Content(text),
        ChatOutputDelta::Reasoning(text),
    ] {
        let size =
            codec::count(|out| codec::chat_data(out, &plan.contract, delta, u64::MAX)).unwrap();
        assert!(size <= plan.codec_descriptor().unwrap().max_data_envelope_bytes + 6 * text.len());
        assert!(codec::encode(size - 1, |out| {
            codec::chat_data(out, &plan.contract, delta, u64::MAX)
        })
        .is_err());
        let bytes = codec::encode(size, |out| {
            codec::chat_data(out, &plan.contract, delta, u64::MAX)
        })
        .unwrap();
        assert!(bytes.capacity() <= size);
        let events = sse_values(&bytes);
        let field = match delta {
            ChatOutputDelta::Content(_) => "content",
            ChatOutputDelta::Reasoning(_) => "reasoning",
        };
        assert_eq!(events[0]["choices"][0]["delta"][field], text);
    }
}

#[test]
fn chat_output_semantic_capacity_failure_preserves_reserved_error_terminal() {
    let plan = chat_plan(1, false, false);
    let output_pool = pool(&plan);
    let mut budget = RequestOutputBudget::open(&output_pool, limits(&plan), plan).unwrap();
    let permit = begin(&mut budget);
    assert!(matches!(
        budget.encode_chat_data_frame(permit, ChatOutputDelta::Content("12345"), 1),
        Err(OutputFlowError::BoundExceeded)
    ));
    let error = BoundedOutputError::new("projection failed");
    let terminal = budget
        .encode_terminal(OutputTerminal::Error(&error))
        .unwrap();
    assert_eq!(
        sse_values(terminal.payload())[0]["error"]["message"],
        "projection failed"
    );
    assert!(terminal.payload().ends_with(b"data: [DONE]\n\n"));
    drop((budget, terminal));
    assert_eq!(output_pool.snapshot().retained_accounts, 0);
}

#[test]
fn chat_output_plan_uses_resolved_policy_and_separate_semantic_bound() {
    let plan = chat_plan(32, true, false);
    assert_eq!(plan.max_decoded_bytes(), 32);
    assert_eq!(plan.max_semantic_bytes(), 4 * 32);
    assert_eq!(plan.max_data_frames(), 2 * 33);
    assert_eq!(plan.minimum_event_capacity(), 3);
    assert_eq!(
        plan.codec_descriptor().unwrap().kind,
        OutputCodecKind::ChatSse {
            include_usage: false,
            prompt_opened: true
        }
    );
    assert_ne!(
        plan.codec_descriptor().unwrap(),
        chat_plan(32, false, false).codec_descriptor().unwrap()
    );
    let output_pool = pool(&plan);
    let mut short = limits(&plan);
    short.maximum.projection_bytes -= 1;
    assert!(matches!(
        RequestOutputBudget::open(&output_pool, short, plan),
        Err(OutputFlowError::BoundExceeded)
    ));
    assert_eq!(output_pool.snapshot().retained_accounts, 0);
}

#[test]
fn chat_output_plan_rejects_missing_or_conflicting_actual_contract() {
    let contract = Arc::new(OutputProjectionContract::chat_sse(
        "id".into(),
        "alias".into(),
        false,
    ));
    let mut request = chat_request(8, true, false);
    request
        .metadata
        .remove(PROMPT_OPENED_REASONING_METADATA_KEY);
    assert!(matches!(
        RequestOutputPlan::derive(contract.clone(), &BoundedTokenizer::new(1), &request, 3),
        Err(OutputFlowError::Unsupported(_))
    ));
    request = chat_request(8, false, true);
    assert!(matches!(
        RequestOutputPlan::derive(contract.clone(), &BoundedTokenizer::new(1), &request, 3),
        Err(OutputFlowError::Unsupported(_))
    ));
    request = chat_request(8, false, false);
    let Some(ApiRequest::Chat(chat)) = request.api_request.as_mut() else {
        unreachable!()
    };
    chat.tool_choice = Some(ApiToolChoice::Mode("required".into()));
    assert!(matches!(
        RequestOutputPlan::derive(contract, &BoundedTokenizer::new(1), &request, 3),
        Err(OutputFlowError::Unsupported(_))
    ));
    assert!(matches!(
        RequestOutputPlan::derive(
            Arc::new(OutputProjectionContract::cli_text()),
            &BoundedTokenizer::new(1),
            &chat_request(8, false, false),
            3
        ),
        Err(OutputFlowError::Unsupported(_))
    ));
}

#[test]
fn chat_output_cursor_holds_markers_and_flushes_terminal_without_raw_growth() {
    let plan = chat_plan(8, false, false);
    let mut projection = BoundedChatProjection::new(&plan).unwrap();
    assert!(projection.retained_capacity() <= 2 * plan.max_decoded_bytes());
    projection.prepare("<thi", false).unwrap();
    assert!(!projection.has_pending());
    assert!(matches!(
        projection.prepare("too much text", false),
        Err(OutputFlowError::BoundExceeded)
    ));
    assert_eq!(projection.raw_text(), "<thi");
    projection.prepare("", true).unwrap();
    assert_eq!(
        projection.pending_delta(),
        Some(ChatOutputDelta::Content("<thi"))
    );
    projection.advance().unwrap();
    assert!(!projection.has_pending());
    assert!(matches!(
        projection.prepare("", true),
        Err(OutputFlowError::Closed)
    ));
}

#[test]
fn chat_output_one_command_drains_two_channels_with_recyclable_event_credit() {
    let plan = chat_plan(40, false, true);
    let output_pool = pool(&plan);
    let mut budget = RequestOutputBudget::open(&output_pool, limits(&plan), plan.clone()).unwrap();
    let mut projection = BoundedChatProjection::new(&plan).unwrap();
    projection
        .prepare("<think>推理</think>\nanswer", false)
        .unwrap();
    assert!(matches!(
        projection.prepare("next", false),
        Err(OutputFlowError::FrameInFlight)
    ));
    let permit = begin(&mut budget);
    let first = budget
        .encode_chat_data_frame(permit, projection.pending_delta().unwrap(), 17)
        .unwrap();
    projection.advance().unwrap();
    assert_eq!(
        sse_values(first.payload())[0]["choices"][0]["delta"]["reasoning"],
        "推理"
    );
    assert!(matches!(
        budget.try_begin_frame().unwrap(),
        OutputFrameAttempt::Full(_)
    ));
    assert_eq!(
        projection.pending_delta(),
        Some(ChatOutputDelta::Content("answer"))
    );
    drop(first);
    let permit = begin(&mut budget);
    let second = budget
        .encode_chat_data_frame(permit, projection.pending_delta().unwrap(), 17)
        .unwrap();
    projection.advance().unwrap();
    let event = &sse_values(second.payload())[0];
    assert_eq!(event["id"], "id\"\n");
    assert_eq!(event["model"], "wire-中");
    assert_eq!(event["choices"][0]["delta"]["content"], "answer");
    assert!(event["choices"][0]["delta"].get("reasoning").is_none());
    assert!(second.payload().capacity() <= second.credit().bytes);
    assert!(!projection.has_pending());
    projection.prepare("", true).unwrap();
    assert!(!projection.has_pending());
    let terminal = budget
        .encode_terminal(OutputTerminal::Success {
            reason: FinishReason::Length,
            usage: &TokenUsage {
                prompt_tokens: 3,
                completion_tokens: 1,
                total_tokens: 4,
            },
            created: 17,
        })
        .unwrap();
    assert_eq!(terminal.credit().events, 3);
    let events = sse_values(terminal.payload());
    assert_eq!(events[0]["choices"][0]["delta"]["role"], "assistant");
    assert_eq!(events[0]["choices"][0]["delta"]["content"], "");
    assert_eq!(events[0]["choices"][0]["finish_reason"], "length");
    assert_eq!(events[1]["usage"]["completion_tokens"], 1);
    assert!(terminal.payload().ends_with(b"data: [DONE]\n\n"));
    drop(projection);
    drop(budget);
    assert_ne!(output_pool.snapshot().retained_accounts, 0);
    drop((second, terminal));
    assert_eq!(output_pool.snapshot().retained_accounts, 0);
}

#[test]
fn chat_output_reclassification_can_exceed_raw_length_without_exceeding_real_budget() {
    let raw = "hello</think>answer";
    let plan = chat_plan(raw.len(), false, false);
    let output_pool = pool(&plan);
    let mut budget = RequestOutputBudget::open(&output_pool, limits(&plan), plan.clone()).unwrap();
    let mut projection = BoundedChatProjection::new(&plan).unwrap();
    let mut payload_bytes = 0;
    for byte in raw.as_bytes() {
        projection
            .prepare(
                std::str::from_utf8(std::slice::from_ref(byte)).unwrap(),
                false,
            )
            .unwrap();
        while let Some(delta) = projection.pending_delta() {
            payload_bytes += delta.text().len();
            let permit = begin(&mut budget);
            let frame = budget.encode_chat_data_frame(permit, delta, 1).unwrap();
            projection.advance().unwrap();
            drop(frame);
        }
    }
    projection.prepare("", true).unwrap();
    assert!(!projection.has_pending());
    assert!(payload_bytes > plan.max_decoded_bytes());
    assert!(payload_bytes <= plan.max_semantic_bytes());
    let terminal = budget
        .encode_terminal(OutputTerminal::Success {
            reason: FinishReason::EOS,
            usage: &TokenUsage {
                prompt_tokens: 3,
                completion_tokens: raw.len(),
                total_tokens: raw.len() + 3,
            },
            created: 1,
        })
        .unwrap();
    let events = sse_values(terminal.payload());
    assert_eq!(events.len(), 1);
    assert!(events[0].get("usage").is_none());
    assert_eq!(terminal.credit().events, 2);
    drop(projection);
    drop((budget, terminal));
    assert_eq!(output_pool.snapshot().data_used, OutputCreditAmount::ZERO);
}

#[test]
fn chat_output_cursor_matches_legacy_channels_at_every_two_chunk_split() {
    for raw in [
        "reason</think>\nanswer 🦀",
        "<think>x</think>y",
        "hello</think>middle<think>new</think>last",
        "<thi",
    ] {
        for opened in [false, true] {
            for cut in raw
                .char_indices()
                .map(|(offset, _)| offset)
                .chain(std::iter::once(raw.len()))
            {
                let plan = chat_plan(raw.len().max(1), opened, false);
                let mut projection = BoundedChatProjection::new(&plan).unwrap();
                let mut cumulative = String::new();
                let mut sent = [0; 2];
                let mut expected = [String::new(), String::new()];
                let mut actual = [String::new(), String::new()];
                for (delta, terminal) in [(&raw[..cut], false), (&raw[cut..], false), ("", true)] {
                    cumulative.push_str(delta);
                    if terminal
                        || !ferrum_types::should_defer_model_reasoning_stream_delta(
                            ModelOutputProtocol::Text,
                            &cumulative,
                        )
                    {
                        let parsed =
                            ferrum_types::parse_reasoning_response_for_prompt(&cumulative, opened);
                        for (channel, text) in [
                            parsed.content.as_str(),
                            parsed.reasoning.as_deref().unwrap_or(""),
                        ]
                        .into_iter()
                        .enumerate()
                        {
                            if sent[channel] <= text.len() && text.is_char_boundary(sent[channel]) {
                                expected[channel].push_str(&text[sent[channel]..]);
                            }
                            sent[channel] = text.len();
                        }
                    }
                    projection.prepare(delta, terminal).unwrap();
                    while let Some(delta) = projection.pending_delta() {
                        let (reasoning, text) = channel_text(delta);
                        actual[usize::from(reasoning)].push_str(text);
                        projection.advance().unwrap();
                    }
                }
                assert_eq!(actual, expected, "{raw:?} at {cut}, opened={opened}");
            }
        }
    }
}

#[test]
fn chat_output_delayed_completion_requires_actual_capability_and_additional_storage() {
    let contract = Arc::new(OutputProjectionContract::chat_sse(
        "id".into(),
        "model".into(),
        false,
    ));
    let mut tokenizer = BoundedTokenizer::new(8);
    let mut request = chat_request(8, true, false);
    let ordinary = RequestOutputPlan::derive(contract.clone(), &tokenizer, &request, 3).unwrap();
    request.sampling_params.response_completion_boundary =
        ResponseCompletionBoundary::AfterDelimiterAndPayload {
            delimiter: "</think>".into(),
            alternate_envelope: None,
        };
    assert!(matches!(
        RequestOutputPlan::derive(contract.clone(), &tokenizer, &request, 3),
        Err(OutputFlowError::Unsupported(_))
    ));
    tokenizer.atomic_delimiter = true;
    assert!(matches!(
        RequestOutputPlan::derive(contract.clone(), &tokenizer, &request, 3),
        Err(OutputFlowError::Unsupported(_))
    ));
    tokenizer.incremental = true;
    let delayed = RequestOutputPlan::derive(contract.clone(), &tokenizer, &request, 3).unwrap();
    let extra = 2 * ordinary.max_decoded_bytes()
        + 8 * std::mem::size_of::<TokenId>()
        + 3 * "</think>".len()
        + std::mem::size_of::<u32>()
        + std::mem::size_of::<usize>();
    assert_eq!(delayed.projection_bytes, ordinary.projection_bytes + extra);
    assert_eq!(delayed.effective_max_tokens(), 8);
    assert_eq!(
        delayed.bounded_incremental_policy(),
        Some(BoundedIncrementalDecodePolicy::StrictDecodedPrefix)
    );
    let output_pool = pool(&delayed);
    let mut too_small = limits(&delayed);
    too_small.maximum.projection_bytes = ordinary.projection_bytes;
    assert!(matches!(
        RequestOutputBudget::open(&output_pool, too_small, delayed),
        Err(OutputFlowError::BoundExceeded)
    ));
    assert_eq!(output_pool.snapshot().retained_accounts, 0);
    request
        .metadata
        .insert(PROMPT_OPENED_REASONING_METADATA_KEY.into(), false.into());
    assert!(matches!(
        RequestOutputPlan::derive(contract, &tokenizer, &request, 3),
        Err(OutputFlowError::Unsupported(_))
    ));
}
