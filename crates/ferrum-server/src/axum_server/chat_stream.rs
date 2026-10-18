//! Shared semantic Chat stream, consumed directly by Chat and Responses.
use super::*;
use ferrum_output::OutputSession;
use futures::Stream;
use std::pin::Pin;

pub(super) enum ChatStreamEvent {
    Chunk(ChatCompletionsResponse),
    Error(OpenAiError),
    Done,
}

pub(super) struct ChatEventStream {
    pub(super) session: OutputSession,
    pub(super) events: Pin<Box<dyn Stream<Item = ChatStreamEvent> + Send>>,
}

fn error_event(
    message: impl Into<String>,
    error_type: &str,
    param: Option<&str>,
) -> ChatStreamEvent {
    ChatStreamEvent::Error(stream_error_payload(message, error_type, param))
}

pub(super) fn into_response(mut stream: ChatEventStream) -> Response {
    let session = stream.session.clone();
    let frames = async_stream::try_stream! {
        while let Some(event) = stream.events.next().await {
            let encoded = match event {
                ChatStreamEvent::Chunk(chunk) => output::encode_json(&stream.session, None, &chunk).await,
                ChatStreamEvent::Error(error) => {
                    let bytes = output::encode_chat_failure(&stream.session, &error)?;
                    drop(stream);
                    yield bytes;
                    return;
                }
                ChatStreamEvent::Done => output::encode_done(&stream.session).await,
            };
            let (bytes, terminal_error) = output::chat_encoding_result(&stream.session, encoded)?;
            if terminal_error {
                drop(stream);
                yield bytes;
                return;
            }
            yield bytes;
        }
    };
    output::sse_response(session, frames)
}

pub(super) async fn open_chat_events(
    state: AppState,
    openai_request: ChatCompletionsRequest,
    inference_request: InferenceRequest,
    benchmark_correlation: Option<BenchmarkRequestCorrelation>,
) -> std::result::Result<ChatEventStream, ServerError> {
    let engine = state.llm.clone().ok_or_else(|| {
        ServerError::ServiceUnavailable("LLM engine not loaded; chat unavailable".into())
    })?;
    let request_id = inference_request.id.to_string();
    let include_stream_usage = openai_request
        .stream_options
        .as_ref()
        .and_then(|opts| opts.include_usage)
        .unwrap_or(false);
    let output_contract = EffectiveChatOutputContract::resolve(&openai_request);
    let buffer_json_object_stream = matches!(
        output_contract,
        EffectiveChatOutputContract::JsonObjectContent
    );
    let buffer_strict_json_schema_stream = matches!(
        output_contract,
        EffectiveChatOutputContract::StrictJsonSchemaContent
    );
    let stream_api_request = match inference_request.api_request.as_ref() {
        Some(ferrum_types::ApiRequest::Chat(request)) => request.clone(),
        _ => api_chat_request(
            &openai_request,
            openai_request.tool_choice.as_ref(),
            ferrum_types::ApiToolCallProtocol::default(),
        ),
    };
    let buffer_structured_api_stream =
        ferrum_types::chat_api_may_emit_tool_or_function_call(&stream_api_request);
    let model_output_protocol = inference_request.sampling_params.model_output_protocol;
    let buffer_stream_output = buffer_json_object_stream
        || buffer_strict_json_schema_stream
        || buffer_structured_api_stream
        || model_output_protocol == ModelOutputProtocol::HarmonyGptOss;
    // R1-distill-style templates open the think block inside the prompt;
    // the parser must know generation starts mid-think.
    let started_in_think = request_started_in_reasoning(&inference_request);
    let mut native_projector = NativeChatOutputProjector::for_request(&inference_request);
    let replay_request_id = inference_request.id.to_string();
    let profile_request_model = openai_request.model.clone();
    let profile_started_at = Instant::now();
    let request_memory_before = request_memory_sample_before(&state);
    let mut stream = match engine.infer_stream(inference_request).await {
        Ok(stream) => stream,
        Err(e) => {
            let failure_kind = e.observability_failure_kind();
            let error_kind = e.observability_error_kind();
            let error_message = e.to_string();
            if let Err(err) = write_chat_request_profile_event(
                &state,
                &replay_request_id,
                benchmark_correlation.as_ref(),
                &profile_request_model,
                true,
                "chat_completions_stream_start",
                profile_started_at,
                ChatRequestProfileTiming::default(),
                0,
                None,
                Some("error"),
                Some(ProfileError {
                    kind: error_kind.to_string(),
                    message: error_message.clone(),
                    blocking: false,
                }),
            ) {
                warn!("failed to write chat stream failure profile event: {}", err);
            }
            let engine_status = if chat_resource_failure_kind(failure_kind) {
                Some(engine.status().await)
            } else {
                None
            };
            error!(
                "Stream generation failed before first chunk: {}",
                error_message
            );
            if let Err(err) = write_chat_request_failure_diagnostics(
                &state,
                &replay_request_id,
                failure_kind,
                "chat_completions_stream_start",
                error_kind,
                &error_message,
                engine_status.as_ref(),
            ) {
                warn!("failed to write chat stream failure diagnostics: {}", err);
            }
            return Err(server_error_from_ferrum_error(e));
        }
    };
    let request_dump_dir = state.request_dump_dir.clone();
    let admission_summary = state
        .auto_config
        .as_ref()
        .map(|config| config.admission_summary_document());
    let diagnostics_engine = engine.clone();
    let profile_state = state.clone();

    let session = stream.output_session().clone();
    let events = async_stream::stream! {
        let mut current_text = String::new();
        let mut output_token_ids = Vec::new();
        let mut first_engine_chunk_received_us = None;
        let mut first_sse_enqueue_us = None;
        let mut sent_reasoning_len = 0usize;
        let mut sent_content_len = 0usize;

        loop {
            let next = stream.next().await;
            let Some(result) = next else {
                break;
            };
            match result {
                Ok(owned_chunk) => {
                    let chunk = owned_chunk.chunk();
                    if first_engine_chunk_received_us.is_none()
                        && (chunk.token.is_some() || !chunk.text.is_empty())
                    {
                        first_engine_chunk_received_us = Some(elapsed_us_since(profile_started_at));
                    }
                    if let Some(token) = chunk.token {
                        output_token_ids.push(token);
                    }
                    if !chunk.text.is_empty() {
                        current_text.push_str(&chunk.text);
                        if let Some(projector) = native_projector.as_mut() {
                            projector.push(&chunk.text);
                        }

                        if native_projector.is_some()
                            || (!buffer_stream_output
                                && !should_defer_model_reasoning_stream_delta(
                                    model_output_protocol,
                                    &current_text,
                                ))
                        {
                            let parsed_result = if let Some(projector) = native_projector.as_ref() {
                                Ok(ParsedReasoningResponse {
                                    content: projector.visible_prefix().to_owned(),
                                    reasoning: projector.reasoning_prefix().map(str::to_owned),
                                })
                            } else {
                                parse_model_reasoning_response(
                                    model_output_protocol,
                                    &current_text,
                                    started_in_think,
                                )
                            };
                            let parsed = match parsed_result {
                                Ok(parsed) => parsed,
                                Err(error) => {
                                    yield error_event(
                                        error.to_string(),
                                        "internal_server_error",
                                        Some("model_output"),
                                    );
                                    yield ChatStreamEvent::Done;
                                    break;
                                }
                            };
                            let full_reasoning = parsed.reasoning.as_deref().unwrap_or("");
                            let reasoning_delta =
                                stream_text_delta(full_reasoning, &mut sent_reasoning_len);
                            let content_delta =
                                stream_text_delta(&parsed.content, &mut sent_content_len);
                            if !reasoning_delta.is_empty() || !content_delta.is_empty() {
                                // Create streaming response chunk
                                let response_chunk = ChatCompletionsResponse {
                                    id: request_id.clone(),
                                    object: "chat.completion.chunk".to_string(),
                                    created: chrono::Utc::now().timestamp() as u64,
                                    model: openai_request.model.clone(),
                                    choices: vec![ChatChoice {
                                        index: 0,
                                        message: None,
                                        delta: Some(ChatMessage {
                                            role: MessageRole::Assistant,
                                            content: content_delta,
                                            reasoning: (!reasoning_delta.is_empty())
                                                .then_some(reasoning_delta),
                                            name: None,
                                            tool_calls: None,
                                            tool_call_id: None,
                                            function_call: None,
                                        }),
                                        finish_reason: None,
                                    }],
                                    usage: None,
                                };

                                first_sse_enqueue_us.get_or_insert_with(|| elapsed_us_since(profile_started_at));
                                yield ChatStreamEvent::Chunk(response_chunk);
                            }
                        }
                    }

                    if chunk.finish_reason.is_some() {
                        let terminal_finish_reason = chunk
                            .finish_reason
                            .expect("finish_reason presence checked above");
                        if let Err(err) = write_chat_prompt_token_evidence(
                            request_dump_dir.as_ref().map(|root| root.as_path()),
                            &replay_request_id,
                            &profile_request_model,
                            chunk.execution_evidence.as_ref(),
                        ) {
                            warn!("failed to write chat stream prompt-token evidence: {}", err);
                        }
                        let usage = chunk.usage.as_ref().map(openai_usage_from_token_usage);
                        let native_projected = native_projector
                            .take()
                            .map(|projector| projector.finish(terminal_finish_reason));
                        let parsed_output_result =
                            if let Some(projected) = native_projected.as_ref() {
                                Ok(ParsedChatModelOutput {
                                    visible: projected.visible.clone(),
                                    harmony_response: None,
                                })
                            } else {
                                parse_chat_model_output(
                                    model_output_protocol,
                                    &current_text,
                                    started_in_think,
                                    terminal_finish_reason,
                                )
                            };
                        let parsed_model_output = match parsed_output_result {
                            Ok(parsed) => parsed,
                            Err(error) => {
                                let error_event = error_event(
                                    stream_validation_error_message(error),
                                    "internal_server_error",
                                    Some("model_output"),
                                );
                                yield error_event;
                                yield ChatStreamEvent::Done;
                                break;
                            }
                        };
                        let mut parsed_final = parsed_model_output.visible;
                        parsed_final.content = normalize_structured_response_content(
                            &openai_request,
                            &parsed_final.content,
                        );
                        let mut structured_chat_response =
                            finish_reason_allows_structured_api_response(terminal_finish_reason)
                                .then(|| match chunk.api_response.as_ref() {
                                    // The native envelope owns the recipient and whether this is
                                    // a call at all, including a final answer containing JSON.
                                    _ if model_output_protocol
                                        == ModelOutputProtocol::HarmonyGptOss =>
                                    {
                                        parsed_model_output.harmony_response.clone()
                                    }
                                    Some(ferrum_types::ApiResponse::Chat(response)) => {
                                        Some(response.clone())
                                    }
                                    _ if native_projected.is_some() => native_projected
                                        .as_ref()
                                        .and_then(|projected| projected.api_response.clone()),
                                    _ if buffer_structured_api_stream => {
                                        chat_api_response_from_parsed_generated_text(
                                            &stream_api_request,
                                            &parsed_final,
                                            terminal_finish_reason,
                                        )
                                    }
                                    _ => None,
                                })
                                .flatten();

                        if native_projected.is_none()
                            && matches!(
                                chunk.api_response,
                                Some(ferrum_types::ApiResponse::Chat(_))
                            )
                        {
                            if let Some(response) = structured_chat_response.as_mut() {
                                if let Err(error) = project_typed_tool_response_content(
                                    response,
                                    model_output_protocol,
                                    started_in_think,
                                ) {
                                    yield error_event(
                                        stream_validation_error_message(error),
                                        "internal_server_error",
                                        Some("model_output"),
                                    );
                                    yield ChatStreamEvent::Done;
                                    break;
                                }
                            }
                        }

                        if let Some(chat_response) = structured_chat_response.as_ref() {
                            if let Err(e) =
                                validate_structured_tool_response(&openai_request, chat_response)
                            {
                                let error_event = error_event(
                                    stream_validation_error_message(e),
                                    "internal_server_error",
                                    Some("tool_choice"),
                                );
                                yield error_event;
                                yield ChatStreamEvent::Done;
                                break;
                            }
                        } else if tool_choice_required(&openai_request) {
                            log_required_tool_choice_failure(
                                &openai_request,
                                &parsed_final.content,
                                parsed_final.reasoning.as_deref(),
                            );
                            let error_event = error_event(
                                "model output did not satisfy required tool_choice",
                                "invalid_request_error",
                                Some("tool_choice"),
                            );
                            yield error_event;
                            yield ChatStreamEvent::Done;
                            break;
                        }
                        if let Err(e) = validate_hard_structured_response(
                            &openai_request,
                            &parsed_final.content,
                            structured_chat_response.as_ref(),
                        ) {
                            let error_event = error_event(
                                stream_validation_error_message(e),
                                "internal_server_error",
                                structured_response_error_param(output_contract),
                            );
                            yield error_event;
                            yield ChatStreamEvent::Done;
                            break;
                        }

                        if let Some(chat_response) = structured_chat_response.as_ref() {
                            let mut delta = openai_chat_delta_from_api(&chat_response.message);
                            if native_projected.is_some() {
                                delta.content =
                                    stream_text_delta(&delta.content, &mut sent_content_len);
                            }
                            if delta.reasoning.is_none() {
                                delta.reasoning = parsed_final.reasoning.clone();
                            }
                            if native_projected.is_some() {
                                let reasoning = stream_text_delta(
                                    delta.reasoning.as_deref().unwrap_or(""),
                                    &mut sent_reasoning_len,
                                );
                                delta.reasoning = (!reasoning.is_empty()).then_some(reasoning);
                            }
                            let response_chunk = ChatCompletionsResponse {
                                id: request_id.clone(),
                                object: "chat.completion.chunk".to_string(),
                                created: chrono::Utc::now().timestamp() as u64,
                                model: openai_request.model.clone(),
                                choices: vec![ChatChoice {
                                    index: 0,
                                    message: None,
                                    delta: Some(delta),
                                    finish_reason: None,
                                }],
                                usage: None,
                            };

                            first_sse_enqueue_us.get_or_insert_with(|| elapsed_us_since(profile_started_at));
                                yield ChatStreamEvent::Chunk(response_chunk);
                        } else if buffer_structured_api_stream
                            && parsed_final.content.trim().is_empty()
                            // A token-limited response may contain only reasoning.
                            // Preserve its terminal reason and usage, as sync does;
                            // required tool and hard content contracts were checked above.
                            && terminal_finish_reason != FinishReason::Length
                        {
                            let error_event = error_event(
                                "model output did not satisfy tool/function call request",
                                "internal_server_error",
                                Some("tool_choice"),
                            );
                            yield error_event;
                            yield ChatStreamEvent::Done;
                            break;
                        } else if !current_text.is_empty() {
                            // Flush safe text held with an incomplete trailing
                            // marker, including when text and finish share a chunk.
                            // Buffered streams have sent lengths of zero.
                            let content_delta =
                                stream_text_delta(&parsed_final.content, &mut sent_content_len);
                            let reasoning_delta = stream_text_delta(
                                parsed_final.reasoning.as_deref().unwrap_or(""),
                                &mut sent_reasoning_len,
                            );
                            if !content_delta.is_empty() || !reasoning_delta.is_empty() {
                                let response_chunk = ChatCompletionsResponse {
                                    id: request_id.clone(),
                                    object: "chat.completion.chunk".to_string(),
                                    created: chrono::Utc::now().timestamp() as u64,
                                    model: openai_request.model.clone(),
                                    choices: vec![ChatChoice {
                                        index: 0,
                                        message: None,
                                        delta: Some(ChatMessage {
                                            role: MessageRole::Assistant,
                                            content: content_delta,
                                            reasoning: (!reasoning_delta.is_empty())
                                                .then_some(reasoning_delta),
                                            name: None,
                                            tool_calls: None,
                                            tool_call_id: None,
                                            function_call: None,
                                        }),
                                        finish_reason: None,
                                    }],
                                    usage: None,
                                };

                                first_sse_enqueue_us.get_or_insert_with(|| elapsed_us_since(profile_started_at));
                                yield ChatStreamEvent::Chunk(response_chunk);
                            }
                        }
                        // Send final chunk. OpenAI-style streaming
                        // clients (e.g. `vllm bench serve`) blindly
                        // read `choices[0]["delta"]` on every chunk
                        // that has any `choices` entries, so the
                        // last chunk must include `delta` even when
                        // empty. Skipping it triggers
                        // `KeyError: 'delta'` on the client side
                        // and the request is reported as failed
                        // despite returning a 200 with content.
                        let final_finish_reason = structured_chat_response
                            .as_ref()
                            .and_then(|response| response.finish_reason.clone())
                            .or_else(|| chunk.finish_reason.as_ref().map(finish_reason_to_string))
                            .or(Some("length".to_string()));
                        let final_chunk = ChatCompletionsResponse {
                            id: request_id.clone(),
                            object: "chat.completion.chunk".to_string(),
                            created: chrono::Utc::now().timestamp() as u64,
                            model: openai_request.model.clone(),
                            choices: vec![ChatChoice {
                                index: 0,
                                message: None,
                                delta: Some(ChatMessage {
                                    role: MessageRole::Assistant,
                                    content: String::new(),
                                    reasoning: None,
                                    name: None,
                                    tool_calls: None,
                                    tool_call_id: None,
                                    function_call: None,
                                }),
                                finish_reason: final_finish_reason.clone(),
                            }],
                            usage: None,
                        };

                        first_sse_enqueue_us.get_or_insert_with(|| elapsed_us_since(profile_started_at));
                        let completion_token_count = chunk
                            .usage
                            .as_ref()
                            .map(|usage| usage.completion_tokens)
                            .unwrap_or(output_token_ids.len());
                        let replay_output_token_ids = chunk
                            .execution_evidence
                            .as_ref()
                            .map(|evidence| evidence.output_token_ids.as_slice())
                            .filter(|tokens| tokens.len() == completion_token_count)
                            .unwrap_or(output_token_ids.as_slice());
                        if let Err(err) = write_chat_request_completion_replay_bundle(
                            request_dump_dir.as_ref().map(|root| root.as_path()),
                            &replay_request_id,
                            &parsed_final.content,
                            replay_output_token_ids,
                            final_finish_reason.as_deref(),
                        ) {
                            warn!("failed to write chat stream replay bundle: {}", err);
                        }
                        if let Err(err) = write_chat_request_profile_event(
                            &profile_state,
                            &replay_request_id,
                            benchmark_correlation.as_ref(),
                            &profile_request_model,
                            true,
                            "chat_completions_stream_complete",
                            profile_started_at,
                            ChatRequestProfileTiming {
                                engine_evidence: chunk.execution_evidence.as_ref(),
                                first_engine_chunk_received_us,
                                first_sse_enqueue_us,
                            },
                            completion_token_count,
                            chunk.usage.as_ref(),
                            final_finish_reason.as_deref(),
                            None,
                        ) {
                            warn!("failed to write chat stream profile event: {}", err);
                        }
                        if let Err(err) = maybe_write_first_request_memory_stage(
                            &profile_state,
                            &replay_request_id,
                            benchmark_correlation.as_ref(),
                            &profile_request_model,
                            true,
                            profile_started_at,
                            request_memory_before,
                        ) {
                            warn!("failed to write chat stream memory profile event: {}", err);
                        }
                        // Persist terminal diagnostics before yielding the terminal
                        // choice: a client may drop immediately after reading it.
                        yield ChatStreamEvent::Chunk(final_chunk);
                        if include_stream_usage && usage.is_some() {
                            let usage_chunk = ChatCompletionsResponse {
                                id: request_id.clone(),
                                object: "chat.completion.chunk".to_string(),
                                created: chrono::Utc::now().timestamp() as u64,
                                model: openai_request.model.clone(),
                                choices: vec![],
                                usage,
                            };
                            yield ChatStreamEvent::Chunk(usage_chunk);
                        }
                        yield ChatStreamEvent::Done;
                        break;
                    }
                }
                Err(e) => {
                    let failure_kind = e.observability_failure_kind();
                    let error_kind = e.observability_error_kind();
                    let error_message = e.to_string();
                    let engine_status = if chat_resource_failure_kind(failure_kind) {
                        Some(diagnostics_engine.status().await)
                    } else {
                        None
                    };
                    error!("Stream generation error: {}", error_message);
                    if let Err(err) = write_chat_request_profile_event(
                        &profile_state,
                        &replay_request_id,
                        benchmark_correlation.as_ref(),
                        &profile_request_model,
                        true,
                        "chat_completions_stream_next",
                        profile_started_at,
                        ChatRequestProfileTiming {
                            engine_evidence: None,
                            first_engine_chunk_received_us,
                            first_sse_enqueue_us,
                        },
                        output_token_ids.len(),
                        None,
                        Some("error"),
                        Some(ProfileError {
                            kind: error_kind.to_string(),
                            message: error_message.clone(),
                            blocking: false,
                        }),
                    ) {
                        warn!("failed to write chat stream chunk profile event: {}", err);
                    }
                    if let Err(err) = write_chat_request_failure_diagnostics_at_root(
                        request_dump_dir.as_ref().map(|root| root.as_path()),
                        admission_summary.as_ref(),
                        engine_status.as_ref(),
                        &replay_request_id,
                        failure_kind,
                        "chat_completions_stream_next",
                        error_kind,
                        &error_message,
                    ) {
                        warn!(
                            "failed to write chat stream chunk failure diagnostics: {}",
                            err
                        );
                    }
                    yield error_event(
                        error_message,
                        "internal_server_error",
                        None,
                    );
                    yield ChatStreamEvent::Done;
                    break;
                }
            }
        }
    };
    Ok(ChatEventStream {
        session,
        events: Box::pin(events),
    })
}
