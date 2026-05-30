use super::*;

impl EngineInner {
    // ── stream helper ──────────────────────────────────────────────────

    pub(super) async fn send_stream_update(&self, request_id: &RequestId, token: TokenId) {
        // Decode the full generated-token history (skip_special=true matches
        // the final-response decode in `complete_request`) and emit only
        // the delta that hasn't been streamed yet. Per-token decode is
        // wrong for any model whose vocab can split a multi-byte UTF-8
        // sequence across BPE pieces — Qwen3 / Qwen2.5 routinely do this
        // for Chinese chars and emoji, and the single-token decode then
        // returns a `\u{FFFD}` replacement char that renders as a square /
        // `?` glyph in the terminal.
        //
        // Algorithm: hold the write lock once to (a) clone sender, (b)
        // decode current full history, (c) if the decoded text ends in
        // `\u{FFFD}` defer the emit (a later token will complete the
        // multi-byte sequence), (d) otherwise carve off the substring
        // past `streamed_text_len` and bump the watermark. Buffering is
        // bounded — the longest multi-byte sequence is 4 bytes, so at
        // most one or two tokens get deferred before flushing.
        let (sender, delta, ttft_s, itl_s) = {
            let mut sequences = self.sequences.write();
            let Some(seq) = sequences.get_mut(request_id) else {
                return;
            };
            let sender = seq.stream_sender.clone();
            let full = self
                .tokenizer
                .decode(&seq.generated_tokens, true)
                .unwrap_or_else(|_| format!("token_{}", token.get()));
            if full.ends_with('\u{FFFD}') {
                // Partial multi-byte UTF-8 at the tail; wait for the next
                // token. Do NOT advance streamed_text_len so the bytes get
                // re-considered once the sequence completes.
                return;
            }
            let delta = full[seq.streamed_text_len..].to_string();
            seq.streamed_text_len = full.len();

            // Latency-metric tracking (PLAYBOOK § 7 definitions).
            // We capture timestamps in the critical section so the
            // first-emit point matches the moment we commit to streaming
            // the delta — not the moment the chunk actually crosses the
            // socket, which the engine can't observe.
            let mut ttft_s: Option<f64> = None;
            let mut itl_s: Option<f64> = None;
            if !delta.is_empty() {
                let now = Instant::now();
                match seq.first_emit_at {
                    None => {
                        ttft_s = Some(now.duration_since(seq.start_time).as_secs_f64());
                        seq.first_emit_at = Some(now);
                    }
                    Some(_) => {
                        if let Some(prev) = seq.last_emit_at {
                            itl_s = Some(now.duration_since(prev).as_secs_f64());
                        }
                    }
                }
                seq.last_emit_at = Some(now);
                seq.emitted_chunks = seq.emitted_chunks.saturating_add(1);
            }

            (sender, delta, ttft_s, itl_s)
        };

        if let Some(t) = ttft_s {
            histogram!("ferrum.engine.ttft_seconds").record(t);
        }
        if let Some(t) = itl_s {
            histogram!("ferrum.engine.itl_seconds").record(t);
        }

        if let Some(tx) = sender {
            if delta.is_empty() {
                return;
            }
            let chunk = StreamChunk {
                request_id: request_id.clone(),
                text: delta,
                token: Some(token),
                finish_reason: None,
                usage: None,
                created_at: chrono::Utc::now(),
                metadata: HashMap::new(),
            };
            let _ = tx.send(Ok(chunk)).await;
        }
    }

    // ── completion ─────────────────────────────────────────────────────

    pub(super) async fn complete_request(
        &self,
        request_id: &RequestId,
        finish_reason: FinishReason,
    ) -> Result<()> {
        let (response, stream_sender, response_sender, has_kv_cache, model_cache_id) = {
            let mut sequences = self.sequences.write();
            if let Some(seq) = sequences.remove(request_id) {
                let text = self
                    .tokenizer
                    .decode(&seq.generated_tokens, true)
                    .unwrap_or_default();
                let api_response = api_response_from_generated_text(&seq.original_request, &text);

                // TPOT histogram (PLAYBOOK § 7 definition):
                //   tpot = (e2e − ttft) / (output_tokens − 1)
                // Only meaningful when first_emit_at is set (i.e. at
                // least one stream chunk landed) and ≥ 2 chunks were
                // emitted to give a non-degenerate decode window.
                if let (Some(first), Some(last)) = (seq.first_emit_at, seq.last_emit_at) {
                    if seq.emitted_chunks >= 2 {
                        let decode_s = last.duration_since(first).as_secs_f64();
                        let tpot_s = decode_s / (seq.emitted_chunks - 1) as f64;
                        histogram!("ferrum.engine.tpot_seconds").record(tpot_s);
                    }
                }

                let response = InferenceResponse {
                    request_id: request_id.clone(),
                    text,
                    tokens: seq.generated_tokens.clone(),
                    finish_reason,
                    usage: TokenUsage::new(seq.input_tokens.len(), seq.generated_tokens.len()),
                    latency_ms: seq.start_time.elapsed().as_millis() as u64,
                    created_at: chrono::Utc::now(),
                    metadata: HashMap::new(),
                    api_response,
                };

                let has_kv = seq.kv_cache.is_some();
                let cache_id = seq.model_cache_id.clone();
                (
                    response,
                    seq.stream_sender,
                    seq.response_sender,
                    has_kv,
                    cache_id,
                )
            } else {
                return Ok(());
            }
        };

        // Release model executor's KV cache for this sequence (frees GPU memory).
        if let Some(ref cache_id) = model_cache_id {
            self.model_executor.release_cache(cache_id);
        }

        if has_kv_cache {
            let _ = self.kv_cache.deallocate(request_id.clone()).await;
        }

        self.scheduler
            .complete(request_id.clone(), &response)
            .await?;

        if let Some(tx) = response_sender {
            let _ = tx.send(response.clone());
        }

        if let Some(tx) = stream_sender {
            let final_chunk = StreamChunk {
                request_id: request_id.clone(),
                text: String::new(),
                token: None,
                finish_reason: Some(finish_reason),
                usage: Some(response.usage.clone()),
                created_at: chrono::Utc::now(),
                metadata: HashMap::new(),
            };
            let _ = tx.send(Ok(final_chunk)).await;
        }

        debug!(
            "Request {} completed: {} tokens, {:?}",
            request_id,
            response.tokens.len(),
            finish_reason
        );

        Ok(())
    }
}

fn api_response_from_generated_text(
    request: &InferenceRequest,
    text: &str,
) -> Option<ferrum_types::ApiResponse> {
    let ferrum_types::ApiRequest::Chat(chat_request) = request.api_request.as_ref()? else {
        return None;
    };
    if api_tool_choice_is_none(chat_request) {
        return None;
    }

    if !chat_request.tools.is_empty() {
        let tool_calls = parse_tool_calls_from_generated_text(text, &chat_request.tools)?;
        if !tool_calls.is_empty() {
            return Some(ferrum_types::ApiResponse::Chat(
                ferrum_types::ApiChatResponse {
                    message: ferrum_types::ApiChatMessage {
                        role: ferrum_types::ApiMessageRole::Assistant,
                        content: String::new(),
                        name: None,
                        tool_calls,
                        tool_call_id: None,
                        function_call: None,
                    },
                    finish_reason: Some("tool_calls".to_string()),
                },
            ));
        }
    }

    if !chat_request.legacy_functions.is_empty() {
        let function_call =
            parse_legacy_function_call_from_generated_text(text, &chat_request.legacy_functions)?;
        return Some(ferrum_types::ApiResponse::Chat(
            ferrum_types::ApiChatResponse {
                message: ferrum_types::ApiChatMessage {
                    role: ferrum_types::ApiMessageRole::Assistant,
                    content: String::new(),
                    name: None,
                    tool_calls: Vec::new(),
                    tool_call_id: None,
                    function_call: Some(function_call),
                },
                finish_reason: Some("function_call".to_string()),
            },
        ));
    }

    None
}

fn api_tool_choice_is_none(chat_request: &ferrum_types::ApiChatRequest) -> bool {
    matches!(
        chat_request.tool_choice.as_ref(),
        Some(ferrum_types::ApiToolChoice::Mode(mode)) if mode.eq_ignore_ascii_case("none")
    )
}

fn parse_tool_calls_from_generated_text(
    text: &str,
    tools: &[ferrum_types::ApiTool],
) -> Option<Vec<ferrum_types::ApiToolCall>> {
    let value = parse_json_value_from_generated_text(text)?;
    if let Some(calls) = value.get("tool_calls").and_then(|value| value.as_array()) {
        let parsed = calls
            .iter()
            .enumerate()
            .filter_map(|(index, value)| parse_tool_call_value(value, index, tools))
            .collect::<Vec<_>>();
        return (!parsed.is_empty()).then_some(parsed);
    }
    if let Some(tool_call) = value.get("tool_call") {
        return parse_tool_call_value(tool_call, 0, tools).map(|call| vec![call]);
    }
    parse_tool_call_value(&value, 0, tools).map(|call| vec![call])
}

fn parse_tool_call_value(
    value: &serde_json::Value,
    index: usize,
    tools: &[ferrum_types::ApiTool],
) -> Option<ferrum_types::ApiToolCall> {
    let tool_type = value
        .get("type")
        .and_then(|value| value.as_str())
        .unwrap_or("function");
    if tool_type != "function" {
        return None;
    }
    let function = value.get("function").unwrap_or(value);
    let name = function.get("name").and_then(|value| value.as_str())?;
    if !tools.iter().any(|tool| tool.function.name == name) {
        return None;
    }
    let arguments = api_arguments_to_string(function.get("arguments"));
    let id = value
        .get("id")
        .and_then(|value| value.as_str())
        .map(str::to_string)
        .unwrap_or_else(|| format!("call_{index}"));

    Some(ferrum_types::ApiToolCall {
        id,
        tool_type: "function".to_string(),
        function: ferrum_types::ApiFunctionCall {
            name: name.to_string(),
            arguments,
        },
    })
}

fn parse_legacy_function_call_from_generated_text(
    text: &str,
    functions: &[ferrum_types::ApiFunction],
) -> Option<ferrum_types::ApiFunctionCall> {
    let value = parse_json_value_from_generated_text(text)?;
    let function = value.get("function_call").unwrap_or(&value);
    let name = function.get("name").and_then(|value| value.as_str())?;
    if !functions.iter().any(|function| function.name == name) {
        return None;
    }
    Some(ferrum_types::ApiFunctionCall {
        name: name.to_string(),
        arguments: api_arguments_to_string(function.get("arguments")),
    })
}

fn parse_json_value_from_generated_text(text: &str) -> Option<serde_json::Value> {
    let trimmed = strip_single_json_fence(text.trim());
    serde_json::from_str(trimmed).ok().or_else(|| {
        let start = trimmed.find('{')?;
        let end = trimmed.rfind('}')?;
        (start <= end)
            .then(|| serde_json::from_str(&trimmed[start..=end]).ok())
            .flatten()
    })
}

fn strip_single_json_fence(text: &str) -> &str {
    let Some(rest) = text.strip_prefix("```") else {
        return text;
    };
    let rest = rest.strip_prefix("json").unwrap_or(rest).trim_start();
    rest.strip_suffix("```").map(str::trim).unwrap_or(text)
}

fn api_arguments_to_string(arguments: Option<&serde_json::Value>) -> String {
    match arguments {
        Some(serde_json::Value::String(raw)) => raw.clone(),
        Some(value) => serde_json::to_string(value).unwrap_or_else(|_| "{}".to_string()),
        None => "{}".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chat_request_with_tool(
        tool_choice: Option<ferrum_types::ApiToolChoice>,
    ) -> InferenceRequest {
        InferenceRequest::new("rendered prompt", "mock-model").with_api_request(
            ferrum_types::ApiRequest::Chat(ferrum_types::ApiChatRequest {
                messages: vec![ferrum_types::ApiChatMessage {
                    role: ferrum_types::ApiMessageRole::User,
                    content: "Use the weather tool".to_string(),
                    name: None,
                    tool_calls: Vec::new(),
                    tool_call_id: None,
                    function_call: None,
                }],
                tools: vec![ferrum_types::ApiTool {
                    tool_type: "function".to_string(),
                    function: ferrum_types::ApiFunction {
                        name: "weather".to_string(),
                        description: None,
                        parameters: None,
                        strict: None,
                    },
                }],
                tool_choice,
                legacy_functions: Vec::new(),
                legacy_function_call: None,
                response_format: None,
                stream_options: None,
            }),
        )
    }

    #[test]
    fn generated_tool_call_json_becomes_structured_chat_response() {
        let request =
            chat_request_with_tool(Some(ferrum_types::ApiToolChoice::Mode("auto".to_string())));
        let text = r#"{"tool_calls":[{"id":"call_1","type":"function","function":{"name":"weather","arguments":{"city":"Paris"}}}]}"#;

        let Some(ferrum_types::ApiResponse::Chat(response)) =
            api_response_from_generated_text(&request, text)
        else {
            panic!("expected structured chat tool response");
        };

        assert_eq!(response.finish_reason.as_deref(), Some("tool_calls"));
        assert_eq!(response.message.content, "");
        assert_eq!(response.message.tool_calls.len(), 1);
        let call = &response.message.tool_calls[0];
        assert_eq!(call.id, "call_1");
        assert_eq!(call.function.name, "weather");
        assert_eq!(call.function.arguments, r#"{"city":"Paris"}"#);
    }

    #[test]
    fn tool_choice_none_keeps_generated_text_unstructured() {
        let request =
            chat_request_with_tool(Some(ferrum_types::ApiToolChoice::Mode("none".to_string())));
        let text = r#"{"name":"weather","arguments":{"city":"Paris"}}"#;

        assert!(api_response_from_generated_text(&request, text).is_none());
    }

    #[test]
    fn unregistered_tool_name_keeps_generated_text_unstructured() {
        let request =
            chat_request_with_tool(Some(ferrum_types::ApiToolChoice::Mode("auto".to_string())));
        let text = r#"{"name":"calendar","arguments":{"city":"Paris"}}"#;

        assert!(api_response_from_generated_text(&request, text).is_none());
    }

    #[test]
    fn generated_legacy_function_call_json_becomes_structured_chat_response() {
        let request = InferenceRequest::new("rendered prompt", "mock-model").with_api_request(
            ferrum_types::ApiRequest::Chat(ferrum_types::ApiChatRequest {
                messages: Vec::new(),
                tools: Vec::new(),
                tool_choice: None,
                legacy_functions: vec![ferrum_types::ApiFunction {
                    name: "weather".to_string(),
                    description: None,
                    parameters: None,
                    strict: None,
                }],
                legacy_function_call: None,
                response_format: None,
                stream_options: None,
            }),
        );
        let text = r#"```json
{"function_call":{"name":"weather","arguments":{"city":"Paris"}}}
```"#;

        let Some(ferrum_types::ApiResponse::Chat(response)) =
            api_response_from_generated_text(&request, text)
        else {
            panic!("expected structured legacy function response");
        };

        assert_eq!(response.finish_reason.as_deref(), Some("function_call"));
        let function_call = response.message.function_call.unwrap();
        assert_eq!(function_call.name, "weather");
        assert_eq!(function_call.arguments, r#"{"city":"Paris"}"#);
    }
}
