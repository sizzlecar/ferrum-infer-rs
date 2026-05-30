# OpenAI API Compatibility Status - 2026-05-30

This is a focused status packet for the product API work in
`docs/dev-loop-product-api-goal-2026-05-30.md`. It documents the current
server-side contract tests added for the deterministic stub path; it is not a
claim that Milestone F or G is complete.

## Covered By Non-Ignored Tests

All entries below are covered by `cargo test -q -p ferrum-server`.

The product-facing compatibility matrix is documented in
[`docs/openai-api-compatibility.md`](../openai-api-compatibility.md), including
explicit supported/rejected fields, usage accounting, error mapping, and the
`json_object` best-effort caveat.

| Surface | Current behavior | Evidence |
|---|---|---|
| Basic chat | `/v1/chat/completions` returns OpenAI chat envelope with assistant message and usage | `route_basic_chat_contract_uses_stub_engine` |
| Chat prompt rendering layer | HTTP handlers delegate model-family prompt rendering to `ferrum-server::chat_template`, preserving Qwen, Llama 3, and fallback templates | `qwen3_renders_chatml_with_think_marker`, `qwen2_renders_chatml_without_think`, `llama3_renders_header_format`, `unknown_model_uses_tinyllama_fallback` |
| Token usage accounting | Chat/completions usage is populated from engine `TokenUsage`, not HTTP whitespace counts | `route_basic_chat_contract_uses_stub_engine`, `route_completions_contract_uses_stub_engine`, `chat_stream_options_include_usage_controls_stream_usage`, `streaming_chat_does_not_synthesize_whitespace_usage`, `streaming_completions_do_not_synthesize_whitespace_usage` |
| Engine tokenizer usage fixture | Engine `TokenUsage.prompt_tokens` equals tokenizer-encoded prompt length, and `completion_tokens` equals generated/emitted token count | `non_streaming_usage_matches_tokenizer_and_generated_tokens`, `streaming_final_usage_matches_tokenizer_and_emitted_tokens` in `crates/ferrum-engine/tests/continuous_batch_test.rs`; `usage_counts_byte_tokenizer_prompt_tokens_not_words` in `crates/ferrum-engine/tests/regex_guided_test.rs` |
| Streaming chat | SSE returns chat chunks and `[DONE]` | `route_streaming_chat_include_usage_contract` |
| `stream_options.include_usage` | Emits a separate final usage chunk with `choices: []` before `[DONE]` | `route_streaming_chat_include_usage_contract`, `chat_stream_options_include_usage_controls_stream_usage` |
| `max_completion_tokens` | Chat accepts OpenAI's newer completion budget field and lets it override legacy `max_tokens` when both are present | `chat_accepts_stop_string_and_max_completion_tokens` |
| `stop` string or array | Chat and legacy completions accept `stop` as either one string or an array and strip a trailing stop suffix from returned text | `chat_accepts_stop_string_and_max_completion_tokens`, `stop_string_strips_chat_and_completion_suffixes` |
| `n=2` | Rejected with HTTP 400, `type=invalid_request_error`, `param=n` | `chat_rejects_n_not_one_with_openai_error_param` |
| Structured internal API request | `InferenceRequest` now carries `api_request` for chat/completion semantics alongside the rendered prompt | `inference_request_can_carry_structured_chat_api_request`, `route_tool_request_reaches_engine_structured_boundary`, `tool_requests_and_tool_messages_parse_into_structured_api_request` |
| Structured internal API response | `InferenceResponse` can carry `api_response` for assistant tool-call responses without overloading text metadata | `inference_response_can_carry_structured_chat_tool_call`, `route_chat_serializes_structured_tool_call_response` |
| `tools` request parsing | Function tools parse through `/v1/chat/completions` and are carried in structured `api_request` plus compatibility metadata | `route_tool_request_reaches_engine_structured_boundary`, `tool_requests_and_tool_messages_parse_into_structured_api_request` |
| Non-function tools | Non-function tool types reject explicitly instead of being accepted and ignored | `route_rejects_non_function_tools_with_openai_error_param` |
| Tool role messages | `role=tool` parses through `/v1/chat/completions`, is rendered into the chat template, and remains available in structured `api_request` | `route_tool_request_reaches_engine_structured_boundary`, `tool_requests_and_tool_messages_parse_into_structured_api_request` |
| Legacy `functions` / `function_call=auto` | Legacy function metadata parses and reaches the structured internal request | `route_tool_request_reaches_engine_structured_boundary`, `legacy_function_role_messages_parse_into_structured_api_request` |
| Legacy `role=function` messages | Legacy function result messages parse, render through the chat-template layer, and remain available in structured `api_request` | `legacy_function_role_messages_parse_into_structured_api_request`, `fallback_preserves_legacy_function_and_tool_roles` |
| Assistant `tool_calls` serialization | Assistant message serializes OpenAI `tool_calls[]` shape; non-streaming chat responses can serialize structured internal tool-call responses | `assistant_tool_call_serializes_openai_shape`, `route_chat_serializes_structured_tool_call_response` |
| Assistant `function_call` serialization | Legacy assistant `function_call` responses serialize in the OpenAI shape with `finish_reason=function_call` | `route_chat_serializes_legacy_function_call_response` |
| Unsupported multimodal content | Non-text content parts are rejected at the HTTP boundary with 400 instead of dropped | `route_rejects_multimodal_content_with_400` |
| `logit_bias` | Non-empty `logit_bias` is rejected with HTTP 400 and `param=logit_bias` | `route_rejects_logit_bias_with_openai_error_param`, `chat_rejects_logit_bias_and_logprobs_explicitly` |
| `logprobs` / `top_logprobs` | Rejected with HTTP 400 and field-specific `param` | `chat_rejects_logit_bias_and_logprobs_explicitly` |
| Forced tool/function selection | Forced `tool_choice` and legacy `function_call` reject with field-specific HTTP 400 | `route_rejects_forced_tool_and_function_selection` |
| Engine unavailable | Missing LLM engine returns HTTP 503 with `service_unavailable_error` for chat and completions | `route_chat_engine_unavailable_maps_to_503`, `route_completions_engine_unavailable_maps_to_503` |
| Generation failure | Non-streaming LLM generation failure returns HTTP 500 with `internal_server_error`; streaming start/chunk failures emit OpenAI-shaped SSE error events followed by `[DONE]` | `route_chat_generation_failure_maps_to_500`, `route_completions_generation_failure_maps_to_500`, `route_chat_stream_generation_failure_emits_openai_error_event`, `route_chat_stream_chunk_failure_emits_openai_error_event`, `route_completions_stream_generation_failure_emits_openai_error_event`, `route_completions_stream_chunk_failure_emits_openai_error_event` |
| `/v1/chat/completions` invalid JSON | Malformed request JSON maps to an OpenAI error envelope instead of Axum's default rejection body | `route_chat_invalid_json_maps_to_openai_error` |
| `/v1/completions` | Implemented for non-streaming stub LLM path with OpenAI text-completion envelope | `route_completions_contract_uses_stub_engine`, `completions_endpoint_uses_stub_engine` |
| `/v1/completions` streaming | Emits text-completion SSE chunks, a separate usage chunk, and `[DONE]` | `route_completions_streaming_contract_uses_stub_engine` |
| `/v1/completions` invalid JSON | Malformed or schema-invalid request JSON maps to OpenAI error envelope instead of Axum's default rejection body | `route_completions_invalid_json_maps_to_openai_error` |
| `/v1/completions` unsupported fields | `n != 1`, `logprobs`, non-empty `logit_bias`, and non-string or missing `prompt` reject explicitly with field-specific `param` | `route_completions_rejects_unsupported_fields_explicitly`, `route_completions_rejects_non_string_prompt_with_field_param` |
| `response_format=json_object` | Best-effort JSON mode; non-streaming responses strip one outer markdown fence but are not schema/JSON validated at the HTTP boundary | `json_object_strips_single_markdown_fence_as_best_effort_repair`, `json_object_remains_best_effort_not_strict_validation` |
| Non-strict `response_format=json_schema` | Parsed and preserved in structured request data, but does not enable hard guided decoding or strict response validation | `non_strict_json_schema_is_preserved_but_not_hard_masked` |
| Unsupported strict JSON Schema | `response_format.json_schema.strict=true` rejects unsupported schema subsets at request boundary | `unsupported_strict_json_schema_is_rejected_at_boundary` |
| Missing strict JSON Schema body | `response_format.type=json_schema` with no inner `schema` parses far enough to return HTTP 400 with `param=response_format.json_schema` | `missing_json_schema_schema_rejects_with_field_param` |
| Unknown response format type | Unknown `response_format.type` values reject with HTTP 400 and `param=response_format.type` | `route_rejects_unknown_response_format_type_with_openai_error_param` |
| Strict JSON response validation | Supported strict schema parses returned JSON and validates non-streaming output before returning; streaming buffers strict-schema content until validation passes, then emits the content plus final chunk, or emits an OpenAI-shaped SSE error without invalid partial deltas | `strict_json_schema_validates_non_streaming_response`, `strict_json_schema_invalid_model_output_fails_before_response`, `strict_json_schema_does_not_rely_on_markdown_fence_stripping`, `strict_json_schema_validates_streaming_final_response`, `strict_json_schema_invalid_streaming_output_emits_error_event` |
| Strict JSON deterministic repeat | Supported strict schema passes 100 consecutive deterministic route runs without invalid JSON | `route_strict_json_schema_supported_schema_passes_100_runs` |

## Current Explicit Rejections

| Field / combination | Status | Error shape |
|---|---|---|
| `n != 1` | Unsupported | HTTP 400, `invalid_request_error`, `param=n` |
| non-empty `logit_bias` | Unsupported | HTTP 400, `invalid_request_error`, `param=logit_bias` |
| `logprobs=true` | Unsupported | HTTP 400, `invalid_request_error`, `param=logprobs` |
| `top_logprobs > 0` | Unsupported | HTTP 400, `invalid_request_error`, `param=top_logprobs` |
| `/v1/completions logprobs` | Unsupported | HTTP 400, `invalid_request_error`, `param=logprobs` |
| malformed `/v1/completions` JSON | Invalid request | HTTP 400, `invalid_request_error`, `param=null` |
| `/v1/completions` missing/non-string `prompt` | Invalid request | HTTP 400, `invalid_request_error`, `param=prompt` |
| malformed `/v1/chat/completions` JSON | Invalid request | HTTP 400, `invalid_request_error`, `param=null` |
| `stream_options` with `stream != true` | Invalid request | HTTP 400, `invalid_request_error`, `param=stream_options` |
| non-function tools | Unsupported | HTTP 400, `invalid_request_error`, `param=tools` |
| forced or `required` `tool_choice` | Unsupported generation behavior | HTTP 400, `invalid_request_error`, `param=tool_choice` |
| forced legacy `function_call` | Unsupported generation behavior | HTTP 400, `invalid_request_error`, `param=function_call` |
| unknown `response_format.type` | Invalid request | HTTP 400, `invalid_request_error`, `param=response_format.type` |
| unsupported strict schema subset | Unsupported | HTTP 400, `invalid_request_error`, `param=response_format.json_schema` |
| missing `response_format.json_schema.schema` | Invalid request | HTTP 400, `invalid_request_error`, `param=response_format.json_schema` |

## Ignored Real-Model SDK Smokes

These tests are ignored because they load a local model and are intended for
manual GPU/Metal validation, not the always-on stub path.

| SDK | Coverage | Test |
|---|---|---|
| `async-openai` | Basic chat response parsing | `test_openai_client_chat_basic` |
| `async-openai` | Chat SSE parsing | `test_openai_client_chat_streaming` |
| `async-openai` | Typed tool request fields plus `stream_options.include_usage` final usage chunk | `test_openai_client_tools_stream_options_include_usage` |
| `async-openai` | `response_format=json_object` best-effort parseable JSON smoke | `test_openai_client_response_format_json_object` |
| `async-openai` | Strict `json_schema` simple object, 20/20 runs at temperature 0 | `test_openai_client_strict_json_schema_20_runs` |
| `async-openai` | Multi-turn typed message builders | `test_openai_client_multi_turn` |
| Python OpenAI SDK | Non-streaming chat plus streaming chat with `stream_options.include_usage` | `test_python_openai_sdk_chat_and_stream_smoke` |

## Known Remaining Gaps

- Always-on engine tests cover tokenizer-backed mock-executor paths, including a byte-level tokenizer fixture that differs from whitespace counting. Real-model tokenizer-vs-usage fixture tests are still needed outside the mock executor path.
- Tool-call generation is not implemented; only parsing, serialization, structured request/response carry, compatibility metadata carry, and explicit rejection of forced selection are covered.
- Engines still consume `InferenceRequest.prompt`; `api_request` preserves product semantics for the next structured generation boundary but does not yet replace prompt execution.
- Strict schema validation is regex-subset based; strict-schema streaming now
  buffers content until final validation passes, so invalid partial deltas are
  not sent, but strict streaming no longer has token-by-token latency.
- `response_format=json_object` remains best-effort.
- Real-model SDK tests remain ignored; deterministic route tests cover the always-on path. The strict-schema 20-run smoke exists but has not been executed in the always-on local gate.
- The Python OpenAI SDK smoke requires `python3 -m pip install openai` and can
  be pointed at a specific Python with `FERRUM_PYTHON`.
