# OpenAI API Compatibility

Ferrum exposes OpenAI-shaped HTTP endpoints for local serving. This document
describes the current product contract for the always-on server path.

## Endpoints

| Endpoint | Status | Notes |
|---|---|---|
| `POST /v1/chat/completions` | Supported | Non-streaming and streaming chat responses. |
| `POST /v1/completions` | Supported | Non-streaming and streaming text completions with a single string `prompt`; prompt arrays/objects are rejected with `param=prompt`. |
| `GET /v1/models` | Supported | Lists models known to the server. |
| `POST /v1/embeddings` | Supported for embedding servers | Text and image embedding support depends on the loaded model. |
| `POST /v1/audio/transcriptions` | Supported for ASR servers | Multipart form input. |
| `POST /v1/audio/speech` | Supported for TTS servers | Speech output depends on the loaded TTS model. |

## Chat Fields

| Field | Status | Behavior |
|---|---|---|
| `model` | Supported | Required by OpenAI clients; routed to the loaded Ferrum model. |
| `messages` | Supported | `system`, `user`, `assistant`, `tool`, and legacy `function` roles parse into structured request data and are rendered by the chat-template layer. Assistant `tool_calls` / legacy `function_call` history is included in the rendered prompt for caller-owned tool-result loops. |
| string `content` | Supported | Rendered through the model-family chat template layer. |
| text content parts | Supported | `content: [{"type":"text","text":"..."}]` is accepted and concatenated. |
| multimodal content parts | Rejected | Non-text parts return HTTP 400 instead of being silently dropped. |
| `max_tokens` | Supported | Legacy completion budget. |
| `max_completion_tokens` | Supported | Overrides `max_tokens` when both are supplied. |
| `temperature`, `top_p` | Supported | Mapped into Ferrum sampling parameters. |
| `stop` | Supported | Accepts a string or string array and strips a trailing stop sentinel from returned text. |
| `stream` | Supported | Emits OpenAI-shaped SSE chunks followed by `[DONE]`. |
| `stream_options.include_usage` | Supported with `stream=true` | Emits a final usage chunk with `choices: []`; `stream_options` without streaming is rejected. |
| `n` | Restricted | Only `n=1` is supported; other values return HTTP 400 with `param=n`. |
| `logit_bias` | Rejected | Non-empty maps return HTTP 400 with `param=logit_bias`. |
| `logprobs` | Rejected | Returns HTTP 400 with `param=logprobs`. |
| `top_logprobs` | Rejected | Values greater than zero return HTTP 400 with `param=top_logprobs`. |
| `tools` | Partially supported | Function tool definitions parse, are carried through the structured request boundary, and are included in the rendered chat-template prompt. Engine output that emits matching tool-call JSON is returned as OpenAI `tool_calls` for non-streaming responses and streaming deltas; non-function tool types return HTTP 400 with `param=tools`. Tool execution is caller-owned, matching OpenAI/vLLM API semantics. |
| `tool_choice=auto/none` | Supported | Parsed and carried through structured request metadata. `none` keeps generated tool-call JSON as ordinary assistant content. |
| specific `tool_choice` | Supported | Selector objects such as `{"type":"function","function":{"name":"weather"}}` validate against declared tools, render into prompt context, and constrain generated JSON parsing to the selected tool. Undeclared tool names return HTTP 400 with `param=tool_choice`. |
| `tool_choice=required` | Rejected | Returns HTTP 400 with `param=tool_choice` until hard tool-call forcing is implemented. |
| legacy `functions` / `function_call=auto/none` | Supported | Parsed for SDK compatibility and carried through structured request data. Assistant `function_call` responses serialize in the legacy OpenAI shape, including non-streaming responses and streaming deltas when engine output emits matching function-call JSON. |
| specific legacy `function_call` | Supported | Named function-call selectors validate against declared legacy functions and constrain generated function-call JSON parsing to the selected function. Undeclared function names return HTTP 400 with `param=function_call`. |

## Structured Output

| Request | Status | Behavior |
|---|---|---|
| `response_format={"type":"text"}` | Supported | Default behavior. |
| `response_format={"type":"json_object"}` | Best-effort | Ferrum asks the model for JSON and repairs one outer markdown fence, but it does not hard-validate JSON at the HTTP boundary. Invalid JSON can still be returned. |
| `response_format={"type":"json_schema","json_schema":{"strict":true,...}}` | Supported for a subset | Supported schemas are validated before non-streaming responses return. Strict-schema streaming buffers content until validation passes, then emits the content and final chunk; invalid output emits an OpenAI-shaped SSE error without invalid partial deltas. Unsupported schema subsets are rejected at request validation with `param=response_format.json_schema`. |
| non-strict `json_schema` | Best-effort | Parsed and preserved, but strict validation only applies when `strict=true`. |
| unknown `response_format.type` | Rejected | Returns HTTP 400 with `param=response_format.type`. |

Strict schema support is intentionally conservative. It currently depends on
Ferrum's schema-to-regex subset; unsupported constructs should fail fast rather
than silently degrading to best-effort generation.

## Usage Accounting

Usage fields come from engine token accounting, not HTTP whitespace counting:

- `prompt_tokens` is produced by the model/tokenizer path.
- `completion_tokens` tracks generated or streamed tokens.
- `total_tokens = prompt_tokens + completion_tokens`.

When streaming and `stream_options.include_usage=true`, usage is emitted in a
separate final SSE chunk before `[DONE]`.

## Error Mapping

| Case | HTTP status | Error type |
|---|---:|---|
| Invalid request JSON or invalid field combination | 400 | `invalid_request_error` |
| Unsupported explicit feature | 400 | `invalid_request_error` |
| No compatible engine loaded for the endpoint | 503 | `service_unavailable_error` |
| Generation failure | 500 for non-streaming; OpenAI-shaped SSE error event plus `[DONE]` for streaming | `internal_server_error` |

Every explicit rejection should include the relevant OpenAI-style `param` when
the failing field is known.

## Test Evidence

The always-on compatibility path is covered by non-ignored Rust tests:

```bash
cargo test -q -p ferrum-server
cargo test -q -p ferrum-types requests_tests
```

The tracked status matrix with individual test names is maintained in
[`docs/status/openai-api-compat-2026-05-30.md`](status/openai-api-compat-2026-05-30.md).
Ignored SDK smoke tests exist for `async-openai` and the Python OpenAI SDK;
those require a real served model and are intended for manual GPU/Metal
validation.
