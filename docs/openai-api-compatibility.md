# OpenAI API Compatibility

Ferrum exposes OpenAI-shaped HTTP endpoints for local serving. This document
describes the current product contract for the always-on server path.

Ferrum v0.8 release support is language-model serving: chat completions, text
completions, and model discovery. Embedding, ASR, and TTS routes remain in the
source tree for legacy/specialized engines, but are outside the v0.8 release
support matrix and are hidden from the default CLI help.

## Endpoints

| Endpoint | Status | Notes |
|---|---|---|
| `POST /v1/chat/completions` | Supported | Non-streaming and streaming chat responses. |
| `POST /v1/responses` | Supported, stateless | Ordered text/reasoning/tool history, non-streaming and streaming output, usage, and caller-owned function/namespace-tool loops. |
| `POST /v1/completions` | Supported | Non-streaming and streaming text completions with a single string `prompt`; prompt arrays/objects are rejected with `param=prompt`. |
| `GET /v1/models` | Supported | Lists models known to the server. |
| `POST /v1/embeddings` | Experimental / outside v0.8 release scope | Text and image embedding support depends on a specialized loaded model. |
| `POST /v1/audio/transcriptions` | Experimental / outside v0.8 release scope | Multipart form input for specialized ASR engines. |
| `POST /v1/audio/speech` | Experimental / outside v0.8 release scope | Speech output depends on a specialized TTS engine. |

## Modality Endpoint Fields

| Endpoint / field | Status | Behavior |
|---|---|---|
| `/v1/embeddings input` | Supported | Accepts a string, array of strings, object with `text` or `image`, or array of those objects. |
| `/v1/embeddings encoding_format=float` | Supported | Returns numeric float embeddings. |
| `/v1/embeddings encoding_format=base64` | Rejected | Returns HTTP 400 with `param=encoding_format`; base64 embedding encoding is not implemented. |
| `/v1/audio/transcriptions response_format=json` | Supported | Returns `{ "text": ... }`. |
| `/v1/audio/transcriptions` other `response_format` values | Rejected | Returns HTTP 400 with `param=response_format`; text/SRT/VTT/verbose JSON transcription formats are not implemented. |
| `/v1/audio/speech response_format=wav` | Supported | Returns 16-bit mono WAV bytes with `content-type: audio/wav`. |
| `/v1/audio/speech response_format=pcm` | Supported | Returns raw 16-bit little-endian mono PCM bytes with `content-type: audio/pcm`. |
| `/v1/audio/speech` other `response_format` values | Rejected | Returns HTTP 400 with `param=response_format`; compressed speech formats are not implemented. |

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
| `top_k`, `min_p`, `repetition_penalty` | Supported extension | vLLM-compatible sampling fields. `top_k=-1/0` and `min_p=0` disable their filters. |
| `stop` | Supported | Accepts a string or string array and strips a trailing stop sentinel from returned text. |
| `stream` | Supported | Emits OpenAI-shaped SSE chunks followed by `[DONE]`. |
| `stream_options.include_usage` | Supported with `stream=true` | Emits a final usage chunk with `choices: []`; `stream_options` without streaming is rejected. |
| `chat_template_kwargs.enable_thinking` | Supported when the model template reads it | Boolean vLLM-compatible chat-template variable. Ferrum forwards it to the model-provided template. `ferrum serve --enable-thinking` or `--disable-thinking` sets the default for omitted requests; the request value wins. Templates that do not use `enable_thinking` are unaffected; non-boolean values return HTTP 400. |
| `n` | Restricted | Only `n=1` is supported; other values return HTTP 400 with `param=n`. |
| `logit_bias` | Rejected | Non-empty maps return HTTP 400 with `param=logit_bias`. |
| `logprobs` | Rejected | Returns HTTP 400 with `param=logprobs`. |
| `top_logprobs` | Rejected | Values greater than zero return HTTP 400 with `param=top_logprobs`. |
| `tools` | Partially supported | Function tool definitions parse, are carried through the structured request boundary, and are included in the rendered chat-template prompt. Engine output that emits matching tool-call JSON is returned as OpenAI `tool_calls` for non-streaming responses and streaming deltas; non-function tool types return HTTP 400 with `param=tools`. Tool execution is caller-owned, matching OpenAI/vLLM API semantics. |
| `tool_choice=auto/none` | Supported | Parsed and carried through structured request metadata. `none` keeps generated tool-call JSON as ordinary assistant content. |
| specific `tool_choice` | Supported | Selector objects such as `{"type":"function","function":{"name":"weather"}}` validate against declared tools, render into prompt context, and constrain generated JSON parsing to the selected tool. Undeclared tool names return HTTP 400 with `param=tool_choice`. |
| `tool_choice=required` | Supported | Requires at least one function tool. Ferrum steers generation toward the first declared tool's argument schema and returns OpenAI-shaped `tool_calls`. If no valid tool call can be parsed, non-streaming requests return HTTP 400 with `param=tool_choice`; streaming requests emit an OpenAI-shaped SSE error and `[DONE]` without first leaking invalid content. |
| legacy `functions` / `function_call=auto/none` | Supported | Parsed for SDK compatibility and carried through structured request data. Assistant `function_call` responses serialize in the legacy OpenAI shape, including non-streaming responses and streaming deltas when engine output emits matching function-call JSON. |
| specific legacy `function_call` | Supported | Named function-call selectors validate against declared legacy functions and constrain generated function-call JSON parsing to the selected function. Undeclared function names return HTTP 400 with `param=function_call`. |

Both Chat Completions and Responses first render system and developer messages
in their original positions. If a model-owned template explicitly rejects a
non-leading system message, the server retries with system text coalesced in
wire order into one leading message. This compatibility retry is enabled by
default. Operators can disable it with
`ferrum serve --disable-interleaved-system-coalescing` or set
`server.interleaved_system_coalescing = false` in `ferrum.toml`; Ferrum then
returns the model-owned template's original error.

## Stateless Responses API

`POST /v1/responses` reuses the same model, tokenizer, sampling, structured-output,
and function-call path as Chat Completions. Input may be a string or an ordered
array containing `message`, readable `reasoning`, `function_call`, and
`function_call_output` items. The caller owns the loop and resends the complete
ordered history on each turn; Ferrum correlates tool results by `call_id`.

Assistant message `phase` values (`commentary` and `final_answer`) are validated,
preserved in caller-owned history, and exposed to model chat templates. Generated
text before a function call is labelled `commentary`; terminal text is labelled
`final_answer`. A streaming `response.output_item.added` omits this optional
field until later tool calls are known, while `response.output_item.done` and the
terminal response contain the resolved phase.

The endpoint accepts `instructions`, `max_output_tokens`, `temperature`, `top_p`,
`stream`, function and namespace `tools`, `tool_choice`, `parallel_tool_calls`, supported
`reasoning.effort` values, `text.format`, `prompt_cache_key`, and opaque client
metadata. `include=["reasoning.encrypted_content"]` is accepted, but local model
reasoning has no provider-owned encrypted state, so generated items return
`encrypted_content: null`. An encrypted-only reasoning item cannot be replayed
and is rejected instead of silently losing context.

Supported namespace tools contain nested function tools; `allowed_callers` and
`defer_loading` are rejected. `text.format` supports `text`, `json_object`, and
`json_schema`; `text.verbosity` and non-empty JSON Schema descriptions are
rejected because local model templates cannot preserve those semantics.

Namespace tools are flattened only inside Ferrum's Chat-template bridge. A
request-local, collision-safe mapping restores the original `namespace` and
child `name` in every Responses output item and during caller-owned replay;
internal aliases are never part of the public Responses contract.

Streaming uses typed Responses lifecycle events with contiguous sequence numbers.
It emits the applicable semantic terminal event (`response.completed`,
`response.incomplete`, or `response.failed`) and then the transport sentinel
`[DONE]`.

Ferrum does not store Response objects. Requests for `store=true`,
`previous_response_id`, conversations, background execution, built-in tools, or
remote MCP return HTTP 400 with the failing field in `param`. Tool execution
remains caller-owned.

## Structured Output

| Request | Status | Behavior |
|---|---|---|
| `response_format={"type":"text"}` | Supported | Default behavior. |
| `response_format={"type":"json_object"}` | Hard constrained | Ferrum applies tokenizer-aware constrained decoding and final exact validation. The result must be one JSON object with no markdown fence or surrounding text. Ferrum does not repair malformed model output. |
| `response_format={"type":"json_schema","json_schema":{"strict":true,...}}` | Hard constrained | Ferrum compiles the supplied JSON Schema into the same tokenizer-aware constrained-decoding runtime and validates the exact final JSON value against the schema. There is no silent fallback to prompt-only generation. |
| non-strict `json_schema` | Best-effort | Parsed and preserved, but strict validation only applies when `strict=true`. |
| unknown `response_format.type` | Rejected | Returns HTTP 400 with `param=response_format.type`. |

Hard structured streams are buffered until final validation passes, so malformed
partial JSON is not emitted to the client. If a grammar cannot be compiled, the
request fails before admission. If generation reaches a token or context limit
without a complete valid value, non-streaming requests fail and streaming
requests emit an OpenAI-shaped SSE error followed by one `[DONE]`.

Schema support follows the embedded grammar compiler rather than Ferrum's old
schema-to-regex subset. Constructs such as `oneOf` are passed to that compiler;
schemas it cannot support fail explicitly instead of degrading to best-effort
generation.

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

The always-on compatibility path is covered by Rust unit and route tests:

```bash
cargo test -q -p ferrum-server
cargo test -q -p ferrum-types requests_tests
```
