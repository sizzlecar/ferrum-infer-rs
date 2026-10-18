# OpenAI API Compatibility

Ferrum exposes OpenAI-shaped HTTP endpoints for local serving. This document
describes the current product contract for the always-on server path.

Ferrum serves text language models through chat completions, Responses, text
completions, and model discovery. Experimental standalone embedding, ASR, and
TTS implementations have been retired. Their HTTP routes return 404; the
`embed`, `transcribe`, and `tts` CLI commands are no longer available.

This also removes the experimental embedding/speech engines, executors,
task traits, and dedicated model loaders from the Rust API. Text generation
continues through `run`, `serve`, and the LLM engine interfaces; future tasks
will define their own contracts at the model registration and assembly boundary.

## Endpoints

| Endpoint | Status | Notes |
|---|---|---|
| `POST /v1/chat/completions` | Supported | Non-streaming and streaming chat responses. |
| `POST /v1/responses` | Supported, stateless | Ordered text/reasoning/tool history, non-streaming and streaming output, usage, and caller-owned function/namespace-tool loops. |
| `POST /v1/completions` | Supported | Non-streaming and streaming text completions with a single string `prompt`; prompt arrays/objects are rejected with `param=prompt`. |
| `GET /v1/models` | Supported | Lists public model names and optional effective serving capacity. |
| `POST /v1/embeddings` | Removed | Returns HTTP 404; requests are not routed to a text model. |
| `POST /v1/audio/transcriptions` | Removed | Returns HTTP 404. |
| `POST /v1/audio/speech` | Removed | Returns HTTP 404. |

## Output Limits

`run` and `serve` share these optional `ferrum.toml` settings (shown with defaults):

```toml
[output]
max_sessions = 2048
engine_queue_items = 100
semantic_bytes_total = 268435456
semantic_bytes_per_session = 33554432
queued_bytes_total = 67108864
encoded_bytes_total = 67108864
max_event_bytes = 33554432

[tokenizer_decode_cache]
max_entries = 1000
max_bytes = 67108864
```

Output limits must be positive; per-session content and individual encoded events
must fit their respective per-engine shared pools. Generated history, completion results,
cached assistant messages and CLI history retain their content credit until their
last owner is dropped. A session means one inference output, not a conversation:
retaining several turns can occupy several sessions. Exhausted session capacity
rejects new requests before model admission. Persistent content growth fails
explicitly when full; slow wire consumers pause their own generation. Each session
has one separate error allowance of at most 1 KiB.

Semantic limits count logical retained payload; queued/encoded limits count wire
content. Each tokenizer instance has its own token-ID plus UTF-8 cache byte limit;
either cache limit set to zero disables caching. Oversized entries use ordinary
decoding without entering the cache. These limits exclude tokenizer/parser scratch,
allocation capacity/overhead and transport buffers, so they are not a process-memory
limit. Library callers configure `EngineConfig.output` and
`EngineConfig.model.tokenizer.decode_cache`; borrow `InferenceOutput::response()`
or `OutputChunk::chunk()`. Cloning `InferenceOutput` shares its content and credit.

## Profile Delivery Limits

`run` and `serve` share optional journal limits in `ferrum.toml` (defaults shown):

```toml
[profile_journal]
queue_capacity = 256
payload_capacity_bytes = 16777216
max_record_bytes = 1048576
max_batch_bytes = 4194304
max_batch_records = 1024
```

Limits apply per physical journal. Values must be positive, with
`max_record_bytes <= max_batch_bytes <= payload_capacity_bytes`. Record sizes
include the trailing newline; batches are accepted or rejected whole. The payload
budget covers measured JSONL bytes retained by nonblocking delivery, including
active writer work. It excludes object/allocation overhead and is not an RSS limit.
Startup/shutdown profile records use the same bounded queue. Explicit shutdown
waits for the original writer to flush and finish.

Explicit settings for a shared journal path must match. Library callers use
`EngineConfig.runtime.profile_journal` and `ServerConfig.profile_journal`.
Library components sharing a file take separate producer scopes from one
`JsonlJournalOwner`, through `EngineBuilder::with_journal_scopes` and
`AxumServer::with_journal_scopes`. Each component finishes its own scope; the
owner closes the file after all scopes and final records finish. Opening a second
independent owner for the same live file is a configuration error. Check both the
close result and the returned receipt's `result()` for writer failures.
Optional telemetry loss is reported without replaying inference. Replay, Verify
and Full require complete evidence; configuration, encoding and internal contract
failures are never treated as optional delivery loss.

`run` and `serve --profile-detail off --profile-jsonl ...` write the standard
`FerrumProfileEvent` product schema through those same bounded journals. Off
keeps device timing and detailed execution facts disabled; explicit artifacts
still contain startup, request/output and close observations. Aliased profile
and scheduler paths share one writer and producer scopes.

Serve's `--profile-commit-sha`, `--profile-env-hash`, `--profile-model`,
`--profile-concurrency` and `--profile-runtime-flags-json` are preserved as
`declared_metadata` in one `product_session_metadata` event per invocation.
These labels do not override actual model identity, execution geometry or
capacity. The environment-only versions of these metadata controls and the old
process-global profiler/trace switches are rejected, including explicitly
forced-off values. Historical native-profile files remain readable by the
historical parser; new product artifacts do not use that schema.

The optional Engine batch diagnostic (`runtime.batch_decode_prof`) writes
`iter_prof` and `stream_ttft_prof` through its own instance sink when a typed
journal is configured. First-text duration retains the original request ingress
anchor and ends when the Engine commits nonempty text to output. It is not
client-observed SSE latency. Retired legacy per-operator/graph/Marlin/MoE
profiling switches do not implicitly enable detailed profiles on the retained
executor.

## Model Capacity

For loaded language models, each model card includes `max_model_len` when the
engine reports its effective per-request capacity. This is the same input-plus-output
limit used by request admission, including runtime and executor capacity constraints.
Public aliases and LoRA adapters report their base engine's limit. Unreported capacity
is omitted; it is not inferred from the model name or nominal weights context
window. This discovery field affects `serve`;
`run` continues to use the same existing engine admission boundary.

LoRA aliases require an execution implementation that applies adapters per
sequence. Single-device LlamaFamily supports this with prefix reuse disabled;
its token-only prefix cache does not distinguish adapters. Typed plan execution
and layer-split execution currently reject adapter registration and requests.
Unsupported or malformed adapter selections never fall back to
base-model inference. Custom Rust engines/executors must explicitly declare
adapter execution support; exposing alias metadata alone is insufficient.

## Chat Fields

| Field | Status | Behavior |
|---|---|---|
| `model` | Supported | Required by OpenAI clients; routed to the loaded Ferrum model. |
| `messages` | Supported | `system`, `user`, `assistant`, `tool`, and legacy `function` roles parse into structured request data and are rendered by the chat-template layer. Assistant `tool_calls` / legacy `function_call` history is included in the rendered prompt for caller-owned tool-result loops. |
| string `content` | Supported | Rendered through the model-family chat template layer. |
| text content parts | Supported | `content: [{"type":"text","text":"..."}]` is accepted and concatenated. |
| assistant `reasoning` / `reasoning_content` | Supported extension | Historical reasoning accepts either field. A string in canonical `reasoning` wins, including an empty string; missing or null `reasoning` falls back to `reasoning_content`. The selected value must be a string or null. The model's chat template decides whether to retain it in the prompt. Chat response messages and SSE deltas emit only canonical `reasoning`. |
| multimodal content parts | Rejected | Non-text parts return HTTP 400 instead of being silently dropped. |
| `max_tokens` | Supported | Legacy completion budget. |
| `max_completion_tokens` | Supported | Overrides `max_tokens` when both are supplied. |
| `temperature`, `top_p` | Supported | Mapped into Ferrum sampling parameters. |
| `top_k`, `min_p`, `repetition_penalty` | Supported extension | vLLM-compatible sampling fields. `top_k=-1/0` and `min_p=0` disable their filters. |
| `stop` | Supported | Accepts a string or string array. Both streaming and non-streaming text end before the first matched stop, including matches across tokens or inside a token. Streaming holds possible stop prefixes and releases unmatched text when generation ends. Empty stop strings are ignored. |
| `stream` | Supported | Emits OpenAI-shaped SSE chunks followed by `[DONE]`. |
| `stream_options.include_usage` | Supported with `stream=true` | Emits a final usage chunk with `choices: []`; `stream_options` without streaming is rejected. |
| `reasoning_effort` | Supported according to model capabilities | Standard reasoning control, shared with Responses `reasoning.effort`; see the behavior below. |
| `chat_template_kwargs.enable_thinking` | Supported when the model template reads it | Boolean vLLM-compatible chat-template variable. Ferrum forwards it to the model-provided template. `ferrum serve --enable-thinking` or `--disable-thinking` sets the default for omitted requests; explicit request controls win. Templates that do not use `enable_thinking` are unaffected; null acts as omission, and other non-boolean values return HTTP 400. |
| `chat_template_kwargs.reasoning_effort` | Supported extension | Passes a recognized effort string to the model template. Used alone, this legacy extension does not derive an `enable_thinking` value or override its server default. |
| `n` | Restricted | Only `n=1` is supported; other values return HTTP 400 with `param=n`. |
| `logit_bias` | Rejected | Non-empty maps return HTTP 400 with `param=logit_bias`. |
| `logprobs` | Rejected | Returns HTTP 400 with `param=logprobs`. |
| `top_logprobs` | Rejected | Values greater than zero return HTTP 400 with `param=top_logprobs`. |
| `tools` | Partially supported | Function tool definitions parse, are carried through the structured request boundary, and are included in the rendered chat-template prompt. Engine output that emits matching tool-call JSON is returned as OpenAI `tool_calls` for non-streaming responses and streaming deltas; non-function tool types return HTTP 400 with `param=tools`. Tool execution is caller-owned, matching OpenAI/vLLM API semantics. |
| `tool_choice=auto/none` | Supported | Parsed and carried through structured request metadata. `none` keeps ordinary tool-call JSON as assistant content. A native protocol tool call that violates `none` is rejected as a generation error (HTTP 500, or an SSE error followed by `[DONE]`) without returning the call. |
| specific `tool_choice` | Supported | Selector objects such as `{"type":"function","function":{"name":"weather"}}` validate against declared tools, render into prompt context, and constrain generated JSON parsing to the selected tool. Undeclared tool names return HTTP 400 with `param=tool_choice`. |
| `tool_choice=required` | Supported | Requires at least one function tool. Ferrum steers generation toward the first declared tool's argument schema and returns OpenAI-shaped `tool_calls`. If no valid tool call can be parsed, non-streaming requests return HTTP 400 with `param=tool_choice`; streaming requests emit an OpenAI-shaped SSE error and `[DONE]` without first leaking invalid content. |
| legacy `functions` / `function_call=auto/none` | Supported | Parsed for SDK compatibility and carried through structured request data. Assistant `function_call` responses serialize in the legacy OpenAI shape, including non-streaming responses and streaming deltas when engine output emits matching function-call JSON. |
| specific legacy `function_call` | Supported | Named function-call selectors validate against declared legacy functions and constrain generated function-call JSON parsing to the selected function. Undeclared function names return HTTP 400 with `param=function_call`. |

Chat `reasoning_effort` and Responses `reasoning.effort` accept `none`, `minimal`,
`low`, `medium`, `high`, `xhigh`, and `max`. Omission or null preserves the server
and template defaults. An explicit `none` passes both `reasoning_effort="none"`
and `enable_thinking=false` to the template; other explicit values pass the
requested effort and `enable_thinking=true`, overriding the server default.
Conflicting standard and extension controls return HTTP 400.

Accepted values depend on the loaded model. Ferrum rejects a standard effort
outside a model's declared support set before inference; for example, GPT-OSS
declares `low`, `medium`, and `high`. If support is unknown, Ferrum forwards the
controls and preserves the model template's behavior. Templates may reject a
value with HTTP 400. A template that only consumes the thinking switch gets
on/off control, without a guarantee of distinct computation budgets for each
positive effort. Templates that consume neither variable remain unchanged.
Legacy-only template kwargs retain their existing direct template semantics.

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
`[DONE]`. If an event exceeds the encoding limit, the stream instead ends with
`response.failed` and no `[DONE]`.

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
A user stop that cuts inside the constrained JSON value is also a generation
error; a stop after the complete value can terminate normally.

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
| Input plus requested output exceeds the effective model/KV context capacity | 400 | `invalid_request_error` with `code=context_length_exceeded` |
| No compatible engine loaded for the endpoint | 503 | `service_unavailable_error` |
| Generation failure | 500 before a response/stream is established; OpenAI-shaped SSE error event plus `[DONE]` after streaming starts | `internal_server_error` |

Every explicit rejection should include the relevant OpenAI-style `param` when
the failing field is known.

Context-capacity rejection uses the same structured code on Chat Completions,
Completions, and Responses, including requests with `stream=true`. The rejection
happens before generation and before SSE headers; clients must shorten the input
or change the requested output budget before retrying. Ordinary validation errors
do not receive this code. The shared engine uses the same capacity boundary for
`ferrum run`, which reports the capacity and requested input/output counts as a
CLI error. Failures after a stream starts cannot certify that no generation took
place and must not be treated as a pre-generation capacity rejection.

## Test Evidence

The always-on compatibility path is covered by Rust unit and route tests:

```bash
cargo test -q -p ferrum-server
cargo test -q -p ferrum-types requests_tests
```
