# Agent Serving with Ferrum

Ferrum exposes an OpenAI-compatible serving path for agent workloads that need
structured JSON, tool calls, and streaming responses that standard OpenAI
clients can parse.

G2 focuses on a practical subset, not a full grammar engine or tool executor.
Tools are caller-owned: Ferrum can ask the model to produce tool-call arguments
and return OpenAI-shaped `tool_calls`, but it never executes the tool.

## Structured JSON Output

Use `response_format={"type":"json_object"}` when you want best-effort JSON
mode:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"qwen3:0.6b",
    "messages":[{"role":"user","content":"Return JSON: {\"answer\":\"ok\"}"}],
    "response_format":{"type":"json_object"},
    "temperature":0,
    "max_tokens":128
  }'
```

Ferrum repairs one outer markdown JSON fence when present. For strict validation,
use `json_schema` with `strict: true`.

## Strict JSON Schema Subset

Ferrum supports a conservative strict schema subset:

- `type: object`
- `properties`
- `required`
- `additionalProperties: false` or omitted
- scalar `string`, `number`, `integer`, and `boolean`
- `enum` of strings or numbers
- arrays with homogeneous `items` drawn from the same scalar/object subset

Unsupported strict constructs fail before generation with HTTP 400 and
`param=response_format.json_schema`:

- `oneOf`, `anyOf`, `allOf`
- `patternProperties`
- recursive schemas
- external `$ref`
- complex string formats
- regex `pattern` when not enforced
- `minItems` / `maxItems` when not enforced

Example strict request:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"qwen3:0.6b",
    "messages":[{"role":"user","content":"Return {\"answer\":\"ok\"}"}],
    "temperature":0,
    "max_tokens":128,
    "response_format":{
      "type":"json_schema",
      "json_schema":{
        "name":"Answer",
        "strict":true,
        "schema":{
          "type":"object",
          "properties":{"answer":{"type":"string"}},
          "required":["answer"]
        }
      }
    }
  }'
```

For streaming strict schema requests, Ferrum buffers generated content until it
can validate the final JSON. It does not send invalid partial content first. If
validation fails, the stream emits an OpenAI-shaped SSE error followed by exactly
one `data: [DONE]`.

## Tool Calling

Ferrum supports OpenAI function tools:

- `tool_choice="none"`: generated tool-call JSON remains ordinary assistant content.
- `tool_choice="auto"`: Ferrum may return ordinary assistant content or parsed `tool_calls` when model output matches a declared function tool.
- Named `tool_choice`: Ferrum validates the function name exists and constrains parsing to that tool.
- `tool_choice="required"`: Ferrum requires at least one function tool, steers generation toward the first declared tool argument schema, and returns at least one OpenAI-shaped `tool_call` when generation succeeds.

Ferrum does not execute tools. The caller reads `tool_calls`, executes tools in
application code, and sends tool results back in the next `messages` request.

Example required tool-call request:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"qwen3:0.6b",
    "messages":[{"role":"user","content":"Use calc. Return arguments for 123+456."}],
    "temperature":0,
    "max_tokens":128,
    "tools":[{
      "type":"function",
      "function":{
        "name":"calc",
        "description":"Evaluate an arithmetic expression.",
        "parameters":{
          "type":"object",
          "properties":{"expression":{"type":"string","enum":["123+456"]}},
          "required":["expression"]
        }
      }
    }],
    "tool_choice":"required"
  }'
```

If `tool_choice="required"` cannot be satisfied after generation, non-streaming
requests return an OpenAI-style error with `param=tool_choice`. Streaming
requests emit an OpenAI-style SSE error and exactly one `[DONE]` without first
streaming invalid assistant content.

## Validation Expectations

G2 release validation covers:

- `cargo test -p ferrum-server structured_output`
- `cargo test --release -p ferrum-cli --test server_structured_output -- --ignored --test-threads=1`
- `cargo test --release -p ferrum-cli --test server_agent_tools -- --ignored --test-threads=1`
- real-model strict schema smoke, 20/20 at `temperature=0.0`
- real-model required tool-call smoke, 10/10 at `temperature=0.0`
- streaming tests with exactly one `[DONE]`
