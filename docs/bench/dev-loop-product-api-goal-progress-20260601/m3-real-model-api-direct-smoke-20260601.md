# Real-Model Direct OpenAI API Smoke - 2026-06-01

Purpose: provide release evidence for OpenAI-compatible API behavior against a
real model without using the ignored `cargo test` path that blocked in a debug
CUDA build script.

Remote artifact:

- `/workspace/m3-real-model-api-direct-smoke-20260601`

Model:

- `Qwen/Qwen3-0.6B`
- snapshot:
  `/root/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca`

Server:

```bash
FERRUM_MODEL_PATH=/root/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca \
  target/release/ferrum serve \
  /root/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca \
  --port 18580
```

Checks:

| Check | Result | Evidence |
|---|---|---|
| `/health` | pass | `health.json` |
| non-streaming chat schema/status | pass | `chat_basic.json` |
| real-model Paris answer | pass | content `Paris` |
| usage accounting fields | pass | `prompt_tokens=24`, `completion_tokens=2`, `total_tokens=26` |
| streaming chat `[DONE]` | pass | `chat_streaming.sse` |
| streaming `stream_options.include_usage` | pass | final usage chunk present |
| `response_format=json_object` | pass | content contains `{"city": "Paris"}` |
| three-turn recall | pass | content contains `basalt` and `Paris` |

Summary:

```json
{
  "all_passed": true
}
```

Important product note:

- `ferrum serve qwen3:0.6b` resolved the model snapshot but failed tokenizer
  creation unless `FERRUM_MODEL_PATH` was supplied or the snapshot path was
  passed directly. This direct smoke therefore proves the real-model OpenAI API
  path, not the ergonomic alias serve path.
- The root cause for `ferrum pull qwen3:0.6b` failing was a missing Qwen3 alias
  in `crates/ferrum-cli/src/commands/pull.rs`; this checkpoint adds the
  missing `qwen3:0.6b`, `qwen3:1.7b`, and `qwen3:4b` aliases.
