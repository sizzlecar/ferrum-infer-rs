# vLLM to Ferrum migration guide

Ferrum can serve common single-node OpenAI-compatible LLM workloads with a Rust-native binary and no Python runtime in the serving path.

This guide is for vLLM users who want to try Ferrum without changing their OpenAI client code.

## Quick mapping

| vLLM concept / flag | Ferrum equivalent | Notes |
|---|---|---|
| `vllm serve <MODEL>` | `ferrum serve --model <MODEL>` or `ferrum serve <MODEL>` | Ferrum accepts HF ids, cached aliases, local safetensors dirs, and supported GGUF paths. |
| `--host`, `--port` | `--host`, `--port` | Same meaning. |
| `--gpu-memory-utilization` | `--gpu-memory-utilization` | Auto-sizes KV pool on CUDA when model weights are local. |
| `--max-model-len` | `--max-model-len` | Maps to `FERRUM_MAX_MODEL_LEN`. |
| `--max-num-seqs` | `--max-num-seqs` | Maps to `FERRUM_PAGED_MAX_SEQS`. |
| `--max-num-batched-tokens` | `--max-num-batched-tokens` | Maps to `FERRUM_MAX_BATCHED_TOKENS`. |
| `--enable-prefix-caching` | `--enable-prefix-caching` / `--enable-prefix-cache` | Requests product prefix-cache behavior and observability. Unsafe engine-level KV prefix reuse is forced off and recorded in effective config until correctness gates prove it safe. |
| `--no-enable-prefix-caching` | `--no-enable-prefix-caching` / `--disable-prefix-cache` | Disables product prefix cache and records `FERRUM_PREFIX_CACHE=0`. |
| `--quantization gptq_marlin` | auto-detected | Ferrum auto-detects supported GPTQ / Marlin paths from model metadata. |
| OpenAI `/v1/chat/completions` | supported | Non-streaming and streaming. |
| OpenAI streaming SSE | supported | Emits OpenAI-shaped `data:` chunks and one `[DONE]`. |
| `/v1/models` | supported | Lists the loaded model id. |

## Serve examples

Ferrum equivalent of a common vLLM command:

```bash
vllm serve Qwen/Qwen3-0.6B \
  --host 127.0.0.1 \
  --port 8000 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 2048 \
  --max-num-seqs 8 \
  --max-num-batched-tokens 2048
```

```bash
ferrum serve Qwen/Qwen3-0.6B \
  --host 127.0.0.1 \
  --port 8000 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 2048 \
  --max-num-seqs 8 \
  --max-num-batched-tokens 2048
```

For a cached alias:

```bash
ferrum pull qwen3:0.6b
ferrum serve qwen3:0.6b --host 127.0.0.1 --port 8000
```

## OpenAI client compatibility

Use the same OpenAI base URL shape as vLLM:

```bash
export OPENAI_BASE_URL=http://127.0.0.1:8000/v1
export OPENAI_API_KEY=dummy-key-not-checked
```

Ferrum ignores API keys for local serving unless an external proxy enforces authentication.

Ferrum accepts vLLM-compatible `top_k`, `min_p`, and `repetition_penalty`
fields on chat-completion requests. `top_k=-1` or `0` and `min_p=0` disable the
corresponding filter; invalid ranges return an OpenAI-shaped 4xx response.

## Structured-output behavior in v0.8.0

Ferrum v0.8.0 changes `response_format` from prompt steering and response repair
to tokenizer-aware constrained decoding:

- `json_object` returns exactly one valid JSON object. Markdown fences,
  explanatory prefixes, suffixes, and non-object JSON roots are errors.
- strict `json_schema` uses the same constrained decoder and validates the exact
  final JSON value. Unsupported grammars fail before request admission; they do
  not silently fall back to ordinary sampling.
- streaming hard-structured responses are buffered until validation succeeds.
  A generation failure produces an OpenAI-shaped error event and one `[DONE]`
  without first exposing invalid partial JSON.
- reasoning-capable templates may generate a reasoning block first. The grammar
  starts at the template's structured-content boundary, and reasoning is not
  included in the validated JSON value.
- non-strict `json_schema` remains best-effort for compatibility.

Clients migrating from Ferrum 0.7.x must not depend on outer markdown-fence
removal or brace extraction. They should handle an explicit request or generation
error when a schema cannot be compiled or a valid value cannot be completed
within the request limits.

## Benchmark comparison

Use Ferrum's canonical HTTP benchmark client for both Ferrum and vLLM endpoints:

```bash
ferrum bench-serve \
  --base-url http://127.0.0.1:8000 \
  --model Qwen/Qwen3-0.6B \
  --tokenizer ~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/<sha> \
  --dataset random \
  --random-input-len 256 \
  --random-output-len 128 \
  --num-prompts 64 \
  --warmup-requests 8 \
  --concurrency 8 \
  --n-repeats 3 \
  --fail-on-error
```

Run the same command against a vLLM server by changing only `--base-url`.

## Unsupported or intentionally different behavior

Ferrum does not aim for full vLLM CLI or engine parity.

- No Ray or multi-node compatibility in this migration pack.
- No vLLM internal engine API compatibility.
- No claim that Ferrum's scheduler behavior is identical to vLLM.
- Unknown vLLM flags are not accepted as no-op flags.
- Prefix caching is exposed as a runtime flag here; G3 adds productized cache metrics and session-cache gates.
