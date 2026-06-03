# vLLM compatibility matrix

This matrix describes Ferrum's tested vLLM-facing migration surface. It is not a claim of full vLLM parity.

| Area | Ferrum status | Evidence / gate |
|---|---|---|
| `ferrum serve <MODEL>` as `vllm serve <MODEL>` replacement | Supported for cached supported models and local paths | G1 migration smoke |
| `--host`, `--port` | Supported | CLI help and server smoke |
| `--gpu-memory-utilization` | Supported on CUDA auto-size path | G0/G1 effective config artifacts |
| `--max-model-len` | Supported | Maps to `FERRUM_MAX_MODEL_LEN` |
| `--max-num-seqs` | Supported | Maps to `FERRUM_PAGED_MAX_SEQS` |
| `--max-num-batched-tokens` | Supported | Maps to `FERRUM_MAX_BATCHED_TOKENS` |
| `--enable-prefix-caching` | Supported as runtime flag | Full cache productization is G3 |
| `--no-enable-prefix-caching` | Supported as runtime flag | Effective config reflects `FERRUM_PREFIX_CACHE=0` |
| `--quantization gptq_marlin` | Auto-detected where supported | Qwen3-30B-A3B GPTQ/Marlin CUDA gates |
| `/v1/models` | Supported | G1 migration smoke |
| `/v1/chat/completions` non-streaming | Supported | G1 migration smoke |
| `/v1/chat/completions` streaming SSE | Supported | G1 migration smoke, one `[DONE]` required |
| `stream_options.include_usage` | Supported | G1/OpenAI compatibility smoke |
| Python OpenAI SDK | Supported when SDK is installed | G1 optional Python smoke |
| `async-openai` Rust SDK | Supported | Existing and G1 smoke tests |
| vLLM CLI flag superset | Not supported | Unsupported flags are intentionally rejected |
| Ray / distributed serving | Out of scope | Not implemented |
| vLLM internal APIs | Out of scope | Not implemented |
