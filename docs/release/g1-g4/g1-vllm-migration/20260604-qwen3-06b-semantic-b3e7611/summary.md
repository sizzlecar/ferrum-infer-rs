# G1 vLLM Migration Compatibility Gate

Status: PASS

Model: `Qwen/Qwen3-0.6B`

Validated:
- vLLM-compatible `ferrum serve` flags are visible in help output.
- CLI flags are reflected in effective runtime config with `cli` source.
- `--max-model-len` returned OpenAI-shaped HTTP 400 when prompt + max_tokens exceeded the configured context limit.
- `--max-num-seqs` was tested with concurrent product requests and `/health` active/queued observations.
- `--max-num-batched-tokens` was tested at scheduler batch-plan level with prompt-token admission evidence.
- Prefix-cache vLLM/product enable aliases both produced runtime cache hits and saved prefill tokens.
- Prefix-cache vLLM/product disable aliases both kept runtime cache hits at zero.
- `/v1/models`, non-stream chat, stream chat, and usage stream smoke passed.
- Streaming emitted exactly one `[DONE]`; usage stream emitted exactly one usage chunk before `[DONE]`.
- Server log was scanned for forbidden release-blocker patterns.
