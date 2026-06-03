# G1 vLLM Migration Compatibility Gate

Status: PASS

Model: `qwen3:0.6b`

Validated:
- vLLM-compatible `ferrum serve` flags are visible in help output.
- CLI flags are reflected in effective runtime config with `cli` source.
- `/v1/models`, non-stream chat, stream chat, and usage stream smoke passed.
- Streaming emitted exactly one `[DONE]`; usage stream emitted exactly one usage chunk before `[DONE]`.
- Server log was scanned for forbidden release-blocker patterns.
