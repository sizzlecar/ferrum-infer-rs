# G2 Agent Serving Gate

Status: PASS

Model: `qwen3:0.6b`

Validated:
- `cargo test --workspace --all-targets`
- `cargo test -p ferrum-server structured_output`
- real-model strict schema smoke 20/20 via `server_structured_output`
- real-model required tool-call smoke 10/10 via `server_agent_tools`
- streaming tests require exactly one `[DONE]` and no invalid pre-validation content
