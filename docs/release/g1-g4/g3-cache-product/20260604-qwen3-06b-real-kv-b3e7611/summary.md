# G3 Cache Product Gate

Status: PASS

Model: `Qwen/Qwen3-0.6B`

Validated:
- `cargo test --workspace --all-targets`
- `cargo test -p ferrum-server cache_metrics_contract`
- real-model prefix cache product correctness
- real-model session cache correctness
- shared-prefix `bench-serve --fail-on-error --require-ci` disabled/enabled runs, unless explicitly skipped
- `/health.cache.prefix_cache.position == real-kv-reuse`
- enabled metrics include prefix hits and saved prefill tokens
- cache-enabled performance meets the configured improvement threshold
