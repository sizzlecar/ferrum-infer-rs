# Release-bin Metal smoke: Qwen3-0.6B

- Commit: `e6f24d1`
- GitHub Actions run: `26770107576`
- Binary: `ferrum 0.7.4`
- Artifact: `ferrum-macos-aarch64-dry-run-e6f24d1`
- Model: `Qwen/Qwen3-0.6B`

## Gates

| Gate | Result |
|---|---|
| Paris correctness | pass |
| Multi-turn Paris | pass |
| Three-round multi-turn Paris | pass |

## Performance smoke

| c | prompts | input/output tokens | output throughput |
|---:|---:|---|---:|
| 4 | 16 | 64 / 32 | 56.8 tok/s |

Raw artifacts in this directory include `health.json`, `paris.json`, `multiturn.json`, `multiturn_3round.json`, `bench.json`, `bench.log`, and server stdout/stderr.
