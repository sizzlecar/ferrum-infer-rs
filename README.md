# Ferrum Infer (Rust)

Ferrum Infer is a Rust workspace for local LLM inference with an Ollama-like CLI and an
OpenAI-compatible HTTP server entrypoint.

Current phase: MVP is runnable, but still evolving quickly.

## Project Status

- MVP available: local chat, embedding generation, model pull/list, HTTP serving.
- Not production-ready yet: APIs and internals may change.
- Focus is correctness, cross-platform build stability, and iterative performance work.

## What Works Today

- CLI commands:
  - `ferrum run` interactive chat
  - `ferrum embed` BERT embeddings
  - `ferrum pull` download from Hugging Face
  - `ferrum list` inspect local model cache (ready/incomplete)
  - `ferrum serve` start HTTP server
  - `ferrum stop` stop server by PID file
- Inference stack:
  - Candle-based runtime path
  - Model loading from Hugging Face cache/snapshots
  - Streaming text generation in CLI
- Server endpoints:
  - `POST /v1/chat/completions`
  - `GET /v1/models`
  - `GET /health`

## Known Limits

- Still pre-production: limited hardening/observability.
- Feature/performance differs by model architecture and device backend.
- CUDA builds require CUDA toolchain on host.
- Some models may appear as `incomplete` in cache if snapshot is partial.

## Quick Start

Prerequisites:

- Rust stable toolchain
- Network access to Hugging Face for model pull
- Optional private model token: `HF_TOKEN`

Build CLI:

```bash
cargo build --release -p ferrum-cli --bin ferrum
```

Show help:

```bash
./target/release/ferrum --help
```

Download a model:

```bash
./target/release/ferrum pull Qwen/Qwen2.5-0.5B-Instruct
```

List local models:

```bash
./target/release/ferrum list
```

Run interactive chat:

```bash
./target/release/ferrum run Qwen/Qwen2.5-0.5B-Instruct
```

Start server:

```bash
./target/release/ferrum serve --model Qwen/Qwen2.5-0.5B-Instruct --host 127.0.0.1 --port 8000
```

Stop server:

```bash
./target/release/ferrum stop
```

## Workspace Layout

Core contracts:

- `crates/ferrum-types`
- `crates/ferrum-interfaces`

Implementations:

- `crates/ferrum-runtime`
- `crates/ferrum-scheduler`
- `crates/ferrum-tokenizer`
- `crates/ferrum-sampler`
- `crates/ferrum-kv`
- `crates/ferrum-models`
- `crates/ferrum-engine`
- `crates/ferrum-server`
- `crates/ferrum-cli`

## Build and Validation

Use the same checks as CI:

```bash
cargo fmt --all -- --check
cargo check --workspace --all-targets
cargo clippy --workspace --all-targets -- -A warnings
cargo build -p ferrum-cli --bin ferrum
cargo test -p ferrum-cli --test cli_e2e
```

Useful local full pass:

```bash
cargo test --workspace
```

## Device Notes

- CPU path works cross-platform.
- Metal path is for Apple targets (`feature = "metal"`).
- CUDA path is optional and environment-dependent.

## Cache and Environment

- Default config file: `ferrum.toml`
- Hugging Face cache defaults to `~/.cache/huggingface` (override via `HF_HOME`)
- Private model token:
  - `HF_TOKEN`
  - `HUGGING_FACE_HUB_TOKEN`

## Roadmap (Near Term)

1. Improve model download resilience (`resume/retry/cleanup` for partial snapshots).
2. Expand CI gates and regression tests for cross-platform device paths.
3. Improve runtime performance and backend parity.
4. Continue OpenAI API compatibility and operational hardening.

## License

MIT. See `LICENSE`.
