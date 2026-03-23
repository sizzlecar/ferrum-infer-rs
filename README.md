# Ferrum Infer

A Rust-native LLM inference engine. Load models from Hugging Face, chat locally or serve via OpenAI-compatible API. Single binary, no Python, no runtime dependencies.

[中文说明](README_zh.md)

## Quick Start

Prerequisites: Rust stable toolchain.

For gated models (e.g. Llama 3.2), set your Hugging Face token first:
```bash
export HF_TOKEN=hf_your_token_here
```

```bash
# Build
cargo build --release -p ferrum-cli --bin ferrum

# Download a model
./target/release/ferrum pull qwen3:0.6b

# Chat
./target/release/ferrum run qwen3:0.6b

# Or start an API server
./target/release/ferrum serve --model qwen3:0.6b --port 8000
```

## Supported Models

| Alias | Model | Architecture |
|-------|-------|-------------|
| `qwen3:0.6b` / `1.7b` / `4b` | Qwen3 | Qwen3 |
| `qwen2.5:0.5b` / `1.5b` / `3b` / `7b` | Qwen2.5-Instruct | Qwen2 |
| `llama3.2:1b` / `3b` | Llama-3.2-Instruct | LLaMA |
| `tinyllama` | TinyLlama-1.1B-Chat | LLaMA |

Any Hugging Face model ID with a supported architecture also works directly:
```bash
./target/release/ferrum run Qwen/Qwen3-0.6B
```

## Commands

| Command | Description |
|---------|-------------|
| `ferrum run <model>` | Interactive chat |
| `ferrum serve --model <model>` | OpenAI-compatible HTTP server |
| `ferrum stop` | Stop running server |
| `ferrum pull <model>` | Download model from Hugging Face |
| `ferrum list` | Show cached models |
| `ferrum embed <model>` | Generate BERT embeddings |

## API Endpoints

```bash
# Chat completions (OpenAI-compatible)
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3:0.6b","messages":[{"role":"user","content":"Hello"}]}'

# List models
curl http://localhost:8000/v1/models

# Health check
curl http://localhost:8000/health
```

## Current Status

**v0.2.0 — Functional MVP, pre-production.**

What works:
- CLI chat and HTTP serving with streaming
- Qwen3, Qwen2/2.5, LLaMA 3.x, TinyLlama architectures
- Metal GPU acceleration (macOS), CPU cross-platform
- Top-k/top-p/temperature/repetition-penalty sampling
- Hugging Face model download and cache management

What's in progress:
- Backend abstraction layer (KernelOps) for pluggable Metal/CUDA/CPU kernels
- PagedAttention integration for production-grade KV cache management
- Continuous batching for concurrent request serving

## Roadmap

1. **Kernel backend abstraction** — unify Metal/CUDA/CPU behind a single trait interface
2. **CUDA kernel FFI** — bind FlashAttention/FlashInfer for NVIDIA GPUs
3. **Production batching** — iteration-level continuous batching with preemption
4. **Quantization** — GPTQ/AWQ/GGUF support for larger models on consumer hardware

See [docs/ROADMAP.md](docs/ROADMAP.md) for full details.

## Build Options

```bash
# CPU only (default)
cargo build --release -p ferrum-cli

# With Metal acceleration (macOS)
cargo build --release -p ferrum-cli --features metal
```

Prerequisites: Rust stable toolchain.

## Project Structure

```
crates/
├── ferrum-types        # Shared type definitions
├── ferrum-interfaces   # Core trait contracts (ComputeBackend, KernelOps, ModelExecutor)
├── ferrum-runtime      # Backend implementations (Candle, CPU)
├── ferrum-engine       # Metal kernels, model orchestration
├── ferrum-models       # Model architectures (LLaMA, Qwen2, Qwen3, BERT)
├── ferrum-tokenizer    # Tokenization
├── ferrum-sampler      # Sampling strategies
├── ferrum-scheduler    # Request scheduling
├── ferrum-kv           # KV cache management
├── ferrum-server       # HTTP API server
├── ferrum-cli          # CLI binary
└── ferrum-testkit      # Testing utilities
```

## License

MIT
