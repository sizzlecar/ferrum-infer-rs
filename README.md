# Ferrum Infer

[![Crates.io](https://img.shields.io/crates/v/ferrum-cli.svg)](https://crates.io/crates/ferrum-cli)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A Rust-native LLM inference engine. Load models from Hugging Face, chat locally or serve via OpenAI-compatible API. Single binary, no Python, no runtime dependencies.

[中文说明](README_zh.md)

## Install

```bash
# From crates.io
cargo install ferrum-cli

# Or build from source
cargo build --release -p ferrum-cli --bin ferrum
```

## Quick Start

For gated models (e.g. Llama 3.2), set your Hugging Face token first:
```bash
export HF_TOKEN=hf_your_token_here
```

```bash
# Download a model
ferrum pull qwen3:0.6b

# Chat
ferrum run qwen3:0.6b

# Or start an API server
ferrum serve --model qwen3:0.6b --port 8000
```

## Supported Models

| Alias | Model | Architecture | CUDA Runner |
|-------|-------|-------------|-------------|
| `qwen3:0.6b` / `1.7b` / `4b` | Qwen3 | Qwen3 | Yes |
| `qwen2.5:0.5b` / `1.5b` / `3b` / `7b` | Qwen2.5-Instruct | Qwen2 | — |
| `llama3.2:1b` / `3b` | Llama-3.2-Instruct | LLaMA | Yes |
| `tinyllama` | TinyLlama-1.1B-Chat | LLaMA | Yes |

GPTQ INT4 quantized models are auto-detected and use the Marlin fused kernel:
```bash
./target/release/ferrum run JunHowie/Qwen3-4B-GPTQ-Int4
```

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
| `ferrum bench <model>` | Performance benchmark |
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

## Performance

Benchmarked on **RTX PRO 6000 (Blackwell)**:

### Qwen3-4B

| Mode | FP16 | INT4 (GPTQ + Marlin) |
|------|------|----------------------|
| Single request decode | 88.1 tok/s | **130.4 tok/s (+48%)** |
| 4 concurrent (batch) | 109.4 tok/s | **124.2 tok/s** |
| VRAM | ~8 GB | **~2.5 GB (-69%)** |

### TinyLlama-1.1B (Llama architecture)

| Mode | Candle | CUDA Runner |
|------|--------|-------------|
| Decode | 126 tok/s | **256.5 tok/s (+103%)** |

### Tensor Parallelism (multi-GPU)

| Config | Qwen3-4B FP16 |
|--------|---------------|
| 1× GPU | 82.3 tok/s (TPOT 12.1ms) |
| 2× GPU TP | 26.1 tok/s (TPOT 38.4ms) |

> TP decode uses persistent per-rank threads with NCCL all-reduce. Current bottleneck is PCIe interconnect latency (~0.44ms × 72 NCCL calls/step). TP is most beneficial for models that don't fit on a single GPU, or with NVLink interconnect.

### Key Optimizations

- **Custom CUDA decode runner**: bypasses candle for the decode hot path (Qwen3 + LLaMA)
- **INT4 quantization**: GPTQ models auto-detected, Marlin fused INT4×FP16 kernel
- **Tensor parallelism**: persistent per-rank threads, barrier sync, NCCL all-reduce (Megatron-LM pattern)
- **Batched attention kernel**: single launch for all batch items (SM utilization 17%→67%)
- **Batched RoPE**: per-item positions in single kernel launch
- **Custom CUDA kernels**: fused RmsNorm, SiLU×mul, RoPE, decode attention (all on single stream)
- **Flash Decoding**: split-K for long-context decode (auto at KV > 256)
- **Batch decode**: batched cuBLAS GEMM + batched attention for concurrent requests
- **Paged KV attention**: GPU block pool with block-table indirection
- **Double-buffered residual**: cross-layer norm fusion (-108 kernel launches)

## Current Status

What works:
- CLI chat, HTTP serving with streaming, benchmarking
- Qwen3, Qwen2/2.5, LLaMA 3.x, TinyLlama architectures
- Custom CUDA decode runner for Qwen3 and LLaMA (2x speedup)
- Metal GPU acceleration (macOS), CUDA (NVIDIA), CPU
- INT4 GPTQ quantization with Marlin fused kernel (Blackwell compatible)
- FlashAttention-2 prefill + custom CUDA decode runner
- Paged KV cache with block reclamation
- Continuous batching with batch decode
- Tensor parallelism (multi-GPU NCCL, auto-detects GPU count)
- Top-k/top-p/temperature/repetition-penalty sampling

## Roadmap

- **Speculative decoding** — draft model verification
- **More model architectures** — Mistral, Phi, DeepSeek, etc.
- **Qwen2 CUDA runner** — same pattern as LLaMA

See [docs/ROADMAP.md](docs/ROADMAP.md) for full details.

## Build Options

```bash
# CPU only (default)
cargo install ferrum-cli

# With Metal acceleration (macOS)
cargo install ferrum-cli --features metal

# With CUDA acceleration (NVIDIA, requires CUDA toolkit + nvcc)
cargo install ferrum-cli --features cuda
```

Or build from source:
```bash
cargo build --release -p ferrum-cli                    # CPU
cargo build --release -p ferrum-cli --features metal   # Metal (macOS)
cargo build --release -p ferrum-cli --features cuda    # CUDA (NVIDIA)
cargo build --release -p ferrum-cli --features cuda    # Multi-GPU auto-detected when available
```

Prerequisites: Rust stable toolchain.

## Project Structure

```
crates/
├── ferrum-types          # Shared type definitions
├── ferrum-interfaces     # Core trait contracts (ComputeBackend, KernelOps, ModelExecutor)
├── ferrum-runtime        # Backend implementations (Candle, CPU)
├── ferrum-engine         # Metal kernels, model orchestration
├── ferrum-models         # Model architectures (LLaMA, Qwen2, Qwen3, BERT)
├── ferrum-cuda-kernels   # Custom CUDA kernels + decode runner
├── ferrum-tokenizer      # Tokenization
├── ferrum-sampler        # Sampling strategies
├── ferrum-scheduler      # Request scheduling
├── ferrum-kv             # KV cache management
├── ferrum-server         # HTTP API server
├── ferrum-cli            # CLI binary
└── ferrum-testkit        # Testing utilities
```

## License

MIT
