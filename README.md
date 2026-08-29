# ferrum-infer-rs

[![Crates.io](https://img.shields.io/crates/v/ferrum-cli.svg)](https://crates.io/crates/ferrum-cli)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/sizzlecar/ferrum-infer-rs/blob/main/LICENSE)

> Rust-native LLM inference for OpenAI-compatible local and private serving.

**One binary. No Python runtime. Apple Silicon Metal and NVIDIA CUDA acceleration.**

[中文说明](README_zh.md)

## Quick Start

Install Ferrum:

```bash
brew tap sizzlecar/ferrum

# macOS Apple Silicon
brew install ferrum

# Linux x86_64, NVIDIA CUDA sm89
brew install ferrum-cuda
```

Inspect the installed binary before downloading weights:

```bash
ferrum doctor
```

Run a model directly:

```bash
# macOS Metal (GGUF)
ferrum run qwen3.5:4b-q4_k_m

# Linux CUDA (safetensors)
ferrum run qwen3.5:4b
```

Ferrum does not silently select a model. `run` requires MODEL, and `serve`
requires either `--model` or an intentional `default_model` in `ferrum.toml`.

Serve the same model through an OpenAI-compatible API:

```bash
# macOS Metal
ferrum serve --model qwen3.5:4b-q4_k_m --served-model-name ferrum --port 8000

# Linux CUDA
ferrum serve --model qwen3.5:4b --served-model-name ferrum --port 8000

curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"ferrum","messages":[{"role":"user","content":"Hello"}]}'
```

For a short direct answer from a model whose template enables reasoning by
default, add `--disable-thinking` to `ferrum run` or `ferrum serve`. Omitting
the flag preserves the model template's default; an HTTP request can override
the server default with `chat_template_kwargs.enable_thinking`.

`ferrum doctor <MODEL>` resolves an alias and prints the next `run` and `serve`
commands without downloading the model or starting an inference engine.

### Qwen3.8 27B block-FP8 on CUDA

Current source builds support the pinned official
[`Qwen/Qwen3.8-27B-FP8@017b9c7`](https://huggingface.co/Qwen/Qwen3.8-27B-FP8/tree/017b9c7af6b5689d5dd426a76e0bc077eb5ca20a)
checkpoint as a local Hugging Face snapshot. Build Ferrum with
`cuda,vllm-moe-marlin,vllm-paged-attn-v2`, then use the same snapshot for both
product entrypoints:

```bash
MODEL=/path/to/Qwen3.8-27B-FP8

ferrum run "$MODEL" \
  --backend cuda --gpu-devices 0 --disable-thinking \
  --max-model-len 512 --max-num-seqs 1 \
  --max-num-batched-tokens 1024 --gpu-memory-utilization 0.90

ferrum serve --model "$MODEL" \
  --served-model-name qwen38-27b-fp8 \
  --backend cuda --gpu-devices 0 --disable-thinking \
  --max-model-len 512 --max-num-seqs 2 \
  --max-num-batched-tokens 1024 --gpu-memory-utilization 0.90
```

These bounded settings were validated on a single 48 GB-class CUDA GPU. The
adoption gate does not cover lower-memory devices.

## Features

- `ferrum run` and `ferrum serve` in one Rust binary.
- OpenAI-compatible Chat Completions and stateless Responses APIs, streaming,
  tools, and structured output.
- Apple Silicon Metal and NVIDIA CUDA from the same runtime.
- Continuous batching, paged KV cache, prefix cache, and typed admission control.
- GGUF on Metal and GPTQ/safetensors on CUDA.
- v0.8 covers language-model inference only. Release scope: Qwen3.5 4B, Qwen3.5 35B-A3B,
  Qwen3 30B-A3B, and Llama 3.1 8B dense. [Support matrix](docs/release/runtime-vnext/0.8.0/SUPPORT_MATRIX.md).

## Performance Snapshot

Latest R2 development `ferrum serve` checkpoint. The first three rows use
64-token input / 128-token output on Metal and 256 / 128 on CUDA. Values are
mean tok/s with the 95% confidence-interval half-width across three repeats.

| Model | M1 Max 32 GB Metal | RTX 4090 CUDA |
|---|---:|---:|
| Qwen3.5 4B | c=16 · 61.9 ± 0.1 | c=32 · 241.3 ± 0.6 |
| Qwen3.5 35B-A3B | c=4 · 26.1 ± 0.2 | c=16 · 174.1 ± 1.0 |
| Qwen3 30B-A3B | c=16 · 39.6 ± 1.2 | c=32 · 214.9 ± 2.7 |
| Qwen3.8 27B AWQ INT4 |  | c=4 · 78.19 ± 0.04 · c=16 · 115.12 ± 1.18 · c=32 · 115.18 ± 0.97 |

`c` is active server concurrency. The first three rows completed 100 requests ×
3 repeats with zero errors. [Measurement details](docs/release/runtime-vnext/0.8.0/PERFORMANCE_REPORT.md).

The Qwen3.8 27B AWQ INT4 row uses a different checkpoint and quantization path
from the official block-FP8 model described above.

## OpenAI-Compatible API

Ferrum supports:

- chat completions and streaming usage
- stateless Responses text, streaming, usage, and function tools
- function tools with `auto`, `none`, `required`, or a named function
- `json_object` and strict `json_schema` structured output
- multi-turn sessions, prefix cache, and session cache
- typed concurrency, memory, and scheduler controls

See [OpenAI API compatibility](docs/openai-api-compatibility.md) for the exact
request contract and [cache product controls](docs/cache-product.md) for prefix
and session caching.

## Installation

Homebrew:

```bash
brew tap sizzlecar/ferrum

# macOS Apple Silicon Metal
brew install ferrum

# Linux x86_64 CUDA sm89
brew install ferrum-cuda
```

Prebuilt release tarballs:

```bash
# Linux x86_64 CUDA sm89
curl -L https://github.com/sizzlecar/ferrum-infer-rs/releases/latest/download/ferrum-linux-x86_64-cuda-sm89.tar.gz | tar xz
LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-} ./ferrum --version

# macOS Apple Silicon Metal
curl -L https://github.com/sizzlecar/ferrum-infer-rs/releases/latest/download/ferrum-macos-aarch64.tar.gz | tar xz
./ferrum --version
```

Install from crates.io:

```bash
# macOS Apple Silicon Metal
cargo install ferrum-cli --version 0.8.3 --locked --features metal

# NVIDIA CUDA
cargo install ferrum-cli --version 0.8.3 --locked \
  --features cuda,vllm-moe-marlin,vllm-paged-attn-v2
```

The official prebuilt CUDA asset targets `sm89`. CUDA installation requires a
compatible NVIDIA driver, CUDA runtime, and NCCL runtime on the target host.

## Architecture

- Contracts: `ferrum-types`, `ferrum-interfaces`
- Execution: `ferrum-engine`, `ferrum-scheduler`, `ferrum-kv`, `ferrum-sampler`
- Models and compute: `ferrum-models`, `ferrum-kernels`, `ferrum-native-ops`, `ferrum-quantization`
- Product surface: `ferrum-cli`, `ferrum-server`, `ferrum-tokenizer`
- Validation: `ferrum-bench-core`, `ferrum-testkit`

## License

MIT
