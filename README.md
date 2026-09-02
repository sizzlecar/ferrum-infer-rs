<p align="center">
  <a href="https://ferrum.pandaailabs.com/">
    <img src="assets/brand/ferrum-lockup.svg" alt="Ferrum — Local LLM Runtime" width="520">
  </a>
</p>

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
ferrum --version
ferrum --help
ferrum doctor
```

Use the commands for your platform. `doctor` shows the model-source mapping
without downloading weights or starting the inference engine.

### macOS Apple Silicon

The first run downloads about **2.55 GiB**. Download time depends on your route
to Hugging Face; wait for the progress output before treating the process as
hung.

```bash
ferrum doctor qwen3.5:4b-q4_k_m
ferrum run qwen3.5:4b-q4_k_m --disable-thinking
```

### Linux NVIDIA CUDA

The first run downloads about **8.7 GiB** of repository weights.

```bash
ferrum doctor qwen3.5:4b
ferrum run qwen3.5:4b --disable-thinking
```

Ferrum does not silently select a model. `run` requires MODEL, and `serve`
requires either `--model` or an intentional `default_model` in `ferrum.toml`.

Serve the same model through an OpenAI-compatible API:

```bash
# macOS Metal
ferrum serve --model qwen3.5:4b-q4_k_m --served-model-name ferrum --disable-thinking --port 8000

# Linux CUDA
ferrum serve --model qwen3.5:4b --served-model-name ferrum --disable-thinking --port 8000

curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"ferrum","messages":[{"role":"user","content":"Reply with a short hello from Ferrum."}],"max_tokens":32}'
```

A working request returns HTTP 200 with a non-empty assistant response.

The Quick Start uses `--disable-thinking` so the first response is short and
direct. Omit the flag to preserve the model template's default reasoning
behavior; an HTTP request can override the server default with
`chat_template_kwargs.enable_thinking`.

`ferrum doctor <MODEL>` resolves an alias and prints the next `run` and `serve`
commands without downloading the model or starting an inference engine.

## Features

- `ferrum run` and `ferrum serve` in one Rust binary.
- OpenAI-compatible Chat Completions and stateless Responses APIs, streaming,
  tools, and structured output.
- Apple Silicon Metal and NVIDIA CUDA from the same runtime.
- Continuous batching, paged KV cache, prefix cache, and typed admission control.
- GGUF on Metal and GPTQ/safetensors on CUDA.
- v0.8 covers language-model inference only. Release scope: Qwen3.5 4B,
  Qwen3.5 35B-A3B, Qwen3 30B-A3B, and Llama 3.1 8B dense.

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
curl --fail --location --remote-name https://github.com/sizzlecar/ferrum-infer-rs/releases/download/v0.8.4/ferrum-linux-x86_64-cuda-sm89.tar.gz
curl --fail --location --remote-name https://github.com/sizzlecar/ferrum-infer-rs/releases/download/v0.8.4/ferrum-linux-x86_64-cuda-sm89.tar.gz.sha256
sha256sum --check ferrum-linux-x86_64-cuda-sm89.tar.gz.sha256
tar -xzf ferrum-linux-x86_64-cuda-sm89.tar.gz
LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-} ./ferrum --version

# macOS Apple Silicon Metal
curl --fail --location --remote-name https://github.com/sizzlecar/ferrum-infer-rs/releases/download/v0.8.4/ferrum-macos-aarch64.tar.gz
curl --fail --location --remote-name https://github.com/sizzlecar/ferrum-infer-rs/releases/download/v0.8.4/ferrum-macos-aarch64.tar.gz.sha256
shasum -a 256 --check ferrum-macos-aarch64.tar.gz.sha256
tar -xzf ferrum-macos-aarch64.tar.gz
./ferrum --version
```

Install the Metal build from crates.io:

```bash
# macOS Apple Silicon Metal
cargo install ferrum-cli --version 0.8.4 --locked --features metal
```

The official prebuilt CUDA asset targets `sm89`. CUDA installation requires a
compatible NVIDIA driver, CUDA runtime, and NCCL runtime on the target host.
CUDA source builds also require Ferrum's matching native-operator set, so use
the prebuilt CUDA tarball or Homebrew formula for the supported install path.

## Architecture

- Contracts: `ferrum-types`, `ferrum-interfaces`
- Execution: `ferrum-engine`, `ferrum-scheduler`, `ferrum-kv`, `ferrum-sampler`
- Models and compute: `ferrum-models`, `ferrum-kernels`, `ferrum-native-ops`, `ferrum-quantization`
- Product surface: `ferrum-cli`, `ferrum-server`, `ferrum-tokenizer`
- Validation: `ferrum-bench-core`, `ferrum-testkit`

## License

MIT
