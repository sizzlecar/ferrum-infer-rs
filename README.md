# ferrum-infer-rs

[![Crates.io](https://img.shields.io/crates/v/ferrum-cli.svg)](https://crates.io/crates/ferrum-cli)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/sizzlecar/ferrum-infer-rs/blob/main/LICENSE)

> Rust-native LLM inference for fast, simple, OpenAI-compatible serving.

**One binary. No Python runtime. Hardware-accelerated on Apple Silicon and NVIDIA CUDA.**

Ferrum is a lightweight inference engine for running and serving transformer LLMs with an OpenAI-compatible API.
It is built for developers and teams who want simple deployment, practical serving performance, and a clean Rust-native runtime for local, edge, and production inference.

[中文说明](README_zh.md)

## Quick Start

Install a prebuilt binary:

```bash
brew tap sizzlecar/ferrum
brew install ferrum        # macOS Apple Silicon Metal / Linux x86_64 CPU
brew install ferrum-cuda   # Linux x86_64 CUDA sm89 build
ferrum --version
```

Run a model directly:

```bash
export HF_TOKEN=hf_your_token_here   # only needed for gated models
ferrum run qwen3:4b
```

Serve the same model through an OpenAI-compatible API:

```bash
ferrum serve --model qwen3:4b --port 8000

curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3:4b","messages":[{"role":"user","content":"Hello"}]}'
```

## Why Ferrum?

- **One binary:** ship `ferrum run` and `ferrum serve` without a Python service in the runtime path.
- **OpenAI-compatible API:** reuse existing OpenAI-shaped clients, SDKs, and HTTP tooling.
- **Hardware accelerated:** use Apple Silicon Metal or NVIDIA CUDA from the same project.
- **Rust-native runtime:** fewer moving parts, simpler deployment, and a runtime that is easy to embed or package.
- **Practical serving performance:** continuous batching, paged KV cache, INT4 GPTQ/Marlin paths, CUDA Graphs, and release-tested concurrency gates.

## What Ferrum is good at

Ferrum is built for developers and teams building:

- local AI agents
- private OpenAI-compatible inference services
- Apple Silicon LLM applications
- CUDA-accelerated inference servers
- edge and workstation deployments
- Rust-native AI infrastructure

## Performance Snapshot

Ferrum is designed for practical high-throughput serving on modern accelerators, with raw benchmark logs checked into the repository instead of only summary claims.

Historical CUDA same-pod throughput on RTX 4090 with
`Qwen3-30B-A3B-GPTQ-Int4` for the opt-in FA2 direct-FFI path:

| Concurrency | Ferrum tok/s | vLLM 0.20.2 tok/s | Ferrum / vLLM |
| ---: | ---: | ---: | ---: |
| 1 | `160.4 +/- 0.2` | `183.9 +/- 0.2` | `0.872x` |
| 4 | `446.3 +/- 7.0` | `512.5 +/- 2.8` | `0.871x` |
| 16 | `1185.1 +/- 12.3` | `1331.9 +/- 5.7` | `0.890x` |
| 32 | `1641.9 +/- 4.8` | `1972.9 +/- 18.6` | `0.832x` |

Full CUDA methodology and raw artifacts for that historical comparison are in
[`docs/bench/cuda-rtx4090-2026-05-30-m3-80pct-confirmed/`](docs/bench/cuda-rtx4090-2026-05-30-m3-80pct-confirmed/).
Do not treat this table as a current source-linked/default release-gate claim;
release candidates must use the current G0/G1-G4 CUDA artifacts for their exact
binary, git SHA, runtime config, and same-hardware results.

Release and Metal gates:

| Target | Model / workload | Result | Evidence |
| --- | --- | --- | --- |
| CUDA release binary | Qwen3-30B-A3B GPTQ-Int4, c=32 smoke | `16/16` requests, `0` errors; Paris, multi-turn, and three-round chat gates passed | [`CUDA release-binary validation`](docs/bench/dev-loop-product-api-goal-progress-20260601/release-bin-cuda-qwen3-30b-a3b-v0.7.4-final-05254fb-20260602/) |
| Apple Silicon Metal | Llama/Qwen3 8B and Qwen3-30B-A3B, canonical random `16/64` streaming chat workload | Current G0 gate requires default `serve` startup evidence, `run` and `serve` correctness, concurrent marker/square content-quality probes, plus throughput rows with actual tokenizer-counted input lengths. Qwen3-30B-A3B Metal GGUF release evidence must include a multi-sequence MoE content-quality and throughput cell. | `scripts/metal_readme_regression.py` + `scripts/release/validate_metal_readme_regression.py` |
| Apple Silicon Metal historical report | Older near-empty-prompt README rows | Historical evidence only; do not use those numbers as current release-gate claims | [`metal-readme-regression-20260601-release-candidate-rerun3`](docs/bench/dev-loop-product-api-goal-progress-20260601/metal-readme-regression-20260601-release-candidate-rerun3/) |

## API Compatibility

Ferrum exposes OpenAI-shaped chat completions for local and private deployments. The endpoint contract, explicit rejections, tool-field status, usage accounting, and structured-output limits are documented in [`docs/openai-api-compatibility.md`](docs/openai-api-compatibility.md).

### Tool Calling

`/v1/chat/completions` accepts OpenAI-style function tools. Tool execution is
caller-owned: Ferrum renders tool definitions into the model prompt and returns
model-emitted calls as OpenAI-shaped `tool_calls` in non-streaming responses and
streaming deltas.

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3:4b",
    "messages": [{"role": "user", "content": "Call calc for 123+456"}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "calc",
        "description": "Evaluate an arithmetic expression.",
        "parameters": {
          "type": "object",
          "properties": {"expression": {"type": "string"}},
          "required": ["expression"]
        }
      }
    }],
    "tool_choice": "required",
    "max_tokens": 128
  }'
```

Supported tool choices include `auto`, `none`, `required`, and a specific
function selector. Non-function tool types and undeclared forced tools are
rejected with OpenAI-style 400 errors instead of being silently ignored.

### Structured Output

Ferrum supports OpenAI `response_format` for text, best-effort `json_object`,
and strict `json_schema` validation for a conservative schema subset.

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3:4b",
    "messages": [{"role": "user", "content": "Return the sum of 123+456."}],
    "response_format": {
      "type": "json_schema",
      "json_schema": {
        "name": "Answer",
        "strict": true,
        "schema": {
          "type": "object",
          "additionalProperties": false,
          "properties": {"answer": {"type": "integer"}},
          "required": ["answer"]
        }
      }
    },
    "max_tokens": 128
  }'
```

Strict schemas are validated before non-streaming responses return. Strict
schema streaming buffers generated content until validation passes, then emits
valid content and one final `[DONE]`. Unsupported schema constructs fail fast
with `param=response_format.json_schema`.

### Prefix and Session Cache

Prefix cache is an explicit serving option for repeated or shared-prefix
workloads. Session cache is also opt-in and uses caller-provided session ids.

```bash
ferrum serve qwen3:4b \
  --enable-prefix-cache \
  --session-cache memory \
  --session-cache-max-entries 128 \
  --session-cache-max-tokens 4096

curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-Ferrum-Session: agent-session-1" \
  -d '{"model":"qwen3:4b","messages":[{"role":"user","content":"Remember ferrum-blue."}]}'
```

`/health` reports whether prefix cache is active and whether the path is real
KV reuse. `/metrics` exposes prefix hit/miss, saved prefill token, entry, byte,
and session-cache counters. Details are in [`docs/cache-product.md`](docs/cache-product.md).

## Installation

Homebrew:

```bash
brew tap sizzlecar/ferrum
brew install ferrum        # macOS Metal / Linux CPU
brew install ferrum-cuda   # Linux x86_64 CUDA sm89 build
```

Prebuilt release tarballs:

```bash
# Linux x86_64 CPU
curl -L https://github.com/sizzlecar/ferrum-infer-rs/releases/latest/download/ferrum-linux-x86_64.tar.gz | tar xz
./ferrum --help

# Linux x86_64 CUDA, sm89 build
curl -L https://github.com/sizzlecar/ferrum-infer-rs/releases/latest/download/ferrum-linux-x86_64-cuda-sm89.tar.gz | tar xz
LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-} ./ferrum --help

# macOS Apple Silicon Metal
curl -L https://github.com/sizzlecar/ferrum-infer-rs/releases/latest/download/ferrum-macos-aarch64.tar.gz | tar xz
./ferrum --help
```

Linux x86_64 is the CPU build. Linux x86_64 CUDA is built for `sm89` and requires a compatible NVIDIA driver plus CUDA runtime libraries on the target host. macOS aarch64 is the Metal build.

From source:

```bash
cargo install ferrum-cli
cargo build --release -p ferrum-cli --bin ferrum
```

## Benchmarks / Docs

- CUDA vLLM comparison: [`docs/bench/cuda-rtx4090-2026-05-30-m3-80pct-confirmed/`](docs/bench/cuda-rtx4090-2026-05-30-m3-80pct-confirmed/)
- Apple Silicon regression gate: `scripts/metal_readme_regression.py` and `scripts/release/validate_metal_readme_regression.py`
- OpenAI API compatibility: [`docs/openai-api-compatibility.md`](docs/openai-api-compatibility.md)
- Prefix/session cache product surface: [`docs/cache-product.md`](docs/cache-product.md)
- Module status notes: [`docs/status/`](docs/status/)

## Supported Models

| Architecture | Apple Silicon | CUDA | INT4 (GPTQ) | Tensor Parallel |
|---|:---:|:---:|:---:|:---:|
| LLaMA (3.x, TinyLlama, Vicuna, Mistral) | ✓ | ✓ | ✓ | ✓ |
| Qwen3 dense (0.6B – 8B) | ✓ | ✓ | ✓ | ✓ |
| Qwen3-MoE (30B-A3B) | ✓ | ✓ | ✓ | — |
| Qwen2 / Qwen2.5 | ✓ | ✓ | ✓ | — |
| BERT (embeddings) | ✓ | — | — | — |
| Whisper ASR (tiny → large-v3-turbo) | ✓ | — | — | — |
| Qwen3-TTS (0.6B / 1.7B) | ✓ | — | — | — |
| CLIP / Chinese-CLIP / SigLIP (text + image) | ✓ | — | — | — |

Use any HuggingFace model ID:

```bash
ferrum run Qwen/Qwen3-4B
ferrum run meta-llama/Llama-3.2-3B-Instruct
ferrum run JunHowie/Qwen3-4B-GPTQ-Int4    # INT4 auto-detected
```

### Multi-modal

```bash
# Speech-to-text (WAV/M4A/MP3/FLAC, auto ffmpeg conversion)
ferrum transcribe whisper-turbo recording.m4a -l zh
ferrum serve whisper-turbo

# Text-to-speech (basic synthesis; optional reference-audio cloning)
ferrum tts qwen3-tts "Hello, welcome to Ferrum TTS" -o output.wav
ferrum serve qwen3-tts

# Embeddings (text + image)
ferrum embed OFA-Sys/chinese-clip-vit-base-patch16 --text "sunset at the beach"
ferrum embed google/siglip-base-patch16-224 --image photo.jpg
```

## Build options

```bash
# CPU only (default)
cargo install ferrum-cli

# Metal acceleration (macOS)
cargo install ferrum-cli --features metal

# CUDA acceleration from source (NVIDIA, requires CUDA toolkit + nvcc)
cargo install ferrum-cli --features cuda,vllm-moe-marlin,vllm-paged-attn-v2,fa2-source
```

## Architecture

```
crates/
├── ferrum-types          # Shared types
├── ferrum-interfaces     # Trait contracts (Backend<B>, ModelExecutor, ...)
├── ferrum-runtime        # Backend registry
├── ferrum-engine         # Continuous-batch engine, Metal shader pipeline
├── ferrum-models         # Model architectures (LlamaFamilyModel<B>, MoE, ...)
├── ferrum-kernels        # Custom CUDA + Metal kernels, decode runner
├── ferrum-attention      # Fused-transformer prototype (Metal/CPU)
├── ferrum-quantization   # GPTQ loader, Marlin, native safetensors
├── ferrum-tokenizer      # Tokenization
├── ferrum-sampler        # Top-k/p, temperature, repetition penalty, JSON-mode
├── ferrum-scheduler      # Continuous batching, paged-KV scheduling
├── ferrum-kv             # Paged KV cache (CUDA + Metal pools)
├── ferrum-server         # HTTP API
├── ferrum-cli            # Binary entry point
└── ferrum-testkit        # Test infrastructure
```

Architecture v2 (Model-as-Code) means the model layer is an explicit Rust generic over a `Backend<B>` trait, not a config-driven runner. Adding a backend = implementing the trait, not editing models. See [docs/architecture-v2.md](docs/architecture-v2.md).

## Status

What works today:
- CLI chat, OpenAI-compatible HTTP server with streaming
- Continuous batching, PagedAttention (CUDA + Metal pools), prefix caching, preemption
- OpenAI-style function tool calling, including required tool calls and streaming `tool_calls` deltas
- Custom CUDA decode runner (Qwen3, LLaMA): 2× over Candle baseline
- Apple Silicon MoE inference (Qwen3-30B-A3B) — correctness, multi-turn, default serve startup, and multi-sequence serving gates covered by the Metal README gate
- INT4 GPTQ with Marlin fused kernel (Blackwell + Ampere); also Triton w4a16
- Tensor parallelism (multi-GPU NCCL, persistent per-rank threads)
- Speculative decoding (`--spec-draft <MODEL>` DeepMind accept/reject)
- Structured output (`json_object` best-effort plus strict `json_schema` validation for the supported schema subset)
- Whisper ASR (Metal-accelerated forward pass) + Qwen3-TTS
- Top-k / top-p / temperature / repetition penalty

Known regressions / in-progress:
- Apple Silicon dense at c = 4 underperforms c = 1 on small models (paged-batched is below crossover). Per-token mode remains the default for c ≤ 4 until the small-m path catches up.
- FP8 (Hopper / Blackwell) — INT4 path is at 24% peak DRAM bandwidth, so there's headroom before FP8 becomes the bottleneck.

## Roadmap

See [docs/ROADMAP.md](docs/ROADMAP.md) for the full picture.

Near-term:
- v0.1: CUDA + Apple Silicon production release with concurrent serving benchmarks
- v0.2: Broader release matrix and long-context serving benchmarks
- v0.3: Long-context tuning (32k+), more architectures (Phi, DeepSeek, Gemma)

## License

MIT
