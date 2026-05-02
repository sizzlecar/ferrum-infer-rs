# ferrum-infer-rs

[![Crates.io](https://img.shields.io/crates/v/ferrum-cli.svg)](https://crates.io/crates/ferrum-cli)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/sizzlecar/ferrum-infer-rs/blob/main/LICENSE)

Production-grade LLM inference in Rust. Single binary, OpenAI-compatible, runs on Apple Silicon and CUDA.

[中文说明](README_zh.md)

## What it is

ferrum-infer-rs is a Rust-native inference engine for transformer LLMs:
single binary, no Python, OpenAI-compatible HTTP API, seconds to start.

Designed for single-GPU servers, edge devices, and Apple Silicon —
where Docker image size, cold start time, and Python toolchain friction matter.

## Performance highlight: Apple Silicon at concurrency

The hard case for laptop inference is concurrent serving. ferrum holds its own at single-request decode and pulls ahead as concurrency goes up. Same machine, same `Q4_K_M` GGUFs, same OpenAI-compatible HTTP load — see the audit-quality report at [`docs/bench/macos-2026-05-02/`](docs/bench/macos-2026-05-02/) (env, scripts, raw JSON, logs).

**M1 Max 32 GB · Q4_K_M · output throughput (tok/s)** — best of multiple runs, see [bench report § Methodology](docs/bench/macos-2026-05-02/README.md#methodology--why-two-reruns) for variance and re-run protocol.

| Model | c | ferrum | llama.cpp (b8960) | mistralrs (0.8.1) |
|---|---:|---:|---:|---:|
| LLaMA-3.1-8B | 1 | 29.1 | 28.7 | 30.2 |
| LLaMA-3.1-8B | 8 | **51.3** | 42.3 | 14.6 |
| LLaMA-3.1-8B | 16 | **96.7** | 67.2 | 23.3 |
| Qwen3-8B | 16 | **93.2** | 68.6 | 23.5 |
| Qwen3-30B-A3B (MoE) | 16 | 79.2¹ | 83.4 | panic² |

> ¹ ferrum MoE c ≥ 8 requires `FERRUM_MOE_BATCHED=1 FERRUM_MOE_BATCHED_DECODE=1` (currently opt-in). Without it, MoE c = 16 falls to 48 tok/s. ² mistralrs 0.8.1 PoisonError-panics on Qwen3-30B-A3B-Q4_K_M (`add_request.rs:466`) — not a ferrum issue.

> The Qwen3-30B-A3B (MoE) row is the headline. That's the model where Apple Silicon Rust support was effectively missing two months ago. ferrum closed it from a 51 → 80 tok/s gap to llama.cpp in a single PR (#81) by mirroring the Phase-4 paged-KV scaffolding into `Qwen3MoeModel`. On the dense 8B models, ferrum is +36–44% over llama.cpp at c = 16.

The full 36-cell grid (c = 1, 4, 8, 16 across all three engines and three models, including TPOT / TTFT distributions) is in the [bench report](docs/bench/macos-2026-05-02/README.md).

## Performance highlight: NVIDIA GPUs (CUDA)

ferrum maintains a custom CUDA decode runner with INT4 Marlin support. Numbers from RTX PRO 6000 (Blackwell):

**Qwen3-4B**

| Mode | Decode (tok/s) | VRAM |
|---|---:|---:|
| FP16 (eager) | 70.3 | ~8 GB |
| FP16 + CUDA Graphs | 82.9 (+18%) | ~8 GB |
| INT4 (GPTQ + Marlin) | **130.4 (+85%)** | **~2.5 GB (-69%)** |
| 4 concurrent (INT4) | 124.2 | ~2.5 GB |

**TinyLlama-1.1B**

| Backend | Decode (tok/s) |
|---|---:|
| Candle | 126 |
| ferrum CUDA | **256.5 (+103%)** |

vLLM-style scheduling features included: PagedAttention, continuous batching, FlashAttention-2 prefill, batched decode, custom fused kernels, piecewise CUDA Graphs, NCCL tensor parallel.

## Comparison

|  | ferrum | vLLM | llama.cpp | mistralrs |
|---|---|---|---|---|
| Language | Rust | Python+CUDA | C++ | Rust |
| Single binary | ✓ | ✗ (Docker) | ✓ | ✓ |
| Apple Silicon | ✓ (incl. MoE) | ✗ | ✓ | partial (no MoE) |
| CUDA | ✓ (custom) | ✓ (best) | ✓ | ✓ |
| Concurrent serving | ✓ | ✓ (best) | ✓ | ✓ |
| Continuous batching | ✓ | ✓ | partial | ✓ |
| INT4 quantization | ✓ Marlin / Triton | GPTQ / AWQ | GGUF only | varies |
| OpenAI-compatible API | ✓ | ✓ | ✓ | ✓ |
| Embeddable as a library | ✓ | ✗ | ✓ | ✓ |

## Quick Start

### Prebuilt binaries

```bash
# Linux x86_64
curl -L https://github.com/sizzlecar/ferrum-infer-rs/releases/latest/download/ferrum-linux-x86_64.tar.gz | tar xz
./ferrum --help

# macOS Apple Silicon (Metal)
curl -L https://github.com/sizzlecar/ferrum-infer-rs/releases/latest/download/ferrum-macos-aarch64.tar.gz | tar xz
./ferrum --help
```

Linux x86_64 is the CPU build. macOS aarch64 is the Metal build (the same backend that beats llama.cpp at c=16 on the [Group A bench](docs/bench/macos-2026-05-02/README.md)). For CUDA, build from source.

### From source

```bash
# crates.io
cargo install ferrum-cli

# or git
cargo build --release -p ferrum-cli --bin ferrum
```

### Run

```bash
# Set HF token for gated models (e.g. Llama 3.x)
export HF_TOKEN=hf_your_token_here

# Chat directly
ferrum run qwen3:4b

# Or serve via OpenAI-compatible API
ferrum serve --model qwen3:4b --port 8000
```

API call:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3:4b","messages":[{"role":"user","content":"Hello"}]}'
```

### Docker (CPU)

A prebuilt CPU image is published to GHCR on each tagged release (`:cpu`, `:cpu-<version>`). Runs on any x86_64 Linux host — no Rust toolchain needed.

```bash
docker pull ghcr.io/sizzlecar/ferrum-infer-rs:cpu

docker run --rm -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  ghcr.io/sizzlecar/ferrum-infer-rs:cpu \
  serve --model qwen3:0.6b --port 8000
```

CUDA and Metal images are on the roadmap — for now run Metal natively on macOS, CUDA natively on Linux.

## Supported Models

| Architecture | Apple Silicon | CUDA | INT4 (GPTQ) | Tensor Parallel |
|---|:---:|:---:|:---:|:---:|
| LLaMA (3.x, TinyLlama, Vicuna, Mistral) | ✓ | ✓ | ✓ | ✓ |
| Qwen3 dense (0.6B – 8B) | ✓ | ✓ | ✓ | ✓ |
| Qwen3-MoE (30B-A3B) | ✓ | — | — | — |
| Qwen2 / Qwen2.5 | ✓ | ✓ | ✓ | — |
| BERT (embeddings) | ✓ | — | — | — |
| Whisper ASR (tiny → large-v3-turbo) | ✓ | — | — | — |
| Qwen3-TTS (0.6B / 1.7B, voice clone) | ✓ | — | — | — |
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

# Text-to-speech (incl. voice clone via ICL prompting)
ferrum tts qwen3-tts "Hello, welcome to Ferrum TTS" -o output.wav
ferrum tts qwen3-tts "你好" --ref-audio ref.wav --ref-text "参考文本" -o clone.wav
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

# CUDA acceleration (NVIDIA, requires CUDA toolkit + nvcc)
cargo install ferrum-cli --features cuda
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
- Custom CUDA decode runner (Qwen3, LLaMA): 2× over Candle baseline
- Apple Silicon MoE inference (Qwen3-30B-A3B) — matches llama.cpp at c=16
- INT4 GPTQ with Marlin fused kernel (Blackwell + Ampere); also Triton w4a16
- Tensor parallelism (multi-GPU NCCL, persistent per-rank threads)
- Speculative decoding (`--spec-draft <MODEL>` DeepMind accept/reject)
- Structured output (`response_format: json_object` + `json_schema` with DFA-guided masking)
- Whisper ASR (Metal-accelerated forward pass) + Qwen3-TTS (voice clone, streaming)
- Top-k / top-p / temperature / repetition penalty

Known regressions / in-progress:
- Apple Silicon dense at c = 4 underperforms c = 1 on small models (paged-batched is below crossover). Per-token mode remains the default for c ≤ 4 until the small-m path catches up.
- FP8 (Hopper / Blackwell) — INT4 path is at 24% peak DRAM bandwidth, so there's headroom before FP8 becomes the bottleneck.

## Roadmap

See [docs/ROADMAP.md](docs/ROADMAP.md) for the full picture.

Near-term:
- v0.1: Apple Silicon Group A production release with concurrent serving benchmarks (this PR)
- v0.2: CUDA serving benchmark vs vLLM on commodity hardware (RTX 4090)
- v0.3: Long-context tuning (32k+), more architectures (Phi, DeepSeek, Gemma)

## License

MIT
