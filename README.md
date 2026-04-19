# Ferrum Infer

[![Crates.io](https://img.shields.io/crates/v/ferrum-cli.svg)](https://crates.io/crates/ferrum-cli)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A Rust-native LLM inference engine. Load models from Hugging Face, chat locally or serve via OpenAI-compatible API. Single binary, no Python, no runtime dependencies.

[中文说明](README_zh.md)

## Install

### Docker (recommended — no toolchain, no build)

Pre-built images on GitHub Container Registry:

```bash
# GPU host (NVIDIA + Container Toolkit)
docker run --rm -it --gpus all \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  ghcr.io/sizzlecar/ferrum-infer-rs:latest \
  run qwen3:4b --backend cuda

# CPU-only host
docker run --rm -it \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  ghcr.io/sizzlecar/ferrum-infer-rs:cpu \
  run qwen3:0.6b

# HTTP server (OpenAI-compatible)
docker run --rm -p 8000:8000 --gpus all \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  ghcr.io/sizzlecar/ferrum-infer-rs:latest \
  serve --model qwen3:4b --backend cuda
```

Tags: `latest` (= CUDA), `cuda`, `cpu`, `cuda-0.7.0`, `cpu-0.7.0`, ...

### crates.io

```bash
cargo install ferrum-cli
```

### From source

```bash
# CPU / Metal (macOS auto-selects Metal)
cargo build --release -p ferrum-cli --bin ferrum

# CUDA (Linux + NVIDIA)
CUDA_HOME=/usr/local/cuda cargo build --release --features cuda -p ferrum-cli --bin ferrum
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

## Supported Architectures

Any Hugging Face model using a supported architecture works out of the box:

### Text Generation

| Architecture | CUDA Decode | INT4 (GPTQ) | Tensor Parallel | Example Models |
|-------------|-------------|-------------|-----------------|----------------|
| **LLaMA** | Yes | Yes | Yes | Llama-3.x, TinyLlama, Vicuna, Alpaca, ... |
| **Qwen3** | Yes | Yes | Yes | Qwen3-0.6B ~ 4B |
| **Qwen2** | — | — | — | Qwen2.5-Instruct-0.5B ~ 7B |

### Speech-to-Text (Whisper ASR)

| Architecture | Metal | CUDA | Example Models |
|-------------|-------|------|----------------|
| **Whisper** | Yes | — | whisper-tiny, whisper-base, whisper-small, whisper-medium, whisper-large-v3, **whisper-turbo** (recommended) |

### Text-to-Speech (Qwen3-TTS)

| Architecture | Metal | CPU | Voice Clone | Example Models |
|-------------|-------|-----|-------------|----------------|
| **Qwen3-TTS** | Yes | Yes | Yes (ICL) | Qwen3-TTS-12Hz-0.6B-Base |

### Embeddings (text + image)

| Architecture | Modality | Embedding Dim | Example Models |
|-------------|----------|--------------|----------------|
| **CLIP** | Text + Image | 512/768 | openai/clip-vit-base-patch32 |
| **Chinese-CLIP** | Text + Image | 512 | OFA-Sys/chinese-clip-vit-base-patch16 |
| **SigLIP** | Text + Image | 768 | google/siglip-base-patch16-224 |
| **BERT** | Text | 768 | google-bert/bert-base-chinese |

```bash
# Text generation
ferrum run Qwen/Qwen3-4B
ferrum run llama3.2:3b

# Speech-to-Text (supports WAV/M4A/MP3/FLAC — auto ffmpeg conversion)
ferrum transcribe whisper-turbo recording.m4a -l zh
ferrum transcribe whisper-turbo meeting.wav -l en

# Text-to-Speech
ferrum tts qwen3-tts "Hello, welcome to Ferrum TTS" -o output.wav
ferrum tts qwen3-tts "你好欢迎使用语音合成系统" -o output.wav

# Voice clone (ICL mode — clone any voice from 5s reference audio)
ferrum tts qwen3-tts "你好" --ref-audio ref.wav --ref-text "参考文本" -o clone.wav

# Whisper API server (OpenAI-compatible)
ferrum serve whisper-turbo
curl localhost:8000/v1/audio/transcriptions -F "file=@audio.wav" -F "language=zh"

# Embeddings (text + image)
ferrum embed OFA-Sys/chinese-clip-vit-base-patch16 --text "sunset at the beach"
ferrum embed google/siglip-base-patch16-224 --image photo.jpg

# Embedding API server
ferrum serve --model OFA-Sys/chinese-clip-vit-base-patch16
curl localhost:8000/v1/embeddings -d '{"model":"clip","input":"hello"}'
curl localhost:8000/v1/embeddings -d '{"model":"clip","input":{"image":"/path/to/photo.jpg"}}'
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
| `ferrum transcribe <model> <audio>` | Speech-to-text (Whisper, supports WAV/M4A/MP3) |
| `ferrum tts <model> <text>` | Text-to-speech (Qwen3-TTS, voice clone with `--ref-audio`) |
| `ferrum embed <model>` | Generate embeddings (BERT/CLIP/SigLIP, text + image) |

## API Endpoints

```bash
# Chat completions (OpenAI-compatible)
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3:0.6b","messages":[{"role":"user","content":"Hello"}]}'

# Audio transcription (OpenAI-compatible, multipart form)
curl http://localhost:8000/v1/audio/transcriptions \
  -F "file=@audio.wav" -F "language=zh"

# Embeddings
curl http://localhost:8000/v1/embeddings \
  -d '{"model":"clip","input":"hello"}'

# List models
curl http://localhost:8000/v1/models

# Health check
curl http://localhost:8000/health
```

## Performance

Benchmarked on **RTX PRO 6000 (Blackwell)**:

### Qwen3-4B

| Mode | FP16 (eager) | FP16 + CUDA graph | INT4 (GPTQ + Marlin) |
|------|--------------|-------------------|----------------------|
| Single request decode | 70.3 tok/s | **82.9 tok/s (+18%)** | **130.4 tok/s** |
| 4 concurrent (batch) | 109.4 tok/s | — | **124.2 tok/s** |
| TPOT (p50) | 14.2 ms | **12.1 ms** | — |
| VRAM | ~8 GB | — | **~2.5 GB (-69%)** |

> CUDA graph replay is automatic after 3-step warmup; eliminates per-step launch overhead and sits on the Blackwell + CUDA 13 path we hardened in this cycle (see [docs/phase-e-cuda-status.md](docs/phase-e-cuda-status.md)).

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

### Whisper ASR (Apple Silicon Metal)

| Model | 5-min audio | Realtime factor |
|-------|------------|-----------------|
| whisper-large-v3-turbo | **~72s** | **4.2x realtime** |
| whisper-tiny | ~20s | 15x realtime |

> Custom Whisper forward pass with rustfft STFT. Full decode pipeline: timestamp-based sequential decode, temperature fallback, compression ratio check. Mel precision matches Python whisper exactly.

### Qwen3-TTS (Apple Silicon Metal)

| Text | Audio | Time | RTF |
|------|-------|------|-----|
| 29 chars Chinese | 4.6s | **11.3s** | **2.8x realtime** |
| Voice clone (ICL, 5s ref) | 5.3s | 13.1s | 2.5x realtime |

> All-Metal fused transformer pipeline: custom GEMM (64×32 simdgroup tiles), fused residual+norm, flash attention with layer_scale. Full Mimi-based vocoder with 8-layer pre-transformer. Zero-copy on Apple Silicon unified memory.

### Key Optimizations

- **Custom CUDA decode runner**: bypasses candle for the decode hot path (Qwen3 + LLaMA)
- **INT4 quantization**: GPTQ models auto-detected, Marlin fused INT4×FP16 kernel
- **Tensor parallelism**: persistent per-rank threads, barrier sync, NCCL all-reduce (Megatron-LM pattern)
- **Batched attention kernel**: single launch for all batch items (SM utilization 17%→67%)
- **Batched RoPE**: per-item positions in single kernel launch
- **Custom CUDA kernels**: fused RmsNorm, SiLU×mul, RoPE, decode attention (all on single stream)
- **Flash Decoding**: split-K for long-context decode (auto at KV > 256)
- **Batch decode**: batched cuBLAS GEMM + batched attention for concurrent requests
- **Metal TTS pipeline**: all-Metal fused transformer for talker (28 layers) + SubTalker (5 layers) + vocoder (8 layers), cached GPU buffers, fused residual+norm kernel, layer_scale support
- **TTS voice clone**: ICL prompting with speaker encoder (ECAPA-TDNN) + speech tokenizer (Mimi RVQ)
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
- CLIP/Chinese-CLIP/SigLIP embeddings (text + image, `/v1/embeddings` API)
- Whisper ASR (speech-to-text, Metal accelerated, `/v1/audio/transcriptions` API)
- Multi-format audio support (WAV/M4A/MP3/FLAC via ffmpeg)
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
├── ferrum-models         # Model architectures (LLaMA, Qwen2, Qwen3, BERT, Whisper)
├── ferrum-kernels   # Custom CUDA kernels + decode runner
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
