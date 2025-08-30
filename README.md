# Ferrum Infer (Rust) — vLLM‑inspired Inference Core (Pre‑alpha)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Ferrum Infer is a Rust research project exploring vLLM‑style LLM inference. The repository currently focuses on clean abstractions (traits) and crate layout. There is no end‑to‑end runnable server yet.

## ⚠️ Project Status

- **Pre‑alpha / Work in Progress**
- **Interfaces only**: most modules expose traits and skeletons; many implementations are placeholders
- **Not production‑ready**: APIs and code structure will change

## ✨ Goals (aligned with vLLM)

- **PagedAttention KV cache** with block tables and memory swapping
- **Continuous batching** (prefill/decode separation, token‑level scheduling)
- **Prefix cache / prompt deduplication**
- **Backend abstraction** (Candle/ONNX Runtime; CUDA/ROCm/Metal/CPU)
- **OpenAI‑compatible API** with streaming

## 📦 Crate Overview

- `crates/ferrum-core`: core types and traits (`InferenceEngine`, `Backend`, `Scheduler`, `BatchManager`, `CacheManager`, `MemoryManager`, `Model`)
- `crates/ferrum-engine`: engine skeleton (executor, batch manager, paged KV cache, attention interfaces)
- `crates/ferrum-models`: model abstractions and config
- `crates/ferrum-cache`: cache traits and types
- `crates/ferrum-runtime`: runtime and device/memory abstractions
- `crates/ferrum-scheduler`: scheduling/batching configuration types
- `crates/ferrum-server`: OpenAI‑compatible data types (HTTP server not implemented)
- `crates/ferrum-cli`: CLI scaffolding (TBD)

## ✅ What Exists Today

- Well‑scoped traits for engine, backend, scheduler, cache, memory, and models
- Skeletons for paged KV cache and attention with placeholder logic
- OpenAI‑compatible request/response types under `ferrum-server`

## 🚫 Not Implemented Yet

- Real backends (Candle/ORT) and model execution
- FlashAttention/PagedAttention GPU kernels and efficient memory management
- Prefill/Decode separation, preemption, prefix cache reuse
- OpenAI HTTP server, SSE streaming, auth, and metrics wiring
- Benchmarks and performance guarantees

## 🔧 Build

Prerequisites: Rust 1.70+ (latest stable recommended)

```bash
git clone https://github.com/sizzlecar/ferrum-infer-rs.git
cd ferrum-infer-rs
cargo build
```

Note: Building succeeds for trait/skeleton crates; running a server is not supported yet.

## 🗺️ Roadmap (high‑level)

1) MVP
- Plug in Candle backend, tokenizer, and basic sampling
- Single‑node dynamic batching, naive per‑sequence KV cache
- OpenAI `/v1/chat/completions` with streaming (SSE)

2) vLLM‑style features
- PagedAttention: block tables, allocation/eviction/compaction, GPU/CPU swap
- Prefill/Decode separation, token‑level continuous batching, prefix cache reuse
- Scheduler with fairness/priority and bucketed padding

3) Performance and scale
- FlashAttention/fused kernels via FFI; quantization options
- Multi‑GPU (tensor/pipeline parallel), improved observability and benchmarks

## 🤝 Contributing

Contributions are welcome! The project is rapidly evolving; please open an issue to discuss design/implementation before large changes.

## 📝 License

Licensed under the MIT License. See `LICENSE`.

## 🙏 Acknowledgements

Inspired by the design principles behind vLLM and Rust community projects (Candle, Axum, Tokio).