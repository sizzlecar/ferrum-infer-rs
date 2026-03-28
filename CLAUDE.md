# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What is this?

Ferrum Infer is a Rust-native LLM inference engine. Single binary, no Python — supports Metal (macOS), CUDA (NVIDIA), and CPU backends. Targets vLLM-level performance with PagedAttention, continuous batching, and custom CUDA kernels.

**Current performance (RTX PRO 6000 Blackwell, Qwen3-4B):**

| Mode | FP16 | INT4 (Marlin) |
|------|------|---------------|
| Single request | 88.8 tok/s (TPOT 11.35ms) | **112.4 tok/s (TPOT 8.90ms)** |
| 4 concurrent | 109.4 tok/s | — |
| VRAM usage | ~8 GB | **~2.5 GB (-69%)** |

- INT4 quantization: GPTQ format auto-detected, Marlin fused kernel on Blackwell
- Paged KV attention with block reclamation
- Flash Decoding (split-K) for long contexts

## Build & Development Commands

```bash
cargo check --workspace --all-targets    # Fast compile validation
cargo build --workspace                  # Full build
cargo test --workspace                   # All tests
cargo test -p ferrum-scheduler           # Single crate tests
cargo fmt --all -- --check               # Format check (CI enforced)
cargo clippy --workspace --all-targets -- -A warnings  # Lint (CI advisory)

# Run CLI
cargo run -p ferrum-cli --bin ferrum -- run qwen3:0.6b
cargo run -p ferrum-cli --bin ferrum -- serve --model qwen3:0.6b --port 8000
cargo run -p ferrum-cli --bin ferrum -- pull qwen3:0.6b
cargo run -p ferrum-cli --bin ferrum -- list

# With Metal acceleration (macOS)
cargo run -p ferrum-cli --bin ferrum --features metal -- run qwen3:0.6b

# Benchmarks
cargo run -p ferrum-cli --bin ferrum -- bench qwen3:4b                          # sequential baseline
cargo run -p ferrum-cli --bin ferrum -- bench qwen3:4b --concurrency 4          # batch decode
cargo run -p ferrum-cli --bin ferrum -- bench qwen3:4b --max-tokens 1024        # long decode
cargo run -p ferrum-cli --bin ferrum -- bench qwen3:4b --long-context           # long prompt (~2k tokens)

# CUDA with batch decode + paged KV
FERRUM_MAX_BATCH=8 cargo run -p ferrum-cli --features cuda -- bench qwen3:4b --concurrency 4
FERRUM_PAGED_KV=1 FERRUM_KV_BLOCKS=128 cargo run -p ferrum-cli --features cuda -- bench qwen3:4b --concurrency 4
```

## Architecture

**Dependency layers (bottom-up):**

1. **Foundation (no GPU deps):** `ferrum-types` (shared types, errors), `ferrum-interfaces` (trait contracts: ComputeBackend, ModelExecutor, Scheduler, KvCacheManager, Sampler, Tokenizer)
2. **Core logic (hardware-agnostic):** `ferrum-scheduler` (continuous batching, priority), `ferrum-sampler` (top-k/p, temperature), `ferrum-tokenizer` (HF wrapper), `ferrum-kv` (paged KV cache, block allocation), `ferrum-runtime` (backend abstraction)
3. **Application:** `ferrum-engine` (orchestration, ContinuousBatchEngine), `ferrum-models` (Qwen3/Qwen2/LLaMA/BERT architectures + weight loading), `ferrum-server` (Axum HTTP, OpenAI-compatible API), `ferrum-cli` (binary entry point)
4. **Accelerators (feature-gated):** `ferrum-cuda-kernels` (CudaDecodeRunner with custom CUDA kernels — PTX precompiled at build time)
5. **Testing:** `ferrum-testkit` (mocks for all trait contracts — enables GPU-free testing)

**Key design rules:**
- `cargo check --workspace` and `cargo test --workspace` must pass on Mac without GPU features
- CUDA code lives behind `#[cfg(feature = "cuda")]`, Metal behind target OS gates
- Trait-based abstraction: all hardware-specific behavior goes through interfaces
- Shared types/traits go in `ferrum-types`/`ferrum-interfaces`, never duplicated in impl crates

## Code Style

- Rust 2021, `rustfmt.toml`: 4-space indent, max width 100, reordered imports
- `snake_case` functions/modules, `CamelCase` types/traits, `SCREAMING_SNAKE_CASE` constants
- Conventional commits: `feat(scope):`, `fix(scope):`, `refactor(scope):`

## CUDA Decode Runner

Candle handles weight loading and prefill (FlashAttention-2). Decode is fully controlled by `CudaDecodeRunner` in `ferrum-cuda-kernels`:

**Custom CUDA kernels** (PTX compiled at build time):
- `rms_norm.cu`, `fused_add_rms_norm.cu` — layer normalization
- `rope.cu` — rotary position embedding (Q+K fused)
- `fused_silu_mul.cu` — MLP activation (+ interleaved variant for batch)
- `decode_attention.cu` — single-block warp-cooperative attention
- `flash_decode_attention.cu` — split-K flash decoding for long contexts
- `paged_decode_attention.cu` — block-table indirect attention (+ split-K variant)
- `residual_add.cu` — element-wise residual

**Decode optimizations:**
- Double-buffered residual + cross-layer norm fusion (108 fewer kernel launches)
- Flash Decoding: split KV across blocks for GPU SM utilization (auto at kv_len > 256)
- Batch decode: batched cuBLAS GEMM (m=batch) with per-item attention loop
- Paged KV: GPU block pool with block-table indirection, free-list reclamation

## Build Scripts

- **`ferrum-engine/build.rs`**: Compiles Metal shaders (.metal → .air → .metallib) on macOS via `xcrun`. Generates empty stub on non-Apple platforms.
- **`ferrum-cuda-kernels/build.rs`**: Compiles CUDA .cu files to PTX using `bindgen_cuda`. Requires CUDA_HOME env var. Generates `ptx.rs` in OUT_DIR.

## Model Support

Qwen3 (0.6B–4B), Qwen2.5-Instruct (0.5B–7B), Llama-3.2-Instruct (1B–3B), TinyLlama-1.1B-Chat. Models are downloaded from HuggingFace and cached locally.

## Config

Runtime defaults in `ferrum.toml`. Model cache at `~/.cache/huggingface` (shared with HF Python).
