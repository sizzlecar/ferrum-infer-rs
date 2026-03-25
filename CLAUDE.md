# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What is this?

Ferrum Infer is a Rust-native LLM inference engine (v0.2.0 MVP). Single binary, no Python — supports Metal (macOS), CUDA (NVIDIA), and CPU backends. Targets vLLM-level performance with PagedAttention, continuous batching, and fused CUDA kernels.

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
```

## Architecture

**Dependency layers (bottom-up):**

1. **Foundation (no GPU deps):** `ferrum-types` (shared types, errors), `ferrum-interfaces` (trait contracts: ComputeBackend, ModelExecutor, Scheduler, KvCacheManager, Sampler, Tokenizer)
2. **Core logic (hardware-agnostic):** `ferrum-scheduler` (continuous batching, priority), `ferrum-sampler` (top-k/p, temperature), `ferrum-tokenizer` (HF wrapper), `ferrum-kv` (paged KV cache, block allocation), `ferrum-runtime` (backend abstraction)
3. **Application:** `ferrum-engine` (orchestration, ContinuousBatchEngine), `ferrum-models` (Qwen3/Qwen2/LLaMA/BERT architectures + weight loading), `ferrum-server` (Axum HTTP, OpenAI-compatible API), `ferrum-cli` (binary entry point)
4. **Accelerators (feature-gated):** `ferrum-cuda-kernels` (fused RmsNorm, SiLU+mul — PTX precompiled at build time)
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

## Build Scripts

- **`ferrum-engine/build.rs`**: Compiles Metal shaders (.metal → .air → .metallib) on macOS via `xcrun`. Generates empty stub on non-Apple platforms.
- **`ferrum-cuda-kernels/build.rs`**: Compiles CUDA .cu files to PTX using `bindgen_cuda`. Requires CUDA_HOME env var. Generates `ptx.rs` in OUT_DIR.

## Model Support

Qwen3 (0.6B–4B), Qwen2.5-Instruct (0.5B–7B), Llama-3.2-Instruct (1B–3B), TinyLlama-1.1B-Chat. Models are downloaded from HuggingFace and cached locally.

## Config

Runtime defaults in `ferrum.toml`. Model cache at `~/.cache/huggingface` (shared with HF Python).
