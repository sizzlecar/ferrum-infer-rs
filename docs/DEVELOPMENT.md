# Development Guide

## Overview

Ferrum targets production CUDA inference, but daily development happens on macOS. This works because the project's architecture cleanly separates scheduling logic (pure Rust, no hardware dependency) from compute kernels (GPU-specific). About 70%+ of the codebase — and the core value of the project — lives in the scheduling layer and can be fully developed and tested on Mac.

## Architecture for Testability

The key trait boundaries that enable hardware-independent development:

```
ModelExecutor    — prefill/decode abstraction, mockable
KvCacheManager   — block allocation/deallocation, mockable
ComputeBackend   — device abstraction, mockable
Scheduler        — pure logic, no hardware dependency at all
```

All scheduling, KV cache management, continuous batching, and API serving code depends only on these traits, never on concrete GPU implementations.

## Development Layers

### Layer 1: Scheduling & Orchestration (Mac, no GPU needed)

These crates have zero hardware dependency and are fully testable on Mac:

| Crate | What to test | How |
|---|---|---|
| `ferrum-scheduler` | Continuous batching, priority, preemption, admission control | `#[tokio::test]`, pure logic |
| `ferrum-kv` | PagedAttention block allocation, CoW, eviction, defragmentation | Unit tests with in-memory blocks |
| `ferrum-engine` | Iteration loop, request lifecycle, batch orchestration | MockExecutor + MockKvCacheManager |
| `ferrum-server` | API routing, SSE streaming, middleware | HTTP integration tests |
| `ferrum-sampler` | Temperature, top-k/p, repetition penalty | Unit tests with synthetic logits |
| `ferrum-tokenizer` | Encoding, decoding, chat templates | Unit tests |
| `ferrum-types` | Serialization, type invariants | Unit tests |

### Layer 2: Model Execution (Mac with Candle CPU/Metal)

Use Candle CPU or Metal backend with small models for correctness validation:

| What | How |
|---|---|
| Model loading | Small model (e.g. Qwen2.5-0.5B) from HF cache |
| Prefill/decode correctness | End-to-end with Candle CPU, verify output makes sense |
| Metal acceleration | `--features metal` on Mac for faster local iteration |
| Streaming | CLI `ferrum run` with a small model |

This layer validates that the engine orchestration integrates correctly with a real model, but is **not** for performance benchmarking.

### Layer 3: CUDA Kernels (CI / Remote GPU)

CUDA-specific code only compiles and runs on Linux with NVIDIA GPU:

| What | Where |
|---|---|
| FlashAttention/FlashInfer FFI | CI with GPU runner or remote GPU machine |
| Quantization kernels | CI with GPU runner or remote GPU machine |
| Performance benchmarks | Dedicated GPU machine, on-demand |
| Multi-GPU tensor parallelism | Multi-GPU CI or cloud instance |

## Mock Components

### MockExecutor

For testing the scheduling layer without loading a real model:

```rust
use std::time::Duration;
use ferrum_interfaces::ModelExecutor;

/// Simulates model execution with configurable latency.
/// No model weights, no GPU — pure async simulation.
struct MockExecutor {
    prefill_latency: Duration,
    decode_latency: Duration,
    vocab_size: usize,
}

#[async_trait]
impl ModelExecutor for MockExecutor {
    fn info(&self) -> &ModelInfo {
        // Return static model metadata
    }

    async fn prefill(&self, input: &PrefillInput) -> Result<PrefillOutput> {
        // Simulate prefill compute time
        tokio::time::sleep(self.prefill_latency).await;
        // Return synthetic logits
        Ok(PrefillOutput { /* ... */ })
    }

    async fn decode(&self, input: &DecodeInput) -> Result<DecodeOutput> {
        // Simulate per-token decode latency
        tokio::time::sleep(self.decode_latency).await;
        // Return random token from vocab
        Ok(DecodeOutput { /* ... */ })
    }
}
```

Use cases:
- Continuous batching correctness: 100+ concurrent requests, verify scheduling order
- Preemption testing: inject slow prefill, verify decode requests aren't starved
- Backpressure: saturate the scheduler, verify admission control
- Cancellation safety: cancel mid-stream, verify KV cache cleanup

### MockKvCacheManager

For testing PagedAttention logic without GPU memory:

```rust
/// In-memory block pool simulating GPU memory constraints.
struct MockKvCacheManager {
    block_size: usize,
    total_blocks: usize,
    allocated: DashMap<RequestId, Vec<BlockId>>,
}

#[async_trait]
impl KvCacheManager for MockKvCacheManager {
    async fn allocate(&self, request: &AllocationRequest) -> Result<Arc<dyn KvCacheHandle>> {
        // Allocate from fixed-size block pool
        // Return error when pool exhausted (simulates OOM)
    }

    async fn deallocate(&self, request_id: RequestId) -> Result<()> {
        // Return blocks to pool
    }
}
```

Use cases:
- Block allocation/deallocation correctness
- OOM handling: what happens when blocks run out
- Eviction policy: LRU/priority-based eviction under memory pressure
- CoW: shared prefix blocks with reference counting
- Fragmentation: allocate/free patterns, verify defragmentation

## Feature Flags

The workspace uses feature flags to isolate hardware-specific code:

```toml
# ferrum-runtime/Cargo.toml
[features]
default = ["candle"]          # Mac development default
cuda = []                     # Linux + NVIDIA GPU
metal = []                    # macOS GPU acceleration
```

```toml
# ferrum-engine/Cargo.toml
[features]
default = []
cuda = ["candle-core/cuda"]   # CUDA kernels
metal = ["candle-core/metal", "dep:metal", "dep:mpsgraph", ...]
```

Future CUDA FFI backend:

```toml
# ferrum-runtime/Cargo.toml (planned)
[features]
cuda-ffi = ["dep:cuda-sys"]   # FlashAttention/FlashInfer FFI, Linux+CUDA only
```

Key rule: `cargo check --workspace` and `cargo test --workspace` must always pass on Mac without any `--features` flag. CUDA code is behind `#[cfg(feature = "cuda")]` and never compiles on Mac.

## CI Strategy

```
┌─────────────────────────────────────────────────────┐
│  Every Push (Mac / Linux runners)                   │
│  ├── cargo fmt --check                              │
│  ├── cargo clippy --workspace                       │
│  ├── cargo test --workspace  (no GPU features)      │
│  └── cargo build -p ferrum-cli                      │
├─────────────────────────────────────────────────────┤
│  GPU Runner (on demand / nightly)                   │
│  ├── cargo test --workspace --features cuda          │
│  ├── integration tests with real models             │
│  └── benchmark suite (throughput, latency)           │
└─────────────────────────────────────────────────────┘
```

## Daily Workflow

```bash
# 1. Build and check (always works on Mac)
cargo check --workspace --all-targets
cargo clippy --workspace --all-targets

# 2. Run all non-GPU tests
cargo test --workspace

# 3. Run specific scheduling tests
cargo test -p ferrum-scheduler
cargo test -p ferrum-kv

# 4. End-to-end with a small model (Candle CPU)
cargo run -p ferrum-cli --bin ferrum -- run Qwen/Qwen2.5-0.5B-Instruct

# 5. With Metal acceleration on Mac
cargo run -p ferrum-cli --bin ferrum --features metal -- run Qwen/Qwen2.5-0.5B-Instruct

# 6. HTTP server smoke test
cargo run -p ferrum-cli --bin ferrum -- serve --model Qwen/Qwen2.5-0.5B-Instruct --port 8000
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen2","messages":[{"role":"user","content":"hello"}]}'
```

## Adding New CUDA Kernels

When adding CUDA-dependent code:

1. Define the interface as a trait in `ferrum-interfaces` or `ferrum-runtime` — no CUDA types
2. Implement a mock version that works on CPU for testing
3. Implement the CUDA version behind `#[cfg(feature = "cuda-ffi")]`
4. Write tests that run against the mock on Mac, against real CUDA in GPU CI
5. Never import CUDA types outside of `cfg`-gated modules

```rust
// In ferrum-runtime
pub trait AttentionKernel: Send + Sync {
    fn paged_attention(
        &self,
        query: &TensorRef,
        key_cache: &BlockTable,
        value_cache: &BlockTable,
        scale: f32,
    ) -> Result<TensorRef>;
}

// Mock (always available)
pub struct CpuAttention;
impl AttentionKernel for CpuAttention { /* naive matmul */ }

// CUDA (behind feature flag)
#[cfg(feature = "cuda-ffi")]
pub struct FlashInferAttention { /* FFI handle */ }
#[cfg(feature = "cuda-ffi")]
impl AttentionKernel for FlashInferAttention { /* FFI call */ }
```
