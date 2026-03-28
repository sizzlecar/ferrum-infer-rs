# Ferrum Infer Roadmap

## Vision

Production-grade LLM inference engine in Rust, targeting the same market as vLLM.

Core value proposition: **vLLM-level scheduling capabilities, rewritten with Rust's determinism and performance guarantees.**

- **Stable P99 latency** - no Python GIL, no GC pauses
- **Higher memory utilization** - precise KV cache lifecycle control, no Python object overhead
- **Single-binary deployment** - no conda environment, no Python dependency chain
- **Embeddable** - usable as a library in Rust/C++ services, not just a standalone process

## Current State

MVP phase on the `mvp` branch. Phase 1 (Core Scheduling Engine) is complete.

### What's done

- **PagedAttention (Phase 1.1)** — fully implemented and tested end-to-end
  - Logical-to-physical block mapping with indirection table (`BlockTable`)
  - Block-level KV tensor storage (`BlockStorage`) with per-layer K/V read/write
  - Block memory pool with allocation, deallocation, reuse, and ID 0 sentinel
  - Paged attention kernel (CPU reference): causal masking, GQA head mapping, softmax with numerical stability
  - Cross-block KV gather for non-contiguous memory access
  - `PagedKvCacheManager`: `write_kv()` / `read_kv()` through block table indirection
  - `PagedAttentionExecutor`: test executor using real paged KV (identity projections Q=K=V=embedding)
  - `PrefillInput.kv_cache` field: engine passes pre-allocated KV handle to executor, avoiding double allocation
  - 4 paged lifecycle integration tests + 5 engine integration tests with paged KV
  - 67 unit tests in `ferrum-kv`, all passing

- **Iteration-Level Continuous Batching (Phase 1.2)**
  - `Arc<EngineInner>` pattern for concurrent spawning from trait methods
  - `iteration_lock` (tokio::Mutex) serializes engine steps to prevent double-processing
  - `drive_to_completion()` loop drives iterations until a specific request completes
  - Mixed prefill+decode batches per iteration via scheduler's `next_batch()`
  - Background iteration loop (`start_loop()`) with `work_notify` signaling
  - 11 `continuous_batch_test` integration tests (single/multi/concurrent/streaming/latency/prefix)

- **Preemption (Phase 1.3)**
  - Recomputation-based preemption on KV cache OOM
  - Victim selection: lowest-priority decoding request (fewest generated tokens as tiebreaker)
  - Frees victim's KV cache, resets sequence state, re-submits to scheduler for re-prefill
  - Response/stream channels preserved across preemption
  - Deterministic RNG reset for consistent re-generation
  - 1 preemption integration test with tight block pool (5 concurrent requests, 6-block pool)

- **Prefix Caching (Phase 1.4)**
  - `PrefixCache`: exact prompt match with LRU eviction and stats (hits/misses/hit_rate)
  - Cached last-token logits alongside KV handle — cache hit skips executor prefill entirely
  - `clone_handle()` for KV reuse across requests with identical prompts
  - Engine-level integration: `run_prefill()` checks cache before executor call, stores result on miss
  - 1 prefix cache integration test proving second identical prompt skips prefill (executor.prefill_count() == 1)
  - 5 unit tests for prefix cache (storage, retrieval, filtering, LRU eviction, stats)

- **Scheduler correctness foundation**
  - Per-request KV isolation, cancellation safety, state-based admission control
  - Request state machine: waiting → running transitions with rollback on mismatch

- **Hardware-independent test infrastructure** (`ferrum-testkit`)
  - `MockModelExecutor`, `MockKvCacheManager`, `MockTokenizer`, `MockSampler`, `MockTensor`, `MockTensorFactory`
  - `PagedAttentionExecutor` — real paged KV usage without GPU
  - `TensorFactory` trait injection to decouple engine from candle_core

- **Engine decoupling**
  - KV allocation dims derived from `ModelInfo` instead of hardcoded 32/32/128
  - `TensorFactory` abstraction replaces direct candle tensor construction

- 3 model architectures (Llama, Qwen2, BERT) with Candle backend
- OpenAI-compatible HTTP API and CLI

### What's next

**Phase 2: CUDA Kernel Layer** — Custom CUDA kernels for decode, flash decoding, paged attention, and batch decode. Candle retained for prefill (FlashAttention-2) and weight loading.

## Gap Analysis vs vLLM

| Capability | vLLM | Ferrum | Gap |
|---|---|---|---|
| PagedAttention | Mature | **Done** — GPU paged KV pool, block-table attention kernel, free-list reclamation | Closed |
| Continuous Batching | Iteration-level, prefill/decode mixed | **Done** — iteration-level mixed batching, concurrent requests | Closed |
| Preemption | Swap/recomputation | **Done** — recomputation-based preemption with auto-resubmit | Closed |
| Prefix Caching | Yes | **Done** — exact-match prefix cache with LRU eviction | Closed |
| CUDA Kernel Optimization | FlashAttention / FlashInfer | **Done** — custom decode kernels, flash decoding (split-K), fused ops | Closed |
| Batch Decode | Batched GEMM + attention | **Done** — batched cuBLAS GEMM (m=batch), per-item attention | Partial (attention not yet batched) |
| Quantization | AWQ / GPTQ / FP8 | **Done** — GPTQ INT4 auto-detect, Marlin fused kernel (+48% vs FP16), Blackwell compatible | Closed |
| Tensor Parallelism | Multi-GPU via NCCL | Type stubs only | Large |
| Model Support | Dozens | 4 | Medium |
| Structured Output | JSON mode / grammar-guided | **Done** — JSON mode via logits biasing, OpenAI API support | Partial (grammar-guided future) |
| Benchmarking | Comprehensive | **Done** — sequential, concurrent, long-context modes | Closed |

## Architecture Principle

**Rust manages scheduling and orchestration. CUDA kernels come from mature libraries via FFI.**

The right division of labor:

- **Rust layer**: request lifecycle, KV cache management, continuous batching, memory allocation, kernel launch orchestration, API serving
- **CUDA layer**: FlashAttention, quantized matmul, fused kernels - via FFI bindings to existing battle-tested libraries
- **Candle**: retained as CPU fallback and development/testing backend

The `ferrum-runtime` crate already has extension points (`ComputeBackend` trait) for this layering.

## Phases

### Phase 1: Core Scheduling Engine (P0)

The scheduling layer is Ferrum's moat. This is where Rust's advantages are most tangible.

#### 1.1 True PagedAttention ✅

- ~~Logical block -> physical block mapping with indirection table~~
- Copy-on-Write for shared prefix blocks (framework exists, not yet exercised)
- ~~Dynamic block allocation and reclamation~~
- ~~Block-level memory pool with defragmentation~~
- ~~Block-level KV tensor storage with cross-block gather~~
- ~~CPU reference paged attention kernel (causal mask, GQA, softmax)~~
- ~~End-to-end integration tests with ContinuousBatchEngine~~

#### 1.2 Iteration-Level Continuous Batching ✅

- ~~Prefill and decode requests mixed in the same batch at iteration boundaries~~
- ~~Dynamic insertion: new requests join running batches without waiting for batch completion~~
- ~~Dynamic eviction: completed/cancelled requests leave immediately~~
- ~~Iteration-level scheduling loop replaces current per-request flow~~

#### 1.3 Preemption ✅

- ~~Recomputation-based preemption on KV cache OOM~~
- ~~Victim selection with automatic re-submission to scheduler~~
- ~~Response/stream channels preserved across preemption~~
- SLA-aware scheduling: latency targets per request (future)
- Swap-based preemption: KV cache offload to CPU (future)

#### 1.4 Prefix Caching ✅

- ~~Exact-match prefix cache with LRU eviction~~
- ~~Cached last-token logits — cache hit skips executor prefill entirely~~
- ~~`clone_handle()` for KV reuse across identical prompts~~
- Partial prefix matching / radix tree (future)
- Block-level sharing via `share_prefix_blocks()` for paged KV (future)

### Phase 2: CUDA Kernel Layer (P0)

Without this, performance cannot compete. The strategy is FFI bindings, not reimplementation.

#### 2.1 Kernel Backend Abstraction

- Define `KernelBackend` trait in `ferrum-runtime`
- Candle backend as default/fallback
- CUDA FFI backend as production path
- Clean separation: scheduling logic never touches raw kernel code

#### 2.2 FlashAttention / FlashInfer Integration

- FFI bindings to FlashAttention-2 or FlashInfer
- Support for variable-length sequences in batched attention
- PagedAttention kernel integration (FlashInfer's paged API)

#### 2.3 Quantization Kernels

- FP8 / INT8 quantized matmul via CUTLASS or Marlin kernels
- Weight-only quantization (W8A16, W4A16) as first target
- GPTQ / AWQ weight format loading

### Phase 3: Production Features (P1)

#### 3.1 Benchmark Framework ✅

- ~~Throughput benchmark: requests/sec, tokens/sec at various concurrency levels~~
- ~~Latency benchmark: TTFT (Time To First Token), TPOT (Time Per Output Token), P50/P95/P99~~
- ~~Criterion micro-benchmarks: CPU attention prefill/decode at various seq lengths~~
- ~~Latency profiling tests: sequential and concurrent with percentile reporting~~
- Comparison harness against vLLM on same hardware/model (requires CUDA)
- Automated regression detection in CI (future)

#### 3.2 Observability ✅

- ~~Prometheus metrics: requests, prefills, decodes, preemptions, prefix cache hits, latencies~~
- ~~`/metrics` endpoint with Prometheus text format export~~
- ~~`/health` endpoint with engine status, active/queued requests, throughput~~
- ~~`metrics` crate instrumentation across engine critical path~~
- ~~`init_prometheus_recorder()` for server startup~~
- Grafana dashboard templates (future)
- Distributed tracing with OpenTelemetry (future)

#### 3.3 Tensor Parallelism

- NCCL FFI bindings for multi-GPU all-reduce
- Pipeline parallelism as alternative for memory-constrained setups
- Automatic parallelism strategy selection based on model size and available GPUs

#### 3.4 Structured Output ✅

- ~~JSON mode with logits biasing via `JsonModeProcessor` state machine~~
- ~~`ResponseFormat` enum (`Text`, `JsonObject`, `JsonSchema`) in `SamplingParams`~~
- ~~Integration with engine sampling path (prefill + decode)~~
- ~~OpenAI API `response_format` parameter support~~
- Grammar-guided generation with full tokenizer integration (future)
- JSON Schema constraint enforcement (future)

### Phase 4: Competitive Differentiation (P2)

#### 4.1 Disaggregated Prefill/Decode

- Prefill and decode on separate GPU pools or nodes
- Prefill-optimized instances (high compute, lower memory)
- Decode-optimized instances (lower compute, high memory for KV cache)

#### 4.2 Speculative Decoding

- Draft model + verification pattern
- Type stubs already exist in `ferrum-types` (`speculation` structures)
- Target: 2-3x decode throughput for applicable models

#### 4.3 Multi-LoRA Serving

- Dynamic LoRA adapter loading/unloading
- Shared base model weights across adapters
- Per-request adapter selection

## Priority Summary

| Priority | Item | Rationale |
|---|---|---|
| **P0** | PagedAttention + Continuous Batching | Core value of the project |
| **P0** | CUDA kernel FFI layer | Without this, no competitive performance |
| **P1** | Benchmark framework | Need data to prove Rust scheduling advantage |
| **P1** | FlashAttention/FlashInfer integration | Foundation for attention performance |
| **P1** | FP8/INT8 quantization | Standard cost reduction in production |
| **P1** | Tensor Parallelism | Required for large models |
| **P2** | Disaggregated prefill/decode | Advanced scaling architecture |
| **P2** | Speculative decoding | Throughput multiplier |
| **P2** | Structured output | Product feature, not core engine |
| **Deprioritized** | More model architectures | Get Llama to excellence first, then expand |
| **Deprioritized** | Metal support | Production inference is CUDA; Metal is nice-to-have |
| **Deprioritized** | CLI pull/list polish | That's Ollama's job; production serving doesn't need it |

## Non-Goals

- Competing with Ollama on the edge/desktop inference use case
- Training or fine-tuning
- Reimplementing CUDA kernels from scratch
- Supporting every model architecture early on
