# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What is this

Ferrum Infer is a Rust-native LLM inference engine. Single binary, no Python — supports Metal (macOS), CUDA (NVIDIA), CPU backends. Targets vLLM-level performance with PagedAttention, continuous batching, custom CUDA kernels.

## Current baselines (RTX 4090, ShareGPT apples-to-apples vs vLLM 0.20.2)

| model | c | ferrum | vLLM | ratio |
|---|---:|---:|---:|---:|
| M1 (Llama-3.1-8B FP16) | 32 | see `docs/bench/cuda-rtx4090-2026-05-07/` | | |
| M2 (Llama-3.1-8B-INT4) | 32 | **881** | 2179 | 40% |
| M3 (Qwen3-30B-A3B-Int4) | 32 | **1023** | 1867 | 55% |

M2 currently primary win target via dense Marlin tile tuning + continuous batching. M3 primary via Phase 3 engine-init scratch budget + chunked prefill (see `docs/continuous-batching-redesign.md`). 80%-of-vLLM is the stretch target.

**Latest progress report**: `docs/progress/2026-05-13-continuous-batching-foundation.md` — current state, what shipped, what's next.

## Build & dev commands

```bash
cargo check --workspace --all-targets   # Mac, no GPU features
cargo test --workspace
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -A warnings

# CLI surface
cargo run -p ferrum-cli -- run qwen3:0.6b                                          # interactive chat
cargo run -p ferrum-cli -- serve --model qwen3:0.6b --port 8000                    # OpenAI API
cargo run -p ferrum-cli -- bench qwen3:4b --concurrency 4 --max-tokens 128         # internal bench
cargo run -p ferrum-cli -- bench-serve --base-url http://127.0.0.1:8000 ...        # HTTP bench, vLLM-parity

# Feature gates
cargo run -p ferrum-cli --features metal -- ...
cargo run -p ferrum-cli --features cuda,vllm-moe-marlin -- ...
cargo run -p ferrum-cli --features cuda,triton-kernels -- ...   # FERRUM_TRITON_INT4=1 to enable
```

Internal env vars (`FERRUM_KV_MAX_BLOCKS` / `FERRUM_PAGED_MAX_SEQS` / `FERRUM_VLLM_MOE` etc.) are set by `gpu_mem_autosize` for `serve` and chat-profile defaults for `run`. Users should not set by hand — autosizer's job.

## Bench reproducibility

Full methodology + env blocks live in `bench/v0.2-cuda/` and `docs/bench/`:
- `bench/v0.2-cuda/apples_all_drive.sh` — single driver, M1/M2/M3 × c=1/4/16/32 sweep (use `SKIP_VLLM=1` to skip vLLM rerun)
- `bench/v0.2-cuda/PERF_TRACKER.md` — running tracker (R0-R2 + Wave 1 + Phase 2B retrospectives)
- `docs/bench/macos-2026-05-02/` — Metal apples-to-apples vs llama.cpp + mistralrs (M1 Max 32GB)

Profile probes (opt-in via env): `FERRUM_NEXT_BATCH_PROF`, `FERRUM_SCHED_NONE_PROF`, `FERRUM_BATCH_PREFILL_PROF`, `FERRUM_RBD_PROF`, `FERRUM_BATCH_DECODE_PROF`, `FERRUM_DECODE_OP_PROFILE`.

## Architecture

**Model-as-Code v2**: per-family Rust struct generic over `Backend<B>` + `K: KvDtypeKind`. `LlamaFamilyModel<B, K>` covers Llama / Qwen2.x / Qwen3 / Mistral (per-family quirks are config toggles). `Qwen3MoeModel<B, K>` covers Qwen3-MoE / 30B-A3B. Adding a backend = implementing the relevant supertraits, not editing models.

**5 polymorphism dimensions**: model architecture / compute precision (`Linear<B>`) / weight format (`WeightLoader<B>` + `MarlinExpertStack<B>`) / device (`Backend` + supertraits) / KV precision (`KvDtypeKind` marker).

**13 crates, bottom-up**:
- Foundation: `ferrum-types`, `ferrum-interfaces`
- Core logic (no GPU): `ferrum-scheduler`, `ferrum-sampler`, `ferrum-tokenizer`, `ferrum-kv`
- Compute (feature-gated): `ferrum-kernels`, `ferrum-quantization`
- Application: `ferrum-engine` (only LLM engine = `ContinuousBatchEngine`), `ferrum-models`, `ferrum-server`, `ferrum-cli`
- Testing: `ferrum-testkit`

**Inference flow** (decode, OpenAI chat):
```
HTTP → ContinuousBatchEngine.run_iteration (under iteration_lock)
     → ContinuousBatchScheduler.next_batch (mixed prefill+decode batch)
     → process_batch_unified → model_executor.unified_decode
     → per-layer: rms_norm → qkv_proj → varlen attn → o_proj → MLP/MoE → residual
     → Sampler → mpsc StreamChunk → SSE
```

The unified path lives in `crates/ferrum-engine/src/continuous_engine.rs::process_batch_unified` (Llama already supports `unified_forward`; Qwen3MoE implementation lands dormant pending Phase 3 scratch budgeting).

**Iteration lock**: `EngineInner::iteration_lock` serializes batches. Mixed prefill+decode in one batch recovers intra-iteration parallelism.

## Code style

- Rust 2021, `rustfmt.toml`: 4-space indent, max width 100
- `snake_case` fns/modules, `CamelCase` types/traits, `SCREAMING_SNAKE_CASE` constants
- Conventional commits: `feat(scope):`, `fix(scope):`, `refactor(scope):`
- `cargo check --workspace` and `cargo test --workspace` must pass on Mac without GPU features
- CUDA code behind `#[cfg(feature = "cuda")]`, Metal behind target OS gates, Triton kernels behind `#[cfg(feature = "triton-kernels")]` (implies cuda)
- Shared types/traits go in `ferrum-types`/`ferrum-interfaces`, never duplicated

## Done criteria — engine / scheduler / sampler / CLI run changes

Any PR that touches `ferrum-engine/src/continuous_engine.rs`,
`ferrum-engine/src/sampler/`, `ferrum-scheduler/`, or
`ferrum-cli/src/commands/run.rs` is in the d67fbbb blast radius
(EOS / stop_sequences / stream / multi-turn KV / chat template). It is
NOT done until the chat smoke suites pass locally:

```bash
ferrum pull qwen3:0.6b           # + tinyllama, qwen2.5:0.5b for full matrix
cargo test --release -p ferrum-cli --features metal --test chat_smoke -- --ignored --test-threads=1
cargo test --release -p ferrum-cli --features metal --test chat_pty   -- --ignored --test-threads=1
cargo test --release -p ferrum-cli --features metal --test chat_stress -- --ignored --test-threads=1
```

Total ~70 s on M1 Metal (`--release` is required — debug is ~5× slower).
Nightly CI (`.github/workflows/chat-smoke.yml`) runs the same suites on
macos-latest. PR-time CI (`.github/workflows/ci.yml`) only compiles them
via `cargo check --all-targets` — actual runs are nightly because each
test cold-loads a real model.

## CUDA backend layout

`crates/ferrum-kernels/src/backend/cuda/` split by supertrait:

| file | trait |
|---|---|
| `mod.rs` | core `Backend` + `CudaState` + `KvFp16` |
| `collective.rs` | `BackendCollective` (TP all-reduce) |
| `graph.rs` | `BackendGraph` + multi-slot graph cache |
| `int8_kv.rs` | `BackendInt8KvOps` + KvCacheQuant ctor |
| `quant.rs` | `BackendQuantMarlin` + `BackendQuantGguf` + vllm marlin moe |
| `paged.rs` | `BackendPagedKv` + SplitKScratch + paged dispatchers |
| `moe.rs` | `BackendMoeFused` (route_topk_softmax, moe_align_block_size, moe_combine) |

Custom CUDA kernels (PTX compiled at build time via `ferrum-kernels/build.rs`): `rms_norm.cu`, `fused_add_rms_norm.cu`, `rope.cu`, `fused_silu_mul.cu`, `decode_attention.cu` / `flash_decode_attention.cu` / `paged_decode_attention.cu` (+ split-K + INT8 variants), `residual_add.cu`, `marlin.cu`, `vllm_marlin_moe/ops.cu` (vendored, under `vllm-moe-marlin` feature).

Triton-rs PTX path (`triton-kernels` feature): `crates/ferrum-kernels/triton_ptx/` (offline-compiled, embedded via `include_str!`, loaded by `cudarc::get_or_load_custom_func`). Triton w4a16 INT4 `FERRUM_TRITON_INT4=1` has a known prefill-NaN bug under ContinuousBatch scheduler — Marlin remains default. See `crates/ferrum-kernels/src/backend/cuda/triton_*.rs` for wired kernels.

## Model support

**LLM** (via `LlamaFamilyModel<B>`): Qwen3 (0.6B–4B), Qwen2.5-Instruct (0.5B–7B), Llama-3.x-Instruct (1B–8B), TinyLlama-1.1B-Chat, Mistral (sliding-window).

**MoE** (via `Qwen3MoeModel<B>`): Qwen3-MoE / Qwen3-30B-A3B.

**ASR**: Whisper (tiny → large-v3-turbo). Files: `crates/ferrum-models/src/multimodal/whisper.rs` + `whisper_decoder.rs`. Recommended: `whisper-turbo`. Quirk: Metal float32 matmul accumulation differs from CPU after ~4 encoder layers (~minor char-level drift; hardware-level, not fixable in software).

**TTS**: Qwen3-TTS-0.6B / 1.7B. **Embeddings**: CLIP, Chinese-CLIP, SigLIP, BERT.

**Quantization**: GPTQ INT4 auto-detected at load. CUDA defaults to Marlin (repacks at load); Triton w4a16 (`FERRUM_TRITON_INT4=1`) runs on-disk GPTQ tile layout directly. CPU path uses dequant+cuBLAS fallback.

Models cached at `~/.cache/huggingface` (shared with HF Python).

## Config

Runtime defaults: `EngineConfig::default()` in `ferrum-types::config`. Build script: `ferrum-kernels/build.rs` compiles CUDA .cu → PTX via `bindgen_cuda`, requires `CUDA_HOME`. Metal shaders embed at compile time via `include_str!`.
