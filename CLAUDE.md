# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What is this?

Ferrum Infer is a Rust-native LLM inference engine. Single binary, no Python — supports Metal (macOS), CUDA (NVIDIA), and CPU backends. Targets vLLM-level performance with PagedAttention, continuous batching, and custom CUDA kernels.

**Current baseline (Qwen3-30B-A3B-GPTQ-Int4, RTX 4090, `ferrum bench-serve`):**

| c | tok/s | TPOT | ratio vs vLLM 0.20.1 |
|---|------:|-----:|---------------------:|
| 1  | 96.2  | 10.04ms | 60% |
| 8  | 241.5 | 31.13ms | 58% |
| 16 | 272.1 | 55.52ms | 54% |
| 32 | **318.4** | 94.75ms | **17%** |

`bash bench/v0.2-cuda/m3_bench_serve.sh` for repro. The c=32 cliff vs vLLM is the active perf target.

- INT4 quantization: GPTQ format auto-detected, Marlin fused kernel on Blackwell
- Paged KV attention with block reclamation
- Flash Decoding (split-K) for long contexts

## Build & Development Commands

```bash
cargo check --workspace --all-targets   # Fast compile validation (Mac, no GPU features)
cargo test --workspace                  # Full test suite
cargo fmt --all -- --check              # CI-enforced format check
cargo clippy --workspace --all-targets -- -A warnings  # CI-advisory lint

# Default CLI surface (the only entry points users see)
cargo run -p ferrum-cli --bin ferrum -- run qwen3:0.6b           # interactive chat
cargo run -p ferrum-cli --bin ferrum -- serve --model qwen3:0.6b --port 8000
cargo run -p ferrum-cli --bin ferrum -- bench qwen3:4b --concurrency 4 --max-tokens 128
cargo run -p ferrum-cli --bin ferrum -- bench-serve --base-url http://127.0.0.1:8000 \
    --model /path/to/model --tokenizer /path/to/model --max-concurrency 32   # HTTP bench, vllm-parity

# Feature gates
cargo run -p ferrum-cli --features metal -- ...                  # macOS Metal
cargo run -p ferrum-cli --features cuda,vllm-moe-marlin -- ...   # CUDA + vLLM marlin moe
cargo run -p ferrum-cli --features cuda,triton-kernels -- ...    # CUDA + Triton PTX
# Triton w4a16 INT4: FERRUM_TRITON_INT4=1 (default off; Marlin is faster)
```

Internal env vars (`FERRUM_KV_MAX_BLOCKS` / `FERRUM_PAGED_MAX_SEQS` / `FERRUM_VLLM_MOE` etc.) are set by `gpu_mem_autosize` for `serve` and chat-profile defaults for `run`. Users should not set them by hand — that's the autosizer's job.

## Architecture

**Architecture v2 (Model-as-Code) — done.** The model layer is explicit Rust generic over a `Backend<B>` trait, not a config-driven runner. `LlamaFamilyModel<B, K>` (in `ferrum-models`) covers Qwen3 / Qwen2.x / Llama / Mistral; `Qwen3MoeModel<B, K>` covers Qwen3-MoE / 30B-A3B; per-family quirks (Qwen3 QK-norm, Mistral sliding window) are toggles on the config struct. K is the KV-precision marker (`KvFp16` default; `KvInt8` functional, FP8 ahead). All hardware behavior goes through the `Backend` supertrait stack — adding a backend = implementing the relevant supertraits, not editing models.

**5 polymorphism dimensions** (each is one independent axis, not multiplicative):
1. Model architecture — per-family Rust struct
2. Compute precision — `Linear<B>` impl
3. Weight format — `WeightLoader<B>` (safetensors / GPTQ / GGUF; AWQ / EXL2 pluggable)
4. Inference device — `Backend` + capability supertraits (CUDA / Metal / CPU; AMD pluggable)
5. KV cache precision — `KvDtypeKind` marker + `BackendKvDtype<K>` (FP16 / INT8; FP8 pluggable)

**13 crates, dependency layers (bottom-up):**

1. **Foundation (no GPU deps):** `ferrum-types` (shared types/errors/config), `ferrum-interfaces` (trait contracts: ModelExecutor, Scheduler, KvCacheManager, Sampler, Tokenizer, KvDtypeKind markers)
2. **Core logic (hardware-agnostic):** `ferrum-scheduler` (continuous batching, priority), `ferrum-sampler` (top-k/p, temperature, JSON mode), `ferrum-tokenizer` (HF wrapper), `ferrum-kv` (paged KV cache, block allocation)
3. **Compute (feature-gated):** `ferrum-kernels` (unified `Backend<B>` impls for CPU / CUDA / Metal — owns custom CUDA kernels, Triton PTX, Marlin, paged KV, NCCL, TP decode, fused-transformer attention, Linear<B> quant impls), `ferrum-quantization` (`Linear` trait, GPTQ loader, native safetensors)
4. **Application:** `ferrum-engine` (orchestration, `ContinuousBatchEngine` — the only LLM engine impl; also owns tensor_factory + parallel device manager), `ferrum-models` (`LlamaFamilyModel<B, K>` / `Qwen3MoeModel<B, K>` / BERT / Whisper / Qwen3-TTS), `ferrum-server` (Axum HTTP, OpenAI-compatible API), `ferrum-cli` (binary entry point)
5. **Testing:** `ferrum-testkit` (mocks for all trait contracts — enables GPU-free testing)

**Deleted crates (history):** `ferrum-runtime` (folded into `ferrum-engine` PR #121), `ferrum-attention` (merged into `ferrum-kernels::attention` PR #128).

**Key design rules:**
- `cargo check --workspace` and `cargo test --workspace` must pass on Mac without GPU features
- CUDA code lives behind `#[cfg(feature = "cuda")]`, Metal behind target OS gates, Triton kernels behind `#[cfg(feature = "triton-kernels")]` (which implies `cuda`)
- Trait-based abstraction: all hardware-specific behavior goes through `Backend<B>` / interfaces
- Shared types/traits go in `ferrum-types`/`ferrum-interfaces`, never duplicated in impl crates

## Inference Flow

End-to-end LLM request lifecycle (decode-only, OpenAI chat path):

```
HTTP /v1/chat/completions  (ferrum-server::axum_server)
   ↓ apply chat template, build InferenceRequest
ContinuousBatchEngine  (ferrum-engine::continuous_engine)
   ↓ run_iteration() under tokio::Mutex iteration_lock
ContinuousBatchScheduler.next_batch(BatchHint)  (ferrum-scheduler)
   ↓ returns BatchPlan { prefill_ids, decode_ids } — mixed batch
process_batch:
  • prefill: tokenizer.encode → model.prefill (candle path, cuBLAS GEMM, flash-attn-2)
  • decode:  model.decode (custom CudaDecodeRunner + per-layer kernels)
  • per-request: build SamplingConfig, sample logits, push StreamChunk over mpsc
Model forward  (LlamaFamilyModel<B, K>::forward_layer, ferrum-models)
  per layer: rms_norm → qkv_proj → [qk_norm if Qwen3] → rope → kv_append
           → flash/decode attention → o_proj → fused_add_rms_norm
           → gate_up_proj → fused_silu_mul → down_proj → residual_add
Backend<B> dispatch  (ferrum-kernels::backend::{cuda|metal|cpu})
   ↓ kernel calls (custom .cu PTX, Triton PTX, cuBLAS, Marlin, Metal shaders)
Sampler  (ferrum-sampler) — temperature/top-k/top-p/rep-penalty/JSON-mode
Tokenizer.decode([token_id])  → SSE chunk back to client
```

**Prefill vs decode split.** Prefill is batch-shaped (high m), so it goes through candle + cuBLAS + FlashAttention-2. Decode is m=1 (or small batch), so it runs through `CudaDecodeRunner` with custom kernels and piecewise CUDA Graphs (graphs[0]=pre_attn_0, graphs[i]=post_attn_{i-1}+pre_attn_i, graphs[L]=post_attn+norm+lm_head; attention itself stays eager because kv_len varies per step).

**Iteration lock.** `EngineInner::iteration_lock` serializes batches — only one `process_batch` runs at a time. Keeps KV-cache bookkeeping and scheduler state coherent. Mixed prefill+decode in one batch recovers most of the lost intra-iteration parallelism.

**Per-request sampling state.** Each `SequenceState` owns its own RNG (seeded from `SamplingParams.seed`) and `token_frequencies` map, so repetition penalty and reproducibility are per-request, not global.

## Code Style

- Rust 2021, `rustfmt.toml`: 4-space indent, max width 100, reordered imports
- `snake_case` functions/modules, `CamelCase` types/traits, `SCREAMING_SNAKE_CASE` constants
- Conventional commits: `feat(scope):`, `fix(scope):`, `refactor(scope):`

## CUDA Decode Runner

CUDA backend lives in `crates/ferrum-kernels/src/backend/cuda/`. After PRs #148-#152 (Audit #8), the 4756-line `cuda.rs` is split into 6 supertrait-aligned files:

| File | Owns |
|------|------|
| `mod.rs` (1986 lines) | core `impl Backend for CudaBackend` + `CudaState` + global stream / decode-state slots + `KvFp16` marker impl |
| `collective.rs` | `BackendCollective` (TP all-reduce) |
| `graph.rs` | `BackendGraph` + `GraphSlotRaw` + `DECODE_GRAPHS` multi-slot cache |
| `int8_kv.rs` | `BackendInt8KvOps` + `OptionalCudaInt8` + KvCacheQuant constructor |
| `quant.rs` | `BackendQuantMarlin` + `BackendQuantGguf` + Marlin gather scratch + vLLM marlin moe |
| `paged.rs` | `BackendPagedKv` + SplitKScratch + paged dispatchers |
| `moe.rs` | `BackendMoeFused` (route_topk_softmax, moe_align_block_size, moe_combine) |

**Custom CUDA kernels** (PTX compiled at build time via `ferrum-kernels/build.rs`):
- `rms_norm.cu`, `fused_add_rms_norm.cu` — layer normalization
- `rope.cu` — rotary position embedding (Q+K fused)
- `fused_silu_mul.cu` — MLP activation (+ interleaved variant for batch)
- `decode_attention.cu` / `flash_decode_attention.cu` / `paged_decode_attention.cu` (+ split-K + INT8 variants)
- `residual_add.cu` — element-wise residual
- `marlin.cu` — INT4×FP16 GPTQ GEMM (Blackwell-tuned)
- `vllm_marlin_moe/ops.cu` — vendored vLLM marlin_moe_wna16 (under `vllm-moe-marlin` feature)

**Decode optimizations:**
- Double-buffered residual + cross-layer norm fusion (108 fewer kernel launches)
- Piecewise CUDA Graphs on `LlamaFamilyModel` decode (L+1 graphs); attention stays eager. `FERRUM_UNIFIED_GRAPH=1` opts into full-forward graph capture (~+5% measured). **`Qwen3MoeModel` has no graph capture yet** — primary perf gap to vLLM on MoE.
- Flash Decoding: split KV across blocks for GPU SM utilization (auto at kv_len > 256)
- Batch decode: batched cuBLAS GEMM (m=batch) with per-item attention loop, OR `split_qkv_norm_rope_into_paged_cache_varlen` batched-attn path (PR #102)
- Paged KV: GPU block pool with block-table indirection, free-list reclamation
- Tensor parallel decode: persistent per-rank threads + NCCL all-reduce (use only when single GPU OOMs or NVLink is present — PCIe path is bandwidth-bound)

## Triton-rs PTX Kernels (`triton-kernels` feature)

Alongside the hand-written `.cu` kernels, ferrum can dispatch a set of Triton-DSL kernels compiled to PTX **offline** by [triton-rs](https://github.com/sizzlecar/triton-rs). PTX/JSON artifacts live in `crates/ferrum-kernels/triton_ptx/` and are embedded with `include_str!`; `cudarc::get_or_load_custom_func` loads them at runtime. No Python or LLVM at runtime — the binary stays single-shot.

**Wired kernels** (gated on `--features triton-kernels` + `cuda`):
- `triton_rms_norm_f16` (also view-stride variant) — replaces custom rms_norm in decode
- `triton_fused_add_rms_norm_f16`, `triton_fused_silu_mul_f16`, `triton_residual_add_f16`
- `triton_decode_attention_f16` (HEAD_DIM 64 + 128) — alternative to flash_decode_attention
- `triton_w4a16_gptq_f16` — INT4 GEMM that runs **directly on the on-disk GPTQ tile layout** (no Marlin repack)

**Dispatch toggles:**
- Norm/silu/residual/attn swaps are auto when `triton-kernels` is built in (the launcher cfg-gates inside `Self::launch_*`).
- INT4 GEMM: `FERRUM_TRITON_INT4=1` switches `load_gptq` to `GptqStoreCuda::Triton(TritonGptqWeight)`, then `gemm_gptq` calls `launch_w4a16_gptq_triton`. Default stays Marlin.

**Regeneration:** `scripts/regen-triton-ptx.sh` (run on a Linux+CUDA box). DSL changes are rare, so manual regen is the accepted trade-off for "no runtime JIT".

**Known issue:** under `FERRUM_TRITON_INT4=1`, the ContinuousBatch scheduler's prefill round produces NaN logits ("No valid tokens for sampling"). Marlin works in the same scheduler; Triton works under the Priority scheduler. Likely a batched-matmul shape/stride edge case in the w4a16 kernel — tracked, do not enable Triton INT4 in `bench` until fixed.

**Performance landscape (RTX PRO 4000 Blackwell sm_120, Qwen2.5-3B-Instruct-GPTQ-Int4, single batch):**

| Path | tok/s | TPOT | vs FP16 |
|------|-------|------|---------|
| FP16 cuBLAS | 53.0 | 18.86 ms | 1.00× |
| INT4 Marlin | 79.3 | 12.59 ms | +49% |
| INT4 Triton (interactive) | 56.2 | ~17.8 ms | +6% |

INT4 Marlin uses ~24% of DRAM bandwidth peak — there is ~4× headroom remaining (CUDA Graphs, scheduler overhead, larger batch).

## Build Scripts

- **`ferrum-engine/build.rs`**: Compiles Metal shaders (.metal → .air → .metallib) on macOS via `xcrun`. Generates empty stub on non-Apple platforms.
- **`ferrum-kernels/build.rs`**: Compiles CUDA .cu files to PTX using `bindgen_cuda`. Requires CUDA_HOME env var. Generates `ptx.rs` in OUT_DIR.

## Model Support

**LLM:** Qwen3 (0.6B–4B), Qwen2.5-Instruct (0.5B–7B), Llama-3.x-Instruct (1B–8B), TinyLlama-1.1B-Chat, Mistral (sliding-window). All routed through `LlamaFamilyModel<B>`; per-family quirks are config toggles.

**Quantization:** GPTQ INT4 auto-detected at load. CUDA path defaults to Marlin (repacks weights on load); Triton w4a16 (`FERRUM_TRITON_INT4=1`) runs the on-disk layout directly. CPU path uses dequant+cuBLAS fallback.

**ASR (Speech-to-Text):** Whisper (tiny, base, small, medium, large-v3, large-v3-turbo). Recommended: `whisper-turbo` for best quality/speed tradeoff.

**TTS:** Qwen3-TTS-0.6B / 1.7B (streaming, voice-clone ICL). **Embeddings:** CLIP, Chinese-CLIP, SigLIP, BERT.

Models are downloaded from HuggingFace and cached locally.

## Whisper ASR

Custom Whisper forward pass (candle loads weights only, inference is ours). Files: `crates/ferrum-models/src/multimodal/whisper.rs` + decode in `whisper_decoder.rs`.

- Custom `LayerNorm`/`Softmax`/`Linear` (bypass candle-nn's missing Metal CustomOp)
- rustfft-based STFT + Python-whisper-extracted mel filterbank (float32 parity)
- Decode pipeline matches `whisper.transcribe`: timestamp-based, 3 logit filters (SuppressBlank / SuppressTokens / ApplyTimestampRules), temperature fallback 0.0→1.0 in 0.2 steps, compression-ratio + avg-logprob checks, repetition detection.
- Perf: 5-min Chinese audio whisper-large-v3-turbo on Mac Metal ~72s (vs Python CPU torch 107s; PyTorch MPS broken at the time).
- Known limit: Metal float32 matmul accumulation differs from CPU after ~4 encoder layers — minor char-level drift (e.g. "核销" vs "和销"). Hardware-level, not fixable in software.

CLI:
```bash
cargo run -p ferrum-cli --features metal -- transcribe whisper-turbo audio.m4a -l zh
cargo run -p ferrum-cli --features metal -- serve whisper-turbo
curl -X POST http://localhost:8000/v1/audio/transcriptions -F "file=@audio.wav" -F "language=zh"
```

## Config

Runtime defaults: `EngineConfig::default()` in `ferrum-types::config` (no helper wrapper; PR #153 dropped `simple_engine_config`). Model cache at `~/.cache/huggingface` (shared with HF Python).
