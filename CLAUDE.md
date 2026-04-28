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

# Whisper ASR transcription
cargo run -p ferrum-cli --bin ferrum --features metal -- transcribe whisper-turbo audio.wav -l zh
cargo run -p ferrum-cli --bin ferrum --features metal -- serve whisper-turbo

# Benchmarks
cargo run -p ferrum-cli --bin ferrum -- bench qwen3:4b                          # sequential baseline
cargo run -p ferrum-cli --bin ferrum -- bench qwen3:4b --concurrency 4          # batch decode
cargo run -p ferrum-cli --bin ferrum -- bench qwen3:4b --max-tokens 1024        # long decode
cargo run -p ferrum-cli --bin ferrum -- bench qwen3:4b --long-context           # long prompt (~2k tokens)

# CUDA with batch decode + paged KV
FERRUM_MAX_BATCH=8 cargo run -p ferrum-cli --features cuda -- bench qwen3:4b --concurrency 4
FERRUM_PAGED_KV=1 FERRUM_KV_BLOCKS=128 cargo run -p ferrum-cli --features cuda -- bench qwen3:4b --concurrency 4

# Triton-rs PTX kernels (optional, decode-path swap-in)
cargo run -p ferrum-cli --features cuda,triton-kernels -- run qwen2.5:3b-int4
FERRUM_TRITON_INT4=1 cargo run -p ferrum-cli --features cuda,triton-kernels -- run qwen2.5:3b-int4
```

## Architecture

**Architecture v2 (Model-as-Code) — done.** The model layer is explicit Rust generic over a `Backend<B>` trait, not a config-driven runner. `LlamaFamilyModel<B>` (in `ferrum-models`) covers Qwen3 / Qwen2.x / Llama / Mistral; per-family quirks (Qwen3 QK-norm, Mistral sliding window) are toggles on the config struct. All hardware behavior goes through the `Backend` trait — adding a backend = implementing the trait, not editing models.

**Dependency layers (bottom-up):**

1. **Foundation (no GPU deps):** `ferrum-types` (shared types, errors), `ferrum-interfaces` (trait contracts: ComputeBackend, ModelExecutor, Scheduler, KvCacheManager, Sampler, Tokenizer)
2. **Core logic (hardware-agnostic):** `ferrum-scheduler` (continuous batching, priority), `ferrum-sampler` (top-k/p, temperature, JSON mode), `ferrum-tokenizer` (HF wrapper), `ferrum-kv` (paged KV cache, block allocation), `ferrum-runtime` (backend registry)
3. **Compute (feature-gated):** `ferrum-kernels` (unified `Backend<B>` impls for CPU / CUDA / Metal — owns custom CUDA kernels, Triton PTX, Marlin, paged KV, NCCL, TP decode), `ferrum-attention` (fused-transformer prototype — Metal/CPU shipping, CUDA module is a stub kept around for future use), `ferrum-quantization` (`Linear` trait, GPTQ loader, native safetensors)
4. **Application:** `ferrum-engine` (orchestration, `ContinuousBatchEngine`), `ferrum-models` (`LlamaFamilyModel<B>`, BERT, Whisper, Qwen3-TTS), `ferrum-server` (Axum HTTP, OpenAI-compatible API), `ferrum-cli` (binary entry point)
5. **Testing:** `ferrum-testkit` (mocks for all trait contracts — enables GPU-free testing)

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
Model forward  (LlamaFamilyModel<B>::forward_layer, ferrum-models)
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

Candle handles weight loading and prefill (FlashAttention-2). Decode is fully controlled by `CudaDecodeRunner` in `ferrum-kernels`:

**Custom CUDA kernels** (PTX compiled at build time via `ferrum-kernels/build.rs`):
- `rms_norm.cu`, `fused_add_rms_norm.cu` — layer normalization
- `rope.cu` — rotary position embedding (Q+K fused)
- `fused_silu_mul.cu` — MLP activation (+ interleaved variant for batch)
- `decode_attention.cu` — single-block warp-cooperative attention
- `flash_decode_attention.cu` — split-K flash decoding for long contexts
- `paged_decode_attention.cu` — block-table indirect attention (+ split-K variant)
- `residual_add.cu` — element-wise residual
- `marlin.cu` — INT4×FP16 GPTQ GEMM (Blackwell-tuned, used for decode quant path)

**Decode optimizations:**
- Double-buffered residual + cross-layer norm fusion (108 fewer kernel launches)
- Piecewise CUDA Graphs (`L+1` graphs covering pre_attn / post_attn{i}+pre_attn{i+1} / post_attn+norm+lm_head); attention itself stays eager
- Flash Decoding: split KV across blocks for GPU SM utilization (auto at kv_len > 256)
- Batch decode: batched cuBLAS GEMM (m=batch) with per-item attention loop
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

Custom Whisper forward pass — candle loads weights only, inference is ours (Metal/CUDA/CPU).

**Architecture:**
- Custom LayerNorm, Softmax, Linear (bypasses candle-nn CustomOp which lacks Metal support)
- rustfft-based STFT for mel spectrogram (matches Python whisper to float32 precision)
- Mel filterbank extracted from Python whisper (identical to torch version)
- Self-attention KV cache with positional embedding offset tracking
- Cross-attention KV cache (compute once per segment)

**Decode pipeline (matches Python whisper.transcribe):**
- Timestamp-based sequential decode (not no_timestamps mode)
- Three logit filters: SuppressBlank, SuppressTokens (82 non-speech + special), ApplyTimestampRules
- Temperature fallback: 0.0 → 0.2 → 0.4 → 0.6 → 0.8 → 1.0
- Compression ratio check (real zlib via flate2) + avg logprob threshold
- No-speech detection and segment skipping
- Seek-based segmentation from timestamp tokens
- Repetition detection (consecutive token limit)

**CLI:**
```bash
# Transcribe audio file (WAV/M4A/MP3 — auto ffmpeg conversion)
cargo run -p ferrum-cli --bin ferrum --features metal -- transcribe whisper-turbo audio.m4a -l zh

# HTTP server with /v1/audio/transcriptions endpoint
cargo run -p ferrum-cli --bin ferrum --features metal -- serve whisper-turbo
curl -X POST http://localhost:8000/v1/audio/transcriptions -F "file=@audio.wav" -F "language=zh"
```

**Performance (5-min Chinese audio, whisper-large-v3-turbo):**

| Backend | Time | vs Python |
|---------|------|-----------|
| Rust Metal (release) | ~72s | 1.5x faster |
| Python CPU (torch) | 107s | baseline |
| Python MPS | N/A (PyTorch bug) | — |

**Known limitation:** Metal float32 matmul accumulation order differs from CPU, causing minor character-level differences (e.g., "核销" vs "和销") after 4 encoder transformer layers. Not fixable at software level — hardware floating-point behavior. On CUDA, this is controllable via cuBLAS compute type.

## Config

Runtime defaults in `ferrum.toml`. Model cache at `~/.cache/huggingface` (shared with HF Python).
