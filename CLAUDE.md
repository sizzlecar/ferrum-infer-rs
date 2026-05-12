# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What is this?

Ferrum Infer is a Rust-native LLM inference engine. Single binary, no Python — supports Metal (macOS), CUDA (NVIDIA), and CPU backends. Targets vLLM-level performance with PagedAttention, continuous batching, and custom CUDA kernels.

**Current baseline (Qwen3-30B-A3B-GPTQ-Int4, RTX 4090, random-dataset `ferrum bench-serve`, post Audit #8/#9/#5 reorg + Phase 3 MoE fixes @ commit `0022412`):**

| c | tok/s | TPOT | note |
|---|------:|-----:|------|
| 1  | 146.6 | 6.52ms  | random-len 256/128, num_prompts=c×4 |
| 8  | 541.3 | 13.10ms |  |
| 16 | 725.1 | 19.28ms |  |
| 32 | **811.8** | **34.46ms** | active perf target — see "Performance Testing" below |

`bash bench/v0.2-cuda/m3_bench_serve.sh` for repro (release build needs `--features cuda,vllm-moe-marlin`). Random-dataset bench is fast (~10 min) but not apples-to-apples with the published vLLM number (`bench/v0.2-cuda/run_cell.sh` is ShareGPT). For the gap-closing comparison vs vLLM, use the run_cell.sh path described below.

- INT4 quantization: GPTQ format auto-detected, Marlin fused kernel on Blackwell
- Paged KV attention with block reclamation
- Flash Decoding (split-K) for long contexts

## Performance Testing

### CUDA (RTX 4090) — primary perf target

**Reference model**: `Qwen/Qwen3-30B-A3B-GPTQ-Int4` (M3 tag, ~17 GB MoE GPTQ-Int4). M1 / M2 (Llama-3.1-8B FP16 / GPTQ-Int4) also covered when sweeping the v0.2 matrix; `bench/v0.2-cuda/models.txt` is the canonical model list.

**Methodology** (vendored from PR #102, ShareGPT dataset — this is the apples-to-apples comparison with the published vLLM number):

1. `bash bench/v0.2-cuda/setup.sh` — one-time pod setup (Rust + ferrum release build with `--features cuda,vllm-moe-marlin` + `pip install vllm` + ShareGPT subset).
2. `bash bench/v0.2-cuda/run_sweep.sh` — the 144-cell matrix (vllm + ferrum × M1/M2/M3 × c=1/4/16/32 × 3 repeats). Calls `run_cell.sh` per cell.
3. Each cell uses `vllm bench serve --backend openai-chat --base-url http://...` against the running server. Engine-agnostic measurement tool: same dataset, same client, same metrics across vLLM and ferrum.
4. Dataset: `bench/v0.2-cuda/prompts.jsonl` — ShareGPT V3 subset, 128 prompts, deterministic seed = ferrum repo HEAD short hash (different commits → different prompt subsets, but vLLM and ferrum at the same commit see identical prompts).
5. Workloads: c=1 → 4 prompts (128-tok input, 512-tok output); c≥4 → c×4 prompts (512-tok input, 256-tok output).

**Ferrum server env block** (post Phase 1.5/2/3 fix commits):

```
FERRUM_VLLM_MOE=1
FERRUM_KV_CAPACITY=2048
FERRUM_KV_MAX_BLOCKS=4096
FERRUM_PAGED_MAX_SEQS=32
FERRUM_MOE_BUCKETED=1
FERRUM_MARLIN_SKIP_WS_ZERO=1
FERRUM_MOE_STREAMS=4
# FERRUM_MOE_GRAPH=1 + FERRUM_GRAPH_SKIP_UPLOAD=1 at c ≤ 16 only;
# graph regresses c=32 by ~6% (cuGraphLaunch overhead on 480-node MoE
# graph dominates at high m). See memory project_moe_phase3_graph_bug.
```

**vLLM server args** (matches `bench/v0.2-cuda/run_sweep.sh::vllm_start`):

```
vllm serve <model> --port 8800 --max-num-seqs 64 --max-model-len 4096 \
  --no-enable-prefix-caching --no-enable-log-requests --quantization gptq_marlin
```

**Result files**: `bench/v0.2-cuda/results/{engine}__{model_tag}__c{C}__r{R}.json` (vLLM 0.20 schema: `output_throughput`, `mean_tpot_ms`, `p99_tpot_ms`, `mean_ttft_ms`, `completed`, `failed`).

**Recorded vLLM 0.20 baseline on M3 c=32**: ~1900 tok/s, mean TPOT 15.65 ms (see `results/vllm__M3__c32__r1.json`). This is the gap target.

### Metal (Apple Silicon) — comparison vs llama.cpp + mistralrs

**Reference models** (all `Q4_K_M` GGUF — same files read by all three engines):
- `bartowski/Meta-Llama-3.1-8B-Instruct-GGUF` (4.6 GB)
- `unsloth/Qwen3-8B-GGUF` (4.7 GB)
- `Qwen/Qwen3-30B-A3B-GGUF` MoE (17.3 GB)

**Hardware**: MacBook Pro 16" 2021 (M1 Max, 32 GB unified, macOS 15+). 30B-A3B + paged-KV pool at c=16 lands ~21 GB resident — fits the 32 GB pool.

**Comparison engines**: `llama.cpp` (homebrew, e.g. b8960) + `mistralrs` 0.8.1.

**Methodology** — full reproducible suite at `docs/bench/macos-2026-05-02/`:

1. `bash docs/bench/macos-2026-05-02/run_suite.sh` — 36-cell base suite (3 engines × 3 models × c = 1/4/8/16).
2. `bash docs/bench/macos-2026-05-02/rerun_c16.sh` — clean-state c=16 rerun with 15s cooldown + pkill between cells (controls for run-to-run variance + 36-cell swap pressure).
3. `bash docs/bench/macos-2026-05-02/rerun_moe_batched.sh` — Qwen3-30B-A3B with `FERRUM_MOE_BATCHED=1 FERRUM_MOE_BATCHED_DECODE=1 FERRUM_MOE_BATCH_THRESHOLD=2` (opt-in MoE batched-decode path — required for c ≥ 8 MoE numbers).
4. Bench harness: `bench/scripts/bench_serving.py` — OpenAI `/v1/chat/completions` SSE, temperature 0, `max_tokens=64`, total prompts 8/16/24/32 for c=1/4/8/16, one prewarm chat completion (`max_tokens=4`) per cell.

**Ferrum server env block** (dense path):

```
FERRUM_METAL_PAGED_KV=1
FERRUM_PAGED_MAX_SEQS=$((c * 2))
FERRUM_KV_CAPACITY=512
FERRUM_MAX_BATCH=$c
```

For MoE at c ≥ 8 add `FERRUM_MOE_BATCHED=1 FERRUM_MOE_BATCHED_DECODE=1 FERRUM_MOE_BATCH_THRESHOLD=2`. (Crossover is c ≈ 8 — below that, per-token decode is faster on M1 Max.)

**Headline c = 16 throughput (tok/s, M1 Max, best-of-3 runs, recorded 2026-05-02 PR #81)**:

| Model | ferrum | llama.cpp (b8960) | mistralrs (0.8.1) | ferrum vs llama.cpp |
|---|---:|---:|---:|---:|
| LLaMA-3.1-8B          | **96.7** | 67.2 | 23.3   | **+44%** |
| Qwen3-8B              | **93.2** | 68.6 | 23.5   | **+36%** |
| Qwen3-30B-A3B (MoE)   | 79.2     | 83.4 | panic¹ | −5% (matched) |

¹ mistralrs 0.8.1 `PoisonError`-panics on Qwen3-30B-A3B-Q4_K_M (`mistralrs-core add_request.rs:466`).

c = 1/4/8 grid and TPOT distributions are in [`docs/bench/macos-2026-05-02/README.md`](docs/bench/macos-2026-05-02/README.md).

Smaller Metal models (Qwen3-0.6B / 1.7B / 4B FP16) are used for fast post-refactor smoke (`cargo run --release -p ferrum-cli --features metal -- bench qwen3:0.6b --concurrency 1 --max-tokens 16`) but the headline Metal numbers above are the perf gate.

### Perf targets

| horizon | c=32 vs vLLM 0.20.x on M3 4090 | current |
|---------|--------------------------------|--------:|
| Near-term (1-2 mo) | **≥ 50%** (≥ ~950 tok/s) | 811.8 / 43% (random-dataset) |
| Mid-term (3-6 mo)  | **≥ 65%** (≥ ~1240 tok/s) | — |
| Stretch            | **≥ 80%** (≥ ~1520 tok/s) | — |

c=1 / 8 / 16 are already at vLLM parity or ahead under the random-dataset bench; the c=32 cliff is the **only** active perf target. Path-to-target inputs:

- vLLM internals reference: memory `project_vllm_decode_design.md` (FULL CUDA Graph 3× win, BatchDescriptor cached by m_padded, Marlin tile fixed, PagedAttn V2 grid)
- MoE graph c=32 regression analysis: memory `project_moe_phase3_graph_bug.md`
- Continuous-batch scheduler quality / prefix cache / spec decode — not yet implemented in ferrum; each is a multi-week project

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
3. Weight format — `WeightLoader<B>`, `MarlinExpertStack<B>` (PR #166-#173 closed dim 3 for MoE Marlin — `Backend::type GptqStore` is gone, `load_gptq_stacked` returns `Arc<dyn MarlinExpertStack<B>>` so new Marlin backends only impl the trait); GGUF goes through `StackedExpertGgufLinear<B>`. AWQ / EXL2 still pluggable.
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

- **`ferrum-kernels/build.rs`**: Compiles CUDA .cu files to PTX using `bindgen_cuda`. Requires CUDA_HOME env var. Generates `ptx.rs` in OUT_DIR. Metal shaders embed at compile time via `include_str!` inside the `.rs` files alongside the `.metal` sources (no separate build script).

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
