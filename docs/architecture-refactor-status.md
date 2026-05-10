# Architecture Refactor — 5-Dim Complete-Form Goal & Status

> Started 2026-05-09. Tracks the audit-driven refactor toward a clean
> 5-dimension architecture where new backends / models / quant formats /
> KV precisions can be added independently.

## The 5-dim goal

Inference engines compose 5 orthogonal axes. Naive multiplication is
~7 200 cells (8×5×15×4×3); the right design has each axis as a single
polymorphism point so total maintenance units are ~44 (additive, not
multiplicative).

| # | Dimension | Polymorphism point | Examples |
|---|---|---|---|
| 1 | **Model architecture** | Per-family Rust struct (`LlamaFamilyModel<B>`, `Qwen3MoeModel<B>`, future `DeepseekV3Model<B>`) | Llama / Qwen2 / Qwen3 / Qwen3-MoE / Mistral SW / DeepSeek-V3 MLA / Gemma |
| 2 | **Compute precision** | `Linear<B>` impl | FP32 / FP16 / BF16 / FP8 (E4M3) / INT8 / INT4 |
| 3 | **Weight format** | `WeightLoader<B>` produces format-specific `Linear<B>` impl | safetensors / GPTQ / AWQ / GGUF (Q2K-Q8) / Marlin-packed / HQQ / EXL2 |
| 4 | **Inference device** | `Backend` trait + capability supertraits + bundle aliases | CUDA / Metal / CPU / AMD / Intel / 国产 GPU |
| 5 | **KV cache precision** | `KvDtypeKind` marker + `BackendKvDtype<K>` supertrait | FP16 / BF16 / INT8 / FP8 |

The combinatorial complexity stays additive because each axis only adds
its own units, never multiplies with the others.

### Acceptance for "完全体" (complete form)

Three gates — ALL must pass before the refactor is considered done:

1. **CLI + server use the new architecture** — not just trait切片 in
   ferrum-kernels; CLI commands and HTTP handlers reference the new
   modality-specific traits / Linear impls.
2. **All previously perf-tested models infer correctly** — Qwen3
   (0.6B / 1.7B / 4B / 8B), Qwen2.5-Instruct (FP16 + GPTQ-Int4),
   Qwen3-30B-A3B MoE (Marlin INT4 on CUDA, Q4_K_M GGUF on Metal),
   Qwen3-TTS-0.6B/1.7B, whisper-large-v3-turbo, BERT, CLIP.
3. **Performance does not regress** — Metal smoke for each phase
   relative to per-phase baseline ≤ 3% drift (LTO usually folds trait
   reorganization to identical machine code; thermals are the noise
   floor).

## Progress (as of 2026-05-10, completed dims 1-4 + crate consolidation)

### Dim status

| Dim | Status | Notes |
|---|---|---|
| 1. Model architecture | ✅ done | Architecture v2 (historical, pre-audit) |
| 2. Compute precision | ✅ done | Phase 3e cutover complete (#123 Marlin, #124 GGUF). All quant kernel dispatch lives in `Linear<B>` impls in `ferrum-kernels::quant_linear/`. `BackendQuantMarlin::gemm_gptq*` and `BackendQuantGguf::gemm_quant` deleted. |
| 3. Weight format | ✅ done | `Linear<B>` polymorphism point owns the kernel call; new format = new `Linear<B>` impl in ferrum-kernels (no Backend trait change). |
| 4. **Inference device** | ✅ done | Backend → 5 supertraits + 3 capability bundles (#107-#110) |
| 5. KV cache precision | 🟢 functional done | Model wire-up landed PR #143 (PR B: K type param) + PR #144 (PR C: INT8 model integration). `LlamaFamilyModel<B, K: KvLayer<B>>` with `K::Layer` associated type — `KvCache<B, KvFp16>` for FP16, `KvCacheQuant<B, KvInt8>` for INT8. Trait dispatch (no enum / no `as_fp16()` panic accessors). DecoderOnlyLLM split per K. Bench (RTX 4090, post-#144): Qwen3-0.6B FP16 83.7 tok/s (vs PR B 82.4: +1.6%), Qwen3-30B-A3B GPTQ-Int4 c=4 52.9 tok/s (vs PR B 54.1: -2.2%) — both within 3% drift. INT8 functional: Qwen3-0.6B 43.6 tok/s, TPOT 22.89ms (49% of FP16 — perf parity within 5% deferred to PR D, gap is in `int8_paged_decode_attention.cu` per-token dequant on the read path). FP8 KV is a separate future PR D item. |

**5 of 5 dims functional.** Dim 5 status went from "🟡 structural done" to "🟢 functional done" with PR #143 + #144. INT8 path runs end-to-end (`ferrum bench qwen3:0.6b --kv-dtype int8` works). Remaining items under PR D: INT8 TPOT within 5% of FP16, INT8 batched/unified, INT8 perplexity validation, FP8 KV.

### PRs landed (in order)

| PR | Phase | Effect |
|---|---|---|
| [#106](https://github.com/sizzlecar/ferrum-infer-rs/pull/106) | 1 | Delete 377 lines of dead code (KernelExecutor / DecodeBackend / orphan EngineFactory) |
| [#107](https://github.com/sizzlecar/ferrum-infer-rs/pull/107) | 2.1 + 2.2 | Extract `BackendGraph` + `BackendCollective` supertraits |
| [#108](https://github.com/sizzlecar/ferrum-infer-rs/pull/108) | 2.3 | Extract `BackendQuantMarlin` + `BackendQuantGguf` supertraits |
| [#109](https://github.com/sizzlecar/ferrum-infer-rs/pull/109) | 2.4 | Extract `BackendPagedKv` + `BackendMoeFused` supertraits |
| [#110](https://github.com/sizzlecar/ferrum-infer-rs/pull/110) | 2.5 | `LlmBackend` + `QuantLlmBackend` + `MoeLlmBackend` capability bundles |
| [#111](https://github.com/sizzlecar/ferrum-infer-rs/pull/111) | 3a + 3b | Split `qwen3_moe.rs` (3579 → 2950) + `llama_family.rs` (4305 → 2652); -2282 lines combined in two largest files |
| [#112](https://github.com/sizzlecar/ferrum-infer-rs/pull/112) | 4 | KV-dtype scaffolding (`KvDtypeKind` + `BackendKvDtype<K>` traits, Metal +1.0%) |
| [#113](https://github.com/sizzlecar/ferrum-infer-rs/pull/113) | 3c | `ExpertStack::gemv_*` methods replace 4 `B::gemv_quant_moe_id*` call sites in stacked-decode path (Metal +2.5%) |
| [#114](https://github.com/sizzlecar/ferrum-infer-rs/pull/114) | 5a step 1 | Extract `modality_stubs` helper for the 3 fake engines (-39 lines copy-paste, +59 shared) |
| [#116](https://github.com/sizzlecar/ferrum-infer-rs/pull/116) | 3d | `ExpertStack::gemm_*` + `gemv_*_batched/offset` wrappers replace remaining 16 `B::*_moe_id*` call sites in `moe/forward.rs` + `qwen3_moe.rs` decode path. **Vast 4090: Llama-8B INT4 +0.3%, Qwen3-30B-A3B vLLM Marlin c=4 +3.1% (55.9 vs 54.2 tok/s)**. |
| [#117](https://github.com/sizzlecar/ferrum-infer-rs/pull/117) | 5a step 2 | Split `InferenceEngine` mega-trait into lifecycle base + 4 modality supertraits (`LlmInferenceEngine`/`EmbedEngine`/`TranscribeEngine`/`TtsEngine`). `AppState` 4 fields; handlers 501 when missing. The 3 fake engines now impl only the modality trait their executor serves. |
| [#118](https://github.com/sizzlecar/ferrum-infer-rs/pull/118) | 5b/1 | Delete `DefaultInferenceEngine` (1217 lines net). Builder always returns `ContinuousBatchEngine`. Unblocks the `ferrum-runtime` crate delete (5b/2). |
| [#119](https://github.com/sizzlecar/ferrum-infer-rs/pull/119) | 4 wire-up | `KvCache<B, K = KvFp16>` type-param + PhantomData. **Vast 4090: Qwen3-0.6B 82.9 tok/s (baseline 83.0, perf-neutral ✅)**. |
| [#121](https://github.com/sizzlecar/ferrum-infer-rs/pull/121) | 5b/2 | Delete the `ferrum-runtime` crate; move `CandleBackend` + memory pool into `ferrum-engine/src/backends/`. Drops the legacy CPU `ComputeBackend` impl (no consumers). |
| [#122](https://github.com/sizzlecar/ferrum-infer-rs/pull/122) | 3e/1 | Add concrete `CudaMarlinLinear` + `CudaMarlinStackedExpertLinear` in `ferrum-kernels::quant_linear/cuda_marlin.rs`. Make `marlin_gemm_with_perm` + `launch_vllm_marlin` public so the new types can dispatch the kernel without a trait method. Additive — no behaviour change. |
| [#123](https://github.com/sizzlecar/ferrum-infer-rs/pull/123) | 3e/2 | **Marlin cutover.** `BackendQuantMarlin::load_gptq` returns `Box<dyn Linear<Self>>`; `make_stacked_expert_linear` factory replaces `gemm_gptq_with_offset`. Both `gemm_gptq*` trait methods deleted. `GptqLinear<B>` / `StackedExpertLinear<B>` shrink to delegating wrappers. CPU side gains `load_gptq_stacked` + `make_stacked_expert_linear` for parity tests. |
| [#124](https://github.com/sizzlecar/ferrum-infer-rs/pull/124) | 3e/3 | **GGUF cutover (closes Dim 2/3).** `BackendQuantGguf::load_quant` / `load_quant_fused` return `Box<dyn Linear<Self>>`. `gemm_quant` trait method deleted; the 209-line Metal body extracted into `pub fn metal_gemm_quant_dispatch`. New `MetalGgufLinear` + `CpuGgufLinear` in `ferrum-kernels::quant_linear/`. `QuantLinear<B>` shrinks to wrapper. |
| [#126](https://github.com/sizzlecar/ferrum-infer-rs/pull/126) | Dim 5 cleanup | Move `KvDtypeKind` + the four marker structs out of ferrum-kernels into ferrum-interfaces (no GPU deps). |
| [#127](https://github.com/sizzlecar/ferrum-infer-rs/pull/127) | PR B | Delete the legacy `ComputeBackend` / `KernelOps` / `model_builder` / memory modules from ferrum-interfaces (had no implementations after Phase 3e). Drop the dead `MetalKernelOps` shell, the `ferrum-engine::backends::candle` `ComputeBackend` impl, the dead `custom_backend` builder field, and `ferrum-models::{builder,weights}`. |
| [#128](https://github.com/sizzlecar/ferrum-infer-rs/pull/128) | PR C | Merge `ferrum-attention` into `ferrum-kernels::attention`. The crate's only consumers were `ferrum-kernels::backend::metal` and `ferrum-models` (Qwen3-TTS); merging removes a duplicated metal feature pass-through and the cudarc-0.12 stub. Two-cudarc-version workspace gone. |
| [#129](https://github.com/sizzlecar/ferrum-infer-rs/pull/129) | cleanup | Drop dead `LinearFactory` trait + `DefaultLinearFactory` (zero callers anywhere). |
| [#131](https://github.com/sizzlecar/ferrum-infer-rs/pull/131) | Dim 5 INT8 KV step 1 | CUDA INT8 KV kernels: `paged_decode_attention_int8` (read INT8 + per-token FP16 scale, dequantize on the fly) and `int8_kv_cache_append` (FP16 → per-(token, kv_head) symmetric INT8 + scale). Rust launchers in `ferrum-kernels::int8_kv`. `BackendKvDtype<KvInt8>` marker on `CudaBackend`. Host-reference parity test: cosine 0.99999 vs FP32 ref. |
| [#132](https://github.com/sizzlecar/ferrum-infer-rs/pull/132) | Dim 5 INT8 KV test | End-to-end append→decode composition test. FP16 K/V → `int8_kv_cache_append` → `paged_decode_attention_int8` → compare against FP32 host reference. Cosine 0.99999, validates the kernel pair composes cleanly with no storage / scale convention drift. |
| [#134](https://github.com/sizzlecar/ferrum-infer-rs/pull/134) | Dim 5 type system | `BackendKvDtype<K>` gains `type KvBuffer` + `type KvScales` associated types; for K=KvFp16 they resolve to `Self::Buffer` / `()` (zero-cost), for K=KvInt8 on CudaBackend they resolve to `OptionalCudaInt8` / `OptionalCudaScalesF16`. New parallel `KvCacheQuant<B, K>` struct uses the associated types — leaves the FP16 `KvCache<B, K>` field shape unchanged (no callsite churn). |
| [#135](https://github.com/sizzlecar/ferrum-infer-rs/pull/135) | Dim 5 INT8 cache abstraction | `KvCacheQuant<CudaBackend, KvInt8>::new_paged_cuda(...)` one-call constructor: K/V int8 pool + FP16 scales + block_table + context_lens. New e2e parity test `kv_cache_quant_int8_e2e` drives append→decode through the cache abstraction (no direct cudarc allocs) — cos 0.99999. |
| [#137](https://github.com/sizzlecar/ferrum-infer-rs/pull/137) | post-audit polish | Rename `ferrum-engine/src/backends/` → `tensor_factory/`. After PR #127 the directory contained only `CandleTensorFactory`; the old name mis-suggested engine-level backend dispatch lives there (real dispatch is in `ferrum-kernels::backend`). |
| [#138](https://github.com/sizzlecar/ferrum-infer-rs/pull/138) | post-audit polish | Drop dead `EngineBuilder::backend_name` field + `with_backend()` (zero callers after the resolve_backend_name delete). Reword stale "Phase D/E" comments in `registry.rs` to reflect the current state — TP>1 is the only remaining unsupported feature, framed as a feature port not a "Phase" stage. |
| [#139](https://github.com/sizzlecar/ferrum-infer-rs/pull/139) | Dim 5 CLI scaffolding | `ferrum_types::config::KvCacheDtype` enum (Fp16/Bf16/Int8/Fp8) + `KvCacheConfig.dtype` field. `--kv-dtype DTYPE` flag on `run` / `serve` / `bench` + `FERRUM_KV_DTYPE` env. `apply_kv_dtype_override` helper rejects Int8 / Fp8 with a helpful message until model wire-up ships. Hand-tested on Metal: fp16 runs, int8 errors cleanly, bad value rejects with options listed. |
| [#141](https://github.com/sizzlecar/ferrum-infer-rs/pull/141) | Dim 3 polymorphism point + factory rename (PR A) | Extract `ferrum_models::weight_format::WeightFormat::detect()` enum (Safetensors/Gguf), replacing the `is_gguf_path` short-circuit. Future formats (AWQ/EXL2/HQQ) plug in by adding a variant + `WeightLoader<B>` impl. Rename `CandleExecutorFactory` → `LlmExecutorFactory` (the LLM hot path uses `Backend<B>`, not candle). Registry key `"candle"` → `"llm"`. `pub type CandleExecutorFactory = LlmExecutorFactory` kept as `#[deprecated]` back-compat. |
| [#143](https://github.com/sizzlecar/ferrum-infer-rs/pull/143) | Dim 5 PR B | `LlamaFamilyModel<B, K: KvDtypeKind = KvFp16>` + `Qwen3MoeModel<B, K>` add the K type parameter. Two separate `match config.device` cascades in `LlmExecutorFactory::create()` collapse into one `match (device, kv_dtype) → build_llm::<B, K>(...)` cascade. K = KvFp16 default keeps existing call sites unchanged. **CUDA Qwen3-0.6B 82.4 tok/s, 30B-A3B GPTQ-Int4 c=4 54.1 tok/s — within 3% drift.** |
| [#144](https://github.com/sizzlecar/ferrum-infer-rs/pull/144) | Dim 5 PR C + KvLayer pivot + CLI cleanup | (1) INT8 KV functional on CUDA via `KvLayer<B>: KvDtypeKind` trait dispatch (replaces `LayerKvCache::Fp16/Int8` enum + `as_fp16()` panic accessors). `K::Layer` is associated type; `LlamaFamilyModel<CpuBackend, KvInt8>` is now a compile error (CPU has no `BackendInt8KvOps`). (2) `DecoderOnlyLLM` impl split per K (`KvFp16` keeps batched/unified overrides; `KvInt8` minimal — falls back to per-item via trait default). (3) INT8 perf round 1: drop per-token D2H block_table → use `paged_block_indices` host mirror (+8.5% TPOT). (4) `cli::source_resolver` module: `looks_like_gguf_path` / `detect_format` / `find_cached_model` / `resolve_model_source` centralised. `serve.rs` drops its forked cache-walk; `run.rs` / `bench.rs` delegate. **CUDA Qwen3-0.6B FP16 83.7 (+1.6%), 30B-A3B GPTQ-Int4 c=4 52.9 (-2.2%), INT8 functional 43.6 tok/s.** |

### Backend trait shrinkage

| Phase | Method count | Δ |
|---|---|---|
| Pre-Phase-2 | 94 | — |
| 2.1 + 2.2 (#107) | ~82 | -12 |
| 2.3 (#108) | ~62 | -20 |
| 2.4 (#109) | ~32 | -30 |
| 3e/2 (#123) | ~30 | -2 (gemm_gptq + gemm_gptq_with_offset; +make_stacked_expert_linear) |
| 3e/3 (#124) | ~29 | -1 (gemm_quant; load_quant return type changed) |

### CLI / serve startup → 5-dim implementation matching

`ferrum run / serve / bench <model>` boots through
`LlmExecutorFactory::create()` (see `ferrum-engine/src/registry.rs`).
That function performs the 5-dim selection:

| Dim | Match site | Status |
|---|---|---|
| 1. Model arch | `match model_def.architecture` (Llama / Qwen2 / Qwen3 / Qwen3Moe / Mistral) → `LlamaFamilyModel<B>` or `Qwen3MoeModel<B>` | ✅ wired |
| 2. Compute precision | implicit via `Linear<B>` polymorphism. The loader's `load_linear()` returns the right impl (Dense f16 vs Marlin int4 vs GGUF q4_k) per tensor metadata. | ✅ wired |
| 3. Weight format | `WeightFormat::detect(&model_path)` → `Safetensors {..}` / `Gguf {..}` (PR #141). Future formats (AWQ/EXL2/HQQ) = new enum variant + `WeightLoader<B>` impl, no factory edit. | ✅ wired |
| 4. Inference device | `match config.device` → CPU/CUDA/Metal → static `<B>` type parameter | ✅ wired |
| 5. KV cache precision | `engine_config.kv_cache.dtype` plumbed (PR #139); model wire-up via `K: KvLayer<B>` (PR #143 + #144). Factory cascade `match (device, kv_dtype) → build_llm::<B, K>(...)` covers `(CPU/Metal/CUDA, Fp16)` + `(CUDA, Int8)`. INT8 functional; INT8 TPOT-vs-FP16 within 5% deferred to PR D (attention kernel rewrite needed). | 🟢 functional |

### Remaining work (feature, not architecture)

See `docs/dim5-model-wireup-plan.md` for the detailed PR B + PR C
plan that closes Dim 5 model integration.

| Phase | Scope | Risk | Estimate |
|---|---|---|---|
| **PR B** — model `K` parameter, FP16-only | `LlamaFamilyModel<B, K = KvFp16>` / `Qwen3MoeModel<B, K = KvFp16>` add a `K: KvDtypeKind` type parameter (defaulting to KvFp16). All KvCache<B> sites become KvCache<B, K>. Flatten `LlmExecutorFactory` cascade to `(device, kv_dtype) → build_llm::<B, K>(...)`. Zero behavior change — K only takes KvFp16. | Medium (touches model struct sigs across 4 files; FP16 pathway unchanged) | 1 day. |
| **PR C** — INT8 KV model integration | K=KvInt8 branch on the model: hold `KvCacheQuant<B, KvInt8>` instead of `KvCache<B>`, route per-layer append + paged-decode through `int8_kv` launchers. `LlmExecutorFactory` reads `engine_config.kv_cache.dtype = Int8` → instantiates `LlamaFamilyModel::<CudaBackend, KvInt8>`. Drop the Int8 reject branch in `apply_kv_dtype_override`. INT8 vs FP16 parity bench (≤ 1% accuracy delta + measurable VRAM saving). | Medium-High (real per-layer forward edit + parity bench) | 1-2 days. |
| **PR D** — FP8 KV (deferred) | Mirror PR #131 + #134 + #135 + PR C for FP8: `paged_decode_attention_fp8` + `fp8_kv_cache_append` (use `__nv_fp8_e4m3` storage + per-token f8 scale). `BackendKvDtype<KvFp8>` marker. K=KvFp8 model branch. | Medium (FP8 needs SM ≥ 8.9; otherwise mechanical mirror of INT8) | 2-3 days. |

## GPU testing workflow (Vast.ai 4090)

CUDA-only validation runs on Vast.ai rented GPUs. Mac local handles
Metal + CPU; Vast pod handles CUDA. Cost ~$0.30/hr for an RTX 4090.

### Spin up a pod

Use the Vast HTTP API (the `vastai` CLI has a urllib3 incompatibility on
macOS LibreSSL):

```bash
source .env.local  # exports VAST_API_KEY + HF_TOKEN

# Find cheapest 4090 (filter rentable + verified preferred)
curl -s "https://console.vast.ai/api/v0/bundles/?q=%7B%22gpu_name%22%3A%7B%22eq%22%3A%22RTX%204090%22%7D%2C%22rentable%22%3A%7B%22eq%22%3Atrue%7D%2C%22dph_total%22%3A%7B%22lt%22%3A0.40%7D%7D&order=%5B%5B%22dph_total%22%2C%22asc%22%5D%5D" \
  -H "Authorization: Bearer $VAST_API_KEY" | jq '.offers[0:3] | .[] | {id, dph_total, cuda_max_good, geolocation}'

# Create instance — image must be nvidia/cuda:*-devel-ubuntu22.04
# (NOT pytorch/pytorch:* — that pulls 5-10 GB of unused ML deps)
PUBKEY=$(cat ~/.ssh/id_ed25519.pub)
curl -X PUT "https://console.vast.ai/api/v0/asks/<OFFER_ID>/" \
  -H "Authorization: Bearer $VAST_API_KEY" \
  -H "Content-Type: application/json" \
  -d "$(python3 -c "import json; print(json.dumps({
    'client_id': 'me',
    'image': 'nvidia/cuda:12.4.0-devel-ubuntu22.04',
    'env': {'-p 22:22': '1'},
    'disk': 60,
    'runtype': 'ssh ssh_proxy',
    'onstart': 'apt-get update -qq && apt-get install -y -qq openssh-server pkg-config libssl-dev git build-essential && mkdir -p /run/sshd /root/.ssh && echo \"$PUBKEY\" > /root/.ssh/authorized_keys && chmod 600 /root/.ssh/authorized_keys && /usr/sbin/sshd -D &',
    'label': 'ferrum-cuda-validation',
}))")"
```

Notes:
- ~30% of new pods hit a Vast SSH proxy port-forwarding bug — destroy
  + retry if SSH refuses for >2 min after `actual_status=running`.
- Pod returns `ssh_host` + `ssh_port` (e.g. `ssh1.vast.ai:19862`).

### Bootstrap on the pod

```bash
ssh -p <PORT> root@<HOST>

# One-time: install Rust + clone ferrum
curl -sSf https://sh.rustup.rs | sh -s -- -y
source $HOME/.cargo/env
mkdir -p /workspace && cd /workspace
git clone https://github.com/sizzlecar/ferrum-infer-rs.git
cd ferrum-infer-rs

# HF token for gated models (Qwen3-30B-A3B-GPTQ-Int4 etc).
# Pipe via stdin (NOT --token CLI arg — that exposes in `ps`):
printf '%s\n' "$HF_TOKEN" | ssh -p <PORT> root@<HOST> \
  'mkdir -p /root/.cache/huggingface && cat > /root/.cache/huggingface/token && chmod 600 /root/.cache/huggingface/token'
```

### Build + bench

```bash
# Build (LTO + codegen-units=1 → ~5 min cold, ~3 min incremental)
cargo build --release --features cuda -p ferrum-cli

# For vLLM Marlin MoE path, add the feature:
cargo build --release --features cuda,vllm-moe-marlin -p ferrum-cli

# Standard smokes
HF_HOME=/workspace/.hf_home ./target/release/ferrum bench Qwen/Qwen3-0.6B \
    --rounds 3 --max-tokens 128

# Marlin INT4 on Llama-8B (no auth needed, hugging-quants mirror)
HF_HOME=/workspace/.hf_home ./target/release/ferrum bench \
    hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4 \
    --rounds 3 --max-tokens 128

# MoE Marlin (gated; needs HF login above)
HF_HOME=/workspace/.hf_home FERRUM_VLLM_MOE=1 ./target/release/ferrum bench \
    Qwen/Qwen3-30B-A3B-GPTQ-Int4 --rounds 3 --max-tokens 64 --concurrency 4
```

### Background workflow (don't block on cargo build)

```bash
nohup bash -c "
    cargo build --release --features cuda -p ferrum-cli 2>&1 | tail -3
    HF_HOME=/workspace/.hf_home ./target/release/ferrum bench Qwen/Qwen3-0.6B --rounds 2 --max-tokens 64 2>&1 | tail -10
    echo __DONE__
" > /tmp/build_bench.log 2>&1 &
```

Then check periodically with `ssh ... 'tail /tmp/build_bench.log'`.

### Pod cost / cleanup

- Idle pod: ~$0.30/hr; destroy when not actively benching.
- Destroy: `curl -X DELETE "https://console.vast.ai/api/v0/instances/<INSTANCE_ID>/" -H "Authorization: Bearer $VAST_API_KEY"`
- Or via web console.

### Reference baselines (Vast 4090, post-Phase-3a/b main = `fff0759`)

| Model | Path | tok/s e2e | TPOT |
|---|---|---|---|
| Qwen3-0.6B | FP16 (safetensors) | 83.0 | 11.88 ms |
| Llama-3.1-8B-Instruct | GPTQ-Int4 (Marlin) | 60.4 | 16.55 ms |
| Qwen3-30B-A3B | GPTQ-Int4 (vLLM Marlin moe path, c=4) | 54.2 | 69.95 ms |
| Qwen3-30B-A3B | GPTQ-Int4 (IST-DASLab Marlin path, c=4) | 57.6 | 66.65 ms |

Note: c=4 isn't in the historical bench tables (`docs/bench/cuda-rtx4090-2026-05-09-pr-101-102/` only logged c=1/8/16/32). Treat
these as new baselines.

### Post-consolidation rerun (Vast 4090, main @ `44a87d0` after PRs #126-#129)

Acceptance gate 3 — same model, same comparison method, ≤ 3% drift.

| Model | Path | tok/s e2e (now) | Δ vs baseline | TPOT | Δ TPOT |
|---|---|---|---|---|---|
| Qwen3-0.6B | FP16 (safetensors) | 81.8 | -1.4% (within 3%) ✅ | 12.16 ms | +2.4% |
| Llama-3.1-8B-Instruct | GPTQ-Int4 (Marlin) | 63.0 | **+4.3% improvement** ✅ | 15.86 ms | -4.2% |
| Qwen3-30B-A3B | GPTQ-Int4 (vLLM Marlin moe, c=4) | 54.9 | +1.3% ✅ | 69.34 ms | -0.9% |

Marlin INT4 improvement comes from the Phase 3e/2 cutover — LTO inlines
`Box<dyn Linear<B>>` dispatch better than the previous
`#[cfg]`-branched trait method body. Metal locally: Qwen3-0.6B Q4_K_M
60.3 tok/s, matches pre-consolidation reading.

**Acceptance gate 3 passes** through PRs #126-#129.

### Post-INT8-KV rerun (Vast 4090, main @ `1d3f3af` after PR #131)

Verifies that adding the INT8 KV kernels did not regress the existing
FP16 paths (PR #131 is purely additive — new files + one marker impl).

| Model | Path | tok/s e2e (now) | Δ vs post-cleanup | TPOT |
|---|---|---|---|---|
| Qwen3-0.6B | FP16 | 81.3 | -0.6% ✅ | 12.30 ms |
| Llama-3.1-8B-Instruct | GPTQ-Int4 (Marlin) | 62.1 | -1.4% ✅ | 16.09 ms |
| Qwen3-30B-A3B | GPTQ-Int4 (vLLM Marlin moe, c=4) | 54.4 | -0.9% ✅ | 70.08 ms |

All within ±3% drift (thermal / scheduler noise floor). Functionality
+ performance preserved through Dim 5 step 1.

## References

- Audit memory: `~/.claude/projects/-Users-chejinxuan-rust-ws-ferrum-infer-rs/memory/project_arch_audit_2026_05_09.md`
- Acceptance memory: `~/.claude/projects/-Users-chejinxuan-rust-ws-ferrum-infer-rs/memory/project_arch_refactor_done_criteria.md`
- Vast workflow memory: `~/.claude/projects/-Users-chejinxuan-rust-ws-ferrum-infer-rs/memory/feedback_arch_refactor_validation.md`
