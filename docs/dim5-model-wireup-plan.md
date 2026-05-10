# Dim 5 Model Wire-Up Plan — PR B + PR C

> Created 2026-05-10 after PR #141 (WeightFormat enum + LlmExecutorFactory rename) closed Dim 3. The remaining gap to "INT8 KV cache fully usable end-to-end" is split into two PRs to keep risk bounded.

## Where we are now

The kernel + type + cache abstraction layer for INT8 KV is complete:

- **Kernels** (PR #131): `paged_decode_attention_int8` + `int8_kv_cache_append` validated cosine 0.99999 vs FP32 host reference.
- **Type system** (PR #134): `BackendKvDtype<K>` carries `type KvBuffer` + `type KvScales`. `KvFp16` resolves to `Self::Buffer` / `()` (zero cost). `KvInt8` on CUDA resolves to `OptionalCudaInt8` / `OptionalCudaScalesF16`.
- **Cache abstraction** (PR #135): `KvCacheQuant<B, K>` parallel struct + `KvCacheQuant<CudaBackend, KvInt8>::new_paged_cuda()` constructor + e2e parity test driving append→decode through the cache (cos 0.99999).
- **CLI/serve config** (PR #139): `KvCacheDtype` enum + `--kv-dtype` flag + `FERRUM_KV_DTYPE` env. Currently rejects Int8/Fp8 at the CLI layer with a helpful message pointing here.

The single remaining gap: `LlamaFamilyModel<B>` and `Qwen3MoeModel<B>` are hardcoded to `KvCache<B>` (= `KvCache<B, KvFp16>`). To use INT8 KV for inference, the model decode loop must branch on `K`.

## The 3 acceptance gates apply

The user established three gates that every refactor PR must clear:

1. **CLI + server use the new architecture** — `ferrum run / serve / bench` go through the new dispatch.
2. **Previously perf-tested models still infer correctly** — Qwen3 (0.6B/1.7B/4B/8B), Qwen2.5-Instruct (FP16 + GPTQ-Int4), Qwen3-30B-A3B MoE, plus modality models (Whisper / BERT / CLIP / TTS).
3. **Performance does not regress** — ≤ 3% drift vs each phase's baseline.

PR B is risky because it changes model struct signatures across 4 files. PR C is riskier because it changes the per-layer forward call sites. Both must clear all 3 gates.

## PR B — model `K` parameter, FP16-only (1 day, medium risk)

**Goal**: introduce the `K: KvDtypeKind` type parameter through every model type and the factory cascade. K only takes `KvFp16` after this PR — zero behavior change. The point is to land the type-system shape so PR C only has to add INT8-specific code.

### File-by-file changes

1. `crates/ferrum-models/src/models/llama_family.rs`
   - `pub struct LlamaFamilyModel<B: MoeLlmBackend>` → `pub struct LlamaFamilyModel<B: MoeLlmBackend, K: KvDtypeKind = KvFp16>`
   - `pub kv_caches: HashMap<String, Vec<KvCache<B>>>` → `Vec<KvCache<B, K>>`
   - `kv_free_pool: Vec<Vec<KvCache<B>>>` → `Vec<Vec<KvCache<B, K>>>`
   - `impl<B> LlamaFamilyModel<B>` → `impl<B, K> LlamaFamilyModel<B, K>`
   - 4 KvCache literal construction sites (lines 884 + 898): no field change needed (FP16 KvScales = (), FP16 KvBuffer = B::Buffer), the existing `_kv_dtype: PhantomData` already has the K slot.
   - `impl<B> DecoderOnlyLLM for LlamaFamilyModel<B>` → `impl<B, K> DecoderOnlyLLM for LlamaFamilyModel<B, K>`

2. `crates/ferrum-models/src/models/qwen3_moe.rs`
   - Same pattern: add `K` parameter, `Vec<KvCache<B, K>>`, generic the impls.

3. `crates/ferrum-models/src/models/llama_family_forward_batched.rs`
   - The `forward_batched` impl block already uses `&mut LlamaFamilyModel<B>` — change to `&mut LlamaFamilyModel<B, K>`.
   - Body unchanged: it only reads / writes `KvCache<B, K>` fields that are `B::Buffer`-shaped for `K = KvFp16`.

4. `crates/ferrum-engine/src/registry.rs::LlmExecutorFactory::create()`
   - The two `match &config.device { CPU => Box::new(LlamaFamilyModel::<CpuBackend>::new(...)) ... }` cascades become:
     ```rust
     match (&config.device, config.engine_config.kv_cache.dtype) {
         (Device::CPU,     KvCacheDtype::Fp16) => build_llm::<CpuBackend, KvFp16>(...).await,
         (Device::Metal,   KvCacheDtype::Fp16) => build_llm::<MetalBackend, KvFp16>(...).await,
         (Device::CUDA(_), KvCacheDtype::Fp16) => build_llm::<CudaBackend, KvFp16>(...).await,
         (dev, dt) => Err(unsupported(format!("({dev:?}, {dt:?}) not implemented"))),
     }
     ```
     `build_llm<B, K>` is a generic helper that wraps the existing arch + loader cascade, taking `<B, K>` as type parameters.

### Acceptance for PR B

- `cargo check --workspace --all-targets` passes (CPU + Metal + CUDA)
- `cargo test --workspace --features metal --lib` ≥ 335 passing
- Metal Qwen3-0.6B, Q4_K_M GGUF benches: ≤ 3% drift vs current baselines
- CUDA Qwen3-0.6B FP16 = 82 tok/s, Llama-8B INT4 = 62 tok/s, 30B-A3B vLLM c=4 = 55 tok/s (post-#141 baselines): ≤ 3% drift
- 1 model running each on Metal + CUDA + CPU still produces correct output (smoke check via `ferrum run --prompt "Hello"`)

### Bisect strategy if perf regresses

1. Check binary md5 — if `cargo build` produces identical bytes to PR #141's build, drift is thermal noise.
2. Otherwise, suspect monomorphization explosion: K=KvFp16 should generate identical code to today, but rustc may inline differently. Compare `cargo asm` output for `LlamaFamilyModel::forward` between #141 and PR B.

## PR C — INT8 KV model integration (1-2 days, medium-high risk)

**Goal**: make `LlamaFamilyModel::<CudaBackend, KvInt8>` actually work end-to-end. After this PR, `ferrum bench qwen3:0.6b --kv-dtype int8` runs.

### File-by-file changes

1. `crates/ferrum-models/src/models/llama_family.rs`
   - **The model needs to hold a different cache type per K**. Rust without specialization can't dispatch per-K at struct level, so introduce an enum:
     ```rust
     enum LayerKvCache<B: MoeLlmBackend + BackendKvDtype<K>, K: KvDtypeKind> {
         Fp16(KvCache<B, KvFp16>),                  // when K = KvFp16
         Int8(KvCacheQuant<B, KvInt8>),             // when K = KvInt8
     }
     ```
     Or: make the model's `kv_caches: Vec<KvCache<B, K>>` for K=KvFp16 and `Vec<KvCacheQuant<B, KvInt8>>` for K=KvInt8 — doable via associated types but requires per-K specialization or an `InternalCache<B, K>` newtype.

   - Decision: **enum-based dispatch** (simplest, avoids specialization). The cost is a small runtime match per layer, but the K value is monomorphized so LLVM can fold the enum to a single variant at compile time per K.

2. **Per-layer forward edits** (4-6 call sites in `llama_family.rs::forward_layer*`):
   - Wherever the code does `let cache = &mut self.kv_caches[seq_id][layer]; B::split_qkv_norm_rope_into_paged_cache_varlen(..., &mut cache.k, &mut cache.v, ...)`, branch on K:
     ```rust
     match cache {
         LayerKvCache::Fp16(kv) => {
             B::split_qkv_norm_rope_into_paged_cache_varlen(ctx, ..., &mut kv.k, &mut kv.v, ...);
             B::paged_decode_attention(ctx, q, &kv.k, &kv.v, ...);
         }
         LayerKvCache::Int8(kv) => {
             ferrum_kernels::int8_kv::launch_int8_kv_cache_append(
                 ctx_handle, ..., kv.k.buffer_mut(), kv.v.buffer_mut(),
                 kv.k_scales.buffer_mut(), kv.v_scales.buffer_mut(), ...,
             )?;
             ferrum_kernels::int8_kv::launch_int8_paged_decode_attention(
                 ctx_handle, q, kv.k.buffer(), kv.v.buffer(),
                 kv.k_scales.buffer(), kv.v_scales.buffer(), ...,
             )?;
         }
     }
     ```
   - The K=KvInt8 path goes via the `int8_kv` module's launchers (already validated).

3. `Qwen3MoeModel` — same pattern, but only the dense self-attention layer needs INT8 KV. The MoE FFN is unchanged.

4. `crates/ferrum-engine/src/registry.rs::LlmExecutorFactory::create()`
   - Add the K=KvInt8 arm:
     ```rust
     (Device::CUDA(_), KvCacheDtype::Int8) => build_llm::<CudaBackend, KvInt8>(...).await,
     ```

5. `crates/ferrum-cli/src/commands/run.rs::apply_kv_dtype_override`
   - Drop the Int8 reject branch (let the call through). Keep Fp8 reject because PR D hasn't shipped.

### Acceptance for PR C

- All PR B acceptance criteria still pass (FP16 paths unchanged)
- `ferrum bench qwen3:0.6b --kv-dtype int8` runs on CUDA without erroring
- INT8 vs FP16 parity: same prompt produces tokens with ≤ 1% perplexity delta on a fixed eval set (use `wikitext` first 8 docs; sequential greedy decode)
- VRAM saving measurable: `nvidia-smi` shows ≥ 30% reduction in KV cache footprint at long-context settings (4K tokens, BS=4)
- TTFT / TPOT within 5% of FP16 (INT8 dequant on the read path adds ~5-10% per-token overhead per vLLM benchmarks)

### Risk mitigations

- **Smoke before bench**: `ferrum bench qwen3:0.6b --kv-dtype int8 --rounds 1 --max-tokens 32` first, just to confirm the path runs without crashing.
- **Bisect via env var**: keep an internal `FERRUM_INT8_KV_FALLBACK_FP16=1` that forces the K=KvInt8 branch to call the FP16 launchers — if INT8-vs-FP16 parity diverges, this isolates whether the issue is in the launcher dispatch or the kernel data.
- **Single model first**: ship PR C only for Qwen3 (LlamaFamilyModel). MoE models (Qwen3MoeModel) get a separate follow-up PR — MoE adds expert dispatch complexity orthogonal to KV dtype.

### What can't be done in PR C

- **Metal INT8 KV** — requires Metal kernels that don't exist yet (only CUDA kernels are validated).
- **Mixed-precision intra-model** — every layer of a given model must use the same K. Switching K per-layer is out of scope.

## PR D — FP8 KV (2-3 days, deferred)

After PR C lands, FP8 follows the same pattern:

1. `paged_decode_attention_fp8` + `fp8_kv_cache_append` CUDA kernels (use `__nv_fp8_e4m3` + per-token f8 scale). SM ≥ 8.9 only.
2. `BackendKvDtype<KvFp8>` impl on CudaBackend with `KvBuffer = OptionalCudaFp8`, `KvScales = OptionalCudaScalesF8`.
3. `KvCacheQuant<CudaBackend, KvFp8>::new_paged_cuda()` constructor.
4. K=KvFp8 model branch in LlamaFamilyModel forward.
5. `(Device::CUDA(_), KvCacheDtype::Fp8) => build_llm::<CudaBackend, KvFp8>(...)` factory arm.
6. Drop the Fp8 reject branch in `apply_kv_dtype_override`.
7. Parity bench: FP8 vs FP16, FP8 vs INT8.

## Roll-out timeline

```
this session (done):
  PR A — WeightFormat enum + factory rename (#141)

next session 1:
  PR B — model K parameter, FP16-only (1 day)
  Validates: structural change is safe (zero behavior change)

next session 2:
  PR C — INT8 KV model integration (1-2 days)
  Validates: INT8 path runs, parity holds, VRAM savings measurable

session 3+ (deferred):
  PR D — FP8 KV (2-3 days)
  Optional, only when FP8 deployment hardware (SM ≥ 8.9) is available
```

## What completion means

Dim 5 status moves from `🟡 structural done, model wire-up pending` to `✅ done` when **all four** of:

1. PR B + PR C land
2. CUDA Qwen3 / Llama bench passes with `--kv-dtype int8` end-to-end
3. INT8 perplexity ≤ 1% delta vs FP16 on wikitext-8doc eval
4. The `apply_kv_dtype_override` Int8 reject branch is gone

PR D (FP8) is a nice-to-have, not part of the Dim 5 closure criterion (FP8 needs newer hardware than the validated Dim 5 INT8 CUDA path).
