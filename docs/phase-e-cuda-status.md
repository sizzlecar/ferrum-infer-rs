# Phase E CUDA — Status

Runtime-validated on RTX PRO 6000 Blackwell (SM 12.0, CUDA 12.8).

## ✅ Working end-to-end

### LLM inference (LlamaFamilyModel<CudaBackend>)

| Model | Path | tok/s | Parity | Notes |
|-------|------|-------|--------|-------|
| Qwen3-0.6B FP16 | eager | ~141 | ✅ cos=0.999998 | matches CPU argmax + text |
| Qwen3-4B FP16 | eager | ~81 | ✅ cos=1.000000 | tied-emb wired; 91% of 88.8 target |
| Qwen2.5-3B-GPTQ-Int4 | Marlin | ~108.8 | ✅ text | bias-on-attn projections wired; 97% of 112.4 target |

**Smoke tests:**
- `ferrum run qwen2.5:3b-gptq --backend cuda` → coherent Chinese + English answers.
- `ferrum transcribe whisper-turbo *.wav --backend cuda` → correct transcript.
- `ferrum tts qwen3-tts "..." --backend cuda -o out.wav` → runs, produces audio
  (output quality is a pre-existing voice-clone issue, not a CUDA issue).

- **Parity test**: `crates/ferrum-models/tests/qwen3_cuda_parity_test.rs`
  - `qwen3model_cpu_vs_cuda` — 0.6B, prefill + 5 decode, argmax + cosine ≥ 0.999
  - `qwen3_4b_cpu_vs_cuda` — 4B same gate
  - `qwen3_generate_text_cpu_vs_cuda` — end-to-end 20-token text generation identical to CPU

- **GPTQ parity test**: `crates/ferrum-quantization/tests/gptq_parity_test.rs`
  - `cpu_selfcheck` — synthesised weights, dequant path consistency
  - `cuda_vs_cpu` (#[ignore], needs GPU) — CUDA Marlin vs CPU dequant, relative err < 5%

### GPTQ (Marlin INT4)

- `Backend::GptqStore` associated type + `load_gptq` + `gemm_gptq` methods.
  - CPU: `CpuGptqStore` dequantises to f32 at load, normal GEMM.
  - CUDA: `MarlinWeight` — CPU-side repack via existing
    `crate::marlin::repack_gptq_to_marlin` + `repack_scales_to_marlin`,
    upload to GPU, alloc workspace; forward delegates to `marlin_gemm`.
  - Metal: `type GptqStore = ();` — load/gemm return unsupported.
- `NativeSafetensorsLoader`:
  - Auto-detects `<name>.qweight` and routes to `load_gptq_linear`.
  - Fuses split GPTQ projections (q/k/v → qkv_proj, gate/up → gate_up_proj)
    by concatenating qweight/scales/qzeros along the N dim at load.
  - Parses `quantization_config` either as a standalone `quantize_config.json`
    or embedded in `config.json` (Qwen-GPTQ, transformers-style).

### ASR (Whisper) + TTS (Qwen3-TTS) + CLIP

- CLI `select_candle_device` extended to handle `--backend cuda` —
  routes to `CandleDevice::new_cuda(0)`, falls back to CPU with message.
- Whisper / TTS / embed commands all run on CUDA via candle.
- CLIP: `from_config_json` now reads hidden_size / layers / heads /
  projection_dim from config.json (was hardcoded to base defaults,
  broke on large variants). Verified text + image embedding work on
  CUDA with `openai/clip-vit-large-patch14`, 768-dim output.
- `auto` backend prefers CUDA > Metal > CPU.

**Voice clone status** (two tests, two outcomes):

- **Metal + CPU: WORKS**. `ferrum tts qwen3-tts '人工智能正在改变
  世界。' --backend metal --ref-audio female_ref_5s.wav --ref-text '...'`
  produces 2.4s audio (RTF 3x). Whisper loopback:
  `"人工智能正在改变世界请不吝点赞 订阅 转发..."` — **target text
  intact**, trailing = training-data YouTube outro bleed.
- **CUDA voice clone via existing candle path**: broken, produces
  YouTube-outro garbage at RTF 45x. Root cause found (not candle's
  fault): `ferrum-attention::FusedTransformer` CUDA module was a
  stub; on Linux w/ CUDA feature but no Metal, forward silently fell
  back to `cpu/transformer.rs` which uses **naive fp64 O(n³) matmul**
  (Accelerate is macOS-only). Over 20 decoder layers × 128 decode
  steps the rounding diverges from training numerics enough to pick
  wrong codec tokens → untrained-distribution output.
- **Phase F fix — `Qwen3TtsTalker<B>` Model-as-Code**: replaces the
  FusedTransformer path with a `LlamaFamilyModel<B>`-backed
  transformer stack that uses ferrum-kernels' production CUDA/Metal
  kernels. Loaded via `NativeSafetensorsLoader` + `PrefixedLoader`
  ("talker." prefix). `Qwen3TTSTalker::backend_override` added; the
  `TtsModelExecutor` installs `TalkerBackboneBackend<CudaBackend>`
  for Talker + SubTalker automatically when device is CUDA. Commits
  `b5931d1`, `8a9c99f`, `7e80f7d`.
- **Runtime impact**: CUDA voice-clone RTF drops from **45x → 1.12x**
  (40× speedup). Audio length becomes reasonable (10s vs 5.9s
  of outro garbage). Multi-stage prefill (role prefix → ICL block
  → decode) is verified correct via trace.
- **Remaining gap — f16 precision drift**: CPU↔CUDA codec tokens
  match bit-for-bit for **steps 0-39**, then diverge at **step 40**
  (CPU=984 vs CUDA=1720). Over the ~120 decode steps in a typical
  voice-clone, steps 40+ drift into training-data attractors (outro
  phrases like "Thank you", "感谢观看"). Confirmed via
  `cuda_long_decode` / `cpu_long_decode` in
  `qwen3_tts_backend_smoke.rs` — dumps argmax tokens for side-by-side
  diff. Root cause: our `Backend<CudaBackend>` stores intermediate
  activations in f16; 28 layers × 40 steps accumulated enough
  rounding error to flip a codec selection. Fix options: (a) add
  f32 activation mode to `CudaBackend` scratch buffers, or (b) keep
  f16 and tolerate drift for short prompts.

### Concurrent HTTP loadtest (Qwen3-0.6B)

**After batched decode impl** (LlamaFamilyModel::decode_batch_internal):

| Concurrency | Throughput | p50 latency | p99 latency |
|-------------|-----------|-------------|-------------|
| 1           | 175 tok/s |  283ms      |  528ms      |
| 4           | **314 tok/s** |  699ms | 1065ms      |
| 8           | **344 tok/s** | 1220ms | 2169ms     |

**Before** batched decode (sequential loop inside Mutex): conc=8
saturated at 244 tok/s. Now scales to 344 (+41%).

Implementation (`decode_batch_internal`):
- GEMM-heavy ops run with m=M: qkv_proj, o_proj, gate_up_proj,
  down_proj. Natural batching → significant win on cuBLAS GEMM fixed
  overhead.
- rms_norm, split_qkv, fused_silu_mul, residual_add: all take
  `tokens` parameter, run with tokens=M.
- Per-item attention loop: qk_norm_rope + kv_cache_append +
  flash_attention called M times with tokens=1. copy_slice extracts
  each item's Q/K/V from the M-batched q_buf/k_buf/v_buf into single-
  item scratch (q_single, etc), runs attention, copies result back
  into attn_flat.
- Final norm + lm_head run with m=M into `batch_logits` buffer
  (sized max_tokens * vocab); to_vec slices out M * vocab floats.

GPTQ concurrent improves more (+50%) because Marlin kernel has larger
fixed overhead that GEMM batching amortises.

## ⛔ Graph capture — Rust-side trigger not pinned (2026-04-19)

Earlier diagnosis said "libcuda.so driver bug". Further investigation
ruled that out:

- `scripts/graph_repro{,_v2..v5}.cu` — five bare-C++ reproducers
  covering runtime API, `cuCtxCreate_v4`, memory pool, cross-thread
  capture+replay, cuBLAS sgemm inside capture. **All pass** with 6
  `cuGraphLaunch` replays on Blackwell + CUDA 13.
- `crates/ferrum-kernels/tests/cudarc_graph_repro.rs`:
  - `cudarc_graph_default_pool` — fails at `begin_capture` with
    `STREAM_CAPTURE_ISOLATION` (because cudarc's default event
    tracking injects cross-stream deps during warm-up allocs).
  - `cudarc_graph_no_event_tracking` — **passes** all 6 replays.
  - `cudarc_graph_with_many_allocs` — 200 pre-existing bufs + event
    tracking off — **passes** all 6 replays.

So cudarc + CUDA 13 graph **works** on this box. The ferrum bench
path still SIGSEGVs though, even with event tracking verified off
(`is_event_tracking()=false` logged at `begin_capture`). The
remaining trigger is ferrum-kernels-specific state that the
minimal repros don't exercise. Most likely suspects:

- **cuBLAS workspace reuse across capture + replay** — we share a
  process-global 32 MB workspace, cuBLAS may record it into the
  graph in a way the `_v2` reproducer doesn't stress.
- **Multiple PTX modules** — ferrum loads ~20 distinct kernel PTX
  blobs. Graph capture may be sensitive to module lifetime or JIT
  state across them.
- **`set_decode_state` memcpy_htod pre-launch** — we do 3 small
  htods to device state buffers right before `cuGraphLaunch`.
  Nothing in the repros does this pattern.

**Our-side fixes already landed**:

1. Shared-memory OOB — dynamic shared capped at 32 KB, `cuFuncSet
   Attribute` for >48 KB cases.
2. Per-ctx resource lifetime — cuBLAS handle, workspace, alpha/beta,
   decode-state buffers, captured graph all moved to process-global
   `OnceLock<RwLock<...>>`.
3. cuBLAS host-pointer alpha/beta → DEVICE pointer mode +
   process-global `CudaSlice<f32>` scalars.
4. Event tracking disabled in `default_stream` **before any buffer
   is allocated** (previously only in `new_context` — too late).
5. Graph upload after `end_capture` to skip lazy JIT.
6. `scratch.residual: Option<Buffer>` + `.take()` — no placeholder
   `B::alloc(1)` leaks.
7. Raw FFI `cuStreamEndCapture` + `cuGraphInstantiateWithFlags(0)`
   (cudarc default flag didn't agree with this driver).
8. `Drop for GraphSlotRaw` → `cuCtxSynchronize` + proper destruction.
9. `ctx.bind_to_thread()` pre-`cuGraphLaunch` so tokio-worker switch
   doesn't hit un-bound ctx.
10. Skip candle's `Device::new_cuda(0)` probe in `bench.rs` when
    `FERRUM_CUDA_GRAPH=1` (candle instantiates another
    `CudaContext` with default event tracking).

**Default**: eager (graph disabled). `FERRUM_CUDA_GRAPH=1` still
crashes in `ferrum` binary. Infrastructure kept in place for
follow-up work to isolate the remaining trigger.

### Our-side fixes that DID land (kept for future retry)

All these were real bugs, just shadowed by the driver SIGSEGV:

1. **Shared memory OOB** — `extern __shared__` sized per current
   kv_len; captured graph bakes in smaller size than later replays
   need. **Fixed**: cap dynamic shared at 32 KB; `FERRUM_CUDA_MAX_KV`
   env to raise (auto opts into >48KB via `cuFuncSetAttribute`).
2. **Per-ctx resource lifetime** — captured graph held pointers to
   workspace/alpha/beta/decode-state buffers that were freed when the
   capturing ctx dropped. **Fixed**: moved cuBLAS handle + workspace +
   decode-state buffers + captured graph to process-global
   `OnceLock<RwLock<...>>` slots.
3. **cuBLAS host-pointer alpha/beta** — HOST pointer mode queues
   memcpy with stack-local scalar pointer, stale at replay. **Fixed**:
   `cublasSetPointerMode(DEVICE)` + process-global
   `alpha_f32/beta_f32: CudaSlice<f32>` holding 1.0/0.0.
4. **Event tracking cross-stream deps** — cudarc auto event recording
   fails capture with `STREAM_CAPTURE_ISOLATION`. **Fixed**: disable
   event tracking globally in `new_context`.
5. **Graph upload timing** — first replay triggers JIT. **Fixed**:
   `graph.upload()` immediately after `end_capture`.
6. **Placeholder-alloc drop** — `mem::replace(&mut scratch.residual,
   B::alloc(1))` dropped a stream-ordered alloc after graph capture,
   corrupting pool state. **Fixed**: `scratch.residual:
   Option<Buffer>` + `.take()`.
7. **cudarc `end_capture` non-zero default flag** — split into raw
   FFI `cuStreamEndCapture` + `cuGraphInstantiateWithFlags(flags=0)`.
8. **Graph ref lifetime across requests** — `cuGraphExec` held
   memory-pool refs. **Fixed**: `Drop for GraphSlotRaw` calls
   `cuCtxSynchronize` + `cuGraphExecDestroy` + `cuGraphDestroy`;
   `release` invokes `sync → reset_graph → sync` before cache return.
9. **Thread binding** — `cuGraphLaunch` from un-bound tokio worker
   hung silently. **Fixed**: `ctx.bind_to_thread()` pre-launch.

Default: eager. `FERRUM_CUDA_GRAPH=1` now errors out with the driver
trace. Impact: 4B runs at 82 tok/s (~92% of 88.8 baseline); the
~7 tok/s gap is what graph would recover if/when the driver is fixed.

### Kernel infrastructure for graph capture (retained)

New `_dyn` kernel variants that read dynamic scalars from device:

- `embedding_lookup_f16_dyn` — reads token id from `token_buf`
- `qk_norm_rope_transpose_f16_dyn` — reads `pos_offset` from `pos_buf`
- `kv_cache_append_head_major_f16_dyn` — reads `cache_len` from `pos_buf`
- `decode_attention_head_major_f16_dyn` — reads `valid_kv_len` from `kv_buf`

`Backend::set_dev_state_mode(enable)` toggles the scalar vs dynamic
kernel dispatch. `Backend::set_decode_state(ctx, token, step)` updates
the process-global device buffers via memcpy_htod_async.

## ❌ Not done (future work)

- **Bert Model-as-Code** (Phase D.5c) — Backend extensions landed
  (layer_norm, gelu, add_bias). Model port pending.
- **Whisper Model-as-Code** (Phase D.5e) — currently runs via candle.
  Full port needs `B::conv1d` / `conv2d` kernels.
- **TTS Model-as-Code** (Phase D.6) — largest port; vocoder needs
  `B::conv_transpose1d` + CausalConv variants.
- **Graph capture** — see above, currently broken.
- **NCCL TP all_gather / broadcast** — stubs only. `all_reduce` skeleton
  is wired to `crate::nccl_comm::NcclRank`.
- **Flash-decode split-K path** — present in old runner but not ported;
  long-context decode (>256 kv) would benefit.

## Reproducing

```bash
# Box setup (rented GPU with CUDA 12.8, Rust 1.80+):
git clone https://github.com/sizzlecar/ferrum-infer-rs.git -b feat/cuda-tts
cd ferrum-infer-rs
bash scripts/phase-e-verify.sh qwen3:0.6b

# Manual benches:
export CUDA_HOME=/usr/local/cuda-12.8
cargo build --release --features cuda -p ferrum-cli --bin ferrum
target/release/ferrum bench qwen3:0.6b --backend cuda --max-tokens 128
target/release/ferrum bench qwen3:4b   --backend cuda --max-tokens 128
target/release/ferrum bench qwen2.5:3b-gptq --backend cuda --max-tokens 128
target/release/ferrum transcribe whisper-turbo audio.wav --backend cuda
target/release/ferrum tts qwen3-tts "hello" --backend cuda -o out.wav
```
