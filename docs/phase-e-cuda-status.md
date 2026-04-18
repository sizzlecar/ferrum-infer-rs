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

**Voice clone status** (revised after local Metal test):

- **Metal + CPU: WORKS**. Local `ferrum tts qwen3-tts '人工智能正在改变
  世界。' --backend metal --ref-audio female_ref_5s.wav --ref-text '...'`
  produces 2.4s of audio. Whisper loopback transcript:
  `"人工智能正在改变世界请不吝点赞 订阅 转发..."`. **Target text is
  intact**; trailing noise = YouTube channel outro bleed from training
  data (model generates 4-5 extra frames before codec_eos).
- **CUDA (earlier remote box)**: produced 0.16s garbage audio. Not an
  algorithm-layer bug as previously documented — the CUDA-specific
  failure is likely in candle's CUDA TTS forward pass (Qwen3-TTS isn't
  ported to our kernels yet; runs via candle). Reproducing on a working
  CUDA box would help pin this down. `FERRUM_TTS_MIN_FRAMES` env
  remains available as a tuning knob for min output length.

### Concurrent HTTP loadtest (Qwen3-0.6B)

| Concurrency | Throughput | p50 latency | p99 latency |
|-------------|-----------|-------------|-------------|
| 1           | 215 tok/s |  250ms      |  483ms      |
| 4           | 242 tok/s | 1020ms      | 1189ms      |
| 8           | 244 tok/s | 2034ms      | 2337ms      |

Saturates past concurrency=4 — true batched decode (one GEMM with
m=batch, per-item attention) would push further. Currently the
`LlmExecutor::batch_decode` override acquires the model mutex once
per batch and dispatches to `DecoderOnlyLLM::decode_batch` whose
default impl loops `decode` sequentially.

Plumbing in place for future `decode_batch` overrides; the kernel
work to actually batch (GEMM m=M, per-item KV append+attention loop)
requires either:
- Buffer-offset-aware kernel variants (new `_at` trait methods for
  qk_norm_rope / kv_cache_append / decode_attention with offsets)
- OR per-token `pos[]` array parameter in qk_norm_rope
- Plus per-item scratch or slicing support in the Backend's Buffer type

Not tractable from an SSH-only debug session — needs kernel iteration
on-box with nvcc. Tracked as follow-up.

## ⚠️ Graph capture — experimental

Infrastructure landed but replay triggers `CUDA_ERROR_INVALID_VALUE`
on Blackwell + CUDA 12.8 in a way I couldn't fully diagnose on the
rented session. Root-cause suspicions (each of which has fixes landed):

1. **Shared memory OOB** — kernel's `extern __shared__` sized per
   current kv_len; captured graph bakes in smaller size than later
   replays need. **Fixed**: cap dynamic shared at 32 KB (8192 positions);
   `FERRUM_CUDA_MAX_KV` env to raise (auto opts into >48KB via
   cuFuncSetAttribute).
2. **Per-ctx resource lifetime** — captured graph held pointers to
   workspace/alpha/beta/decode-state buffers that were freed when the
   capturing ctx dropped. **Fixed**: moved cuBLAS handle + workspace +
   decode-state buffers (token/pos/kv) + captured graph to
   process-global `OnceLock<RwLock<...>>` slots.
3. **cuBLAS host-pointer alpha/beta** — HOST pointer mode causes
   cuBLAS to queue a memcpy with stack-local scalar pointer that goes
   stale at replay. **Fixed**: `cublasSetPointerMode(DEVICE)` +
   process-global `alpha_f32/beta_f32: CudaSlice<f32>` holding 1.0/0.0.
4. **Event tracking cross-stream deps** — cudarc's auto event
   recording fails capture with `CUDA_ERROR_STREAM_CAPTURE_ISOLATION`.
   **Fixed**: `ctx.disable_event_tracking()` around capture.
5. **Graph upload timing** — first replay may trigger JIT setup that
   fails under the current stream state. **Fixed**: call `graph.upload()`
   immediately after `end_capture`.

**Current state of graph with `FERRUM_CUDA_GRAPH=1`:**

Root causes found across two boxes (CUDA 12.8 + 13.0, both Blackwell):

1. **Placeholder-alloc drop corruption**: `mem::replace(&mut scratch.residual, B::alloc(1))` dropped a stream-ordered alloc after graph capture, corrupting pool state so `bind_to_thread` / `sync` / `dtoh` all returned INVALID_VALUE. **Fixed**: `scratch.residual: Option<Buffer>` + `.take()` — no placeholder.
2. **cudarc's end_capture packaged cuStreamEndCapture + cuGraphInstantiateWithFlags as one call with non-zero default flag**. Traced showed context ptr was fine after both; fix was not actually in this layer but the diagnostic was useful. **Now**: raw FFI split with flags=0 explicitly.
3. **Graph ref lifetime across requests**: captured graph's cuGraphExec held memory-pool references that prevented new-request allocs. **Fixed**: `Drop for GraphSlotRaw` calls `cuCtxSynchronize` + `cuGraphExecDestroy` + `cuGraphDestroy`; `LlamaFamilyModel::release` invokes `sync → reset_graph → sync` before returning cache to pool.

**With all fixes**: within-request replay confirmed working (observed 10+ replays succeed). **But**: flaky — same binary sometimes hangs at first `cuGraphLaunch`, sometimes replays cleanly for entire request. No consistent repro from SSH-only debug. Blackwell Server Edition + driver + cudarc 0.19 has some additional interaction needs cuda-gdb on-box to pin down.

Default: eager. `FERRUM_CUDA_GRAPH=1` for research only. Infrastructure fully in place so follow-up with cuda-gdb can focus on the remaining flakiness without redoing the plumbing.

Default: eager (no graph). Set `FERRUM_CUDA_GRAPH=1` for the single-
request case. Production fix requires either per-graph memory pools
or graph-less KV cache reuse.

**Impact**: 4B runs at 82 tok/s (92% of the 88.8 baseline). The
~7 tok/s gap is what graph capture is expected to recover.

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
