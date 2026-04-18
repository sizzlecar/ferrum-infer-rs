# Phase E CUDA — Status

Runtime-validated on RTX PRO 6000 Blackwell (SM 12.0, CUDA 12.8).

## ✅ Working end-to-end

### LLM inference (LlamaFamilyModel<CudaBackend>)

| Model | Path | tok/s | Parity | Notes |
|-------|------|-------|--------|-------|
| Qwen3-0.6B FP16 | eager | ~141 | ✅ cos=0.999998 | matches CPU argmax + text |
| Qwen3-4B FP16 | eager | ~81 | ✅ cos=1.000000 | tied-emb wired; 91% of 88.8 target |
| Qwen2.5-3B-GPTQ-Int4 | Marlin | ~106.5 | ✅ text | bias-on-attn projections wired; 95% of 112.4 target |

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

### ASR (Whisper) + TTS (Qwen3-TTS)

- CLI `select_candle_device` extended to handle `--backend cuda` —
  routes to `CandleDevice::new_cuda(0)`, falls back to CPU with message.
- Both commands run on CUDA via the existing candle-based executor
  (no Model-as-Code port yet — that's Phase D.5e / D.6 proper).
- `auto` backend prefers CUDA > Metal > CPU.

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

- ✅ Single-request capture + replay works: warmup (14-16 replays) runs
  clean, all kernels record into graph, replay reuses them.
- ❌ Multi-request breaks: on the 2nd request, `stream.alloc()` for the
  new KV cache fails with CUDA_ERROR_INVALID_VALUE. Even with
  sync→reset_graph→sync before the cache free, the stream allocator
  pool ends up in a bad state the next alloc can't recover from.

The fundamental issue: captured graphs hold raw device pointers into
the request's KV cache. Freeing that cache while the stream's
allocator still has outstanding graph-bound reservations corrupts
the pool. vLLM avoids this with a process-global paged KV pool that
never gets freed between requests.

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
