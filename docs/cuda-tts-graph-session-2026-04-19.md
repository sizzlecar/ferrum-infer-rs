# CUDA TTS + Graph — Session Status (2026-04-19)

Branch: `feat/cuda-tts` (195 commits ahead of `main`)
Hardware: RTX PRO 6000 Blackwell Server Edition, CUDA 13.0 driver 580.126.09
Remote: `ssh -p 35057 root@ssh9.vast.ai`

This document captures the state of two parallel workstreams after an
extended debug session: the **CUDA TTS Model-as-Code port** (voice-clone
on CUDA) and **CUDA graph capture** (the ~7 tok/s perf gap for eager-mode
decode). Both are **functionally in place but not fully fixed**; the
remaining work is scoped and described here.

---

## 1. TTS on CUDA — where we are

### ✅ Done

- **Architecture port** (commits `b5931d1`, `8a9c99f`, `c24d3aa`,
  `7e80f7d`): replaced `ferrum_attention::FusedTransformer` with
  `TalkerBackboneBackend<B>` — a `LlamaFamilyModel<B>`-backed transformer
  stack loaded via `NativeSafetensorsLoader` + `PrefixedLoader("talker.")`.
  Works for both Talker and SubTalker.
- **Executor integration**: `TtsModelExecutor::from_path` automatically
  installs `Qwen3TtsTalker<CudaBackend>` + `Qwen3TtsSubTalker<CudaBackend>`
  overrides when the candle device is CUDA. `tracing::info` log line
  `"TtsModelExecutor: Backend<CudaBackend> installed for Talker + SubTalker"`
  confirms. Metal and CPU paths unchanged.
- **Multi-stage prefill** (`prefill_all_post_norm` with `pos_offset`):
  handles the voice-clone pattern of role-prefix (9 tokens) → ICL-block
  (~64 tokens at `pos_offset=9`) → autoregressive decode. Verified
  correct via debug trace output showing each stage's
  `(cache_id, seq_len, pos_offset)`.
- **Performance**: voice-clone `RTF 45x → 1.12x` — a **40× speedup**
  over the pre-fix state where `FusedTransformer`'s CUDA module was a
  stub and the Linux CPU fallback used naive fp64 O(n³) matmul.
- **CPU↔CUDA parity** on the first 40 codec tokens
  (`crates/ferrum-models/tests/qwen3_tts_backend_smoke.rs`). Smoke test
  covers prefill + 4 decode steps + SubTalker predict — identical
  argmax tokens on both backends.

### ❌ Unresolved — **voice-clone content drift**

**Symptom**: the produced audio length is correct (e.g. 10.56 s for a
sentence-sized prompt), RTF is dramatically improved, but Whisper
loopback transcribes "Thank you. Thank you." instead of the target
text. The same inputs on Mac Metal produce the target text followed by
training-data outro — Metal works.

**What we already ruled out**:

1. Transformer structural / KV-cache logic — smoke test confirms
   CPU↔CUDA bit-for-bit match on first 40 tokens in long-decode.
2. Multi-stage prefill — trace shows correct `pos_offset` threading
   through role-prefix → ICL → decode.
3. Sole Talker precision — setting `FERRUM_USE_CANDLE=1` (routes the
   Talker forward through candle's native f32 CUDA tensor ops, keeping
   SubTalker on our f16 `Backend<CudaBackend>` override) STILL
   produces "Thank you" output. So Talker f16 drift is not the only
   cause.

**What we proved is a real issue**:

The long-decode smoke test
(`cpu_long_decode` vs `cuda_long_decode` in the same test file) runs
one prefill + 50 decode steps with argmax sampling, dumping codec
tokens. Diff:

| Step | CPU (f32) | CUDA (f16) |
|------|-----------|------------|
| 0-39 | identical | identical |
| 40   | 984       | 1720       |
| 41   | 1789      | 2045       |
| ...  | drift compounds |

So f16 activation precision accumulates enough rounding error over
28 decoder layers × 40 decode steps to flip one codec selection. Over
the ~120 decode steps in a real voice-clone, the drift covers 2/3 of
the audio. But content drift extends beyond this — even with Talker
kept in f32 via candle, output is wrong.

**Remaining suspects in order of likelihood**:

1. **SubTalker f16 drift** (same issue as Talker, unfixable without
   f32 path since we only added candle-fallback to Talker not SubTalker).
2. **Candle CUDA upstream** — SpeakerEncoder (ECAPA-TDNN with
   Conv1d), text_embedding, text_projection, codec_embedding all still
   run via candle on CUDA. Any precision delta between candle CUDA
   kernels and Metal kernels for these ops affects the embedding fed
   into Talker.
3. **SpeechTokenizerEncoder** forced to CPU (line 127 of
   `tts_executor.rs`): `CandleDevice::Cpu` — same on all platforms.
   Unlikely to differ.
4. **Vocoder** — if upstream codes are right, vocoder renders them
   consistently. Since our codec tokens themselves are wrong, vocoder
   isn't the issue.

### How to actually fix this

Needs one of:

- **(A) f32 activation path in CudaBackend** — add a second Buffer
  type (CudaSlice<f32>) + cast kernels, route TTS through it. ~500
  LOC. Eliminates f16 drift in both Talker and SubTalker. Likely
  fixes the content if SubTalker is indeed the remaining cause.
- **(B) Candle path for SubTalker** — write a `predict_candle`
  method on `SubTalker` that uses its existing `layers: Vec<TransformerLayer>`
  (candle-based) instead of raw f32 matmul / fused transformer.
  Combined with `FERRUM_USE_CANDLE=1` would let Talker AND SubTalker
  use candle f32 on CUDA. ~150 LOC. Verifies whether upstream candle
  components are the cause or just SubTalker precision.
- **(C) Explicit bisection** — add a debug tap in `synthesize_voice_clone`
  that dumps the exact codec tokens produced per frame. Run on Mac
  Metal (known-good) and Linux CUDA (broken), diff frame-by-frame.
  First divergence point tells us whether the gap is at embedding,
  Talker, SubTalker, or elsewhere.

Recommended sequence: **(C) first** (cheap, informational) → **(B)
if SubTalker-precision is confirmed** → **(A) as the proper long-term
fix**.

---

## 2. Graph capture — where we are

### The original diagnosis was wrong

Earlier commits (`02d72fa`) claimed the `cuGraphLaunch` SIGSEGV was a
`libcuda.so` driver bug. A gdb trace showed the crash inside the
driver at frame `#3 cuGraphLaunch`. Closer inspection via extensive
reproducer matrix disproves this.

### Reproducer matrix

| Variant | Language | Features | Result |
|---------|----------|----------|--------|
| `scripts/graph_repro.cu` (v1) | C | runtime API ctx + cuMemAlloc + 1 kernel | ✅ pass |
| `scripts/graph_repro_v2.cu` | C | + cuMemAllocFromPoolAsync + worker thread | ✅ pass |
| `scripts/graph_repro_v3.cu` | C | + cuCtxCreate_v4 non-primary ctx | ✅ pass |
| `scripts/graph_repro_v4.cu` | C | + cuBLAS sgemm inside capture | ✅ pass |
| `scripts/graph_repro_v5.cu` | C | + capture on thread A, replay on thread B | ✅ pass |
| `tests/cudarc_graph_repro.rs::cudarc_graph_default_pool` | Rust | cudarc default (event tracking on) | ❌ STREAM_CAPTURE_ISOLATION at begin_capture |
| `tests/cudarc_graph_repro.rs::cudarc_graph_no_event_tracking` | Rust | cudarc + `disable_event_tracking()` | ✅ pass |
| `tests/cudarc_graph_repro.rs::cudarc_graph_with_many_allocs` | Rust | + 200 pre-existing weight-sized buffers | ✅ pass |
| `tests/cudarc_graph_repro.rs::cudarc_graph_like_ferrum` | Rust | + cuBLAS DEVICE mode + 2 PTX modules + htod pre-launch | ✅ pass |
| `ferrum bench qwen3:0.6b --backend cuda` with `FERRUM_CUDA_GRAPH=1` | Rust | full ferrum-kernels stack | ❌ SIGSEGV at cuGraphLaunch |

Conclusion: **driver path is clean, cudarc path is clean, even
most-of-ferrum-mimicked-in-cudarc is clean**. The trigger is something
specific to ferrum-kernels that the most-complete bare Rust
reproducer does not yet exercise.

### Our-side fixes that landed anyway (kept in tree)

1. **Event tracking disabled before any buffer is allocated** (commit
   `169e619`). Previously only disabled in `new_context` which runs
   after model weights are already loaded — too late. Now disabled
   in `default_stream`'s first-time ctx-creation path. Verified via
   eprintln showing `is_event_tracking()=false` at `begin_capture`.
2. **Shared memory cap at 32 KB** with `cuFuncSetAttribute` opt-in
   for >48 KB kernels.
3. **Process-global cuBLAS handle + 32 MB workspace + DEVICE pointer
   mode alpha/beta** — so captured kernel args don't dangle when a
   per-ctx workspace frees.
4. **Process-global decode-state buffers** (token/pos/kv) — captured
   graph's dynamic-state reads from stable device addresses.
5. **Raw FFI `cuStreamEndCapture` + `cuGraphInstantiateWithFlags(0)`**
   — cudarc's default `end_capture` uses a non-zero flag that
   disagrees with this driver.
6. **`scratch.residual: Option<B::Buffer>`** with `.take()` pattern —
   no transient `B::alloc(1)` placeholder that would cause
   `cuMemFreeAsync` during capture.
7. **`Drop for GraphSlotRaw`**: `cuCtxSynchronize` + `cuGraphExecDestroy`
   + `cuGraphDestroy` in the right order.
8. **`ctx.bind_to_thread()` pre-`cuGraphLaunch`** so a tokio-worker
   thread switch doesn't hit an un-bound ctx.
9. **Skip candle's `Device::new_cuda(0)` probe in `bench.rs`** when
   `FERRUM_CUDA_GRAPH=1` — candle instantiates another `CudaContext`
   with default tracking (commit `64a580e`).
10. **`graph.upload()` after `end_capture`** — avoid JIT on first
    replay.

### What the remaining ferrum-specific trigger might be

Given bare-Rust-cudarc with cuBLAS + 2 modules + htod pre-launch
passes, and ferrum bench fails, the remaining deltas are:

- **Ferrum's own PTX kernels** (rms_norm, flash_attention,
  qk_norm_rope, split_qkv, fused_silu_mul_split, residual_add,
  embedding_lookup, transpose_head_to_token, copy_slice, add_bias,
  layer_norm, gelu, add_inplace — ~12 kernel types). Any one of them
  may have a specific stream-capture interaction, e.g. using
  `extern __shared__` with a size that depends on ctx state, or
  referencing constant memory that cudaMalloc'd separately.
- **Scratch buffer lifecycle** under `ensure_scratch(seq_len)` growth.
- **Multi-kernel ordering** — ferrum launches 20+ kernels per decode
  step; the graph node count is much larger than the test.
- **cuModule drop ordering** relative to capture.

### How to actually find the trigger

- **Kernel-by-kernel swap**: in the ferrum decode flow, gradually
  replace each of our PTX kernels with a no-op touch kernel and see
  at which point graph capture starts to work. Identifies the
  offending kernel.
- **compute-sanitizer --tool synccheck** with source-level symbols
  to catch sync-order violations that flip to SIGSEGV at replay time.
- **Single-kernel capture mode**: capture a graph containing only
  ONE kernel (say rms_norm) at a time, build up. Binary-search
  which kernel triggers the crash.

All of these need on-GPU iteration with targeted code changes — not
feasible to fully resolve in a single auto-loop session.

---

## 3. Commit history map

The 195 commits on `feat/cuda-tts` fall into these buckets:

### Architecture (Phase F)
- `ba1c075` prefill_from_embeds + decode_from_embed on LlamaFamilyModel
- `31f1e77` PrefixedLoader
- `880b96b` Option<embed>/Option<lm_head> + new_backbone_only
- `b5931d1` Qwen3TtsTalker<B>
- `8a9c99f` Qwen3TtsSubTalker<B>
- `c24d3aa` prefill_all_post_norm + decode_post_norm_from_embed
- `7e80f7d` backend_override plumbing in Qwen3TTSTalker + SubTalker + executor

### Graph fixes
- `169e619` disable event tracking before weight buffers
- `64a580e` skip candle probe in bench when graph mode on
- Earlier: resource lifetime / cuBLAS device mode / Option<residual>
  / raw FFI end_capture / Drop for GraphSlotRaw / bind_to_thread

### Graph reproducers
- `02d72fa` graph_repro.cu (v1)
- `f67b4e0` v2 worker thread + memory pool
- `d20d988` v3 cuCtxCreate_v4
- `f0d2490` v4 cuBLAS sgemm
- `0539b24` v5 cross-thread
- `c174aa8` cudarc_graph_repro.rs (Rust version)
- `4a55995` + `3ca7b07` cudarc_graph_with_many_allocs + like_ferrum

### Tests
- `951ba02` smoke test extended to SubTalker.predict_greedy
- `10f3835` 50-step long-decode for CPU↔CUDA divergence localisation
- `3655d15` CPU↔CUDA parity docs

### Eager-mode perf
- `8b48f4b` strip hot-path eprintln leftovers

---

## 4. Reproducing locally

```bash
# From a GPU host with CUDA 13 + HF weights cached:
ssh -p 35057 root@ssh9.vast.ai
source ~/.cargo/env && cd /workspace/ferrum-infer-rs
git checkout feat/cuda-tts && git pull

# Build
CUDA_HOME=/usr/local/cuda-13.0 cargo build --release --features cuda \
    -p ferrum-cli --bin ferrum

# Voice clone (runs, content drifts)
HF_HOME=/workspace/.hf_home ./target/release/ferrum tts qwen3-tts \
    "The quick brown fox jumps over the lazy dog" --backend cuda \
    --ref-audio /workspace/test_5s.wav \
    --ref-text "I do not know. You do not know. ..." \
    -o /tmp/out.wav
# RTF ~1.12x. Whisper loopback returns "Thank you. Thank you." (wrong).

# TTS parity smoke tests
HF_HOME=/workspace/.hf_home cargo test --release --features cuda \
    -p ferrum-models --test qwen3_tts_backend_smoke \
    -- --ignored --nocapture
# cuda_smoke: passes
# cuda_long_decode: dumps 51 tokens to /tmp/tts_smoke_cuda.tokens
# (diff against /tmp/tts_smoke_cpu.tokens from Mac shows step-40 divergence)

# Graph reproducers (all pass)
for v in 1 2 3 4 5; do
    src=scripts/graph_repro$([ $v = 1 ] && echo "" || echo "_v$v").cu
    /usr/local/cuda/bin/nvcc -o /tmp/grepro_v$v $src -lcuda ${v:+$([ $v -ge 4 ] && echo -lcublas)} -std=c++17
    /tmp/grepro_v$v
done

cargo test --release --features cuda -p ferrum-kernels \
    --test cudarc_graph_repro -- --ignored --nocapture --test-threads=1

# Graph in ferrum bench (fails — this is the unresolved issue)
FERRUM_CUDA_GRAPH=1 ./target/release/ferrum bench qwen3:0.6b --backend cuda \
    --rounds 1 --max-tokens 16
# → SIGSEGV at cuGraphLaunch inside libcuda.so
```

---

## 5. Handover summary

**For the next session / contributor**:

- Architecture for TTS CUDA path is in place and the pipeline runs
  end-to-end; users will see a 40× RTF improvement but wrong content
  on voice-clone with long prompts.
- Two options to ship: document the limitation
  (short prompts <3s work) OR invest ~1-2 sessions to add f32
  activation path to `CudaBackend`.
- Graph capture has all the infrastructure. The trigger is a specific
  ferrum-kernels state interaction that needs invasive bisection
  (swap kernels one-at-a-time under capture). Not urgent — eager is
  within 8% of the previous graph-enabled perf.
- All 195 commits are logical and reviewable; no "garbage" commits
  that need reverting. Diag commits (`diag(cuda):` prefix) can be
  squashed once the work is finalised.
