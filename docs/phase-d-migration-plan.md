# Phase D.5 / D.6 Migration Plan

Status: **D.5a + D.5b shipped** (Backend extensions). D.5c (Bert) onward is
work that needs dedicated sessions with model-specific end-to-end
verification environments (audio files, reference embeddings, ASR ground
truth).

This document records *what* has to be done for each model, so those
sessions can execute without re-planning.

---

## Backend trait — done

The LLM hot path uses:
  `gemm`, `rms_norm`, `fused_add_rms_norm`, `qk_norm_rope`, `split_qkv`,
  `fused_silu_mul_split`, `flash_attention`, `kv_cache_append_head_major`,
  `transpose_head_to_token`, `add_inplace`, `copy_slice`,
  `embedding_lookup`.

Phase D.5a added, with Metal kernel + CpuBackend impl:
  `layer_norm(x, γ, β, eps, out, tokens, dim)`
  `gelu(x, out, len)`
  `add_bias(data, bias, rows, cols)`

Phase D.5b: `DenseLinear<B>` gained optional bias
  (`from_rows_with_bias`, `with_bias`). `forward` dispatches `gemm + add_bias`.

## Still to add (per target model)

| Op | Needed for | Backend default |
|----|------------|-----------------|
| `cross_attention(q_dec, kv_enc, out, ...)`          | Whisper decoder | `Err(Unsupported)` stub (no caller yet) |
| `conv1d(input, kernel, bias, stride, pad, out)`     | Whisper mel frontend | Skip: mel runs on CPU, one-time |
| `fused_bias_gelu(a, bias, out, rows, cols)` (opt.)   | Bert FFN perf       | `gemm + add_bias + gelu` suffices |
| `tanh`                                               | Bert pooler, some TTS | 2 lines CpuBackend; 1 shader  |

Cross-attention is the main real need; others are small.

---

## D.5c — `BertModel<B>` (encoder-only)

### Scope
~500 LOC Rust + shaders (none new needed, uses D.5a additions).

### Structure
```rust
// ferrum-models/src/models/bert.rs
pub struct BertConfig {
    hidden_size, num_heads, num_layers, intermediate_size,
    max_position_embeddings, type_vocab_size,  // = 2 for Bert
    vocab_size, layer_norm_eps,
}

pub struct BertLayer<B: Backend> {
    attn_q: Box<dyn Linear<B>>,    // with bias
    attn_k: Box<dyn Linear<B>>,
    attn_v: Box<dyn Linear<B>>,
    attn_o: Box<dyn Linear<B>>,
    attn_ln_gamma: B::Buffer,
    attn_ln_beta: B::Buffer,
    ff_intermediate: Box<dyn Linear<B>>,  // with bias
    ff_output: Box<dyn Linear<B>>,
    ff_ln_gamma: B::Buffer,
    ff_ln_beta: B::Buffer,
}

pub struct BertModel<B: Backend> {
    cfg: BertConfig,
    word_emb: B::Buffer,       // [vocab, h]
    position_emb: B::Buffer,   // [max_pos, h]
    token_type_emb: B::Buffer, // [2, h]
    embed_ln_gamma, embed_ln_beta: B::Buffer,
    layers: Vec<BertLayer<B>>,
    pooler_dense: Option<Box<dyn Linear<B>>>,  // with bias; optional
    scratch: BertScratch<B>,
}
```

### Forward pipeline
```
1. embeddings[t] = word_emb[tokens[t]] + position_emb[t] + token_type_emb[seg[t]]
2. LayerNorm over embeddings → residual
3. N × BertLayer:
   a. q/k/v projections (with bias) → q_head_major / k_head_major / v_head_major
      (reuse split_qkv shader? No — Bert has separate q_proj/k_proj/v_proj
       in checkpoints. Either (1) concat on load like Llama does, or
       (2) issue three separate gemms. Option 1 keeps the forward aligned
       with LlamaFamilyModel pattern.)
   b. flash_attention(q, k, v, out, AttnConfig { causal: false, ... })
      — full bidirectional attention, no causal mask. Already supported;
        just set causal=false.
   c. O-proj (with bias); residual += O
   d. LayerNorm over residual (post-LN, not pre-LN like Llama)
   e. FFN: gemm(intermediate, with bias) + gelu + gemm(output, with bias)
   f. residual += FFN; LayerNorm
4. Final hidden = residual
5. Pooling:
   - "CLS": hidden[:, 0, :]  (requires copy_slice)
   - "mean": mean over tokens dim (needs a mean_reduce op — or do on host
     after to_vec since the embedding is small)
6. Optional pooler: tanh(pooler_dense(pooled))
```

### Families
Implement `EmbeddingModel` trait from `ferrum-models::common::families`:
```rust
impl<B: Backend> EmbeddingModel for BertModel<B> {
    fn embed(&mut self, tokens: &[u32]) -> Vec<f32> {
        // fresh forward — no KV cache for encoder
        // pool (CLS or mean), return Vec<f32>
    }
}
```

### Executor integration
Option A: new `EmbeddingExecutor` wrapping `Box<dyn EmbeddingModel>`.
Option B: leave `BertModelExecutor` but swap its internals.

A is cleaner. Wire registry accordingly.

### Verification
- Keep existing `BertModelExecutor` alive via env flag `FERRUM_BERT_NEW=1`.
- Write parity test: same input tokens → both backends → compare pooled
  embedding cosine. Target: cos > 0.9999.
- Known-good reference: `bert-base-chinese` in HF cache.
- Metric: with deterministic inputs, new path should produce embeddings
  identical to candle-bert to ~1e-4 absolute.

### Weight name mapping (safetensors → BertLayer)
```
bert.embeddings.word_embeddings.weight         → word_emb
bert.embeddings.position_embeddings.weight     → position_emb
bert.embeddings.token_type_embeddings.weight   → token_type_emb
bert.embeddings.LayerNorm.weight / .bias       → embed_ln_gamma / beta
bert.encoder.layer.{i}.attention.self.query.weight / .bias   → attn_q.weight / .bias
...                                            .key.*         → attn_k
...                                            .value.*       → attn_v
bert.encoder.layer.{i}.attention.output.dense.weight / .bias   → attn_o
bert.encoder.layer.{i}.attention.output.LayerNorm.weight / .bias → attn_ln_*
bert.encoder.layer.{i}.intermediate.dense.weight / .bias     → ff_intermediate
bert.encoder.layer.{i}.output.dense.weight / .bias           → ff_output
bert.encoder.layer.{i}.output.LayerNorm.weight / .bias       → ff_ln_*
bert.pooler.dense.weight / .bias               → pooler_dense (optional)
```

`NativeSafetensorsLoader` already handles f32/f16/bf16 dtype conversion
and concat-on-load for fused projections, so adding Bert is a matter of
new weight name mapping only.

---

## D.5d — `ClipModel<B>` (text encoder + vision encoder)

### Scope
~800 LOC — two encoders in one file or two files.

Text encoder is essentially Bert (minus token_type_embeddings; uses
RoPE in newer CLIP; original CLIP uses learned position embeddings).

Vision encoder (ViT):
  - `conv2d` patch embedding (16×16 patches → sequence)
  - CLS + position embeddings
  - N × encoder layer (same as Bert-style: self-attn + FFN)
  - Final LayerNorm + projection to shared embedding space

New op needed: `conv2d_patch_embed` or just a model-level reshape + gemm
(ViT patches are a reshape of image pixels to `[num_patches, patch_pixels]`
then a linear projection — no real convolution needed).

### Executor
`EmbeddingModel` for text-only; multimodal needs a split trait. Simpler
path: `ClipTextModel<B>` implements `EmbeddingModel`, `ClipVisionModel<B>`
implements a new `VisionEncoder` trait.

### Verification
- `OFA-Sys/chinese-clip-vit-base-patch16` in HF cache.
- Parity against existing `ClipModelExecutor` for both text and image
  branches.

---

## D.5e — `WhisperModel<B>` (encoder-decoder)

### Scope
The largest of this group: **~1500-2000 LOC**. Timestamp handling alone
is ~300 LOC.

### Components
1. **Mel frontend** (CPU, already exists in `ferrum-models::mel`)
2. **Encoder**: conv downsample + N × encoder layer (GELU, LayerNorm)
3. **Cross-attn decoder**: N × (self-attn + cross-attn + FFN)
4. **Sampler**: temperature fallback, compression-ratio check, logprob
   threshold, no-speech threshold
5. **Timestamp rules**: SuppressBlank, SuppressTokens (82 non-speech),
   ApplyTimestampRules, seek-based segmentation
6. **Language detection** (optional per-segment)

### Backend gaps
- `cross_attention(q, k_enc, v_enc, out, ...)` — same math as flash, but
  K/V come from encoder output, so `causal=false` and `kv_len` is fixed
  per-sequence. Could be handled by adding a
  `Backend::cross_attention` method or just reuse `flash_attention` with
  `causal=false` and stride=0.
- Mel + STFT stay CPU (existing `rustfft` code).

### Validation
- 5-min Chinese audio in `~/ferrum-test-data/`, Python baseline output
  saved (see `metal-performance-status.md` for exact command).
- Transcribe + diff: target Chinese character level match within ~1%
  (CPU-vs-Metal float drift).
- Also bench: Python 107s baseline vs Rust Metal current ~72s; new path
  should match or beat (no regression, decode shouldn't change since
  flash_attention already dominates).

### Executor
`Transcriber` trait, wrapping `Box<dyn EncoderDecoderLM>` + mel frontend
+ sampler.

---

## D.6 — `Qwen3-TTS` migration

### Scope
**Largest of all, ~2000-2500 LOC.** Three sub-models plus speaker encoder
plus flow-matching vocoder finisher.

### Components and current state
1. **SpeakerEncoder** (ECAPA-TDNN, small) — currently candle CPU; already
   known-precision-fragile on Metal, so keep on CPU. Not a migration target
   by itself; just ensure it can feed its output into the new pipeline.

2. **Talker** — 28-layer Qwen3-style decoder with QK-norm.
   **Can reuse `LlamaFamilyModel<B>`** as-is, just need different
   `LlamaFamilyConfig` constructor that matches Qwen3-TTS talker config.
   Output: per-step hidden state + codec token logits.

3. **SubTalker** — small 5-layer decoder that maps Talker hidden to
   codec tokens (multi-codebook). Also Llama-family-ish; can probably
   share infrastructure.

4. **Vocoder** — 8-layer transformer (LayerScale'd attention + MLP) +
   flow-matching ODE solver. The transformer is Llama-family-like but
   has `attn_layer_scale` and `mlp_layer_scale`. **Need model code to
   apply scales.** `LlamaFamilyModel` doesn't.

5. **Speech tokenizer encoder** — CPU currently.

### Backend gaps
- **Layer-scale attention** (Vocoder transformer): gate the attention
  and MLP outputs through a learnable scale vector before residual.
  One extra `add_bias`-like op (actually a broadcast multiply):
  `fused_scale_add(a, b, scale, out, len, scale_len)` — already exists as
  `fused_scale_add_enc` in pipelines; just needs a Backend trait method
  to expose it (and a CPU fallback).

- **Flow-matching ODE solver** — model-level logic, no new Backend ops.

### Approach
1. Add `Backend::fused_scale_add` trait method (expose existing shader).
2. Write `Qwen3TtsTalker<B>` (extends `LlamaFamilyModel<B>` pattern;
   outputs multi-codebook logits instead of single-vocab).
3. Write `Qwen3TtsSubTalker<B>` — similar but smaller.
4. Write `Qwen3TtsVocoder<B>` — transformer with layer-scale.
5. Orchestrator `Qwen3TtsPipeline<B>` implementing `TtsModel` trait.

### Validation
- Voice clone is the hardest. `~/ferrum-test-data/py_clone_5s.wav` has a
  Python reference. Metric: ASR the output, compare text match.
- Past-hidden at step 0 has a known Python value
  `[6.198, 2.600, -3.834, 3.134, 3.208]` — parity at model level can be
  checked there (see `fused-transformer-status.md`).

### Risk
Voice clone already has documented fragility on Metal — decoder sampling
diverges from Python. The new migration should match current behaviour,
not try to fix that drift (that's orthogonal).

---

## Suggested order and sessions

| Session | Work | Hours | Verify via |
|---------|------|:-----:|-----------|
| next    | D.5c Bert (pilot of non-decoder M-as-C) | 2-3 | bert-base-chinese cosine parity |
| +1      | D.5d Clip (text + vision, shares Bert code) | 2-3 | chinese-clip text + image match |
| +2 / +3 | D.5e Whisper (biggest; timestamp rules) | 4-6 | ferrum-test-data ASR |
| +4 / +5 | D.6 TTS Talker + SubTalker (reuse LlamaFamily) | 3-4 | talker parity at past-hidden step 0 |
| +6 / +7 | D.6 Vocoder + pipeline | 3-4 | full voice clone + ASR loop |

Each session stands alone: migrate one model, parity-verify, delete the
candle wrapper it replaces, commit. No cross-session invariants beyond
"Backend trait doesn't regress."

## Performance gates — mandatory, no regression

Migrating Bert / Clip / Whisper / TTS to Model-as-Code **must not regress
latency or quality** versus the current candle-based executors. Before
flipping any default, the following numbers have to be re-measured and
compared against their current baseline:

### Whisper ASR (`ferrum transcribe whisper-turbo`)

- **Baseline** (from CLAUDE.md, 5-min Chinese audio, M-series Mac, Metal release):
  - Total time: **~72 s**
  - Python CPU (torch) for the same audio: 107 s
  - Rust Metal must stay **≤ 80 s** (10% margin).
- **Command to re-bench**:
  ```bash
  time target/release/ferrum transcribe whisper-turbo \
    <5-min-zh-audio.wav> -l zh
  ```
- **Quality check**: ASR transcript must match pre-migration output
  character-for-character (or within ~1% CER drift from known Metal
  float32 matmul accumulation differences — see
  `metal-performance-status.md` for the pre-existing "核销" vs "和销"
  note; this is a hardware-level issue, not a migration target).

### Qwen3-TTS synthesis (`ferrum tts qwen3-tts "..."`)

- **Baseline bench targets** (Mac, Metal, 5-min output from ~100-char input):
  - Measure before any migration code lands:
    ```bash
    time target/release/ferrum tts qwen3-tts \
      "你好,欢迎使用语音合成系统,今天天气真不错,适合出门散步" \
      -o /tmp/tts_baseline.wav
    ```
  - Record: total wall time, audio duration, wall/audio ratio.
  - **Migration must stay within 10% of this baseline.**
- **Quality check**: ASR the generated wav through `whisper-turbo`; must
  transcribe back to the input text.

### Voice clone (`ferrum tts qwen3-tts --ref-audio ...`)

- **Baseline command** (from `fused-transformer-status.md`):
  ```bash
  target/release/ferrum tts qwen3-tts \
    "你好欢迎使用语音合成系统今天天气真不错适合出门散步" \
    --ref-audio ~/ferrum-test-data/female_ref_5s.wav \
    --ref-text "对你协调的结果不认同你不会说表示继续去沟通我要求你" \
    -o /tmp/rust_clone.wav
  ```
- **Quality check**:
  - ASR the output; text must match input.
  - Voice similarity via speaker-embedding cosine vs the reference audio
    (currently documented to drift from Python — migration should
    *preserve* the current behaviour, not introduce new drift).
- **Latency**: re-measure and stay within 10%.

### Process for each of D.5c / D.5d / D.5e / D.6

Before the migration:
  1. Run all three baseline benchmarks above, save numbers.
  2. Save the produced `.wav` files as reference.
  3. Commit a `docs/perf-baseline-YYYYMMDD.md` with the numbers.

During the migration:
  4. Hide the new path behind an env flag (`FERRUM_WHISPER_NEW=1`, etc.)
     so the baseline path stays callable.
  5. Only when the new path matches quality AND performance within 10%,
     flip the default and delete the old path.

This is non-negotiable: working features with real users
(ASR / voice clone / TTS) take precedence over architectural purity. If
a migration fails the perf gate, the old path stays until the kernel
gap is closed.

## What blocks here

None of the above is blocked on the LLM architecture; they are blocked on
*verification environments* — test audio, Python reference values, and
end-to-end smoke checks. Do them when those environments are set up.
