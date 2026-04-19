# Phase F — Qwen3-TTS Model-as-Code port

## Why

CUDA voice-clone produces YouTube-outro garbage (`"感谢观看我是超"` instead of
target text) + runs at RTF **45x** (vs Metal's 3x).

Root cause: `ferrum-attention::FusedTransformer` has only Metal + CPU
backends. The CUDA module at `crates/ferrum-attention/src/cuda/transformer.rs`
is a **stub** (line 125: `TODO: Implement cuda_layer_forward()`) and
`fused_attention` on CUDA (mod.rs line 16-20) falls back to CPU. So on a
CUDA box with `--features cuda` (no metal), TTS runs the Talker on:

- Linux CPU path → naive O(n³) fp64-accumulating matmul
  (`cpu/transformer.rs:170-178`) because Accelerate is macOS-only

Through 20 decoder layers × 128+ decode steps, the accumulated fp64
rounding + different kernel ordering from Metal's fp32 SGEMM diverges
enough to pick wrong codec tokens, landing in the untrained / low-probability
region of the vocabulary — which maps to YouTube-outro training data
memorized during pretraining.

## What we are porting

`crates/ferrum-models/src/architectures/qwen3_tts.rs`:

1. **Qwen3TTSTalker** — 20-layer Qwen3 LM, hidden=1024, heads=16/2, head_dim=64.
   Currently uses `ferrum_attention::FusedTransformer` + candle embeddings.
2. **SubTalker** (code predictor) — 5-layer Qwen3-style transformer.
   Same shape, same port pattern.

We keep (no changes needed):
- Vocoder (`qwen3_tts_vocoder.rs`) — uses candle `Conv1d`/`ConvTranspose1d`.
  Vocoder doesn't generate codes; it renders them. If codes are right,
  vocoder output is right regardless of numeric drift.
- Speaker encoder (`speaker_encoder.rs`) — CPU-only path, runs once per
  clone. Doesn't cause the divergence.
- Speech-tokenizer encoder (`speech_tokenizer_encoder.rs`) — CPU, one-shot.

## Target shape

Mirror `LlamaFamilyModel<B>` from `llama_family.rs`:

```rust
pub struct Qwen3TtsTalker<B: Backend> {
    // Embeddings
    text_embed: EmbeddingTable<B>,      // [text_vocab, text_hidden]
    codec_embed: EmbeddingTable<B>,     // [vocab, hidden]
    text_proj: DenseLinear<B>,          // [text_hidden -> hidden]
    // Transformer
    layers: Vec<Qwen3TtsLayer<B>>,
    final_norm: B::Buffer,
    // Output head
    codec_head: DenseLinear<B>,         // [hidden -> vocab]
    // KV cache
    kv_cache: HashMap<String, Vec<KvCacheLayer<B>>>,
    scratch: Scratch<B>,
    cfg: TalkerConfig,
}
```

Forward uses:
- `B::embedding_lookup` for text/codec embeddings
- `B::gemm` for text_projection (2-layer MLP), codec_head, all proj weights
- `B::rms_norm` / `B::fused_add_rms_norm` for norms
- `B::split_qkv` + `B::qk_norm_rope` + `B::kv_cache_append_head_major` for attention setup
- `B::flash_attention` for the attention kernel
- `B::fused_silu_mul_split` + `B::add_inplace` for MLP

All Backend methods are already in `traits.rs`.

## Weight loading

Reuse `NativeSafetensorsLoader` from ferrum-models. TTS weights live in
`talker/` and `talker/model/` subspaces inside the safetensors files —
need to extend the loader with prefix handling or use direct tensor
reads.

`Qwen3TTSTalker::load_backend(cfg, repo_path)` → reads safetensors →
builds `Qwen3TtsTalker<B>`. No candle in the hot path.

## Integration

`TtsModelExecutor` currently instantiates `Qwen3TTSTalker::load()`. Add
a generic `Qwen3TtsTalker<B>` variant, select at runtime based on
`--backend`:

- `cuda` → `Qwen3TtsTalker<CudaBackend>`
- `metal` → `Qwen3TtsTalker<MetalBackend>`
- `cpu` → `Qwen3TtsTalker<CpuBackend>`

Keep the candle-based `Qwen3TTSTalker` as `legacy` for precision testing
until the ported version matches bit-for-bit against a Metal baseline.

## Order of implementation

1. **Skeleton** — `qwen3_tts_backend.rs` with struct + constructor stub.
2. **Weight loading** — wire `NativeSafetensorsLoader` for talker prefix.
3. **Forward (text-only)** — prefill + decode producing first codebook.
4. **SubTalker port** — same pattern.
5. **Integration in executor** — dispatch based on backend.
6. **CUDA smoke test** — verify voice-clone produces target text.
7. **Metal parity test** — CPU vs Metal vs CUDA cosine ≥ 0.999 on first
   N hidden states.

## Known risks

- **f16 vs f32 precision**: Backend<CudaBackend> uses f16 internally
  (CudaSlice<f16>). LlamaFamilyModel works fine in f16 for LLM argmax
  decoding, so TTS codec-argmax should also work. If not, add a f32
  mode toggle.
- **Text embedding size mismatch**: text_hidden=2048 ≠ hidden=1024, need
  projection (2-layer MLP with GELU). Backend has `gelu` method already.
- **Different positional offset per sequence**: talker's KV cache is
  per-sequence, need per-sequence cache indexing like LlamaFamilyModel.
