# Qwen3-TTS Voice Clone Implementation Status

## Overview

Branch: `feat/whisper-asr`
Model: `Qwen/Qwen3-TTS-12Hz-0.6B-Base` (0.6B, safetensors, Apache 2.0)
Python baseline: `~/Qwen3-TTS/` (official repo, uv venv)
Test data: `~/ferrum-test-data/`

## What Works

### 1. Speaker Encoder (ECAPA-TDNN)
- **File**: `crates/ferrum-models/src/architectures/speaker_encoder.rs`
- Mel spectrogram (24kHz, n_fft=1024, hop=256, 128 mels) + ECAPA-TDNN → 1024-dim x-vector
- Weights load from `model.safetensors` under `speaker_encoder.*`
- Output matches Python to f32 precision

### 2. Speech Tokenizer Encoder (Mimi-based)
- **File**: `crates/ferrum-models/src/architectures/speech_tokenizer_encoder.rs`
- 15-layer CausalConv stack (960x downsample) + 8-layer Transformer + 2x Downsample + Split RVQ
- Conv output matches Python exactly (1e-5 precision)
- Transformer output matches within 1e-3
- **Codebook 0 matches Python perfectly** (e.g., frame 0 = 404, frame 1 = 394, frame 2 = 935)
- Codebook 1-15 diverge due to RVQ residual chain sensitivity to transformer precision

### 3. ICL Prompt Construction
- Dual-stream text+codec summing matches Python
- Chat template tokenization (`<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n`)
- Codec prefix with language ID + speaker embedding injection
- Prefill shape matches: `[1, 73, 1024]` for 5s reference audio
- **All 73 prefill positions match Python to f32 precision** (when using Python's codec tokens)

### 4. Decode Loop Structure
- Token suppression: [vocab_size-1024, vocab_size) except EOS
- Greedy token 0 = 1566, matches Python exactly
- Greedy token 1 = 2007, matches Python exactly
- SubTalker runs with per-codebook lm_heads + codec_embeddings

### 5. Bug Fixes Applied
- `hidden` variable shadowing in decode loop (was `let hidden =`, should be `hidden =`)
- MAX_CODEC_TOKENS 4096 → 2000 (prevent OOM from vocoder upsampling)
- RVQ SplitQuantizer: semantic and acoustic get SAME hidden (not residual between them)
- Missing downsample CausalConv1d(512→512, k=4, s=2) between transformer and RVQ
- CausalConv extra right padding for ceil output length
- Trim conv output from start (not end)
- Speaker encoder reflect_pad contiguous() for CPU compatibility

### 6. CLI
- `--ref-audio` and `--ref-text` arguments on `tts` command
- `FERRUM_REF_CODES=/path/to/codes.bin` env var for pre-computed codec tokens (debug)

## The Problem

**TTS voice clone output is unintelligible.** ASR on generated audio produces garbage text.

### Root Cause Chain

```
Layer 0 input (layernorm output)     ← MATCHES Python exactly
         ↓
Layer 0 attention output             ← DIFFERS by 0.001
         ↓ (28 layers)
Final hidden state (after norm)      ← DIFFERS by 0.5-1.0
         ↓                              Python: [9.320, 3.522, -4.352]
         ↓                              Rust:   [9.903, 2.946, -4.882]
         ↓
SubTalker input (past_hidden)        ← Same 0.5-1.0 difference
         ↓
SubTalker codebook tokens            ← COMPLETELY DIFFERENT
         ↓                              Python: [1566, 1800, 106, 1546, ...]
         ↓                              Rust:   [1566, 1993, 178,   92, ...]
         ↓
Combined embedding (sum of 16 codebook embeddings + tts_pad)
         ↓                           ← VERY DIFFERENT
         ↓
Next decode step input               ← Wrong → cascading failure
```

### Key Observations

1. **Not a Metal issue** — CPU produces identical results to Metal
2. **Not a config issue** — head_dim=128, num_heads=16, num_kv_heads=8 all correct
3. **Not a weight loading issue** — layernorm output matches exactly
4. **Not a prompt issue** — all 73 prefill positions match (with Python codec tokens)
5. **Not a MRoPE issue** — for same position_ids, MRoPE ≡ 1D RoPE (verified mathematically)
6. **Divergence starts at Layer 0 attention** — input matches, output differs by 0.001

## Hypothesis

The attention computation in `qwen3_tts.rs` differs numerically from Python's `Qwen3TTSTalkerAttention` at the f32 level. Possible causes:

### H1: Manual softmax implementation order
Rust:
```rust
let max = attn.max_keepdim(D::Minus1)?;
let shifted = attn.broadcast_sub(&max)?;
let exp = shifted.exp()?;
let sum = exp.sum_keepdim(D::Minus1)?;
exp.broadcast_div(&sum)?
```
vs PyTorch's fused `F.softmax`. While mathematically identical, the intermediate rounding in f32 differs because PyTorch may use a different accumulation order for max/sum.

### H2: QK Norm application
Rust applies ManualRmsNorm to Q/K after reshape to [B, H, T, 128]. Python uses `Qwen3TTSRMSNorm`. Both should produce the same result but the computation graph differs.

### H3: Attention matmul precision
`q.matmul(&k.transpose(2, 3)?)` — the dot product of 128-dim vectors accumulates f32 rounding errors. PyTorch may use a different BLAS routine (possibly with FMA or different blocking) than candle.

### H4: candle rope_slow vs Python rotate_half
The RoPE application might differ in how `rotate_half` handles the split. `rope_slow` splits at dim/2, while Python's `rotate_half` does `cat(-x[..., dim//2:], x[..., :dim//2])`.

## Debug Method

### Verification approach (don't guess, dump and compare)

1. **Layer-by-layer dump**: Compare each layer's output at the last position
2. **Sub-layer dump**: Within layer 0, compare after each sub-operation:
   - After input_layernorm ← DONE (matches)
   - After Q/K/V projection ← NEXT
   - After QK norm
   - After RoPE
   - After attention matmul (pre-softmax)
   - After softmax
   - After V matmul + output projection
   - After residual + MLP
3. **Find exact divergence point** between consecutive operations

### Python comparison script pattern
```python
# In ~/Qwen3-TTS/
import torch, numpy as np
from qwen_tts import Qwen3TTSModel

tts = Qwen3TTSModel.from_pretrained('Qwen/Qwen3-TTS-12Hz-0.6B-Base', device_map='cpu', dtype=torch.float32)
ie = np.load('~/ferrum-test-data/py_talker_input_embeds.npy')
ie_tensor = torch.from_numpy(ie).float()

m = tts.model.talker.model
layer0 = m.layers[0]
# ... dump intermediate values
```

### Rust comparison pattern
```rust
// In qwen3_tts.rs Attention::forward, add:
if pos_offset == 0 && x.dim(1).unwrap_or(0) > 1 {
    // Dump q/k after projection, after norm, after rope, etc.
    // Compare with Python at last position
}
```

## Test Files

| File | Purpose |
|------|---------|
| `~/ferrum-test-data/female_ref_5s.wav` | 5s reference audio (female customer) |
| `~/ferrum-test-data/female_ref_24k.wav` | 20s reference audio (24kHz) |
| `~/ferrum-test-data/py_ref_codes_5s.npy` | Python codec tokens for 5s audio (63x16) |
| `~/ferrum-test-data/py_ref_codes_5s.bin` | Same as above, flat u32 binary for Rust |
| `~/ferrum-test-data/py_talker_input_embeds.npy` | Python prefill embeddings (1x73x1024) |
| `~/ferrum-test-data/py_trailing_text_hidden.npy` | Python trailing text (1x1x1024) |
| `~/ferrum-test-data/py_spk_embed.npy` | Python speaker embedding (1024) |
| `~/ferrum-test-data/py_mel.npy` | Python mel spectrogram (1x1875x128) |
| `~/ferrum-test-data/py_clone_5s.wav` | Python ICL voice clone output (working) |
| `~/ferrum-test-data/py_clone_greedy.wav` | Python greedy output (ASR: correct text) |
| `~/ferrum-test-data/mel_filters_spkenc.bin` | Mel filterbank (128x513, librosa slaney) |

## Architecture Notes

### Talker Config (from config.json)
```
hidden_size: 1024, intermediate_size: 3072, num_hidden_layers: 28
num_attention_heads: 16, num_key_value_heads: 8, head_dim: 128
rope_theta: 1e6, rms_norm_eps: 1e-6
rope_scaling: {interleaved: true, mrope_section: [24, 20, 20], type: default}
Q proj: [2048, 1024], K proj: [1024, 1024], O proj: [1024, 2048]
```

### MRoPE vs 1D RoPE Equivalence (for TTS)
For pure TTS (no vision), all 3 position_ids are identical. With interleaved=true and mrope_section=[24,20,20]:
- `apply_interleaved_rope(cos[0..3], 3)` copies same values when all modalities match
- Result: cos_final = cos[0] = standard 1D cos
- **Verified**: mathematically equivalent to 1D RoPE for same positions

### Python SubTalker (code_predictor)
- Uses HF generate internally
- Per-codebook lm_heads: `lm_head.{0-14}` (ModuleList of 15 Linear [2048, 1024])
- Per-codebook embeddings: `model.codec_embedding.{0-14}`
- `small_to_mtp_projection`: Identity (no-op for 0.6B model)
- `generation_steps` tracked across forward calls via `_update_model_kwargs_for_generation`

### Python Talker forward (decode step)
```python
# input_ids = [last_sampled_codec_token]
last_id_hidden = codec_embedding(input_ids)        # [1, 1, 1024]
# Run SubTalker with previous hidden state
predictor_result = code_predictor.generate(
    inputs_embeds=cat(past_hidden, last_id_hidden), # [1, 2, 1024]
    max_new_tokens=15,
)
# Sum all 16 codebook embeddings
codec_hiddens = cat([last_id_hidden] + [sub_embeds...])  # [1, 16, 1024]
inputs_embeds = codec_hiddens.sum(1, keepdim=True)       # [1, 1, 1024]
# Add trailing text component
inputs_embeds = inputs_embeds + trailing_text_hidden[:, gen_step]
# Forward through transformer with KV cache
outputs = self.model(inputs_embeds=inputs_embeds, past_key_values=...)
# Return: logits, past_hidden=hidden_states[:,-1:], generation_step+1
```

## Next Steps

1. **Dump Q/K after projection** in both Python and Rust layer 0 → find if projection diverges
2. **Dump after QK norm** → find if norm is the source
3. **Dump attention scores (pre-softmax)** → find if matmul accumulation differs
4. **Dump after softmax** → find if softmax implementation differs
5. Once divergence point found:
   - If softmax: try candle's built-in softmax or reorder accumulation
   - If matmul: try contiguous() or different precision
   - If QK norm: check weight dimensions match exactly
6. **Alternative**: if divergence is inherent to f32, make SubTalker more robust:
   - Use temperature>0 for SubTalker sampling (reduce sensitivity)
   - Use top-p instead of greedy for SubTalker
   - Quantize/round hidden states before SubTalker input
