# Qwen3-TTS Architecture

## Overview

端到端 TTS pipeline：文本 → codec tokens → 波形音频。支持普通合成和 ICL 声音克隆。

```
Text ──→ Talker (28L transformer) ──→ semantic token
                                          │
                            ┌──────────────┘
                            ▼
                   SubTalker (5L transformer, 15 steps)
                            │
                            ▼
                    16 codec tokens/frame
                            │
                            ▼
              Vocoder (8L pre-transformer + BigVGAN)
                            │
                            ▼
                       24kHz WAV
```

## Crate 依赖

```
ferrum-cli
  └─ ferrum-engine
       └─ ferrum-models          ← TTS model 架构 + executor
            ├─ ferrum-attention   ← FusedTransformer (Metal/CPU)
            ├─ candle-core/nn/transformers  ← 权重加载 + speech tokenizer
            └─ rubato            ← sinc resample
```

## 文件结构 (7083 行)

| 文件 | 行数 | 职责 |
|------|------|------|
| `tts_executor.rs` | 1453 | Pipeline 编排：加载、合成、声音克隆 |
| `qwen3_tts.rs` | 1256 | Talker (28L) + SubTalker (5L) 模型 |
| `qwen3_tts_vocoder.rs` | 933 | Vocoder: RVQ decode → pre-transformer → upsampler → BigVGAN |
| `speaker_encoder.rs` | 657 | ECAPA-TDNN 说话人编码器 |
| `speech_tokenizer_encoder.rs` | 165 | Mimi 语音编码器 (candle 组件) |
| `audio_processor.rs` | 314 | 音频加载 + rubato sinc resample |
| `ferrum-attention/lib.rs` | 833 | FusedTransformer: Metal/CPU 双路径 |
| `metal/transformer.rs` | 327 | Metal 单层 forward (13 compute encoders) |
| `metal/pipelines.rs` | 761 | Metal kernel 注册 + helper functions |
| `cpu/transformer.rs` | 384 | CPU 单层 forward (Accelerate cblas) |

## 模型组件

### Talker (热路径)

28 层 Qwen3 transformer。0.6B: hidden=1024, 1.7B: hidden=2048。

**Forward 路径**：FusedTransformer (Metal GEMM 64x32 tiles + flash attention + GPU RMSNorm)

每步 decode ~126ms (1.7B)：
- Metal command buffer: 1 次 commit+wait
- 28 层 × 12 kernel dispatches = 336 dispatches/step
- GPU-side final RMSNorm

### SubTalker (热路径, 69% 耗时)

5 层 transformer，每帧 15 步 autoregressive（预测 acoustic codebooks 1-15）。

**Forward 路径**：FusedTransformer (同 Talker 但 hidden=1024, 5L)

每帧 ~287ms = 15 步 × ~19ms/步：
- lm_head: cblas_sgemm (CPU, [1,1024]×[2048,1024]^T)
- argmax: CPU
- embedding lookup: CPU raw f32
- projection: cblas_sgemv (CPU, 2048→1024, 仅 1.7B)
- fused forward: Metal (5 layers)

**1.7B 特有**：`small_to_mtp_projection` 在 predict 首次 concat 后 project (candle Linear)，per-step 用 cached raw weights + cblas_sgemv。

### Vocoder (冷路径)

RVQ decode → 8 层 pre-transformer (FusedTransformer, hidden=512) → upsampler → BigVGAN decoder。

运行一次，~0.5s。

### Speaker Encoder (冷路径)

ECAPA-TDNN: Conv → 3× SE-Res2Net block → MFA → ASP → FC。

**关键修复点**：
- Res2Net 第一个 sub-block 不加前一个输出
- ASP epsilon: 1e-5
- SE residual 无 relu
- FC output dim 从 config 读 (0.6B=1024, 1.7B=2048)
- 输入 mel: [B, T, 128] (forward 内部 transpose)

运行一次，~35ms。

### Speech Tokenizer Encoder (冷路径)

**全部使用 candle-transformers mimi 组件**（正确性优先）：
- SeaNetEncoder (conv stack, 960x downsample)
- ProjectedTransformer (8 layers, sliding window)
- ConvDownsample1d (25Hz → 12.5Hz)
- SplitResidualVectorQuantizer (1 semantic + 15 acoustic codebooks)

运行一次，~400ms。

## Voice Clone ICL 流程

```
ref_audio ──→ [rubato sinc resample 24kHz]
                    │
         ┌──────────┴──────────┐
         ▼                     ▼
  Speaker Encoder       Speech Tokenizer
  (ECAPA-TDNN)          (Mimi encoder)
         │                     │
    spk_embed [2048]      ref_codes [44, 16]
         │                     │
         └──────┬──────────────┘
                ▼
        Build ICL Prompt
                │
    ┌───────────┴───────────┐
    ▼                       ▼
  Prefill (9 pos)     ICL Block (45 pos)
  [role + codec_prefix    [text + ref_codec
   + spk_embed]            element-wise sum]
    │                       │
    └───────┬───────────────┘
            ▼
     Talker forward_step × 2
     (KV cache 连续)
            │
            ▼
      Decode Loop (max ~126 steps)
      ┌─────────────────────────┐
      │ logits → suppress →     │
      │ rep_penalty → sample →  │
      │ SubTalker predict →     │
      │ combined_embed →        │
      │ trailing text/pad →     │
      │ Talker forward_step     │
      └─────────────────────────┘
            │
            ▼
    [ref_codes + gen_codes] → Vocoder → WAV
```

## Metal 内核

| Shader 文件 | 内核 | 用途 |
|-------------|------|------|
| `gemm_f32.metal` | `gemm_f32_v2` | 64×32 tile GEMM (simdgroup_multiply_accumulate) |
| `flash_attn.metal` | `flash_attn_f32` | Flash attention with GQA |
| `norm_rope.metal` | `qk_norm_rope_transpose_f32` | Fused QK-norm + RoPE + transpose (3 modes) |
| `transformer_ops.metal` | `rms_norm_f32` | RMSNorm (simd reduction) |
| | `silu_mul_f32` | SiLU × gate |
| | `add_f32` | Elementwise add |
| | `fused_residual_norm_f32` | Residual + scale + norm (3→1 dispatch) |
| | `fused_scale_add_f32` | Scale + add (2→1 dispatch) |
| | `argmax_f32` | Simd argmax reduction |
| | `embedding_lookup_f32` | Single-index table lookup |
| `softmax.metal` | `softmax_last_dim_f32` | Softmax (simd reduction) |

## 性能 (Apple Silicon, 1.7B, voice clone)

| 阶段 | 耗时 | 后端 |
|------|------|------|
| Model load | ~3s | mmap safetensors |
| Speaker encoder | 36ms | candle (Metal) |
| Speech tokenizer | 400ms | candle mimi (CPU) |
| ICL prefill (9 pos) | ~50ms | FusedTransformer (Metal) |
| ICL block (45 pos) | ~180ms | FusedTransformer (Metal) |
| **Decode (70 steps)** | **~20s** | FusedTransformer (Metal) + cblas |
| Vocoder | ~0.5s | FusedTransformer (Metal) + candle |
| **Total** | **~25s** | RTF 4.4x |

vs 参考 Rust 项目 (CPU): 88s, RTF 15.3x → **我们快 3.4x**

## Sampling 参数

| 参数 | Normal TTS | Voice Clone (ICL) |
|------|-----------|-------------------|
| temperature | 0.9 | 0.9 |
| top_k | 50 | 50 |
| top_p | 0.9 | 0.9 |
| repetition_penalty | 1.05 | 1.5 |
| min_new_tokens | 0 | 2 |
| max_tokens | 2000 | max(75, text_len×6) |
| SubTalker | temperature=0.9 | **argmax (greedy)** |
| RNG | LCG (subsec_nanos seed) | LCG (subsec_nanos seed) |
| Repetition detection | no | 3× pattern match → early stop |

## 关键设计决策

1. **FusedTransformer vs candle**：热路径用 FusedTransformer (Metal GEMM + custom kernels)，冷路径用 candle (speech tokenizer encoder)。

2. **CPU embedding + GPU forward**：SubTalker 的 embedding lookup 和 projection 在 CPU (cblas)，transformer forward 在 GPU。Apple Silicon 统一内存下 CPU cblas_sgemv 比 Metal GEMM (M=1) 更快。

3. **Sinc resample**：音频加载用 rubato sinc interpolation（不是线性插值）。线性插值导致 speech tokenizer 编码出不同的 ref_codes，voice clone 质量不稳定。

4. **LCG RNG**：使用 `subsec_nanos + counter × 1103515245` LCG（匹配参考项目）。更高质量的 PRNG (thread_rng) 反而导致 voice clone 不稳定——特定 logit 分布下 LCG 的随机数序列碰巧总能采样到好 token。

5. **2-step ICL forward**：prefill (9 pos) 和 ICL block (45 pos) 分开调用 forward_step。FusedTransformer 通过 KV cache 维持 position offset 连续性。

## 已知限制

- 1.7B 模型 CPU 推理很慢（每步 ~500ms），只适合 Metal 加速
- 中文 voice clone 对参考音频质量敏感（需要 ≥4s、清晰、无噪声）
- SubTalker 占 69% 耗时，瓶颈在 15× Metal command buffer sync
- Vocoder 和 speech tokenizer 未做 Metal 优化（冷路径，优先级低）
