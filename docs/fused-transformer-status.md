# Fused Transformer Implementation Status

## 当前卡点

**TTS 运行超时**（>60s 无输出）。需要定位是哪一步卡住：
- Speech tokenizer encoder (candle Metal, 63帧×32 codebooks)
- Prefill (73 tokens × 28 layers, 新的 fused path)
- Decode loop

**部分计时数据收集到**（进程被管道 SIGPIPE 杀死，但前几步数据在）：

```
Step 2 (speaker embed):    36.8ms  ✓ 正常
Step 3 (speech tokenizer): 587.1ms ✓ 正常
Steps 4-5 (tokenize+prompt): 89.7ms ✓ 正常
Prefill (28 layers fused): ??? — 进程被杀，未出数据
```

**卡点确认在 Prefill** — 28 层 fused transformer forward。每层 4 次 Metal cmd.commit()+wait() = 112 次 GPU sync。

## 架构

```
ferrum-attention/
├── src/
│   ├── lib.rs          # FusedTransformer — N层 transformer，自动选 Metal/CPU
│   ├── cpu/
│   │   ├── mod.rs      # Accelerate sgemm fused attention
│   │   └── transformer.rs  # CPU transformer layer (Accelerate sgemm + fused softmax)
│   └── metal/
│       ├── mod.rs
│       ├── pipelines.rs    # Metal pipeline cache + dispatch (GEMM 改用 Accelerate on shared buf)
│       ├── transformer.rs  # Metal transformer layer forward
│       └── shaders/
│           ├── flash_attn.metal       # Fused flash attention (tested, 1.68ms/prefill)
│           └── transformer_ops.metal  # RMSNorm, SiLU, add (tested), GEMM (8x8, 太慢已弃用)
└── tests/
    ├── metal_test.rs       # 4 tests pass (flash attention correctness + perf)
    └── transformer_test.rs # 1 test pass (single layer prefill + decode)
```

## Metal Transformer Layer 数据流

```
每层 forward (metal_layer_forward):

1. RMSNorm        → Metal kernel, 1 dispatch
2. Q/K/V proj     → cblas_sgemm on shared buffer (零拷贝，同步)
3. QK-norm + RoPE → CPU (element-wise)
4. KV cache       → CPU Vec append
5. Flash Attention → Metal kernel, 1 dispatch
6. Untranspose    → CPU
7. O proj         → cblas_sgemm
8. Residual add   → Metal kernel  ┐
9. Post RMSNorm   → Metal kernel  ┘ 1 dispatch
10. Gate/Up proj  → cblas_sgemm (×2)
11. SiLU×gate     → Metal kernel, 1 dispatch
12. Down proj     → cblas_sgemm
13. Residual add  → Metal kernel, 1 dispatch

总计：每层 4 次 Metal dispatch + 5 次 sgemm 同步调用
28 层 = 112 Metal dispatch + 140 sgemm
```

## 性能问题

1. **Metal GEMM 8x8 kernel 太慢** — 已替换为 Accelerate cblas_sgemm on shared buffers
2. **每层 4 次 `cmd.commit(); wait()`** — GPU sync 开销。可优化为批量提交
3. **Speech tokenizer encoder** — 原 candle Metal 代码，63帧×32 codebooks，可能也很慢
4. **管道截断问题** — `| grep` 导致进程被 SIGPIPE，看不到日志

## 已验证的精度数据

| 配置 | past_hidden 差异 vs Python |
|------|---------------------------|
| candle 原始 (Metal) | 2.0-5.0 |
| fused attention only (CPU) | 0.1-0.6 |
| fused transformer (Metal+Accel) | 未测到 |

## 下一步

1. **定位卡点**：重定向日志到文件，找出哪步耗时最长
2. **减少 Metal sync**：合并 Metal dispatches，或改为 CPU-only path 先验证正确性
3. **验证 ASR 输出**
4. **优化 Metal GEMM**：移植 llama.cpp 的 `kernel_mul_mm`（64×32 tiles, 4 simdgroups）或用 MPS
5. **Metal QK-norm + RoPE kernel**：消除 CPU 来回

## 参考代码

**Python baseline (Qwen3-TTS 官方实现):**
- 路径: `~/Qwen3-TTS/`
- 虚拟环境: `~/Qwen3-TTS/.venv/` (用 `cd ~/Qwen3-TTS && source .venv/bin/activate`)
- 关键文件: `qwen_tts/core/models/modeling_qwen3_tts.py`
  - `Qwen3TTSTalkerAttention` (line 727) — attention 实现
  - `Qwen3TTSTalkerForConditionalGeneration.forward` (line 1664) — decode loop
  - `generate_icl_prompt` (line 1968) — ICL prompt 构造
  - `apply_multimodal_rotary_pos_emb` (line 660) — RoPE
  - `eager_attention_forward` (line 634) — attention with softmax
- 运行 Python voice clone:
  ```bash
  cd ~/Qwen3-TTS && source .venv/bin/activate
  python3 -c "
  import torch, soundfile as sf, sys; sys.path.insert(0, '.')
  from qwen_tts import Qwen3TTSModel
  tts = Qwen3TTSModel.from_pretrained('Qwen/Qwen3-TTS-12Hz-0.6B-Base', device_map='cpu', dtype=torch.float32)
  wavs, sr = tts.generate_voice_clone(
      text='你好欢迎使用语音合成系统今天天气真不错适合出门散步',
      language='Chinese',
      ref_audio='$HOME/ferrum-test-data/female_ref_5s.wav',
      ref_text='对你协调的结果不认同你不会说表示继续去沟通我要求你',
      max_new_tokens=2048, do_sample=True, top_k=50, temperature=0.9, repetition_penalty=1.05,
      subtalker_dosample=True, subtalker_top_k=50, subtalker_temperature=0.9,
  )
  sf.write('/tmp/py_clone.wav', wavs[0], sr)
  print(f'Done: {len(wavs[0])/sr:.2f}s')
  "
  ```

**llama.cpp Metal 参考实现:**
- 路径: `~/rust_ws/llama.cpp/`
- Metal shader: `ggml/src/ggml-metal/ggml-metal.metal` (10549 行)
  - `kernel_flash_attn_ext_impl` (line 5758) — fused flash attention, simdgroup_multiply_accumulate
  - `kernel_mul_mm` (line 9267) — 高效 GEMM, 64×32 tiles, 4 simdgroups, 128 threads
  - `kernel_soft_max` (line 1846) — fused softmax with simd_sum
  - `kernel_rms_norm_fuse_impl` (line 2981) — fused RMSNorm with simd_sum
  - `kernel_swiglu_f32` (line 1457) — SiLU × gate

## 测试数据

**参考音频 (5 秒女声客服录音):**
- 路径: `~/ferrum-test-data/female_ref_5s.wav`
- 内容 (ASR): "对你协调的结果不认同你不会说表示继续去沟通我要求你"
- 24kHz, ~120000 samples

**完整参考音频 (20 秒):**
- 路径: `~/ferrum-test-data/female_ref_24k.wav`

**Python 预计算数据 (用于精度对比):**
- `~/ferrum-test-data/py_talker_input_embeds.npy` — Python prefill embedding [1, 73, 1024]
- `~/ferrum-test-data/py_ref_codes_5s.npy` — Python codec tokens [63, 16]
- `~/ferrum-test-data/py_ref_codes_5s.bin` — 同上, flat u32 binary (FERRUM_REF_CODES 用)
- `~/ferrum-test-data/py_spk_embed.npy` — Python speaker embedding [1024]
- `~/ferrum-test-data/py_clone_greedy.wav` — Python greedy voice clone 输出 (ASR: 正确)
- `~/ferrum-test-data/py_clone_5s.wav` — Python ICL voice clone 输出

## 验证方法

### 1. 快速功能验证 (TTS → ASR)
```bash
# Rust voice clone
RUST_LOG=warn target/release/ferrum tts qwen3-tts \
  "你好欢迎使用语音合成系统今天天气真不错适合出门散步" \
  --ref-audio ~/ferrum-test-data/female_ref_5s.wav \
  --ref-text "对你协调的结果不认同你不会说表示继续去沟通我要求你" \
  -o /tmp/rust_clone.wav 2>&1 > /tmp/tts_log.txt

# ASR 验证
target/release/ferrum transcribe whisper-turbo /tmp/rust_clone.wav -l zh

# 期望输出包含: "你好欢迎使用语音合成系统今天天气真不错适合出门散步"
# Python baseline ASR: "你好,欢迎使用语音合成系统,今天天气真不错,适合出门散步"
```

### 2. 精度对比 (past_hidden)
```bash
# 在 tts_executor.rs 里有 "step 0 past_hidden first 5" 日志
# Python 的 past_hidden (step 0):
#   [6.198, 2.600, -3.834, 3.134, 3.208]
#
# 差异阈值:
#   < 0.01 per element = SubTalker 应该能出正确 codebook
#   < 0.1 = 可能对，需要 ASR 验证
#   > 0.5 = 肯定不对
```

### 3. SubTalker 隔离测试
```bash
# 用 Python 的 exact past_hidden 喂 Rust SubTalker
# 如果 SubTalker 输出正确 = 问题在 main talker 精度
# 如果仍然错 = SubTalker 有 bug
FERRUM_PY_HIDDEN=/tmp/py_past_hidden_step0.npy \
FERRUM_PY_CODEC_EMBED=/tmp/py_first_codec_embed.npy \
RUST_LOG=info target/release/ferrum tts qwen3-tts ...
# 看日志 [ST-i*] token=xxx，对比 Python codec_ids
```

### 4. 性能 benchmark
```bash
# 计时日志在 tts_executor.rs
# 期望:
#   Speaker embed: <50ms
#   Speech tokenizer: <1s
#   Prefill (73 tokens, 28 layers): <500ms (Metal) / <2s (CPU)
#   Decode per step: <50ms (Metal) / <200ms (CPU)
#   Total TTS (5s audio): <15s
```

### 5. 单元测试
```bash
# ferrum-attention 全部测试
cargo test -p ferrum-attention --features metal -- --nocapture

# 期望: 7 tests pass
# - 2 CPU attention tests
# - 4 Metal flash attention tests (correctness + perf)
# - 1 Metal transformer layer test (prefill + decode)
```

## 关键文件

- `crates/ferrum-attention/src/metal/transformer.rs` — Metal 层 forward 逻辑
- `crates/ferrum-attention/src/metal/pipelines.rs` — Metal dispatch 和 GEMM（Accelerate）
- `crates/ferrum-models/src/architectures/qwen3_tts.rs` — Talker forward_step 集成
- `crates/ferrum-models/src/executor/tts_executor.rs` — TTS 执行器（有计时日志）
- `~/rust_ws/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal` — 参考实现（kernel_mul_mm, kernel_flash_attn_ext）
