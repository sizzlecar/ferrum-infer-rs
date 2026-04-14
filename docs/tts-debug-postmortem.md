# TTS Debug 复盘 — 2026-04-11/13

## 真实根因

**Vocoder（`Qwen3TTSVocoder`）缺失了 8 层 transformer。** 代码里写着 `// pre_transformer omitted for now — TODO`，然后再也没实现。

Vocoder 的完整 decode 链路：`RVQ(512) → pre_conv(512→1024) → 8层transformer(1024) → output_proj(1024→512) → upsampler → decoder → 音频`

我们跳过了 transformer，直接 `pre_conv → upsampler`。93 个权重文件被完全忽略。输出是未经语义处理的原始 codebook 向量，导致 3 倍响度、全程削波、不可懂。

**这是一个 5 分钟能定位的 bug，但花了 20+ 小时。**

---

## 时间线与错误决策

### 第一阶段：错误归因（4-11 20:00 ~ 4-12 04:00，~8 小时）

**假设**：音频垃圾 = token 不对 = transformer 精度问题

**做了什么**：
- 对比 Layer 0 每个 op 的精度（RMSNorm、Q/K/V proj、QK-norm、RoPE、softmax）
- f64 softmax、f32 softmax、vDSP_dotpr RMSNorm 各种组合
- 研究 PyTorch SDPA 源码、exp_u20() 实现
- 下载 llama.cpp 源码分析 GEMM 精度
- 比较 cblas_sgemm vs sgemm_ column-major 布局

**结果**：全部是死胡同。因为前提就错了——token 是对的，vocoder 是错的。

**应该做的**：用 Python 导出的 codec tokens 直接喂 vocoder，检查输出波形。

### 第二阶段：错上加错（4-12 04:00 ~ 4-12 12:00，~8 小时）

**假设**：精度差异来自 candle 框架 vs 自定义 ops

**做了什么**：
- 写了 All-Metal GEMM kernel（64x32 tiles, simdgroup_multiply_accumulate）
- 写了 Metal QK-norm + RoPE fused kernel
- 重构了整个 Metal transformer layer（单 command buffer）
- 改了 softmax 5 次（f32/f64/vDSP/decomposed/softmax_last_dim）
- 改了 RMSNorm 4 次（f32/f64/vDSP_dotpr/candle_nn::rms_norm）
- 改了 RoPE 2 次（rope_slow → narrow+rotate）

**结果**：这些改动本身是有价值的工程工作（性能优化），但没有解决 TTS 质量问题，因为问题不在这。

### 第三阶段：发现参考项目（4-12 12:00 ~ 4-12 16:00，~4 小时）

**转折点**：搜到 `qwen3-tts-rs` 参考项目，它用同样的 candle + Rust 能出正确音频。

**做了什么**：
- 对比参考项目代码
- 发现 prompt 结构差异 → 修复
- 发现 sampling 差异（RNG + top_p）→ 修复
- 发现 GQA repeat_kv bug → 修复
- 发现 ManualRmsNorm 公式差异 → 修复

**关键进展**：prefill input 精确匹配，step 0 argmax 精确匹配。但音频仍然不对。

### 第四阶段：移植参考项目（4-12 16:00 ~ 4-12 20:00，~4 小时）

**做了什么**：
- 完整移植参考项目的 TalkerModel + CodePredictor + Sampling（2100 行）
- token 序列 100% 匹配参考项目

**发现**：token 完全一致但音频仍然是噪声 → **vocoder 是唯一问题**。

### 第五阶段：定位 vocoder bug（4-12 20:00 ~ 20:30，30 分钟）

- 换用参考项目的 `Decoder12Hz` → 音频正确
- 对比两个 vocoder 的 `decode()` 流程 → 发现缺失 transformer
- 确认 `Qwen3TTSVocoder` line 643-650 写着 `// TODO: implement transformer`

---

## 核心反思

### 1. 没有隔离测试

整个 TTS 链路是：`text → tokenizer → talker → SubTalker → vocoder → 音频`。当音频不对时，我直接假设是 talker/SubTalker 的问题（最复杂的部分），从来没想过测 vocoder（最后一步）。

**正确做法**：从后往前隔离。
- 先拿 Python 的 codec tokens 喂 vocoder → 如果输出正确，vocoder OK → 问题在 token 生成
- 如果输出不正确 → vocoder 有 bug → 直接修 vocoder

这一步只需要 5 分钟。我甚至中途保存了 Python 的 codec tokens（`/tmp/py_codec_tokens.bin`），但从来没用它测 vocoder。

### 2. 被波形特征误导

音频波形分析早就显示：
- RMS = 6600（参考 1991）→ 3 倍响度
- max = 32767 → 全程削波
- 零静音段

这些是 **vocoder 问题的明显标志**：错误的 token 序列不会导致 3 倍响度和全程削波。正常的错误 token 会产生错误内容但正常响度的语音。全程高能量噪声意味着信号处理链有根本性缺陷。

**我看到了这些数据但没有正确解读。**

### 3. 确认偏误

当第一个假设是"精度问题"时，所有后续实验都在验证这个假设。每次实验失败时，我不是质疑假设本身，而是换一种精度优化方式继续试。

- f32 softmax 没改善 → 试 f64 softmax
- f64 softmax 没改善 → 试 vDSP_dotpr
- vDSP_dotpr 没改善 → 试改 GEMM layout
- GEMM layout 没改善 → 试改 RoPE
- RoPE 没改善 → 试改 RMSNorm
- 全都没改善 → 搜网上解决方案
- 网上说"已知问题" → 接受"精度不可解"

**正确做法**：连续 3 次实验没改善时，应该质疑假设而不是换参数。

### 4. 过度分析，缺少对照实验

花了大量时间在理论推导上（f32 累积误差模型、BLAS tiling 顺序、exp_u20 多项式近似），但一直没做最关键的对照实验：**用已知正确的 token 测 vocoder**。

前几天的 debug 文档（`voice-clone-debug-session.md`）甚至写了"过度分析不如动手验证"这条教训，但我没有吸取。

### 5. 忽略了 TODO 注释

vocoder 代码里明确写着 `// pre_transformer omitted for now — TODO`。如果一开始就审查 vocoder 的完整性，而不是假设它是对的，bug 一目了然。

---

## 有价值的产出

虽然方向错了，但过程中产出了有价值的工程成果：

1. **GQA bug 修复**：`repeat_kv` 导致 Q head 2-15 attend 到错误 KV head（真实 bug，只是不影响音频质量因为 vocoder 更严重）
2. **All-Metal GEMM kernel**：64x32 tiles, simdgroup_multiply_accumulate, 正确性验证通过
3. **All-Metal transformer layer 骨架**：单 command buffer, GPU KV cache
4. **Prompt 结构修正**：匹配参考项目的 prefill 构建
5. **采样逻辑修正**：top_p, repetition_penalty, 时间种子 RNG
6. **qwen3_tts_v2.rs**：完整移植参考项目的推理引擎（2100 行），token 序列 100% 匹配
7. **codec_v2/**：参考项目的 vocoder 移植，正确输出音频

---

## 以后的规则

1. **从后往前隔离**：当输出不对时，先测最后一个组件（vocoder），再测前面的（token 生成）。不要从最复杂的中间组件开始。

2. **有 TODO/SKIP 注释的代码不算完成**：审查时要检查所有 `TODO`、`skip`、`omitted` 注释，确认跳过的功能是否影响正确性。

3. **三次失败换方向**：如果连续 3 次实验没有改善目标指标，停下来质疑假设，不要继续试同一类修改。

4. **波形特征先于 token 分析**：音频问题先看波形（RMS、max、silence）。异常的能量分布（3x 响度、全程削波）指向信号处理 bug，不是 token 问题。

5. **对照实验优先于理论分析**：先做 5 分钟的隔离实验，再花 5 小时分析理论。
