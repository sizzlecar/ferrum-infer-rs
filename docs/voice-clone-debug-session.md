# Voice Clone Debug Session — 2026-04-10/11

## 问题

TTS voice clone 输出不可懂。Python baseline 正常，Rust 输出噪音。

## 根因链

```
Rust prefill (73 tokens) → 28 层 transformer forward
    → candle 的 decomposed ops (5-pass softmax, 分步 matmul, 分步 RMSNorm)
    → 每个 op 写回 f32 中间 tensor，额外舍入
    → Layer 0 o_proj 就和 Python 差 0.001
    → 28 层累积到 past_hidden 差 0.5-1.0
    → SubTalker 输入偏差太大
    → codebook 5+ greedy argmax 翻转（top2 logit 差仅 0.28）
    → 错误 codec embedding 反馈 → 级联崩溃
```

## 关键验证结果

| 验证项 | 结果 |
|--------|------|
| prefill 输入（73 位置） | Rust vs Python 逐位匹配 ~1e-7 ✅ |
| Layer 0 q_proj | 匹配 ~1e-7 ✅ |
| Layer 0 QK-norm | 匹配 ~1e-6 ✅ |
| Layer 0 RoPE | 匹配 ~1e-6 ✅ |
| Layer 0 pre-softmax (QK^T/√d) | 所有 16 heads 匹配 ~1e-5 ✅ |
| Layer 0 post-softmax | 差 ~0.001 ❌ |
| Layer 0 o_proj | 差 ~0.001 ❌ |
| 28 层后 past_hidden | 差 0.5-1.0 ❌ |
| SubTalker（Python exact input） | codebook 0-14 全部正确 ✅ |
| SubTalker（Rust past_hidden） | codebook 5+ 偏离 ❌ |
| CPU Accelerate vs Metal gemm | 结果完全一致（都不对）|
| f64 softmax | 无改善（softmax 不是问题）|
| fused attention (CPU raw slices) | past_hidden 改善到差 0.1-0.6 |
| ferrum-attention Metal kernel | 单元测试通过，精度 < 1e-3 vs CPU |

## 根因确认

**不是某个 op 的 bug，是 candle 框架的 decomposed ops 累积舍入。**

candle 的 softmax 用 5 个独立 kernel（max → sub → exp → sum → div），每步 materialize 中间 tensor 到 f32。PyTorch 的 F.softmax 是 fused 单 pass，中间值在寄存器里不额外舍入。

在 Python 内部验证：分步 softmax vs fused softmax 差 3.58e-7（不影响）。但 candle 的 tensor ops 在 Metal/CPU 上每步的实际舍入行为和 PyTorch 不同，导致 softmax 后差 0.001。

这个差异通过 28 层 residual connection 放大，SubTalker 的 greedy decode 对此极度敏感。

## 解决方案

**自建 fused kernel，绕过 candle tensor ops：**

1. ✅ ferrum-attention crate — Metal + CPU fused flash attention
2. 🔧 fused transformer layer — Linear/RMSNorm/SiLU 也自建
3. 后续：Metal zero-copy（直接操作 candle Metal buffer，不经 CPU）

## 修复的其他 bug

1. **language_id 未解析**：`auto` 没有映射到 `chinese`，导致 prefill 少 1 位（72 vs 73）
   - 文件：`tts_executor.rs:473`
   - 状态：已修复（本次改动中）

2. **trailing_text_hidden 处理差异**（已识别，未修复）：
   - Python：trailing text 不在 prefill，decode 时逐步加到输入
   - Rust：trailing text 拼到 prefill 末尾，decode 每步加 tts_pad
   - 影响：text > codec 时输出错误
   - 状态：待修复

## debug 过程中的教训

1. **一开始方向错误**：花大量时间在 "f32 精度累积" 假设上，尝试了 f64 softmax、f64 RMSNorm、Accelerate BLAS，都无效。应该更早做 Python 端的控制实验（在 Python 内部对比分步 vs fused softmax，发现差异仅 1e-7）。

2. **没有及时 release build**：debug build 太慢导致 TTS 命令 hang，浪费大量时间等待。

3. **ref-text 不匹配**：用了错误的 ref-text 对比，导致 prefill 内容不同。应该一开始就 ASR 参考音频获取正确 ref-text。

4. **测试工具链不完善**：应该一开始就建立 TTS→ASR 的自动化对比流程，而不是逐个 dump 值。

5. **过度分析不如动手验证**：很多时间花在理论推导"差异可能来自哪里"，不如直接做对照实验。Python SubTalker 直喂实验在 5 分钟内就确认了 SubTalker 本身没 bug。

## 文件变更

```
新增：
  crates/ferrum-attention/           # 新 crate
  ├── Cargo.toml
  ├── build.rs
  ├── src/lib.rs
  ├── src/cpu/mod.rs                 # Accelerate sgemm + fused softmax
  ├── src/metal/mod.rs               # Metal dispatch
  ├── src/metal/shaders/flash_attn.metal  # Metal flash attention kernel
  └── tests/metal_test.rs

  docs/voice-clone-debug-session.md  # 本文档

修改：
  Cargo.toml                         # workspace 加 ferrum-attention
  crates/ferrum-models/Cargo.toml    # 依赖 ferrum-attention
  crates/ferrum-models/src/architectures/qwen3_tts.rs  # attention 改用 ferrum-attention
  crates/ferrum-models/src/executor/tts_executor.rs     # language_id 修复
  crates/ferrum-cli/Cargo.toml       # accelerate feature
  crates/ferrum-engine/Cargo.toml    # accelerate feature

待清理：
  crates/ferrum-models/src/architectures/fused_attention.rs  # 旧的 fused attention（被 ferrum-attention 取代）
  crates/ferrum-models/src/architectures/fused_transformer.rs # 旧的 fused transformer（半成品）
  ferrum-test-data/compare_l0_attn.py  # debug 脚本
```
