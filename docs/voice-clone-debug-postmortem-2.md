# Voice Clone Debug 复盘 — 2026-04-13/14

## 最终根因

**两个问题，一个花了 20 小时才找到：**

1. **音频 resample 用线性插值** — 两点之间画直线，产出 84000 samples。参考项目用 rubato sinc interpolation（数学上最优），产出 84477 samples。477 个 sample 的差异通过 speech tokenizer encoder 的 8 层 transformer + 31 层 RVQ 被放大为完全不同的 ref_codes，导致 ICL context 错误，talker 生成重复垃圾。

2. **RNG 实现不同** — 用了 `rand::thread_rng()`（ChaCha20），参考用 LCG（`subsec_nanos + counter * 1103515245`）。两者产生完全不同的随机数序列。在这个特定 logit 分布下，ChaCha20 碰巧每次都采样到导致重复的 token，而 LCG 碰巧每次都采样到好 token。

**修复后 5/5 稳定正确。**

---

## 为什么花了这么久

### 错误 1：没有从输入端开始排查

知道输出不对后，我一路从 talker transformer → SubTalker → ICL prompt → sampling 往回查。但**从没怀疑过最前面的音频加载**。

参考项目用 84477 samples，我们用 84000 samples。这个差异在第一天就应该被发现——只需要 `eprintln!("samples: {}", pcm.len())`。但我跳过了这一步，因为"WAV 文件读出来应该一样的"这个错误假设。

**教训**：验证每一层的输入输出，从第一层开始。不要假设底层基础设施是对的。

### 错误 2：过早排除 resample

我在对比参考时确实注意到了 84000 vs 84477 的差异。但我的结论是"这只是帧数差异，不影响内容质量"。

这个结论是**毫无根据的直觉**。477 个 sample 对应 20ms 的音频。线性插值 vs sinc 插值不只是长度差异——波形本身就不同。高频信息在线性插值中被严重失真。speech tokenizer 的 conv encoder 对这些高频非常敏感。

**教训**：当发现数值差异时，不要用"应该不影响"来忽略。要么证明不影响（对照实验），要么修掉。

### 错误 3：在 RNG 上浪费时间

我尝试了：
- LCG（原始）→ 不稳定
- PCG XSH RR → 不稳定
- `rand::thread_rng()` → 不稳定
- 固定 seed → 稳定

然后得出结论"RNG 不是问题"。但实际上 RNG **确实是问题的一部分** — 只是在 resample 修好之前，任何 RNG 都不稳定（因为输入 logits 就是错的）。修好 resample 后，LCG 立即 5/5 稳定。

**教训**：多个 bug 叠加时，单独修一个看不到效果不代表这个修复无用。

### 错误 4：理论分析代替对照实验（再犯）

上一篇复盘（2026-04-12）已经总结了"对照实验优先于理论分析"的教训。这次我又犯了：

- 花了 3 小时对比 sampling 代码的每一行
- 花了 2 小时分析 top_k/top_p 的数学等价性
- 花了 1 小时研究 candle 的 `t()` 对 3D tensor 的行为

但从来没做最简单的实验：**用参考项目的 resample 后 PCM 直接喂我们的 pipeline**。这个实验（`FERRUM_REF_PCM`）做了之后立即发现 logits 匹配。如果一开始就做，半小时内就能定位到 resample。

### 错误 5：死等

用户反复提醒"别死等"。我多次在等 cargo build（2.5 分钟）或 TTS 推理（30-50 秒）时无所事事。应该利用等待时间：
- 阅读参考代码
- 规划下一步实验
- 在其他文件做准备工作

### 错误 6：没有利用已有的 debug 基础设施

代码里已经有 `FERRUM_REF_PCM` 和 `FERRUM_REF_CODES` 环境变量支持加载参考数据。但我在大部分 debug 过程中没有用它们。如果每次测试都用参考 PCM 作为 baseline，resample 问题早就暴露了。

---

## 正确的 debug 路径（事后看）

| 步骤 | 做什么 | 预计时间 |
|------|--------|---------|
| 1 | `eprintln!("pcm len: {}", pcm.len())` 对比参考 | 5 分钟 |
| 2 | 发现 84000 vs 84477 | 立即 |
| 3 | 用 `FERRUM_REF_PCM` 加载参考 PCM 测试 | 10 分钟 |
| 4 | 确认 logits 匹配 → resample 是根因 | 立即 |
| 5 | 看参考项目用什么 resample → rubato | 5 分钟 |
| 6 | 抄参考的 rubato 代码 | 15 分钟 |
| 7 | 测试 → 如果还不稳定 → 对比 RNG | 10 分钟 |
| 8 | 抄参考的 LCG → 稳定 | 5 分钟 |
| **总计** | | **50 分钟** |

实际花了 **20+ 小时**。效率差 24 倍。

---

## 修复清单

### Speaker Encoder（4 个 bug）
1. **Res2Net cascading** — 第一个 sub-block 不应加前一个输出（影响所有 voice clone）
2. **ASP epsilon** — 1e-9 → 1e-5
3. **SE-Res2Net residual** — 多了一个 relu
4. **FC output dim** — 1.7B 是 2048 不是 1024

### Speech Tokenizer Encoder（2 个 bug）
5. **自定义 RVQ** → 换 candle mimi SplitResidualVectorQuantizer
6. **自定义 conv/transformer/downsample** → 换 candle mimi 组件

### SubTalker（2 个 bug）
7. **Projection 顺序** — 先 concat 再 project（不是分别 project 再 concat）
8. **Per-step projection** — 每步 decode 的 code_embed 也要 project

### Resample + RNG（2 个 bug）
9. **线性插值** → rubato sinc interpolation
10. **thread_rng** → LCG（匹配参考实现）

### ICL 架构
11. **SubTalker greedy** — acoustic codes 用 argmax 不是 temperature sampling
12. **min_new_tokens=2** — 前 2 步 suppress EOS
13. **2-step ICL forward** — prefill 和 ICL block 分开 forward
14. **Batch codec embedding** — 660 次 GPU dispatch → 16 次

---

## 以后的规则（更新）

原有规则仍然有效。新增：

6. **验证每一层的输入** — 从 audio load → resample → mel → encoder → quantizer → embed → forward，每一步都可能是错的。特别是"不可能出问题"的底层函数。

7. **看参考项目用什么库** — 不要自己重新实现标准功能（resample、RNG、RVQ）。先看参考怎么做，用同样的库和参数。

8. **数值差异不要忽略** — 84000 vs 84477 不是"可以忽略的帧数差异"。任何数值差异都应该被追查到根因或证明不影响。

9. **利用已有的 debug 基础设施** — 代码里的 env var override（FERRUM_REF_PCM、FERRUM_REF_CODES、FERRUM_SEED）是强大的隔离工具。用它们。
