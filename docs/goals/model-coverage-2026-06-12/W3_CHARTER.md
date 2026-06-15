# W3 立项合同 — Gated DeltaNet 混合注意力子系统(草案,W1+W2 PASS 后生效)

> GOAL.md 对 W3 的交付物定义:本目标只交付这份立项合同;实现与验收属于
> 独立 goal。进入条件:`MODEL_COVERAGE_W1 GOAL PASS` + `MODEL_COVERAGE_W2
> GOAL PASS` 均已打印(W1 已于 2026-06-13 达成;W2 已于 2026-06-14 达成)。

> 发布级补充:本 charter 的 `≥0.5× llama.cpp` 只是模型覆盖阶段 sanity floor。
> W2/W3 若要进入 release-grade/README 发布矩阵,以
> `RELEASE_GRADE_GOAL.md` 为准:正确性必须全过,性能必须达到同硬件同模型
> 主流推理引擎至少 80%。

## 选型决议

**默认选项:Gated DeltaNet**(对 gpt-oss-20b 备选的裁决理由,引 GOAL.md):
解锁 Qwen3.5/3.6 整条 2026 主线家族(24GB 需求第一)+ 官方 GPTQ-Int4 可
直接喂 Marlin + llama.cpp CUDA 路径仍慢(issue #17822)形成差异化窗口;
gpt-oss-20b 三件套(attention sinks / MXFP4 / harmony 模板)只解锁一个家族。

## 执行合同(CLAUDE.md 格式)

```text
Lever: chunked delta-rule kernel + recurrent state cache + 512-expert
  shared-expert MoE 变体,三件套构成 DeltaNet 子系统
Expected gain: ferrum run qwen3.5:35b-a3b / qwen3.6 全线可跑,
  并按 W2 新家族 gate 标准认证(≥0.5× llama.cpp 起步)
Files/paths to inspect or change:
  - crates/ferrum-kernels/triton_ptx/(fla Triton kernel 离线编译嵌入通道,
    现有 triton-kernels feature 即移植路径)
  - crates/ferrum-interfaces/(新增 recurrent-state cache trait —— 与
    KvDtypeKind 平行的第六个多态维度,设计先行)
  - crates/ferrum-models/(Qwen3.5-MoE 家族结构,复用 Qwen3MoeModel 的
    router/expert 资产 + shared-expert 扩展)
Correctness gate(分阶段):
  S0 可移植性:50 行级 fla chunked delta-rule microbench(Triton→PTX→
     cudarc 加载),数值 vs fla 官方 PyTorch 实现 atol 1e-3 —— 不过即停,
     整个 W3 重新评估
  S1 单层:DeltaNet 层 CPU 参考实现 vs HF transformers 逐层 dump(沿用
     gemma3_l1_compare 方法)
  S2 整模型:Qwen3.5 最小尺寸 BF16 greedy vs HF byte-equal(W2 修订 #3
     平局规则适用)
  S3 Paris 级 known-answer + 行为/agent gate(W1 阶梯复用)
Benchmark gate: 同卡同量化 vs llama.cpp ≥0.5×(新家族 sanity floor);
  vs vLLM 不设硬目标(那是后续单模型 perf goal)
Budget cap: 设计+S0 microbench 纯本地;S1 起每周 ≤1 pod-day,
  每周五 stop/go 决策,连续两周无 gate 进展即冻结
Stop condition: S0 不可移植 / recurrent cache trait 设计评审发现与
  paged-KV 体系不可共存 → 停,出报告转 gpt-oss-20b 备选评估
```

## 风险登记(开工前已知)

1. **fla kernel 的 Triton 版本锚定**:triton_ptx 通道是离线编译,fla 上游
   kernel 对 Triton 版本敏感;S0 需同时锚定 (fla rev, triton ver, PTX arch)。
2. **recurrent state 与 ContinuousBatch 调度**:DeltaNet 状态是 O(1) 不是
   KV 序列 —— 抢占/恢复语义与现 scheduler 的 block 思维不同,接口设计是
   第一周交付物(纸面)。
3. **W1/W2 开放问题不得带病开工**:schema-500(开放问题 #1)与 CUDA
   autosizer(#4)若仍未关闭,先各给半天定位,因为 DeltaNet 验收同样依赖
   这两条路径。

## 不做清单(防 scope 蔓延)

- 不做 Qwen3.5 视觉塔(归 VLM goal)。
- 不做 MTP/投机解码组合(独立 lever)。
- 不在本 goal 内追 0.8× vLLM(单模型 perf goal 范畴)。
