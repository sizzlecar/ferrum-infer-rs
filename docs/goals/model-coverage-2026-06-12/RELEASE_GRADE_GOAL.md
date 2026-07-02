# W2/W3 发布级目标 — 正确性硬门 + 主流引擎 80% 性能线

> 本文是 `GOAL.md` / `W3_CHARTER.md` 之后的发布级目标补充。
> 既有 W2 PASS 只能证明 Gemma 3 27B 文本路径已经可用、可诊断、可被产品入口跑通;
> 它不等价于 release-grade。W3 仍未进入实现阶段。后续对外称"支持"、
> README 发布矩阵、release notes 或性能宣传,以本文门槛为准。
> 命名说明:本文沿用仓库现有 W2/W3 命名;若外部讨论写作 M2/M3,本目标指同一批
> "第二阶段可用态"与"第三阶段未启动战略件"。

## 目标声明

把 W2/W3 从"能跑"推进到"可发布":

- 正确性是前置硬门。任何模型、格式或后端只要正确性 gate 未过,性能数据只能
  标为 diagnostic,不能进入性能结论、README 性能表或 release-ready 叙述。
- 性能必须达到同硬件、同模型、同量化/格式、同产品入口下主流推理引擎的
  **至少 80%**。低于 80% 时,即便功能可用,也只能标为 functional / preview /
  known-gap,不能标为 release-grade。
- 所有证据必须来自用户可复现的产品路径:typed CLI/config/defaults,不得依赖
  隐藏 env 组合。`ferrum run` 与 `ferrum serve` 都必须覆盖。

本目标不是官方 tag/release 发布目标;它定义的是模型支持达到 release-grade
之前必须完成的工程、正确性、性能和证据标准。

## 当前判定

| 项 | 当前状态 | 发布级判定 |
|---|---|---|
| W2 / Gemma 3 27B text, CUDA GPTQ | L0/L1/L2-GPTQ/L3/L4/L5 已跑通; c=32 通过 typed admission cap 16 完成; artifact: `artifacts/w2_c32_admission16_l5_pass_2026-06-14/` | functional, not release-grade |
| W2 / Gemma 3 27B perf | Ferrum c=1 decode 25.252 tok/s vs llama.cpp tg128 50.478 tok/s, ratio 0.500260 | 只过旧 sanity floor,距离 80% 目标约 1.6x |
| W2 / Gemma 3 27B GGUF | 既有 W2 中 waived | 不可作为发布级 GGUF/Metal 支持 |
| W3 / Gated DeltaNet | 只有 `W3_CHARTER.md` 立项草案,未实现 | not started |

结论:W2 当前是可用阶段,不是发布阶段。W3 当前是立项阶段,不是实现阶段。

## 主流引擎基线定义

性能基线按每个 lane 单独选择,并写入 artifact manifest:

1. CUDA + HF/safetensors/GPTQ/AWQ:优先 vLLM;如果 vLLM 当前不支持该模型或
   该量化格式,用能运行同模型同格式的主流引擎中最快者。
2. GGUF / Metal / llama.cpp-only 格式:优先 llama.cpp。
3. 如果 vLLM 和 llama.cpp 都能运行可比 lane,以同硬件同量化下更快者作为
   80% 分母;另一个作为参考行记录。
4. 如果没有可比主流基线,不得宣称 release-grade performance。先补 baseline
   artifact,或把该 lane 标为 unsupported / diagnostic。

同硬件要求:

- CUDA:同一台 RTX 4090 或同一台 2x RTX 4090,同驱动/CUDA 可见性。
- Metal:同一台 Mac 型号、同内存级别、同电源/thermal 状态说明。
- 不能用跨机器、跨云主机、跨量化格式的数字做 80% 结论。跨格式数字只能作为
  sanity 或 known-gap 说明。

## 性能通过标准

每个 release-grade lane 必须同时满足:

- `bench-serve --fail-on-error --require-ci --seed 9271 --n-repeats 3`。
- 并发 cell:至少 c=1/4/16;30B 级或 MoE/GPTQ lane 还必须含 c=32。
- 每个 cell `completed_per_run` 全满、`errored_per_run` 全 0。
- streaming 请求必须包含 usage,且 release 报告必须使用 tokenizer/usage 计数,
  不能用估算 token 数。
- Ferrum throughput lower confidence bound 不低于 baseline throughput 的 80%。
  若只保留均值,则必须 N>=5;否则 N>=3 但要保存 CI/方差。
- TTFT/ITL 不能用吞吐掩盖严重尾延迟:同 cell 下 p95 ITL 不得高于 baseline
  1.25x,除非目标文件明确该 lane 只发布 offline throughput。
- c=32 若通过 admission cap 或 scheduler cap,必须公开有效 active concurrency,
  baseline 也要用相同有效 cap。不能把 server cap 16 的结果宣传成 true c=32
  release performance。

## 正确性硬门

本文不降低既有 L0-L5。发布级前置门如下:

- L0 chat template/tokenizer golden,禁止 silent fallback。
- L1 数值/逐层或代表路径证明,按 `GOAL.md` 修订后的可行规则执行。
- L2 真实尺寸量化路径语义正确;若某格式 waived,该格式不能发布。
- L3 多轮、stream/non-stream、一致停止、reasoning 提取。
- L4 tools + strict JSON schema 真模型 smoke。
- L5 concurrency cell 全满零错误。
- `ferrum run` 与 `ferrum serve` 都必须有 artifact。

任何 correctness gate 失败时,停止性能测量;已经采到的性能数字必须标记为
diagnostic only。

## W2 发布级工作项

W2 的发布级目标是把 Gemma 3 27B text 从 functional 推到 release-grade。

### W2-P0:发布口径修正

- README/支持矩阵不得把 W2 当前状态写成 release-grade。
- 当前可写为 "Gemma 3 27B text: CUDA GPTQ functional; performance known-gap"。
- GGUF/Metal lane 仍为 waived/unsupported,直到独立正确性和 80% 性能 gate 通过。

### W2-P1:正确性补齐

- 关闭任何 waived lane 或明确从发布范围删除该 lane。
- 如果保留 CUDA-only 发布范围,文档必须明确 GGUF/Metal 不在本次发布级支持内。
- 保持 L0-L5 全 PASS,并补 `ferrum run` 与 `ferrum serve` 的产品入口 artifact。

### W2-P2:性能差距收敛到 80%

优先级按预期收益排序:

1. 移除 Gemma3 正确性修复中的 host/F32 residual shadow 热路径成本,改为设备端、
   无 per-layer host sync/copy 的 sandwich-norm/GeGLU/post-norm 实现。
2. 让 batched/unified decode 路径真正支持 Gemma3 语义。当前 sandwich-norm 家族
   禁用 batched/varlen fast path 是合理的 correctness 防线,但 release-grade 需要
   把这些语义补进 fast path,不能靠单序列路径硬撑。
3. 重新审计 c=32 内存模型。若产品限制 active admission=16,则发布文档只能宣称
   admitted c=16;若要宣称 c=32,必须实现真实 c=32 或同等有效吞吐。
4. 对照 vLLM/llama.cpp 的 Gemma3 Q/K/V、GeGLU、rope/local attention 实现,确认没有
   低效 fallback、重复 transpose、重复 host readback 或未融合的热点。
5. 只有 correctness 完整保持后,再做 kernel-level 优化;不得用错误输出或过滤输出
   换性能。

W2 发布级 PASS 行:

```text
MODEL_RELEASE_GRADE_W2 PASS: <out_dir>
```

该 `<out_dir>` 必须包含 correctness artifacts、baseline artifacts、Ferrum artifacts、
ratio report、git SHA/dirty status、binary SHA256(如可得)、runtime config snapshot、
GPU/Mac hardware snapshot、最终 validator manifest。

## W3 发布级工作项

W3 的目标不是"先能跑再说",而是从设计开始就按 release-grade 建门。
默认 W3 方向仍是 Gated DeltaNet 混合注意力子系统,但本文提高验收标准:
W3 进入 README release-grade 支持前,必须达到主流引擎 80%。

### W3-S0:设计与微基准

- 设计 recurrent state cache trait,明确它与 paged-KV、ContinuousBatch、
  preemption/release 语义如何共存。
- 移植或实现 chunked delta-rule kernel 的最小 microbench。
- microbench 必须 vs 官方/主流参考实现数值对齐,并保存 PTX arch、Triton/fla rev、
  build command、输入分布、误差统计。

### W3-S1:单层正确性

- DeltaNet 层 CPU/reference vs Ferrum 单层 dump 对齐。
- shared-expert / 512-expert MoE 变体要分别验证 router、expert layout、
  shared expert merge 语义。

### W3-S2:整模型产品路径

- 接入 tokenizer/template/loader/config 后,先跑 `ferrum run`,再跑 `ferrum serve`。
- L0-L5 全过之前不得进入性能结论。
- 不允许通过隐藏 env 打开核心行为;必须是 typed runtime config、CLI/config
  或模型默认解析。

### W3-S3:80% 性能门

- 首选 baseline:vLLM/fla 路径(若当前支持对应模型与量化);否则按"主流引擎基线定义"
  选择最快可比主流引擎。
- 必须覆盖 c=1/4/16/32;如果模型尺寸只能支持较低 active concurrency,需在
  发布范围内明确最大并发,并用相同 cap 比较 baseline。
- 任何 cell 低于 80% 即 W3 release-grade fail。可以继续标为 experimental,
  但不能进入 release-grade 矩阵。

W3 发布级 PASS 行:

```text
MODEL_RELEASE_GRADE_W3 PASS: <out_dir>
```

## GPU 执行合同

每次 paid GPU 前必须写入 artifact notes:

- lane: W2 release-grade perf / W3 S0 / W3 S1 / W3 full gate 等。
- expected runtime/cost。
- stop condition。
- correctness gate command。
- performance command。
- baseline engine/version/build command。

失败策略:

- correctness 首败即停。
- performance 低于 80% 时,保留 artifact,停止重复 full sweep,转为 profiler/最小
  A/B 诊断。
- 不为省机器反复重装环境;优先复用有模型缓存的 native CUDA 机器,但 GPU 不能空闲。

## 明确非目标

- 本文不追求"支持更多模型"。coverage PASS 与 release-grade 是两个不同层级。
- 不用 0.5x sanity floor 做发布口径。
- 不把跨格式 llama.cpp 对照当作 CUDA GPTQ 的正式 80% 证据。
- 不接受错误输出、乱码、`<unk>`、缺 usage、schema 失败、工具调用失败后仍跑性能。
- 不把 cap 后的并发写成未 cap 的并发。

## 需要新增或调整的验证器

- `scripts/release/model_release_grade_goal_gate.py`
  - 聚合 W2/W3 correctness manifests。
  - 校验 Ferrum/baseline 同硬件、同模型、同量化、同 prompt/cell。
  - 校验每个 cell ratio >= 0.8。
  - 校验 no hidden env / runtime config snapshot / git dirty status / binary hash。
  - 打印本文 PASS 行。

命令入口:

```bash
python3 scripts/release/model_release_grade_goal_gate.py w2 <out_dir>
python3 scripts/release/model_release_grade_goal_gate.py w3 <out_dir>
```

`<out_dir>` 必须包含 `model_release_grade_manifest.json`。validator 会写入
`model_release_grade_goal_gate.manifest.json`;只有该 manifest 为 `status=pass`
且 stdout 打印对应 exact PASS line 时,该 lane 才算发布级完成。

没有该 validator 的 exact PASS line,不得宣称本文目标完成。
