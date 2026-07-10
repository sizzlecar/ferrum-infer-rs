# Ferrum Runtime vNext / v0.8.0 总目标

## 状态

Open。创建于 2026-07-10。

本目标不是在现有 Architecture v2、`Backend` 大 trait、模型专属 runner 和
`run`/`serve` 分叉上继续打补丁。目标是重新设计 Ferrum 的核心推理架构，并把
已经可靠的测试、benchmark、artifact、kernel 和 release 能力收敛到新架构。

完成本目标意味着实际发布 `v0.8.0`，而不是仅达到 release-ready。只有发布后的
Metal/CUDA 安装产物通过最终验证器并打印下面这一行，目标才算完成：

```text
FERRUM RUNTIME VNEXT V0.8.0 RELEASE GOAL PASS: <out_dir>
```

计划中的总验证器：

```text
scripts/release/runtime_vnext_goal_gate.py
```

该脚本只允许聚合 `run_gate.py` 产生的 manifest、校验 DAG/freshness 和打印最终 line，
不得自行重跑或重新解释业务 gate。

## 1. 为什么必须彻底重构

当前问题不是缺少一个 registry 或几条测试，而是五个核心边界同时失效：

1. 模型语义进入通用 `Backend` trait，Qwen3.5 接入新增架构命名方法和热路径
   capability 分支。
2. 模型自行复制 prefill/decode/unified runner、KV/recurrent state 和 scratch 管理，
   通用生命周期无法复用。
3. `run` 与 `serve` 独立解析 source、alias、config、preset、capability 和 runtime
   defaults，曾使 `run` 未应用 `serve` 已使用的 resolved auto-config；同一真实模型修复前后
   从约 `9.5` 提升到 `54.3 tok/s`，证明产品组合分叉会直接隐藏正确性和性能路径。
4. 测试数量很多，但真实入口、真实 feature set、真实模型和 artifact freshness 没有
   形成闭合证据图；顶层自测甚至可以遗漏已失败的子门禁。
5. CUDA 编译、GPU 正确性集成和性能定位共用一个几十分钟级反馈循环，导致 GPU 被
   当作资源状态机和架构假设的第一次完整测试。

历史量级支持彻底重构：当前 `qwen35.rs` 单文件有 `18,239` 个物理行（其中包含 test
module），六个 Qwen3.5 命名生产源码文件的物理行合计 `24,317`；production LOC 必须由
G00 按统一分类器另算。2026-06-17 至 06-26 的 git subject 中有 `255` 个匹配
`Qwen3.5|Qwen35`，其中 `13` 个含 revert/rollback；这些主题提交触达 `86` 个去重后的
`crates/`/`scripts/` Rust、CUDA、build、test 或 gate 路径。已有复盘还记录了最近 500
提交中 `qwen35.rs` 被改动 114 次、128 个 Qwen35 artifact 目录和约 40.28 小时本地 ledger。
继续局部收敛只会把现有耦合固化为下一轮重构的前置债务。

上述证据分别来自
[`test-architecture HANDOFF`](../test-architecture-2026-06-10/HANDOFF.md) 和
[`W3 Qwen3.5 retrospective`](../model-coverage-2026-06-12/W3_QWEN35_RETROSPECTIVE_20260626.md)。
G00 必须把统计命令和结果重新写入 artifact；本段数字只用于立项，不替代 baseline PASS。

### 1.1 仓库证据快照

| 痛点 | 当前代码/历史证据 | 对应重构 Goal |
|---|---|---|
| 核心抽象泄漏模型语义 | [`Backend`](../../../crates/ferrum-kernels/src/backend/traits.rs) 文件 2,341 个物理行，当前含 32 处 Qwen35 命名符号；[`ModelExecutor`](../../../crates/ferrum-interfaces/src/model_executor.rs) 与 [`ContinuousEngine`](../../../crates/ferrum-engine/src/continuous_engine.rs) 共同分担 lifecycle | G01、G03、G04 |
| 单模型接入膨胀 | [`qwen35.rs`](../../../crates/ferrum-models/src/models/qwen35.rs) 18,239 个物理行；同名六个生产源码文件共 24,317 行 | G01、G03、G08 |
| `run`/`serve` 组合分叉 | [`run.rs`](../../../crates/ferrum-cli/src/commands/run.rs) 与 [`serve.rs`](../../../crates/ferrum-cli/src/commands/serve.rs) 分别解析 source/config；历史 handoff 记录真实 CUDA 路径 `9.5 -> 54.3 tok/s` 修复 | G05 |
| 测试晚发现 | W3 复盘记录 128 个 Qwen35 artifact 目录，GPU 逐次发现 resource-state bug；近期 500 提交中 test/docs/perf/fix 高密度交替 | G02、G04、G10 |
| 性能定位反复试错 | 同一复盘记录约 40.28 小时 ledger，多个 c32 candidate 只改变局部 trace、仍远离目标 | G06、G09 |
| CUDA 编译阻塞 | test-architecture handoff 记录 CUDA L1 cold `1906s`、warm `18s`；重型 native source 与 release LTO 扩大失效域 | G07 |
| CUDA/Metal 乒乓回归 | [`release regression hardening goal`](../release-regression-hardening-2026-06-28/GOAL.md) 已归档多次 backend 边界、资源和人工 release smoke 问题 | G02、G03、G10 |

物理行数和 symbol 次数是 2026-07-10 的只读快照，不是最终 LOC 指标；G00 analyzer 才是后续
减少比例、分类和 PASS 的唯一数据源。

### 1.2 当前核心执行链

```text
ferrum run / serve
  -> ferrum-cli source/alias/config/preset resolution
  -> ferrum-engine registry + builder
  -> ContinuousEngine + ferrum-scheduler + ferrum-kv/recurrent state
  -> ferrum-interfaces::ModelExecutor
  -> ferrum-models architecture-specific loader/executor/runner
  -> ferrum-kernels Backend traits + CUDA/Metal providers/native kernels
  -> sampler/tokenizer -> terminal or ferrum-server OpenAI/SSE response
```

`bench-serve` 经 `ferrum-bench-core::BenchReport` 测 HTTP 产品路径，release scripts 再把
unit、Metal、CUDA、tarball、Homebrew 和 completion artifact 聚合。问题不在这些模块完全不存在，
而在 model semantics、backend capability、resource lifecycle 和 product defaults 横跨多层并形成
第二真相；vNext 保留已验证实现，重建它们之间的 ownership 和 contract。

## 2. 总策略

采用四类处理方式，禁止含糊的“以后再清理”：

| 类别 | 处理方式 |
|---|---|
| 核心 contract、trait、runner、资源所有权、产品组合根 | 重新设计并替换 |
| CUDA/Metal kernel、quant 实现、`bench-serve`、`BenchReport` | 保留实现，适配新 contract |
| resource invariant、observability/replay、scenario/gate schema | 修复依赖图后收敛复用 |
| legacy factory、架构命名 Backend API、重复入口、env-only 产品策略 | 迁移完成后删除 |

迁移使用受控双轨：legacy 只能作为 baseline、shadow comparison 和回滚参考，不能进入
`v0.8.0` release binary。每个 adapter 必须声明 owner、创建 Goal、删除 Goal 和最晚删除
里程碑；没有 sunset 的 adapter 不得合并。

## 3. 优先级

1. CUDA 是第一实现和性能优化后端。
2. Metal 在核心 contract 稳定后跟进，但最终发布不允许 waiver。
3. 正确性先于性能；任一真实模型 correctness cell 失败时，对应性能数据只能标记为
   diagnostic。
4. 三个主模型优先于长尾兼容；主模型迁移成功后，再迁移或撤销其他 support row。
5. 快速开发循环优先于单次漂亮 benchmark；正式 release 构建与开发构建分离。

## 4. 主三模型

主矩阵选择新、热门且能覆盖不同执行结构的 Qwen 系模型。热度快照和精确格式见
[`MODEL_MATRIX.md`](MODEL_MATRIX.md)。

| 模型 | 角色 | 核心覆盖 |
|---|---|---|
| `Qwen/Qwen3.5-4B` | 高频 dense-hybrid canary | Gated DeltaNet、full attention、recurrent state、dense FFN；目标态低成本双端回归（当前 Metal unsupported） |
| `Qwen/Qwen3.5-35B-A3B-GPTQ-Int4` | 战略主模型和最难验收对象 | hybrid attention、recurrent state、256 experts、8 routed + 1 shared、GPTQ、资源压力 |
| `Qwen/Qwen3-30B-A3B-GPTQ-Int4` | 成熟传统 MoE 控制组 | full attention、128 experts/top-8、QK norm、Marlin、paged KV、历史性能锚点 |

三个主模型均只认证 language path。Qwen3.5 vision tower 不在本目标范围；图片输入必须
显式返回 unsupported/4xx，禁止静默忽略并按文本请求处理。

附加但不占主三模型名额的强制 lane：

- `Qwen/Qwen3-Coder-30B-A3B-Instruct`：agent/tool XML 格式和历史 CUDA empty-answer bug kill。
- `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B`：reasoning、特殊 EOS、think-history 和模板回归。
- `meta-llama/Llama-3.1-8B-Instruct`：只作为仓库 G0 发布政策要求的 compatibility lane，不作为产品
  优先模型，也不能替代任何主模型 PASS。

## 5. vNext 核心结构

### 5.1 稳定基础层

`DeviceRuntime` 只负责设备、buffer、stream/command、同步和错误边界。它不知道模型、
Transformer、Qwen、Llama、MoE 或 scheduler。

### 5.2 可版本化 operation contract

Attention、linear attention、MoE、quantized linear、norm、sampling 等均为独立 operation
contract。每个 operation 必须具备：

- stable operation id 与 schema version；
- shape/dtype/layout/resource contract；
- CPU oracle 或明确的高精度 reference；
- 每个支持 backend 的 provider 与 conformance test；
- 显式 unsupported 结果；
- profiler phase、resource scope 和 fault-injection point。

禁止为某个模型向通用设备 trait 添加 `qwen35_*`、`gemma_*` 一类方法。

### 5.3 模型语义层

`ModelFamilyProvider` 负责：

- 解析官方 typed config；
- 声明 weight schema 与映射；
- 构造由 semantic blocks 和 state specs 组成的 `ModelProgram`；
- 声明需要的 operation contracts；
- 提供 chat/template/EOS 元数据，不决定 backend 实现。

### 5.4 计划层

`ExecutionPlanner` 在模型加载阶段把 `ModelProgram + BackendCapabilities + RuntimePolicy`
解析为不可变 `ExecutionPlan`。计划必须可序列化、可 snapshot、可 diff，并包含每个选择、
fallback 和拒绝原因。capability 判断不得留在 token hot loop。

### 5.5 执行与资源层

共享 `ExecutionRuntime` 负责 batching、prefill/decode、layer traversal、logits、sampling
边界和清理。唯一 `ResourceTransaction` 管理 KV、recurrent state、scratch、graph workspace、
admission、commit、rollback、release。scheduler、engine 和 model 不得分别拥有半套资源真相。

### 5.6 产品组合层

`run` 和 `serve` 通过唯一 `ResolvedModelPlan` 进入同一个 engine。二者只保留 terminal 与
HTTP/SSE 的 I/O 适配，不得各自解析模型 alias、能力、默认值、优化开关或 chat template。

## 6. 架构硬约束

以下指标是最终状态，不是“尽量减少”：

| 指标 | v0.8.0 目标 |
|---|---:|
| 通用 trait 中架构命名方法 | `0` |
| token hot loop 中 `supports_*` / backend feature 决策 | `0` |
| model/engine 中未批准的 `cfg(cuda|metal)` | `0` |
| 核心 runtime 直接读取隐藏 `FERRUM_*` 环境变量 | `0` |
| 同一模型的产品 source/config/capability 决策实现 | `1` |
| model-owned scheduler/KV/recurrent manager | `0` |
| release binary 中 legacy executor/factory/runtime | `0` |
| silent fallback / silent default success | `0` |
| 未声明 sunset 的 compatibility adapter | `0` |

必须通过四个扩展演练：

1. 新增只使用现有 op 的 synthetic model family，核心 runtime、Backend、`run`、`serve`
   生产代码改动均为 `0`。
2. 新增一个 novel op，只修改 op contract、provider、capability catalog 和 conformance test；
   planner/runtime 主循环改动为 `0`。
3. 新增 reference backend，模型生产代码改动为 `0`。
4. 注入 unsupported capability、kernel failure、资源泄漏和坏输出，单个 artifact 必须给出
   request、plan node、operation、phase、资源状态和 first failure event。

## 7. 正确性总标准

三个主模型在 CUDA 和 Metal 上都必须真实执行，不能由另一个模型、mock、stub 或
synthetic fixture 代替：

- `ferrum run`：单轮、JSONL 三轮、交互三轮、长输出、多字节 UTF-8。
- `ferrum serve`：non-stream、stream、multi-turn、六轮 stateful loop。
- known-answer `20/20`。
- legacy 可比 lane：相同 backend/format 下，冻结旧 binary 与 vNext 在 `temperature=0`
  的前 64 个生成 token 完全一致，`20/20`，PASS exception 数量 `0`；near-tie logit margin 只作
  诊断，不能把 generated token flip 改写为 PASS。
- 新增 Metal/Qwen3.5 lane：官方 HF config/tokenizer/template 是 metadata 真值；独立 CPU
  FP32/Transformers checkpoint 是 op/layer 数值真值；同 GGUF llama.cpp 只做量化端到端
  token/logit 交叉验证和性能参考。三层证据均通过 [`MODEL_MATRIX.md`](MODEL_MATRIX.md) 的明确
  数值门，并绑定 checked-in `runtime_vnext_numerical_tolerances.json` row/blob SHA；artifact 不得
  自带或事后放宽 tolerance。
- required tool call `20/20`，auto tool call `20/20`，tool-result 回填 `20/20`。
- streamed tool-call delta 重组 `20/20`，arguments 必须通过声明的 JSON schema。
- strict `json_schema` `50/50`，`json_object` `50/50`。这是 Ferrum server contract，
  不是模型卡能力声明。
- stream 与 non-stream 内容、finish reason、usage 一致 `20/20`。
- 每个 stream 恰好一个 `[DONE]`、恰好一个 usage、至少一个输出 token。
- natural EOS、custom stop、`max_tokens`、context limit、cancel、timeout 全部通过。
- Qwen3.5 默认 thinking 与 `enable_thinking=false` 硬切换、Qwen3 的硬/软 thinking 切换、
  content/final/history 隔离均按 [`MODEL_MATRIX.md`](MODEL_MATRIX.md) 分别验证。
- panic、OOM、resource leak、串话、`<unk>`、`[PAD]`、U+FFFD、mojibake、特殊 token
  泄漏、missing/duplicate DONE 均为 `0`。
- CUDA client c=1/4/16/32、Metal client c=1/4/16 marker/checksum 隔离全部正确；每 cell
  必须记录 typed admission cap、observed max-active 和 active timeline。最高 required cell 的硬
  active floor 为 CUDA M1/M2/M3=`32/16/32`、Metal M1/M2/M3=`16/4/16`，eligible interval
  duty-cycle `>=0.80`；typed cap 等于 floor 时 observed max-active 必须等于 cap。floor/cap
  变化必须走 reviewed Goal amendment，不能把排队的 client c32 冒充 active c32。
- `temperature=0` 只用于 deterministic parity；另以官方推荐 sampling + fixed seed 跑用户
  默认 smoke，不要求逐 token 相等但仍要求无垃圾、正确终止、tools/schema 和资源闭合。C21
  五组各 `4`；required-tool 与 strict response format 同时出现时必须确定性选择 tool priority，
  standalone strict-schema 另行成功。

任何 required probe 被 skip、waive、placeholder 或手填为 PASS 时，总 Goal 必须失败。

## 8. 性能总标准

### 8.1 证据协议

G00 先在冻结 legacy SHA `cff4c47765ef3259b8a04890187d99c60da86394` 上采集同机
基线。正式比较必须：

- 冻结 SHA 标识 legacy `run`/`serve` product binary；HTTP 采集使用单独锁定 source/tree/binary
  SHA 的 G00 canonical `bench-serve` client。该 client 只增加 payload 和证据字段，不改变
  被测 server；同一 comparison 的 A/B 必须使用完全相同的 client binary；
- 使用同一台机器、模型 revision、文件 SHA256、dataset、seed、CLI/config 和 feature set；
- 保存 Git SHA、dirty status、binary SHA256、硬件、driver/runtime、完整命令和非空日志；
- `serve-legacy`、`serve-external`、`run-legacy`、`run-vs-serve` 按适用范围分别执行独立
  `ABBA-BAAB`，不同 comparator 不复用 A rows 或 CI；
- 每个 serve comparison 的每实现每 cell 至少 `1200` measured requests 和 12 个 repeat
  samples；run comparison 每实现 12 个真实 `ferrum run` measured samples；
- 正确性 gate 先通过；
- HTTP 吞吐只使用 `ferrum bench-serve` 与 `ferrum-bench-core::BenchReport`。
- G00 legacy `ferrum run` 只比较 JSONL 暴露的完整 `engine.infer` E2E：
  `generated_tokens * 1000 / assistant.ms`。该边界包含 prefill/decode/sampling/text、排除 load 和
  shutdown，不得改名为 TTFT 或 steady decode；G06 后另采 token-commit 指标，并保留同一 E2E
  边界用于 legacy no-regression。

指标词典必须区分两个 ITL 来源。G00 `sse_delta_events` 仅是 client 收到完整 SSE output event 的
间隔 proxy；一个 request 只有在 usage completion tokens 等于 non-empty delta event 数、interval
数等于 `tokens-1`、且 transport 没有把多个 output event 合并成一次可观察 read 时才 eligible。
任一 paired request ineligible 时对应 repeat/cell 不生成正式 client-SSE ITL ratio，不能从 eligible
子集补数。G06 `engine_token_events` 使用同一 monotonic clock 的 token-commit timestamp，是独立
指标；两种 source 禁止合并、互相改名或用 tokenizer 重切 delta 合成不存在的到达时间。

标准命令必须包含：

```text
--fail-on-error --require-ci --seed 9271 --enable-thinking false --num-prompts 100 --warmup-requests 10 --n-repeats 3
```

CUDA 固定 1x RTX 4090，覆盖 random `256/128` 与固定 ShareGPT 数据集，
`c=1/4/16/32`。Metal 固定本机 `32GB / 24-GPU-core Apple M1 Max`，M2 使用固定
Q4_K_S，覆盖 `c=1/4/16`；Qwen3.5-35B 不允许因 32GB 机器不足而 waiver。物理硬件确实无法
满足 headroom/active floor 时，只能按 MODEL_MATRIX 的 reviewed amendment policy 更换明确硬件或
format，并重采全部受影响 baseline；不能原地降低 correctness、active floor 或 performance target。

### 8.2 不回退标准

对 legacy 已支持 cell：

- 每个 cell candidate throughput 中位数 `>= legacy`，且 ratio 的 95% CI 下界 `>=0.97`；
- 全部 cell throughput 几何平均值 `>=1.00x legacy`；
- TTFT、TPOT 的 candidate 中位数不得高于 legacy，ratio 95% CI 上界 `<=1.05`；只有 A/B
  paired request 全部 eligible 时，client-SSE-event ITL 执行同一门；G06 Ferrum token-commit ITL
  必须另行采集并满足对应 no-regression；
- `ferrum run` 单请求 decode tok/s 中位数 `>= legacy`，ratio LCB `>=0.97`；
- peak device/unified memory 增加 `<=3%`；
- completed `100%`，所有错误计数为 `0`。

对原本 unsupported 的 Metal/Qwen3.5 cell，不能伪称 no-regression：

- 与同机、同 GGUF、同 workload 的 llama.cpp 比较；
- Metal 全部 required c=1/4/16 的 throughput ratio LCB `>=0.80`；
- TTFT/TPOT p95 不高于 llama.cpp `1.25x`；全 paired request eligible 时 client-SSE-event ITL p95
  也不高于 `1.25x`，否则不生成该 ratio；G06 Ferrum token-commit ITL 仍为必需证据；
- 正确性先通过 reference gate。

CUDA 三个主模型还必须达到同机 vLLM 相同模型/格式/数据集 throughput LCB 的
`>=0.80`。仅守住低性能 legacy 基线不足以完成目标。

Qwen3-30B-A3B 有两套独立历史向量，禁止拼成一个不存在的 baseline：0.7.7 默认路径为
`164.2 / 353.3 / 636.9 / 706.0 tok/s`；历史 FA2 direct 路径为
`160.4 / 446.3 / 1185.1 / 1641.9 tok/s`。G00 必须分别绑定 artifact、SHA、feature、preset
和命令；只有证明模型、workload、硬件和产品可见配置可比后，才允许把逐 cell 最大有效
LCB 作为绝对防倒退线。不能把两套均值直接拼接，也不能静默降低。

Qwen3.5-35B-A3B 历史 vLLM ShareGPT 均值约为 c=1/4/16/32
`136.1 / 405.4 / 1190.7 / 1708.5 tok/s`；按该 artifact 的正式 LCB 计算，80% 历史参考线为
`107.495 / 324.046 / 896.239 / 1349.917 tok/s`。这些数字不是当前同机结论，最终以 G00
新鲜 same-host LCB 为准。

## 9. 构建和开发效率标准

固定 CUDA 构建机同机运行 5 次，记录 p50/p95；p95 使用 nearest-rank，五个样本时等于最慢
样本。cache、edit/fsync、Cargo argv 到 runnable-binary smoke 的精确边界按 G07 冻结：

| 场景 | p95 上限 |
|---|---:|
| no-op / 无内容变化 | `30s` |
| Rust model leaf edit 到可运行 dev binary | `90s` |
| runtime leaf edit 到可运行 dev binary | `90s` |
| 单个 core PTX 修改到可运行 dev binary | `120s` |
| 单个 Marlin/MoE native TU 修改 | `5min`，且只重编受影响 TU |
| clean official CUDA release build | `15min` |

正式 release 继续使用 LTO；开发 profile 禁止因 release LTO/link 阻塞反馈。大体量第三方
C++/CUDA 模板源码不得继续作为普通 workspace build 输入，必须走版本化 native operator
artifact 和独立 source-build lane。

## 10. 子目标与依赖

| ID | 文档 | 依赖 | 目标 |
|---|---|---|---|
| G00 | [`G00_BASELINE.md`](G00_BASELINE.md) | 无 | 冻结 legacy、真实六个主 lane、外部基线和编译基线 |
| G01 | [`G01_CORE_CONTRACTS.md`](G01_CORE_CONTRACTS.md) | G01A<-G00a；G01B<-G00+G01A；G01 聚合 A/B | 新核心 type/trait/program/plan/resource contract |
| G02 | [`G02_TEST_EVIDENCE.md`](G02_TEST_EVIDENCE.md) | G00,G01 | 先建立可信测试层次、artifact 图和 historical bug kill |
| G03 | [`G03_BACKEND_OPS.md`](G03_BACKEND_OPS.md) | G01,G02 | operation contracts、CPU oracle、CUDA/Metal providers |
| G04 | [`G04_RUNTIME_RESOURCES.md`](G04_RUNTIME_RESOURCES.md) | G01,G02,G03 | 共享 runtime、scheduler、资源事务与状态所有权 |
| G05 | [`G05_PRODUCT_API.md`](G05_PRODUCT_API.md) | G02,G04 | 唯一产品组合根和 OpenAI API 语义 |
| G06 | [`G06_OBSERVABILITY_PERF_LAB.md`](G06_OBSERVABILITY_PERF_LAB.md) | G01,G02,G04,G05 | 定位、replay、统一 profile 和性能实验协议 |
| G07 | [`G07_BUILD_NATIVE_OPS.md`](G07_BUILD_NATIVE_OPS.md) | G07A<-G00+G01；G07B<-G03+G07A；G07 聚合 A/B | crate/build graph、native ops、增量编译 |
| G08 | [`G08_MODEL_MIGRATION.md`](G08_MODEL_MIGRATION.md) | G03-G07；内部 A->B->C->D | 三主模型逐个迁移、parity、legacy 删除和长尾处置 |
| G09 | [`G09_PERFORMANCE.md`](G09_PERFORMANCE.md) | G00,G06,G07,G08 | 三模型双端性能恢复及 80% 外部线 |
| G10 | [`G10_RELEASE.md`](G10_RELEASE.md) | G10A<-G00-G09 dev PASS；G10A->fresh G08-RC/G09-RC->G10B | release freeze、候选 SHA 重验、发布、安装后回归和最终 PASS |

### Canonical gate 入口

所有阶段必须注册到现有 `scripts/release/run_gate.py`，不能形成一套可独立 PASS 的 sidecar：

```text
python3 scripts/release/run_gate.py vnext-g00 --out <out>
...
python3 scripts/release/run_gate.py vnext-g10 --out <out>
```

`run_gate.py --list-lanes` 必须列出 G00a、G00-G10、G01A/G01B、G07A/G07B、G08A-G08D、
G10A/G08-RC/G09-RC/G10B，以及 G10 定义的三模型 source/published/prepromotion lanes。
stage-specific validator 可以作为内部模块存在，但有效 PASS 必须来自 `run_gate.py` 写出的统一
`gate.manifest.json`。
manifest 记录 command、SHA、dirty、binary/model/config hashes、child artifacts 和 PASS line。

所有 canonical `--out` 必须解析到 Git 源码工作树之外。Goal 文档中的 artifact tree 都是相对
`<out_dir>` 的逻辑布局，不表示 collector 应直接写入未忽略的 `docs/release/`。collector、被测
legacy worktree 和 validator worktree 必须分别保持可辨识；不能通过忽略 output path、过滤整个
`git status` 或把证据预先放进源码树来伪造 clean source。需要随仓库保存的 compact manifest 或
结论在 gate 完成后另行提交，它只能引用 canonical artifact SHA，不能替代原始 artifact。

现有 G0 lane 与 vNext 重合时只执行一次并引用同一 artifact；禁止复制数据生成两个互相
独立的 PASS。`g0_release_summary.py` 和 completion manifest 必须把三主模型矩阵设为
v0.8.0 required input，不能把它留成可漏跑的旁路。

依赖 DAG。`G00a` 是只读 inventory、historical bug catalog、模型解析与 preset 锁全部通过
各自 validator 后形成的不可变 checkpoint；它只解锁 G01A 的 ADR、纯 contract 和无产品路由的
compile/unit/trybuild 工作，不代表 G00 PASS，也不能解锁 G01B、性能结论或模型迁移：

```text
G00a -> G01A
G00 + G01A -> G01B
G01A + G01B -> G01 -> G02 -> G03 -> G04 -> G05
                              |       |             |      |
                              |       +------------>G06<---+
G00 + G01 ------------------->G07A
G03 + G07A ------------------>G07B
G07A + G07B ----------------->G07

G03 + G04 + G05 + G06 + G07
  -> G08A -> G08B -> G08C -> G08D -> G08 -> G09-dev
  -> G10A-release-freeze
  -> G08-RC + G09-RC
  -> G10B-stage-publish-promote
  -> G10
```

`G10A-release-freeze` 生成唯一 `release_candidate_sha`，完成 version/release notes/workflow policy
修改、以 production workflow 冻结 staged tarball/binary SHA 并保持 checkout clean；未来 `v0.8.0`
tag 只能指向该 SHA。G08-RC 必须直接用 staged binary 在该 SHA 重跑完整六
lane correctness，G09-RC 必须在该 SHA 重跑全部正式 comparison。二者的 candidate binary SHA
必须相同，并与 staged Metal/CUDA tarball 中对应 binary SHA 一致；不一致时重新构建并重跑，不能
拼接旧 G08/G09 rows。G10B 只能消费 fresh G08-RC/G09-RC。G00 legacy binary 仍固定
`cff4c477...`，这是 comparator 身份，不受 candidate SHA 相等规则影响。

### G00a 事实检查点

G00a 的 canonical 输入必须是源码工作树外的两个真实 artifact：冻结 `cff4c477...` 由当前
checked-in analyzer 重算得到的 `coupling-inventory.json`，以及 clean current HEAD 通过实时
Hugging Face HTTPS 解析得到的 `model-resolution.json`。有效入口是：

```text
python3 scripts/release/run_gate.py vnext-g00a \
  --coupling-inventory <external-cff4-inventory.json> \
  --model-resolution <external-current-head-resolution.json> \
  --out <external-g00a-out>
```

checkpoint 必须冻结 12/12 model/backend lane、M1-M3 四类 generation preset、15 个 historical
bug family/28 个 concrete case 的 catalog 事实和完整 analyzer/catalog/goal source identity。collector
必须先把两个外部输入读取成不可变快照，再用 checked-in resolver 重新执行一次实时 HF 解析；
调用方解析与 live recheck 的完整 model facts 必须相等，二者都作为 artifact 保存；HF
`model-info`/`repo-tree` 原始响应必须可按 SHA/size 重放，safetensors index 的 weight map、shard
集合和编号必须完全一致，不能只比较自报 request 摘要。M2 Metal Q4_K_S 的 live LFS OID/size
还必须分别等于 model catalog 的 `expected_sha256`/`expected_size_bytes`；catalog 约束不能只被复制
进 lock 而不执行。产物为 `manifest.json`、
`model-facts.lock.json`、inventory 输入快照、resolution
输入快照、live resolution recheck 和四个 catalog 副本；8 个非 manifest artifact 必须被
SHA256/size index 100% 覆盖。historical 部分明确是 `catalog_only`，不冒充 G00 要求的完整
reproducer corpus。child validator 和统一入口分别必须打印：

```text
FERRUM RUNTIME VNEXT G00A FACT CHECKPOINT PASS: <out_dir>
FERRUM GATE vnext-g00a PASS: <out_dir>
```

任一 collector contract、model catalog、resolver、inventory analyzer、模型解析请求、目标文档、
Git HEAD/tree 或 clean 状态变化都会使 checkpoint stale。该 PASS 的机器可读 `unlocks` 只能是
`["G01A"]`；`G00`、`G01B`、模型迁移、性能和发布均必须位于 `does_not_prove`。
`model-facts.lock.json` 只保存 normalized facts/fingerprint，必须在相同事实下字节确定；带
`generated_at`、绝对路径和原始输入 SHA 的采集 provenance 只进入 manifest/index。

G01A 的设计或测试发现 G00a 锁定事实有误时必须废弃 checkpoint 并重采；G01 的最终 PASS
必须同时消费完整 G00 PASS 和 G01A artifact。G03/G04 的实现允许小型纵切验证，但 G08 前
不得批量迁移模型。G07A 可与 G02-G06 并行，
G07B 必须消费 G03 冻结的 operation catalog。
任何后续 Goal 修改已通过的核心 contract，都必须自动 invalidate 受影响的上游/模型 artifact
并重新运行，不能靠人工判断“应该没影响”。

## 11. 里程碑统筹

| 里程碑 | 包含 | 退出条件 |
|---|---|---|
| M0 事实冻结 | G00 | baseline artifact 完整，已支持/缺口不再靠口头判断 |
| M1 架构合同 | G01-G02 | contract、failure model、测试证据先于生产迁移 |
| M2 新运行时纵切 | G03-G05 | tiny real-weight vNext runtime 经统一 run/serve composition root 完成纵切；不提前宣称主模型迁移 |
| M3 可诊断可迭代 | G06-G07 | 一次失败可定位；dev compile 达标 |
| M4 三模型迁移 | G08 | 三模型 CUDA -> Metal，旧路径随模型删除 |
| M5 性能闭环 | G09 | legacy 不回退且三模型达到主流实现 80% |
| M6 发布 | G10A -> G08-RC/G09-RC -> G10B -> G10 | release-candidate SHA 重验后，`v0.8.0` 已发布且安装产物双端复验 |

CUDA 优先顺序：Qwen3.5-4B -> Qwen3.5-35B-A3B -> Qwen3-30B-A3B。Metal 在每个模型
CUDA contract 稳定后开始，但不得把三个 Metal lane 全部拖到发布前一次性补做。

## 12. 分支、提交和停止规则

- 每个子 Goal 使用小而可审阅的提交；核心 contract、kernel 优化、release gate 大改不得混在
  同一个 patch。
- 长期分支提交前执行 `git pull --rebase --autostash`，验证后及时 push。
- correctness 失败时停止性能 sweep。
- 同一 paid GPU failure class 连续两个 REJECT 后，必须回到 source/artifact 分析。
- 每个性能候选只验证一个主要假设，必须预先写 expected signal 和 reject threshold。
- 双轨期间 legacy/vNext 差异必须有 artifact；不能用输出过滤掩盖 token 或状态错误。
- 任何子 Goal 的 PASS artifact 在受影响代码变化后自动 stale。

## 13. 最终发布条件

G10 必须实际完成：

1. workspace version 升到 `0.8.0`，迁移说明和 release notes 完整。
2. 三个主模型在 Metal/CUDA 上 correctness 与 performance 全过，且 final G08-RC/G09-RC 均
   绑定未来 tag SHA 和 staged/published binary SHA。
3. Llama 8B-class dense supplemental release evidence 满足仓库政策。
4. unit、Metal、CUDA、tarball、Homebrew、release summary 和 completion gate 全过。
5. GitHub tag/release、正式资产和 checksum 已发布。
6. workspace crates 已按依赖顺序发布，crates.io 可查询 `0.8.0`，clean
   `cargo install ferrum-cli --version 0.8.0 --locked` 通过。
7. 从已发布 tarball/Homebrew 安装的 binary 再运行 `run` 与 `serve`。
8. 最终 `G0 RELEASE PASS: docs/release/g0/0.8.0` 存在。
9. `FERRUM RELEASE COMPLETION PASS` 引用发布后的资产和安装验证。
10. Docker 当前不维护：v0.8.0 Docker image/tag 发布数量 `0`，现有 tag-trigger workflow 已禁用。

禁止在第 5 步之前打印总 Goal PASS，也禁止把 source-ready、release-ready 或 staged asset
描述为已经发布。
