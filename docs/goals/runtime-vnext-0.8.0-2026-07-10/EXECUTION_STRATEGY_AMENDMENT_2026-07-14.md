# Runtime vNext v0.8.0 执行策略修订（2026-07-14）

## 状态与效力

- 状态：Accepted。
- 本修订只改变开发顺序、阶段依赖、进度口径和验证时机，不降低
  [`GOAL.md`](GOAL.md)、[`MODEL_MATRIX.md`](MODEL_MATRIX.md)、G08、G09、G10 的最终
  correctness、性能、双后端、发布和真实资产标准。
- 本修订优先于原文中“完整 G00 在任何 production refactor 前完成”、G01A/G01B 串行解锁
  G02-G08，以及“先完成全部证据设施再做生产纵切”的执行条款。原 artifact 的历史事实仍有效，
  但不能继续阻塞新的执行 DAG，也不能被重新解释为 production migration PASS。
- G00-G10 继续表示最终能力和验收维度；本文件中的 S0-S7 是实际开发 work package。两者不得
  再混为一个串行 DAG。

## 决策背景

2026-07-11 至 2026-07-14 的实现产生了约 46K 行 `ferrum-interfaces::vnext` contract、约
25K 行外部 contract tests 和约 19K 行 release/gate 脚本，但没有真实 vNext production caller。
完整 G00 的 collector、artifact freshness 和发布级请求矩阵先于生产纵切，导致大量时间用于证明
尚未接入的架构，而不是让 `ferrum run`/`ferrum serve` 消费它。

这不意味着动态资源、ownership、completion、profile 和测试合同应被简化掉。capacity-derived
admission、defer/resume、transaction/lease、fence 后回收、无全局 HOL、replay identity 等问题是
高性能连续批处理 runtime 的不可约复杂度。修订目标是先拆清所有权和依赖，再以真实产品路径验证并
激进重写，而不是以减少 LOC 为目标删除核心不变量。

## 不可妥协的最终标准

以下标准保持不变：

1. 三主模型为 Qwen3.5-4B、Qwen3.5-35B-A3B、Qwen3-30B-A3B；迁移顺序为每模型 CUDA 后
   立即 Metal，不把全部 Metal 回归拖到发布前。
2. `run`、`serve`、streaming usage、multi-turn、required/auto/streamed tools、tool result、
   `json_object`、strict `json_schema`、cancel/disconnect 和 concurrency required path 缺口为 `0`。
3. runtime concurrency 由 typed physical capacity 动态决定；配置值只是 ceiling。不得按模型名、
   GPU 名、显存档位或任意固定并发决定正常 admission。
4. 最终三模型 x CUDA/Metal correctness `6/6 PASS`，required failure、skip、waiver、stale 数均为 `0`。
5. G09 的 legacy no-regression、vLLM/llama.cpp 竞争线、正式 ABBA-BAAB 和置信区间标准不变。
6. G10 继续直接验证 frozen staged/published binary；candidate、tarball 和 published binary SHA
   必须一致。Docker 仍不发布。
7. Llama 8B-class dense 模型只作为 release policy 的 supplemental architecture evidence，不替代
   三主模型，也不改变主迁移优先级。

## 复杂度准入规则

复杂不是拒绝设计的理由，但每个复杂抽象必须至少满足以下一项，并在 contract map 中记录：

- 拥有一个不能由更低层可靠表达的 safety/correctness invariant；
- 删除一套真实的 model/backend/product 重复生命周期；
- 让已有 operation/provider 可被另一个模型直接复用；
- 提供可测量的吞吐、延迟、内存、编译或定位收益；
- 阻止已在 historical corpus 中出现过的错误构造或状态转换。

每个 public abstraction 还必须记录 owner、lifetime、production consumer 或目标 S milestone、
misuse prevented、hot-path overhead 和 compile invalidation domain。只有被自身 synthetic test 引用、
没有上述理由且没有目标 consumer 的 public API 必须删除。不得用单纯 LOC、type 数或 test 数作为
架构质量指标。

## S0：优先拆分现有 vNext contract

S0 是文档修订后的第一个实现任务。它分两步，不能合并成一次大改。

### S0A：保持语义的结构拆分

先运行 `inventory_tree.py` 并保存 inventory，然后仅移动/拆分模块和测试，不改变 public behavior、
状态机终态、capacity 计算或资源释放语义。概念所有权至少拆成以下边界；最终文件名可按依赖图调整：

1. identity/version/error；
2. semantic demand、work shape 与 storage contract；
3. capacity authority、physical accounting 与 elastic provisioning；
4. dynamic pool、extent/view 与 allocation quantum；
5. transaction、reservation、commit、rollback、lease 与 release epoch；
6. request/sequence/session lifetime；
7. step/batch/invocation authority、scratch 与 physical submission；
8. fence、completion、reaper、retry、recovery 与 quarantine；
9. event schema、profile identity 与 replay records；
10. 每个边界对应的 unit/property/fault/compile-fail tests。

结构门槛：

- `resource.rs`、`execution.rs`、`event.rs` 最终只能是 facade/re-export 或单一职责模块，单文件
  production logical LOC `<=2,500`，facade logical LOC `<=500`；generated code 除外。
- 单个 contract test target logical LOC `<=2,000`；测试按 invariant owner 拆分，不能只按数量均分。
- 新增循环模块依赖数 `0`；lower-level demand/capacity/pool 不得反向依赖 scheduler/product/model。
- public item 的 old path -> new owner map 覆盖率 `100%`；S0A 无意删除的 public item 丢失数 `0`。
- S0A 不新增 operation/model/product feature，不扩展正式 G00 collector，不运行 paid GPU。
- 每个拆分提交使用 bounded focused test；全部拆分完成后运行 bounded
  `cargo test -p ferrum-interfaces --all-targets`。同一 invariant 不因拆文件而重复执行多份等价测试。

### S0B：以真实用途驱动的语义重构

S0A 稳定后，允许对不合理 API、泛型边界、typestate、event schema 和 test topology 做 breaking
rewrite。S0B 必须与 S1 的 Qwen3.5-4B CUDA production consumer 同一里程碑完成，不能再次形成
只有 isolated contract/test 的大提交。

允许删除或合并现有 contract tests；保留标准是能证明 invariant、捕获 historical failure、保护
production consumer 或验证扩展成本，而不是保持当前 test count。新增 public contract 没有 live
consumer 或明确 S1 consumer 的数量必须为 `0`。

## 动态资源核心必须保留并进入生产

S0/S1 不得弱化以下合同：

- capacity 只来自已成功分配并安装的真实 backing segment；metadata 或剩余显存数字不能直接发布
  logical capacity；
- request 在 committed lease 前不得执行 provider encode、device submit 或 prefill；
- 暂时不可满足的请求返回 typed `Deferred`，永久超过物理上限的请求返回 typed `Impossible`；
- active decode 和其他 eligible waiting request 不因队首大请求而停止；全局 capacity HOL block 为 `0`；
- 所有提高 effective capacity 的 release/grow transition 递增 monotonic release epoch 并封闭
  register/recheck lost-wakeup 窗口；
- multi-resource reserve/commit 的第 N 项失败必须逆序补偿，rollback/release/cancel 幂等；
- invocation/extent/sequence ownership 持有到 typed fence terminal；possibly-submitted 不得直接重试
  或提前归还资源；
- normal request thread 不执行 blocking fence wait；reaper/recovery/quarantine 保持 owner 可达；
- plan descriptor 保持 `O(graph)`，steady decode host allocations/token=`0`，正常 admission 的
  device allocations/request=`0`；
- dynamic policy、chunked prefill、preempt/recompute 都通过 typed config/preset 暴露，隐藏环境变量
  和模型/GPU 特判数为 `0`。

## 生产纵切执行 DAG

```text
S0A contract/test structural split
  -> S0B + S1 Qwen3.5-4B CUDA basic production slice
  -> S2 Qwen3.5-4B CUDA complete product contract
  -> S3 Qwen3.5-4B Metal + M1 legacy deletion
  -> S4 Qwen3.5-35B-A3B CUDA -> Metal + M2 legacy deletion
  -> S5 Qwen3-30B-A3B CUDA -> Metal + M3 legacy deletion
  -> S6 legacy zero + full correctness/performance/build/diagnostic evidence
  -> S7 staged assets -> publish -> installed-asset regression
```

S1 必须让实际 Qwen3.5-4B CUDA 请求经过：

```text
typed source/config/weights
  -> ModelProgram
  -> capability/provider resolution
  -> immutable ExecutionPlan
  -> shared dynamic ExecutionRuntime
  -> CUDA operation providers
  -> ferrum run + ferrum serve
```

允许通过单向 provider adapter 复用已验证 kernel，但不得回退到 legacy scheduler、model-specific
unified runner、KV/resource manager 或两套 product composition。S1 至少携带 basic/resource profile，
否则不能进入 S2。

S2 完成 stream/tools/schema/multi-turn/cancel/concurrency 和 M1 CUDA historical resource failures。
S3 使用同一 ModelProgram 增加 Metal providers/GGUF mapping，并在 dual-backend milestone 内删除 M1
legacy entry。S4/S5 每完成一个模型立即删除对应 legacy factory/runner/flag/fallback；adapter 必须有
当前里程碑 sunset，不能延后到总发布。

## G00 分层

完整 G00 不再阻塞 S0-S5：

- `G00F`：facts checkpoint。只包含 source/coupling inventory、model/revision/weight facts、preset、
  historical catalog 和最小入口事实；它是 S0/S1 唯一 baseline 前置。
- `G00M1/G00M2/G00M3`：逐模型 just-in-time baseline。在删除该模型 legacy 前采集必要 token、
  resource、correctness 和 performance-smoke rows。
- `G00P`：当前完整六 lane external/legacy、ABBA-BAAB、build timing 和 artifact authenticity 基线。
  它只阻塞 G09 正式性能结论和 G10 发布，不阻塞生产架构实现。

现有 G00 root、partial lane 和 G01A artifact 保留为 historical/diagnostic input。策略或 source 变化后
它们可以标记 stale，但不得删除、复制成新 PASS 或继续作为 S0-S5 的硬 blocker。

## 测试与 profile 的激进重构

测试和 profile 是新架构的一部分，不是 release 阶段附加脚本：

- L0 plan/state/oracle/property/fault tests warm `<=60s`；
- L1 tiny real weights + reference runtime warm `<=5min`；
- product-visible change 的开发 gate 必须同时执行 `run`/`serve` focused smoke；
- PR/change-impact required gate p95 `<=10min`，不包含三模型正式性能；
- 每个 S milestone 只执行受影响的 actual-model/backend correctness 和低成本 performance smoke；
- 完整 C01-C21、正式 ABBA-BAAB、置信区间和 published asset matrix 只在 S6/S7 执行；
- basic profile 在 S1 可用，resource/latency 在 S2 可用，kernel/replay 在 S3 前可用；
- identity 必须贯穿 run/request/sequence/plan/node/op/resource/backend/kernel，隐藏 profile env 数 `0`；
- basic profile overhead `<=2%`，resource/latency `<=5%`，自动 bottleneck report `<=60s`。

focused validation 不是降低 correctness。任何当前纵切的 basic correctness、resource ownership 或
`run`/`serve` smoke 失败都阻止进入下一 S milestone；只是未受影响的完整 release matrix不再每次执行。

## 编译反馈并行线

G07A 在 S1 后立即与 S2/S3 并行，S4 前必须证明 model/runtime/native invalidation 已分离。目标保持：

- no-op `<=30s`；
- Rust model leaf `<=90s`；
- runtime leaf `<=90s`；
- single core PTX `<=120s`；
- single Marlin/MoE TU `<=5min`；
- clean official CUDA release `<=15min`，使用已验证 versioned native artifacts。

模型 program 修改不得运行 nvcc。Metal-only 修改不得使 CUDA provider/native artifact dirty，反向同理。

## 进度与提交规则

状态报告按以下产品事实排序，而不是按已写脚本或 elapsed time：

1. vNext production entry：`run`/`serve`；
2. 已迁移 model/backend：`n/6`；
3. 当前 actual correctness、性能差距和 compile p95；
4. 已删除 legacy executor/factory/flag/fallback；
5. 最新 artifact、失败分类和下一个停止条件。

S0A 使用多个可独立 review/revert 的 mechanical commits；S0B、runtime semantics、provider、product
wiring、test/profile 和 gate 不得重新压成数万行单提交。长分支验证后及时 commit/push。除修复已复现
的 runner blocker 外，不允许连续两个以上 gate-only commit；正式 collector 工作在 S6 前不得再次
取代 production milestone。

## Artifact 与 PASS 语义

本修订生效后：

- 已有 `FERRUM GATE vnext-g01a PASS` 只证明历史 isolated contract checkpoint；
- 已有 G00 `collecting`/partial roots 只证明其中实际保存的事实；
- S0A 只证明结构拆分和既有语义保持，不证明 production migration；
- S1 以后只有实际 `run`/`serve` production artifact 才能证明纵切；
- G08/G09/G10 的最终 PASS line 和 release completion 语义完全不变。
