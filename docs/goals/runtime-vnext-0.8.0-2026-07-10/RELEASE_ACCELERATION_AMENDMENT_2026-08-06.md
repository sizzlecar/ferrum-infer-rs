# Runtime vNext v0.8.0 发布加速修订（2026-08-06）

## 状态与效力

- 状态：Active。
- 本修订响应“缩短 v0.8.0 发布周期”的目标调整，优先于 [`GOAL.md`](GOAL.md)、
  [`EXECUTION_STRATEGY_AMENDMENT_2026-07-14.md`](EXECUTION_STRATEGY_AMENDMENT_2026-07-14.md)
  和 G00-G10 中与发布阻塞依赖、阶段重复验证、全仓治理完成时机冲突的条款。
- 性能范围随后由
  [`PERFORMANCE_ACCEPTANCE_AMENDMENT_2026-08-06.md`](PERFORMANCE_ACCEPTANCE_AMENDMENT_2026-08-06.md)
  进一步收敛；该文件优先于本修订中的 external comparator、ABBA 和 ratio 条款。
- R1 correctness 重复采样规模随后由
  [`CORRECTNESS_ACCEPTANCE_AMENDMENT_2026-08-07.md`](CORRECTNESS_ACCEPTANCE_AMENDMENT_2026-08-07.md)
  收敛；该文件优先于本修订中要求三个模型各自执行同一 exhaustive 分母的条款。
- 本修订不降低真实产品正确性、动态资源、profile、编译效率、CUDA/Metal 或发布后安装验证标准。
- G00-G10 的未完成项目不删除。未列为本修订 v0.8.0 硬门的项目转入 v0.8.1/0.9 hardening backlog，
  不能被误报为已经完成。
- v0.8.0 的正式进度分母从 `G00-G10 0/11` 改为下文 `R0-R3 0/4`。旧 `0/11` 只保留为原始
  exhaustive roadmap 的历史口径，不再阻止 release freeze。
- 总目标仍只有在实际发布后的安装产物通过最终验证并打印
  `FERRUM RUNTIME VNEXT V0.8.0 RELEASE GOAL PASS` 后完成。

## 调整原因

从 2026-07-10 到 2026-08-06，工程已经实现共享 execution/resource contracts、真实 CUDA/Metal
产品纵切、703/702-case 历史矩阵、typed profile/replay、native artifact 和大量 canonical gate。
但原目标把四类工作串成同一个发布前分母：

1. 核心架构和三主模型产品迁移；
2. 完整测试、profile、benchmark 和 build 平台建设；
3. 全仓历史 mutation、所有 provider conformance、support disposition 与 legacy 物理清零；
4. 最终 staged/published asset 发布。

该依赖图使 focused artifact 在无关源码或 gate 变化后反复 stale，并让中间阶段重复执行最终级全量矩阵。
发布加速的原则是削减证明广度和重复验证，不削减用户实际运行的质量。

## v0.8.0 不可删减硬门

以下要求继续阻塞发布：

1. 主三模型固定为 Qwen3.5-4B、Qwen3.5-35B-A3B、Qwen3-30B-A3B。
2. 三主模型 CUDA/Metal correctness 共 `6/6 PASS`；M1 使用 `703/702` 公共全矩阵，M2 使用
   `112/111`、M3 使用 `120/119` 架构差异矩阵。required failure、skip、waiver、stale 均为 `0`。
3. 三主模型及其共享 Qwen production path 的 legacy entry、factory、runner、fallback 在 release
   binary 中不可达，运行时 legacy selection 次数为 `0`；不得用 hidden env、模型名、GPU 名或固定
   并发切换到另一套实现。非三主模型的既有 legacy path 可以冻结保留到 post-release，但新增 legacy
   adapter、调用点、feature、fallback 和隐藏开关数量必须为 `0`。
4. `ferrum run` 与 `ferrum serve` 均覆盖 multi-turn、stream、唯一 `[DONE]`、usage、Unicode、
   tool required/auto/streamed/tool-result、`json_object`、strict `json_schema`、cancel/disconnect 和并发隔离。
5. 动态资源必须在 kernel submit 前完成 admission；暂时不足使用 typed defer/wait/resume，active decode
   和后续 eligible request 继续推进。OOM、panic、livelock、泄漏、提前 fence 回收和全局 HOL 均为 `0`。
6. CUDA 正式性能保留 c1/c4/c16/c32 和三次 repeat；Metal 保留 c1/c4/c16 和三次 repeat。
   两端按性能收敛修订执行绝对可用线、Ferrum 自身非回退、并发、稳定性和内存门；外部比较不阻塞
   v0.8.0。
7. profile-off 是产品性能真值；公开 typed profile 开关关闭时不改变执行路径。basic/replay/full 必须能把
   请求关联到 plan/node/op/resource/provider/kernel，不能再靠反复排除法定位主瓶颈。
8. G07 的关键编译阈值继续生效：no-op `<=30s`，Rust/runtime leaf `<=90s`，core PTX `<=120s`，
   native TU `<=5min`，clean official CUDA release `<=15min`。
9. Llama 8B-class dense 模型继续作为 accelerator release supplemental evidence，不替代三个主模型。
10. 最终必须验证 exact staged/published Metal/CUDA binary、tarball、Homebrew、crates.io 和 clean install；
    Docker 继续不发布。

## 四阶段发布 DAG

```text
R0 Core Closure
  -> R1 Product Correctness
  -> R2 Performance / Profile / Build
  -> R3 Freeze / Assets / Publish / Installed Regression
```

只有 R0-R3 是 v0.8.0 release-blocking 分母。G00-G10 文档继续提供要求来源和后续 hardening backlog，
但不得再要求它们各自产生 exhaustive aggregate PASS 后才进入 R3。

## R0：Core Closure

目标是在不新增通用 gate 平台的前提下，关闭当前 Qwen3.5-4B shared core 的已知产品和数值缺口。

必须满足：

- 当前 clean source 的 G08A source ownership PASS；五类 lifecycle 只有 shared runtime owner。
- 当前 clean source 的 G08A op、linear-attention、full-attention、full-model、token parity 和
  same-history numerics PASS；阈值仍只来自 checked-in tolerance catalog。
- 当前 clean source 的 M1 CUDA determinism、response-format、API/modality、Unicode stream、
  tool/schema、multi-turn/concurrency、latency/first-failure 和 H02.1/H12.1-H12.4 resource set PASS。
- focused `run` 与 `serve` 实际加载同一 resolved plan/runtime，production legacy selection 为 `0`。
- 任何只影响 artifact packaging、但 raw product/numerical evidence 已通过的 blocker最多允许一次
  source-level authenticity 修复；不得继续扩建新的通用 validator framework。

R0 canonical checkpoint 可以复用现有 child validators，只允许做薄聚合：

```text
FERRUM RUNTIME VNEXT R0 CORE CLOSURE PASS: <out_dir>
FERRUM GATE vnext-r0 PASS: <out_dir>
```

## R1：Product Correctness

目标是先让所有必须发布的实际模型和入口正确，再运行任何正式性能矩阵。

必须满足：

- M1/M2/M3 x CUDA/Metal C01-C21 `6/6 PASS`，每 lane 使用真实 model weights、产品默认 typed config、
  `run` 和 `serve`；精确分母按 2026-08-07 correctness 修订执行，不得把 focused sentinel 冒充
  对应的 `703/702/112/111/120/119` 正式分母。
- CUDA/Metal 修复共享源码后使用 change-impact 选择 affected sentinel；只有 affected sentinel 通过后才
  运行该阶段的一次 full matrix。单 case 失败不得从 case 0 重跑。
- 三模型 resolved plans 中 supported provider 均有实际 execution/conformance evidence；未被三模型解析到的
  provider 不阻塞 0.8.0。
- resource final state、cancel/disconnect 后容量复用、capacity pressure 等产品关键不变量全部通过。
- Llama 8B-class supplemental `run`/`serve`/stream correctness 通过。
- 三主模型及共享 Qwen production legacy entry/factory/runner/fallback 在 release binary 中不可达，
  运行选择为 `0`；其他已支持架构的既有 legacy source physical zero 转入 post-release，期间禁止新增。

```text
FERRUM RUNTIME VNEXT R1 PRODUCT CORRECTNESS PASS: <out_dir>
FERRUM GATE vnext-r1 PASS: <out_dir>
```

## R2：Performance、Profile 与 Build

R2 只允许在 R1 correctness 通过后形成正式性能结论。

必须满足：

- 三主模型 CUDA/Metal 按性能收敛修订的 Ferrum-only workload、同硬件和固定
  model/config/dataset/binary 执行。
- CUDA c1/c4/c16/c32 和 Metal required cells 的 completed request、usage-token source、CI、active timeline、
  TTFT/TPOT/throughput/ITL eligibility 字段完整率为 `100%`。
- 每 cell throughput median `>=0.95x` 冻结自身基线、model/backend 几何平均 `>=1.00`，
  TTFT/TPOT p95 `<=1.10x` 自身基线，三次 repeat CV `<=8%`；缺历史基线的 cell 还必须达到
  性能收敛修订中的模型/backend 绝对可用线。
- 性能优化只能由现有 profile/replay artifact 指向的单一 stage/provider/op 假设启动。每个候选先跑 focused
  correctness，再跑 bounded A/B；两次相同 failure class 后停止 GPU sweep，回到源码分析。
- profile-off 产品性能不因诊断功能存在而回退。basic profile overhead 以用户已接受的 `<=7%` 为
  v0.8.0 release 门；超过 `2%` 继续记录为 hardening target。replay/full 不产生产品吞吐 claim。
- G07 上述六个关键编译场景达到阈值，并保存实际命令、cache 状态、binary SHA 和至少一次 cold/hot
  可重现实测；其余五样本全组合矩阵转入 hardening。

```text
FERRUM RUNTIME VNEXT R2 PERFORMANCE BUILD PROFILE PASS: <out_dir>
FERRUM GATE vnext-r2 PASS: <out_dir>
```

## R3：Freeze、发布与安装后回归

R3 仍是实际发布，不是 release-ready 声明。

1. G10A 只依赖 fresh R0、R1、R2 manifest，完成 version `0.8.0`、migration guide、release notes、
   support/performance report、workflow policy 和唯一 release-candidate SHA。
2. production workflow 以 `publish_release=false` 构建并冻结 Metal/CUDA/CPU staged tarball；后续禁止重编。
3. exact staged Metal/CUDA binary 在 release-candidate SHA 上重新执行一次最终三主模型 correctness 和
   performance；这是唯一必须全局 same-SHA/same-binary 的全量重验。
4. source、tarball、Homebrew、Llama supplemental、crates.io package 和 release completion gates 通过。
5. 创建 annotated tag 和 GitHub prerelease，发布相同 bytes；从实际资产和 clean install 重新验证
   `ferrum run`/`serve`。失败保留 failed prerelease，修复进入新版本，禁止重写 tag。
6. promotion 后确认 asset id/SHA 不变，打印 G0 release、completion 和总 Goal 三条最终 PASS。

```text
FERRUM RUNTIME VNEXT R3 V0.8.0 PUBLISHED PASS: <out_dir>
FERRUM GATE vnext-r3 PASS: <out_dir>
FERRUM RUNTIME VNEXT V0.8.0 RELEASE GOAL PASS: <out_dir>
```

## 转入 v0.8.1/0.9 的工作

以下项目不删除，但不再阻塞 v0.8.0：

| 原范围 | v0.8.0 保留 | Post-release backlog |
|---|---|---|
| G00 | 正式模型/cell 使用的 legacy 与 external baseline | 六 lane 通用 collector、全部 build/perf 组合和完整 red-team 平台 |
| G01 | shared owner、hot-loop 边界、真实 dense/hybrid/MoE consumer | 未被发布路径需要的全部 synthetic 扩展演练聚合 |
| G02 | release-critical historical failures、current product scenarios | 16/16 family 全 mutation/replay、完整 planner precision/PR p95 平台 |
| G03 | 三主模型 resolved plan 实际 providers | 未使用 provider、novel-op 和全 catalog dispatch microbench |
| G04 | admission/defer/release/fence/accounting 等产品 safety invariants | 非发布分母的扩展 fault permutations 与单独 exhaustive aggregate packaging |
| G05 | 两个产品入口和发布 API 矩阵 | 未被发布场景消费的完整 target-segment ledger |
| G06 | 可定位实际 top stage/op 的 typed trace/profile | 全历史自动 replay、完整 coverage dashboard 和自动优化报告 |
| G07 | 六个关键编译阈值和 native artifact 链 | 所有场景五样本全组合与非发布 native op |
| G08D | 三主模型及共享 Qwen legacy production path 在 release binary 中不可达且选择为零；Llama supplemental | 非三主模型全仓 support disposition、Coder/DeepSeek 全迁移、其余 legacy physical zero |
| G09 | 正式发布矩阵和竞争阈值 | 非发布 cells、探索性 kernel 候选和长期性能实验 |

## 验证与提交节奏

- 每个源码变化执行：exact reproducer -> affected unit/contract -> affected backend product smoke。
- 只有 R0/R1/R2 阶段退出和 R3 release candidate 执行 aggregate/full gate。
- 单 case 失败只重跑该 case 和 affected group；focused 通过前禁止从头运行完整 702/703 matrix。
- 开发期 focused artifact 不要求与无关 gate/doc 变化共享全局 SHA；是否 stale 由输入闭包 hash 和
  change-impact 决定，禁止人工豁免。R3 必须全局 same SHA/tree/binary。
- 连续 gate-only 提交最多 `1` 个；下一提交必须关闭真实产品、性能、编译或发布缺口。
- 每个 30 分钟窗口继续要求 diff、test result、commit 或 diagnostic artifact。
- 提交前 `git pull --rebase --autostash`，focused 验证后及时 push；不积攒大提交。
- 单一 CUDA lane 同时最多一个 potentially billable instance；模型、build cache 有价值时按有界窗口复用，
  无后续 bounded action 时立即停止计费。

## 时间盒与升级规则

| 阶段 | 规划时间盒 | 超时处理 |
|---|---:|---|
| R0 | 2-3 工作日 | raw 产品/数值正确但 gate packaging 阻塞时降级为 hardening；产品或数值错误继续阻塞 |
| R1 | 7-12 工作日 | 按失败模型/backend 聚焦；跨模型/后端重复时升级为 systemic architecture |
| R2 | 5-8 工作日 | 两次同类性能 REJECT 后停止 sweep，只保留有 profile 因果预测的候选 |
| R3 | 3-5 工作日 | 任一 correctness/asset identity 失败立即停止发布，修复后生成新 RC |

最快路径约 3-4 周，规划承诺为 4-6 周。时间盒不能豁免 correctness、性能或发布硬门，只能阻止
非产品证明、工具完善和无因果性能试验继续占用关键路径。

## 完成口径

- `R0-R3 4/4 PASS` 前不得声称 v0.8.0 完成。
- R0-R2 只表示 development checkpoint，不表示 release-ready 或已经发布。
- Post-release backlog 未完成必须在 release notes/support matrix 中明确，不得宣称原 G00-G10
  exhaustive roadmap 已完成。
- 只有发布后的 exact asset 和 clean install 证据使总 Goal 完成。
