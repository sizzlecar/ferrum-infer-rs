# Runtime vNext v0.8.0 代码改动与回归范围执行计划（2026-08-12）

## 状态与边界

- 状态：Active execution plan。
- 本文件落实
  [`RELEASE_ACCELERATION_AMENDMENT_2026-08-06.md`](RELEASE_ACCELERATION_AMENDMENT_2026-08-06.md)
  的 change-impact 规则，以及
  [`CORRECTNESS_ACCEPTANCE_AMENDMENT_2026-08-07.md`](CORRECTNESS_ACCEPTANCE_AMENDMENT_2026-08-07.md)
  的“exact case -> affected scenario -> architecture sentinel -> milestone full lane”顺序。
- 本文件不修改 R0-R3 的验收分母、阈值或最终 PASS 条件。它只约束开发期间每个代码改动的
  回归范围，避免把“最终候选签字”错误地变成“每次改动后的全量回归”。
- R1/R2/R3 的正式 artifact freshness 与开发期回归是两件事：产品代码变化会使旧正式 artifact
  对最终聚合 stale，但这不授权立即重跑完整矩阵。先完成下述精确影响面验证，所有产品代码冻结后，
  每个阶段只执行一次正式矩阵。

## 1. 每个改动必须先写出的影响集合

测试命令启动前，先把改动映射为以下六个维度。未进入集合的维度不得因为“保险”而自动加入回归：

| 维度 | 可选值 | 判断依据 |
|---|---|---|
| 行为 | API/模板/采样、调度/资源、执行/provider、profile/trace、build/provenance、control-plane | 实际修改的 contract 和调用链，不按文件名猜测 |
| 入口 | `run`、`serve`、两者共享、collector/validator only | 修改点是否位于两入口共享的 resolved plan/runtime 之后 |
| 后端 | CUDA、Metal、两者共享、无 accelerator | 修改分支和实际 provider consumer |
| 架构 | M1 dense-hybrid、M2 hybrid-MoE、M3 full-attention-MoE、Llama dense | 修改的 op/state/model protocol 是否真的被该架构消费 |
| 模式/压力 | default/profile-off/basic/replay/full、c1/c4/c16/c32、cancel/resource pressure | 修改条件是否能在该模式或压力状态到达 |
| 证据 | 产品行为、性能、profile、build、artifact/control-plane | 改动可能改变的原始证据类型 |

每项排除必须有一句可证伪理由，例如“只改 CUDA provider，Metal 不编译该模块”，不能只写
“预计无影响”。如果调用链无法证明排除，先加一个便宜的 unit/contract assertion；不能直接用昂贵
全量矩阵代替影响分析。

## 2. 固定回归阶梯

每个改动只能逐层扩大，不能从源码改动直接跳到完整 R1/R2：

1. **L0 静态边界**：`git diff --check`、format/compile 或脚本语法检查，只覆盖改动文件。
2. **L1 exact reproducer**：一个能在旧代码失败、新代码通过的精确用例；并发测试必须先通过
   `bounded_command.py`。
3. **L2 affected unit/contract**：直接调用者、错误/清理分支和受影响 contract 的窄测试。
4. **L3 affected product sentinel**：最小真实模型、真实入口、真实后端命令，必须实际穿过修改分支。
5. **L4 affected group / architecture sentinel**：仅在 L3 通过后，并且调用链跨场景或跨架构时执行；
   只增加实际消费该分支的场景/架构。
6. **L5 milestone lane**：只在 R0/R1/R2 阶段退出或新的源码冻结点执行一次正式 lane/aggregate。
7. **L6 release candidate**：只对 exact staged/published bytes 执行目标要求的最终完整复验。

扩大范围必须满足至少一个条件：

- 前一层发现同一 failure class 出现在更宽调用链；
- 静态调用链证明修改点被另一入口、后端或架构实际消费；
- 到达目标文档规定的阶段退出/源码冻结点；
- 正式 validator 明确要求相同 source/binary 的最终签字。

“改了 `crates/` 文件”“二进制 SHA 变化”只说明旧正式 artifact 最终不能签新二进制；它本身不是
立即运行所有模型、后端和场景的扩大条件。

## 3. 改动类型到最小回归范围

| 类型 | 开发期必须回归 | 明确不自动回归 |
|---|---|---|
| 文档或 allowlisted control-plane | 相关 parser/self-test；用既有 artifact 重算 validator | 产品 build、GPU、R1/R2 性能 |
| collector/analyzer/validator | 脚本 self-test、构造正负例、对一个既有 raw artifact replay | 模型执行；除非发现 raw 字段本身缺失 |
| profile/trace 产品埋点 | exact unit、受影响 profile mode 的单个真实产品 sentinel、对应 analyzer | default correctness matrix、吞吐矩阵、无关 profile mode |
| 后端本地 provider/kernel | affected op/shape 数值测试；一个实际消费它的模型在该后端跑 `run`/`serve` sentinel；若改资源边界，再加一个 affected pressure cell | 另一后端、未消费该 provider 的模型、完整并发矩阵 |
| shared scheduler/admission/resource | exact 历史 case；直接 contract；每种实际不同执行结构各一个 sentinel；只有共享入口逻辑变化才同时测 `run`/`serve` | 不经过改动状态的 C01-C21 场景和完整性能矩阵 |
| API/template/sampling/structured output | exact case；affected scenario group；模型协议不同时各一个必要架构 sentinel；共享响应路径变化时覆盖 `run`/`serve` | 资源、build、无关模型/backend 全矩阵 |
| model-family loader/plan | 该 family 的 config/weight/plan test；该 family 一个 `run` 和一个 `serve` smoke；修改 shared op contract 时再按 provider 行扩展 | 其他 family 和无关后端 |
| build/native provenance | 受影响的一个 build scenario、cache invalidation 正负例、binary/receipt identity | 模型 correctness/performance，除非产物语义或 feature graph变化 |
| 默认产品语义或跨层公共 contract | exact case、affected unit、每个实际 consumer 类别的最小 product sentinel | 仍不直接运行完整阶段矩阵；到冻结点才执行 L5 |

当一个 patch 同时属于多行时取集合并集，但必须先拆分能独立提交的无关改动；不得把“可能以后会改”
计入当前影响集合。

## 4. 入口、后端和模型的扩大规则

### `run` / `serve`

- 修改发生在共享 `ResolvedModelPlan`/engine/provider 内，且 default 产品行为可达：L3 覆盖一个
  `run` 和一个 `serve` sentinel。
- 修改只在 `ferrum run --profile-detail ...` 的诊断命令组装、profile 文件或 replay bundle：只跑
  对应 `run` profile sentinel；不自动跑 serve correctness。
- 修改只在 HTTP/SSE、tool/schema 或 server response conversion：只跑 serve exact/affected group；
  只有共享 sampling/history 状态也改变时才加入 run。

### CUDA / Metal

- backend-local 文件或 capability 分支只跑该 backend。
- 共享 Rust contract 只有在两个 backend 都实际实现并穿过修改分支时，才各跑一个最小 sentinel。
- Metal 性能和内存敏感命令只允许在 owner 指定的 `22:00-09:00 Asia/Shanghai` 窗口；白天不能以
  “补保险回归”为理由启动。

### M1 / M2 / M3 / Llama

- M1 是公共 API canary，但不能替代 M2 hybrid-MoE/GPTQ 资源路径或 M3 full-attention-MoE 路径。
- 只有修改点涉及相应独有结构时才加入 M2/M3 architecture sentinel。
- Llama 只在 shared dense/attention/tokenizer/release entrypoint 真实受影响，或正式阶段退出要求时加入；
  Qwen-only profile bookkeeping 改动不自动触发 Llama。

## 5. 失败后的停止与扩大

- L1/L2 失败：不得启动 accelerator。
- L3 失败：保存 KEEP/REJECT artifact，回源码/原始 trace；不得升级到 affected group 或 full lane。
- 同一 paid-GPU failure class 连续两个 REJECT：禁止第三次 GPU 确认；必须先产生新的源码级因果预测。
- 非代码问题（网络、路径、认证、SSH quoting）只修复命令/传输并说明一次；不能因此重跑已经有完整
  receipt 和 raw artifact 的产品命令。
- 一个 focused case 通过只关闭该 failure class，不自动证明 full lane；一个 focused case失败也不使
  无关已通过 case 失效。

## 6. 正式 artifact 与源码冻结

1. 开发期允许多个 focused KEEP/REJECT artifact，各自绑定其 source/binary。
2. 修复已知 blocker 期间不运行 R1/R2 全量；先完成所有 L1-L4 影响面验证。
3. 产品源码冻结后，只运行一次 current-source R1 正式矩阵并取得 exact PASS。
4. R1 后禁止主动优化产品代码；只有 R2 暴露 release blocker 才解冻。解冻后仍先走 L1-L4，旧正式
   artifact 标记 stale，但不边改边反复重跑。
5. 再次冻结后才执行一次新的正式阶段矩阵。control-plane-only 且满足现有 source-closure allowlist 的
   修改只重算 validator，不重跑产品 raw evidence。
6. R3 staged binary 的完整复验仍按 active amendments 执行，不能由本计划缩减。

## 7. 当前 `65c965ef` 改动的精确范围

改动文件：
`crates/ferrum-models/src/executor/vnext_executor.rs`。

行为变化：只把“startup reusable ProgramId 的计算/登记”与“sealed catalog 安装后是否允许 direct
replay”解耦。catalog 尚未安装时，Kernel/Verification profile mode 也登记 identity；catalog
安装后仍保持原来的 direct-replay 禁止语义。没有修改默认采样、模型数值、调度/admission、资源预算、
HTTP/SSE 或 profile-off 吞吐路径。

当前开发期回归集合：

- L1/L2：`reusable_program_identity_is_recorded_before_catalog_installation`，真实绑定
  Kernel/Verification 两种 timing mode；已 bounded PASS `1/1`。
- L3：M1 CUDA、`ferrum run`、16-token、`--profile-detail full` 单条精确复现。
- artifact：
  `/workspace/artifacts/runtime-vnext-r2-diagnostic-m1-cuda-profile-full-65c965ef-20260812`。
- 结果：产品命令 `rc=0`，原先的 startup catalog reconciliation 错误消失；该 failure class 已关闭。
  独立 stage-coverage 结果为 `82.146967%`，门槛 `90%`，差 `7.853033` 个百分点，因此 profile
  coverage 仍为 REJECT。

当前明确不运行：R0、R1 六 lane、M1/M2/M3 correctness full matrix、R2 throughput cells、Llama、
Metal 性能、完整 profile sweep。下一步只从该 raw profile 定位 `7,800,332 ns` unattributed interval；
若修改 attribution/计时边界，仍只执行其 CPU exact test和同一条 M1 CUDA full-profile sentinel。

## 8. 每个后续改动的记录模板

```text
Change/commit:
Changed contract and call chain:
Affected behavior / entrypoint / backend / architecture / mode / evidence:
Explicitly excluded dimensions and reason:
L1 exact reproducer:
L2 affected unit/contract:
L3 product sentinel (if needed):
Escalation signal and stop condition:
Next milestone full gate (not run now):
Artifacts/receipts:
```

没有填完该模板，不启动新的 paid accelerator action；模板确定的 L1-L3 未通过，不扩大到 L4/L5。
