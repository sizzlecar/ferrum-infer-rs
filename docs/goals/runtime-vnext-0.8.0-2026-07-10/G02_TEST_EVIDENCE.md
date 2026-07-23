# G02: 测试架构、证据图与 Historical Bug Kill

## 状态与依赖

- 状态：Open
- 依赖：L0 随 S0A contract split；L1/change-impact 随 S1-S2；full historical/matrix evidence 在 S6
- 下游：S1-S7、G03-G10

## 目标

随真实生产纵切建立可信测试层次和门禁依赖图，而不是先建设完整 release 设施再接入生产。
PASS 必须证明真实命令执行过，而不是仅证明 JSON 中写了 `PASS`。本 Goal 吸收并激进重构现有
test-architecture、release-hardening、scenario 和 change-impact 资产，不再创建平行的第四套 gate。

## 分阶段交付

- S0A：测试按 contract invariant owner 拆分，单 target logical LOC `<=2,000`，bounded focused
  test 和 aggregate interface test 通过；不追求保留当前 test count。
- S1：L0/L1 与 Qwen3.5-4B CUDA basic `run`/`serve` smoke 接入 change-impact planner。
- S2-S5：每个 model/backend 只补齐受影响的 actual scenarios 和 historical mutations。
- S6：完成本文件后续的 full historical corpus、mutation、matrix consumer 和 PR timing PASS。

测试可以因架构重写而删除、合并或重写；保留依据是 invariant coverage、historical kill、真实产品
consumer 或 backend conformance，而不是已有测试数量。任何 product-visible change 同时计划
`run`/`serve` 的规则从 S1 开始生效。

## 失败后的分层回归与证据复用

开发阶段不得因一个 case 失败立即从 C01 重跑完整 C01-C21。统一采用三层调度：

1. `exact replay`：先复跑失败 case 或最小 historical reproducer，保存失败前后同口径 artifact。
2. `change-impact`：由 checked-in path/domain 规则选择 L0/L1、`run`/`serve` product scenario 和
   actual-model/backend 的受影响纵切；tool protocol、stream、structured output、sampling/EOS、
   resource/admission 等 capability 必须显式出现在计划中，未知路径 hard fail。
3. `full matrix`：代码冻结后在 S6/S7、正式性能前和发布候选阶段执行完整 C01-C21；同一冻结候选
   只因正式 artifact 自身失败或代码再次变化而重跑。

断点或分片证据只有在 source git SHA、dirty status、binary SHA256、model revision/files、
effective config、runner/dependency SHA 和输入 SHA 全部相同时才允许聚合。代码变化后，旧分片只能
作为诊断输入；最终 full-matrix PASS 禁止混合不同 evidence identity。基础设施中断可在相同 identity
下继续，产品正确性失败后的修复必须先跑 exact replay 和受影响纵切，再等待下一冻结点做全量。

量化门：

- warm exact replay（非并发压力 case）`<=2min`；
- PR/change-impact required gate p95 `<=10min`，并同时包含所需 `run`/`serve` scenario；
- planner 未分类路径数 `0`，漏选入口/backend/capability 数 `0`；
- 非 C09/C18 case 不重复嵌入 scheduler trace 行，资源语义证据缺失数 `0`；
- 不同 evidence identity 的分片被聚合器拒绝率 `100%`；
- 完整 C01-C21 仍保持 MODEL_MATRIX 精确分母与最终 validator，不以 focused PASS 替代。

## 测试层次

| 层 | 内容 | 时间目标 | 实现/消费 Goal |
|---|---|---:|---|
| L0 | plan/config/shape/state machine/oracle/property/fault tests | warm `<=60s` | G02 建立并硬化 |
| L1 | tiny real weights + G01 reference vNext runtime，无 mock-owned 关键资源 | warm `<=5min` | G02 建立；G03-G06 持续消费 |
| L2 | Qwen3.5-4B actual `run+serve`，CUDA/Metal | each `<=20min` warm | G08 首次必须通过；G02 只定义 runner/schema |
| L3 | 三主模型完整 correctness matrix | 按模型预算 | G08 |
| L4 | 三主模型性能与外部基线 | 仅 correctness 后 | G09 |
| L5 | tarball/Homebrew/发布后安装 | release only | G10 |

G02 PASS 只证明 L0/L1 和 L2-L5 的 runner、schema、依赖图、negative fixtures 是可信的；
不得提前声称 Qwen3.5 actual vNext lane 已通过。G02 使用 G00 legacy actual artifacts 验证
L2-L5 artifact consumer 能拒绝缺入口、缺 backend、stale 和伪造结果。

## Historical bug corpus

以下 H01-H15 是稳定的 bug family 分母。每个 family 内逗号分隔的独立失败必须拆为
`Hxx.y` concrete case；每个 case 保存 source commit、原始 artifact/issue、最小 reproducer、
受影响入口/backend、expected failure class 和 mutation/revert patch。family 只有在所属 concrete
case 全部 kill 时才算 1/15。G00 冻结 concrete case 总数 `M`；后续只能追加 case，不能合并或
删除失败来提高通过率。

至少包含以下 15 个 family，使用真实最小输入和预期失败层：

- H01：run/serve auto-config 未物化导致 CUDA MoE 慢路径。
- H02：serve 默认值导致 run context-full/swap/hang。
- H03：UTF-8/BPE 逐 token 解码乱码。
- H04：stream U+FFFD。
- H05：EOS/stop 泄漏。
- H06：position offset 错误。
- H07：CUDA graph padded key/stale slot。
- H08：scratch realloc 后未初始化。
- H09：Metal 调用 CUDA-only varlen primitive。
- H10：shared default dtype 把 CUDA f16 residual 变 f32。
- H11：Metal head_dim=256 accumulator 越界/NaN。
- H12：KV/admission OOM、defer、rollback、release 不闭合。
- H13：missing/duplicate `[DONE]` 或 usage。
- H14：Qwen3-Coder CUDA 首 token EOS/empty answer。
- H15：Qwen3.5 c32 recurrent/KV resource livelock 或错误复用。

## Gate graph

- 顶层 self-test 必须机器枚举所有注册子 gate；漏接线数量为 `0`。
- 当前失败的 `product_observability_wiring_gate.py --self-test` 必须被修复并纳入顶层。
- change-impact planner 接入 PR CI、`run_gate.py` 和 release manifest。
- 建立至少 100 个历史/合成 change fixture 的 gold set，覆盖 shared contract/runtime/product、
  CUDA provider/native op、Metal provider、model/template、gate-only 和 docs-only；每项保存预期 lane。
- artifact 记录 parent/child gate、输入 SHA、产出 SHA、invalidated_by 和 freshness。
- required probe 缺失、skip 或 placeholder -> `INCOMPLETE/FAIL`，绝不能 PASS。
- matrix step 必须引用真实 command/log/artifact；文字计划不算执行。
- correctness runner schema 必须把 MODEL_MATRIX 的 case 分母作为 exact partition，而非只有总数：
  C14 `P_NO_THINKING` required/type/additionalProperties/enum=`13/13/12/12`、`P_THINKING`
  四类各 `5`；C16 invalid tool/schema/stream option/model/context 各 `6`。任一 variant 被另一 variant
  的重复 case 补足总数时 hard fail。
- C01 四类各 `5`；C06 每个 stream case 必须有 matching C05 ref，C12 每个 ordinal 必须有同
  model/preset/ordinal 的 matching C11 auto-tool ref；C07 必须是
  6 conversation x 5 history-carrying rounds；C17 每种字符 run/stream=`10/10`；C19/C21 使用
  MODEL_MATRIX 的 exact partition。C21 五组各 `4`；required-tool 组同时携带 strict response format
  并确定性选择 tool priority，standalone strict-schema 组单独成功。重复 stateless payload、
  non-stream 冒充 stream 或 plain prompt 冒充 tool/schema 都必须失败。

## 验收

- historical corpus `15/15` family、`M/M` concrete case 都有证据引用、最小 fixture、期望层和
  failure class，orphan/重复 evidence 数 `0`；validator fixture kill rate `100%`。至少
  `10/15` family 还必须在冻结 legacy/replay 输入上重现。
- G08 必须把 `15/15` family 和 `M/M` concrete case 全部提升为 vNext production
  mutation/revert-to-bug kill；G02 的
  synthetic fixture PASS 不能替代该下游条件。
- 顶层 self-test 注入任一子 gate failure 时，顶层 `100%` fail。
- C14/C16 每个 partition 分别做少 1、重复 evidence、错 preset 和复用 payload mutation，validator
  kill `100%`；C14 四类 payload/schema fingerprint 去重率 `100%`。
- C01 缺 golden bytes、semantic/tokenizer source lock、runtime `resolution_evidence` 或真实 product
  fail-closed receipt，unknown case 仅由普通缺文件/权重失败代替；C06/C12 只保存 pair id、未从
  persisted input 重算 C05/C11 canonical payload deep-equal，或同时伪造 pair 两侧和 registry；C07
  不携带 history、C12 未重组
  tool JSON、C17 无 raw stream chunk、C19 缺 hard/soft 冲突或 history、C21 缺任一 4-case 子组、
  required-tool 未选择 tool priority、arguments/schema invalid、content 伪造 schema JSON，或 standalone
  schema 未成功的 mutation kill `100%`。
- SSE byte stream 的每个单切点/逐字节切分语义必须一致；缺 ITL evidence、错 cardinality/source、
  usage/event 或 interval 不一致仍声称 eligible、coalescing 伪装 eligible、repeat totals 伪造、
  ineligible ITL 仍生成 ratio 的 mutation kill `100%`。transport coalescing 同时伪写 server
  `stream_bulk_flush` 必须被拒绝。
- 手写 `status=PASS`、stale SHA、空 log、错误 binary hash、少入口、少 backend fixture
  均被拒绝。
- parent hostile `FERRUM_*` 注入、child 继承 hidden product env、保存 env 与实际 process receipt
  不一致的 mutation kill `100%`；release scenario child 只能消费 typed CLI/config/preset。
- shared runtime 改动的 planner classification coverage `100%`。
- change-impact gold set required-lane recall `100%`（漏跑 Metal/CUDA/run/serve 数 `0`），precision
  `>=95%`；backend-local 实现改动不得无理由触发另一端 full gate，但 shared contract/op schema/
  model/runtime/product 变化必须自动计划 CUDA+Metal，并在 manifest 中写 invalidation 原因。
- PR required gate p95 `<=10min`，且不包含 paid full-model performance。
- L1 关键资源路径使用 MockKv/StubLlm 计数 `0`；它使用 G01 reference runtime 和 tiny real
  weights。mock 只能测试 API adapter 自身。
- 所有 product-visible 变化同时计划 `run` 与 `serve`。
- `.github/workflows/ci.yml` 中不存在已删除 crate、无效 `continue-on-error` correctness step。

## 产物与 PASS

```text
docs/release/runtime-vnext/0.8.0/g02-test-evidence/
  gate-registry.json
  gate-graph.json
  historical-bugs/
  mutation-report.json
  ci-timings.json
  planner-fixtures/
```

```text
FERRUM RUNTIME VNEXT G02 TEST EVIDENCE PASS: <out_dir>
FERRUM GATE vnext-g02 PASS: <out_dir>
```
