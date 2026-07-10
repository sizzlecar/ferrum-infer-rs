# G01: 核心 Contract、Trait 与 Execution Plan

## 状态与依赖

- 状态：Open
- 依赖：G01A 可在 G00a inventory/model-resolution/preset checkpoint 后并行；G01 最终 PASS
  依赖完整 G00 PASS
- 下游：G02-G10

## 目标

从零设计 vNext 的稳定核心边界，不迁就现有 `Backend` 大 trait 或巨型 architecture match。
本 Goal 先完成 ADR、type contract、最小 reference implementation 和扩展演练，再允许批量迁移。

G01A 只允许新增隔离的 `ferrum-interfaces::vnext` 纯契约、ADR、compile/unit/trybuild test 和
legacy mapping；不得接管 `run`/`serve` 路由、改变 runtime 默认值或产生性能结论。G01B 在
G00 PASS 后消费真实 baseline 与 hardware/model lock，完成 reference implementation、扩展演练和
overhead 测量；二者共同通过才构成 G01 PASS。

### Canonical checkpoint

G01A/G01B 是可独立 freshness 校验的 DAG node，不是人工 checklist：

```text
python3 scripts/release/run_gate.py vnext-g01a --g00a <g00a-manifest> --out <external-out>
python3 scripts/release/run_gate.py vnext-g01b --g00 <g00-manifest> --g01a <g01a-manifest> --out <external-out>
python3 scripts/release/run_gate.py vnext-g01 --g01a <g01a-manifest> --g01b <g01b-manifest> --out <external-out>
```

G01A manifest 必须引用 G00a manifest/artifact index SHA、ADR/contract/trybuild Git blob SHA 和
source tree；其 `unlocks` 只能是 `G01B`/`G01`，不得声称 runtime、模型或性能已迁移。G01B 必须
引用完整 G00 manifest、hardware/model lock、G01A manifest 和相同 contract blob。aggregate G01
必须逐字节消费两个 child manifest，验证 source/contract/model-input freshness；任一 child stale、
contract blob 不同或 G00a/G00 facts 不一致时拒绝，不能复制 child summary 重新签发 PASS。

## 必须定义的 contract

1. `DeviceRuntime`：buffer、stream/command、copy、sync、device error。
2. `OperationContract`：version、shape/dtype/layout、resource、oracle、provider、profile phase。
3. `ModelFamilyProvider`：typed config、weight schema、semantic program、template/EOS metadata。
4. `ModelProgram`：blocks、state specs、weight refs，不包含 backend implementation。
5. `ExecutionPlanner`：program + capability + typed runtime policy -> immutable plan。
6. `ExecutionPlan`：stable node id、selected provider、memory plan、fallback/reject reason、plan hash。
7. `ResourceTransaction`：reserve/commit/rollback/release 的唯一状态机。
8. `ExecutionEventSink`：request/plan/node/op/resource identity，disabled 时近零开销。
9. `ResolvedModelPlan`：产品 source/config/template/runtime/engine 的唯一结果。

## 设计规则

- 不要求 runtime dynamic plugin；优先编译期注册和版本化 provider catalog。
- 不允许 `Any`/字符串 downcast 成为 hot path 类型系统。
- operation 可以使用 trait object，但每 token 小 op 必须通过 benchmark 证明 overhead 可忽略。
- capability 在 plan build 时解析一次；plan 执行期间只消费已选择 provider。
- 所有 default method 必须是语义安全的纯组合；unsupported op 不得有成功默认实现。
- `Architecture` 可作为外部 metadata id 保留，但不能驱动 engine 中的巨型 concrete match。
- 未知 architecture/config/weight layout fail-closed，禁止 fallback 到 Llama/ChatML。
- 事件、错误和 replay identity 是 contract 的组成部分，不是后续 logging 装饰。

## 扩展演练

1. synthetic dense family：只新增 provider/program/test，核心改动 `0`。
2. synthetic recurrent family：复用 state contract，runtime 主循环改动 `0`。
3. novel op：只新增 operation/provider/oracle/catalog，planner/runtime 主循环改动 `0`。
4. reference backend：模型代码改动 `0`。
5. unsupported backend：在权重分配前返回包含 missing op/version 的结构化错误。

## 验收

- ADR 至少比较“聚合小 capability traits”和“typed operation registry”两种实现，包含
  compile-time、runtime overhead、object safety、错误定位和扩展成本数据。
- 选定方案通过 5/5 扩展演练。
- 通用 contract 中架构命名符号数量 `0`。
- vNext plan execution hot loop 中 capability/backend feature branch 数量 `0`；legacy 路径
  在 G01 只登记 mapping，G08D 负责全仓零值。
- silent success default 数量 `0`。
- unknown architecture -> Llama fallback 数量 `0`。
- plan snapshot 在相同输入上 deterministic `100/100`。
- plan schema round-trip `100/100`，schema breaking change 会被 version gate 拒绝。
- disabled event sink 对 CPU microbench overhead `<=1%`；enabled basic sink `<=2%`。
- 新增 existing-op family 的允许生产改动：family package、registration manifest、fixtures；
  shared runtime/Backend/product entry 改动均为 `0`。

## 删除条件

G01 不立即删除 legacy trait，但必须产出逐项 mapping 和删除 owner。任何无法映射的 legacy
method 必须被分类为 stable device primitive、versioned op、model semantic 或 dead code，
不允许以“特殊情况”留在新核心。

## 产物与 PASS

以下均为 canonical external `<out_dir>` 下的逻辑路径：

```text
g01a-contract-checkpoint/
  manifest.json
  adr.md
  contract-map.json
  compile-unit-trybuild.json
g01b-reference-contract/
  manifest.json
  extension-drills.json
  plan-snapshots/
  overhead.json
g01-contracts/
  manifest.json
  adr.md
  contract-map.json
  extension-drills.json
  plan-snapshots/
  overhead.json
```

```text
FERRUM RUNTIME VNEXT G01A CONTRACT CHECKPOINT PASS: <out_dir>
FERRUM GATE vnext-g01a PASS: <out_dir>
FERRUM RUNTIME VNEXT G01B REFERENCE CONTRACT PASS: <out_dir>
FERRUM GATE vnext-g01b PASS: <out_dir>
FERRUM RUNTIME VNEXT G01 CORE CONTRACTS PASS: <out_dir>
FERRUM GATE vnext-g01 PASS: <out_dir>
```
