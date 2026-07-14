# G01: 核心 Contract、Trait 与 Execution Plan

## 状态与依赖

- 状态：Open
- 依赖：G01A/S0A 只依赖 G00F facts；G01B/S0B 与 S1 Qwen3.5-4B CUDA production slice
  同里程碑；不依赖完整 G00P
- 下游：G02-G10

## 目标

彻底重构 vNext 的稳定核心边界，不迁就现有 `Backend` 大 trait 或巨型 architecture match，也不把
当前约 46K 行 isolated contract 当作不可修改的既成架构。本 Goal 先保持语义拆分现有 contract/test，
再由实际 Qwen3.5-4B CUDA `run`/`serve` consumer 驱动 breaking semantic rewrite 和扩展演练。

G01A/S0A 只做保持语义的 contract/test 结构拆分，不新增 operation/model/product feature。G01B/S0B
允许激进修改 public API、typestate、泛型边界、event/profile schema 和测试拓扑，但必须与 S1 actual
Qwen3.5-4B CUDA production consumer 同一里程碑，完成 reference implementation、扩展演练和
overhead 测量。二者共同通过才构成新的 G01 PASS。

2026-07-13 的既有 `FERRUM GATE vnext-g01a PASS` 保留为 historical isolated-contract checkpoint；
它不证明当前 contract split、production wiring 或 G01 完成，也不冻结现有 API。执行顺序和 artifact
语义以 [`EXECUTION_STRATEGY_AMENDMENT_2026-07-14.md`](EXECUTION_STRATEGY_AMENDMENT_2026-07-14.md)
为准。

## S0A：保持语义的结构拆分

移动或拆分 `crates/` 文件前先运行 `inventory_tree.py`。S0A 按 identity/error、demand/work、capacity/
provisioning、pool/extent、transaction/lease、request/sequence/session、step/invocation、completion/
recovery、event/profile 和 owner-aligned tests 拆分依赖图。

- `resource.rs`、`execution.rs`、`event.rs` 最终单文件 production logical LOC `<=2,500`；作为
  facade/re-export 时 logical LOC `<=500`。
- 单个 contract test target logical LOC `<=2,000`；同一 invariant 不因拆分重复成多套测试。
- lower-level resource contract 反向依赖 scheduler/product/model 数 `0`，新增循环模块依赖数 `0`。
- public item old path -> new owner map 覆盖 `100%`；S0A 无意删除的 public item 丢失数 `0`。
- S0A 每个提交执行 bounded focused tests；最终执行 bounded
  `cargo test -p ferrum-interfaces --all-targets`。paid GPU 和完整 G00P collector 运行次数 `0`。

S0A 的文件大小门是所有权和可审阅性门，不是总 LOC 删除门。capacity-derived admission、dynamic
pool、transaction/lease、fence/reaper/recovery 和 event/profile invariant 必须原样保留到 S0B 审计。

## S0B/S1：真实 consumer 驱动的语义重构

每个 public abstraction 必须记录 invariant owner、production consumer 或 S1 target、misuse
prevented、hot-path overhead 和 compile invalidation domain。只有 synthetic self-test 引用、没有
不可约 invariant 且没有目标 consumer 的 API 必须删除；对架构、性能、扩展和诊断有明确收益的复杂
contract 允许保留并继续强化。

G01B 必须让 actual Qwen3.5-4B CUDA 请求通过 vNext config/weights/program/plan、共享动态 runtime、
operation providers，并同时进入 `ferrum run` 和 `ferrum serve`。单向 adapter 可复用已验证 kernel，
但 legacy scheduler/model runner/resource manager/product composition fallback 数必须为 `0`。

### Canonical checkpoint

修订后的 G01A/G01B 是可独立 freshness 校验的 DAG node，不是人工 checklist：

```text
python3 scripts/release/run_gate.py vnext-g01a --g00f <g00f-manifest> --out <external-out>
python3 scripts/release/run_gate.py vnext-g01b --g00f <g00f-manifest> --g01a <g01a-manifest> \
  --s1 <qwen35-4b-cuda-production-manifest> --out <external-out>
python3 scripts/release/run_gate.py vnext-g01 --g01a <g01a-manifest> --g01b <g01b-manifest> --out <external-out>
```

G01A manifest 必须引用 G00F、inventory、public owner map、拆分前后 API/behavior evidence 和 bounded
tests。其 `unlocks` 只能是 `G01B`/`S1`，不得声称 runtime、模型或性能已迁移。G01B 必须引用 G00F、
G01A、actual S1 `run`/`serve` artifact 和相同 contract source。aggregate G01 必须逐字节消费两个 child
manifest并验证 source/contract/model-input freshness；任一 child stale、contract 不同或 facts 不一致时
拒绝，不能复制 child summary 重新签发 PASS。

## 必须由生产纵切证明的 contract

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
- runtime concurrency 是 capacity-derived dynamic admission；配置值只是 ceiling。Plan 中
  sequence/request 资源必须是 `O(graph)` scoped descriptor，禁止按 ceiling 展开 per-slot Vec、
  全量 claim 或全量 buffer allocation。
- admission 返回 typed `Admitted`、`Deferred` 或 `Impossible`：temporary defer 携带逐 capacity
  domain 的 requested/available/release-epoch；独占总容量也不可运行的请求必须永久拒绝，不能
  无限排队。在 committed lease 之前 provider encode、device submit 和 prefill launch 均不可达。
- provider workspace estimator 只能返回 core-validated actual-shape formula；不得读取 global
  concurrency ceiling 或返回已乘 ceiling 的 opaque bytes。Plan/Sequence/Invocation scope 明确，
  invocation formula按实际 sequences/tokens/pages 或有界多维 bucket 求值。
- semantic state capacity 只允许 fixed-per-scope 或 token-scaled；物理 page/block 是 selected storage
  contract 的 allocation quantum，不得由 `PageScaled`/`Pages` demand 推断。provider binding/workspace
  requirement、device offer 和 typed runtime preference 必须联合求交；模型 program 不含 allocator、
  physical view、pool、backend/device id。
- plan-static provisioning 与 logical sequence admission 使用不同 sealed authority；static/no-op
  provisioning 无法创建 execution stream、激活 sequence、生成 event active evidence 或 dispatch。
- plan runtime 必须是 `Arc<PlanRuntimeResources<R>>` owning root；`NoStatic` 和 committed
  `ResourceTransaction` 只能按值 handoff，borrowed static lease 不能生成 trusted binding。
  Request -> Sequence -> Session -> Step -> Invocation 全链删除 `'plan` 并满足 `Send + 'static`。
- completion reaper 不得借用 plan。root 与 reaper 只能是 sibling 或 `Weak` 关系；close 在仍有 child/
  reaper hold 时返回 typed `Referenced`，cleanup failure 保留 retry/quarantine ownership。动态 chunk
  buffer 必须先于 capacity grant 释放，static buffer 必须先于 static capacity claim 释放。
- mutable model state 不允许 `PlanStatic` 跨请求共享；state demand 与 Request/Sequence/Step lifetime
  必须一致。minimum runnable memory 使用 resident state 加同时 live workspace peak，不能把顺序
  node scratch 全图求和。

## 扩展演练

1. synthetic dense family：只新增 provider/program/test，核心改动 `0`。
2. synthetic recurrent family：复用 state contract，runtime 主循环改动 `0`。
3. novel op：只新增 operation/provider/oracle/catalog，planner/runtime 主循环改动 `0`。
4. reference backend：模型代码改动 `0`。
5. unsupported backend：在权重分配前返回包含 missing op/version 的结构化错误。

## G01B/G01 验收

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
- `maximum_active_sequences=1/32/4096/u32::MAX` 的 plan build 内存、descriptor 数和 build-time
  allocation 数完全相同且保持 `O(graph)`；除 `0` 外不得用任意固定常量拒绝合法 ceiling，
  theoretical ceiling 超过设备容量也不能替代运行时动态 admission。
- 同一 provider/node/shape 在上述四个 ceiling 下的 estimator fingerprint、workspace formula、
  单位 demand 和 minimum runnable bytes 完全一致；provider 对 global ceiling 的可见字段数 `0`。
- 同一 typed pool 内，依赖闭包且执行合同证明“前驱 fence terminal 后后继才可 claim”的两个
  invocation scratch minimum runnable 增量取 `max(a,b)` 而非 `a+b`；只有 submission 全序不构成
  复用证明。无法证明互斥的 node 先保守求和，并把 per-pool
  `TotalOrderCompletionReuse/ConservativeConcurrent` 写入 plan evidence/hash。
- static provisioning 生成 stream/active-sequence/dispatch authority 的可达路径 `0`；zero-static
  dynamic-only plan 的 no-op provisioning panic/伪造 receipt 数 `0`。
- dynamic capacity 只由已提交的真实 backing segment 生成；从剩余显存数字直接发布 capacity
  的路径数 `0`。每个 dynamic provider view 均绑定 exact device buffer、segment generation、offset
  和 length；metadata-only authority 可达 dispatch 的路径数 `0`。
- dynamic descriptor 的 storage contract 和 core-derived pool id 必须进入 plan hash；compatibility
  key 至少绑定 allocator、contiguous/paged view、usage、dtype、layout fingerprint、alignment。
  provider 指定 pool id、资源层由 demand 类型猜 view、同 id/different key、按 descriptor 重新分配
  domain 的路径数均为 `0`。每个 segment 带 chunk generation/ordinal；contiguous claim 只可返回一个
  extent，paged claim 才可跨 extent/chunk。
- 同一 synthetic KV program 在 CUDA-like fixed-block+paged 与 Metal-like linear+contiguous offer 下
  program fingerprint 完全相同，selected memory plan/hash 不同；provider/storage 联合求解必须在
  compatible fallback provider 存在时选中它，不能因先贪心选了 storage-incompatible provider 而误报
  unsupported。
- lifecycle authority 必须具有 `PlanRuntimeResources -> AdmittedRequestResources ->
  AdmittedSequenceResources -> StepResourceLease -> InvocationResourceLease` 五个语义 scope；Request
  state 对 N 个 child sequence 只 claim 一次，Sequence state 每 child 独立，Step state 跨每个
  participant 的 exact frame 全部 node，Invocation scratch 只属于一次 exact batch node invocation。
  Step/Invocation 必须持有 canonical non-empty participant parent set；把 batch capacity 挂在 leader
  sequence、按 participant 重复 claim scratch、把 Request 并入 Sequence、Step 并入 Invocation，或
  用 Node 同时表示结构 owner 与时间 lifetime 的路径数均为 `0`。
- `PlanRuntimeSource<'plan, _>`、`TrustedPlanRuntimeBinding<'plan, _>`、五层 owning authority 的
  `'plan` 参数以及 `CompletionReaper<'plan, _>` source count 均为 `0`；compile-pass 必须证明 root、
  binding 和 Request -> Invocation durable chain 为 `Send + 'static`，所有 `Arc` target 为 `Sync`。
- NoStatic/Static 两条 consuming handoff 与 close fault matrix `100%`：任一 child/reaper hold 存活时
  close 只返回 `Referenced`；释放后 retry 成功；partial static release 可 retry 或 quarantine；root/reaper
  strong-reference cycle 数 `0`，dynamic buffer-before-grant 与 static buffer-before-claim 顺序违规数 `0`。
- public admission 接收 raw `DynamicResourceShape` 或任意 caller aggregate token/page 数的路径数为
  `0`。model program 必须显式声明 token-bearing value/axis，plan node 的 `NodeWorkContract` 必须把它
  解析到 exact `ResolvedValueBinding` role/ordinal/rank；page metric 必须绑定 exact
  state/resource/pool/storage profile 和 committed page authority。缺失或含糊 source 时 plan build
  必须失败，禁止按 tensor 总元素、模型名或 backend 猜测。
- core 从 per-participant token span、full-input 和 committed page-set authority 生成不可伪造的
  `BatchWorkShape`；其 sequence 数必须等于 exact participant 数，immediate/fit token/page metric
  必须 checked 聚合并写入 canonical fingerprint。该 authority/fingerprint 必须随
  `ClaimedBackingTransaction`、Step、Invocation 和 `OperationInvocation` 贯穿到 fence；claim 后替换
  work、仅验证 minimum bytes 或空 demand 丢失 shape authority 的成功数均为 `0`。32-participant
  continuous batch 必须产生 `1` 个 invocation child capacity claim、`1` 份 batch scratch、`1` 个
  command 和 `1` 个 fence。
- `SequenceSession` 的 run/request/plan/coordinator/sequence generation/active fingerprint 在 live
  lifetime 内冻结；Step/Invocation 持有 owning parent，session 仅在自身 participant-flight=`0` 后
  complete/abort。sequence terminalization drain 共享 lane、替换 session identity 或阻塞其他
  participant 的路径数均为 `0`。
- `BatchStepId`/`BatchInvocationId` 与 per-participant `ExecutionFrameId`/request node identity 必须
  分离；frame `7/2/19` 的 participant batch 原样保留映射，强制同 frame、同 node invocation id 或
  leader identity 投影的路径数 `0`。物理 batch ledger 的
  Prepared/InFlight/Retired 各至多一次，InFlight 安装 durable fence 后才可发布 Submitted receipt；
  N 个 request journal 只引用同一 submission/completion fingerprint，不能把 command/fence/吞吐计数
  放大 N 倍。
- invocation topology collision 必须按
  `(sequence authority, request authority, ExecutionFrameId, NodeId)` 判断，不能按可重新生成的
  `BatchInvocationId` 判断；任意 participant key overlap 必须整批原子拒绝。状态只允许
  `Prepared -> InFlight -> Retired` 或 `Prepared -> NotSubmitted -> Prepared`；后一条 retry 只能消费
  exact `DefinitelyNotSubmitted` 生成的 sealed authority，且 topology/work fingerprint 必须不变。
  换 attempt id 重复提交、Retired tombstone 提前删除或 possibly-submitted error retry 的成功数均为 `0`。
- Invocation resource、Request-state hazard permit 和 backing extent 必须持有到 typed device fence
  completion；`submit Err` 只允许表示 definitely-not-submitted，任何 possibly-submitted 错误必须
  返回 fence 并在 terminal failed-but-quiescent 或 quarantine 路径处理。`submit` 返回后 fence 未
  完成即复用的路径数 `0`。正常 scheduler 通过 poll/await 或 completion reaper 前进，不得在请求
  线程同步等待。
- final sequence/completion owner 的 Drop 后端调用数和内部新建 worker 数均为 `0`；未静默释放、
  `mem::forget` 或变成不可达的 cleanup owner 数为 `0`。每个 plan-domain 的显式 recovery
  maintenance 单轮至多选择 `64` 个 owner 且每个至多尝试一次，backend 调用期间 registry lock
  不得被持有，一个阻塞任务不得扣留 sibling lane。失败、panic、quarantine 后 owner 和 freshness
  必须可观测并可重试；积压达到 `64` 时只阻止新的 execution-authority derivation，不限制正常用户
  并发，也不能拒收第 `65` 个已存在 owner。第二次成功维护后 close 必须收敛。
- `NodeRetired` 必须消费 exact shared batch completion receipt 的 participant projection；仅有
  submission receipt 即进入 node/frame terminal 的成功数 `0`。
- dynamic operation view 只能通过 checked logical-range -> physical-region translator 取得 buffer；
  provider 可见的脱离 physical offset/extent 的 dynamic arena 裸 buffer accessor 数量 `0`。
- static buffer 与 backing segment 在全局 device account 中各计费一次，logical slice/page 不重复
  计费；两个 plan/pool 不能分别认领同一份物理余量。initial/grow 失败不得改变 published capacity
  或 capacity epoch。
- capacity-exact、少 `1 byte`、释放后重试和资源不足队首场景 `100/100`：defer 时 prefill
  submit=`0`，active decode 继续，eligible waiting request 不被阻塞，release epoch 后成功 admission。
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
g01a-contract-split/
  manifest.json
  adr.md
  contract-map.json
  public-owner-map.json
  split-inventory.json
  compile-unit-trybuild.json
g01b-reference-contract/
  manifest.json
  qwen35-4b-cuda-production.json
  run-serve-evidence/
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
FERRUM RUNTIME VNEXT G01A CONTRACT SPLIT PASS: <out_dir>
FERRUM GATE vnext-g01a PASS: <out_dir>
FERRUM RUNTIME VNEXT G01B PRODUCTION REFERENCE CONTRACT PASS: <out_dir>
FERRUM GATE vnext-g01b PASS: <out_dir>
FERRUM RUNTIME VNEXT G01 CORE CONTRACTS PASS: <out_dir>
FERRUM GATE vnext-g01 PASS: <out_dir>
```
