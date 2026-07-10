# G04: 共享 Execution Runtime、Scheduler 与资源事务

## 状态与依赖

- 状态：Open
- 依赖：G01、G02、G03
- 下游：G05、G06、G08-G10

## 目标

建立唯一共享执行 runtime。模型只描述 program/block/state；runtime 管理 batch、prefill、
decode、资源、执行次序和退出。解决当前 engine/model/KV manager 多重所有权以及模型复制
unified runner 的问题。

## Runtime 职责

1. request admission、capacity、defer、resume、cancel。
2. prefill/decode/mixed-batch 计划和公平调度。
3. immutable `ExecutionPlan` 驱动的 layer/block traversal。
4. KV、recurrent state、scratch、graph workspace 的唯一事务。
5. logits/final-token selection 与 sampler 边界。
6. 所有正常、错误、cancel、disconnect、timeout 路径的 deterministic cleanup。

## Transaction 与 Lease 状态机

```text
unallocated -> reserved -> committed -> released
                     \-> rolled_back
committed -> deferred -> resumed -> committed
committed/deferred -> cancelled -> released
```

每个 transition 记录 request、resource kind、owner、amount、before/after、reason、plan node。
禁止通过 allocator/kernel OOM 才发现容量不足。

transaction state 与单个 lease state 必须分离。一次 request 可以原子包含 KV、recurrent、
scratch 和 graph workspace 多个 lease：

- reserve 阶段要么全部成功，要么按逆序补偿，不能留下 partial reservation；
- commit 顺序、provider failure after partial commit 和补偿 action 必须写入 contract；
- rollback/release/cancel transition 必须幂等，重复事件返回明确 already-finalized 而非 underflow；
- defer 明确哪些 lease 保留、哪些归还，resume 必须重新验证 generation/version；
- prefix/session cache retention 是显式 policy，最终 expected balance 与 leak 分开计算；
- disconnect/timeout 在任意 transition 点都有唯一合法终态。

## 状态类型

- full-attention paged KV；
- linear-attention recurrent state；
- hybrid model 同时拥有 KV + recurrent state；
- model/backend scratch；
- CUDA graph/Metal pipeline workspace；
- prefix/session cache lease。

这些状态通过 `StateSpec` 与 `ResourceLease` 管理，模型代码不得保存独立 hashmap/manager
作为第二真相。

## 测试

- seeded model checking / property test 至少 100,000 个状态序列。
- capacity=0/1/exact/overflow；并发 reserve；cancel at every transition；provider failure；
  partial prefill；mixed final/non-final；reallocation。
- 多资源第 N 个 reserve/commit 失败、逆序补偿、重复 rollback/release、defer retain/release、
  cache-retention policy 的组合 fault injection。
- Qwen3.5 hybrid state：KV 释放不提前释放 recurrent state，反之亦然；两者最终均释放。
- scheduler trace 可 replay 并得到相同 transition 和 batch membership。

## 验收

- 100,000 状态序列中 leak、underflow、double release、use-after-release 均为 `0`。
- multi-resource partial reserve/commit fault grid `100%` 达到 contract 指定终态，补偿遗漏 `0`。
- vNext program/runtime 与 G04 L0/L1 纵切中 model-owned KV/recurrent manager 数量 `0`；
  legacy 模型的全仓/source-tree 零值由 G08D 验收。
- EngineBuilder 构造后丢弃 scheduler/manager 的路径数量 `0`。
- 所有 capacity reject/defer 在 kernel launch 前发生 `100%`。
- cancel/disconnect 后同容量下一请求成功率 `100%`。
- mixed-batch output 与逐请求 reference 一致 `100/100` seeds。
- resource final state 在所有 G04 L0/L1 scenario 中为 empty/expected cache `100%`；actual
  L2-L3 的同项验收归 G08。
- scheduler/replay determinism `100/100`。
- Qwen3.5 c32 historical resource fixtures `100%` 被本地 state-machine 或 replay gate 捕获，
  paid GPU 不再是第一次完整 invariant 测试。
- synthetic dense、MoE、hybrid 三类 program 的 setup/admission/state/finalize/cleanup 全部
  使用同一 lifecycle implementation，重复主循环数量 `0`。G08 在真实三模型迁移后重新
  执行同一 ownership analyzer；不再使用可通过改变分母操纵的“复用率”指标。

## 性能约束

- L1 reference workload scheduler bookkeeping 占 runtime wall time `<=5%`；真实 CUDA c32
  `<=2%` 由 G09 验收。
- disabled event path overhead `<=1%`。
- resource transaction 不增加每 token host allocation；steady decode host allocations/token=`0`。

## 产物与 PASS

```text
docs/release/runtime-vnext/0.8.0/g04-runtime-resources/
  state-model-report.json
  resource-fixtures/
  scheduler-replays/
  allocation-profile.json
  qwen35-resource-kills.json
```

```text
FERRUM RUNTIME VNEXT G04 RUNTIME RESOURCES PASS: <out_dir>
FERRUM GATE vnext-g04 PASS: <out_dir>
```
