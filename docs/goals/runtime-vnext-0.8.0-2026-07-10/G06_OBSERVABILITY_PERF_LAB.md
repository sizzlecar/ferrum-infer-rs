# G06: Observability、Replay 与统一性能实验室

## 状态与依赖

- 状态：Open
- 依赖：basic/resource identity 随 S1；latency 随 S2；kernel/replay 在 S3 前可用；full G06 在 S6
- 下游：S1-S7、G08-G10

## 目标

让正确性、资源、延迟和 kernel 性能使用同一 identity 和 artifact，而不是依赖约 50 个
隐藏 profile/trace env、stderr regex 和多个互不关联 schema。一次失败必须自动给出首个失败
阶段和可验证的下一步假设。

profile 是 execution/resource/op contract 的核心组成，不再作为迁移完成后的附加设施。S1 没有
basic/resource artifact 不能退出；S2 没有 latency 和 first-failure attribution 不能进入大模型迁移；
S3 前完成 kernel/replay 入口。完整 benchmark catalog、historical attribution 和正式 overhead matrix
仍在 S6 聚合，避免 profile gate 再次先于 production caller 膨胀。

## 统一 identity

```text
run_id -> request_id -> sequence_id -> plan_hash -> node_id -> operation_id
       -> resource_lease_id -> backend_span_id -> kernel/native_op_id
```

所有 run/serve、scheduler、resource、op provider、kernel profile 和 benchmark row 使用相同
identity。公共 analyzer 不得解析 backend-specific 日志才能判断 PASS。

## Typed profiling 入口

提供 documented CLI/config preset，例如：

```text
ferrum run ... --profile <preset> --artifact-out <dir>
ferrum serve ... --profile <preset> --artifact-out <dir>
```

preset 至少有 `off`、`basic`、`resource`、`latency`、`kernel`、`replay`。release gate 不能
要求用户不知道的 env 组合。

## 统一阶段

- source/config/plan；
- weights load/convert/upload；
- admission/queue；
- prefill prepare/attention/FFN-or-MoE；
- decode prepare/attention-or-DeltaNet/FFN-or-MoE；
- state/KV update；
- logits/sampling/decode text；
- API serialize/stream flush；
- cancel/release。

每个阶段记录 wall/device time、count、bytes、tokens、batch shape、provider 和 first error。
并发场景还必须记录每次 admission/start/defer/release 的 monotonic timestamp 和 active sequence
count，由 analyzer 计算 requested concurrency、typed cap、observed max-active、eligible interval 与
time-weighted active duty-cycle；只从日志最大值或 client queue depth 推断 active concurrency 禁止通过。

### 时间覆盖率计算

并行 CPU task、CUDA stream 和 Metal command buffer 可能重叠，覆盖率不得用 span duration 直接
相加。每个请求/sequence 必须保存 monotonic-clock interval 与 parent/child/async-link：

- `decode_wall_time` 是第一个 decode step ready 到最后一个 token commit/cancel 终态的 critical-path
  wall interval；排队、HTTP flush 和调用者 backpressure 单列，不混入 decode。
- `stage_accounted_time` 是所有 decode stage interval 与上述 critical-path interval 的交集并集长度；
  同时发生的 span 只计一次。`stage coverage = union(stage intervals) / decode_wall_time`。
- `device_busy_time` 是目标 device 上所有相关 kernel/native-op event interval 的并集；跨 stream 重叠
  只计一次。每个 op 的 inclusive duration 可以单列，但 top-op coverage 的分子必须先按归属规则
  消除 parent/child 和 fusion 重复，再求 interval union。
- `unattributed_time = decode_wall_time - stage_accounted_time`，负值、coverage `>100%` 或缺失
  clock-domain conversion 一律 validator REJECT。

wall 和 device clock 不能直接相减；backend 必须记录同步 anchor、转换误差和 event source。正式
artifact 中时钟转换误差须 `<=0.5% decode_wall_time`，否则只可作为 diagnostic。

## 实验 contract

每个性能候选必须有机器可读：

- baseline artifact；
- 单一主要 hypothesis；
- expected trace/counter change；
- correctness pre-gate；
- performance command；
- accept target、reject threshold、stop condition；
- KEEP/REJECT 结果与 remaining gap。

禁止无范围 env sweep。相同 failure class 两次 REJECT 后，runner 必须阻止新的 paid full run，
直到提交 source-level hypothesis 和本地/plan-only evidence。

## Benchmark 收敛

- `/v1/chat/completions` 正式性能客户端只有 `ferrum bench-serve`。
- BenchReport 每个 row 必须记录 requested/effective thinking preset 和实际发送的
  `chat_template_kwargs`；performance validator 对缺字段、server 未生效或 external engine
  payload 不一致一律 REJECT。
- report schema 只有 `ferrum-bench-core::BenchReport`。
- `BenchReport` 必须保存 typed per-request ITL source、output event、usage token、observed interval、
  transport coalescing 和 eligibility；validator 从这些字段重算 repeat counts/totals。client
  `sse_delta_events` 与 runtime `engine_token_events` 是不同指标，source 混用数量必须为 `0`。
- `engine_token_events` 必须来自同一 monotonic clock 的 token-commit timestamps，interval 数精确为
  `tokens-1`；HTTP flush/backpressure 单列。只有 server-side token-commit/flush span 能归因 server
  bulk flush，client transport coalescing 不能。
- `m3_ab_runner.py` 可保留为 orchestration，但不得重算 token、SSE correctness 或统计。
- legacy wrappers 要么变成单行 manifest adapter，要么在 inventory 后删除/归档。
- raw HTTP client 只允许 correctness scenario allowlist，不得产生正式 throughput claim。

## Replay

replay bundle 包含 tokenized input、template hash、plan/config snapshot、sampling seed、scheduler
decisions、resource events、expected output/token ids 和 model/weight fingerprint。不得复制权重或
secret。run 与 serve bundle 使用同 schema。

### CUDA replay timing 合同缺口（2026-07-24）

`e5a6e6c1` 的真实 RTX 4090 artifact 已证明现有 profile 不是完全缺失，而是停在 replay
边界：exact submission counter 可测得 decode device execution 平均 `8.206444 ms`，
但 `18,450` 条 native event 中有 `17,625` 条 command timing unavailable；其中
`16,605` 条 replayed compute 全部为 `BackendUnsupported`。根因是 CUDA runtime 一旦
launch reusable executable，就丢弃已经收集的 eager command event，并把整次
`DeviceSubmissionExecutionTiming` 标为 unavailable。engine consumer 又按 timing 数组位置
关联 attribution row，而不是按显式 command identity 关联，因此不能安全表达部分覆盖。

这属于观测合同缺陷，不允许用重复 throughput sweep 补偿。合同必须能够表达物理执行 span：

- eager command span 精确覆盖 `[command_index, command_index + 1)`；
- replay executable span 精确覆盖 `[start_command_index, end_command_index)`；
- span 使用同一 submission-relative device clock，保持有序、非重叠并验证 command range；
- 一个 replay span 的 elapsed time 不得复制给范围内每个 logical command；
- eager event 采集失败不得抹掉已经成功测得的 replay span，反之亦然；缺失按 span/range 明示；
- analyzer 先报告 replay program、eager boundary 和 unattributed device time；只有存在真实
  subwork interval 时才继续下钻到 op/kernel，不得伪造 top-op coverage。

下一 paid CUDA diagnostic 前，本地合同至少证明 partial timing、range validation、显式
command-index join、mixed eager/replay 顺序和 terminal RAII；profile `off` specialization
不得新增 event、分配或热路径分支。下一 artifact 的 expected signal 是 replay segment timing
coverage `100%`，并把 exact decode device time 分成 replay program 与
binding/embedding/logits eager work；在此之前禁止再次运行完整吞吐 sweep。

## 验收

- 顶层 observability 自测执行全部子组件；漏接线 `0`。
- 当前 `product_observability_wiring_gate` SHA mismatch fixture 修复并被总 self-test 捕获。
- profile 默认 `off`，只允许通过 typed CLI/config 与 artifact path 开启；off 路径创建 sink、采集线程、
  profile/trace 文件、事件序列化/入队和 device completion timing 的数量均为 `0`。默认路径性能由
  G09 正式 no-regression/外部竞争门验收。
- basic profile overhead `<=2%`、resource/latency profile `<=5%` 保留为开启模式的报告目标；
  profile ABBA-BAAB 的单次 CV 或 overhead 超标标为 `noisy`/`target_miss`，不阻塞架构里程碑。
  kernel profile 仍为 diagnostic，不得用于正式 throughput。
- stage accounting 覆盖 decode wall time `>=90%`，重叠/未解释部分有字段而非丢失。
- top operations/kernels 累计覆盖 device time `>=80%`。
- G09 同配置 5 次正式 throughput CV `<=5%`；超过时该性能 claim 的 artifact REJECT 并报告噪声源，
  但不得反向阻塞默认关闭的 profile 功能合同。
- historical failures：first failure class `15/15 family`、`M/M concrete case` 正确，replay
  同样达到 `15/15` 和 `M/M`。
- 从已有 artifact 到自动 top bottleneck/target gap 报告耗时 `<=60s`。
- 正式 HTTP benchmark client 数量 `1`；正式 regex-derived throughput 数量 `0`。
- 官方性能 lane `100%` 输出 BenchReport 和 KEEP/REJECT。
- hidden-env-only profiling option 数量 `0`。
- performance row 的 thinking requested/effective/payload 字段完整率 `100%`。
- ITL typed 字段完整率与 eligibility/count/interval 重算一致率 `100%`；ineligible request 进入正式
  ITL ratio 数 `0`，client SSE source 冒充 engine token-commit 数 `0`。
- 三主模型 performance/concurrency row 的 active timeline、eligible interval、max-active 和 duty-cycle
  字段完整率 `100%`；MODEL_MATRIX active floor/duty-cycle 计算可从原始事件确定性重放 `100/100`。

## 产物与 PASS

```text
docs/release/runtime-vnext/0.8.0/g06-observability-perf-lab/
  schema/
  profiles/
  replay/
  historical-failure-report.json
  benchmark-catalog.json
  wrapper-inventory.json
  overhead.json
```

```text
FERRUM RUNTIME VNEXT G06 OBSERVABILITY PERF LAB PASS: <out_dir>
FERRUM GATE vnext-g06 PASS: <out_dir>
```
