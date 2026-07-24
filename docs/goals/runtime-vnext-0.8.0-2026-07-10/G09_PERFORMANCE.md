# G09: 三模型双后端性能恢复与优化

## 状态与依赖

- 状态：Open
- 依赖：G00P、G06、G07、G08；只在 S6 correctness/legacy-zero 后形成正式性能结论
- 下游：G10

## 目标

在正确性和 legacy 删除后恢复并超过重构前性能。CUDA 是主优化方向；Metal 必须达到
可发布正确性和同机主流实现逐 cell 90% / 主矩阵几何平均 95% 的竞争线，不得在最后发布
阶段才第一次运行。

## 统一流程

每个 model/backend/concurrency candidate：

1. 引用 G00 legacy/external baseline。
2. 运行 G08 correctness pre-gate。
3. 写单一主要 hypothesis、expected profile signal、accept/reject threshold。
4. 运行一次 scoped diagnostic；未命中 expected signal 即 REJECT。
5. 命中后按每个 required `comparison_id` 独立运行外层 `ABBA-BAAB`：serve comparisons
   使用 `bench-serve` 的 `100 x 3 inner repeats`，run comparisons 使用模型矩阵定义的真实
   `ferrum run` runner；legacy、external、run-vs-serve 的 A 实现和统计样本不得混用。
6. 保存 KEEP/REJECT、绝对目标差距和下一个 bottleneck。

禁止将多个 kernel、scheduler 和 runtime policy 改动堆在一个无法归因的候选中。

另设 capacity-pressure competitive lane：使用与 same-host vLLM 相同模型、请求序列、max
model len、KV budget、prefix-cache/speculative 设置和 client arrival trace，混合长短 prompt 并让
outstanding requests 持续高于可用 active capacity。Ferrum 必须保存每次 admit/defer/release
epoch、prefill submit 和 active timeline；固定并发、模型名/GPU 名分支或启动时按 ceiling 全量
claim 均直接失败。资源暂不可满足的队首请求不得停止 active decode 或其他 eligible request。
两侧还必须记录并对齐完整 input reservation/chunked-prefill/preempt-recompute 策略；artifact
分别报告 immediate allocated units、fit-requested units、实际 future-reserved units（未启用时为
`0`）和无资源占用的配置 ceiling。

对 G00 标记为 legacy PASS/comparable 的 cell，G00 必须保存可执行 legacy binary、动态
依赖/native artifacts 和 binary SHA256。G09 在实际性能硬件上重新运行该 binary 与 vNext
的 ABBA；G00 summary 本身不能代替同机交错。若该冻结 binary 无法在主机执行，对应
legacy-comparison cell 为 BLOCKED，不能退化为跨时间均值比较。G00 已标记 BLOCKED/new 的
cell 不要求 legacy binary，只执行 numerical/external/product 门，不能因此阻塞整个 G09。

## Paid CUDA 合同

每次创建/启动/resume RTX 4090 前必须查询现有实例并写入 run manifest：lane、offer/instance、
小时价、预计 runtime、预计 cost、correctness command、performance command、stop condition。

- 同时最多 1 个 potentially billable instance。
- 单个 scoped diagnostic 最长 `45min`；未出现 expected signal 立即 REJECT/stop。
- 单模型 full candidate round 最长 `2h`；三模型一轮最多 `6 GPU-hours`。
- 超过一轮或预算必须先有新的 source-level hypothesis 和用户批准。
- correctness failure、两个同类 REJECT、OOM 未分类、绝对目标仍差 >20% 且 signal 未变化时
  立即停止 paid work并 copy back artifacts。
- 优先复用含模型/build cache 的 stopped instance；结束后轮询到 `actual_status=exited`。

## CUDA 目标

Hardware 固定 1x RTX 4090，三模型使用 [`MODEL_MATRIX.md`](MODEL_MATRIX.md) 的精确
revision/format，random `256/128` + frozen ShareGPT，c=1/4/16/32。

typed active admission 硬下限为 M1/M2/M3=`32/16/32`。最高 client cell 的
`observed_max_active` 必须达到对应下限；在 warmup 后且 outstanding request 数仍不少于该下限的
eligible interval 内，active sequences 达到下限的 wall-time fraction 必须 `>=0.80`。每个 outer
slot 保存 scheduler active timeline、eligible interval 和 duty-cycle；瞬时 max 或 client queue 深度
不能替代该门。typed cap 等于 floor 时 `observed_max_active == typed_admission_cap`；cap 高于 floor
时 observed max-active 至少达到 floor。external engine 使用相同 cap，但不得通过把双方 cap 同时
降到下限以下获得 PASS；cap/floor 关系变化必须走 reviewed Goal amendment。

### CUDA 通用硬门

- 以下 legacy non-regression 指标只适用于 G00 产出可执行 legacy PASS 的 cell；新增或
  BLOCKED lane 不得制造 legacy ratio：每 cell candidate throughput 中位数 `>=legacy`，
  ratio LCB `>=0.97`，几何平均 `>=1.00`。
- same-host vLLM throughput ratio LCB `>=0.90`，每个 cell 都必须达到；三个主模型所有
  required cells 的 throughput ratio 几何平均 LCB `>=0.95`。
- capacity-pressure lane 的 completed-token throughput ratio LCB `>=0.95x` same-host vLLM，
  eligible-interval device-busy duty-cycle 差值不低于 `-5` 个百分点，TTFT p95 `<=1.15x`；
  TPOT p95、short/long request queue-time p95 均 `<=1.15x`，recompute-token fraction 不高于
  vLLM `+2` 个百分点；Ferrum/vLLM admission failure、OOM 和错误请求均为 `0`。该 lane 不允许
  用普通 c=1/c=32 吞吐 cell 代替。
- 对 G00 legacy PASS cell，TTFT/TPOT candidate 中位数 `<=legacy`、ratio UCB `<=1.05`；只有
  paired A/B request 全部 eligible 时 client-SSE-event ITL 执行同一门，任一 ineligible 时该 ratio
  不得存在；G06 Ferrum token-commit ITL 独立执行 no-regression；
  run 的匹配 `engine.infer` E2E tok/s 中位数 `>=legacy`、ratio LCB `>=0.97`；peak VRAM
  `<=1.03x legacy`。禁止拿 G06 steady decode 与 G00 E2E 直接相除。
- 对新增/BLOCKED cell，延迟和 run decode 使用 [`MODEL_MATRIX.md`](MODEL_MATRIX.md) 定义的
  external/product 门；显存不与另一引擎做比例，而是不得超过 typed preflight budget，并保留
  `>=512 MiB` physical headroom。所有 cell 的 OOM/admission failure 都为 `0`。
- 三模型全部新增/BLOCKED CUDA cell 的 TTFT/TPOT p95 均须 `<=1.15x` same-host vLLM，
  paired latency ratio 95% CI 上界 `<=1.15`；M1/M2/M3 一视同仁。legacy PASS cell 仍执行更严格的
  candidate median `<=legacy`、ratio UCB `<=1.05`，不能用 external 门降级。client-SSE-event ITL
  仅在全 paired request eligible 时执行对应门；G06 token-commit ITL 仍必须采集。
- 每个 required HTTP `comparison_id` 的每实现每 cell `1200/1200` measured requests 完成；
  run comparison `12/12`，全部 warmup/error/bad-output/malformed stream `0`。
- 每个 CUDA 模型同 binary 的 G06 token-commit `ferrum run` 稳态 decode / Ferrum
  serve-c1 decode median `>=0.95`；G00 legacy PASS lane 的 run 另以匹配的完整
  `engine.infer` E2E 边界满足 legacy no-regression。

### M2 Qwen3.5-35B 历史审计向量

历史 vLLM ShareGPT artifact 的均值和当时计算的 80% LCB 为：

| c | vLLM mean tok/s | 80% LCB 参考线 |
|---:|---:|---:|
| 1 | 136.1 | 107.495 |
| 4 | 405.4 | 324.046 |
| 16 | 1190.7 | 896.239 |
| 32 | 1708.5 | 1349.917 |

该历史 artifact 未可靠保存 vLLM active cap，且曾报告异常的 `49140 MiB` visible VRAM，所以上表
只是审计线索，不进入 validator，也不能用 c32 值拒绝 Ferrum active-cap-16 的结果。G00 新鲜、
same-host、相同 active-cap baseline 的 LCB 才是正式标准。

2026-07-23 的 addressed vLLM paged-attention scoped diagnostic 绑定 clean source
`6a596558` 和 binary SHA256 `e8d12cc...`，把 decode attention 从
`5.826 ms/wave` 降到 `0.887 ms/wave`。随后 clean source `d0d2e4f5` 和 binary
SHA256 `3b09cb93...` 消除部分 resource/identity 重建：profile-off c1 从
`39.5687 +/- 2.0459 tok/s` 提升到 `46.0342 +/- 0.7937 tok/s`，
`submitted_wave_total=12.888 ms` 达到 `<=13.5 ms`，但
`resource_prepare_attempt=3.603 ms` 仍比 `<=3.5 ms` 硬线高 `0.103 ms`。
因此 scoped diagnostic 仍为 REJECT，且正式 `76.158 tok/s` legacy 90% 线仍差
`30.124 tok/s`。完整证据记录在
[`G08B_QWEN35_35B.md`](G08B_QWEN35_35B.md)；下一次 paid M2 performance
diagnostic 暂停。事后源码审计确认 `bench-serve` c1 的 outer batch 构造发生在
`resource_prepare_attempt` 计时区间之外，因此 artifact 中“缓存 singleton batch 可把
该指标降到 `<=3.45 ms`”的初始假设无效。先在现有 typed `wave_timing` 中把该指标拆为
work-shape/request preparation、step admission 和 submission-wave preparation，
再基于占比最高的实测阶段提出下一项源码预测；不得用另一轮 GPU sweep 代替该定位。

2026-07-24 的本地真实 Qwen3.5 Metal 诊断完成了该拆分。step admission 中
`backing_claim=4.174 ms`，占 `step_admission=4.591 ms` 的 `90.9%`；demand、
logical claim 和 frame lease 均不是首要瓶颈。源码根因是 backend-neutral
`ReusableExecutionPolicy` 被错误绑定到 CUDA graph/device executable capture 开关，导致
graph-disabled CUDA 和 Metal 每个 wave 重复申请/释放 workspace backing。历史 CUDA
703-case 和 `d0d2e4f5` artifact 均明确记录 `FERRUM_BATCHED_GRAPH=0`、startup
preparation disabled、dynamic-pool `live_segments=0`，因此它们证明 vNext 正确性路径，
不证明 workspace reuse 已启用。

dirty candidate `89d3f66d + source.diff` 将 workspace bucket policy 与 device capture
解耦后，同请求的 `backing_claim` 降至 `0.262 ms`，`step_admission` 降至
`0.692 ms`，`resource_prepare_attempt` 从 `7.528 ms` 降至 `1.170 ms`；device capture
仍为 disabled/unsupported，domains 1/4/5 则保留 lane-stable live segments。32/32 waves
完成，failed wave 和 product-visible deferral 均为 `0`。该结果仅为
`KEEP_DIAGNOSTIC`，不是正式性能或正确性 PASS；artifact 位于
`/Users/chejinxuan/ferrum-bench/artifacts/runtime-vnext-20260724/workspace-reuse-metal-dirty-89d3f66d/`。

clean source `e66ade7f0c0cd88ffc55c9a3c5a9ac902c68f58d` 已按上述合同完成一次
1x RTX 4090 bounded smoke。`c03-001 run`、`c05-001 serve`、`c06-001 stream`
正确性 `3/3 pass` 后，profile-off c1 `random 64/32`、`100 x 3` 完成
`300/300` measured requests，error/quality issue 均为 `0`，token source 为 usage。
`resource_prepare_attempt` 从 `3.6031 ms` 降至 `1.1426 ms`（`-68.29%`），
graph capture 保持关闭且 lane-stable live segments 非零，命中预期 signal。
吞吐从 `46.0342` 提升至 `55.5897 +/- 0.2898 tok/s`（`+20.76%`），但仍比
`76.1583 tok/s` 正式 floor 低 `20.5685 tok/s`（`27.01%`），所以结论仅为
`KEEP_DIAGNOSTIC`，不是 G09 PASS。三次 client event/usage mismatch 使 ITL comparison
不具资格，不得从该 artifact 得出 ITL no-regression 结论。

完整 artifact 通过 GitHub transfer 回传，SHA256 为
`e25500b9da8ef546f5cb70b0785cd81a15f8b26b0d4a94fa1d2618a439363445`；详情见
[`G08B_QWEN35_35B.md`](G08B_QWEN35_35B.md)。Vast `45319871` 已
`stopped/exited`。在下一次 paid run 前，必须从现有 artifact 的
`host_encode_submit=6.224 ms`、`completion_round_trip=5.629 ms` 和 device timing 中提出
单一 source-level hypothesis；不得因本轮 KEEP 直接启动 full sweep。

同一 clean source/binary 的后续 bounded full-profile 已完成该定位：4/4 请求正确，
75 个 decode wave 中每 wave 均有 174 条 eager command、40/40 个 reusable candidate
落在 startup preparation 之外，replay 为 0。`device_runtime_submit=5.682 ms/wave` 中
`enqueue_commands=5.614 ms/wave`，而 completion-worker queued wait 仅
`0.046 ms/wave`。因此失败类收敛为
`stable-decode-command-program-outside-reusable-preparation`，不是 paged-attention
或 scheduler/resource admission 重新设计问题。full-profile 的 `7.4649 tok/s` 仅用于
归因，不可与 profile-off 性能比较。

历史实现只证明 reusable executable 的产品路径真实存在，不能直接提供本轮性能目标。
`beb3e63c` 的 `1.819 ms/wave` 来自真实 Qwen3.5-4B `serve`，但差分中的 36 个
wave 混合 prefill/decode，且 measured 区间仍发生 6 次 capture。`b38e9645` 的
`2432/2432` 也是 Qwen3.5-4B 真实请求，但它表示 76 个 decode wave 的 candidate
segment 全命中，不是全部 command replay；同一 artifact 实际为 `7980` replay、
`2660` eager（75% command replay），enqueue `3.27695 ms/wave`。模型、workload 和
统计口径均与当前 M2 不同；此外 `a0038a0e` 使用 full kernel attribution，会在 eager
command 前后记录 CUDA event，因此 owner 数可用而 enqueue 时长不能与历史 basic-profile
直接比较。二者不得作为 `a0038a0e` 或下一候选的直接验收线。

`a0038a0e` 的 `>=150` replay、`<=24` eager、`<=2.0 ms` enqueue 是该次 paid
diagnostic 预先声明且已失败的假设，保留为 immutable REJECT 证据，不循环复用。
该候选已证明 typed product policy integration 生效；后续问题不是再接一次开关，也不是
新增 paged-attention 内核或第二套 command cache。

诊断产物 SHA256 为
`ebb2e401276fc5767ef96bfa66373967f6d242d9bbacd7ccf938dac27fbb59b6`，本机验证路径为
`/Users/chejinxuan/ferrum-bench/artifacts/runtime-vnext-20260724/current-sha-full-profile-e66ade7f/`。
该轮约 12 分钟、`$0.0939`；Vast `45319871` 已 `stopped/exited`，计费或过渡 sibling 为 0。

clean candidate `a0038a0e` 已把 typed reusable program 从 legacy graph policy 中独立出来，
且 `run/serve/config` 使用同一产品可见策略。受影响本地 gate 全部通过后，同一 RTX 4090 的
`c03/c05/c06` 正确性通过 `3/3`。默认配置保持 `FERRUM_BATCHED_GRAPH=0`，
同时 `FERRUM_REUSABLE_EXECUTION=1`；startup 在 `1.389 s` 内准备 240 个 executable，
captured/uploaded/resident 均为 240，rejected/deferred 为 0，
`eager_fallback_required=false`。

该候选只命中部分 performance signal：40/40 candidate segment 每 wave 均 cache-hit/replay，
outside-preparation 和 request-time capture/upload 均为 0；但只 replay
`121 commands/wave`，仍有 `53 commands/wave` eager。enqueue 从 `5.614 ms` 降到
`2.509 ms/wave`，没有达到 `<=2.0 ms`；profile-off 完成 `300/300` 且错误为 0，但
`55.4898 +/- 2.1109 tok/s` 没有超过 `55.5897` 既有候选，正式
`76.1583 tok/s` floor 仍差 `20.6685 tok/s`。因此 artifact 明确为
`typed-reusable-program-partial-command-coverage` REJECT，不是 G09 progress PASS：

```text
CUDA REUSABLE PROGRAM INTEGRATION REJECT: /workspace/ferrum-artifacts/runtime-vnext-reusable-program-a0038a0e-20260724T0423/diagnostic-summary.json
```

完整 GitHub artifact SHA256 为
`f7ab54160372956f39b868df20cc6ffedca6c06ce953cad43110953d13dbb80d`，本机已验证于
`/Users/chejinxuan/ferrum-bench/artifacts/runtime-vnext-20260724/reusable-program-cuda-a0038a0e/`。
该轮约 29 分钟、`$0.2269`，实例已 `stopped/exited`。下一轮 paid work 被 source
analysis 阻塞：必须先按 typed command owner 分类 53 条 eager command，并预测哪一类进入
prepared replay 后可把 enqueue 降到 `<=2.0 ms`；不得继续试跑发现 owner。

保存的 scheduler trace 已完成该分类。每 wave 的 174 条命令由 `1` 条 token upload 和
173 条 provider command 构成：

| owner | commands/wave | 当前路径 |
|---|---:|---|
| RMSNorm（40 层 + final） | 41 | replay |
| routed/shared MoE | 40 | replay |
| residual add | 40 | replay |
| gated-delta recurrent attention | 30 | eager compute |
| causal paged attention compute | 10 | eager compute |
| causal address binding | 10 | eager dynamic binding |
| token embedding | 1 | eager compute |
| last-token logits projection | 1 | eager compute |
| token upload | 1 | eager input boundary |

因此 `121 replay + 53 eager = 174` 已无未知 owner。源码同时证明当前抽象把两类责任混在
同一个 `CudaDeviceCommand`：一类是 CUDA graph 真正读取的稳定 kernel/BLAS 参数，另一类是
只为 fence 生命周期保留的动态 KV/state region。causal compute 虽然从 binding workspace
读取 page address，仍把动态 page region 放入 compute payload，导致 reusable-address
合同被剥离；recurrent attention、embedding 和 logits 则仍直接绑定 sequence/request 地址。
把这些地址强行标记稳定或把 dynamic binding 捕获进 graph 都会破坏正确性。

源码改造按 owner 精确分三步：先拆 captured launch region 与 fence-only dependency，
使十个 causal compute 恢复 replay，预测 `131 replay / 43 eager`；再引入 lane-stable
recurrent-state binding，预测 `161 / 13`；最后将 embedding/logits 移入 lane-stable
I/O staging，预测 `163 / 11`。十个 causal address binding 与一个 upload 保持显式 eager
边界；这些数字是下一 artifact 的可证伪预测，不是提前宣告 PASS。

最终 source checkpoint 是 backend-neutral 的“plan-owned static command program +
typed per-wave binding patch”合同及 CUDA 实现：cache hit 不再逐 node 重新
`encode_selected`；动态 state/IO 只能通过 typed binding 或显式 eager boundary 更新；
cached program 不得持有 request-owned resource；completion fence 必须继续覆盖所有 binding
target。Metal 可以复用 program/binding 生命周期合同，但保留自己的 command-buffer/pipeline
实现。完成本地 contract、生命周期和故障测试前禁止再开 paid GPU；下一轮必须重新声明将减少
的具体 owner 数、`provider_node_encode`/`enqueue_commands` 预测和停止线。正式
`76.1583 tok/s` floor 不变。

clean source `3ac6b65a` 随后验证了第一阶段 owner 移动，但也暴露了 program segmentation
的根缺陷。`c03 run`、`c05 serve`、`c06 streaming` 真实 CUDA 正确性 `3/3 pass`；
75 个 decode wave 精确达到预言的 `131 replay / 43 eager`。但是同一 full-profile
workload 中，exact-shape causal varlen prefill 进入 reusable segment 后，使相邻稳定
RMSNorm/MoE/residual 的 prefill replay 分别丢失 `275/250/250` 条，只换回 `50` 条 causal
prefill replay。整个 workload 的 replay 最终只增加 `25` 条，而不是 decode 局部看到的
`750` 条。

profile-off c1 `random 64/32`、`100 x 3` 完成 `300/300`，错误和质量问题均为 `0`，
但吞吐为 `51.5331 +/- 4.3387 tok/s`，比 `a0038a0e` 的 `55.4898` 低 `7.13%`，
距正式 floor 仍差 `24.6252 tok/s`（`32.33%`）。因此本轮按
`mixed-topology-replay-segment-poisoning` REJECT：

```text
CUDA CAUSAL REPLAY ENVELOPE REJECT: /workspace/ferrum-artifacts/runtime-vnext-causal-envelope-3ac6b65a-20260724T0548/diagnostic-summary.json
```

下一步不是提高 executable cache 容量，也不是继续搬 recurrent owner。必须先让 replay
eligibility 成为 typed launch-topology contract：decode V1/V2 可复用，尚无稳定 envelope
的 varlen/fallback prefill 是显式 eager barrier，单个 dynamic-key miss 不能污染相邻 stable
command。下一付费 artifact 的可证伪预测为：

- decode 保持 `131 replay / 43 eager`；
- prefill RMSNorm/MoE/residual replay 恢复为 `1230/1200/1200`；
- causal prefill replay 为 `0`；
- scoped workload 总计 `13455 replay / 5985 eager`；
- 在这些结构数字命中前，不运行另一轮 `100 x 3` 性能。

本轮压缩包 SHA256 为
`1e0b9774ff7822ffe3336b39c7afb96b78171a40e5c0a17ba9f4f9863108b8d7`。
后续已通过临时 GitHub branch 传回并在本机校验，路径为
`/Users/chejinxuan/ferrum-bench/artifacts/runtime-vnext-20260724/causal-envelope-cuda-3ac6b65a/`。

clean source `4df3d63a` 随后把 eligibility 固化为 typed launch-topology contract：
decode V1/V2 继续 replay，exact-shape varlen/fallback prefill 成为带独立 fence
dependency 的 eager barrier。CUDA candidate build 与 manifest READY，`c03 run`、
`c05 serve`、`c06 streaming` 再次 `3/3 pass`。full-profile 精确命中全部预言：

- scoped native replay `12730 -> 13455`，目标 `13455`；
- prefill RMSNorm/MoE/residual replay 为 `1230/1200/1200`；
- prefill causal replay 为 `0`；
- 75 个 decode wave 均为 `131 replay + 42 eager provider + 1 upload`。

因此 topology isolation 作为结构改造 KEEP；它没有改变 decode 主瓶颈。正式
profile-off c1 `random 64/32`、`100 x 3`、seed `9271` 完成 `300/300`，
usage token count、请求错误和质量问题均合格，但只有
`48.3721 +/- 4.9645 tok/s`。相对 `3ac6b65a` 为 `-6.13%`，相对
`a0038a0e` 稳定 checkpoint 为 `-12.83%`；两个近期候选的 CI 重叠，因此不把这段小差异
另行归因。绝对 floor 仍差 `27.7862 tok/s`（`36.48%`），本轮按
`formal-throughput-floor-miss-after-topology-isolation-hit` REJECT：

```text
CUDA TOPOLOGY REPLAY BARRIER REJECT: /workspace/ferrum-artifacts/runtime-vnext-topology-barrier-4df3d63a-20260723T231634Z/diagnostic-summary.json
```

archive 为 `35,174,029` bytes，SHA256
`c433fffb9e77bfea543a66a6e5df5f4420f6cb58d80743e61f4dd2516084657d`。
首次 GitHub upload 在远端连接超时，archive 暂留 retained instance `45319871`；
实例已确认 `stopped/exited`。禁止为重复 benchmark 单独恢复实例。

下一 source lever 不再调整 replay key 或 prefill segmentation，而是建立
plan-owned immutable CUDA command program 与 typed per-wave binding patch。
当前 decode 仍有 `43` 条 eager command，其中 `30` 条属于 recurrent attention；
必须先用 lane-stable state indirection 证明 recurrent state 地址、RAII 和 fence
生命周期，再降低 eager owner 数。下一 paid run 前必须有本地 command-program
ownership、binding lifetime、recurrent address-scope 和无硬编码合同，并预言
`decode eager <43`、`recurrent eager <30` 或可测的 host enqueue 下降；否则停止在
source analysis。

clean source `393a9a40` 随后完成 recurrent state indirection 和 wave-level
program-binding prelude。真实 CUDA `c03 run`、`c05 serve`、`c06 streaming`
正确性为 `3/3 pass`。75 个 decode wave 全部直接观测到
`161 replay / 2 node-attributed eager`；补入唯一 coalesced binding prelude 和
显式 input upload 后，完整结构精确命中预言的 `161/4`，相对 `131/43` 大幅减少
eager command。

结构命中后才运行 profile-off c1 `random 64/32`、`100 x 3`、十次 warmup、
seed `9271`。`300/300` 请求完成，usage token、错误和质量门均通过，但性能只有
`44.947749 +/- 3.671515 tok/s`：相对 `4df3d63a` 下降 `7.079%`，相对预声明
KEEP 线 `55.5897` 仍差 `19.144%`，相对正式 `76.1583` floor 仍差
`31.210551 tok/s`（`40.981%`）。因此本轮按
`formal-throughput-floor-and-keep-threshold-miss-after-decode-structure-hit`
REJECT：

```text
CUDA PROGRAM BINDING PERFORMANCE REJECT: /workspace/ferrum-artifacts/runtime-vnext-program-binding-393a9a40-20260724T004652Z/diagnostic-summary.json
```

同机、同 workload 的 host-wall 边界给出可归因结果：

| decode boundary | `4df3d63a` | `393a9a40` | delta |
|---|---:|---:|---:|
| resource prepare | `1.507994ms` | `1.894054ms` | `+25.601%` |
| host encode/submit | `6.122343ms` | `5.066258ms` | `-17.250%` |
| completion round trip | `6.689556ms` | `8.351333ms` | `+24.841%` |
| submitted wave total | `13.626184ms` | `14.433627ms` | `+5.926%` |

这证明 replay 和 host-submit 方向有效，但当前 coalescer 只合并 Rust command 外壳，
没有合并 30 个 recurrent 加 10 个 causal provider binding 的 native transfer 与
resource workspace。下一 source checkpoint 必须把 binding ownership 上移到 compiled
plan：一个 typed `ProgramBindingLayout`、一个 lane-owned device arena、一个连续 host
patch 和每 wave 最多一次 H2D upload；provider compute program 只编译一次并使用固定
offset。不得继续在 per-node invocation binding workspace 上打补丁。

另一 paid run 前，本地合同必须证明 aggregate workspace、单 upload、request-owned
address 隔离、RAII/fence 和无硬编码。下一 artifact 必须保持 `161/4`，把 binding native
transfer 降到 `<=1`，并把 decode resource prepare、completion round trip、total
分别恢复到 `<=1.507994ms`、`<=6.689556ms`、`<=13.626184ms`；之后才允许以
`>=55.5897 tok/s` 作为最小 KEEP 线。`393a9a40` 禁止原样复跑。

完整 archive SHA256 为
`38dba973258d622f40d550794b2b2d5b829fe4a85fe0ce075d913382aa2e4146`，
暂留 stopped Vast instance `45319871`。实例已确认 `stopped/exited`，无 billable
sibling；GitHub transfer 尚未完成，不能把远端留存写成已 copy-back。

clean source `e5a6e6c1` 随后完成 compiled `ProgramBindingLayout`、lane-owned
binding arena 和 typed aggregate host patch。真实 CUDA `c03 run`、`c05 serve`、
`c06 streaming` 为 `3/3 pass`；75 个 decode wave 均有 40 个 recurrent/causal
binding owner、恰好一个 aggregate prelude 和恰好一次 physical transfer，缺失、重复、
非法 attribution 均为 `0`：

```text
CUDA PHASE B AGGREGATE BINDING TRACE PASS: /workspace/ferrum-artifacts/runtime-vnext-phase-b-trace-e5a6e6c1-20260724T030818Z/full-profile/trace-summary.json
```

结构命中不等于性能通过。profile-off c1 `random 64/32`、`100 x 3`、十次 warmup、
seed `9271` 完成 `300/300`，错误和质量问题均为 `0`，但只有
`45.931942 +/- 2.411397 tok/s`。距预声明 KEEP 线 `55.5897` 仍差
`9.657758 tok/s`（`17.37%`），距正式 floor `76.1583` 仍差
`30.226358 tok/s`（`39.69%`），因此只 KEEP 结构、REJECT 性能：

```text
CUDA PHASE B C1 THROUGHPUT REJECT: /workspace/ferrum-artifacts/runtime-vnext-phase-b-trace-e5a6e6c1-20260724T030818Z/perf-off/throughput-summary.json
```

同 workload 的 profile-off host wall 相对 `393a9a40` 显示 host encode/submit
`5.066258 -> 4.710335 ms`（`-7.03%`），completion round trip 仅
`8.351333 -> 8.264359 ms`（`-1.04%`），submitted wave total
`14.433627 -> 13.908279 ms`（`-3.64%`）。full profile 的 exact submission
counter 已把主区间缩到 device：decode device execution 平均 `8.206444 ms`，
completion worker queued wait 仅 `55.987 us`，readback host 为 `348.865 us`；
device execution 与 fence host wait 可能重叠，禁止相加。

剩余阻塞不再需要另一轮吞吐排除法。当前 CUDA kernel profile 在任何 reusable
executable replay 后直接把整批 command timing 置为 `BackendUnsupported`。本 artifact
的 `18,450` 条 native event 只有 `825` 条 measured，`17,625` 条 unavailable；
其中 `16,605` 条 replayed compute 全部 unavailable。因此 exact submission 能证明
`8.206444 ms` 在 device program 内，却不能把它继续分解到 replay segment/op/kernel。
下一 source checkpoint 是 typed physical timing span：eager command 使用
`[command_index, command_index + 1)`，replay executable 使用真实
`[start_command_index, end_command_index)`，consumer 按显式 span join，禁止把一个
segment duration 复制到多个 logical command。完成本地 contract 和 CUDA feature compile
前，不允许再开 paid GPU 或重跑 `100 x 3`。

本轮 diagnostic 决策为
`REJECT_PERFORMANCE_KEEP_STRUCTURE`，不计 G09 formal progress。archive 为
`5,157,454` bytes，SHA256
`b0624ac1c8281897a1d6e1f4db77a95cd5959b69a1edcd5858b98c955a51ba7d`，
已通过 GitHub branch
`artifact/runtime-vnext-phase-b-e5a6e6c1-20260724` 传回并在本机校验，路径为
`/Users/chejinxuan/ferrum-artifacts/runtime-vnext-phase-b-trace-e5a6e6c1-20260724T030818Z/`。
retained instance `45319871` 已再次确认 `stopped/exited`，无 billable sibling。

clean source `33fc6e46` 随后实现 typed physical execution span，并用同一 retained
1x RTX 4090 完成一次 bounded diagnostic。CUDA binary SHA256 为
`7e2b279eb8899c3dd3c7a0da938b00cbc11580dca679986347844ebb78101eb6`；`c03 run`、
`c05 serve`、`c06 streaming` 均为 `KEEP`。本轮没有重跑 `100 x 3`，只用
c1 `random 64/16`、4 requests、1 repeat 获取 full-profile shape。

物理计时覆盖已经从 replay 全部 unavailable 收敛为 `405/405 measured`，logical replay
command `16,605` 条的 duplicated elapsed 为 `0`，ownership/range error 为 `0`。decode
device interval 平均 `8.203624ms`，精确拆为 replay `6.995859ms`（`85.3879%`）、
eager `1.197175ms` 和 gap `6.527573us`。这把下一瓶颈从“device program”进一步收敛为
“CUDA graph replay body”；program binding、scheduler gap 和 completion queue 不再是本轮
第一假设。eager 的主要次级项是 last-token dense projection
`1.069725ms/decode frame`。

```text
CUDA REPLAY EXECUTION SPAN TRACE PASS: /workspace/ferrum-artifacts/runtime-vnext-replay-span-33fc6e46-20260724T041818Z/full-profile/replay-span-summary.json
CUDA REPLAY EXECUTION BREAKDOWN PASS: /workspace/ferrum-artifacts/runtime-vnext-replay-span-33fc6e46-20260724T041818Z/full-profile/replay-breakdown-summary.json
```

该 artifact 的 decision 为 `KEEP_OBSERVABILITY_CHECKPOINT`，
`formal_performance_goal_progress=false`，因此不改变 `76.1583 tok/s` floor 或 G09 Open
状态。archive SHA256 为
`74c0e665fd7e2830e12c681831deb7775b35b504d45811462e6a18155ab679d3`，已通过
GitHub branch `artifact/runtime-vnext-replay-span-33fc6e46-20260724` 回传并在本机验证；
branch 最新 lifecycle commit 为 `d3f70381`。Vast `45319871` 已
`stopped/exited`，无 billable sibling。

下一 paid performance work 仍被 source/profile hypothesis 阻塞。只允许先对 decode graph
运行 kernel-activity scoped diagnostic，要求 `>=95%` replay duration 被 kernel/native-op
分类；未命中则停止并修复 profiler，不运行吞吐。命中后只优化累计占比最高的一个 kernel
family，并预先声明该 family 的 device-time 降幅及对 `8.203624ms` decode interval 的预测，
再决定是否值得运行 profile-off performance smoke。

该 scoped diagnostic 已在 clean source `36b1b3af` 上按停止规则执行并
`REJECT_INSTRUMENTATION_LIFECYCLE`：12 个 fingerprint 全部 join，但 405 个 NVTX replay
range 未正常闭合，kernel-duration sum / Ferrum replay wall-time=`1.170497`；原始 SQLite
显示 405 个 start 共享同一个 profiler terminal end，v2 projection 重放把 405/405 标记为
nested。它没有运行 throughput，也没有产生 G09 performance progress；受污染的 GEMV/Marlin/
MoE-align 排名不得用于选择优化项。

修复提交 `8f44dd8c` 改为显式成对标记，并把 analyzer coverage 口径切到 projected GPU
wall-time。下一次 paid work 仍只允许复用同一 stopped 4090 做一次 bounded attribution：
CUDA feature compile、`c03/c05/c06` 必须先 PASS；range count 必须逐 fingerprint 对齐、
nesting/child=`0`、projection coverage=`0.95..1.10`。任一条件失败即保存 REJECT 并停机；
全部命中后才接受 top kernel family，仍不直接运行 `100 x 3` 或 G09 full sweep。完整失败
artifact、GitHub commit `804cb22a` 和 SHA256
`b6f0558f98675ad043d31f5958aa45e60f1c0ed38b6fad7f0173d45d02d8a143`
记录在 [`G06_OBSERVABILITY_PERF_LAB.md`](G06_OBSERVABILITY_PERF_LAB.md)。

后续 clean source `32c53a6b` 的同范围 bounded diagnostic 已命中全部 correlation stop
condition，精确结果记录在 [`G06_OBSERVABILITY_PERF_LAB.md`](G06_OBSERVABILITY_PERF_LAB.md)。
这只关闭 profiler instrumentation blocker，不改变 G09 Open 状态或 `76.1583 tok/s` floor。
dominant decode graph 的 owner kernel work 为 recurrent attention `46.0304%`、MoE
`40.3663%`、causal attention `11.8639%`；paged-attention kernel 本体仅 `1.2260%`。

recurrent attention 内部第一组可修改瓶颈是五个 F16 projection，其中 QKV GEMV 占全部
kernel work `16.7828%`、Z GEMV `8.7463%`、output GEMV `8.8117%`，B/A 各约 `0.95%`。
当前 vNext 每层分别提交 QKV、Z、B、A 四次 projection；当前 vLLM Qwen GDN 则把它们组织为
QKVZ 和 BA 两个 merged projection。Ferrum 采用自己的 typed weight/program 合同：在模型
prepare 时把现有 immutable source weight 物化为 QKVZ/BA composite，热路径只执行两次
projection，不引入 model-name/GPU-name 分支，也不复制 Python runtime 设计。

下一项 CUDA candidate 的必需信号为：30 个 recurrent layer 每层 projection kernel 从
`4` 降到 `2`，dominant decode graph 总 kernel node 至少减少 `60`；`c03/c05/c06` 正确性
必须先通过，replay wall 和 profile-off throughput 均不得回退。若 graph-node 信号未命中，
立即 REJECT source contract；只有命中且窄 profile 显示绝对时间改善，才运行一次
profile-off c1 smoke，不直接运行 G09 full sweep。

### 2026-07-24 GDN Input Fusion Result

Clean source `cefb4de25036a818fd3d0628a63b4fde3b74d81d` implements the typed
QKVZ/BA input fusion and the reviewed v5 operation, weight-layout, numerical
tolerance, and backend extent contracts. The bound one-RTX-4090 CUDA binary
SHA256 is
`fa76fc9af5cdef07d6a887cc821b4347d139263a6b1618b542af0f29cb947800`;
the final incremental release build took `295.619278 s`.

Correctness preceded every performance measurement:

- the CUDA packed-extent, CUDA/CPU packed numerical parity, and production replay
  symbol tests each passed `1/1`;
- actual-model `c03-001 ferrum run`, `c05-001 ferrum serve`, and `c06-001`
  streaming serve passed `3/3`;
- the focused runner printed
  `FERRUM RUNTIME VNEXT FOCUSED DIAGNOSTIC KEEP`, with zero blocker-scan match.

The predeclared structural signal was exact. The same Qwen3.5 program has 30
recurrent layers; the `32c53a6b` trace reported `13` compute dispatches per layer
and `390` per correlation, while `cefb4de2` reported `11` and `330`. This is an
observed reduction of exactly `60` compute dispatches per correlation, not a
source-count estimate.

The same-hardware full-profile timing also moved in the predicted direction over
75 decode correlations:

| signal | `32c53a6b` | `cefb4de2` | delta |
|---|---:|---:|---:|
| replay elapsed | `7.362019 ms` | `6.888088 ms` | `-0.473931 ms` (`-6.44%`) |
| complete device span | `8.608580 ms` | `8.097945 ms` | `-0.510635 ms` (`-5.93%`) |
| GDN dispatches/correlation | `390` | `330` | `-60` |

Profile-off c1 used the canonical diagnostic workload: random `64/32`, 100
measured requests and 10 warmups per repeat, three repeats, seed `9271`,
`--fail-on-error --require-ci`, and usage-derived token counts. It completed
`300/300` with zero request, stream, output, or quality error. Repeat throughput
was `48.2964 / 46.4895 / 55.4540 tok/s`, with mean
`50.0800 +/- 11.7781 tok/s` and CV `9.4667%`.

This is `4.1480 tok/s` (`9.03%`) above the latest available current-path
profile-off checkpoint at `45.931942 tok/s`, so the optimization source is kept.
It remains `5.5097 tok/s` (`9.91%`) below the higher `55.5897 tok/s` KEEP line
and `26.0783 tok/s` (`34.24%`) below the unchanged `76.1583 tok/s` formal floor.
The decision is therefore:

```text
CUDA GDN INPUT FUSION CHECKPOINT KEEP: /workspace/ferrum-artifacts/runtime-vnext-gdn-fusion-cuda-cefb4de2-20260724T080613Z/diagnostic-summary.json
CUDA GDN INPUT FUSION FORMAL PERFORMANCE REJECT: /workspace/ferrum-artifacts/runtime-vnext-gdn-fusion-cuda-cefb4de2-20260724T080613Z/diagnostic-summary.json
```

The complete GitHub-transfer archive is
[runtime-vnext-gdn-fusion-cuda-cefb4de2-20260724T080613Z.tar.zst](https://github.com/sizzlecar/ferrum-infer-rs/releases/download/untagged-711d3e8abdfcbe0c8b41/runtime-vnext-gdn-fusion-cuda-cefb4de2-20260724T080613Z.tar.zst),
asset id `488205719`, size `30,043,998` bytes, SHA256
`0521a5baa3b98398ce2e4683576b3e558500da74f6cf2479b369d53fef41e144`.
It was downloaded and revalidated locally at
`/Users/chejinxuan/ferrum-artifacts/runtime-vnext-gdn-fusion-cuda-cefb4de2-20260724T080613Z/`.
Vast instance `45319871` is confirmed `stopped/exited`, with no running or
scheduling sibling.

No repeat of this candidate or G09 full sweep is authorized. The next work is
offline source/profile analysis of the post-fusion graph. It must compare the
remaining recurrent output projection against the MoE routing/alignment/Marlin
family, select one predicted high-return lever, and name an absolute device-time
change before another paid run. The retained `330` dispatch count, `6.888088 ms`
replay time, product `3/3`, and zero-error profile-off result become no-regression
requirements for that candidate.

### 2026-07-24 Single-Token MoE Direct-Alignment Result

Clean source `992153a4de14bee734c97f54d2b78b754d7737f7` replaces the
single-token Qwen3.5 MoE route-plus-generic-align sequence with one typed
`SingleTokenDirectMarlin` plan. The CUDA router writes the stable
expert/token/block metadata required by Marlin directly; multi-token prefill
continues to use the generic alignment path. The bound one-RTX-4090 binary
SHA256 is
`393f377659560db9c5df564bf544fbf7d435780059c2fb71fbfbe1e797d1ae1a`.

Correctness ran before performance:

- the focused replay source contract passed `8/8` and the kernel library tests
  passed `17/17`;
- actual Qwen3.5-35B `c03-001 ferrum run`, `c05-001 ferrum serve`, and
  `c06-001` streaming serve passed `3/3`;
- the focused runner printed
  `FERRUM RUNTIME VNEXT FOCUSED DIAGNOSTIC KEEP`;
- positive blocker-log matches and structured request-quality issues were both
  `0`.

The full-profile structural result exactly matched the predeclared prediction.
Across 75 decode correlations and 3,000 MoE node observations, the old
`cefb4de2` candidate used `12` physical compute dispatches per single-token MoE
node; `992153a4` uses `11`. Multi-token prefill remains at `12`, so the change
did not broaden into the generic route. Mean replay device time moved from
`6.888088 ms` to `6.233041 ms`, a reduction of `0.655048 ms` or `9.51%`.
This agrees with the prior Nsight attribution of about `0.679 ms/token` to
`moe_align_block_size_pair_ids_f32`.

The canonical profile-off c1 workload remained random `64/32`, 100 measured
requests plus ten warmups per repeat, three repeats, seed `9271`,
`--fail-on-error --require-ci`, and usage-derived token counts. It completed
`300/300` with zero request or quality error. Repeat throughput was
`45.7588 / 46.4369 / 50.4481 tok/s`, with mean
`47.5479 +/- 6.2963 tok/s`. This is `2.5320 tok/s` (`5.06%`) below the
`cefb4de2` mean; the two three-repeat confidence intervals overlap, so this run
does not prove an end-to-end regression or improvement. It remains
`28.6104 tok/s` (`37.57%`) below the unchanged `76.1583 tok/s` formal floor.

The decision therefore separates the deterministic structural evidence from
the noisy product metric:

```text
CUDA MOE DIRECT ALIGN STRUCTURAL KEEP: /workspace/ferrum-artifacts/runtime-vnext-moe-direct-align-cuda-992153a4-20260724T091507Z/diagnostic-summary.json
CUDA MOE DIRECT ALIGN CANONICAL PERFORMANCE REJECT: /workspace/ferrum-artifacts/runtime-vnext-moe-direct-align-cuda-992153a4-20260724T091507Z/diagnostic-summary.json
```

The complete GitHub-transfer archive is
[runtime-vnext-moe-direct-align-cuda-992153a4-20260724T091507Z.tar.zst](https://github.com/sizzlecar/ferrum-infer-rs/releases/download/untagged-711d3e8abdfcbe0c8b41/runtime-vnext-moe-direct-align-cuda-992153a4-20260724T091507Z.tar.zst),
asset id `488260567`, size `29,911,858` bytes, SHA256
`0550e682170a20bed55a53627ac879c98cd2a522d1d1f8dc3f115cd02fc51eff`.
It was range-resumed after an interrupted download and revalidated locally at
`/Users/chejinxuan/ferrum-artifacts/runtime-vnext-moe-direct-align-cuda-992153a4-20260724T091507Z/`.
Vast instance `45319871` is confirmed `stopped/exited`, with no paid sibling.

The workspace all-target test compiled and then encountered two existing
network-sensitive CLI E2E failures: the inherited host proxy returned a
Hugging Face `404 Repository not found` before the tests could observe their
expected local missing-model message. Therefore this checkpoint does not claim
the unit source-gate PASS line.

No repeat of `992153a4` or G09 full sweep is authorized. Offline work must now
rank the post-fusion projection/GEMV, recurrent kernel, Marlin, and
shared-versus-routed scheduling costs against current source and the vLLM
baseline. Another paid run requires one falsifiable lever, a named device-time
or trace-shape delta, and a reject threshold; the current `11` MoE dispatches,
`6.233041 ms` replay time, product `3/3`, and zero-error c1 result are the new
structural no-regression requirements.

### 2026-07-24 Typed Direct-Program And Binding-Only Results

Clean source `f435bec9498d51e77c0e08b71ea29016f3eb74ed` moved cache hits onto
the exact typed CUDA direct-program path. Qwen3.5-35B `c03/c05/c06` passed
`3/3`; the canonical c1 run completed `300/300` requests at
`59.887970 +/- 17.927647 tok/s`. Health evidence reported `12,210` direct
waves, `32,010` direct segments, `468,600` binding-node applications, and zero
direct fallback or catalog-epoch miss. This is the best current-path mean in
the immediate sequence of CUDA direct-program candidates, but it remains
`16.270330 tok/s` (`21.36%`) below the unchanged `76.1583 tok/s` floor and
therefore does not satisfy G09.

Clean follow-up `8c58e3ea0017c85865c5b5f56d0b02e94f36063a` installed specialized
binding-only encoders for recurrent and causal attention. It retained the same
direct counts and again passed `c03/c05/c06` `3/3`, but the same c1 command
produced only `50.272873 +/- 3.385823 tok/s`. The predeclared
`host_encode_submit <2.319187 ms` signal also missed at `3.076369 ms`.
Resource preparation and host postprocess regressed by similar proportions,
while completion changed only from `7.482255 ms` to `7.628776 ms`; the run is
classified as
`profile-off-host-wide-latency-regression-with-specialized-binding-coverage`.
It cannot prove that the simpler binding source path caused or fixed the
end-to-end delta.

The immutable local evidence roots are:

- `/Users/chejinxuan/ferrum-artifacts/runtime-vnext-direct-program-cuda-f435bec9-20260724T115319Z-gated/`
- `/Users/chejinxuan/ferrum-artifacts/runtime-vnext-binding-only-cuda-8c58e3ea-20260724T122544Z/`

The latter contains `diagnostic-summary.json` with the paired stage metrics,
binary SHA256, clean source SHA, correctness KEEP line, command evidence, and
stopped-instance receipt. Both source checkpoints remain below the absolute
floor, so no repeat c1 or full sweep is authorized. The next paid diagnostic
requires existing full-profile or bounded same-process attribution to isolate
`provider_node_encode` from host-wide variance, followed by one source-level
hypothesis with a named stage delta.

### 2026-07-24 Same-Session Direct-Binding Attribution

The requested bounded attribution subsequently completed on the same retained
RTX 4090 with both binaries, one model cache, and `A-B-B-A` ordering. Each slot
used `profile-detail=basic`, random `64/32`, c1, 25 measured requests plus five
warmups, and seed `9271`. All `100/100` measured requests passed with usage
token counts, and all 4,440 direct waves completed with zero direct fallback
or catalog-epoch miss.

The reversal followed the binary in both halves:

| direct-path metric | A `f435bec9` | B `8c58e3ea` | B vs A |
|---|---:|---:|---:|
| provider-node encode | `1498.316 us` | `860.026 us` | `-42.60%` |
| host encode/submit | `3598.252 us` | `2534.590 us` | `-29.56%` |
| completion round trip | `7565.833 us` | `7495.563 us` | `-0.93%` |
| diagnostic throughput | `47.0844 tok/s` | `54.8719 tok/s` | `+16.54%` |

This resolves the earlier unpaired profile-off ambiguity in favor of retaining
the specialized binding-only source. It does not satisfy the formal
`76.1583 tok/s` floor and is not a G09 performance PASS. The immutable result
is:

```text
CUDA DIRECT BINDING AB ATTRIBUTION KEEP: /Users/chejinxuan/ferrum-artifacts/runtime-vnext-binding-ab-attribution-20260724T130106Z/diagnostic-summary.json
```

### 2026-07-24 Rejected Single-Token Packed-Decode Candidate

Clean candidate `67921b1c55a093c43e5e6a4ea5f60c7916a962df`, binary SHA256
`1a6c582c6af39bf22bfd9f9d6d9a138334ba1b397a6280d955bb376531b1d8d3`,
implemented a typed single-token recurrent topology. The CUDA build completed
in `295.411386 s`. Correctness ran before profiling or throughput:

- actual-model `c03-001 ferrum run`, `c05-001 ferrum serve`, and `c06-001`
  streaming serve passed `3/3`;
- all three cases passed execution-envelope, expectation, model-binding, and
  scenario-oracle checks;
- the runner printed `FERRUM RUNTIME VNEXT FOCUSED DIAGNOSTIC KEEP`.

The full-profile structural prediction was exact. Across 75 decode
correlations and 2,250 recurrent-layer observations, each recurrent layer
moved from `11 compute + 2 transfer` commands to `9 compute + 0 transfer`.
The recurrent contribution therefore moved from `330` to `270` compute
dispatches per decode correlation. Mean physical replay span moved from
`6.888088 ms` to `6.195692 ms`, an improvement of `0.692397 ms` or `10.05%`.
The full-profile diagnostic throughput also moved from `7.454970` to
`7.918762 tok/s`; profile-on throughput is not a product performance claim.

The bounded profile-off check used random `64/32`, concurrency one, 25 measured
requests plus five warmups per repeat, three repeats, seed `9271`,
`--fail-on-error --require-ci`, and usage token counts. It completed `75/75`
with zero request or quality error. Repeat throughput was
`54.6932 / 58.6079 / 50.5373 tok/s`, mean
`54.6128 +/- 10.0265 tok/s`. This missed the predeclared current-path
`59.887970 tok/s` line by `5.275174 tok/s` and the unchanged
`76.1583 tok/s` formal floor by `21.545504 tok/s` (`28.29%`). The smaller
diagnostic request count does not prove a formal regression, but it cannot
accept the candidate or authorize a full sweep.

The result is therefore structural KEEP and candidate REJECT:

```text
CUDA PACKED DECODE STRUCTURAL KEEP: dispatch-diagnostic/dispatch-summary.json
CUDA PACKED DECODE CANDIDATE REJECT: diagnostic-summary.json
```

Commit `b0286270` explicitly reverted the two candidate commits after targeted
verification: the CUDA replay source contracts passed `9/9`,
`cargo check -p ferrum-kernels --all-targets` passed, and formatting passed.
The rejected implementation remains in Git history for diagnosis but is not
present in the active source tree.

The sanitized GitHub evidence asset is
[runtime-vnext-packed-decode-cuda-67921b1c-20260724T140119Z-sanitized.tar.zst](https://github.com/sizzlecar/ferrum-infer-rs/releases/download/untagged-711d3e8abdfcbe0c8b41/runtime-vnext-packed-decode-cuda-67921b1c-20260724T140119Z-sanitized.tar.zst),
asset id `488525573`, size `28,247,157` bytes, SHA256
`152c6cb4257c7656ab7f3fd3722713da97f0e109b7216b9f2c7970969838cc07`.
It was verified locally under
`/Users/chejinxuan/ferrum-artifacts/runtime-vnext-packed-decode-cuda-67921b1c-20260724T140119Z/`.
Vast instance `45319871` is confirmed `stopped/exited`, with no paid sibling.

This exact candidate must not be rerun. Source review found that its packed
delta kernel normalizes Q/K inside each value-tile block. The initial successor
hypothesis was to move normalization to a once-per-key-head stage while
preserving the typed topology and indirect-state lifetime contract. That exact
hypothesis has now been evaluated and rejected below; it must not be proposed
again as an untested next step.

### 2026-07-24 Rejected Separate-Normalization Packed Decode

Clean source `a884f5d44e9bb68542e2dfe67d8310fb2071f227`, binary SHA256
`64e137337df0f120d7a4d415198a2d9a7a0921df4c34f3d8b7808fd6992bf357`,
implemented the once-per-key-head normalization hypothesis as a separate CUDA
dispatch followed by a prenormalized packed-delta kernel. The release build
completed in `296.234294 s`. Correctness preceded performance:

- the actual RTX 4090 CUDA parity test
  `normalized_packed_decode_matches_varlen_cuda_state_and_output` passed `1/1`;
- actual-model `c03 run`, `c05 serve`, and `c06 streaming serve` passed `3/3`;
- request errors, quality issues, and blocker-log matches were all `0`.

The topology reached its exact prediction of `10 compute + 0 transfer` per
recurrent layer, but the physical result missed its predeclared stop
condition. Across 75 decode correlations and 2,250 recurrent observations,
mean replay was `6.221576 ms`, which is `0.025885 ms` or `0.4178%` slower than
the `67921b1c` packed candidate's `6.195692 ms`. The additional normalization
dispatch cost more than the removed per-value-tile normalization saved.
Profile-off was intentionally not run after the structural miss.

The immutable decision is:

```text
CUDA NORMALIZED PACKED DECODE CANDIDATE REJECT: diagnostic-summary.json
```

Commit `784d5bf2` reverted the candidate; the active source contains neither
rejected packed-decode implementation. The GitHub-transfer artifact is
[runtime-vnext-normalized-packed-decode-cuda-a884f5d4-20260724T153606Z-sanitized.tar.zst](https://github.com/sizzlecar/ferrum-infer-rs/releases/download/untagged-711d3e8abdfcbe0c8b41/runtime-vnext-normalized-packed-decode-cuda-a884f5d4-20260724T153606Z-sanitized.tar.zst),
asset id `488576890`, size `27,926,534` bytes, SHA256
`38e715a12017b0771f8da38b2c80b7b4fbdab358c23a67f61ccba3ad77e67080`.
It was downloaded and revalidated locally under
`/Users/chejinxuan/ferrum-artifacts/runtime-vnext-normalized-packed-decode-cuda-a884f5d4-20260724T153606Z/`.
Vast instance `45319871` is confirmed `stopped/exited`, with no paid sibling.

Neither `67921b1c` nor `a884f5d4` may repeat the old unpaired benchmark
protocol. Because the profiling-path audit below invalidates the historical
mean comparison used to reject `67921b1c`, one final bounded re-attribution of
the already-built baseline and `67921b1c` binaries is allowed: correctness
first, direct-path basic `A-B-B-A`, then profile-off `A-B-B-A` only if basic
follows the binary in both reversed pairs. Any miss permanently rejects that
candidate without another build or full sweep.

If `67921b1c` is permanently rejected, a future GDN candidate must avoid both
known failure modes: value-tile repeated Q/K normalization and a separate
normalization launch. It must retain at most `9 compute + 0 transfer`, pass
local numeric/ABI/graph contracts, predict replay below `6.195692 ms`, and
then prove a same-session direct-path improvement rather than compare against
a historical product mean.

### 2026-07-24 vLLM Source Comparison And Measurement Amendment

The current source comparison uses local vLLM commit
`426d48bfa149582664d48f89df21ec9beae5c37b`. For Qwen3.5 non-speculative
decode, vLLM runs `causal_conv1d_update` and then
`fused_recurrent_gated_delta_rule_packed_decode`
(`qwen_gdn_linear_attn.py:1206-1218,1564-1615`). Its Triton program uses grid
`(NV, B * HV)` and reloads and normalizes Q/K inside every value tile
(`third_party/flash_linear_attention/ops/fused_recurrent.py:282-315,448-475`).
This deliberately accepts repeated normalization to avoid another launch.
Accordingly, `67921b1c` was structurally closest to vLLM, while `a884f5d4`
tested a Rust/CUDA-specific alternative and proved that a separate launch is
worse. Ferrum must use this evidence to improve the trade-off, not copy either
implementation blindly.

The same review also corrected a profiling assumption. Current Ferrum maps
`off -> DeviceTimingMode::Off`, `basic/debug -> Completion`, and
`full -> Kernel`. The typed direct-program path is selected only outside
`Kernel`; full kernel attribution explicitly falls back to complete logical
provider encoding. Therefore:

- `full` is authoritative for topology, graph-node composition, and device
  replay deltas only when its reusable executable/graph fingerprints match the
  paired direct path;
- `full` host-stage timings and throughput are not product-path evidence;
- `basic` is the bounded direct-path attribution lane, with fallback and epoch
  miss required to remain `0`;
- candidate triage uses same-instance, same-cache `A-B-B-A`; each reversed
  adjacent pair must move in the predicted direction. Only after it passes may
  profile-off `A-B-B-A` run;
- formal G09 evidence remains the complete `ABBA-BAAB`, the unchanged
  `76.1583 tok/s` floor, and the external/legacy ratio gates above.

For a device candidate, structural KEEP alone is insufficient. It must also
improve direct-path completion/device time or paired profile-off throughput.
Comparing a 25-request candidate mean against a stale 300-request historical
mean is no longer an acceptance method. The G06 execution-path fingerprint
contract is a dependency of the next paid GDN run.

### M3 Qwen3-30B historical floors

保留两套独立 random `256/128` 向量：

| source | c1 | c4 | c16 | c32 |
|---|---:|---:|---:|---:|
| 0.7.7 default | 164.2 | 353.3 | 636.9 | 706.0 |
| historical FA2 direct | 160.4 | 446.3 | 1185.1 | 1641.9 |

G00 必须为每个向量记录 source artifact、SHA、binary、feature、preset、命令和是否为 typed
产品配置。只有同模型 revision、workload、硬件和产品可见配置可比时，才能把逐 cell 最大
有效 LCB 作为 G09 floor。hidden-env diagnostic 不能自动升级为 release floor；不可复现时
必须产生 REJECT artifact 和正式 amendment。

### M1 Qwen3.5-4B

没有可信 legacy performance artifact。G00 fresh legacy 能运行时同时执行 no-regression；
无论 legacy 是否可用，vNext 必须达到 same-host vLLM LCB `>=0.90`。不能以缺 baseline 为由
只做功能 smoke。TTFT/TPOT p95 还必须 `<=1.15x vLLM`；全 paired request eligible 时
client-SSE-event ITL 也执行该门；`ferrum run` 稳态 decode
必须 `>=0.95x` 同 binary、同 prompt/length 的 Ferrum serve-c1 decode。

## Metal 目标

固定 32GB / 24-GPU-core M1 Max，同一 GGUF、同一 llama.cpp revision/config，c=1/4/16：

- typed active admission 硬下限为 M1/M2/M3=`16/4/16`；c16 的 observed max-active 必须达到
  对应下限，eligible interval active duty-cycle `>=0.80`，并保存完整 active timeline。若物理
  preflight 不能满足，按 MODEL_MATRIX hardware amendment policy 更换明确硬件/format 并重采，不能
  临时降低 floor；typed cap 等于 floor 时 `observed_max_active == typed_admission_cap`，cap 高于
  floor 时 observed max-active 至少达到 floor；

- 三模型 c1/c4/c16 throughput ratio LCB 全部 `>=0.90x llama.cpp`，所有 required cells 的
  ratio 几何平均 LCB `>=0.95`；external baseline 失败时必须修复 baseline，不能 waiver
  Ferrum cell。
- TTFT/TPOT p95 `<=1.15x llama.cpp`；全 paired request eligible 时 client-SSE-event ITL 也执行该门。
- 仅 G00 已有 legacy PASS 的 lane 同时满足同 workload no-regression；M1/M2 Metal 不得套用
  不存在的 legacy ratio。
- M3 旧 `16 input / 64 output`、c16 约 `68.5 tok/s` 只作 sanity reference；本 Goal 的
  `64/128` workload 不得直接套用该绝对值，正式 floor 由 G00 同 workload fresh baseline 决定。
- measured cell swap growth `0`，thermal throttling artifact `0`，peak unified memory 不越过
  preflight budget。
- 每个 Metal 模型同 binary、同 prompt/length 的 G06 token-commit `ferrum run` 稳态 decode /
  Ferrum serve-c1 decode median `>=0.95`；M3 等 G00 legacy PASS lane 的 run 另以匹配的
  `engine.infer` E2E 边界满足 legacy no-regression。

## Profiling 完整性

- decode wall time stage coverage `>=90%`。
- top op/kernel coverage `>=80%` device time。
- basic profiling overhead `<=2%`，正式 benchmark profile=off/basic validated mode。
- 每个未达标 cell 自动输出 top 5 bottlenecks、target gap 和 source mapping。
- performance optimization 不得引入 hidden env-only product behavior。

## 停止规则

- correctness failure：立即停止该 model/backend 全部性能工作。
- 两个同类 REJECT：停止 paid run，回到 source/replay。
- absolute floor 仍差 >20% 且 profile signal未变化：禁止再跑 full sweep。
- 每次 paid failure copy back artifact 后 stop/retention decision，不能只报告“跑了多久”。

## Release-candidate 刷新

G09 的开发 PASS 不能跨过 version/release-workflow freeze 自动成为发布证据。G10A 产生唯一
`release_candidate_sha` 后，必须执行：

```text
python3 scripts/release/run_gate.py vnext-g09-rc \
  --g10a <g10a-manifest> \
  --g08-rc <g08-rc-manifest> \
  --g00 <g00-manifest> \
  --out <external-out>
```

G09-RC 重跑本文件全部 required comparison，不复用 G09-dev 的 candidate rows 或 CI；冻结
G00 legacy binary 与 external binary bits 可以复用缓存，但 A server 仍须在本次 ABBA outer slots
实际运行。manifest 必须满足 `source_git_sha == release_candidate_sha`、`dirty=false`，并保存
Metal/CUDA candidate binary SHA。G10A staged tarball 中对应 binary SHA 必须与 G09-RC 完全一致；
否则 G09-RC stale 并完整重跑。必需 line：

```text
FERRUM RUNTIME VNEXT G09 RELEASE CANDIDATE PERFORMANCE PASS: <out_dir>
FERRUM GATE vnext-g09-rc PASS: <out_dir>
```

## 产物与 PASS

```text
docs/release/runtime-vnext/0.8.0/g09-performance/
  cuda/<model>/
  metal/<model>/
  external/<engine>/<model>/
  comparisons/
  profiles/
  candidate-ledger.jsonl
  final-matrix.json
  release-candidate/
    manifest.json
    g10a-binding.json
    binary-bindings.json
```

```text
FERRUM RUNTIME VNEXT G09 PERFORMANCE PASS: <out_dir>
FERRUM GATE vnext-g09 PASS: <out_dir>
```
