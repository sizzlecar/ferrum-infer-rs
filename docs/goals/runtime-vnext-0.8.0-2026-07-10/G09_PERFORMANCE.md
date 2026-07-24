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
