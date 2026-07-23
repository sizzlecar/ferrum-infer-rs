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
diagnostic 阻塞于缓存 immutable singleton batch membership、借用 spans 的本地验证，
源码预测为 `resource_prepare_attempt <=3.45 ms`，不得用另一轮 GPU sweep 代替该验证。

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
