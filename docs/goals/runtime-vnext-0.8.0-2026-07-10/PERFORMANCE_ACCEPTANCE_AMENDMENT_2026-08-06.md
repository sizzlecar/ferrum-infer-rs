# Runtime vNext v0.8.0 性能验收收敛修订（2026-08-06）

## 状态与效力

- 状态：Active。
- 本修订响应 owner 对 v0.8.0 性能范围的明确调整，优先于
  [`GOAL.md`](GOAL.md)、[`MODEL_MATRIX.md`](MODEL_MATRIX.md)、
  [`G00_BASELINE.md`](G00_BASELINE.md)、[`G09_PERFORMANCE.md`](G09_PERFORMANCE.md)、
  [`G10_RELEASE.md`](G10_RELEASE.md) 和
  [`RELEASE_ACCELERATION_AMENDMENT_2026-08-06.md`](RELEASE_ACCELERATION_AMENDMENT_2026-08-06.md)
  中与外部性能比较、ABBA 执行次数和重复发布性能回归冲突的条款。
- 本修订只降低 v0.8.0 的性能竞争性证明范围，不降低三主模型 CUDA/Metal 正确性、产品 API、
  动态资源、legacy-zero、profile 可定位性、编译阈值或最终资产验证要求。
- vLLM 和 llama.cpp 仍可作为非阻断诊断工具，但不再是 v0.8.0 performance PASS 的 comparator，
  不要求下载、启动、同机 ABBA、ratio 或 CI。
- 外部竞争性性能门转入 v0.8.1/0.9，不得误报为 v0.8.0 已证明与 vLLM/llama.cpp 持平。

## 目标

v0.8.0 的性能结论改为：Ferrum 在固定产品配置、固定模型和固定硬件上达到可用的绝对底线，
且最终候选不显著回退于自身已冻结的可信基线。它不再声称与外部引擎有确定比例关系。

性能证据仍必须晚于对应 correctness PASS。正确性失败、错误输出、隐藏配置、legacy selection、
OOM 或资源不变量失败时，性能数据只能作为诊断，不能通过本门。

## 固定矩阵

### CUDA

- 硬件：exactly one RTX 4090。
- 三主模型：M1 Qwen3.5-4B、M2 Qwen3.5-35B-A3B-GPTQ、M3 Qwen3-30B-A3B-GPTQ。
- 主矩阵：random `256 input / 128 output`，c=`1/4/16/32`，每 cell `100 requests x 3 repeats`。
- 真实 workload sentinel：frozen ShareGPT，c=`1` 和最高有效并发，每 cell
  `30 requests x 3 repeats`。
- `ferrum run`：固定 prompt/output policy，三个独立进程，采集 steady decode 和完整
  `engine.infer` E2E。

### Metal

- 硬件：固定 `32GB / 24-GPU-core Apple M1 Max`；只有经 Goal amendment 明确替换后才能更改。
- 同三个主模型和锁定 GGUF。
- 主矩阵：random `64 input / 128 output`，c=`1/4/16`，每 cell
  `100 requests x 3 repeats`。
- 真实 workload sentinel：frozen real-chat，c=`1` 和最高有效并发，每 cell
  `30 requests x 3 repeats`。
- `ferrum run` 与 CUDA 使用相同的三进程统计边界。

全部 HTTP 测量继续使用唯一产品客户端：

```text
ferrum bench-serve ... \
  --fail-on-error \
  --require-ci \
  --seed 9271 \
  --n-repeats 3
```

## Ferrum 自身基线

每个 `(model, backend, dataset, concurrency, metric)` 只能绑定一个 checked-in floor row。
row 必须包含 source git SHA、dirty status、binary SHA256、model SHA256、硬件、driver/OS、完整命令、
typed config、artifact 路径、三个 raw repeats 和冻结时间。

基线选择规则：

1. 只接受 clean source、真实产品入口、对应 correctness 已通过、profile-off、无 hidden env 的 Ferrum
   artifact。
2. model SHA、量化/精度、prompt/output 长度、dataset、并发、active cap 和硬件必须与 release cell
   完全一致；跨硬件或不同 workload 数字不得换算。
3. 存在多个合格 artifact 时，选择吞吐中位数最高且延迟字段完整的一份；不能为降低 floor 选择更慢
   artifact。
4. 没有同口径历史 artifact 的新 cell，在 R1 correctness PASS 后采一次 clean calibration 并立即冻结。
   calibration 必须先满足下文绝对可用性、并发和稳定性门，不能只因“当前能跑”自动成为基线。
5. floor catalog 必须在 R3 staged binary 测量开始前提交。测量后不得修改 row；任何修改都需要单独
   reviewed Goal amendment，并使受影响 performance artifact stale。
6. 当前 release binary 的生产运行时 legacy selection 仍必须为 `0`。基线数字可以来自旧 Ferrum
   release artifact，但 R2/R3 不要求重新启动 legacy binary，也不允许由此恢复 legacy 路径。

## 量化硬门

### 绝对可用性

对没有历史同口径基线的 calibration，`ferrum run` c1 steady decode 至少达到：

| Backend | M1 Qwen3.5-4B | M2 Qwen3.5-35B-A3B | M3 Qwen3-30B-A3B |
|---|---:|---:|---:|
| CUDA RTX 4090 | `>=50 tok/s` | `>=50 tok/s` | `>=100 tok/s` |
| Metal M1 Max | `>=20 tok/s` | `>=5 tok/s` | `>=5 tok/s` |

这些是 v0.8.0 的最低可用线，不是竞争性性能声明。已有更高的合格自身基线时，必须执行更严格的
自身非回退门，不能退回本表。

### 自身非回退

- 每个 required cell 的 candidate throughput median `>=0.95 x frozen Ferrum baseline`。
- 每个 model/backend 的 required-cell throughput ratio 几何平均 `>=1.00`。
- TTFT p95 和 TPOT p95 各自 `<=1.10 x frozen Ferrum baseline`。
- `ferrum run` steady decode median `>=0.90 x` 同 binary、同 prompt/output 的
  `ferrum serve` c1 steady decode。
- peak accelerator/unified memory `<=1.05 x` 自身基线，且不得超过 typed preflight budget。

### 并发与动态资源

- CUDA active admission floor 保持 M1/M2/M3=`32/16/32`；Metal 保持 `16/4/16`。
- 最高 client concurrency 的 observed max active 必须达到对应 floor；eligible interval 内 active
  sequences 达到 floor 的 wall-time fraction `>=0.80`。
- 最高有效并发 throughput 至少为 c1 的 `1.25x`（CUDA）或 `1.10x`（Metal）。如果硬件 preflight
  只允许更低 active cap，必须走 Goal amendment，不能静默降低。
- defer/wait/resume、cancel/release 和 active decode progress 仍是 correctness 硬门；不得通过拒绝、
  串行化或缩短输出制造吞吐 PASS。

### 稳定性与完整性

- 每个 required cell 三次 throughput repeat 的 CV `<=8%`。
- measured 和 warmup completion `100%`；request error、bad output、malformed SSE、missing/duplicate
  `[DONE]`、zero output、panic、OOM、admission failure、resource leak 均为 `0`。
- output token source 为 usage 的请求比例 `100%`；输入长度必须来自实际 tokenizer count。
- Metal measured interval swap growth `0`，thermal throttling `0`；CUDA 保留至少 `512 MiB`
  physical VRAM headroom。
- TTFT、TPOT、throughput、active timeline、memory 和 raw request 字段完整率 `100%`。

### Profile 与编译

- profile-off 仍是产品性能真值；basic profile overhead `<=7%`。
- basic/replay/full 仍须把实际瓶颈关联到 plan/node/op/resource/provider/kernel；不得因取消外部比较
  删除 profile 可定位性验收。
- G07 的 no-op、Rust leaf、PTX、native TU 和 clean release build 阈值不变。

## R2 与 R3 执行次数

1. R2 在 R1 correctness PASS 后，对 development candidate 执行一次上述 Ferrum-only 矩阵。
2. R3 对 exact staged Metal/CUDA binary 各执行一次相同矩阵；这是最终三主模型性能 release evidence。
3. staged tarball 与 published asset 的 tarball SHA256、binary SHA256 完全相同时，发布后只重跑安装、
   version/dependency 和三主模型 `run`/`serve` correctness smoke，不第三次执行完整性能矩阵。
4. 任一 published byte 与 staged byte 不同，R3 performance 全部 stale，必须对实际发布 binary 重跑。
5. 单 cell 失败先跑 exact reproducer 和 affected cell；focused 通过前禁止重跑完整矩阵。

## 外部引擎的地位

- vLLM/llama.cpp 可以在 profile 已把问题收敛到算法/内核差异时运行一次 bounded diagnostic。
- 外部数字只标记 `diagnostic`, `KEEP` 或 `REJECT`，不得阻塞 v0.8.0 R2/R3。
- v0.8.0 release notes 必须准确写明性能报告是 Ferrum 自身基线回归，不声称达到 vLLM/llama.cpp
  的任何比例。
- same-host external ABBA、竞争 ratio、capacity-pressure competitive lane 和外部 CI 转入
  v0.8.1/0.9 performance hardening。

## 必需 PASS

R2 和 R3 validator 必须检查原始 benchmark、floor catalog 和本文件所有阈值，并分别打印：

```text
FERRUM RUNTIME VNEXT R2 PERFORMANCE BUILD PROFILE PASS: <out_dir>
FERRUM RUNTIME VNEXT R3 V0.8.0 PUBLISHED PASS: <out_dir>
```

在出现这些行和对应 artifact 目录前，不得声称 v0.8.0 performance-ready 或 release-ready。
