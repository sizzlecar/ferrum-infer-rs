# P0：干净候选与首轮实测

状态：候选 `c068e3d7` 保留轻量 SLO-aware，补上负 decode 成本的约束拟合；workspace 格式、全目标编译/测试（4626 通过、0 失败、73 ignored）、Clippy、Metal 全目标编译及 CUDA 三 feature 编译均通过，两端正式实机 E2E 正在验证。CUDA Q6 恢复版静态 C1 已恢复吞吐；a890 主负载静态/动态对比未见 G 收益，M4 相同主负载 C1 仍因 TTFT 未达标。版本及适用范围见下文，尚未发布。目标与口径见 [目标文档](goal-slo-throughput.zh.md)。

候选 `57ee5ccb` 基于 main `2c9998a2`，仅接入既有静态执行优化 `c3c9ee0b` 与独立客户端测量；没有服务 SLO 控制器、启动校准或成本模型。旧恢复分支已作本地归档 tag `archive/slo-recovery-20261006`。原脏工作树未修改。

候选已有 workspace 格式、编译、测试（4591 通过、72 项现有 ignored）、Clippy、Metal 全目标编译和 CUDA CLI 三 feature 编译通过。RTX 5090 / Qwen3.5-9B Q4_K_M 已跑通自然 `run`、非流式 `serve` 和现有 mixed 调度断流恢复 E2E（5.43s）：取消一个请求后，同批请求和新 prefill 正常完成，资源排空，随后三个并发请求成功。自然问答解释 KV cache，21 输入 / 32 输出 token，正常 stop。

M4 / 同模型也通过自然 `run`、非流式及 SSE 流式请求，以及相同断流恢复 E2E（14.61s）。三种自然问答均完整回答三段，91 输入 / 215 输出 token，正常 stop；流式 terminal / usage 齐全。本次 M4 服务已关闭，端口释放。功能 E2E 采用 mixed、active-decode chunk 128 / budget 256、context 4096 / slots 4，与性能表固定容量配置分别记录。

本机 M1 Max 使用已有 Qwen3.5-4B Q4_K_M 也通过上述功能及流式自然结束检查。它是附加功能证据，不替代目标 M4 的性能或 E2E。

新增 `bench-serve --latency-slo ttft:200,tpot:15,itl:50 --latency-slo-fail-on-violation` 独立评估每次重复的三项 P99、错误和拒绝。真实服务上已验证 JSON/Markdown 报告及退出码：宽松阈值通过，过紧阈值先写失败报告再返回非零。这个小样本合成 E2E 只验证接口，不作为性能证据；主性能负载仍为固定 ShareGPT。

动态候选新增 `run` / `serve --scheduler-slo ttft:200,tpot:15,itl:50`，按实际执行反馈调整 prefill 总预算、优先处理 TTFT 临期请求，未配置时保持静态路径。本机 M1 Max debug / 4B 功能 E2E 已通过：反馈波 15→156、实际动态 prefill 步 0→59、预算更新 1→42；自然 `run`、流式/非流式均正常结束，断流后 peer / 新 prefill / 后续请求完成，资源排空。此结果只验证功能。

`bc85eec2` 在 RTX 5090 和 M4 / Qwen3.5-9B Q4_K_M 也通过动态断流 E2E（8.22s / 23.03s），实际动态 prefill 步数分别为 151 / 106；自然输出及资源排空正常。但以下短负载揭示了功能测试不能发现的性能问题。

## 动态调度短负载诊断

固定同机同模型、context 4096、slots 4、batch / chunk / aggregate prefill budget 均为 64、mixed、FP16 KV。两个 32 输入 / 512 输出请求启动后插入 256 输入 / 1 输出请求；预热一次、测量一次，每格 3 个测量请求，seed 42。唯一配置变量为是否开启 `--scheduler-slo`。这是合成干扰诊断，所列 P95 是干扰期间可见 SSE 文本事件间隔，**不是主负载 P99 SLO 验收**。

| 后端 / 模式 | 新请求 TTFT ms | 干扰间隔 P50/P95 ms | 最大间隔 ms | 输出 tok/s | 内存峰值 |
| --- | ---: | ---: | ---: | ---: | --- |
| CUDA / 静态 | 736.38 | 111.51 / 152.06 | 152.07 | 62.52 | NVML 7003 MiB；RSS 6069628 KiB |
| CUDA / bc85 动态 | 10108.79 | 37.28 / 38.27 | 117.28 | 54.84 | NVML 6970 MiB；RSS 6069364 KiB |
| CUDA / 739f 静态 | 733.36 | 110.74 / 153.23 | 153.25 | 62.49 | NVML 7002 MiB；RSS 6069796 KiB |
| CUDA / 739f 动态 | 2400.78 | 43.13 / 44.73 | 47.13 | 61.42 | NVML 6973 MiB；RSS 6069348 KiB |
| M4 / bc85 静态 | 1776.10 | 283.80 / 362.68 | 362.68 | 30.14 | RSS 283213824 B；footprint 887081168 B |
| M4 / bc85 动态 | 10696.93 | 187.03 / 199.62 | 227.46 | 25.58 | RSS 290095104 B；footprint 883214544 B |
| M4 / 739f 动态 | 3542.06 | 200.79 / 202.81 | 203.59 | 29.00 | RSS 260767744 B；footprint 856836400 B |
| M4 / 0a7 仿射动态 | 3835.67 | 200.68 / 204.34 | 206.19 | 29.04 | RSS 256180224 B；footprint 855689496 B |
| M4 / fe46 仿射动态＋TTFT 回退 | 2954.04 | 204.52 / 283.77 | 365.35 | 29.47 | RSS 259948544 B；footprint 966150376 B |

已完成各格均成功、错误/拒绝 0，未观察到 coalescing；M4 本轮未采集设备分配峰值，各内存口径不相加。首版缩短了输出停顿，却将新请求 TTFT 拖到约 10 秒并降低吞吐，因此判为失败。修正版区分累计 TPOT 余量与单次 ITL、只从纯 prefill 学习单位成本，并对纯 decode 已不可行的目标降级；复测沿用同一负载与原阈值。CUDA TTFT 仍为静态的 3.27 倍，M4 为 1.99 倍且超出原 3400ms 目标，不能因 ITL 改善就称 SLO 优化成功。两端均已收尾；CUDA 原服务已恢复且健康，M4 端口已释放。两轮失败数据均保留，不继续修补此单位成本算法。

`739f9653` 已通过格式检查、workspace 全目标编译/测试（4600 通过、0 失败、73 ignored）、Clippy、Metal 全目标编译及 CUDA CLI 三 feature 编译。两端自然 `run` 正常完成；动态断流 E2E 分别通过 7.19s / 22.16s，验证实际预算使用、peer / 新请求 / 后续请求完成及资源排空。这些是功能结果，不代表性能达标。

`0a7c0cfc` 使用原 P3 的 `a + b·decode + c·prefill` 模型；M4 自然 `run` 和动态断流 E2E（14.82s）通过，预算确实使用，但相同诊断 TTFT 仍为静态的 2.16 倍。拟合过程中存在不可识别后回静态的情况，未隐藏。后续只补预算对活跃 prefill 剩余 TTFT 的必要可行性判断，不更换模型。

M4 该轮短负载运行源码为 `fe46d6b5`，保留 `bc85eec2` 的既有静态结果作对照；沿用上述容量、负载、阈值和一次重复 / 3 个测量请求。自然 `run` 与动态断流 E2E（14.80s）通过；诊断实际动态 prefill 12 步，TTFT 可行性回退 1 次。新请求 TTFT 从上一版 3835.67ms 降至 2954.04ms，但仍比静态慢 66%，吞吐低 2.24%；最大可见间隔 365.35ms 也不能作为 359ms P99 达标证据。设备分配峰值未采集，主机内存口径不相加。这是功能通过及短负载取舍，**不是主负载 SLO 或 G 改善证明**。服务已清理，端口释放；[结果摘要](/Users/chejinxuan/ferrum-handoffs/20261006-throughput-p0/metal/slo-r4-summary.json)与[原始证据](/Users/chejinxuan/ferrum-handoffs/20261006-throughput-p0/metal/slo-r4-evidence.tar.gz)已取回，早期失败结果保留。

`0d0e4788` 前向移植已测 RN-F16 快路径，显式选择 `--numerical-profile qwen3_5.f32-master.gguf-f16-projections.ffn-rn-fragment-m1to8`，Auto 保持原行为。未带回旧 SLO / 校准 / 成本观测接口；这是保留 P1 执行基线，仍有常驻 FP16 与双物理布局的显存代价，不代表 P2 的低显存量化内核目标完成。工作区编译、测试（4621 通过、0 失败、73 ignored）、Clippy 及 CUDA 三 feature 编译已通过；实机验证进行中。

RN 实机启动先后暴露缺失的 KV 操作别名和静态权重归属支持，分别在 `2c4237d8` / `504d290d` 补齐既有直接依赖，未绕过检查；两次失败日志保留，原服务均已恢复。`a890afc9` 将仿射统计衰减从 0.8 调整为 0.99，延长稀少 prefill 观测的保留时间。停止追加合成负载调参，主负载改为同一 RN/mixed 二进制与固定容量的静态/动态 C1→C2→C4→C8 对比；本轮只改变 SLO 开关，完整结果见下表。

`504d290d` 的 CUDA RN 自然 `run` 与动态断流 E2E 已通过：21/32 输入/输出 token 正常 stop；E2E 实际动态 prefill 步 0→1，资源排空，原服务恢复且健康。最终归属补丁通过 workspace 全目标检查/测试（4625 通过、0 失败、73 ignored）、Clippy 和 Metal 编译；0.99 的既有 SLO 定向测试也通过。但 `a890afc9` 的 M4 自然 `run` 通过后，同一动态 E2E 因实际动态 prefill 步仍为 0 而失败（反馈波 142、预算更新 2）。请求已完成并排空，现有起止快照不足以定位何时形成可用模型；保留失败，不用旧版通过结果替代。证据为仓库外 `cuda/rn-affine-r3/functional-r1/summary.json` 与 `metal/slo-r5-summary.json`。

M4 定位重跑 r6 仍失败：长 prefill 到达前只有 4 个反馈波，仿射系数为空；原请求完成后系数有效，但实际动态步仍为 0。`ff47c0a3` 保留原冷启动长请求、取消与后续请求，并在模型形成后追加较小 prefill，单独要求该阶段实际采用预算。相同 `a890afc9` 产品通过修订 E2E（17.65s）：冷长请求阶段动态步为 0，最终累计为 7，新增已学习阶段的独立增长断言通过；7 个请求完成、失败 0、资源排空。r5/r6 失败原样保留；这是验证前提的修正，不是控制器变更或性能收益证据。仓库外摘要为 `metal/slo-r6-summary.json`、`metal/slo-r7-summary.json`。

## 首轮 CUDA 5090 数据

以下均为单次探索：32 预热 + 64 测量请求、C1。Qwen3.5-9B Q4_K_M，FP16 KV，context 2048、服务 slots 32、batch 2048；容量在扫描中固定。ShareGPT seed 42，首轮用户/参考答案长度回放，输入 4–1024、最少输出 4、总长含 32-token 模板预留 ≤2048，ignore_eos、thinking off、temperature 0、top_p 1、repetition penalty 1。两臂相同完整选择 hash：`03676b00ee0b9e8407f59e3fb531eeff99dde5ccc9e59844b6abf7d22f1d5df2`。

| 实现 | C | TTFT P50/P99 ms | last-visible TPOT P50/P99 ms/token | 可见 SSE 文本事件 ITL P50/P99 ms | 输出 tok/s | GPU allocated 峰值 | NVML 整卡峰值 MiB | OS footprint 峰值 | RSS 峰值 KiB | 本次 SLO |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- | ---: | --- |
| G32 Off，RN-F16 fragment profile | 1 | 19.75 / 104.60 | 9.60 / 10.04 | 9.73 / 10.58 | 102.08 | 未采集 | 22141 | 未采集 | 6157288 | 三项通过 |
| llama.cpp b11065 | 1 | 28.70 / 107.57 | 5.99 / 7.01 | 5.85 / 7.00 | 164.97 | 未采集 | 10187 | 未采集 | 5931552 | 三项通过 |
| 静态基线 57ee5ccb，Auto f32-master | 1 | 104.98 / 1758.71 | 18.83 / 19.15 | 18.94 / 19.92 | 50.83 | 未采集 | 7418 | 未采集 | 6067220 | TTFT、TPOT 超标 |

阈值为 P99 200/15/50ms；G32 Off 和 llama.cpp 均 64/64 完成、错误/拒绝 0。这两行是 C1 的达标点，不是容量边界或重复验收。G32 Off 记录 19742 个非空文本事件、19678 个间隔；8 个 event/usage 不一致保留，未观察到 transport coalescing。协议/完整性检查未发现异常，模型语义另由自然问答 E2E 检查。内存口径相互重叠，不相加。

静态基线使用同一选择 hash，同样 64/64 完成、错误/拒绝 0，19744 个文本事件、19680 个间隔，8 个 event/usage 不一致，未观察到 coalescing。这条路径没有移植 G32 的常驻 FP16 投影 / RN 特化，因此不能用其内存下降掩盖吞吐回退；纯 decode 已超过 TPOT 目标，动态调度不能单独解决这个单请求性能缺口。当前基线扫描未找到可行档。

## a890 CUDA RN 主负载静态 / 动态对比

两种模式均为 `a890afc9`、同一 RN-F16 profile、mixed，唯一变量为 `--scheduler-slo ttft:200,tpot:15,itl:50` 开关；P99 阈值 200/15/50ms。沿用上述 ShareGPT 选择 hash、输出长度政策和客户端；固定 context 2048 / slots 32 / batch 2048 / 24GiB runtime budget、FP16 KV、prefix/session cache off。每格 32 预热、64 测量请求、1 次重复，均完成 64/64、错误/拒绝 0；TTFT/TPOT 各 64 个样本。

| 模式 | C | TTFT P50/P99 ms | last-visible TPOT P50/P99 ms/token | 可见 SSE 文本事件 ITL P50/P99 ms | 输出 tok/s | 文本事件 / 间隔数 | 本次 SLO |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 静态 | 1 | 20.68 / 104.71 | 10.69 / 11.11 | 10.82 / 11.64 | 91.989 | 19742 / 19678 | 三项通过 |
| 静态 | 2 | 34.36 / 116.59 | 13.48 / 14.13 | 13.59 / 14.89 | 143.848 | 19742 / 19678 | 三项通过 |
| 静态 | 4 | 49.94 / 131.03 | 18.89 / 19.72 | 18.79 / 28.41 | 202.107 | 19740 / 19676 | TPOT 超标 |
| 静态 | 8 | 84.48 / 180.48 | 30.22 / 32.54 | 29.69 / 49.44 | 247.060 | 19747 / 19683 | TPOT 超标 |
| 动态 | 1 | 20.51 / 105.96 | 10.70 / 11.11 | 10.82 / 11.63 | 91.995 | 19742 / 19678 | 三项通过 |
| 动态 | 2 | 51.23 / 132.70 | 13.51 / 14.17 | 13.61 / 15.06 | 143.234 | 19740 / 19676 | 三项通过 |
| 动态 | 4 | 49.92 / 130.32 | 18.93 / 19.93 | 18.82 / 28.73 | 201.833 | 19740 / 19676 | TPOT 超标 |
| 动态 | 8 | 84.79 / 179.08 | 30.47 / 32.49 | 29.79 / 50.47 | 246.689 | 19747 / 19683 | TPOT、ITL 超标 |

内存为**每种模式整个服务生命周期、跨四格的峰值**，不能归到单格：静态 NVML 21340 MiB / RSS 6072812 KiB，动态 21232 MiB / 6072648 KiB；NVML 分别 3430 / 3437 个采样。GPU allocated 与 OS footprint 未采集，各口径不相加。各格均保留 8 个 event/usage 不一致，无缺失 usage 或观测到的 coalescing，协议检查错误为 0。

本轮已扫描达标格中的 G 为静态 **143.848 tok/s@C2**、动态 **143.234 tok/s@C2**，动态低 **0.43%**，未见收益；一次重复不足以声称最佳静态或完成主负载验收。新静态 C1 仍比旧 RN G32 的 102.079 tok/s 低 9.9%；新 mixed / 旧 split 的配置差异保留，不能把这两版比较当成单独的 SLO 开关实验。

两种模式的 C4 均有 63/64 请求 TPOT 超过 15ms，C8 均为 64/64；可见间隔中位数分别约 18.8ms、29.7ms。超过 50ms 的长间隔仅占跨请求累计间隔时长的约 0.9%（C4）及 2.1%–2.5%（C8）；这是间隔时长之和，非并发测量墙钟。数据更支持持续输出节奏偏慢，单独减少少数长停顿不足以解决 TPOT 超标。记录缺少逐波工作标签与设备计时，尚不能精确分解 GPU decode、prefill 和主机开销。全部可见间隔仍计入正式 SLO。

动态整趟实际采用 prefill 预算 85 步、预算更新 254 次、TTFT 可行性回退 13 次、纯 decode 超载 10498 步；这是机制运行证据，不改变各格 SLO 结论。两模式最终请求排空，原服务已恢复。完整命令、配置与原始数据见仓库外 [结果摘要](/Users/chejinxuan/ferrum-handoffs/20261006-throughput-p0/cuda/rn-primary-a890-r1/summary.json)。

## Q6 CUDA 路径恢复

`07523c72` 恢复现有 Q6_K F32 单行及 tiled 投影内核，共 55 行；实际模型 output.weight 为 Q6_K。未改 RN profile 或调度策略。CUDA 三 feature 编译、既有真实 GPU 混合矩阵数值测试和自然 `run` 通过；`run` 为 21/32 输入/输出 token、正常 stop，文本与前版一致。此版通过 workspace 格式、全目标编译/测试（4625 通过、0 失败、73 ignored）、Clippy 及 Metal 编译。

静态 C1 使用与 a890 相同的 RN/mixed、服务容量、24GiB runtime budget、客户端及 ShareGPT 选择 hash；32 预热 + 64 测量、1 次重复，64/64 完成，错误/拒绝 0：

| 版本 | TTFT P50/P99 ms | last-visible TPOT P50/P99 ms/token | 可见 SSE 文本事件 ITL P50/P99 ms | 输出 tok/s | NVML 整卡峰值 MiB | RSS 峰值 KiB | 本次 SLO |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 07523 Q6 恢复 | 19.52 / 104.35 | 9.66 / 10.10 | 9.80 / 10.66 | 101.438 | 20219 | 6074464 | 三项通过 |

相比 a890 静态 C1 提升 **10.27%**，与旧 G32 C1 参考相差 **−0.63%**；这验证了该内核恢复的收益，不代表已重测本版的并发容量 G。TTFT/TPOT 各 64 个样本、文本事件 19742 / 间隔 19678，保留 8 个 event/usage 不一致，无 coalescing、坏输出或协议错误。GPU allocated 与 OS footprint 未采集，各内存口径不相加；原服务已恢复。证据见仓库外 `cuda/rn-q6-07523-r1/c1-summary.json`。

同版 `ff47c0a3` 动态 E2E 失败：原冷请求、取消与三个后续请求正常完成并排空，但反馈 140 波后模型系数仍为空、动态步数 0，在追加已学习阶段前触发断言。失败原样保留于 `cuda/rn-q6-07523-r1/lifecycle-failure.json`，不能用数值测试、自然输出或静态 C1 通过替代动态功能验证。当前仅对同一负载做一次拟合统计诊断，未改变策略或阈值。

单次临时诊断在同一负载再次失败，最终 137 个反馈波。统计矩阵满秩（相对行列式 0.9844），无约束解为 `a=13.3563ms、b=−1.4266ms/请求、c=0.231515ms/token`，明确由负 `b` 触发整体失效。修正只对 `b ≥ 0` 做闭式边界重拟合，得到 `a=11.4729、b=0、c=0.236625`；其余非法条件继续回静态。临时诊断的打印不进入产品，也不用于性能结论；原始统计见 `cuda/rn-q6-diagnostic-r1/summary.json`。修正后的正式实机 E2E 待验证。

## 首轮 M4 数据

沿用同一 ShareGPT 选择规则及 32 预热 + 64 测量请求、一次重复；context 4096、slots 32、batch 2048、FP16 KV，其余服务容量固定。G32 Off 与 a890 动态 C1 均完成 64/64、错误/拒绝 0；两轮全部 96 条样本身份及顺序一致：

| 实现 | C | TTFT P50/P99 ms | last-visible TPOT P50/P99 ms/token | 可见 SSE 文本事件 ITL P50/P99 ms | 输出 tok/s | MTLDevice 分配峰值 B | 本次 SLO |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| G32 Off | 1 | 333.80 / 4260.72 | 56.56 / 57.37 | 56.64 / 58.96 | 17.04 | 8590311424 | TTFT 超标 |
| a890 SLO-aware / Auto / mixed | 1 | 334.69 / 4274.67 | 56.67 / 57.51 | 56.77 / 59.04 | 17.00 | 未采集 | TTFT 超标 |

阈值为 P99 3400/212/359ms。当前扫描未找到可行档，不记为 G=0。保留 8 个 event/usage 不一致，未观察到 transport coalescing；内存收集 6596 个样本、错误 0。OS footprint 峰值 1628850672 B、RSS 峰值 380387328 B；MTLDevice 分配量不代表整机物理内存占用，各口径不相加。

最长请求为模板前 954 token、服务端 966 token，TTFT 4985.16ms；约 190 tok/s 是当前实现的有效速率，不是硬件物理上限。该结果说明当前 C1 长 prompt 已超标，不能推出所有配置均不可行。原目标表中的 llama.cpp C1 是另一组 32 个测量样本、最长 606 token，不能用其 P99 直接证明本轮 64 样本的同负载可行性；本次保留原阈值和筛选规则。

a890 的最长请求 TTFT 为 4972.05ms，4/64 请求 TTFT 超过 3400ms；TTFT/TPOT 各 64 个样本，可见 SSE 文本事件 19744 / 间隔 19680，保留 8 个 event/usage 不一致，无 coalescing、缺失 usage 或协议/机械输出检查异常。主机内存采样 6619 次，RSS 峰值 251674624 B、physical footprint 峰值 1668057344 B，二者不相加；设备分配峰值未采集。请求排空，服务退出且端口释放。此行为 a890 的性能证据，不作为后续 c068 拟合修正的实测结果。

两版本存在明确配置差异：a890 为 Auto→`qwen3_5.f32-master`、mixed、active-decode chunk 128 / budget 256；旧 G32 为 `f16-head`、split、budget 0。两者 effective reusable execution 均为 0。此表用于报告现状，不能单独归因于 SLO 开关。a890 本轮实际动态 prefill 步数为 0，故本格也不能代替已有混合请求 E2E 的动态使用验证。完整结果见仓库外 [M4 C1 摘要](/Users/chejinxuan/ferrum-handoffs/20261006-throughput-p0/metal/slo-sharegpt-c1-r1-summary.json)。

原始报告、完整命令和配置位于仓库外 `~/ferrum-handoffs/20261006-throughput-p0/`（CUDA 旧版基线为 `cuda/c1-r1/`，静态候选为 `cuda/run-smoke-r2/`、`cuda/candidate-serve-r1/`，本机静态/动态功能为 `local-smoke/`、`local-slo-smoke/`，M4 为 `metal/`）。上述 CUDA 测量已结束并恢复原服务，M4 两轮主负载服务均已清理。a890 CUDA 主负载首轮未见动态收益；M4 相同主负载 C1 仍因 TTFT 未达标。
