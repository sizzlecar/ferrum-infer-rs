# P0：干净候选与首轮实测

状态：静态候选功能及两端 E2E 已通过，性能验收进行中，尚未发布。目标与口径见 [目标文档](goal-slo-throughput.zh.md)。

候选 `57ee5ccb` 基于 main `2c9998a2`，仅接入既有静态执行优化 `c3c9ee0b` 与独立客户端测量；没有服务 SLO 控制器、启动校准或成本模型。旧恢复分支已作本地归档 tag `archive/slo-recovery-20261006`。原脏工作树未修改。

候选已有 workspace 格式、编译、测试（4591 通过、72 项现有 ignored）、Clippy、Metal 全目标编译和 CUDA CLI 三 feature 编译通过。RTX 5090 / Qwen3.5-9B Q4_K_M 已跑通自然 `run`、非流式 `serve` 和现有 mixed 调度断流恢复 E2E（5.43s）：取消一个请求后，同批请求和新 prefill 正常完成，资源排空，随后三个并发请求成功。自然问答解释 KV cache，21 输入 / 32 输出 token，正常 stop。

M4 / 同模型也通过自然 `run`、非流式及 SSE 流式请求，以及相同断流恢复 E2E（14.61s）。三种自然问答均完整回答三段，91 输入 / 215 输出 token，正常 stop；流式 terminal / usage 齐全。本次 M4 服务已关闭，端口释放。功能 E2E 采用 mixed、active-decode chunk 128 / budget 256、context 4096 / slots 4，与性能表固定容量配置分别记录。

本机 M1 Max 使用已有 Qwen3.5-4B Q4_K_M 也通过上述功能及流式自然结束检查。它是附加功能证据，不替代目标 M4 的性能或 E2E。

新增 `bench-serve --latency-slo ttft:200,tpot:15,itl:50 --latency-slo-fail-on-violation` 独立评估每次重复的三项 P99、错误和拒绝。真实服务上已验证 JSON/Markdown 报告及退出码：宽松阈值通过，过紧阈值先写失败报告再返回非零。这个小样本合成 E2E 只验证接口，不作为性能证据；主性能负载仍为固定 ShareGPT。当前服务端采用静态调度配置，没有按 SLO 自动调整调度的动态功能。

## 首轮 CUDA 5090 数据

以下均为单次探索：32 预热 + 64 测量请求、C1。Qwen3.5-9B Q4_K_M，FP16 KV，context 2048、服务 slots 32、batch 2048；容量在扫描中固定。ShareGPT seed 42，首轮用户/参考答案长度回放，输入 4–1024、最少输出 4、总长含 32-token 模板预留 ≤2048，ignore_eos、thinking off、temperature 0、top_p 1、repetition penalty 1。两臂相同完整选择 hash：`03676b00ee0b9e8407f59e3fb531eeff99dde5ccc9e59844b6abf7d22f1d5df2`。

| 实现 | C | TTFT P50/P99 ms | last-visible TPOT P50/P99 ms/token | 可见 SSE 文本事件 ITL P50/P99 ms | 输出 tok/s | GPU allocated 峰值 | NVML 整卡峰值 MiB | OS footprint 峰值 | RSS 峰值 KiB | 本次 SLO |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- | ---: | --- |
| G32 Off，RN-F16 fragment profile | 1 | 19.75 / 104.60 | 9.60 / 10.04 | 9.73 / 10.58 | 102.08 | 未采集 | 22141 | 未采集 | 6157288 | 三项通过 |
| llama.cpp b11065 | 1 | 28.70 / 107.57 | 5.99 / 7.01 | 5.85 / 7.00 | 164.97 | 未采集 | 10187 | 未采集 | 5931552 | 三项通过 |

阈值为 P99 200/15/50ms；两臂均 64/64 完成、错误/拒绝 0。这是 C1 的达标点，不是容量边界或重复验收。G32 Off 记录 19742 个非空文本事件、19678 个间隔；8 个 event/usage 不一致保留，未观察到 transport coalescing。协议/完整性检查未发现异常，模型语义另由自然问答 E2E 检查。内存口径相互重叠，不相加。

## 首轮 M4 数据

沿用同一 ShareGPT 选择规则及 32 预热 + 64 测量请求、一次重复；context 4096，其余服务容量固定。G32 Off C1 完成 64/64、错误/拒绝 0：

| 实现 | C | TTFT P50/P99 ms | last-visible TPOT P50/P99 ms/token | 可见 SSE 文本事件 ITL P50/P99 ms | 输出 tok/s | MTLDevice 分配峰值 B | 本次 SLO |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| G32 Off | 1 | 333.80 / 4260.72 | 56.56 / 57.37 | 56.64 / 58.96 | 17.04 | 8590311424 | TTFT 超标 |

阈值为 P99 3400/212/359ms。当前扫描未找到可行档，不记为 G=0。保留 8 个 event/usage 不一致，未观察到 transport coalescing；内存收集 6596 个样本、错误 0。OS footprint 峰值 1628850672 B、RSS 峰值 380387328 B；MTLDevice 分配量不代表整机物理内存占用，各口径不相加。

原始报告、完整命令和配置位于仓库外 `~/ferrum-handoffs/20261006-throughput-p0/`（CUDA 旧版基线为 `cuda/c1-r1/`，候选功能为 `cuda/run-smoke-r2/`、`cuda/candidate-serve-r1/`，本机功能为 `local-smoke/`，M4 为 `metal/`）。CUDA 旧版基线 guard 已恢复原服务并验证健康，候选 C1 测量仍在运行；M4 本次服务已清理。尚无候选吞吐或 SLO 达标结论。
