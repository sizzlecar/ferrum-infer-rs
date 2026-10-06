# P0：干净候选与首轮实测

状态：进行中，尚未发布。目标与口径见 [目标文档](goal-slo-throughput.zh.md)。

候选 `57ee5ccb` 基于 main `2c9998a2`，仅接入既有静态执行优化 `c3c9ee0b` 与独立客户端测量；没有服务 SLO 控制器、启动校准或成本模型。旧恢复分支已作本地归档 tag `archive/slo-recovery-20261006`。原脏工作树未修改。

候选已有 workspace 格式、编译、测试、Clippy 和 Metal 全目标编译通过；这些不代表实机 E2E 或性能达标。当前优先完成两端自然 `run`、流式/非流式 `serve`、并发断流后同批请求存活与资源恢复。

## 首轮 CUDA 5090 数据

以下均为单次探索：32 预热 + 64 测量请求、C1。Qwen3.5-9B Q4_K_M，FP16 KV，context 2048、服务 slots 32、batch 2048；容量在扫描中固定。ShareGPT seed 42，首轮用户/参考答案长度回放，输入 4–1024、最少输出 4、总长含 32-token 模板预留 ≤2048，ignore_eos、thinking off、temperature 0、top_p 1、repetition penalty 1。两臂相同完整选择 hash：`03676b00ee0b9e8407f59e3fb531eeff99dde5ccc9e59844b6abf7d22f1d5df2`。

| 实现 | C | TTFT P50/P99 ms | last-visible TPOT P50/P99 ms/token | 可见 SSE 文本事件 ITL P50/P99 ms | 输出 tok/s | GPU allocated 峰值 | NVML 整卡峰值 MiB | OS footprint 峰值 | RSS 峰值 KiB | 本次 SLO |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- | ---: | --- |
| G32 Off，RN-F16 fragment profile | 1 | 19.75 / 104.60 | 9.60 / 10.04 | 9.73 / 10.58 | 102.08 | 未采集 | 22141 | 未采集 | 6157288 | 三项通过 |
| llama.cpp b11065 | 1 | 28.70 / 107.57 | 5.99 / 7.01 | 5.85 / 7.00 | 164.97 | 未采集 | 10187 | 未采集 | 5931552 | 三项通过 |

阈值为 P99 200/15/50ms；两臂均 64/64 完成、错误/拒绝 0。这是 C1 的达标点，不是容量边界或重复验收。G32 Off 记录 19742 个非空文本事件、19678 个间隔；8 个 event/usage 不一致保留，未观察到 transport coalescing。协议/完整性检查未发现异常，模型语义另由自然问答 E2E 检查。内存口径相互重叠，不相加。

原始报告、完整命令和配置位于仓库外 `~/ferrum-handoffs/20261006-throughput-p0/cuda/c1-r1/`。CUDA guard 已恢复原服务并验证健康。M4 首档测量及两端候选构建/E2E 正在进行，尚无候选吞吐结论。
