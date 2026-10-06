# 性能结果与发布状态（2026-10-06）

当前候选 `4d373a5c` 已恢复 CUDA 输出 head 的连续行合批入口，自然 `run`、CUDA 数值/边界检查与动态断流恢复 E2E 均通过；**静态 C4 为 267.548 tok/s，TPOT P99 15.053ms 仍超标，尚未追平 G32，不能发布**。目标见[目标文档](goal-slo-throughput.zh.md)。

最新发布条件是追平旧 G32、静态 CUDA C4 满足三项 SLO，并通过真实 run/serve E2E；最终只做既定 C4/C8 检查。原 P2 内核完整路线与 200 样本 × 2 次重复不再阻塞本次发布，但本页的一次重复不能用于宣称统计稳定性。当前动态调度未测到收益，默认关闭；仅 ITL 触发的保守修订已通过定向调度测试，最终双端 E2E 尚待完成，不再开动态调参轮次。

## 固定口径与配置

- 模型：同机同一 Qwen3.5-9B Q4_K_M GGUF，FP16 KV；输出质量与协议检查独立于计时。数据集 SHA256：`35f0e213ce091ed9b9af2a1f0755e9d39f9ccec34ab281cd4ca60d70f6479ba4`。
- 主表选择 SHA256：`03676b00ee0b9e8407f59e3fb531eeff99dde5ccc9e59844b6abf7d22f1d5df2`。每格 seed 42、32 预热 + 64 测量、1 次重复；两实现及并发档保持全部 96 个样本身份和顺序相同。
- 首轮用户/参考答案回放；输入 4–1024 token、输出至少 4、输入+参考输出+32-token 模板预留 ≤2048；输出长度取参考答案，ignore_eos、thinking off、temperature 0、top_p 1、repetition penalty 1。使用原冻结 Rust HTTP 客户端，版本/hash、完整命令与配置见各结果摘要。
- 三项分别判 P99，并要求错误/拒绝均为 0；TPOT 截止 last-visible output，ITL 是跨请求 pooled 的连续非空**可见 SSE 文本事件间隔**，排除 role-only/空/finish-only 事件；不沿用旧 joint 总状态，不丢弃长停顿。
- CUDA RTX 5090：P99 TTFT/TPOT/ITL = **200/15/50ms**；context 2048、slots 32、batch 2048、24GiB Ferrum runtime budget，prefix/session cache off。容量在并发扫描中固定。
- CUDA RN profile 为 `qwen3_5.f32-master.gguf-f16-projections.ffn-rn-fragment-m1to8`；旧 G32（`e3b9dda0`）为 Off/split/budget 0，c068 与 4d373 主表为 mixed、无显式 chunk/budget，c068 静态/动态之间只改变 `--scheduler-slo`。llama.cpp 为 b11065，保留其独立 KV/实现配置。
- M4：context 4096、slots 32、batch 2048、Ferrum runtime budget 8GiB、KV capacity 4096、cache off；旧 G32 为 f16-head/split/budget 0，a890 为 Auto→f32-master/mixed/chunk 128/budget 256，两者 effective reusable execution 均为 0。llama.cpp b10964 / `b29c606e2-device-memory` 使用 unified KV 32×4096、ubatch 512、Flash Attention，模板及长度政策相同。
- M4 下表按原 **3400/212/359ms** 判断；用户授权后续 TTFT P99 改为 **5500ms**，TPOT/ITL 仍为 212/359ms、样本不变。历史数据没有重跑，旧失败标签不改写为新阈值验收。

## CUDA 主负载结果

下表延迟单位 ms（TPOT 为 ms/token）。每格均 64/64 完成、错误/拒绝/坏输出/协议异常为 0，TTFT/TPOT 各 64 样本；所有内存口径互有重叠，不相加。

| 实现 / 模式 | C | TTFT P50/P99 | last-visible TPOT P50/P99 | 可见 SSE ITL P50/P99 | 输出 tok/s | 本次 SLO |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| G32 Off / split | 1 | 19.75 / 104.60 | 9.60 / 10.04 | 9.73 / 10.58 | 102.08 | 三项通过 |
| G32 Off / split | 2 | 29.99 / 114.04 | 11.62 / 12.71 | 11.48 / 21.23 | 167.632 | 三项通过 |
| G32 Off / split | 4 | 34.47 / 117.13 | 13.18 / 14.33 | 12.83 / 29.33 | 287.582 | 三项通过 |
| llama.cpp b11065 | 1 | 28.70 / 107.57 | 5.99 / 7.01 | 5.85 / 7.00 | 164.97 | 三项通过 |
| c068 静态 / mixed | 1 | 19.54 / 103.68 | 9.68 / 10.08 | 9.79 / 10.63 | 101.492 | 三项通过 |
| c068 静态 / mixed | 2 | 31.45 / 114.37 | 11.50 / 12.12 | 11.55 / 12.84 | 168.545 | 三项通过 |
| c068 静态 / mixed | 4 | 42.82 / 125.53 | 14.82 / 15.74 | 14.64 / 24.21 | 255.865 | TPOT 超标 |
| c068 动态 / mixed | 1 | 19.69 / 104.35 | 9.68 / 10.14 | 9.80 / 10.66 | 101.382 | 三项通过 |
| c068 动态 / mixed | 2 | 73.18 / 194.68 | 11.49 / 12.05 | 11.49 / 16.01 | 166.844 | 三项通过 |
| c068 动态 / mixed | 4 | 58.71 / 137.36 | 15.02 / 16.01 | 14.81 / 23.40 | 252.159 | TPOT 超标 |
| c068 静态 / split | 4 | 33.03 / 116.93 | 14.92 / 16.83 | 14.64 / 30.35 | 254.597 | TPOT 超标 |
| 4d373 静态 / mixed | 4 | 41.67 / 126.39 | 14.05 / 15.053 | 13.90 / 24.57 | 267.548 | TPOT 超标 |

| 内存采样范围（含加载、相应模式的全部格） | NVML 整卡峰值 MiB | RSS 峰值 KiB |
| --- | ---: | ---: |
| G32 C1 / G32 C2+C4 | 22141 / 22315 | 6157288 / 6153984 |
| llama.cpp C1 | 10187 | 5931552 |
| c068 mixed 静态 C1+C2+C4 | 20803 | 6073324 |
| c068 mixed 动态 C1+C2+C4 | 20676 | 6073492 |
| c068 split 静态 C4 | 20481 | 6073236 |
| 4d373 mixed 静态 C4 | 20471 | 6073836 |

CUDA GPU allocated 与 OS footprint 未采集；跨格峰值不能当作逐格值。c068 各格保留 8 个 event/usage 不一致，无缺失 usage 或观测到的 coalescing；静态 C1/C2、动态 C1 为 19742/19678 个文本事件/间隔，动态 C2 为 19746/19682，C4 为 19740/19676。4d373 C4 同样为 19740/19676、8 个 event/usage 不一致、无 coalescing 或缺失 usage。完整 llama 与旧版事件计数见原始摘要。

已扫描达标 G：旧 G32 至少 **287.582 tok/s@C4**；c068 mixed 静态 **168.545@C2**、动态 **166.844@C2（−1.01%）**。动态 C2 TTFT P99 升至 194.68ms；动态确实采用预算 396 步，但不构成收益。改回 split 或 chunk 256/budget 256 仍未恢复 C4，不能单独归因于 mixed，也不能用 C1 追平代替并发追平。

证据：[G32/llama C1](/Users/chejinxuan/ferrum-handoffs/20261006-throughput-p0/cuda/c1-r1/)、[G32 C2/C4](/Users/chejinxuan/ferrum-handoffs/20261006-throughput-p0/cuda/g32-c2c4-r1/summary.json)、[c068 主表](/Users/chejinxuan/ferrum-handoffs/20261006-throughput-p0/cuda/rn-nonnegative-r1/primary-summary.json)、[split 对照](/Users/chejinxuan/ferrum-handoffs/20261006-throughput-p0/cuda/fixed-boundary-split-c4-r1/summary.json)、[chunk 256 对照](/Users/chejinxuan/ferrum-handoffs/20261006-throughput-p0/cuda/fixed-boundary-chunk256-c4-r1/summary.json)。

## M4 同负载 C1 历史结果

下表三轮使用同一主表选择、全部 96 个样本同序；各 64/64 完成，错误/拒绝/协议及机械输出检查异常为 0。延迟仍按原 3400/212/359ms 判断，尚未做新 5500ms 阈值下的最终版本验收。

| 实现 | TTFT P50/P99 ms | last-visible TPOT P50/P99 ms/token | 可见 SSE ITL P50/P99 ms | 输出 tok/s | 本次 SLO |
| --- | ---: | ---: | ---: | ---: | --- |
| G32 Off | 333.80 / 4260.72 | 56.56 / 57.37 | 56.64 / 58.96 | 17.04 | TTFT 超标 |
| a890 动态 / Auto / mixed | 334.69 / 4274.67 | 56.67 / 57.51 | 56.77 / 59.04 | 17.00 | TTFT 超标 |
| llama.cpp b10964 | 344.01 / 4268.42 | 54.75 / 55.10 | 54.72 / 56.41 | 17.62 | TTFT 超标 |

| 实现 | MTLDevice 分配峰值 B | OS physical footprint 峰值 B | RSS 峰值 B |
| --- | ---: | ---: | ---: |
| G32 Off | 8590311424 | 1628850672 | 380387328 |
| a890 | 未采集 | 1668057344 | 251674624 |
| llama.cpp | 12383600640 | 6357181608 | 11856625664 |

MTLDevice 分配量不是整机物理占用，各口径不相加；a890 缺设备分配值，不能宣称设备内存优势。a890/llama 的事件/间隔分别为 19744/19680、19740/19676，event/usage 不一致分别 8/9 个，均无 coalescing 或缺失 usage。a890 C1 的动态 prefill 采用数为 0，不能代替混合请求 E2E。

最长请求为模板前 954 / 服务端 966 token，G32 最大 TTFT 4985.16ms（a890 4972.05ms、llama 4970.64ms）。约 190 tok/s 是该实现有效速率，不是物理上限；旧 llama 32 测量样本/最长 606 token 不可与本轮直接比较。原阈值下当前扫描未找到可行档，不记 G=0，也不推出所有配置不可能。

证据：[G32](/Users/chejinxuan/ferrum-handoffs/20261006-throughput-p0/metal/off-c1-r1-summary.json)、[a890](/Users/chejinxuan/ferrum-handoffs/20261006-throughput-p0/metal/slo-sharegpt-c1-r1-summary.json)、[同负载 llama](/Users/chejinxuan/ferrum-handoffs/20261006-throughput-p0/metal/llama-sharegpt-c1-r1-summary.json)。这些是各自版本结果，不能改署后续 c068/4d；M4 服务已清理。

## 功能验收与热点限制

`c068e3d7` 已通过 workspace 格式/全目标编译/测试/Clippy、Metal 全目标编译和 CUDA 三 feature 编译；两端自然 run 与原 ff47 动态 E2E 均通过：CUDA/M4 各 7 请求成功、0 失败、资源排空，实际动态 prefill 累计 7/10 步。见 [CUDA 功能](/Users/chejinxuan/ferrum-handoffs/20261006-throughput-p0/cuda/rn-nonnegative-r1/functional-summary.json)、[M4 功能](/Users/chejinxuan/ferrum-handoffs/20261006-throughput-p0/metal/slo-r8-summary.json)。功能通过不等于性能达标。

`4d373a5c` 仅恢复 CUDA 连续物理行的 output-head 合批，非法行序/别名回退逐请求执行；既有 CUDA 数值检查、两项边界检查、自然 run（21/32 输入/输出 token，正常 stop）和原动态 E2E（2.66s、7 完成/0 失败、排空）通过，原服务已恢复。workspace 与 Metal 编译检查也通过。见 [4d 功能摘要](/Users/chejinxuan/ferrum-handoffs/20261006-throughput-p0/cuda/rn-packed-head-r1/functional-summary.json)、[4d 静态 C4](/Users/chejinxuan/ferrum-handoffs/20261006-throughput-p0/cuda/rn-packed-head-r1/primary-summary.json)；G32/4d 同配置 Nsight 双版比较仍在进行，不据单点追加补丁。

历史 c068 Off C8 Nsight 短诊断（8 预热+8 测量）记录 Q6 scalar 发射 **4634** 次、累计 2.81777s（完整 trace kernel 累计时间 19.0%），Q6 tiled **0**；包含加载/预热，不能当作测量窗口 GPU 利用率。RN fragment 实际使用。见[原采样摘要](/Users/chejinxuan/ferrum-handoffs/20261006-throughput-p0/cuda/native-profile-c068-c8-r1/summary.json)。

M4 c068 Off C8 同样只作短诊断：首个 fingerprint 的 484 条已观察 native 记录，GPU 时间戳区间合计 9444.07ms。整个 GDN 为 3050.81ms（32.30%），含投影 1950.90ms；真正 recurrent_scan 为 **969.35ms（10.26%）**。投影已 packed，scan 仍逐 participant dispatch：24 个 GDN 记录各 7 participants/7 scan 区间，共 168；源码 `gated_delta_attention.rs:1670,1881` 的 grid 无序列维度，`.metal:342` 在序列内按 token 递推。

该 Metal 子集缺 physical submission 总记录、阶段标签且 command index 有 8 个内部缺口，不能证明完整 submission 或代表 decode/prefill；全局 GPU busy、host gap、权重带宽缺测。跨序列 scan 合批仅为后续候选，不能据此归因每新增请求约 16ms 的增量或承诺收益。两份 8.34GB 原始文件留 M4，路径/大小/hash 和原 Rust 子集摘要见[采样摘要](/Users/chejinxuan/ferrum-handoffs/20261006-throughput-p0/metal/native-profile-r8-c8-r1-summary.json)；instrumentation 改变边界，不作 SLO/G 证据。

## 历史失败与证据保留

早期动态合成诊断虽改善可见间隔，却将 TTFT 拖至约 10 秒；仿射/回退修正版仍有 TTFT 与吞吐取舍，不计主表成功。M4 a890 冷启动动态 E2E r5/r6 的 adapt=0 失败保留，后续 ff47 保留原请求/取消断言并另加已学习阶段；CUDA 负 decode 拟合失败由 c068 边界重拟合修正。没有删除失败或原始证据。

历史入口：[CUDA 短诊断 r1](/Users/chejinxuan/ferrum-handoffs/20261006-throughput-p0/cuda/dynamic-r1/summary.json)、[r2](/Users/chejinxuan/ferrum-handoffs/20261006-throughput-p0/cuda/dynamic-r2/summary.json)、[M4 r4 及更早证据](/Users/chejinxuan/ferrum-handoffs/20261006-throughput-p0/metal/slo-r4-summary.json)、[r5](/Users/chejinxuan/ferrum-handoffs/20261006-throughput-p0/metal/slo-r5-summary.json)、[r6](/Users/chejinxuan/ferrum-handoffs/20261006-throughput-p0/metal/slo-r6-summary.json)、[r7](/Users/chejinxuan/ferrum-handoffs/20261006-throughput-p0/metal/slo-r7-summary.json)、[CUDA 拟合失败](/Users/chejinxuan/ferrum-handoffs/20261006-throughput-p0/cuda/rn-q6-07523-r1/lifecycle-failure.json)、[a890 主表](/Users/chejinxuan/ferrum-handoffs/20261006-throughput-p0/cuda/rn-primary-a890-r1/summary.json)。全部日志、命令、配置、二进制绑定与采样保留在仓库外 `~/ferrum-handoffs/20261006-throughput-p0/`。
