# 性能结果与发布状态（2026-10-06）

本次版本以 `2929f339` 产品代码、`ec322405` E2E 客户端收口。CUDA 静态 C4 **284.688 tok/s，三项 SLO 全部通过**，比旧 G32 的 287.582 tok/s 低 **1.01%**，满足[目标文档](goal-slo-throughput.zh.md) P1 既定的 ≤3% 基线保留条件。两端真实 run/E2E 和必需检查均通过；C8 的 TPOT 超标，保留失败，不再追加优化或调参。

最新发布条件是追平旧 G32、静态 CUDA C4 满足三项 SLO，并通过真实 run/serve E2E；最终只做既定 C4/C8 检查。原 P2 内核完整路线与 200 样本 × 2 次重复不再阻塞本次发布，但本页的一次重复不能用于宣称统计稳定性。当前动态调度未测到收益，默认关闭；仅 ITL 触发的修订已通过两端真实 run/E2E，不再开动态调参轮次。

## 固定口径与配置

- 模型：同机同一 Qwen3.5-9B Q4_K_M GGUF，FP16 KV；输出质量与协议检查独立于计时。数据集 SHA256：`35f0e213ce091ed9b9af2a1f0755e9d39f9ccec34ab281cd4ca60d70f6479ba4`。
- 主表选择 SHA256：`03676b00ee0b9e8407f59e3fb531eeff99dde5ccc9e59844b6abf7d22f1d5df2`。每格 seed 42、32 预热 + 64 测量、1 次重复；两实现及并发档保持全部 96 个样本身份和顺序相同。
- 首轮用户/参考答案回放；输入 4–1024 token、输出至少 4、输入+参考输出+32-token 模板预留 ≤2048；输出长度取参考答案，ignore_eos、thinking off、temperature 0、top_p 1、repetition penalty 1。使用原冻结 Rust HTTP 客户端，版本/hash、完整命令与配置见各结果摘要。
- 三项分别判 P99，并要求错误/拒绝均为 0；TPOT 截止 last-visible output，ITL 是跨请求 pooled 的连续非空**可见 SSE 文本事件间隔**，排除 role-only/空/finish-only 事件；不沿用旧 joint 总状态，不丢弃长停顿。
- CUDA RTX 5090：P99 TTFT/TPOT/ITL = **200/15/50ms**；context 2048、slots 32、batch 2048、24GiB Ferrum runtime budget，prefix/session cache off。容量在并发扫描中固定。
- CUDA RN profile 为 `qwen3_5.f32-master.gguf-f16-projections.ffn-rn-fragment-m1to8`；旧 G32（`e3b9dda0`）与最终 `2929f339` 为静态/split/budget 0，c068 与 4d373 主表为 mixed、无显式 chunk/budget，c068 静态/动态之间只改变 `--scheduler-slo`。llama.cpp 为 b11065，保留其独立 KV/实现配置；可复制的最终 Ferrum 命令见 [README](../README.md#runtime-settings)。
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
| 2929 最终静态 / split | 4 | 31.42 / 116.10 | 13.35 / 14.67 | 12.98 / 29.25 | 284.688 | 三项通过 |
| 2929 最终静态 / split | 8 | 40.99 / 166.47 | 17.43 / 20.62 | 16.53 / 41.41 | 414.436 | TPOT 超标 |

| 内存采样范围（含加载、相应模式的全部格） | NVML 整卡峰值 MiB | RSS 峰值 KiB |
| --- | ---: | ---: |
| G32 C1 / G32 C2+C4 | 22141 / 22315 | 6157288 / 6153984 |
| llama.cpp C1 | 10187 | 5931552 |
| c068 mixed 静态 C1+C2+C4 | 20803 | 6073324 |
| c068 mixed 动态 C1+C2+C4 | 20676 | 6073492 |
| c068 split 静态 C4 | 20481 | 6073236 |
| 4d373 mixed 静态 C4 | 20471 | 6073836 |
| 2929 split 静态 C4+C8 | 21122 | 6072936 |

CUDA GPU allocated 与 OS footprint 未采集；跨格峰值不能当作逐格值。c068 各格保留 8 个 event/usage 不一致，无缺失 usage 或观测到的 coalescing；静态 C1/C2、动态 C1 为 19742/19678 个文本事件/间隔，动态 C2 为 19746/19682，C4 为 19740/19676。4d373 C4 及最终 2929 C4/C8 均为 19740/19676、8 个 event/usage 不一致、无 coalescing 或缺失 usage。最终两格各 19801 usage 输出 token；所选输入 min/median/max/mean 为 5/28.5/954/101.45 token，指定输出为 6/288/788/309.39 token。完整 llama 与旧版事件计数见原始摘要。

已测可行 G：最终 `2929f339` **284.688 tok/s@C4**，旧 G32 至少 **287.582@C4**。此前 c068 mixed 静态 **168.545@C2**、动态 **166.844@C2（−1.01%）**；动态 C2 TTFT P99 升至 194.68ms，采用预算 396 步但没有收益。最终 ITL 修订仅做功能 E2E，未重跑动态性能对照，不能宣称其改善 G 或已证明“不伤害”。

最终两格包含预热共 192/192 成功、0 失败、资源排空；静态 health 的 `scheduler.slo=null`。client/server/guard 均退出 0，原 CUDA 服务恢复并通过 health 检查。版本、二进制与客户端 SHA、完整命令、P50/P99、事件数及共享内存峰值见最终 C4/C8 摘要（`/Users/chejinxuan/ferrum-handoffs/20261006-throughput-p0/cuda/rn-batched-attn-r1/primary-summary.json`）及同目录原始小包；1 次重复不能说明统计稳定性或全局最优。

证据：G32/llama C1（`/Users/chejinxuan/ferrum-handoffs/20261006-throughput-p0/cuda/c1-r1/`）、G32 C2/C4（`/Users/chejinxuan/ferrum-handoffs/20261006-throughput-p0/cuda/g32-c2c4-r1/summary.json`）、c068 主表（`/Users/chejinxuan/ferrum-handoffs/20261006-throughput-p0/cuda/rn-nonnegative-r1/primary-summary.json`）、split 对照（`/Users/chejinxuan/ferrum-handoffs/20261006-throughput-p0/cuda/fixed-boundary-split-c4-r1/summary.json`）、chunk 256 对照（`/Users/chejinxuan/ferrum-handoffs/20261006-throughput-p0/cuda/fixed-boundary-chunk256-c4-r1/summary.json`）。

## M4 同负载 C1 历史结果

下表三轮使用同一主表选择、全部 96 个样本同序；各 64/64 完成，错误/拒绝/协议及机械输出检查异常为 0。延迟仍按原 3400/212/359ms 判断，尚未按新 5500ms 阈值重跑最终版本的主负载性能。

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

证据：G32（`/Users/chejinxuan/ferrum-handoffs/20261006-throughput-p0/metal/off-c1-r1-summary.json`）、a890（`/Users/chejinxuan/ferrum-handoffs/20261006-throughput-p0/metal/slo-sharegpt-c1-r1-summary.json`）、同负载 llama（`/Users/chejinxuan/ferrum-handoffs/20261006-throughput-p0/metal/llama-sharegpt-c1-r1-summary.json`）。这些是各自版本结果，不能改署后续 c068/4d；M4 服务已清理。

## 功能验收与热点限制

`2929f339` 已通过 workspace 格式/全目标编译/测试/Clippy、Metal 全目标编译和 CUDA 三 feature 编译。两端使用同一 `ec322405` E2E 客户端，保留取消/恢复/后续请求和独立动态采用断言：CUDA 2.51s、M4 19.37s，各 7 完成/0 失败/0 拒绝、资源排空，实际缩预算累计 2/10 步。CUDA 自然 run 为 21/32 输入/输出 token、862.39ms，M4 为 91/215、12663.43ms，均自然 stop、内容正确。M4 产品仍为共享 ITL 策略 `0b2a8a0c`，后续产品改动仅涉及 CUDA；M4 使用 5500/212/359ms，两端功能测试 chunk/budget 均为 512。见 CUDA 功能（`/Users/chejinxuan/ferrum-handoffs/20261006-throughput-p0/cuda/rn-batched-attn-r1/functional-summary.json`）、M4 run（`/Users/chejinxuan/ferrum-handoffs/20261006-throughput-p0/metal/slo-r9-summary.json`）、M4 最终 E2E（`/Users/chejinxuan/ferrum-handoffs/20261006-throughput-p0/metal/slo-r10-summary.json`）。功能通过不等于性能达标。

CUDA 恢复的连续 output-head、注意力 V1/V2 合批、实时长度收集、上传合并均复用既有数值/边界检查；多序列不同长度与 replay 真 GPU 回归通过。最终 E2E 将已学习阶段改为完整 512-token 分块压力并保留全部断言：此前短输入在新版 CUDA 预测约 46ms，小于 50ms ITL，原“必须限流”断言失败；7 请求仍正常完成/排空。早期调用参数和测试 URL 遗漏也分别保留，最终通过记录与失败分开。

G32/4d 的完整 C4 Nsight 清单为 42/41 个 kernel 名称：Q6 tiled、RN 次数与单次时间接近；新版唯一缺名为 GPU length gather，注意力 V1/V2 全为单序列 grid，旧版有多序列 grid。注意力含 reduce 累计多 3.474s，占全部 kernel 增量 3.619s 的约 96%；HtoD 调用 74162→297356。`2929f339` 一批恢复注意力合批、实时 length gather 与跨间隔等宽上传合并。旧异步 DtoH 与新版同步读回的 API 累计差约 1.157s，涉及另一套 completion 所有权接口，本次保留该差异。两版同 RN/FP16 KV、split/budget 0、相同容量和 32+64 样本；旧版另有 startup 工作区预热，新版按需分配，不能称严格配置等价。窗口分配 API 仅约 10–12ms，不支持其解释主要差距。完整名称、次数、耗时、形状及 API 表见 G32 采样（`/Users/chejinxuan/ferrum-handoffs/20261006-throughput-p0/cuda/native-profile-g32-c4-r1/summary.json`）、4d 采样（`/Users/chejinxuan/ferrum-handoffs/20261006-throughput-p0/cuda/native-profile-packed-head-c4-r1/summary.json`）所在目录；带采样的成绩不作发布验收。

M4 c068 Off C8 同样只作短诊断：首个 fingerprint 的 484 条已观察 native 记录，GPU 时间戳区间合计 9444.07ms。整个 GDN 为 3050.81ms（32.30%），含投影 1950.90ms；真正 recurrent_scan 为 **969.35ms（10.26%）**。投影已 packed，scan 仍逐 participant dispatch：24 个 GDN 记录各 7 participants/7 scan 区间，共 168；源码 `gated_delta_attention.rs:1670,1881` 的 grid 无序列维度，`.metal:342` 在序列内按 token 递推。

该 Metal 子集缺 physical submission 总记录、阶段标签且 command index 有 8 个内部缺口，不能证明完整 submission 或代表 decode/prefill；全局 GPU busy、host gap、权重带宽缺测。跨序列 scan 合批仅为后续候选，不能据此归因每新增请求约 16ms 的增量或承诺收益。两份 8.34GB 原始文件留 M4，路径/大小/hash 和原 Rust 子集摘要见采样摘要（`/Users/chejinxuan/ferrum-handoffs/20261006-throughput-p0/metal/native-profile-r8-c8-r1-summary.json`）；instrumentation 改变边界，不作 SLO/G 证据。

## 历史失败与证据保留

早期动态合成诊断虽改善可见间隔，却将 TTFT 拖至约 10 秒；仿射/回退修正版仍有 TTFT 与吞吐取舍，不计主表成功。M4 a890 冷启动动态 E2E r5/r6 的 adapt=0 失败保留，后续 ff47 保留原请求/取消断言并另加已学习阶段；CUDA 负 decode 拟合失败由 c068 边界重拟合修正。没有删除失败或原始证据。

历史入口：CUDA 短诊断 r1（`/Users/chejinxuan/ferrum-handoffs/20261006-throughput-p0/cuda/dynamic-r1/summary.json`）、r2（`/Users/chejinxuan/ferrum-handoffs/20261006-throughput-p0/cuda/dynamic-r2/summary.json`）、M4 r4 及更早证据（`/Users/chejinxuan/ferrum-handoffs/20261006-throughput-p0/metal/slo-r4-summary.json`）、r5（`/Users/chejinxuan/ferrum-handoffs/20261006-throughput-p0/metal/slo-r5-summary.json`）、r6（`/Users/chejinxuan/ferrum-handoffs/20261006-throughput-p0/metal/slo-r6-summary.json`）、r7（`/Users/chejinxuan/ferrum-handoffs/20261006-throughput-p0/metal/slo-r7-summary.json`）、CUDA 拟合失败（`/Users/chejinxuan/ferrum-handoffs/20261006-throughput-p0/cuda/rn-q6-07523-r1/lifecycle-failure.json`）、a890 主表（`/Users/chejinxuan/ferrum-handoffs/20261006-throughput-p0/cuda/rn-primary-a890-r1/summary.json`）。全部日志、命令、配置、二进制绑定与采样保留在仓库外 `~/ferrum-handoffs/20261006-throughput-p0/`。
