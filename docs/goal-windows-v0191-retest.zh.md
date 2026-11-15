# Goal：Windows v0.19.1 外部反馈与文本推理性能

## 授权、起点与顺序

用户要求先完成 FastH3 视频，再处理外部 Windows 报告的问题。视频阶段已在独立分支提交 `47611945b5371b1df7496bbf3f0a155c6cbf7f96`，固定 5090 短片性能、CLI/HTTP、连续批处理及容量资格完成；该提交的 `docs/goal-fast-h3-video-5090.zh.md` 第 9–10 节保留验收和后续队列。

本阶段从 `origin/main` 的 `7bbea02877df04132ae7407a194ecb9d669265c2` 开始，分支 `fix/windows-v0191-followup-20261116`，独立工作树 `/Users/chejinxuan/rust_ws/ferrum-windows-v0191-followup-20261116`。不修改原 `ferrum-infer-rs` 工作树的其他性能工作，也不把未提交代码或视频功能混入 v0.19.1 对照。用户已授权继续使用 5090 测试和准备答复，未授权发送邮件。

外部依据：CJ Zwart / OpenSourcesAI 的 `OSAI-TR-20261113-FE Rev 1`，PDF SHA-256 `b3cbd30ee964f4453ed0c648abb2088e820570454d19e6b79846defc9d6cda5b`。附件与 54 单元重算在 `/Users/chejinxuan/ferrum-handoffs/20261114-v0191-windows-review`。2448/2448 完成是对方的观测和报告内部核查，不是本地复测。

## 必须处理的事项

| 顺序 | 问题 | 交付与边界 |
| --- | --- | --- |
| 1 | C8/C16 首 token 仍比 llama.cpp 慢约 25%/31% | 分开观测排队、prefill、首次采样、输出传输；在共享执行/调度层改进，以同机固定负载检验吞吐与全部延迟 SLO。 |
| 2 | C1 吞吐约为 llama.cpp 的 0.87× | 首 token 已接近；定位后续 decode 的 Host 提交、同步、设备计算与内存成本，覆盖 run/serve，避免模型名特判。 |
| 3 | Windows EXE/安装器未签名 | 核查发布流程和现有签名身份；准备签名、时间戳、产物收据及验签闭环。真实身份/凭据缺失单独列为依赖，不把校验第三方安装器当作 Ferrum 已签名。 |
| 4 | 4070 Ti 与 5090 的中位数改善不同 | 说明跨版本短提示词对照与同二进制 profile A/B、ShareGPT 的差别。既有 5090 短提示词 C16 中位数也下降；未测得因果前不归因于 SM89。 |
| 5 | temperature=0 跨批次文本不一致 | 固定请求、种子和数值 profile，区分数值路径、采样和状态隔离；检查语义正确性，不能只称“预期行为”。 |
| 6 | 证据与答复 | 保留每次重复、逐请求原始记录、错误和资源漏采；区分同场旧版 1.87×/2.09× 与另一次历史约 2.5×。准备第 8 节四问的回信草稿，不发送。 |

## 实验合同

先用现成 5090 做诊断，真实可访问设备是 RTX 5090 / WSL2。它不能替代对方 RTX 4070 Ti / Windows 11 的复测。外部客户端的完整原始 prompts 和逐请求数据尚未提供；现有 `email-short-topics.json` 是从话题构造的合成诊断，不能声称逐请求重现了外部报告。

模型固定为 `unsloth/Qwen3.5-4B-GGUF` revision `e87f176479d0855a907a41277aca2f8ee7a09523` 的 Q4_K_M，GGUF SHA-256 `00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4`。复用 5090 的正式 v0.19.1 发布包，llama.cpp b12069 为初始比较基线；记录两侧二进制、native lock、驱动及客户端身份。

服务固定容量 32、每请求上下文 2048、batch token 2048、FP16 KV、thinking 和 prefix cache 关闭；Ferrum 运行时预算固定 `10066329600` 字节。llama.cpp 使用总上下文 65536、parallel 32、ubatch 512、unified KV、flash attention、全 GPU、fit off。服务配置在 C1/C8/C16 扫描中不改变。

主负载采用固定 ShareGPT revision `745745adf6cd15b84e4f1c4a5a051fb4304f9342`、文件 SHA-256 `35f0e213ce091ed9b9af2a1f0755e9d39f9ccec34ab281cd4ca60d70f6479ba4`：首个 human/gpt 对，seed 42，选择池 74，输入 4–1024 tokens、参考输出 128–256、总长 ≤2048、模板预留 32。输出预算来自对应参考答案，ignore EOS；实际 usage 独立记录。固定 256 输出的合成短提示词另列诊断。每个实验的样本 ID、顺序、客户端模板和输出策略在候选比较前冻结，两侧及各并发格一致。

沿用上一轮同硬件工作负载的预声明绝对门槛：每格 **TTFT P99 ≤500ms、最后可见文本定义的 TPOT P99 ≤50ms、可见 SSE 文本事件 ITL P99 ≤100ms、请求/协议错误为 0**。报告每次重复的 P50/P99、吞吐、完整样本数、实际输出、GPU/RSS 峰值与采样覆盖。角色、空和仅结束 SSE 不计可见文本事件；传输合并保留在主 ITL 中，严格单 token 时序单列。

第一轮只做诊断 pilot：每个实现使用同一组 16 个样本，C1/C8/C16、各一次、每格 3 个非计时预热。低样本 P99 只能作为观测，不能宣布正式 SLO 或性能胜出。profile/NSight 独立运行，其耗时不混入基线。先找到实测瓶颈再改代码；正式候选比较另行冻结配对重复和置信区间规则，不反复查看结果追加样本。

## 核心设计约束与验证

优化围绕已有 provider、执行 owner、调度候选和数值合同；不增加第二套分配/提交权威，不用客户端名称或模型名称选择捷径。共享实现同时检查 run/serve；保持采样策略、模型模板、请求隔离、容量及 GPU 完成边界。只有真实证据允许的零提交失败才能重试。

复用 Rust 基准和正确性测试。先跑受影响检查，稳定候选才做仓库与后端验证；不写 Python/shell 测试包装器，不因文档变化重跑整仓。日志、模型、原始报告、可执行文件和 profile 全部留在库外。

## 当前状态

2026-11-16：视频目标已完成，本目标进行中。单请求注意力拓扑修复已单独保存为提交 `930c46b66`；后续 C16 性能资格仍未通过，没有发布或合并。正式 v0.19.1 Linux SM120 二进制重新计算的 SHA-256 为 `377e6ce053cc2f55ccdd147d941603b66bebac27d262fff42021cf2fc7339387`；模型与数据集的实际字节均已核对。原主工作树未修改。证据和完整执行参数保存在 `/Users/chejinxuan/ferrum-handoffs/20261116-windows-v0191-followup` 及 5090 的同名目录。

### 首轮同机诊断

每格 16 个计时请求、1 次重复、3 次预热；96/96 计时请求完成，错误为 0。两侧和各并发格的选样证据、实际输入 token 数、输出长度完全一致，每格输出 2652 usage tokens。以下是小样本观测，不是正式 P99 资格或 Windows 4070 Ti 结论。

| 实现 / 并发 | TTFT P50 / P99 ms | 最后可见文本 TPOT P50 / P99 ms | 可见 SSE ITL P50 / P99 ms | 输出 tokens/s | 观测 SLO |
| --- | --- | --- | --- | --- | --- |
| llama.cpp C1 | 26.97 / 58.99 | 4.37 / 4.66 | 3.92 / 25.89 | 221.68 | 通过 |
| Ferrum C1 | 45.65 / 71.03 | 4.19 / 29.14 | 4.13 / 54.39 | 153.50 | 通过 |
| llama.cpp C8 | 82.65 / 221.90 | 11.51 / 12.82 | 8.65 / 57.86 | 600.62 | 通过 |
| Ferrum C8 | 79.04 / 332.19 | 6.73 / 8.51 | 5.43 / 41.36 | 891.73 | 通过 |
| llama.cpp C16 | 549.48 / 607.30 | 16.46 / 18.49 | 14.93 / 55.72 | 699.54 | TTFT 超限 |
| Ferrum C16 | 169.01 / 226.12 | 8.13 / 8.60 | 6.47 / 30.48 | 1399.18 | 通过 |

llama.cpp 客户端退出 1 是预声明的 TTFT SLO 失败，不能抹去；请求本身没有失败。服务器均正常停止。部分请求的可见事件数与 usage token 数不一致，严格单 token 诊断资格不足，但可见 SSE ITL 仍保留。资源监测器记录设备总量、RSS 和间隔；WSL 没有目标进程的显存归属数据，设备采样峰值不能称为独占进程峰值。

C1 慢点集中在输入 444、511 tokens 的两个请求：跨过 512 后，逐步耗时由约 4.4ms 升至约 20–35ms。首次普通运行有 193 次同一 decode topology 的 catalog miss；第二次无 profile 复跑为 183.41 tokens/s，新形状延迟到第二个长请求才建立程序。另一次 latency 诊断成功建立后恢复约 4.4ms/token。basic 诊断明确记录 210 个 `amortization_deferred` 段、零捕获，表明持续回退来自程序准备准入；不能把 profile 运行更快当作优化成果。

候选改动复用已有固定容量注意力合同：规范 packed decode 即使只有一个参与者，也以当前设备 binding 选择 V1/V2，不让上下文跨分区改变已捕获拓扑。保留单请求原有非 paired 算法、KV 所有权、真实执行上下文上限和容量核算；不修改自动捕获成本策略，不加入模型名特判。增加单请求的边界、实际 CUDA replay 位一致性、KV 页增长和请求替换检查。真实 CUDA 和 run/serve 正确性已通过；扩大样本后暴露 C16 长尾超限，仍未记为性能修复完成。

### 同源码三组配对：C1 回退改善，C16 资格未通过

冻结合同为库外的 `paired-r1.contract.json`：每次服务启动依次扫 C1/C8/C16，每格 64 个计时请求、10 个预热；三个独立服务启动配对，顺序为 baseline→candidate、candidate→baseline、baseline→candidate。服务容量、模型、精度、客户端、输入样本与输出长度相同；逐格核对选样证据及 usage 数组一致。**1152/1152 个计时请求完成，错误为 0，但候选第 3 组 C16 的 TTFT P99 失败，不能认定本候选性能验收通过。**

同源码控制二进制 SHA-256 为 `cc06d16cdb41f77fbf85c3b596475460533b862fce80b009f8d02721bfb5ca52`，实际候选为 `3545da62626dc6fd7d9b0c80b72b6efe7e1767065151317a7c1566e820faa2d9`；两者均用 `cuda-correctness`（opt=3、无 LTO、16 codegen units、incremental off）、相同 CUDA features 和 native operator set。该对照与正式发布二进制分开。首次候选构建因源文件 mtime 早于控制构建，Cargo 复用了控制 CLI；在计时前通过二进制哈希发现，刷新改动文件 mtime 并重编。这个无效 CLI 没有用于测量，源文件字节没有改变。

下表保留每次重复的 Rust 客户端分位数；每行 n=64，错误=0，输出=12138 usage tokens。TPOT 采用最后可见文本，ITL 采用**可见 SSE 文本事件间隔**。有限样本的经验 P99 不构成所有请求的延迟保证。

| 配对 | 实现 | 并发 | TTFT P50 / P99 ms | TPOT P50 / P99 ms | 可见 SSE ITL P50 / P99 ms | 输出 tokens/s | 观测 SLO |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | baseline | 1 | 46.51 / 100.68 | 4.29 / 13.35 | 4.20 / 30.03 | 202.81 | pass |
| 1 | baseline | 8 | 52.50 / 412.80 | 7.73 / 12.14 | 5.54 / 51.42 | 898.00 | pass |
| 1 | baseline | 16 | 53.12 / 345.06 | 10.42 / 14.40 | 6.69 / 70.18 | 1350.71 | pass |
| 1 | candidate | 1 | 46.12 / 100.38 | 4.33 / 4.77 | 4.22 / 4.85 | 216.65 | pass |
| 1 | candidate | 8 | 53.68 / 198.84 | 7.71 / 10.77 | 5.56 / 68.90 | 926.72 | pass |
| 1 | candidate | 16 | 72.82 / 316.62 | 10.51 / 14.42 | 6.69 / 72.51 | 1329.57 | pass |
| 2 | baseline | 1 | 46.37 / 101.34 | 4.34 / 38.94 | 4.21 / 57.21 | 134.36 | pass |
| 2 | baseline | 8 | 53.80 / 286.26 | 7.74 / 10.74 | 5.54 / 55.73 | 928.85 | pass |
| 2 | baseline | 16 | 71.40 / 465.29 | 11.30 / 18.52 | 6.73 / 78.51 | 1219.74 | pass |
| 2 | candidate | 1 | 45.75 / 152.56 | 4.34 / 4.78 | 4.22 / 9.56 | 215.57 | pass |
| 2 | candidate | 8 | 50.72 / 330.21 | 7.70 / 10.72 | 5.55 / 52.16 | 921.58 | pass |
| 2 | candidate | 16 | 71.97 / 288.47 | 10.61 / 17.58 | 6.69 / 74.64 | 1243.51 | pass |
| 3 | baseline | 1 | 46.09 / 100.95 | 4.35 / 38.70 | 4.20 / 57.46 | 130.98 | pass |
| 3 | baseline | 8 | 65.88 / 239.00 | 7.87 / 10.91 | 5.52 / 70.74 | 917.53 | pass |
| 3 | baseline | 16 | 71.35 / 461.16 | 10.94 / 16.11 | 6.70 / 73.63 | 1279.29 | pass |
| 3 | candidate | 1 | 45.71 / 100.24 | 4.34 / 4.74 | 4.22 / 4.95 | 216.71 | pass |
| 3 | candidate | 8 | 48.83 / 361.42 | 7.62 / 12.29 | 5.52 / 50.11 | 897.98 | pass |
| 3 | candidate | 16 | 72.34 / 761.78 | 11.84 / 18.58 | 6.69 / 74.05 | 1148.23 | fail |

C1 候选吞吐为 215.57–216.71 tokens/s，配对比值为 1.068×、1.604×、1.655×；改善幅度随控制端触发回退的程度变化，不能概括为固定百分比提速。控制端记录的单请求 decode 程序缺失计数分别为 271、1243、1334；候选未记录同类行，但 ledger 有少量 overflow，不能把有限记录称为完整的零缺失证明。C8 配对吞吐变化为 −2.13%～+3.20%；C16 为 −10.24%～+1.95%，最后一组 TTFT P99 为 761.78ms，超过 500ms。三次启动不报告统计置信区间，不用 C1 收益掩盖 C16 失败。

资源监测覆盖每次并发扫描及其前后监测窗口；设备值是整张 GPU 的采样峰值，WSL 无目标进程显存归属，存在查询失败和约 2.2 秒的最大间隔，实际峰值可能更高。RSS HWM 是进程生命周期峰值，包含测量窗口前的模型加载。

| 配对 / 实现 | 设备采样峰值 GiB | RSS 采样峰值 / 生命周期 HWM GiB | GPU 样本 / 错误 / 最大间隔 ms | Host 样本 / 错误 |
| --- | --- | --- | --- | --- |
| 1 / baseline | 5.999 | 1.221 / 3.596 | 429 / 2 / 2002 | 488 / 0 |
| 1 / candidate | 5.991 | 1.237 / 3.592 | 1365 / 4 / 2185 | 1423 / 0 |
| 2 / baseline | 5.995 | 1.230 / 3.596 | 473 / 8 / 2171 | 625 / 0 |
| 2 / candidate | 6.304 | 1.188 / 3.593 | 436 / 3 / 2205 | 492 / 0 |
| 3 / baseline | 5.991 | 1.225 / 3.595 | 491 / 14 / 2206 | 672 / 0 |
| 3 / candidate | 5.991 | 1.235 / 3.578 | 336 / 7 / 2205 | 425 / 0 |

每格均保留严格单 token 时序资格计数、usage 与事件数不一致的请求，以及完整可见 SSE 间隔；不使用报告中不适用的传统 token ITL 零值。汇总为库外 `paired-r1.summary.json`，完整原始记录仍保留。

### 正确性及 C16 后续定位

- 定向 Rust 检查：54 项通过；另外在 5090 独立运行单请求边界及 M8/M32 固定容量 CUDA replay 两项，均通过位一致性、KV 所有权和越界 guard 检查。
- 复用正式发布的 Rust `model_regression`，11 个产品验收阶段全部通过，包含 run/serve 算术、记忆、清空历史、双请求历史隔离和长上下文检索。HTTP 同步/流式输入均为 1093 tokens，续问为 1123 tokens，均准确返回目标标识。
- 复用 Rust `agent_regression replay-http`，控制与候选各 64 个请求，合计 128 个独立请求 ID，传输完成、usage 完整。每版在 C1、C8 正序、C8 逆序中的 24 个状态请求均准确返回自己的标识，8 个无历史请求均返回 NONE。此处既有 assistant OK 为固定 wire fixture；真实历史写入及续问由上述 model_regression 覆盖。
- 每版八个开放问答的 C1 前后文本 8/8 相同，控制与候选 C1 文本也 8/8 相同。两版 C1 对 C8 正/逆序均为 0/8 相同，C8 正逆序之间也是 0/8。开放问答有 256-token 上限，部分以 length 结束，只用于文字差异诊断；不据此声称完整质量资格或跨批次字节一致。固定标识与空历史检查没有观察到串请求。
- C16 latency 诊断再次出现 TTFT P99 723.67ms。一个 27-token 输入在引擎首次 token 提交约 257ms 后，继续 decode 调度等待约 454ms，客户端首个可见文本约 744ms；同期 1028-token 请求约 711ms 收到首个可见文本。HTTP 首次入队与引擎流输出接近，排查重点已缩小到 prefill 提交和执行调度等待。
- 独立 basic 诊断记录了完整 1147 个 dispatch waves、零 ledger overflow：最慢预热 prefill Host encode/submit 为 225.95ms，主要在 enqueue 212.65ms；另一 prefill 为 159.35ms，enqueue 141.68ms，均未进行 graph capture。这轮 TTFT P99 为 349.99ms，但带诊断的结果不覆盖原无 profile 配对失败。
- 格式、diff、Metal 全 targets 编译及 Linux workspace 全 targets check 已通过。发现 C16 失败后，主动停止后续整仓测试构建；workspace tests 未完成，Clippy 和标准 CUDA CLI compile check 尚未运行。候选没有被标为 PR 已验证，不为重复跑整仓检查延后性能定位。
- 已实测现有公开 `--scheduler-slo ttft:500,tpot:50,itl:100`，孤立 C16 诊断仍失败：TTFT P99 1025.64ms、TPOT P99 21.98ms、可见 ITL P99 74.93ms，吞吐 1082.63 tokens/s、64/64 请求完成。仅打开自适应策略不能解决当前长尾，不把它作为已交付方案。

随后通过公开 `--scheduler-prefill-step-chunk 256 --scheduler-active-decode-prefill-token-budget 256` 做一次完整扫描：C1/C8/C16 吞吐为 215.75/949.95/1197.44 tokens/s，TTFT P99 为 138.14/249.07/397.58ms，192 个请求无错误。这一轮通过不能代替重复验收，因此另行冻结 `prefill-policy-paired-r1.contract.json`，在同一个候选二进制上比较是否启用上述两个限制。每次独立启动仅测 C16，保留相同 64 个计时样本及 10 个预热；三组顺序为 unbounded→bounded、bounded→unbounded、unbounded→bounded。**384/384 请求完成、零请求错误，两种策略均未通过完整资格。**

下表每行 n=64，TPOT 为最后可见文本口径，ITL 为可见 SSE 文本事件口径；额外列出超过 500ms 的请求数，避免把有限样本的插值 P99 解读为逐请求保证。内存列是设备总量与进程 RSS 的采样峰值，单位 MiB。

| 配对 / 策略 | TTFT P50 / P99 ms | TPOT P50 / P99 ms | 可见 SSE ITL P50 / P99 ms | 输出 tokens/s | GPU / RSS MiB | TTFT >500ms 请求数 | 观测 SLO |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 / unbounded | 63.90 / 647.91 | 11.70 / 18.43 | 6.75 / 79.95 | 1144.47 | 5975 / 1226 | 6 | fail |
| 1 / bounded | 75.09 / 609.37 | 12.68 / 20.51 | 6.73 / 79.53 | 1064.66 | 5783 / 1244 | 6 | fail |
| 2 / unbounded | 58.76 / 678.96 | 12.21 / 20.70 | 6.71 / 77.72 | 1089.90 | 6007 / 1256 | 6 | fail |
| 2 / bounded | 73.31 / 371.20 | 13.07 / 19.40 | 6.73 / 77.54 | 1098.03 | 5783 / 1224 | 0 | pass |
| 3 / unbounded | 76.62 / 643.55 | 11.05 / 18.93 | 6.71 / 81.20 | 1174.86 | 5815 / 1285 | 1 | fail |
| 3 / bounded | 75.33 / 339.12 | 12.33 / 18.23 | 6.69 / 76.02 | 1142.37 | 5815 / 1242 | 0 | pass |

选样和实际输入/输出 usage 数组逐格一致，原始逐请求记录、严格单 token 资格及资源记录在库外 `prefill-policy-paired-r1.summary.json` 和对应服务目录。每次 GPU 采样只有 15–72 个成功样本、3–4 次查询失败，最大间隔 2003–2184ms；峰值只是观测下界，不能作显存下降结论。原始完整扫描的失败仍保留，不以这次独立 C16 实验替换。

分块的三次配对吞吐变化约为 −6.97%、+0.75%、−2.77%，没有稳定收益且仍一次超限，暂不作为推荐配置。接下来使用本机已有 Nsight Systems 对 CUDA API 和 kernel 时序作诊断，区分内存分配、同步及逐算子提交；该跟踪有额外开销，不计为性能资格。停止继续盲调分块参数，也不更改预声明 SLO。

### 仍需完成

1. 解决上述 C16 TTFT 长尾及候选资格失败，再冻结新合同验证；C1 回退改善不等同于关闭外部短提示词 C1 差距。
2. C8/C16 的外部短提示词 TTFT 分解。5090 跨批次文字差异和状态隔离已做局部实测，仍不承诺跨批次字节一致或代替外部原始请求复测。
3. Windows 4070 Ti 实机复测和原始请求数据。5090 WSL2 诊断不能代替这一步。
4. Windows 正式签名身份和验签闭环。仓库级 secrets 未发现签名凭据，仓库 variables 和 environments 列表为空；现有流程仅验第三方安装器签名。尚未配置或使用任何正式 Ferrum 签名身份。
5. 第 8 节四问的回信草稿已备，性能结果确定后更新；不发送邮件，不把计划描述成已发布功能。

## Windows 签名接入方案（身份未配置，尚未执行）

接入点是 `.github/workflows/release-windows.yml` 的原有产物链，避免新增第二套打包器。正式发布身份由公开可信代码签名证书或受管签名服务提供；目前未获得该身份，不建立自签证书冒充正式签名，也不采购或开通服务。

1. 编译得到唯一的 core EXE 后，在 portable spec 和最终 SHA-256 生成前完成签名及 RFC 3161 时间戳。记录编译前后不同阶段的字节身份；现有 `build-profile.json` 的原始编译哈希不能继续假称为最终签名产物哈希。
2. 稳定 launcher 首次签名必须发行新的不可变资产并记录签名后 SHA-256；不能修改已经发布并被固定 URL/hash 引用的 `ferrum-windows-launcher-v1.exe`。复用 launcher 时，除哈希外还验证预期发行者和时间戳；旧未签名资产不能算入完整签名验收。
3. Inno Setup 的安装包和内嵌卸载程序分别进入签名合同：编译时使用受控 `SignTool` 与 `SignedUninstaller`，安装包形成后再验签。这样也覆盖 Setup 的临时执行副本；仅在编译后手动签署最外层 EXE 不足以覆盖它们。没有签名工具/身份的构建不能被标为 signed。[Inno SignTool](https://jrsoftware.org/inno-setup/documentation/topic_setup_signtool.htm)、[SignedUninstaller](https://jrsoftware.org/inno-setup/documentation/topic_setup_signeduninstaller.htm)。
4. 保持第三方 CUDA/VC DLL 的原始字节和发行方身份，不重新用 Ferrum 身份签署。ZIP 保持普通归档，内部 Ferrum EXE 必须为已验签字节；安装器携带同一份已验证 payload。
5. 验签后生成最终 portable receipt、staging provenance 和资产 SHA-256；解压 ZIP、检查安装器嵌入 payload 后再次比对哈希。收据记录原始/签名后哈希、发行者、证书指纹、时间戳、验证状态和工具版本；无私钥和访问令牌。

Windows SDK SignTool 的证书存储后端可使用下列形式；占位符必须来自已批准的发布配置，不能直接执行。若选受管签名服务，只替换签名执行适配器，后续验签和最终哈希阶段保持一致。

```text
signtool sign /s My /sha1 <certificate-thumbprint> /fd SHA256 /tr <approved-rfc3161-url> /td SHA256 <ferrum-exe>
signtool verify /pa /all /tw /v <ferrum-exe>
```

流程将 SignTool 非零退出（包括时间戳警告）视为未通过，同时检查 `Get-AuthenticodeSignature` 的 Valid 状态、预期 SignerCertificate 和 TimeStamperCertificate。`/pa` 使用 Authenticode 策略，不能套用驱动验签默认策略。[SignTool 官方参数](https://learn.microsoft.com/en-us/windows/win32/seccrypto/signtool)、[验签说明](https://learn.microsoft.com/en-us/windows/win32/seccrypto/using-signtool-to-verify-a-file-signature)。Inno 集成还须按所用固定编译器版本核对实际签名调用；上述方案尚未作为已运行工作流宣称完成。

## 第 8 节回信草稿（未发送，性能结果仍待候选验证）

Hi CJ,

Thank you for the report and the complete appendix. The reported completion totals and the main comparisons are consistent with our recalculation of the rounded appendix cells. That checks internal consistency; we cannot reconstruct the original per-request percentiles from rounded summaries alone.

1. **The SM89 median:** the difference does not yet establish a GPU-architecture cause. Your test compares two releases on short prompts, while our main 5090 table compared numerical profiles in one binary on ShareGPT. The warm-up and percentile aggregation also differ. Our separate short-prompt 5090 diagnostic did show a C16 median improvement, from 112.37 to 86.77 ms. Prefill taking a larger share on the 4070 Ti is a plausible hypothesis, but we have not measured that decomposition on your card.

2. **The remaining first-token gap:** yes, it is on the optimization list. We are separating queueing, prefill, first-token sampling and delivery costs. We are not treating the 25–31% median gap as an unavoidable price for throughput. Candidates must retain the workload's latency limits and report throughput, tail latency and errors together.

3. **Single-request throughput:** yes. Your C1 first-token result is already close to llama.cpp, so the remaining throughput gap calls for decode investigation. Our 5090 diagnostics have identified a separate context-boundary replay issue and a candidate is being validated. This is not yet a fix or reproduction of your short-prompt 4070 Ti result; we will keep those claims separate.

4. **Signing:** we plan to sign Ferrum's Windows executable and installer, including the launcher and uninstaller chain. v0.19.1 remains unsigned. We have identified the build and verification steps that need to change, but a production signing identity is not configured, so we cannot promise a signed release date yet.

We are also following up on the temperature-0 cross-batch text differences. Different batch shapes can select different floating-point paths, but that explanation is not a substitute for checking output correctness and request-state isolation.

Your same-session 1.25×/1.31× first-token ratios should remain separate from the approximately 2.5× ratio in the earlier session. Your distinction between maximum latency and our per-repetition P99 aggregation is useful and should stay in the report.

Best,
Jinxuan
