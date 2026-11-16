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

2026-11-17：视频目标已完成，本目标进行中。单请求注意力拓扑修复与显式 `within-budget` 准备选项已通过 PR #517 合入主线，v0.20.0 冻结源码为 `a08e735d3d1da090bb858fecffb54868513a9836`，已包含后续独立 Cargo 包修复。最终包功能验收与公开安装验收分别记录。r5b 在该显式选项下通过 5090 固定 ShareGPT 负载的三轮配对 SLO 资格，1152 次计时请求零错误；默认仍为按需分配。新 Linux SM120 包（二进制 SHA-256 `f4988a7fd24f466d3a9927a0221a82db78037f5e70910f6c7d8df703e72cc506`）已重新通过 35 项文本功能案例，以及显式准备选项的 6 项功能案例；这些验证不替代性能配对实验，也不沿用旧候选的包验收。4070 Ti Windows、外部短提示词差距及新样本 llama.cpp 主表仍未完成；Windows 签名按用户明确决定延期。正式 v0.19.1 Linux SM120 二进制重新计算的 SHA-256 为 `377e6ce053cc2f55ccdd147d941603b66bebac27d262fff42021cf2fc7339387`；模型与数据集的实际字节均已核对。原主工作树未修改。性能证据保存在库外 `/Users/chejinxuan/ferrum-handoffs/20261116-windows-v0191-followup`，最终包验证保存在 `20261116-release-v0200` 及 5090 对应目录。

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

分块的三次配对吞吐变化约为 −6.97%、+0.75%、−2.77%，没有稳定收益且仍一次超限，暂不作为推荐配置。停止继续盲调分块参数，也不更改预声明 SLO。

### CUDA 分配停顿与分页池候选（进行中）

已有 Nsight Systems 2024.6.2 对当前 CUDA 13.3 驱动提示不支持；本次采到了主机端 CUDA API，但没有 GPU kernel activity。对齐 HTTP 计时窗口后，发现 cuMemAllocAsync 单次耗时 147.48ms、484.92ms、1500.57ms。首批慢请求的首字窗口包含前一种慢分配。跟踪结果只用于诊断；嵌套 API 耗时不能相加当作 GPU 执行时间。

随后取消 Nsight，只启用分配器目标的 DEBUG 计时，64/64 请求完成、零错误，TTFT P99 536.64ms、TPOT P99 28.21ms、可见 SSE ITL P99 87.05ms、吞吐 996.15 tokens/s。这轮计时窗口内有 54 次物理分配，最慢的两次为约 2MiB / 1842.50ms、6MiB / 437.47ms；同步部分各约 11μs，主要时间在原生分配调用。两次均属于 F16 分页序列状态池，驱动 reserved/used 同时增长。带 DEBUG 的数值仍不作为无诊断性能资格。

第一个分页池候选 r3b 沿用公开的 PagedGroup4GrowOnly 策略：每序列仍只领取四页组，物理池在预算允许时按既有驻留量成倍增长，额外页保持共享空闲容量。池上限按页对齐裁剪；完整设备预算预留被拒绝时，在原生分配前回到精确组增长，之后仍保留原有按实际需求维护的路径。不改全局 CUDA 池阈值、采样、模型算术或服务容量。更少的扩容可能换来更多驻留显存，必须随延迟和吞吐一起实测。

库外 candidate-r3b.source.json 固定该候选的源码身份，amortized-pool-r1.contract.json 固定相同 SLO 和三组配对规则。9 项分页增长定向用例通过，受影响 runtime crate 的 976 项通过、5 项忽略，覆盖共享页所有权、设备预算及池上限、所需容量的正常维护重试。首次新增边界用例遗漏了原有维护重试流程而失败，补齐测试调用后通过；未修改生产实现来绕过此容量边界。CUDA CLI 构建、格式检查及 run/serve 的 11 项产品阶段通过；二进制 SHA-256 为 82158dc19903a37b30f244eb71c10a1faf25c3e80e6ed451441bbf00f827bcaf。原始 Nsight、分配日志与摘要保留在 c16-nsys-r1 和 c16-alloc-debug-r1。

首个孤立 C16 pilot 的 64 个请求全部完成、零错误，TTFT P50/P99 为 74.60/279.11ms、TPOT 为 11.04/13.19ms、可见 SSE ITL 为 6.67/75.60ms，吞吐 1308.36 tokens/s。结束时该分页状态池为 22 块 / 256MiB；此前未修改的独立运行是 110 块 / 181.5MiB。整卡采样峰值 6.022GiB、RSS 1.199GiB；GPU 95 个样本、4 次查询错误、最大间隔 2002ms，仍不是精确进程峰值。单轮通过没有替代重复验收。

三组配对按预声明的失败即停规则，在第 2 组候选失败后停止，实际完成 3/6 个 arm、576/576 请求、零请求错误。第 1 组 C16 控制/候选吞吐 1354.12/1372.09 tokens/s、TTFT P99 303.67/268.66ms；第 2 组候选为 1329.28 tokens/s，TTFT P99 **505.97ms，失败**，TPOT/可见 SSE ITL P99 为 12.73/76.65ms。最慢请求是 27-token 输入、首个可见文本 525.20ms，同批多数请求约 494.5ms；不能把超过门槛的数值四舍五入为通过。C1/C8 均通过，剩余 3 个 arm 未执行；完整逐请求、分位数、资源和输入一致性证据在 amortized-pool-paired-r1.summary.json 及对应目录。未发布或将该候选记为性能资格通过。

合同 amortized-bounded-c16-r1.contract.json 保留所有原门槛和固定服务容量，在同一候选上加已有公开的 256-token prefill step chunk / active-decode prefill token budget。三次独立 C16 启动验证全部通过：TTFT P99 为 303.56/301.66/422.30ms，TPOT P99 为 12.71/12.65/12.58ms，可见 SSE ITL P99 为 75.90/75.19/76.75ms，吞吐 1328.28/1346.04/1317.70 tokens/s，192/192 请求完成、零错误。复用 Rust replay 工具检查 1093-token 检索请求，C1 与 C8 共 9 个响应都为 violet-583，HTTP 200、自然 stop、usage 和流传输完整；相同请求的重放不作为不同状态隔离的证据。

但随后完整 C1→C8→C16 配对仍失败：第 1 组候选 C16 的 TTFT P99 为 **615.20ms**、TPOT P99 为 12.76ms、可见 SSE ITL P99 为 72.91ms，吞吐 1300.34 tokens/s；对应控制为 262.02ms、14.91ms、76.35ms、1318.96 tokens/s。按冻结规则停止在 2/6 个 arm，384/384 请求无错误，剩余 4 个未执行。孤立 C16 的三轮通过没有覆盖进程内先前执行历史，组合方案仍未获得资格，不能作为推荐配置发布。

完整历史的 basic 诊断中，最慢首字窗口内的物理分配最大约 2.1ms；该次 C16 吞吐降至 383.80 tokens/s，诊断本身影响明显。冷路径 ledger 的前 4096 条已被较早阶段占满，后续 12984 个 wave 溢出，不把缺失记录解释为没有图准备。

随后使用 latency profile 和现有 ferrum.cuda.replay_preparation 目标日志：较大的图实例化耗时（最大 787.24ms）全部在启动期；C16 计时阶段只有一次约 5.23ms 的捕获/拓扑检查/实例化/上传组合，未落在两个最慢请求的首字窗口中。不能用启动期的慢调用解释计时阶段 TTFT。该诊断 C16 吞吐 346.51 tokens/s、TTFT P99 752.59ms，输出约 1GB 的 profile/scheduler 日志；去掉 scheduler trace 后仍为 338.72 tokens/s、745.48ms，说明并非仅重复写文件造成。详细诊断记录了 C16 decode 的 catalog_empty 回退，原无跟踪失败轮次没有同类已记录行；这些诊断不替代无跟踪性能资格，也不据此修改捕获策略。

继续用已有基础计数器和原生调用目标日志缩小范围，关闭执行事件文件；同时检查原无跟踪失败轮次已经保存的阶段计数，避免重复采集。未扩大固定 ledger，也未改变 SLO。此候选的 Metal 全 targets 编译已通过，Linux 整仓测试及 Clippy 仍未重启。新证据位于库外 amortized-bounded-history-latency-r1、amortized-bounded-history-latency-lite-r1 及 aligned-preparation.json。

### 统一物理池维护候选 r4（验证中）

原无跟踪的组合失败轮次已保存的粗粒度计数显示，prefill 的 resource_prepare_attempt 最大 318.34ms。改用 Basic 纯计数器、不写执行事件文件后，完整 C1/C8/C16 扫描恢复正常吞吐：212.52/967.13/1278.83 tokens/s，TTFT P99 157.26/171.25/434.06ms，192/192 请求、零错误、该轮 SLO 均通过。但这仍是带目标 DEBUG 的诊断，不能替代无跟踪配对验收。运行期原生分配日志中，激活池、分页状态池、工作区分别出现 390.34ms、337.04ms、162.78ms 的分配调用；同步各仅数微秒。最大调用的精确计时/预热窗口尚未由请求级记录逐个对齐，不将它们全部称为计时请求的停顿。

r3b 只覆盖分页序列扩容，未覆盖临时激活和 Invocation 工作区，因此 r4 将物理池余量从每序列预留策略中移出。统一的 DemandDrivenElastic 维护对重新验证后的自动增长尝试共享几何余量，适用于分页和连续存储；显式 grow_pool 和最低可运行容量仍按请求大小。全部池的预算预留与驻留上限检查先于任何原生分配；可选余量不被预算接受时先回滚，再重新验证精确需求。只有原生分配前的容量拒绝允许此重试，后端分配错误继续保留原有错误路径。Sequence/Request/Step/Invocation 的逻辑领取量、请求容量、模型计算和调度配置不变。余量可能增加驻留显存，必须继续检验受限预算下的正常维护和回收，以及完整历史的延迟与吞吐。

候选源码基于 7bbea0287 加 C1 修复及此次池维护改动，逐版源码和构建记录冻结在库外。r4 首次新增测试因夹具使用了非十六进制指纹及不支持的 lifetime，在进入生产行为前失败；r4b 仅修正夹具后，两项连续池余量用例和 9 项分页增长用例通过。受影响 runtime crate 为 973 通过、5 失败、5 忽略，未跳过失败继续构建。

其中 3 项回收用例暴露了实际设计问题：容量回收后的重试又加上几何余量，会重新填满刚释放的空间。r4c 将前台和 checkpoint 回收后的重试都保持为精确需求，继续重新验证打包和预算；这 3 项原测试的期望保持不变。另两项逻辑容量测试不应要求物理增长恰好一页，改为验证足够的增长、预算增量、checkpoint 保留和逻辑租约不变。candidate-r4c.source.json / changes.tar.gz 已冻结（归档 SHA-256 8caa64f9733a5702414351a2b699cef4a6eed25cfa60a93be9e6176bf08547da）。两项连续池、9 项分页增长检查通过；runtime 单元测试 978 通过、0 失败、5 忽略，CUDA CLI 构建通过，二进制 SHA-256 120fcfe21136aff7b34c5766e8edec4bd9827d8f73dfa23e75d338ef6f561d54。格式和 diff 检查通过；新版 Metal 编译和整仓检查尚未执行。

run/serve 的 11 项 Rust 产品检查全部通过。随后直接冻结 unified-pool-paired-r1 的三组全历史 C1→C8→C16 配对，不增加孤立 pilot。两侧均保留 C1 修复、默认 prefill 配置及原固定容量，候选只改变物理池维护；顺序 candidate→control、control→candidate、candidate→control，每格 64 计时样本、10 预热。保持原 SLO 和失败即停规则，逐格及跨并发核对样本和 usage，保留所有请求/协议质量计数。

**该轮仍未获得资格：第二组候选 C16 TTFT P99 为 824.34ms，停止于 4/6 个 arm。** 实际 768/768 计时请求完成，零请求/协议错误；每格 12138 output usage tokens，选样及输入/输出长度数组跨全部已执行格一致。以下 TPOT 为最后可见文本口径，ITL 为可见 SSE 文本事件口径；每行 n=64。

| 配对 / 实现 | 并发 | TTFT P50 / P99 ms | TPOT P50 / P99 ms | 可见 SSE ITL P50 / P99 ms | 输出 tokens/s | SLO |
| --- | --- | --- | --- | --- | --- | --- |
| 1 / candidate | 1 | 45.53 / 100.59 | 4.33 / 4.83 | 4.21 / 4.81 | 216.00 | pass |
| 1 / candidate | 8 | 52.97 / 187.40 | 7.65 / 8.92 | 5.55 / 67.57 | 968.07 | pass |
| 1 / candidate | 16 | 75.08 / 271.43 | 11.10 / 13.76 | 6.65 / 78.84 | 1324.33 | pass |
| 1 / control | 1 | 45.43 / 101.34 | 4.33 / 4.74 | 4.21 / 16.53 | 216.79 | pass |
| 1 / control | 8 | 63.45 / 285.32 | 7.82 / 12.57 | 5.55 / 70.17 | 889.32 | pass |
| 1 / control | 16 | 77.06 / 788.24 | 11.71 / 21.29 | 6.70 / 82.19 | 1120.54 | fail |
| 2 / control | 1 | 45.80 / 101.01 | 4.38 / 4.86 | 4.23 / 6.27 | 214.47 | pass |
| 2 / control | 8 | 52.58 / 405.67 | 8.06 / 13.11 | 5.54 / 53.25 | 854.56 | pass |
| 2 / control | 16 | 75.17 / 417.49 | 10.36 / 17.90 | 6.68 / 76.51 | 1218.30 | pass |
| 2 / candidate | 1 | 46.10 / 103.89 | 4.35 / 4.76 | 4.23 / 5.92 | 214.60 | pass |
| 2 / candidate | 8 | 70.39 / 176.26 | 7.94 / 8.96 | 5.59 / 60.53 | 939.48 | pass |
| 2 / candidate | 16 | 76.09 / 824.34 | 11.07 / 13.45 | 6.71 / 76.90 | 1266.47 | fail |

两组 C8 吞吐分别提高 8.86% / 9.94%，但第二组 TTFT 中位数变差，不能声称首字全面改善。C16 吞吐分别提高 18.19% / 3.95%，仍不能抵消第二组尾延迟失败。C1 吞吐变化 −0.37% / +0.06%。第 3 组未执行，不报告完整三组资格或统计置信结论。

| 配对 / 实现 | 设备采样峰值 GiB | RSS 采样峰值 / 生命周期 HWM GiB | GPU 样本 / 错误 / 最大间隔 ms | Host 样本 / 错误 |
| --- | --- | --- | --- | --- |
| 1 / candidate | 6.054 | 1.227 / 3.583 | 286 / 9 / 2204 | 375 / 0 |
| 1 / control | 5.991 | 1.226 / 3.582 | 286 / 8 / 2005 | 387 / 0 |
| 2 / control | 6.304 | 1.255 / 3.589 | 284 / 9 / 2204 | 387 / 0 |
| 2 / candidate | 6.054 | 1.225 / 3.585 | 275 / 9 / 2201 | 375 / 0 |

WSL 仍无目标进程显存归属，设备采样有缺失，峰值只是观测下界；RSS HWM 包含启动。严格单 token ITL 资格不足不抹去可见 SSE 间隔；逐请求事件数与 usage token 数的差异保留在原始 text_timing 中。

失败候选已有的计数显示 prefill resource_prepare_attempt 最大 588.88ms。后续 unified-pool-alloc-http-r1 / r2 只开 Basic 计数器、原生分配目标和现成 tower_http::trace 请求开始日志，不写逐算子事件；每轮 222 个 HTTP 请求与每格 10 次预热 / 64 次计时完整对应，日志各约 180KB。两轮诊断 C16 分别为 1352.10 / 1359.44 tokens/s、TTFT P99 264.84 / 293.26ms，不能代替无跟踪失败。对齐显示约 199 / 510ms 的激活分配发生在 C16 预热；计时阶段均仅剩第一批请求中的一次约 128MiB 分页状态分配，分别耗时 0.85 / 9.36ms。第二轮 C1 计时阶段还出现一次约 84MiB / 155.57ms 激活分配。它们证明仍存在运行期冷扩容，尚未证明失败轮次的 588.88ms 就来自某一次具体调用，不把预热停顿计入正式 TTFT。

第二轮客户端、服务停止及资源监控均正常结束；整理工具读取资源摘要时碰到文件尚未写完，报 JSONDecodeError。随后只读取同一次已完成的原始记录补齐摘要，没有重新运行或修改计时。该恢复说明保存在 orchestration-result.json。

停止继续碰运气重跑。下一步复用 MemoryPlan::startup_workload_peak_bytes 和现有维护事务，为声明的服务容量准备物理 backing：逻辑上限、实际领取量、设备预算和可回收下限必须分开；可选准备不能变成新的不可回收最低驻留量，也不能因预算不足拒绝原本可按需运行的服务。准备量须来自编译后的容量公式及公开配置，不能写死 C16、128MiB 或测试样本。必须同时披露就绪时间和驻留显存代价，重新做 run/serve 及同配置配对验证。r4c 诊断结束时该设计尚未实现，后续进展见下节。

本轮源码、测试、产品检查日志、四个配对 arm 和两个轻量诊断的 192 份小文件已归档为库外 unified-pool-r4c.evidence-r1.tar.gz（3998699 bytes，SHA-256 53664f92e1aa310b8074023f6b29fa7c0cc15f38dd78f273c5965b4eef225969），已回传本机并逐文件校验。产品 verifier 的完整 report.json 约 53MB，保留远端，哈希记入同包 manifest；没有复制模型或二进制进仓库。工作树源码与实际测试的 r4c 冻结清单再次逐文件核对一致，候选仍未提交为已验收的性能修复。

### 启动容量准备 r5b：限定负载三轮资格通过，默认行为未改

新增公开选项 `--runtime-pool-preparation on-demand|within-budget`，`run` / `serve` 共用；配置文件对应 `[runtime] runtime_pool_preparation = "within-budget"`。默认仍为 `on-demand`，不通过隐藏环境变量启用。编译后的 MemoryPlan 给出每个物理池的目标，现有维护事务只补足当前驻留量与目标之间的差额，不修改请求逻辑上限、领取量或可回收下限。

可选准备在原执行器的 startup 生命周期内、Ready 之前执行。总目标超过预算，或完整批次在原生分配前遇到类型化容量拒绝时，继续原有按需分配；不为准备目标驱逐现有 owner。原生分配或发布错误进入原有启动失败与资源清理流程，不伪造准备成功。

r5 初轮夹具把同时违反总预算和池上限的请求误写为只应返回池上限错误；实际优先返回类型化总预算不足且无原生分配，生产逻辑未改。修正夹具以充足总预算单独验证池上限，冻结 r5a 的 28 份 Rust 源文件，库外 archive SHA-256 `9c7eb60d8eb57fcae6b4b469298ab1fcd87e84a5854f2777ef6b0cca589c6ea3`。

r5a 已通过 2 项绝对目标/批次拒绝/失败回滚测试、5 项启动容量公式测试、1 项准备空间回收测试和 2 项类型化配置测试。首次 CLI 检查命令遗漏启用 native-op-artifact 的产品特性，构建脚本在测试前拒绝；改用产品特性后，编译发现共享启动异步块需要显式 `Result` 类型。r5b 仅补这一类型标注，增量源码 archive SHA-256 `ad24e9a5fb217e0927f744234c412a0af816bdfda2a3e7c01d04eaa2fcf920ab`，完整清单仍验证 28 份 Rust 文件。两次失败日志均保留。

r5b CUDA 产品构建通过，二进制 SHA-256 `668035c8c2e28b7b85cfd4aec6c5614b178a84a7dae7492887d17e34515ab964`。CLI 首次选中 bin 目标，实际运行 0 项测试，只计编译；随后正确选择 lib，4 项选项解析/配置优先级/来源测试通过。Rust model_regression 新增的参数传递与生效核验测试 1 项通过。连同源码未变的 10 项 Core/types 检查，共 15 项受影响检查；格式检查通过，未执行新全仓或 Metal 检查，不宣称 PR 已完成全部验证。

prepared-pool-product-r1 的 11 项真实 CUDA 功能验证全部通过，包括 run/serve、历史/重置、C2 隔离、长上下文检索和关闭；验证器 SHA-256 `fb5e293604f44d49f38902d37573e72ca76faba5dd29699f555cd8cb198c675c`。四个产品进程均记录实际 `prepared`，serve 的计划总占用 7326820016 bytes（约 6.82GiB），低于 10066329600 bytes 预算。实际 GPU 物理峰值仍须另测，不能用计划字节代替。

prepared-pool-paired-r1 在发出请求之前，因采集工具过早核验 systemd MainPID 对应的可执行文件而停止，0 个计时请求，不属于推理或 SLO 失败。保留原记录；工具改为有界等待该 PID 实际执行冻结二进制后再启动内存监控，产品代码未改。

prepared-pool-paired-r2 已冻结合同并开始配对比较同一个 r5b 二进制的 `within-budget` 与 `on-demand`，只改变公开准备策略，维持原有 32 序列、2048 context/batch、10066329600 bytes 预算、ShareGPT 样本和三项 SLO。除推理吞吐/延迟外记录 HTTP health 暴露执行器 startup Ready 的时间及从启动开始的内存观测；功能验证通过后才启动性能验收，部分重复或旧诊断通过都不算新候选完成。

六个 arm 全部完成：1152/1152 次计时请求成功、180 次非计时预热，样本身份、实际输入 usage 和输出长度数组跨实现及并发格一致，每格 n=64、输出 12138 usage tokens。candidate 的九个单元均满足 TTFT/TPOT/可见 SSE ITL 的 P99 SLO；control 的三次 C16 均为 TTFT 超限，客户端退出 1 属于 SLO 失败，非请求错误。三次配对不报告统计置信区间，也不据此承诺任意负载或 Windows 4070 Ti 的时延。

| 配对 | 策略 | C | TTFT P50 / P99 ms | TPOT P50 / P99 ms | 可见 SSE ITL P50 / P99 ms | tokens/s | SLO |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | candidate | 1 | 47.91 / 100.52 | 4.33 / 4.76 | 4.21 / 5.51 | 215.81 | pass |
| 1 | candidate | 8 | 51.02 / 171.16 | 7.69 / 8.70 | 5.53 / 50.32 | 966.46 | pass |
| 1 | candidate | 16 | 64.10 / 296.63 | 10.67 / 13.25 | 6.71 / 76.55 | 1347.38 | pass |
| 1 | control | 1 | 46.96 / 100.66 | 4.35 / 4.80 | 4.22 / 4.89 | 214.79 | pass |
| 1 | control | 8 | 48.74 / 194.71 | 7.56 / 9.01 | 5.57 / 48.43 | 972.65 | pass |
| 1 | control | 16 | 76.09 / 1329.48 | 10.63 / 12.55 | 6.69 / 78.13 | 1207.07 | fail |
| 2 | control | 1 | 46.48 / 109.85 | 4.36 / 4.76 | 4.23 / 4.93 | 215.07 | pass |
| 2 | control | 8 | 51.84 / 172.83 | 7.65 / 8.56 | 5.54 / 50.58 | 979.33 | pass |
| 2 | control | 16 | 52.58 / 610.20 | 11.29 / 13.12 | 6.72 / 71.98 | 1243.46 | fail |
| 2 | candidate | 1 | 48.24 / 100.55 | 4.35 / 4.79 | 4.21 / 5.08 | 215.55 | pass |
| 2 | candidate | 8 | 48.30 / 205.48 | 7.63 / 8.91 | 5.53 / 49.77 | 969.10 | pass |
| 2 | candidate | 16 | 76.74 / 244.50 | 10.87 / 13.06 | 6.75 / 80.93 | 1334.89 | pass |
| 3 | candidate | 1 | 45.44 / 100.77 | 4.36 / 4.78 | 4.21 / 5.92 | 215.97 | pass |
| 3 | candidate | 8 | 51.23 / 172.75 | 7.73 / 8.65 | 5.55 / 52.18 | 959.18 | pass |
| 3 | candidate | 16 | 77.73 / 272.68 | 10.69 / 13.34 | 6.66 / 78.82 | 1339.54 | pass |
| 3 | control | 1 | 46.60 / 100.56 | 4.35 / 4.87 | 4.22 / 4.84 | 214.48 | pass |
| 3 | control | 8 | 58.79 / 177.23 | 7.63 / 9.27 | 5.55 / 61.25 | 958.01 | pass |
| 3 | control | 16 | 78.65 / 2016.85 | 11.55 / 20.94 | 6.68 / 76.78 | 1121.49 | fail |

C16 吞吐配对变化 +11.62%、+7.35%、+19.44%；C1 为 +0.48%、+0.22%、+0.70%，C8 为 −0.64%、−1.04%、+0.12%。收益集中在 C16 长尾及其吞吐，不能概括为全部 TTFT 改善：第二组 C16 中位数反而从 52.58 增至 76.74ms。C8 中位数和尾延迟也有双向波动，外部短提示词的中位数差距尚未关闭。

| 配对 / 策略 | health Ready 秒 | 设备总显存峰值 GiB | RSS 峰值 / HWM GiB | GPU 样本 / 错误 / 最大间隔 ms | Host 样本 / 错误 |
| --- | --- | --- | --- | --- | --- |
| 1 / candidate | 13.526 | 7.647 | 3.584 / 3.599 | 310 / 10 / 2190 | 429 / 0 |
| 1 / control | 15.157 | 6.054 | 3.592 / 3.594 | 311 / 11 / 2177 | 436 / 0 |
| 2 / control | 14.766 | 6.054 | 3.579 / 3.593 | 306 / 10 / 2005 | 434 / 0 |
| 2 / candidate | 13.179 | 7.647 | 3.586 / 3.606 | 293 / 12 / 2166 | 428 / 0 |
| 3 / candidate | 14.354 | 7.647 | 3.578 / 3.583 | 288 / 14 / 2204 | 432 / 0 |
| 3 / control | 14.911 | 6.054 | 3.578 / 3.593 | 308 / 12 / 2227 | 447 / 0 |

显存观测增加约 1.59GiB。监控含启动，WSL 无目标进程显存归属且 GPU 有漏采，峰值为设备总量的观测下界。health 每 2 秒轮询，表中为观测到 Ready 的时间，不据此宣称精确的启动提速。candidate 三轮运行前后的九个池驻留量均相同，总预算领取量保持 7326820016 bytes；control 从 4989733328 增至 5608769360 bytes。这是首尾状态证据，不冒充逐次分配跟踪。每格有 2–5 个请求的可见事件数与 usage 不同，严格单 token ITL 仍不可用；没有缺失成功请求的 SSE 证据或被丢弃的可见文本停顿，客户端未标记传输合并。

另以 prepared-pool-fallback-r1 把预算降至 6442450944 bytes（6GiB），低于完整准备目标，保持上下文和并发上限。四个进程均记录超预算后 `on_demand` 回退，run/serve 的 11 项功能验证仍全部成功。容量准备不足未导致拒绝启动；不吞掉真正原生执行错误。候选稳定后只补跑受影响 runtime crate：980 passed、0 failed、5 ignored，执行 0.21 秒；未跑全仓检查。

源码、失败及通过的检查日志、两次功能验证的小文件和完整配对证据已在 5090 归档为 `prepared-pool-r5b.evidence-r1.tar.gz`，5147688 bytes，SHA-256 `f966cfa79c53fa94df08fb2245b8cf2425b29999d8cc9d096b9cf86d5c04ca58`，含 manifest 共 290 项。归档已回传本机，289 份内容逐文件哈希及当前 28 份源码均已核对。两个约 53MB 的产品完整 report.json 保留远端并在 manifest 记录哈希；模型、二进制和原始结果不进仓库。测量服务已停止，GPU 回到约 84MiB 空闲占用。

### 仍需完成

1. 完成新样本、足量重复的同机 llama.cpp 主对照和外部短提示词差距分解。r5b 仅在显式准备选项及当前 ShareGPT 负载通过资格；默认仍为 on-demand，是否推广为默认策略、最终集成和发布验证未完成。C1 回退改善不等同于关闭外部短提示词 C1 差距。
2. C8/C16 的外部短提示词 TTFT 分解。5090 跨批次文字差异和状态隔离已做局部实测，仍不承诺跨批次字节一致或代替外部原始请求复测。
3. Windows 4070 Ti 实机复测和原始请求数据。5090 WSL2 诊断不能代替这一步。
4. Windows 正式签名身份和验签闭环。仓库级 secrets 未发现签名凭据，仓库 variables 和 environments 列表为空；现有流程仅验第三方安装器签名。尚未配置或使用任何正式 Ferrum 签名身份。
5. 第 8 节四问的回信草稿已备，性能结果确定后更新；不发送邮件，不把计划描述成已发布功能。

## Windows 签名接入方案（身份未配置，尚未执行）

2026-11-16 发布准备补充：用户确认目前没有代码签名证书或云签名身份，随后明确本次不做签名。该项按用户决定延期，不再作为本次发布前置条件，不能标为已解决；Windows 包继续明确未签名。此前仅写方案、未完成流水线接入。视频与文本分支已在独立 `release/fast-h3-v0200-20261116` 工作树无冲突合并；联合验证和正式发布尚未完成。

已核查 [SignPath Foundation 条件](https://signpath.org/terms) 和 [GitHub 接入要求](https://docs.signpath.io/trusted-build-systems/github)：免费开源项目签名需要项目审核、每次签名人工批准，签名前工作流要求使用 GitHub 托管 runner。Ferrum 是公开 MIT 项目且已有 Windows 发布，但当前 CUDA Windows 构建和部分前置验证使用自托管 runner，因此尚不能宣称符合免费服务要求。CUDA/VC 运行库的分发资格也须按服务的 System Libraries 例外确认。不能仅把自托管构建产物转存到 GitHub 托管 runner 来绕过来源校验。

可供提交的申请说明如下；尚未发送、未注册账户或接受服务条款：

> Project: Ferrum — a Rust-native local inference engine with CLI and HTTP serving. Repository: https://github.com/sizzlecar/ferrum-infer-rs (MIT). Project website: https://ferrum.pandaailabs.com/. Existing Windows releases include CPU and CUDA SM89 portable ZIPs and Inno Setup installers. We seek signing coverage for our own engine executable, stable launcher, installer and uninstaller; bundled Microsoft/NVIDIA runtime DLLs retain their original vendor signatures. Our Windows CUDA builds currently use a self-hosted GitHub Actions runner. Before applying, please confirm the required hosted-build arrangement and eligibility of the bundled runtime libraries. We can provide the immutable source, build logs and artifact provenance. Maintainer contact, signing approver and final project privacy/code-signing policy will be supplied by the repository owner.

后续接入沿现有产物链完成：签署核心 EXE → 生成 portable manifest/hash → 验证并共享唯一已签 launcher → 编译时签署 Inno Setup/卸载程序 → 验签、生成最终收据 → 最终解包与安装验证。身份、来源要求和证书发行者未确认前，不填造组织 ID、凭据或 signed 状态；发布说明须继续明确未签名。

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

2. **The remaining first-token gap:** yes, it is on the optimization list. A 5090 candidate that prepares physical pools before readiness has now completed three paired ShareGPT sweeps: all 1,152 timed requests completed, and its C16 TTFT P99 was 245–297 ms versus 610–2,017 ms with on-demand allocation in the same binary. The tradeoff was about 1.6 GiB more observed device memory; C8 throughput moved between −1.0% and +0.1%, and median TTFT did not improve consistently. This is an explicit option in an unpublished candidate, not a reproduction or resolution of your Windows short-prompt median gap. We are not treating that 25–31% gap as unavoidable.

3. **Single-request throughput:** yes. Your C1 first-token result is already close to llama.cpp, so the remaining throughput gap calls for decode investigation. Our 5090 diagnostics identified and corrected a separate context-boundary replay issue; the latest ShareGPT candidate sustained about 216 output tokens/s at C1. We still need the full matched llama.cpp comparison for these samples, and this does not establish a fix for your short-prompt 4070 Ti result.

4. **Signing:** we plan to sign Ferrum's Windows executable and installer, including the launcher and uninstaller chain. v0.19.1 remains unsigned. We have identified the build and verification steps that need to change, but a production signing identity is not configured, so we cannot promise a signed release date yet.

We are also following up on the temperature-0 cross-batch text differences. Different batch shapes can select different floating-point paths, but that explanation is not a substitute for checking output correctness and request-state isolation.

Your same-session 1.25×/1.31× first-token ratios should remain separate from the approximately 2.5× ratio in the earlier session. Your distinction between maximum latency and our per-repetition P99 aggregation is useful and should stay in the report.

Best,
Jinxuan
