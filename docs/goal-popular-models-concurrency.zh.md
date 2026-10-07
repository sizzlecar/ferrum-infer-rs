# Ferrum 目标 v2：热门模型的并发吞吐与完整 SLO-aware

制定于 2026-10-07。本文件承接 [`goal-slo-throughput.zh.md`](goal-slo-throughput.zh.md)（已随 #402 合入 main），沿用其中的公式、指标口径和"按并发扫描"的测量方式。

执行进度见 [P0 结果页](performance-popular-models-p0.zh.md)。尚未完成 P0，不将格式支持或基线准备当作吞吐验收。

## 1. 目标

按顺序完成三层目标，模型范围为第 3 节。Metal 只对照 llama.cpp；vLLM 仅在 CUDA 上测试，第 2 层及第 3 层的 vLLM 比值仅在 CUDA 验收。

1. **并发超过 llama.cpp**：同机、同一模型文件、同一负载下，Metal 的 C = 4、8、16，CUDA 的 C = 4、8、16、32，每一档的输出吞吐都高于 llama.cpp。
2. **CUDA 吞吐达到 vLLM 的 80%**：CUDA 上 C = 8、16、32 每一档的输出吞吐都不低于 vLLM 的 80%。
3. **完整可用的 SLO-aware**：在适用的前两层目标基础上，用户只给出模型和阈值 S，Ferrum 自动选好配置；两个后端按公式算出的得分 G 不低于手调最优配置，CUDA 还须不低于 vLLM 同条件 G 的 80%；默认命令直接达到，不需要手写参数。

公式与旧文档相同：

```text
G(θ) = max over C of  成功请求的输出 tokens / 测量时间
       约束：P99(TTFT) ≤ S_TTFT，P99(TPOT) ≤ S_TPOT，P99(可见 SSE 文本事件 ITL) ≤ S_ITL，错误 = 拒绝 = 0
目标：max over θ of G(θ)
```

第 1、2 层比较的是固定并发下的吞吐，每格同时报告三项延迟的 P50/P99，以防用延迟换吞吐。第 3 层按公式验收。

## 2. 目标机器

| 后端 | 机器 | 说明 |
| --- | --- | --- |
| CUDA | RTX 5090 32GB（WSL 主机） | 与 waifugemma4 服务共用，测量时由现有 guard 停止并恢复该服务 |
| Metal | M1 Max 32GB（本机 MacBook Pro） | 测量期间本机不能有其他重负载。M4 Mac mini 只有 16GB，只用于 Qwen3.5-9B 回归 |

同一台机器上，所有引擎都用同一个模型文件。可用的设备内存以 Ferrum 启动时探测到的值为准，不靠修改系统内存上限来塞下模型。

## 3. 模型名单

选择标准：
1. 2026 年 10 月的热度，依据 Hugging Face 30 天下载量、Ollama 累计拉取量和第三方 RTX 5090 选型指南的共识。
2. 4bit 权重加上并发所需的 KV，在两台 32GB 机器上都放得下。
3. 覆盖 dense、MoE 和混合注意力三类架构。
4. llama.cpp 在两后端支持，vLLM 在 CUDA 支持；不要求 vllm-metal。

考虑时间成本，这一版只做两个目标模型。两者同属 Qwen3.5 架构族，dense 和 MoE 各一个，共用 GDN 和注意力内核；Ferrum 在两个后端都已支持，不需要先补架构。

| 模型 | 架构 | 参数 | 热度（HF 官方 30 天下载 / Ollama 累计拉取） | 4bit 文件大小 | Ferrum 现状 |
| --- | --- | --- | --- | --- | --- |
| **Qwen3.8-27B** | dense，GDN 与全注意力混合（qwen35） | 27.3B | 677 万（另有 FP8 版 479 万、unsloth GGUF 657 万，GGUF 下载第一；官方仓库 1.7 万赞）；qwen3.8 330 万（8 月发布） | Q4_K_M 16.5GB；IQ4_XS 14.3GB | 架构已支持。CUDA 的 AWQ INT4 在 4090 上 c16 到 c32 都停在约 115 tok/s；Metal 只验证过三值量化版 Bonsai 2 |
| **Qwen3.6-35B-A3B** | MoE 与混合注意力（qwen35moe），激活约 3B | 34.7B | 348 万（另有 FP8 版 496 万、NVIDIA NVFP4 版 573 万）；qwen3.6 700 万；第三方 5090 选型指南普遍首推 | Q4_K_M 22.1GB；Q4_K_S 20.9GB；IQ4_XS 17.7GB | 支持 Qwen3.5-35B-A3B（同架构）。L40S 上 FP8 版从 c8 到 c32 都停在约 93 tok/s |

M1 Max 上 Qwen3.6-35B-A3B 用 Q4_K_S 或 IQ4_XS，Qwen3.8-27B 用 Q4_K_M；Ferrum 与 llama.cpp 用同一个文件。

Qwen3.5-9B（同架构，当前优化最充分：5090 上静态配置 C4 达标时 285 tok/s）只作回归：每次合并前跑 C4/C8，不低于 #402 的结果，不进入验收矩阵。

**推迟到下一版：**
- Gemma 4 26B-A4B（HF 官方下载第一，1263 万）：Ferrum 只测过 Gemma 4 12B，26B-A4B 的 MoE 版本没有验证过，Metal 端也没有支持。先补架构再做性能，估计多 3–4 周。
- gpt-oss-20b（628 万）：Metal 端未支持，MXFP4 MoE 和注意力 sink 需要单独的内核，估计多 2–3 周。

**没有入选的：**
- Qwen3.8-Flash-Next（177B）、GLM-5.3-Flash（320B）：32GB 放不下。
- Gemma 4 31B：与 26B-A4B 一起推迟。
- Qwen3-Coder-30B-A3B：发布于 2025 年 7 月，与 Qwen3.6-35B-A3B 定位重合。

## 4. 对照方法与测量口径

- **负载**：沿用旧文档固定的 ShareGPT 选择（输入 4–1024 token，输出长度取参考答案并开启 ignore_eos，thinking 关闭，temperature 0）。删除 C1；Metal 测 C = 4、8、16，CUDA 测 C = 4、8、16、32。两台机器并行，同一机器内串行测量；扫描期间服务容量固定。每格预热 8 个请求。探索阶段正式请求数为 max(32, 4×C)，跑 1 次；验收阶段 M1 Max 每格正式请求数为 max(64, 4×C)，CUDA 每格 200 个正式请求，均跑 2 次。样本数不同的格使用同一冻结选择的前缀，同格跨引擎完全相同；预热与正式请求分开计数。第 3 层另加一个长 prompt 负载（ShareGPT 中输入 1K–4K token 的长度分档），用来检验调度。
- **llama.cpp**：固定一个发行版本，用同一个 GGUF 文件。开启 continuous batching 和 flash attention、FP16 KV、关闭 prompt cache。CUDA 固定 32 槽；M1 Max 先用短测试确定各模型全 GPU 能承载的最大槽位，再对两个引擎固定该容量扫描 C = 4、8、16。客户端并发可以高于槽位并排队，不为每格改容量。必须核对实际 GPU 放置；部分层在 CPU 的结果不能作为公平基线。
- **vLLM（CUDA）**：固定版本的 Docker 镜像（需支持 Blackwell）。模型用 vLLM 最擅长的 4bit 格式：两个模型都用社区 AWQ INT4 检查点（如 `cyankiwi/Qwen3.8-27B-AWQ-INT4`、`QuantTrio/Qwen3.6-35B-A3B-AWQ`，P0 固定具体版本）。FP8 版本在 32GB 显存放不下 35B-A3B，不作为对照。Ferrum 跑同一份检查点，80% 目标按同一份检查点计算；Ferrum 的 GGUF 结果另列，只用于和 llama.cpp 比较。Ferrum 不支持该检查点格式时，先补支持，不改用别的格式凑数。
- **Metal 对照**：仅 Ferrum 与 llama.cpp，同一个 GGUF 文件；不测试 vllm-metal 或 MLX 格式。
- **每一格都要报告**：TTFT、TPOT、可见 ITL 的 P50/P99，输出吞吐，显存/内存峰值，错误数，样本数，以及完整的版本、命令和配置。
- **输出质量**：每个模型用固定的 64 题做贪心生成，检查输出可用；Ferrum 与 llama.cpp 用同一个 GGUF 时，报告 token 一致率。量化格式不同时不要求逐 token 一致，但必须说明差异。

## 5. 起点

- 5090 上还没有任何 vLLM 实测数据。上表的 Ferrum 数据来自 4090 和 L40S，只能说明一个共同问题：**并发上不去**，吞吐在 c8 到 c16 就停住了。
- 已知的原因：
  - 每一轮迭代的 kernel 启动次数远多于 vLLM，因为 vLLM 用 CUDA Graph 捕获了整个前向。
  - 小 batch 的矩阵乘和 MoE 内核效率低。
  - 有些 kernel 按单个序列派发，没有跨请求合批。CUDA 的 attention 已经在 #402 修好；Metal 的 GDN 递推仍然是逐序列派发。
- Qwen3.5-9B 在 5090 上单请求时 TPOT 为 9.7ms，llama.cpp 为 6.0ms。

## 6. 阶段

每个阶段都有时间上限和检查点。检查点的结论以主负载的实测吞吐或 G 为准。

### P0 基线矩阵（≤2 天，不改产品代码）

1. 准备对照引擎：在 5090 上部署 vLLM 的 Docker 镜像和 llama.cpp，在 M1 Max 上部署 llama.cpp；下载第 3 节的模型文件，记录 hash。
2. 确认 Ferrum 能在两个后端正常 `run` 和 `serve` 两个模型，包括 CUDA 上加载 AWQ INT4 的 35B-A3B。跑不起来的列入缺口清单，不在这一步修复。
3. 删除 C1。Metal 测 C = 4、8、16；CUDA 测 C = 4、8、16、32。每格预热 8 个、正式 max(32, 4×C) 个请求、跑 1 次。每模型 CUDA 含 Ferrum GGUF、llama.cpp GGUF、Ferrum AWQ、vLLM AWQ 四组；Metal 含 Ferrum GGUF、llama.cpp GGUF 两组。因此共 2 × (4×4 + 2×3) = 44 格（Metal 12、CUDA 32）；不可运行格明确列为支持缺口。已完成的旧 3 格不计入新矩阵：llama.cpp 两格有 CPU 卸载，Ferrum 一格容量和请求口径不同，均按新口径重测。
4. 对差距最大的两三格做 profiler 对比（CUDA 用 nsys，Metal 用 GPU capture）：CUDA 的 Ferrum 对 vLLM，Metal 的 Ferrum 对 llama.cpp，对比 kernel 名称、调用次数、耗时和每轮启动次数。

退出条件：交出一张差距表（两后端每格 Ferrum 与 llama.cpp 的比值，CUDA 另含同检查点 vLLM 比值）和按耗时排序的缺口清单。

### P1 并发与吞吐：Qwen3.8-27B 与 Qwen3.6-35B-A3B（4–6 周）

按 P0 的缺口清单成批修复。预期方向：
- CUDA：decode 整个前向按 batch 大小捕获成 CUDA Graph；小 batch（m=8–64）的 4bit 矩阵乘达到 Marlin 级别；MoE 内核针对小 batch 优化；GDN 递推跨请求合批；去掉每层的 host 同步。
- Metal：batched decode 每多一行的增量开销降下来；GDN 递推跨请求合批；W4 矩阵乘在 m=8–32 时使用 simdgroup matrix。

检查点：
- 第 2 周末：两个模型在两个后端上，C ≥ 4 的每一档吞吐都超过 llama.cpp。
- 第 4 周末：CUDA 的 C = 8–32 时达到 vLLM 的 50% 以上。如果达不到，停下来复盘目标和路线，不追加零散补丁。
- 第 6 周末：CUDA 达到 vLLM 的 80%；Metal 保持相对 llama.cpp 的并发吞吐要求。

### P2 完整 SLO-aware（≤2 周）

1. **自动选配置**：根据模型、硬件和 S，自动选择数值 profile、split/mixed 模式、分块方式、KV 容量和并发上限。`ferrum serve --model <名称>` 加上 S 就能达到手调最优，不再需要手写一长串参数。
2. **知道自己的容量**：用在线步时模型估算在 S 下能撑多少并发，通过 `/health` 和启动日志报告；提供可选的并发上限，负载超过容量时让新请求排队，保住已接纳请求的 TPOT 和 ITL。
3. **动态 prefill 调度**：按 TTFT 截止时间安排 prefill，并把 ITL P99 允许的约 1% 超标额度留给少数长 prompt；在长 prompt 负载和 M1 Max 高并发档上验证。G 没有提升就保持默认关闭。

验收：每个模型在两个后端上，G（自动配置）不低于手调最优 G 的 97%；CUDA 还须不低于 vLLM 同条件 G 的 80%。

### P3 验收与发布（≤1 周）

- 两台机器并行、2 个模型，跑上述完整并发矩阵。每格预热 8 个请求；M1 Max 每格正式 max(64, 4×C) 个，CUDA 每格正式 200 个，均跑 2 次。CUDA 表含 Ferrum、llama.cpp、vLLM，Metal 表只含 Ferrum、llama.cpp。Qwen3.5-9B 按回归要求单独核对。
- README 只写实测范围内的结论，为每个模型写好可以直接运行的命令。
- 通过 release 流程发布安装包。

## 7. 规则

- **先测后改**：每个改动都必须对应 P0 或检查点测出来的具体缺口，并用主负载的吞吐或 G 做前后对比。不搭建证明、校准或资格判定一类的框架。
- **成批修复**：先用 profiler 对比列出完整清单，再按清单集中修改。不走"跑一次、修一个、再跑"的循环。
- **范围固定**：这一版只有第 3 节的两个目标模型；推迟的模型等这一版发布后再开。
- **控制规模**：单个 PR 的非测试代码超过约 3000 行时，先拆分。
- **不能取巧**：同一组样本、同一个文件、不降低质量；看到结果之后不改 S，也不换更容易的样本。
- **文档保持短**：只有本文件，加上每个阶段一页结果表；过程日志放在仓库外。
- 测试和基准逻辑写在 Rust 里；不按模型名字写特例，按架构和声明的能力处理。

## 8. 明确不做

- 32GB 放不下的模型。
- Gemma 4 和 gpt-oss：推迟到下一版，见第 3 节。
- 启动校准、每波搜索或认证式规划（旧 SLO 线已经证明不可行）。
- 只优化单请求速度。C1 已从性能矩阵移除；短测试不计入矩阵完成数。

## 9. 用户决定（2026-10-07 已确认）

1. 目标模型为 Qwen3.8-27B 和 Qwen3.6-35B-A3B；Qwen3.5-9B 只作回归；Gemma 4 和 gpt-oss 推迟到下一版。
2. Metal 目标机用本机 M1 Max 32GB。测量期间这台机器不跑其他重负载。
3. 计算 vLLM 的 80% 时，vLLM 和 Ferrum 跑同一份 AWQ 4bit 检查点；和 llama.cpp 比较时，两边跑同一个 GGUF 文件。
4. S 按以下规则，在 P2 开始前为每个模型、每台机器算出具体数值并写进本节：S_TPOT 取该机器上 llama.cpp 单请求 TPOT 的 2 倍，S_ITL 取 S_TPOT 的 3 倍，S_TTFT 取最长 prompt 单请求 prefill 时间再加余量。算出后不再根据 Ferrum 的结果调整。
5. 工期约 7–9 周（P0 2 天、P1 4–6 周、P2 2 周、P3 1 周）。P1 第 4 周检查点如果还不到 vLLM 的 50%，停下来重新评估，不追加零散补丁。
6. 2026-10-07 更新：Metal 只与 llama.cpp 比较，vLLM 只在 CUDA 环境测试；取消 Metal 的 vllm-metal/MLX 对照准备和比值要求。两后端其余性能与 SLO 目标保留。

7. 2026-10-07 更新：采用第 4 节的新请求数、预热、并发档和全 GPU 固定槽位口径；M1 Max 与 CUDA 并行执行。用户估计探索 M1 Max 两模型约 3–5 小时、CUDA 约 1–2 小时，实际时间按新全 GPU 结果更新，不将准备或旧 CPU 卸载结果计为完成。CUDA 统一使用 `~/ferrum-handoffs/20261006-g32/ssh-via-tailscale-mini.cjs`，经 Mac mini 的本地密钥进入 WSL。

## 10. 数据来源（2026-10-07 查询）

- Hugging Face API：text-generation 和 GGUF 模型按 30 天下载量和 trending 排序，以及各候选仓库的元数据和文件大小。
- Ollama 模型库：按热度排序的累计拉取量。
- RTX 5090 第三方选型指南：[llmconfigurator](https://llmconfigurator.com/en/best-models/rtx-5090)、[runaihome](https://www.runaihome.com/blog/best-llm-every-rtx-50-series-gpu-2026/)、[atomic.chat](https://atomic.chat/blog/guides/best-local-llms-for-rtx-5090)、[modelfit](https://modelfit.io/gpu/rtx-5090/)。
- vllm-metal 发布说明：[vLLM blog 2026-09-22](https://vllm.ai/blog/2026-09-22-vllm-metal-v0-28-0)。
- Reddit（r/LocalLLaMA）无法直接访问，搜索和抓取都被拒绝，这里没有使用它的数据。
- Ferrum 现状来自 main（0680840b）的 README 性能快照，以及 #402 的结果页。
