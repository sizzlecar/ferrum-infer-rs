<p align="center">
  <a href="https://ferrum.pandaailabs.com/zh/">
    <img src="assets/brand/ferrum-lockup.svg" alt="Ferrum — Rust 原生大模型服务" width="520">
  </a>
</p>

[English](README.md) · [发布版本](https://github.com/sizzlecar/ferrum-infer-rs/releases) · [性能对比](#性能) · [MIT](LICENSE)

# Ferrum

Rust 文本与图像推理引擎，一个二进制提供 CLI 和 OpenAI 兼容 API，支持 Metal、CUDA 和 CPU。

目标：满足首 token 延迟（TTFT）、单 token 耗时（TPOT）和输出间隔（ITL）要求，最大化输出吞吐。

固定硬件、模型和输出策略，优化执行策略 $\pi\in\Pi$：

$$
\begin{gathered}
\max_{\pi\in\Pi}\;\frac{N_{\mathrm{out}}(\pi;W,T)}{T} \\
\text{s.t.} \\
Q_{p_k}\!\left(L_k(\pi;W,T)\right)\le S_k \\
\forall k\in\{\mathrm{TTFT},\mathrm{TPOT},\mathrm{ITL}\} \\
C(\pi;W,T)\ge\rho
\end{gathered}
$$

- $W$：固定请求集与到达时序；$T>0$：固定窗口，结束即共同截止点。$N_{\mathrm{out}}$：截止前成功完成请求的有效输出 token 数。
- $L_k$：延迟样本；$Q_{p_k}$：分位数；$S_k$：SLO 上限。TTFT 包含排队时间，缺失延迟证据不算达标。
- $C$：$W$ 中截止前成功完成的比例，分母包含拒绝、失败、超时和未完成请求。测量前确定 $p_k$、$S_k$ 和最低完成率 $0<\rho\le1$。[采样口径](docs/performance-rtx5090-gpt-oss-20261102.zh.md)。

## 安装

macOS Apple Silicon / Linux x86_64：

```sh
curl -fsSL https://ferrum.pandaailabs.com/install.sh | sh
```

Windows x64，在 PowerShell 中执行：

```powershell
irm https://ferrum.pandaailabs.com/install.ps1 | iex
```

安装脚本校验发布包。安装后打开新终端，再次执行即可升级。

| 平台 | 后端 |
|---|---|
| macOS Apple Silicon | Metal |
| Linux / WSL x86_64 | CUDA SM89 或 SM120；CPU 回退 |
| Windows x64 | CUDA SM89；CPU 回退 |

自动安装在 GPU 架构不受支持或混合时回退 CPU。CUDA 需要匹配的 NVIDIA 驱动，详见[运行环境要求](docs/release-regression.md#prepare-the-candidate)。

## 文本生成

```sh
# 终端对话
ferrum run qwen3.5:4b-q4_k_m --disable-thinking

# OpenAI 兼容服务
ferrum serve --model qwen3.5:4b-q4_k_m \
  --served-model-name ferrum --disable-thinking --port 8000
```

首次运行自动下载权重。客户端 API 地址填 **`http://127.0.0.1:8000/v1`**，模型名填 **`ferrum`**。删除 `--disable-thinking` 可恢复模型模板默认行为。

执行策略和容量按硬件、可用内存自动选择，显式设置优先。支持 Chat Completions、无状态 Responses、SSE、工具调用和结构化输出，字段范围见 [API 兼容说明](docs/openai-api-compatibility.md)。

## 图像生成与编辑

使用本地 [Qwen-Image-2.1-Turbo](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo) 模型目录。Linux CUDA 示例，文本编码器和 DiT 使用 FP8：

```sh
ferrum serve /path/to/Qwen-Image-2.1-Turbo --backend cuda \
  --image-precision fp8_linear_experimental_v1 \
  --served-model-name qwen-image --port 8000
```

| API | 输入 |
|---|---|
| `POST /v1/images/generations` | JSON 文本提示词 |
| `POST /v1/images/edits` | Multipart 提示词 + 单张 8 位 RGB/RGBA PNG |

均返回 PNG 的 `b64_json`。CUDA 在容量允许时自动合批，已验证 L20（SM89）和 RTX 5090（SM120），需要 cuDNN 9。Metal 使用 GGUF DiT，已在 M4 Mac mini、16 GiB 内存上验证 1024×1024 生图与编辑。

CLI 示例、Metal 配置和 CUDA LoRA 见[图像使用指南](docs/image-generation.md)。

## 支持范围

| 范围 | 模型、格式与后端限制 |
|---|---|
| 文本架构 | Llama、Mistral、Qwen2、Qwen3 dense/MoE、Qwen3.5 hybrid dense/MoE、Gemma 3 文本 |
| 权重格式 | GGUF，含 Q4_K/Q5_K/Q6_K/Q8_0；SafeTensors |
| CUDA 量化 | 部分模型路径支持 GPTQ INT4、compressed-tensors INT4、block-FP8 |
| CUDA 专用模型 | Gemma 4 compressed-tensors W4A16；GPT-OSS MXFP4 |
| PQ2_0 | Metal 已验证完整模型；CUDA 仅有算子覆盖 |
| 图像 | Qwen-Image-2.1-Turbo 生图与单张 PNG 编辑；Linux CUDA SM89/SM120、Metal。M4 Metal 已验证 1024×1024 |

支持范围取决于架构、格式和后端组合；文本支持不代表视觉支持。CPU 为兼容后端。已验证组合与覆盖范围见[回归目录](docs/release-regression-catalog.md)。

Embedding、ASR、TTS 入口和旧执行 API 已退役，见[迁移说明](https://github.com/sizzlecar/ferrum-infer-rs/pull/441)。

## 性能

**RTX 5090：所有已测热门模型的最新记录。** 吞吐单位 tok/s，每格为 **Ferrum / 对照**；C8、C16 表示客户端并发。版本逐行标注，未将旧版本数据冒充新版复测。

| 模型 / 源权重 | 实测版本 | 对照 | C8 tok/s | C16 tok/s | SLO：C8 / C16 |
|---|---|---|---:|---:|---|
| Qwen3.5-4B · Q4_K_M | 0.18.3 | llama.cpp | 960.8 / 634.7 | 1294.8 / 729.4 | 通过 / Ferrum ITL 超限 |
| Qwen3.5-9B · Q4_K_M | 0.18.3 | llama.cpp | 731.6 / 550.5 | 942.0 / 654.7 | 通过 / Ferrum ITL 超限 |
| Qwen3.5-4B · BF16 | 0.18.3 | vLLM | 840.0 / 684.6 | 1306.8 / 1153.9 | 通过 / 通过 |
| Qwen3.5-9B · BF16 | 0.18.3 | vLLM | 563.4 / 508.2 | 875.0 / 864.3 | 通过 / 通过 |
| Qwen3-30B-A3B · GPTQ INT4 | 0.18.3 | vLLM | 630.5 / 824.0 | 848.8 / 1263.7 | 通过 / 通过 |
| Qwen3.5-35B-A3B · GPTQ INT4 | 0.18.3 | vLLM | 717.6 / 625.6 | 978.3 / 947.4 | 通过 / vLLM ITL 超限 |
| GPT-OSS-20B · MXFP4 | 0.18.4 R11 | vLLM | 890.9–892.3 / 819.8 | 1383.5–1385.9 / 1429.8 | 通过 / 通过 |
| Qwen3.8-27B · UD-Q4_K_M | 0.13 候选 | llama.cpp | 356.42 / 291.60 | 535.51 / 377.41 | 双方未达标 / 双方未达标 |
| Qwen3.8-27B · CT INT4 | 0.13 候选 | vLLM | 374.32 / 447.06 | 654.04 / 733.86 | Ferrum 未达标 / 双方未达标 |

“通过”表示双方两轮均达标；ITL 超限发生在该档第一轮。4B/9B/30B/35B 每引擎每档 64 请求×2 轮，全部成功、零错误/拒绝；GPT-OSS 为 32 请求×2 独立进程，对照保留一轮。27B 为 200 请求×2 轮、零错误，使用更严格的独立 SLO，不能与其他行直接排名。

4B/9B/30B/35B/GPT-OSS 的 P99 上限为 TTFT 2000 ms、TPOT 50 ms、可见 SSE ITL 100 ms。同一行使用同源权重和相同负载；BF16 源权重组及 35B 的 Ferrum 激活/KV 为 F16，vLLM 为 BF16。30B 仍明显落后；9B BF16 和 35B 的小幅吞吐领先在不同轮次中会反转，不宣称稳定优势。

完整 TTFT/TPOT/ITL P50/P99、显存/RSS、命令与采样口径：[4B / 9B / 30B / 35B](docs/performance-rtx5090-popular-models-20261101.zh.md) · [GPT-OSS](docs/performance-rtx5090-gpt-oss-20261102.zh.md) · [27B](docs/performance-popular-models-p0.zh.md)。

<details>
<summary>GPT-OSS 最新优化：延迟和资源明细</summary>

**RTX 5090 · GPT-OSS 20B · MXFP4：** **0.18.4** 合入的优化在 C8 达到 vLLM 吞吐的 **108.7–108.9%**，C16 达到 **96.8–96.9%**，均满足预设延迟 SLO。数据来自该优化的 R11 候选版本，未对最终发布包重新压测。

同 GPU、源权重与固定 ShareGPT 样本；自然 EOS，最多输出 512 token。Ferrum 使用 F16 外层激活/KV、BF16 专家路径，vLLM 使用 BF16 激活/KV。延迟为 **P50 / P99，单位毫秒**；ITL 按客户端非空可见 SSE 文本事件计算。

| 并发 | 引擎 / 独立进程 | 输出 tok/s | TTFT | TPOT | SSE ITL |
|---:|---|---:|---:|---:|---:|
| 8 | Ferrum R11 #1 | 890.93 | 155.70 / 227.27 | 8.17 / 8.62 | 8.00 / 14.38 |
| 8 | Ferrum R11 #2 | 892.30 | 157.81 / 226.31 | 8.21 / 8.61 | 7.95 / 13.02 |
| 8 | vLLM 0.31.0 | 819.77 | 94.75 / 387.13 | 8.71 / 8.96 | 8.16 / 33.00 |
| 16 | Ferrum R11 #1 | 1383.53 | 346.93 / 462.27 | 10.00 / 10.29 | 9.88 / 11.73 |
| 16 | Ferrum R11 #2 | 1385.86 | 340.35 / 444.62 | 10.10 / 10.39 | 9.89 / 12.27 |
| 16 | vLLM 0.31.0 | 1429.75 | 174.08 / 183.29 | 10.14 / 10.38 | 9.47 / 34.24 |

所有行均满足 P99 TTFT ≤2000 ms、TPOT ≤50 ms、SSE ITL ≤100 ms；每格成功 32/32，错误和拒绝均为 0。Ferrum 用两个独立进程各跑一轮 C8→C16，共成功 128/128；vLLM 复用既有基线。C16 首字延迟仍高于 vLLM。

整轮峰值设备显存 / 窗口进程 RSS：Ferrum #1 为 **15.978 / 1.177 GiB**，#2 为 **16.165 / 1.192 GiB**；vLLM 为 **29.937 / 3.145 GiB**。不是逐并发峰值；vLLM 预分配 KV，且其 RSS 仅统计 EngineCore，不能据此推导内存效率倍数。

[负载、命令、输出有效性与完整证据](docs/performance-rtx5090-gpt-oss-20261102.zh.md)。结论仅覆盖该模型与配置。

</details>

## 文档

- [API 兼容说明](docs/openai-api-compatibility.md) · [Agent 服务](docs/agent-serving.md)
- [数值执行策略](docs/numerical-execution.zh.md) · [缓存说明](docs/cache-product.md)

## 项目

Ferrum 使用 AI 辅助开发，仍在迭代，内部 API 可能变化。欢迎贡献代码和提供硬件测试。

感谢 [vLLM](https://github.com/vllm-project/vllm) 与 [llama.cpp](https://github.com/ggml-org/llama.cpp) 的代码和实现参考，相关组件许可证随分发保留。
