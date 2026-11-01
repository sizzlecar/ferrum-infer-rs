<p align="center">
  <a href="https://ferrum.pandaailabs.com/zh/">
    <img src="assets/brand/ferrum-lockup.svg" alt="Ferrum — Rust 原生大模型服务" width="520">
  </a>
</p>

[English](README.md) · [发布版本](https://github.com/sizzlecar/ferrum-infer-rs/releases) · [MIT](LICENSE)

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
- $C$：$W$ 中截止前成功完成的比例，分母包含拒绝、失败、超时和未完成请求。测量前确定 $p_k$、$S_k$ 和最低完成率 $0<\rho\le1$。[采样口径](docs/performance-popular-models-p0.zh.md)。

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

## 文档

- [API 兼容说明](docs/openai-api-compatibility.md) · [Agent 服务](docs/agent-serving.md)
- [数值执行策略](docs/numerical-execution.zh.md) · [缓存说明](docs/cache-product.md)
- 历史实测：[文本，v0.13 候选](docs/performance-popular-models-p0.zh.md) · [图像，v0.15 候选](https://github.com/sizzlecar/ferrum-infer-rs/releases/tag/v0.15.0#qwen-image-performance)。包含负载、延迟和内存数据，仅代表对应版本与配置。

## 项目

Ferrum 使用 AI 辅助开发，仍在迭代，内部 API 可能变化。欢迎贡献代码和提供硬件测试。

感谢 [vLLM](https://github.com/vllm-project/vllm) 与 [llama.cpp](https://github.com/ggml-org/llama.cpp) 的代码和实现参考，相关组件许可证随分发保留。
