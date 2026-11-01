<p align="center">
  <a href="https://ferrum.pandaailabs.com/zh/">
    <img src="assets/brand/ferrum-lockup.svg" alt="Ferrum — Rust 原生大模型服务" width="520">
  </a>
</p>

[English](README.md) · [发布版本](https://github.com/sizzlecar/ferrum-infer-rs/releases) · [MIT](LICENSE)

# Ferrum

Rust 文本与图像推理引擎，一个二进制提供 CLI 和 OpenAI 兼容 API，支持 Metal、CUDA 和 CPU。

目标：满足首 token 延迟（TTFT）、单 token 耗时（TPOT）和输出间隔（ITL）要求，最大化输出吞吐。

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

## 模型与文档

文本架构包括 Llama、Mistral、Qwen2、Qwen3 dense/MoE、Qwen3.5 dense/MoE 和 Gemma 3。支持 GGUF、SafeTensors，量化与模型支持范围取决于后端，已验证组合见[回归目录](docs/release-regression-catalog.md)。

- [API 兼容说明](docs/openai-api-compatibility.md) · [Agent 服务](docs/agent-serving.md)
- [数值执行策略](docs/numerical-execution.zh.md) · [缓存说明](docs/cache-product.md)
- 历史实测：[文本，v0.13 候选](docs/performance-popular-models-p0.zh.md) · [图像，v0.15 候选](https://github.com/sizzlecar/ferrum-infer-rs/releases/tag/v0.15.0#qwen-image-performance)。包含负载、延迟和内存数据，仅代表对应版本与配置。

## 项目

Ferrum 使用 AI 辅助开发，仍在迭代，内部 API 可能变化。欢迎贡献代码和提供硬件测试。

感谢 [vLLM](https://github.com/vllm-project/vllm) 与 [llama.cpp](https://github.com/ggml-org/llama.cpp) 的代码和实现参考，相关组件许可证随分发保留。
