<p align="center">
  <a href="https://ferrum.pandaailabs.com/zh/">
    <img src="assets/brand/ferrum-lockup.svg" alt="Ferrum — Rust 原生大模型服务" width="520">
  </a>
</p>

[English](README.md) · [Crates.io](https://crates.io/crates/ferrum-cli) · [MIT](LICENSE)

# Ferrum

Rust 文本与图像推理，一个二进制提供 CLI 和 OpenAI 兼容 API。

**[v0.16.0](https://github.com/sizzlecar/ferrum-infer-rs/releases/tag/v0.16.0)** 新增 Metal Qwen-Image 图像生成与编辑，支持 GGUF DiT。

核心目标是在满足延迟 SLO 的前提下，最大化输出吞吐：

固定硬件、模型和输出策略，优化候选执行策略 $\pi\in\Pi$：

$$
\begin{gathered}
\max_{\pi\in\Pi}\;\frac{N_{\mathrm{out}}(\pi;W,T)}{T} \\
\text{s.t.} \\
Q_{p_k}\!\left(L_k(\pi;W,T)\right)\le S_k \\
\forall k\in\{\mathrm{TTFT},\mathrm{TPOT},\mathrm{ITL}\} \\
C(\pi;W,T)\ge\rho
\end{gathered}
$$

在固定负载和测量窗口下，最大化成功请求的有效输出吞吐，同时满足尾延迟 SLO 和最低请求完成率。

- $W$：固定请求集与到达时序；$T>0$：固定测量窗口，结束时为共同统计截止点。$N_{\mathrm{out}}$：窗口内成功完成请求的有效输出 token 数。
- $L_k$：延迟样本；$Q_{p_k}$：指定分位数；$S_k$：上限。TTFT 包含排队时间，缺失延迟证据不能视为达标。[采样口径](docs/performance-popular-models-p0.zh.md)。
- $C$：$W$ 中截止前成功完成的请求比例；分母不剔除拒绝、失败、超时和未完成请求。测量前预设 $p_k$、$S_k$ 及 $0<\rho\le1$。

## 安装与使用

macOS Apple Silicon / Linux x86_64：

```sh
curl -fsSL https://ferrum.pandaailabs.com/install.sh | sh
```

Windows x64，在 PowerShell 中执行：

```powershell
irm https://ferrum.pandaailabs.com/install.ps1 | iex
```

脚本校验发布包并配置 PATH；安装后打开新终端。再次执行即可升级，模型与配置保留。命令获取最新已发布正式版。

macOS 使用 Metal；Linux/WSL CUDA 支持 SM89（CUDA 12.4），SM120（CUDA 12.8）需 v0.13.0 或更新版本，需匹配的 NVIDIA 驱动及 CUDA/NCCL 运行库。Windows 与 Homebrew CUDA 面向 SM89；Windows 包内含 CUDA/VC 运行库，驱动需单独安装。自动安装在 GPU 不受支持或 CUDA 无法启动时回退 CPU；指定版本缺少对应资产会明确报错。

运行同一个 GGUF 示例，选择终端对话或 API 服务：

```sh
ferrum run qwen3.5:4b-q4_k_m --disable-thinking
ferrum serve --model qwen3.5:4b-q4_k_m --served-model-name ferrum --disable-thinking --port 8000
```

首次运行自动下载权重；上下文、批处理及并发按可用内存适配，显式设置优先。示例关闭思考，删除 `--disable-thinking` 即恢复模型模板默认行为。

应用的 API 地址设为 `http://127.0.0.1:8000/v1`，模型名设为 `ferrum`。支持 Chat Completions、无状态 Responses、流式输出、工具调用和结构化输出，具体字段见 [API 兼容说明](docs/openai-api-compatibility.md)。下载资产与发行状态见 [Releases](https://github.com/sizzlecar/ferrum-infer-rs/releases)。

## 图像生成与编辑

CUDA 使用完整的本地 [Qwen-Image-2.1-Turbo](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo) 模型目录。已验证 RTX 5090、Linux CUDA 12.8、cuDNN 9；文本编码器/DiT 使用 FP8，视觉编码器/VAE 使用 BF16。

Metal 支持 Qwen-Image 生图与单张 PNG 编辑，使用 Q4_K_M GGUF DiT、FP8 文本编码器和 BF16 视觉编码器/VAE。已在 M4 Mac mini、16 GiB 统一内存上验证 1024×1024。通过 `--image-config` 配置后端、资产和内存预算。

**Metal 配置：**将 [qwen-image-2.1-turbo-metal.json](crates/ferrum-cli/examples/config/qwen-image-2.1-turbo-metal.json) 复制为 `image-metal.json`，与 `models/` 目录并列。原模型的文本编码器、VAE 和元数据放在 `models/Qwen-Image-2.1-Turbo/`，[Q4_K_M DiT](https://huggingface.co/m9ter1a/Qwen-Image-2.1-Turbo-GGUF/blob/a94df596a422bc8dfe406655154c0364f79ba042/qwen-image-2.1-turbo-Q4_K_M.gguf) 放在 `models/Qwen-Image-2.1-Turbo-GGUF/`；也可直接修改 JSON 路径，相对路径以配置文件所在目录为基准。示例保留单个活动图像、组件按阶段加载释放、请求内编辑缓存、2 GiB Host 预算和 Metal 推荐工作集的 90%，仍需通过容量检查；不需要原始 BF16 DiT 权重。

推荐配置启用融合注意力和单次范围不超过 128 MiB 的文本编码器权重预读；将 `backend.attention_policy` 设为 `reference` 可切回原注意力路径。

<details>
<summary>模型快照缺少 transformer/config.json 时</summary>

将下面匹配 Qwen-Image-2.1-Turbo 的架构保存为 `models/Qwen-Image-2.1-Turbo/transformer/config.json`；已有匹配文件则保留：

```json
{
  "_class_name": "QwenImage21Transformer2DModel",
  "_diffusers_version": "0.41.0.dev0",
  "attention_head_dim": 128, "axes_dims_rope": [16, 56, 56],
  "causal_condition": true, "context_in_dim": 4096, "eps": 0.000001,
  "in_channels": 64, "mlp_ratio": 3, "num_attention_heads": 32,
  "num_layers": 32, "out_channels": 64, "patch_size": 1
}
```

</details>

```sh
ferrum run --image-config image-metal.json \
  --prompt "A red ceramic mug on a white table" --width 1024 --height 1024 --seed 42 --output mug.png
ferrum run --image-config image-metal.json --image mug.png \
  --prompt "Make the mug blue" --width 1024 --height 1024 --seed 43 --output edited.png
ferrum serve --image-config image-metal.json --host 127.0.0.1 --port 8000
```

API 模型名为 `qwen-image-turbo-metal`。精度在该 JSON 中配置；`--image-config` 不与位置参数模型或 `--image-precision` 同用。

CUDA 示例：

~~~sh
ferrum run /path/to/Qwen-Image-2.1-Turbo \
  --image-precision fp8_linear_experimental_v1 \
  --prompt "A red fox in a snowy forest" --width 1024 --height 1024 --output fox.png

ferrum run /path/to/Qwen-Image-2.1-Turbo --image fox.png \
  --image-precision fp8_linear_experimental_v1 \
  --prompt "Make the fox blue" --width 1024 --height 1024 --output edited.png

ferrum serve --model /path/to/Qwen-Image-2.1-Turbo --served-model-name qwen-image \
  --image-precision fp8_linear_experimental_v1 --port 8000
~~~

HTTP：`POST /v1/images/generations` 接收 JSON，`POST /v1/images/edits` 接收 multipart。编辑支持单张 8 位 RGB/RGBA PNG，返回 PNG 的 `b64_json`；背景和细节可能重绘。

CUDA 本地目录默认启用 cuDNN SDPA、请求内编辑前缀复用和有界多形状计划缓存：每阶段最多 16 条，序列化编译元数据总计最多 128 MiB。`run` 和 `serve` 均按需填充缓存，不按上限预分配内存。CUDA 目录支持最高 2048×2048，执行前仍须通过容量检查；Metal 保持 1024×1024 和每阶段最近计划缓存。

### 对比 vLLM-Omni

v0.15.0 候选版实测 · RTX 5090 · Qwen-Image-2.1-Turbo · 8 步 · 文本编码器/DiT FP8、VAE BF16。每项 4 对发布候选配对测试，共 640 个正式请求，零错误。单位：张/秒；并发为客户端并发。

| 任务 | 并发 | Ferrum | vLLM-Omni | Ferrum / Omni |
|---|---:|---:|---:|---:|
| 1024² 生成 | 8 | 0.5193 | 0.4428 | 1.173× |
| 1024² 编辑 | 1 | 0.2943 | 0.2631 | 1.119× |
| 2048² 生成 | 2 | 0.0799 | 0.0772 | 1.035× |
| 2048² 编辑 | 2 | 0.0686 | 0.0685 | ≈1.00×* |

两端均满足对应 P99 上限：20 / 5 / 30 / 30 秒。*2K 编辑的比值 95% 区间为 0.9976–1.0077，差异尚不显著。表格采用常驻服务配置，不代表默认 CLI 冷启动性能。[配置、延迟与内存](https://github.com/sizzlecar/ferrum-infer-rs/releases/tag/v0.15.0#qwen-image-performance)。

## 模型与格式

文本架构包括 Llama、Mistral、Qwen2/Qwen3、Qwen3 MoE、Qwen3.5 dense/MoE 和 Gemma 3。支持 GGUF（含 Q4_K/Q5_K/Q6_K/Q8_0）与 SafeTensors；部分 CUDA 路径支持 GPTQ INT4、compressed-tensors INT4 和 block-FP8。Gemma 4 W4A16、GPT-OSS MXFP4 为专用 CUDA 文本路径；PQ2_0 完整模型目前仅在 Metal 实测。

并非任意架构、格式和设备组合都受支持，也不由文本支持推断视觉能力；范围见[回归目录](docs/release-regression-catalog.md)。

本次架构更新退役 embedding、ASR、TTS 入口及旧执行 API，升级见[迁移说明](https://github.com/sizzlecar/ferrum-infer-rs/pull/441)。

默认执行策略按设备、算子能力和容量选择；合资格的 SM120 GGUF 路径可采用 8 位激活量化及整数点积，数值路径或批布局可能改变输出。`--numerical-profile qwen3_5.f32-master` 保留原混合精度参考路径，见[数值执行策略](docs/numerical-execution.zh.md)与[缓存说明](docs/cache-product.md)。

## 文本性能（0.13 候选）

RTX 5090、Qwen3.8-27B、固定 ShareGPT 样本；服务容量保持 32 槽、上下文及批处理上限均为 2048。每格正式 200 请求 × 2 轮，每轮另有 8 次预热，协议错误为 0。以下为 v0.13.0 候选的正式测量，单位为输出 token/s；两轮按总成功输出 usage token ÷ 总正式时长合并。

本表采用固定请求数，以各轮实际耗时计算吞吐；上式描述固定测量窗口的目标。

**GGUF Q4_K_M，同文件对比 llama.cpp b11429：**

| 并发 | Ferrum | llama.cpp | Ferrum / llama.cpp |
| ---: | ---: | ---: | ---: |
| 4 | 179.25 | 170.13 | 105.36% |
| 8 | 356.42 | 291.60 | 122.23% |
| 16 | 535.51 | 377.41 | 141.89% |
| 32 | 666.24 | 435.57 | 152.96% |

**compressed-tensors INT4、group size 32，对比 vLLM 0.31：**

| 并发 | Ferrum | vLLM | Ferrum / vLLM |
| ---: | ---: | ---: | ---: |
| 4 | 210.32 | 233.45 | 90.09% |
| 8 | 374.32 | 447.06 | 83.73% |
| 16 | 654.04 | 733.86 | 89.12% |
| 32 | 836.79 | 1035.09 | 80.84% |

多档延迟 SLO 未通过；吞吐过线不代表 SLO 达成。INT4 对比中 Ferrum 的激活/KV 为 FP16，vLLM 为 BF16；C32 第二轮比例为 **78.91%**，未证明每轮稳定达到 80%。同 GGUF 文件也不等于内部算术相同，零协议错误不等于所有回答语义正确。这些结果仅覆盖上述模型、硬件与负载。

完整配置、TTFT/TPOT/可见 SSE ITL 的 P50/P99、显存与内存峰值、质量边界见[结果与决定](docs/performance-popular-models-p0.zh.md)。本次重构建立执行与控制接口；吞吐和延迟的联合优化仍属后续工作。

## 项目说明

Ferrum 使用 AI 辅助开发，仍在快速迭代；内部 API 暂不承诺稳定，支持范围与性能持续更新。个人硬件资源有限，欢迎提供显卡测试支持；质量标准和如实报告失败、未验证范围的要求不会因此降低。

感谢 [vLLM](https://github.com/vllm-project/vllm) 与 [llama.cpp](https://github.com/ggml-org/llama.cpp)，部分实现参考或复用了其代码，相关许可证随分发保留。Ferrum 自身采用 [MIT 许可证](LICENSE)。
