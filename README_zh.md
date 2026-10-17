<p align="center">
  <a href="https://ferrum.pandaailabs.com/zh/">
    <img src="assets/brand/ferrum-lockup.svg" alt="Ferrum — Rust 原生大模型服务" width="520">
  </a>
</p>

[English](README.md) · [Crates.io](https://crates.io/crates/ferrum-cli) · [MIT](LICENSE)

# Ferrum

Rust 编写的本地大模型推理引擎，一个二进制提供终端对话和 OpenAI 兼容 API，无需 Python 运行时。

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

## 模型与格式

文本架构包括 Llama、Mistral、Qwen2/Qwen3、Qwen3 MoE、Qwen3.5 dense/MoE 和 Gemma 3。支持 GGUF（含 Q4_K/Q5_K/Q6_K/Q8_0）与 SafeTensors；部分 CUDA 路径支持 GPTQ INT4、compressed-tensors INT4 和 block-FP8。Gemma 4 W4A16、GPT-OSS MXFP4 为专用 CUDA 文本路径；PQ2_0 完整模型目前仅在 Metal 实测。

并非任意架构、格式和设备组合都受支持，也不由文本支持推断视觉能力；范围见[回归目录](docs/release-regression-catalog.md)。

默认执行策略按设备、算子能力和容量选择；合资格的 SM120 GGUF 路径可采用 8 位激活量化及整数点积，数值路径或批布局可能改变输出。`--numerical-profile qwen3_5.f32-master` 保留原混合精度参考路径，见[数值执行策略](docs/numerical-execution.zh.md)与[缓存说明](docs/cache-product.md)。

## 近期性能

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

完整配置、TTFT/TPOT/可见 SSE ITL 的 P50/P99、显存与内存峰值、质量边界见[结果与决定](docs/performance-popular-models-p0.zh.md)。联合 SLO 优化安排在下一版本。

## 项目说明

Ferrum 使用 AI 辅助开发，仍在快速迭代；内部 API 暂不承诺稳定，支持范围与性能持续更新。个人硬件资源有限，欢迎提供显卡测试支持；质量标准和如实报告失败、未验证范围的要求不会因此降低。

感谢 [vLLM](https://github.com/vllm-project/vllm) 与 [llama.cpp](https://github.com/ggml-org/llama.cpp)，部分实现参考或复用了其代码，相关许可证随分发保留。Ferrum 自身采用 [MIT 许可证](LICENSE)。
