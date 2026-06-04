# ferrum-infer-rs

[![Crates.io](https://img.shields.io/crates/v/ferrum-cli.svg)](https://crates.io/crates/ferrum-cli)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/sizzlecar/ferrum-infer-rs/blob/main/LICENSE)

> Rust-native LLM inference for fast, simple, OpenAI-compatible serving.

**一个二进制。Serving 路径不需要 Python runtime。Apple Silicon 与 NVIDIA CUDA 硬件加速。**

Ferrum 是一个 Rust 原生的 LLM 推理引擎，用于运行和部署 OpenAI 兼容的 transformer LLM 服务。
它面向希望简化部署、保留实际 serving 性能，并使用干净 Rust-native runtime 的开发者和团队，覆盖本地、边缘和生产推理场景。

[English](README.md)

## 快速开始

安装预编译二进制：

```bash
brew tap sizzlecar/ferrum
brew install ferrum        # macOS Apple Silicon Metal / Linux x86_64 CPU
brew install ferrum-cuda   # Linux x86_64 CUDA sm89 构建
ferrum --version
```

直接运行模型：

```bash
export HF_TOKEN=hf_your_token_here   # 仅 gated 模型需要
ferrum run qwen3:4b
```

通过 OpenAI 兼容 API 提供服务：

```bash
ferrum serve --model qwen3:4b --port 8000

curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3:4b","messages":[{"role":"user","content":"Hello"}]}'
```

## 为什么选择 Ferrum？

- **一个二进制：**同时提供 `ferrum run` 和 `ferrum serve`，runtime 路径不需要 Python 服务。
- **OpenAI 兼容 API：**可以复用现有 OpenAI 风格的 client、SDK 和 HTTP 工具。
- **硬件加速：**同一项目覆盖 Apple Silicon Metal 与 NVIDIA CUDA。
- **Rust-native runtime：**更少依赖、更简单部署，也更容易嵌入或分发。
- **实用 serving 性能：**支持 continuous batching、paged KV cache、INT4 GPTQ/Marlin、CUDA Graphs，并有 release 级并发 gate。

## Ferrum 适合什么场景？

Ferrum 面向这些开发和部署场景：

- 本地 AI agent
- 私有 OpenAI 兼容推理服务
- Apple Silicon LLM 应用
- CUDA 加速推理服务
- 边缘设备和工作站部署
- Rust-native AI 基础设施

## 性能快照

Ferrum 面向现代加速器上的高吞吐 serving 场景，并把原始 benchmark 日志和证据保存到仓库，而不是只给摘要数字。

RTX 4090 + `Qwen3-30B-A3B-GPTQ-Int4` 的历史 same-pod 吞吐，
对应 opt-in FA2 direct-FFI 路径：

| 并发 | Ferrum tok/s | vLLM 0.20.2 tok/s | Ferrum / vLLM |
| ---: | ---: | ---: | ---: |
| 1 | `160.4 +/- 0.2` | `183.9 +/- 0.2` | `0.872x` |
| 4 | `446.3 +/- 7.0` | `512.5 +/- 2.8` | `0.871x` |
| 16 | `1185.1 +/- 12.3` | `1331.9 +/- 5.7` | `0.890x` |
| 32 | `1641.9 +/- 4.8` | `1972.9 +/- 18.6` | `0.832x` |

这组历史对比的完整 CUDA 方法和原始证据见
[`docs/bench/cuda-rtx4090-2026-05-30-m3-80pct-confirmed/`](docs/bench/cuda-rtx4090-2026-05-30-m3-80pct-confirmed/)。
不要把这张表当作当前 source-linked/default release gate 的宣传数字；
release candidate 必须以当前 G0/G1-G4 CUDA artifacts 里的 binary、git SHA、
runtime config 和 same-hardware 结果为准。

Release 和 Metal gates：

| 目标 | 模型 / workload | 结果 | 证据 |
| --- | --- | --- | --- |
| CUDA release 二进制 | Qwen3-30B-A3B GPTQ-Int4, c=32 smoke | `16/16` 请求，`0` errors；Paris、多轮和三轮对话 gates 通过 | [`CUDA release 二进制验证`](docs/bench/dev-loop-product-api-goal-progress-20260601/release-bin-cuda-qwen3-30b-a3b-v0.7.4-final-05254fb-20260602/) |
| Apple Silicon Metal | Qwen3/LLaMA 8B 和 Qwen3-30B-A3B | 覆盖正确性、多轮对话和并发 gates | [`metal-readme-regression-20260601-release-candidate-rerun3`](docs/bench/dev-loop-product-api-goal-progress-20260601/metal-readme-regression-20260601-release-candidate-rerun3/) |
| Apple Silicon Metal 限制 | Qwen3-30B-A3B, c=16 | Ferrum `72.5 tok/s`；记录中的 llama.cpp `83.4 tok/s` | 同上 Metal 报告 |

## API 兼容性

Ferrum 提供 OpenAI 风格的 chat completions API，适合本地和私有部署。Endpoint 合约、明确拒绝的字段、tool 字段状态、usage 统计和 structured-output 限制见 [`docs/openai-api-compatibility.md`](docs/openai-api-compatibility.md)。

## 安装

Homebrew：

```bash
brew tap sizzlecar/ferrum
brew install ferrum        # macOS Metal / Linux CPU
brew install ferrum-cuda   # Linux x86_64 CUDA sm89 构建
```

预编译 release tarball：

```bash
# Linux x86_64 CPU
curl -L https://github.com/sizzlecar/ferrum-infer-rs/releases/latest/download/ferrum-linux-x86_64.tar.gz | tar xz
./ferrum --help

# Linux x86_64 CUDA, sm89 构建
curl -L https://github.com/sizzlecar/ferrum-infer-rs/releases/latest/download/ferrum-linux-x86_64-cuda-sm89.tar.gz | tar xz
LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-} ./ferrum --help

# macOS Apple Silicon Metal
curl -L https://github.com/sizzlecar/ferrum-infer-rs/releases/latest/download/ferrum-macos-aarch64.tar.gz | tar xz
./ferrum --help
```

Linux x86_64 是 CPU 构建。Linux x86_64 CUDA 是 `sm89` 构建，目标机器需要兼容的 NVIDIA driver 和 CUDA runtime libraries。macOS aarch64 是 Metal 构建。

从源码：

```bash
cargo install ferrum-cli
cargo build --release -p ferrum-cli --bin ferrum
```

## Benchmarks / Docs

- CUDA vLLM 对比：[`docs/bench/cuda-rtx4090-2026-05-30-m3-80pct-confirmed/`](docs/bench/cuda-rtx4090-2026-05-30-m3-80pct-confirmed/)
- Apple Silicon 回归报告：[`docs/bench/dev-loop-product-api-goal-progress-20260601/metal-readme-regression-20260601-release-candidate-rerun3/`](docs/bench/dev-loop-product-api-goal-progress-20260601/metal-readme-regression-20260601-release-candidate-rerun3/)
- OpenAI API 兼容性：[`docs/openai-api-compatibility.md`](docs/openai-api-compatibility.md)
- 模块状态记录：[`docs/status/`](docs/status/)

## 支持的模型

| 架构 | Apple Silicon | CUDA | INT4 (GPTQ) | Tensor Parallel |
|---|:---:|:---:|:---:|:---:|
| LLaMA (3.x, TinyLlama, Vicuna, Mistral) | ✓ | ✓ | ✓ | ✓ |
| Qwen3 dense (0.6B – 8B) | ✓ | ✓ | ✓ | ✓ |
| Qwen3-MoE (30B-A3B) | ✓ | ✓ | ✓ | — |
| Qwen2 / Qwen2.5 | ✓ | ✓ | ✓ | — |
| BERT (embeddings) | ✓ | — | — | — |
| Whisper ASR (tiny → large-v3-turbo) | ✓ | — | — | — |
| Qwen3-TTS (0.6B / 1.7B) | ✓ | — | — | — |
| CLIP / Chinese-CLIP / SigLIP (文本 + 图像) | ✓ | — | — | — |

可使用任意 HuggingFace 模型 ID:

```bash
ferrum run Qwen/Qwen3-4B
ferrum run meta-llama/Llama-3.2-3B-Instruct
ferrum run JunHowie/Qwen3-4B-GPTQ-Int4    # INT4 自动识别
```

### 多模态

```bash
# 语音转文字 (WAV/M4A/MP3/FLAC,自动 ffmpeg 转码)
ferrum transcribe whisper-turbo recording.m4a -l zh
ferrum serve whisper-turbo

# 文本转语音 (基础合成；可选参考音频克隆)
ferrum tts qwen3-tts "你好欢迎使用语音合成系统" -o output.wav
ferrum serve qwen3-tts

# Embedding (文本 + 图像)
ferrum embed OFA-Sys/chinese-clip-vit-base-patch16 --text "海边的日落"
ferrum embed google/siglip-base-patch16-224 --image photo.jpg
```

## 构建选项

```bash
# 仅 CPU (默认)
cargo install ferrum-cli

# Metal 加速 (macOS)
cargo install ferrum-cli --features metal

# 从源码构建 CUDA 加速 (NVIDIA, 需要 CUDA toolkit + nvcc)
cargo install ferrum-cli --features cuda,vllm-moe-marlin,vllm-paged-attn-v2,fa2-source
```

## 项目结构

```
crates/
├── ferrum-types          # 共享类型
├── ferrum-interfaces     # Trait 契约 (Backend<B>, ModelExecutor, ...)
├── ferrum-runtime        # Backend 注册
├── ferrum-engine         # Continuous-batch 引擎、Metal shader 流水线
├── ferrum-models         # 模型架构 (LlamaFamilyModel<B>, MoE, ...)
├── ferrum-kernels        # 自研 CUDA + Metal kernels, decode runner
├── ferrum-attention      # Fused-transformer 原型 (Metal/CPU)
├── ferrum-quantization   # GPTQ 加载、Marlin、native safetensors
├── ferrum-tokenizer      # Tokenization
├── ferrum-sampler        # 采样策略 (top-k/p、温度、重复惩罚、JSON-mode)
├── ferrum-scheduler      # 请求调度、paged-KV 调度
├── ferrum-kv             # Paged KV cache (CUDA + Metal pools)
├── ferrum-server         # HTTP API
├── ferrum-cli            # 二进制入口
└── ferrum-testkit        # 测试基础设施
```

Architecture v2 (Model-as-Code) 的意思是: 模型层是显式的 Rust 泛型,基于 `Backend<B>` trait,而不是 config-driven runner。增加一个后端 = 实现 trait,不需要改模型。详见 [docs/architecture-v2.md](docs/architecture-v2.md)。

## 当前状态

已可用:
- CLI 对话、OpenAI 兼容 HTTP server (含流式)
- Continuous batching、PagedAttention (CUDA + Metal pools)、前缀缓存、抢占
- 自研 CUDA decode runner (Qwen3, LLaMA): 比 Candle 快 2×
- Apple Silicon MoE 推理 (Qwen3-30B-A3B)，并通过正确性、多轮对话和并发 gates
- INT4 GPTQ with Marlin fused kernel (Blackwell + Ampere); 同时有 Triton w4a16
- Tensor parallelism (多 GPU NCCL, 持久化 per-rank 线程)
- Speculative decoding (`--spec-draft <MODEL>` DeepMind accept/reject)
- 结构化输出 (`response_format: json_object` + `json_schema`,DFA-guided 硬遮蔽)
- Whisper ASR (Metal 加速 forward pass) + Qwen3-TTS
- Top-k / top-p / 温度 / 重复惩罚

已知 regression / 优化中:
- Apple Silicon dense 在 c = 4 上吞吐低于 c = 1 (paged-batched 在小 m 下处于 crossover 之下)。c ≤ 4 默认仍是 per-token 模式,直到小 m 路径补齐为止。
- FP8 (Hopper / Blackwell) —— INT4 路径目前只占用 24% DRAM 峰值带宽,FP8 还没成为瓶颈。

## 路线图

完整路线图见 [docs/ROADMAP.md](docs/ROADMAP.md)。

近期:
- v0.1: CUDA + Apple Silicon 生产 release，含并发 benchmark
- v0.2: 更完整的 release 矩阵和长上下文 serving benchmark
- v0.3: 长上下文调优 (32k+)、更多架构 (Phi、DeepSeek、Gemma)

## License

MIT
