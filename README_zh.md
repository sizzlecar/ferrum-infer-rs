# ferrum-infer-rs

[![Crates.io](https://img.shields.io/crates/v/ferrum-cli.svg)](https://crates.io/crates/ferrum-cli)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/sizzlecar/ferrum-infer-rs/blob/main/LICENSE)

> Rust 原生 LLM 推理，用于 OpenAI 兼容的本地与私有 serving。

**一个二进制。Serving 路径不需要 Python runtime。Apple Silicon 与 NVIDIA CUDA 硬件加速。**

Ferrum 是一个 Rust 原生的 LLM 推理引擎，用于运行和部署 OpenAI 兼容的 transformer LLM 服务。
它面向希望部署可预测、保留实际 serving 性能，并使用干净 Rust-native runtime 的开发者和团队，覆盖本地、边缘、工作站和生产推理场景。

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

在 24 GB RTX 4090 上运行已认证的 27B 级 CUDA agent 模型：

```bash
ferrum serve --model gemma3:27b-gptq --max-num-seqs 16 --kv-capacity 400 --port 8000
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

当前 0.7.7 source release gates：

| 目标 | 模型 / workload | 结果 | 证据 |
| --- | --- | --- | --- |
| Apple Silicon Metal source gate | Llama-3.1-8B、Qwen3-8B、Qwen3-30B-A3B GGUF；覆盖 `ferrum run`、`ferrum serve`、tool calls、stream、stateful loop 和 `16/64` 吞吐 cells | `FERRUM GATE metal PASS`；Qwen3-30B-A3B c=16 当前 `68.5 tok/s`，`32/32` 完成，`0` errors | [`docs/release/g0/0.7.7/metal/metal-readme/summary.md`](docs/release/g0/0.7.7/metal/metal-readme/summary.md) |
| CUDA RTX 4090 source gate | `Qwen/Qwen3-30B-A3B-GPTQ-Int4`，random `256/128`，c=1/4/16/32，`n_repeats=3` | `FERRUM GATE cuda-full PASS`；c=1/4/16/32 candidate `164.2` / `353.3` / `636.9` / `706.0 tok/s`；每个 cell `384/384` 完成，`0` errors | [`docs/release/g0/0.7.7/cuda-full/summary.json`](docs/release/g0/0.7.7/cuda-full/summary.json) |
| CUDA RTX 4090 dense source gate | `hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4`，random `256/128`，c=1/4/16/32，`n_repeats=3` | `FERRUM GATE cuda-llama-dense PASS`；c=1/4/16/32 output `122.9` / `324.3` / `640.2` / `745.6 tok/s`；每个 cell `288/288` 完成，`0` errors | [`docs/release/g0/0.7.7/cuda-llama-dense/bench-serve.json`](docs/release/g0/0.7.7/cuda-llama-dense/bench-serve.json) |

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
- 当前 0.7.7 G0 source artifacts：[`docs/release/g0/0.7.7/`](docs/release/g0/0.7.7/)
- 2026-06 模型覆盖证据：[`docs/goals/model-coverage-2026-06-12/`](docs/goals/model-coverage-2026-06-12/)
- Apple Silicon 回归 gate：`scripts/metal_readme_regression.py` 和 `scripts/release/validate_metal_readme_regression.py`
- OpenAI API 兼容性：[`docs/openai-api-compatibility.md`](docs/openai-api-compatibility.md)
- Prefix/session cache 产品面：[`docs/cache-product.md`](docs/cache-product.md)
- 模块状态记录：[`docs/status/`](docs/status/)

## 支持的模型

| 架构 | Apple Silicon | CUDA | INT4 (GPTQ) | Tensor Parallel |
|---|:---:|:---:|:---:|:---:|
| LLaMA (3.x, TinyLlama, Vicuna, Mistral) | ✓ | ✓ | ✓ | ✓ |
| Qwen3 dense (0.6B – 32B) | ✓ | ✓ | ✓ | ✓ |
| Qwen3-MoE (30B-A3B, Coder-30B-A3B) | ✓ | ✓ | ✓ | — |
| Gemma 3 text (1B, 27B) | 1B GGUF | ✓ | 27B GPTQ | — |
| Qwen2 / Qwen2.5 | ✓ | ✓ | ✓ | — |
| DeepSeek-R1 (0528-Qwen3-8B; Distill-Qwen-14B/32B; Distill-Llama-70B) | ✓ | ✓ | ✓ | 70B layer-split |
| Mistral Small 3.2 / Magistral Small (24B) | ✓ | — | — | — |
| BERT (embeddings) | ✓ | — | — | — |
| Whisper ASR (tiny → large-v3-turbo) | ✓ | — | — | — |
| Qwen3-TTS (0.6B / 1.7B) | ✓ | — | — | — |
| CLIP / Chinese-CLIP / SigLIP (文本 + 图像) | ✓ | — | — | — |

### 模型覆盖认证 (2026-06)

下面的行有保存下来的 gate 证据，不是只声明“能加载”。W1 行通过了标记
lane 的覆盖阶梯：chat-template 与 `transformers` golden byte-equal (L0)、
temp 0 known-answer 10/10、多轮 KV reuse、stream==non-stream、自然 EOS、
custom stop 和 max_tokens 机制 (L2/L3)，以及 agent gate：required tool-call
10/10 + strict `json_schema` 20/20 (L4, "agent-grade")。W2 新增 Gemma 3
27B CUDA，并通过 L5 并发 gate 与同卡 llama.cpp 性能 floor。证据目录：
[`docs/goals/model-coverage-2026-06-12/artifacts/`](docs/goals/model-coverage-2026-06-12/artifacts/)。
lane 划分：**GGUF 走 Metal lane；CUDA 走 GPTQ/safetensors**。

| 模型 | 别名 | Metal (GGUF Q4_K_M) | CUDA | Agent-grade |
|---|---|:---:|:---:|:---:|
| Qwen3-Coder-30B-A3B-Instruct | `qwen3-coder:30b[-q4_k_m/-gptq]` | ✓ | GPTQ: 已知问题¹ | ✓ (Metal) |
| Gemma 3 27B (text) | `gemma3:27b`, `gemma3:27b-gptq` | 1B GGUF smoke；27B waived⁵ | ✓ GPTQ + L5⁵ | ✓ |
| DeepSeek-R1-0528-Qwen3-8B | `deepseek-r1:8b[-q4_k_m]` | ✓ | ✓ BF16 | ✓ |
| DeepSeek-R1-Distill-Qwen-32B | `deepseek-r1:32b[-q4_k_m/-gptq]` | 32 GB Mac: 不实用² | ✓ GPTQ | tools ✓ / schema³ |
| Qwen3-14B / Qwen3-32B | `qwen3:14b/32b[-q4_k_m/-gptq]` | 14B ✓ / 32B² | 32B ✓ GPTQ | 14B ✓ / 32B³ |
| Qwen2.5-Coder-32B-Instruct | `qwen2.5-coder:32b[-q4_k_m/-gptq]` | —² | ✓ GPTQ | ✓ (CUDA) |
| Mistral Small 3.2 (24B) | `mistral-small:24b-q4_k_m` | ✓ | — | ✓ |
| Magistral Small (24B, reasoning) | `magistral:24b-q4_k_m` | ✓ | — | ✓ |
| DeepSeek-R1-Distill-Llama-70B | `OPEA/...-70B-int4-gptq-sym-inc` | — | ✓ 2×4090 layer-split | chat/reasoning grade⁴ |
| Devstral Small 2 (24B) | — | **不支持** (`mistral3` arch: YaRN-from-GGUF + attention temperature scaling；loader 会明确拒绝) | | |

¹ jart25 GPTQ chat 在 CUDA 上会输出空答案（开放问题；Metal GGUF 和 CUDA random-context bench 正常）。
² 32B dense 在 32 GB Mac 上会逐 token 从 SSD 重读被驱逐权重（约 0.14 tok/s），不适合作为实际部署；使用 CUDA lane。
³ strict `json_schema` 在 32B-GPTQ 上偶发 500（开放问题）；required tool-calls 是 10/10。
⁴ R1-distill template 会强制 `<think>`，70B 会把 tool-call JSON 写进 think block；适合 chat/reasoning，不适合作为 tool-calling 认证。
⁵ Gemma 3 W2 最终 validator 行是
`MODEL_COVERAGE_W2 GOAL PASS: docs/goals/model-coverage-2026-06-12`。CUDA L5 覆盖 random `256/128`、c=1/4/16/32、每个 cell 100 prompts × 3 repeats、零错误、usage token 计数。24 GB RTX 4090 的 c=32 客户端 lane 使用产品 CLI admission `--max-num-seqs 16` 与 `--kv-capacity 400`。同卡 llama.cpp 对比为 `0.500260x`，刚过 floor，因此这是正确性/并发认证并带 known performance gap，不是性能优化完成声明。27B GGUF Metal 在降级的 32 GB Mac 上 waived；Gemma 3 GGUF 架构覆盖由 1B Q4_K_M Metal smoke artifact 固定。

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
- OpenAI-style function tool calling，包含 required tool calls 和 streaming `tool_calls` deltas
- 自研 CUDA decode runner (Qwen3, LLaMA): 比 Candle 快 2×
- Gemma 3 27B GPTQ on CUDA，覆盖 tool-call、strict-schema、streaming、多轮和 c=32 client pressure gates
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
- v0.3: 长上下文调优 (32k+)、更多架构和更完整的 Gemma 覆盖

## License

MIT
