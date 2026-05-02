# ferrum-infer-rs

[![Crates.io](https://img.shields.io/crates/v/ferrum-cli.svg)](https://crates.io/crates/ferrum-cli)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/sizzlecar/ferrum-infer-rs/blob/main/LICENSE)

Rust 编写的生产级 LLM 推理引擎。单二进制，OpenAI 兼容 API，原生支持 Apple Silicon 与 CUDA。

[English](README.md)

## 项目简介

ferrum-infer-rs 是一个 Rust 原生的 LLM 推理引擎: 单二进制、无 Python、
OpenAI 兼容 HTTP API、秒级启动。

为单 GPU 服务器、边缘设备和 Apple Silicon 而设计 ——
Docker 镜像体积、冷启动时间、Python 工具链复杂度真正重要的场景。

## 性能亮点：Apple Silicon 并发

笔记本本地推理的硬骨头是「并发服务」。ferrum 在单请求上和主流引擎打平,并发越高优势越明显。同一台机器、同一份 `Q4_K_M` GGUF、同一份 OpenAI HTTP 压测脚本 —— 完整的可审阅报告(环境、脚本、原始 JSON 与日志)在 [`docs/bench/macos-2026-05-02/`](docs/bench/macos-2026-05-02/)。

**M1 Max 32 GB · Q4_K_M · 输出吞吐 (tok/s)** —— 多次运行取最优值,详细方差与重跑协议见 [bench 报告 § Methodology](docs/bench/macos-2026-05-02/README.md#methodology--why-two-reruns)。

| 模型 | c | ferrum | llama.cpp (b8960) | mistralrs (0.8.1) |
|---|---:|---:|---:|---:|
| LLaMA-3.1-8B | 1 | 29.1 | 28.7 | 30.2 |
| LLaMA-3.1-8B | 8 | **51.3** | 42.3 | 14.6 |
| LLaMA-3.1-8B | 16 | **96.7** | 67.2 | 23.3 |
| Qwen3-8B | 16 | **93.2** | 68.6 | 23.5 |
| Qwen3-30B-A3B (MoE) | 16 | 79.2¹ | 83.4 | panic² |

> ¹ ferrum MoE 在 c ≥ 8 时需要 `FERRUM_MOE_BATCHED=1 FERRUM_MOE_BATCHED_DECODE=1` (当前为 opt-in)。不开启的话 MoE c = 16 跌到 48 tok/s。² mistralrs 0.8.1 在 Qwen3-30B-A3B-Q4_K_M 上 PoisonError-panic (`add_request.rs:466`) —— 不是 ferrum 的问题。

> Qwen3-30B-A3B (MoE) 这一行是头条 —— 两个月前 Apple Silicon 上 Rust 引擎实质性缺失的就是这种模型。ferrum 通过 PR #81 把 `LlamaFamilyModel` 的 Phase-4 paged-KV 镜像到 `Qwen3MoeModel`,把对 llama.cpp 的差距从 51 → 80 tok/s 抹平。在 dense 8B 模型上 c = 16 ferrum 比 llama.cpp 快 +36–44%。

完整的 36-cell 矩阵(c = 1, 4, 8, 16,三引擎 × 三模型,含 TPOT / TTFT 分布)见 [bench 报告](docs/bench/macos-2026-05-02/README.md)。

## 性能亮点：NVIDIA GPUs (CUDA)

ferrum 拥有自研 CUDA decode runner,支持 INT4 Marlin。来自 RTX PRO 6000 (Blackwell) 的数据:

**Qwen3-4B**

| 模式 | Decode (tok/s) | 显存 |
|---|---:|---:|
| FP16 (eager) | 70.3 | ~8 GB |
| FP16 + CUDA Graphs | 82.9 (+18%) | ~8 GB |
| INT4 (GPTQ + Marlin) | **130.4 (+85%)** | **~2.5 GB (-69%)** |
| 4 并发 (INT4) | 124.2 | ~2.5 GB |

**TinyLlama-1.1B**

| Backend | Decode (tok/s) |
|---|---:|
| Candle | 126 |
| ferrum CUDA | **256.5 (+103%)** |

包含 vLLM 风格的全部调度特性: PagedAttention、continuous batching、FlashAttention-2 prefill、batched decode、自研 fused kernel、piecewise CUDA Graphs、NCCL tensor parallel。

## 横向对比

|  | ferrum | vLLM | llama.cpp | mistralrs |
|---|---|---|---|---|
| 语言 | Rust | Python+CUDA | C++ | Rust |
| 单二进制 | ✓ | ✗ (Docker) | ✓ | ✓ |
| Apple Silicon | ✓ (含 MoE) | ✗ | ✓ | 部分 (无 MoE) |
| CUDA | ✓ (自研) | ✓ (最强) | ✓ | ✓ |
| 并发服务 | ✓ | ✓ (最强) | ✓ | ✓ |
| Continuous batching | ✓ | ✓ | 部分 | ✓ |
| INT4 量化 | ✓ Marlin / Triton | GPTQ / AWQ | 仅 GGUF | 视情况 |
| OpenAI 兼容 API | ✓ | ✓ | ✓ | ✓ |
| 可作为库嵌入 | ✓ | ✗ | ✓ | ✓ |

## 快速开始

### Homebrew (macOS Apple Silicon、Linux x86_64)

```bash
brew tap sizzlecar/ferrum
brew install ferrum
ferrum --version
```

### 预编译二进制 (原始 tarball)

```bash
# Linux x86_64
curl -L https://github.com/sizzlecar/ferrum-infer-rs/releases/latest/download/ferrum-linux-x86_64.tar.gz | tar xz
./ferrum --help

# macOS Apple Silicon (Metal)
curl -L https://github.com/sizzlecar/ferrum-infer-rs/releases/latest/download/ferrum-macos-aarch64.tar.gz | tar xz
./ferrum --help
```

Linux x86_64 是 CPU 构建。macOS aarch64 是 Metal 构建(就是在 [Group A bench](docs/bench/macos-2026-05-02/README.md) 中 c=16 击败 llama.cpp 的那个 Metal 后端)。CUDA 用户请从源码构建。

### 从源码

```bash
# crates.io
cargo install ferrum-cli

# 或 git
cargo build --release -p ferrum-cli --bin ferrum
```

### 运行

```bash
# Gated 模型(如 Llama 3.x)需要设置 HF token
export HF_TOKEN=hf_your_token_here

# 直接对话
ferrum run qwen3:4b

# 或启动 OpenAI 兼容 API
ferrum serve --model qwen3:4b --port 8000
```

API 调用:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3:4b","messages":[{"role":"user","content":"Hello"}]}'
```

## 支持的模型

| 架构 | Apple Silicon | CUDA | INT4 (GPTQ) | Tensor Parallel |
|---|:---:|:---:|:---:|:---:|
| LLaMA (3.x, TinyLlama, Vicuna, Mistral) | ✓ | ✓ | ✓ | ✓ |
| Qwen3 dense (0.6B – 8B) | ✓ | ✓ | ✓ | ✓ |
| Qwen3-MoE (30B-A3B) | ✓ | — | — | — |
| Qwen2 / Qwen2.5 | ✓ | ✓ | ✓ | — |
| BERT (embeddings) | ✓ | — | — | — |
| Whisper ASR (tiny → large-v3-turbo) | ✓ | — | — | — |
| Qwen3-TTS (0.6B / 1.7B, 含声音克隆) | ✓ | — | — | — |
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

# 文本转语音 (含 ICL 声音克隆)
ferrum tts qwen3-tts "你好欢迎使用语音合成系统" -o output.wav
ferrum tts qwen3-tts "你好" --ref-audio ref.wav --ref-text "参考文本" -o clone.wav
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

# CUDA 加速 (NVIDIA, 需要 CUDA toolkit + nvcc)
cargo install ferrum-cli --features cuda
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
- Apple Silicon MoE 推理 (Qwen3-30B-A3B) —— c=16 与 llama.cpp 持平
- INT4 GPTQ with Marlin fused kernel (Blackwell + Ampere); 同时有 Triton w4a16
- Tensor parallelism (多 GPU NCCL, 持久化 per-rank 线程)
- Speculative decoding (`--spec-draft <MODEL>` DeepMind accept/reject)
- 结构化输出 (`response_format: json_object` + `json_schema`,DFA-guided 硬遮蔽)
- Whisper ASR (Metal 加速 forward pass) + Qwen3-TTS (声音克隆、流式)
- Top-k / top-p / 温度 / 重复惩罚

已知 regression / 优化中:
- Apple Silicon dense 在 c = 4 上吞吐低于 c = 1 (paged-batched 在小 m 下处于 crossover 之下)。c ≤ 4 默认仍是 per-token 模式,直到小 m 路径补齐为止。
- FP8 (Hopper / Blackwell) —— INT4 路径目前只占用 24% DRAM 峰值带宽,FP8 还没成为瓶颈。

## 路线图

完整路线图见 [docs/ROADMAP.md](docs/ROADMAP.md)。

近期:
- v0.1: Apple Silicon Group A 生产 release,含并发 benchmark (本次 PR)
- v0.2: 普及型硬件(RTX 4090) 上的 CUDA serving benchmark vs vLLM
- v0.3: 长上下文调优 (32k+)、更多架构 (Phi、DeepSeek、Gemma)

## License

MIT
