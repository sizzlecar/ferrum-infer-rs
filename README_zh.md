# Ferrum Infer

[![Crates.io](https://img.shields.io/crates/v/ferrum-cli.svg)](https://crates.io/crates/ferrum-cli)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

纯 Rust 实现的 LLM 推理引擎。从 Hugging Face 加载模型，本地对话或通过 OpenAI 兼容 API 提供服务。单二进制文件，无需 Python，无运行时依赖。

[English](README.md)

## 安装

```bash
# 从 crates.io 安装
cargo install ferrum-cli

# 或从源码编译
cargo build --release -p ferrum-cli --bin ferrum
```

## 快速开始

访问受限模型（如 Llama 3.2）需先设置 Hugging Face token：
```bash
export HF_TOKEN=hf_your_token_here
```

```bash
# 下载模型
ferrum pull qwen3:0.6b

# 对话
ferrum run qwen3:0.6b

# 或启动 API 服务
ferrum serve --model qwen3:0.6b --port 8000
```

## 支持的架构

任何使用以下架构的 Hugging Face 模型都可以直接运行：

### 文本生成

| 架构 | CUDA Decode | INT4 (GPTQ) | 张量并行 | 示例模型 |
|------|-------------|-------------|---------|----------|
| **LLaMA** | 支持 | 支持 | 支持 | Llama-3.x, TinyLlama, Vicuna, Alpaca, ... |
| **Qwen3** | 支持 | 支持 | 支持 | Qwen3-0.6B ~ 4B |
| **Qwen2** | — | — | — | Qwen2.5-Instruct-0.5B ~ 7B |

### 语音转文字（Whisper ASR）

| 架构 | Metal | CUDA | 示例模型 |
|------|-------|------|----------|
| **Whisper** | 支持 | — | whisper-tiny, whisper-base, whisper-small, whisper-medium, whisper-large-v3, **whisper-turbo**（推荐） |

### 文字转语音（Qwen3-TTS）

| 架构 | Metal | CPU | 声音克隆 | 示例模型 |
|------|-------|-----|---------|----------|
| **Qwen3-TTS** | 支持 | 支持 | 支持（ICL） | Qwen3-TTS-12Hz-0.6B-Base |

### 向量化（文本 + 图片）

| 架构 | 模态 | 向量维度 | 示例模型 |
|------|------|---------|----------|
| **CLIP** | 文本 + 图片 | 512/768 | openai/clip-vit-base-patch32 |
| **Chinese-CLIP** | 文本 + 图片 | 512 | OFA-Sys/chinese-clip-vit-base-patch16 |
| **SigLIP** | 文本 + 图片 | 768 | google/siglip-base-patch16-224 |
| **BERT** | 文本 | 768 | google-bert/bert-base-chinese |

```bash
# 文本生成
ferrum run Qwen/Qwen3-4B
ferrum run llama3.2:3b

# 语音转文字（支持 WAV/M4A/MP3/FLAC，自动 ffmpeg 转码）
ferrum transcribe whisper-turbo 录音.m4a -l zh
ferrum transcribe whisper-turbo meeting.wav -l en

# 文字转语音
ferrum tts qwen3-tts "你好欢迎使用语音合成系统" -o output.wav

# 声音克隆（ICL 模式，5 秒参考音频即可克隆任何声音）
ferrum tts qwen3-tts "你好" --ref-audio ref.wav --ref-text "参考文本" -o clone.wav

# Whisper API 服务（OpenAI 兼容）
ferrum serve whisper-turbo
curl localhost:8000/v1/audio/transcriptions -F "file=@audio.wav" -F "language=zh"

# 向量化（文本 + 图片）
ferrum embed OFA-Sys/chinese-clip-vit-base-patch16 --text "海边日落"
ferrum embed google/siglip-base-patch16-224 --image photo.jpg

# Embedding API 服务
ferrum serve --model OFA-Sys/chinese-clip-vit-base-patch16
curl localhost:8000/v1/embeddings -d '{"model":"clip","input":"你好"}'
curl localhost:8000/v1/embeddings -d '{"model":"clip","input":{"image":"/path/to/photo.jpg"}}'
```

## 命令

| 命令 | 说明 |
|------|------|
| `ferrum run <model>` | 交互式对话 |
| `ferrum serve --model <model>` | 启动 OpenAI 兼容 HTTP 服务 |
| `ferrum stop` | 停止服务 |
| `ferrum pull <model>` | 从 Hugging Face 下载模型 |
| `ferrum list` | 查看已缓存模型 |
| `ferrum bench <model>` | 性能基准测试 |
| `ferrum transcribe <model> <audio>` | 语音转文字（Whisper，支持 WAV/M4A/MP3） |
| `ferrum tts <model> <text>` | 文字转语音（Qwen3-TTS，`--ref-audio` 声音克隆） |
| `ferrum embed <model>` | 生成向量（BERT/CLIP/SigLIP，文本 + 图片） |

## API 接口

```bash
# Chat completions（OpenAI 兼容）
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3:0.6b","messages":[{"role":"user","content":"你好"}]}'

# 语音转文字（OpenAI 兼容，multipart form）
curl http://localhost:8000/v1/audio/transcriptions \
  -F "file=@audio.wav" -F "language=zh"

# 向量化
curl http://localhost:8000/v1/embeddings \
  -d '{"model":"clip","input":"你好"}'

# 模型列表
curl http://localhost:8000/v1/models

# 健康检查
curl http://localhost:8000/health
```

## 性能测试

测试环境：**RTX PRO 6000 (Blackwell)**

### Qwen3-4B

| 模式 | FP16 | INT4 (GPTQ + Marlin) |
|------|------|----------------------|
| 单请求 decode | 88.1 tok/s | **130.4 tok/s (+48%)** |
| 4 并发 (batch decode) | 109.4 tok/s | **124.2 tok/s** |
| 显存占用 | ~8 GB | **~2.5 GB (-69%)** |

### TinyLlama-1.1B（Llama 架构）

| 模式 | Candle | CUDA Runner |
|------|--------|-------------|
| Decode | 126 tok/s | **256.5 tok/s (+103%)** |

### 张量并行（多 GPU）

| 配置 | Qwen3-4B FP16 |
|------|---------------|
| 单卡 | 82.3 tok/s (TPOT 12.1ms) |
| 双卡 TP | 26.1 tok/s (TPOT 38.4ms) |

> TP decode 使用持久化 per-rank 线程 + NCCL all-reduce。当前瓶颈为 PCIe 互联延迟（~0.44ms × 72 次 NCCL 调用/步）。TP 主要适用于单卡放不下的大模型，或 NVLink 互联场景。

### Whisper ASR（Apple Silicon Metal）

| 模型 | 5 分钟音频 | 实时率 |
|------|-----------|--------|
| whisper-large-v3-turbo | **~72s** | **4.2 倍实时** |
| whisper-tiny | ~20s | 15 倍实时 |

> 自研 Whisper 前向推理 + rustfft STFT，mel 精度与 Python whisper 完全一致。完整解码管线：带时间戳的顺序解码、温度回退、压缩率检测。

### Qwen3-TTS（Apple Silicon Metal）

| 文本 | 音频时长 | 耗时 | 实时率 |
|------|---------|------|--------|
| 29 字中文 | 4.6s | **11.3s** | **2.8 倍实时** |
| 声音克隆（ICL，5s 参考音频） | 5.3s | 13.1s | 2.5 倍实时 |

> 全 Metal fused transformer 管线：自研 GEMM（64×32 simdgroup tiles）、fused residual+norm、flash attention + layer_scale。完整 Mimi vocoder（8 层 pre-transformer）。Apple Silicon 统一内存零拷贝。

### 核心优化

- **自定义 CUDA decode runner**：绕过 candle 的 decode 热路径（Qwen3 + LLaMA）
- **INT4 量化**：GPTQ 模型自动检测，Marlin fused INT4×FP16 内核
- **张量并行**：持久化 per-rank 线程、Barrier 同步、NCCL all-reduce（Megatron-LM 模式）
- **Batched attention 内核**：单次 launch 处理所有 batch 项（SM 利用率 17%→67%）
- **Batched RoPE**：单次 launch + per-item position 数组
- **自定义 CUDA 内核**：fused RmsNorm、SiLU×mul、RoPE、decode attention（统一 stream 零同步）
- **Flash Decoding**：长上下文 split-K（KV > 256 时自动启用）
- **Batch decode**：batched cuBLAS GEMM + batched attention 支持并发请求
- **Metal TTS 管线**：全 Metal fused transformer，talker（28 层）+ SubTalker（5 层）+ vocoder（8 层），缓存 GPU buffer，fused residual+norm 内核，layer_scale 支持
- **TTS 声音克隆**：ICL 提示 + 说话人编码器（ECAPA-TDNN）+ 语音分词器（Mimi RVQ）
- **Paged KV attention**：GPU block pool + block-table 间接寻址
- **双缓冲 residual**：跨层 norm 融合（-108 次 kernel launch）

## 当前状态

已完成：
- CLI 对话、HTTP 服务（流式输出）、性能基准测试
- Qwen3、Qwen2/2.5、LLaMA 3.x、TinyLlama 架构
- 自定义 CUDA decode runner（Qwen3 + LLaMA，2x 加速）
- Metal GPU 加速（macOS）、CUDA（NVIDIA）、CPU
- INT4 GPTQ 量化 + Marlin fused kernel（Blackwell 兼容）
- FlashAttention-2 prefill + 自定义 CUDA decode runner
- Paged KV cache + block 回收
- 连续批处理 + batch decode
- 张量并行（多 GPU NCCL，自动检测 GPU 数量）
- CLIP/Chinese-CLIP/SigLIP 向量化（文本 + 图片，`/v1/embeddings` API）
- Whisper 语音识别（Metal 加速，`/v1/audio/transcriptions` API）
- Qwen3-TTS 语音合成（Metal 加速，ICL 声音克隆）
- 多格式音频支持（WAV/M4A/MP3/FLAC，自动 ffmpeg 转码）
- Top-k / Top-p / Temperature / 重复惩罚采样

## 路线图

- **推测解码** — draft model 验证
- **更多模型架构** — Mistral、Phi、DeepSeek 等
- **Qwen2 CUDA runner** — 同 LLaMA 模式

详见 [docs/ROADMAP.md](docs/ROADMAP.md)。

## 编译选项

```bash
# 仅 CPU（默认）
cargo install ferrum-cli

# 启用 Metal 加速（macOS）
cargo install ferrum-cli --features metal

# 启用 CUDA 加速（NVIDIA，需要 CUDA Toolkit + nvcc）
cargo install ferrum-cli --features cuda
```

或从源码编译：
```bash
cargo build --release -p ferrum-cli                    # CPU
cargo build --release -p ferrum-cli --features metal   # Metal (macOS)
cargo build --release -p ferrum-cli --features cuda    # CUDA (NVIDIA)
cargo build --release -p ferrum-cli --features cuda    # 多卡自动检测
```

前置条件：Rust stable 工具链。

## 项目结构

```
crates/
├── ferrum-types          # 共享类型定义
├── ferrum-interfaces     # 核心 trait 契约（ComputeBackend, KernelOps, ModelExecutor）
├── ferrum-runtime        # 后端实现（Candle, CPU）
├── ferrum-engine         # Metal 内核、模型编排
├── ferrum-models         # 模型架构（LLaMA, Qwen2, Qwen3, BERT, Whisper）
├── ferrum-kernels   # 自定义 CUDA 内核 + decode runner
├── ferrum-tokenizer      # 分词器
├── ferrum-sampler        # 采样策略
├── ferrum-scheduler      # 请求调度
├── ferrum-kv             # KV 缓存管理
├── ferrum-server         # HTTP API 服务
├── ferrum-cli            # CLI 二进制
└── ferrum-testkit        # 测试工具
```

## 许可证

MIT
