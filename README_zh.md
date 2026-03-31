# Ferrum Infer

[![Crates.io](https://img.shields.io/crates/v/ferrum-cli.svg)](https://crates.io/crates/ferrum-cli)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

纯 Rust 实现的 LLM 推理引擎。从 Hugging Face 加载模型，本地对话或通过 OpenAI 兼容 API 提供服务。单二进制文件，无需 Python，无运行时依赖。

[English](README.md)

## 快速开始

前置条件：Rust stable 工具链。

访问受限模型（如 Llama 3.2）需先设置 Hugging Face token：
```bash
export HF_TOKEN=hf_your_token_here
```

```bash
# 编译
cargo build --release -p ferrum-cli --bin ferrum

# 下载模型
./target/release/ferrum pull qwen3:0.6b

# 对话
./target/release/ferrum run qwen3:0.6b

# 或启动 API 服务
./target/release/ferrum serve --model qwen3:0.6b --port 8000
```

## 支持的模型

| 别名 | 模型 | 架构 | CUDA Runner |
|------|------|------|-------------|
| `qwen3:0.6b` / `1.7b` / `4b` | Qwen3 | Qwen3 | 支持 |
| `qwen2.5:0.5b` / `1.5b` / `3b` / `7b` | Qwen2.5-Instruct | Qwen2 | — |
| `llama3.2:1b` / `3b` | Llama-3.2-Instruct | LLaMA | 支持 |
| `tinyllama` | TinyLlama-1.1B-Chat | LLaMA | 支持 |

GPTQ INT4 量化模型自动检测，使用 Marlin fused 内核：
```bash
./target/release/ferrum run JunHowie/Qwen3-4B-GPTQ-Int4
```

也可直接使用 Hugging Face 模型 ID：
```bash
./target/release/ferrum run Qwen/Qwen3-0.6B
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
| `ferrum embed <model>` | 生成 BERT 向量 |

## API 接口

```bash
# Chat completions（OpenAI 兼容）
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3:0.6b","messages":[{"role":"user","content":"你好"}]}'

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

### 核心优化

- **自定义 CUDA decode runner**：绕过 candle 的 decode 热路径（Qwen3 + LLaMA）
- **INT4 量化**：GPTQ 模型自动检测，Marlin fused INT4×FP16 内核
- **Batched attention 内核**：单次 launch 处理所有 batch 项（SM 利用率 17%→67%）
- **Batched RoPE**：单次 launch + per-item position 数组
- **自定义 CUDA 内核**：fused RmsNorm、SiLU×mul、RoPE、decode attention（统一 stream 零同步）
- **Flash Decoding**：长上下文 split-K（KV > 256 时自动启用）
- **Batch decode**：batched cuBLAS GEMM + batched attention 支持并发请求
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
- Top-k / Top-p / Temperature / 重复惩罚采样

## 路线图

- **张量并行** — 多 GPU NCCL
- **推测解码** — draft model 验证
- **更多模型架构** — Mistral、Phi、DeepSeek 等
- **Qwen2 CUDA runner** — 同 LLaMA 模式

详见 [docs/ROADMAP.md](docs/ROADMAP.md)。

## 编译选项

```bash
# 仅 CPU（默认）
cargo build --release -p ferrum-cli

# 启用 Metal 加速（macOS）
cargo build --release -p ferrum-cli --features metal

# 启用 CUDA 加速（NVIDIA，需要 CUDA Toolkit）
cargo build --release -p ferrum-cli --features cuda

# 启用 CUDA + Marlin INT4 内核（需要 nvcc，SM >= 8.0）
cargo build --release -p ferrum-cli --features cuda,marlin
```

前置条件：Rust stable 工具链。

## 项目结构

```
crates/
├── ferrum-types          # 共享类型定义
├── ferrum-interfaces     # 核心 trait 契约（ComputeBackend, KernelOps, ModelExecutor）
├── ferrum-runtime        # 后端实现（Candle, CPU）
├── ferrum-engine         # Metal 内核、模型编排
├── ferrum-models         # 模型架构（LLaMA, Qwen2, Qwen3, BERT）
├── ferrum-cuda-kernels   # 自定义 CUDA 内核 + decode runner
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
