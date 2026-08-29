# ferrum-infer-rs

[![Crates.io](https://img.shields.io/crates/v/ferrum-cli.svg)](https://crates.io/crates/ferrum-cli)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/sizzlecar/ferrum-infer-rs/blob/main/LICENSE)

> Rust 原生 LLM 推理，用于 OpenAI 兼容的本地与私有服务。

**一个二进制，无需 Python runtime，支持 Apple Silicon Metal 与 NVIDIA CUDA 加速。**

[English](README.md)

## 快速开始

安装 Ferrum：

```bash
brew tap sizzlecar/ferrum

# macOS Apple Silicon
brew install ferrum

# Linux x86_64，NVIDIA CUDA sm89
brew install ferrum-cuda
```

下载权重前先检查安装的二进制：

```bash
ferrum doctor
```

直接运行模型：

```bash
# macOS Metal（GGUF）
ferrum run qwen3.5:4b-q4_k_m

# Linux CUDA（safetensors）
ferrum run qwen3.5:4b
```

Ferrum 不会静默选择模型。`run` 必须提供 MODEL；`serve` 必须提供 `--model`，
或者在 `ferrum.toml` 中有意设置 `default_model`。

通过 OpenAI 兼容 API 提供服务：

```bash
# macOS Metal
ferrum serve --model qwen3.5:4b-q4_k_m --served-model-name ferrum --port 8000

# Linux CUDA
ferrum serve --model qwen3.5:4b --served-model-name ferrum --port 8000

curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"ferrum","messages":[{"role":"user","content":"Hello"}]}'
```

`ferrum doctor <MODEL>` 只解析模型来源并打印下一条 `run`、`serve` 命令，
不会下载模型或启动推理 engine。

## 功能

- 一个 Rust 二进制同时提供 `ferrum run` 和 `ferrum serve`。
- 支持 OpenAI 兼容的 Chat Completions 与无状态 Responses API、流式输出、
  tools 和 structured output。
- 同一 runtime 覆盖 Apple Silicon Metal 与 NVIDIA CUDA。
- 支持 continuous batching、paged KV cache、prefix cache 和 typed admission。
- Metal 使用 GGUF，CUDA 使用 GPTQ/safetensors。
- v0.8 只覆盖语言模型推理。发布范围：Qwen3.5 4B、Qwen3.5 35B-A3B、
  Qwen3 30B-A3B 和 Llama 3.1 8B dense。详见[支持矩阵](docs/release/runtime-vnext/0.8.0/SUPPORT_MATRIX.md)。

## 性能快照

最新 R2 development `ferrum serve` 测量。前三行在 Metal 使用随机 64-token 输入、
128-token 输出，在 CUDA 使用 256 / 128。数值为 3 次 repeat 的平均 tok/s，`±` 为
95% 置信区间半宽。

| 模型 | M1 Max 32 GB Metal | RTX 4090 CUDA | L40S 48 GB CUDA smoke |
|---|---:|---:|---:|
| Qwen3.5 4B | c=16 · 61.9 ± 0.1 | c=32 · 241.3 ± 0.6 |  |
| Qwen3.5 35B-A3B | c=4 · 26.1 ± 0.2 | c=16 · 174.1 ± 1.0 |  |
| Qwen3 30B-A3B | c=16 · 39.6 ± 1.2 | c=32 · 214.9 ± 2.7 |  |
| Qwen3.8 27B AWQ INT4 |  | c=1 · 35.15 |  |
| [Qwen3.8 27B official block-FP8](https://huggingface.co/Qwen/Qwen3.8-27B-FP8/tree/017b9c7af6b5689d5dd426a76e0bc077eb5ca20a) |  |  | ready 119.49 s · c=1 · 16.53 · c=2 · 24.33 |

`c` 为服务端实际活跃并发。前三行均完成 100 请求 × 3 repeats，错误数为零。
[测量详情](docs/release/runtime-vnext/0.8.0/PERFORMANCE_REPORT.md)。

L40S block-FP8 行是 clean commit `ef8dbbee` 的开发 smoke：ready 为冷启动
`serve` 到就绪的时间；c=1 使用 1 次 warmup 和 3 个实测 256-input/32-output-token
请求，c=2 完成 4 个稳定性请求。数值为 usage 计数的聚合输出 tok/s，不是 RTX 4090
release 吞吐。AWQ INT4 行使用不同 checkpoint 和量化路径。

## OpenAI 兼容 API

Ferrum 支持：

- chat completions 与 streaming usage
- 无状态 Responses 文本、流式输出、usage 与 function tools
- `auto`、`none`、`required` 或指定函数的 function tools
- `json_object` 与 strict `json_schema` structured output
- 多轮会话、prefix cache 与 session cache
- typed 并发、内存和 scheduler 控制

精确请求契约见 [OpenAI API 兼容说明](docs/openai-api-compatibility.md)，prefix
与 session cache 配置见 [cache 产品说明](docs/cache-product.md)。

## 安装

Homebrew：

```bash
brew tap sizzlecar/ferrum

# macOS Apple Silicon Metal
brew install ferrum

# Linux x86_64 CUDA sm89
brew install ferrum-cuda
```

预编译 release tarball：

```bash
# Linux x86_64 CUDA sm89
curl -L https://github.com/sizzlecar/ferrum-infer-rs/releases/latest/download/ferrum-linux-x86_64-cuda-sm89.tar.gz | tar xz
LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-} ./ferrum --version

# macOS Apple Silicon Metal
curl -L https://github.com/sizzlecar/ferrum-infer-rs/releases/latest/download/ferrum-macos-aarch64.tar.gz | tar xz
./ferrum --version
```

从 crates.io 安装：

```bash
# macOS Apple Silicon Metal
cargo install ferrum-cli --version 0.8.2 --locked --features metal

# NVIDIA CUDA
cargo install ferrum-cli --version 0.8.2 --locked \
  --features cuda,vllm-moe-marlin,vllm-paged-attn-v2
```

官方预编译 CUDA 资产的目标为 `sm89`。CUDA 安装需要兼容的 NVIDIA driver、
CUDA runtime 和 NCCL runtime。

## 架构

- 契约：`ferrum-types`、`ferrum-interfaces`
- 执行：`ferrum-engine`、`ferrum-scheduler`、`ferrum-kv`、`ferrum-sampler`
- 模型与计算：`ferrum-models`、`ferrum-kernels`、`ferrum-native-ops`、`ferrum-quantization`
- 产品入口：`ferrum-cli`、`ferrum-server`、`ferrum-tokenizer`
- 验证：`ferrum-bench-core`、`ferrum-testkit`

## License

MIT
