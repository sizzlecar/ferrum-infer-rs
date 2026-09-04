<p align="center">
  <a href="https://ferrum.pandaailabs.com/zh/">
    <img src="assets/brand/ferrum-lockup.svg" alt="Ferrum — Local LLM Runtime" width="520">
  </a>
</p>

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
ferrum --version
ferrum --help
ferrum doctor
```

只执行与你的平台对应的命令。`doctor` 只显示模型来源映射，不下载权重，也不启动推理引擎。

### macOS Apple Silicon

首次运行会下载约 **2.55 GiB**。下载耗时取决于本机到 Hugging Face 的网络链路；看到进度输出后
再判断进程是否卡住。

```bash
ferrum doctor qwen3.5:4b-q4_k_m
ferrum run qwen3.5:4b-q4_k_m --disable-thinking
```

### Linux NVIDIA CUDA

首次运行会下载约 **8.7 GiB** 的仓库权重。

```bash
ferrum doctor qwen3.5:4b
ferrum run qwen3.5:4b --disable-thinking
```

Ferrum 不会静默选择模型。`run` 必须提供 MODEL；`serve` 必须提供 `--model`，
或者在 `ferrum.toml` 中有意设置 `default_model`。

通过 OpenAI 兼容 API 提供服务：

```bash
# macOS Metal
ferrum serve --model qwen3.5:4b-q4_k_m --served-model-name ferrum --disable-thinking --port 8000

# Linux CUDA
ferrum serve --model qwen3.5:4b --served-model-name ferrum --disable-thinking --port 8000

curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"ferrum","messages":[{"role":"user","content":"Reply with a short hello from Ferrum."}],"max_tokens":32}'
```

正常时请求会返回 HTTP 200 和非空的 assistant 回答。除非显式设置
`--max-model-len`，Ferrum 会使用模型自身的上下文上限；显式上限必须容纳渲染后的
输入与请求的输出预算之和。

快速开始默认使用 `--disable-thinking`，让首次回答简短直接。删除该参数即可恢复模型模板默认的
推理行为；HTTP 请求也可以通过 `chat_template_kwargs.enable_thinking` 覆盖服务端默认值。

`ferrum doctor <MODEL>` 会解析模型来源并打印下一条 `run`、`serve` 命令，
不会下载模型或启动推理引擎。

## 功能

- 一个 Rust 二进制同时提供 `ferrum run` 和 `ferrum serve`。
- 支持 OpenAI 兼容的 Chat Completions 与无状态 Responses API、流式输出、
  tools 和 structured output。
- 同一 runtime 覆盖 Apple Silicon Metal 与 NVIDIA CUDA。
- 支持 continuous batching、paged KV cache、prefix cache 和 typed admission。
- Metal 使用 GGUF，CUDA 使用 GPTQ/safetensors。
- v0.8 只覆盖语言模型推理。发布范围：Qwen3.5 4B、Qwen3.5 35B-A3B、
  Qwen3 30B-A3B 和 Llama 3.1 8B dense。

## 性能快照

最新 R2 development `ferrum serve` 测量。前三行在 Metal 使用随机 64-token 输入、
128-token 输出，在 CUDA 使用 256 / 128。数值为 3 次 repeat 的平均 tok/s，`±` 为
95% 置信区间半宽。

| 模型 | M1 Max 32 GB Metal | RTX 4090 CUDA | L40S 48 GB CUDA |
|---|---:|---:|---:|
| Qwen3.5 4B | c=16 · 61.9 ± 0.1 | c=32 · 241.3 ± 0.6 |  |
| Qwen3.5 35B-A3B | c=4 · 26.1 ± 0.2 | c=16 · 174.1 ± 1.0 |  |
| Qwen3 30B-A3B | c=16 · 39.6 ± 1.2 | c=32 · 214.9 ± 2.7 |  |
| Qwen3.8 27B AWQ INT4 |  | c=4 · 78.19 ± 0.04 · c=16 · 115.12 ± 1.18 · c=32 · 115.18 ± 0.97 |  |
| [Qwen3.8 27B official block-FP8](https://huggingface.co/Qwen/Qwen3.8-27B-FP8/tree/017b9c7af6b5689d5dd426a76e0bc077eb5ca20a) |  |  | ready 80.91 s · c=1 · 15.23 ± 0.19 · c=8 · 41.75 ± 1.26 · c=32 · 49.75 ± 0.95 |
| [Qwen3.6 27B official block-FP8](https://huggingface.co/Qwen/Qwen3.6-27B-FP8/tree/e89b16ebf1988b3d6befa7de50abc2d76f26eb09) |  |  | ready 93.39 s · c=1 · 15.15 ± 0.05 · c=8 · 42.37 ± 3.04 · c=32 · 50.38 ± 0.29 |
| [Qwen3.6 35B-A3B official block-FP8](https://huggingface.co/Qwen/Qwen3.6-35B-A3B-FP8/tree/95a723d08a9490559dae23d0cff1d9466213d989) |  |  | ready 69.62 s · c=1 · 45.01 ± 7.54 · c=8 · 92.78 ± 2.03 · c=32 · 92.78 ± 0.84 |
| [GPT-OSS 20B official MXFP4](https://huggingface.co/openai/gpt-oss-20b/tree/6cee5e81ee83917806bbde320786a8fb61efebee) |  | ready 23.65 s · c=1 · 61.49 ± 4.19 · c=8 · 77.16 ± 0.70 · c=32 · 77.23 ± 4.37 |  |
| [Gemma 4 12B official W4A16 CT](https://huggingface.co/google/gemma-4-12B-it-qat-w4a16-ct/tree/1d2c2d7f2466070e69d6fb3fd5ce9a7d75f2f6ee) |  | ready 24.90 s · c=1 · 9.79 ± 0.01 · c=8 · 52.91 ± 0.88 · c=32 · 66.05 ± 6.78 |  |

`c` 为服务端实际活跃并发。前三行均完成 100 请求 × 3 repeats，错误数为零。

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
curl --fail --location --remote-name https://github.com/sizzlecar/ferrum-infer-rs/releases/download/v0.8.6/ferrum-linux-x86_64-cuda-sm89.tar.gz
curl --fail --location --remote-name https://github.com/sizzlecar/ferrum-infer-rs/releases/download/v0.8.6/ferrum-linux-x86_64-cuda-sm89.tar.gz.sha256
sha256sum --check ferrum-linux-x86_64-cuda-sm89.tar.gz.sha256
tar -xzf ferrum-linux-x86_64-cuda-sm89.tar.gz
LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-} ./ferrum --version

# macOS Apple Silicon Metal
curl --fail --location --remote-name https://github.com/sizzlecar/ferrum-infer-rs/releases/download/v0.8.6/ferrum-macos-aarch64.tar.gz
curl --fail --location --remote-name https://github.com/sizzlecar/ferrum-infer-rs/releases/download/v0.8.6/ferrum-macos-aarch64.tar.gz.sha256
shasum -a 256 --check ferrum-macos-aarch64.tar.gz.sha256
tar -xzf ferrum-macos-aarch64.tar.gz
./ferrum --version
```

从 crates.io 安装 Metal build：

```bash
# macOS Apple Silicon Metal
cargo install ferrum-cli --version 0.8.6 --locked --features metal
```

官方预编译 CUDA 资产的目标为 `sm89`。CUDA 安装需要兼容的 NVIDIA driver、
CUDA runtime 和 NCCL runtime。CUDA 源码构建还需要与 Ferrum 匹配的
native-operator set，因此受支持的安装路径是预编译 CUDA tarball 或 Homebrew formula。

## 架构

- 契约：`ferrum-types`、`ferrum-interfaces`
- 执行：`ferrum-engine`、`ferrum-scheduler`、`ferrum-kv`、`ferrum-sampler`
- 模型与计算：`ferrum-models`、`ferrum-kernels`、`ferrum-native-ops`、`ferrum-quantization`
- 产品入口：`ferrum-cli`、`ferrum-server`、`ferrum-tokenizer`
- 验证：`ferrum-bench-core`、`ferrum-testkit`

## License

MIT
