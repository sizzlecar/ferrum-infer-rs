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

正常时请求会返回 HTTP 200 和非空的 assistant 回答。

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
curl --fail --location --remote-name https://github.com/sizzlecar/ferrum-infer-rs/releases/download/v0.8.4/ferrum-linux-x86_64-cuda-sm89.tar.gz
curl --fail --location --remote-name https://github.com/sizzlecar/ferrum-infer-rs/releases/download/v0.8.4/ferrum-linux-x86_64-cuda-sm89.tar.gz.sha256
sha256sum --check ferrum-linux-x86_64-cuda-sm89.tar.gz.sha256
tar -xzf ferrum-linux-x86_64-cuda-sm89.tar.gz
LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-} ./ferrum --version

# macOS Apple Silicon Metal
curl --fail --location --remote-name https://github.com/sizzlecar/ferrum-infer-rs/releases/download/v0.8.4/ferrum-macos-aarch64.tar.gz
curl --fail --location --remote-name https://github.com/sizzlecar/ferrum-infer-rs/releases/download/v0.8.4/ferrum-macos-aarch64.tar.gz.sha256
shasum -a 256 --check ferrum-macos-aarch64.tar.gz.sha256
tar -xzf ferrum-macos-aarch64.tar.gz
./ferrum --version
```

从 crates.io 安装 Metal build：

```bash
# macOS Apple Silicon Metal
cargo install ferrum-cli --version 0.8.4 --locked --features metal
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
