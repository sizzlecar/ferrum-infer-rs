<p align="center">
  <a href="https://ferrum.pandaailabs.com/zh/">
    <img src="assets/brand/ferrum-lockup.svg" alt="Ferrum — Rust 原生大模型服务" width="520">
  </a>
</p>

[![Crates.io](https://img.shields.io/crates/v/ferrum-cli.svg)](https://crates.io/crates/ferrum-cli)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/sizzlecar/ferrum-infer-rs/blob/main/LICENSE)

# 用一个二进制文件提供大模型服务。

通过 OpenAI 兼容 API，将本地大模型接入你的应用和编程 Agent。
Ferrum 使用 Rust 编写，以单个二进制文件运行，支持 Apple Silicon Metal 和
[兼容的 NVIDIA GPU](#安装) 上的 CUDA。推理服务自身无需 Python 运行时。

[English](README.md)

| 你想做什么？ | 从这里开始 |
|---|---|
| 在终端运行模型 | [快速开始](#快速开始) |
| 接入应用或编程 Agent | [启动 API 服务](#启动-api-服务) · [API 兼容说明](docs/openai-api-compatibility.md) |
| 在 Apple Silicon 上运行 Bonsai 2 27B PQ2_0 | [Metal 配置与已验证范围](#在-metal-上运行-bonsai-2-pq2_0) |

## 在 Metal 上运行 Bonsai 2 PQ2_0

在 Apple Silicon 上启动 **Bonsai 2 27B** 对话：

```sh
ferrum run bonsai2:27b
```

或者启动本地 API，供应用调用：

```sh
ferrum serve --model bonsai2:27b
```

首次启动自动下载约 **7.2 GB 的 PQ2_0 权重**及匹配元数据，之后复用缓存，
无需手动准备文件。Ferrum 根据可用内存适配上下文、批处理和并发，你明确
指定的配置优先。API 地址为 `http://127.0.0.1:8000/v1`。

已在 **M1 Max / 32 GB** 上验证 Metal 文本推理。需要 Ferrum **0.12.1 或更高版本**。
[安装或更新 Ferrum](#快速开始)。

<details>
<summary><strong>模型来源与实测范围</strong></summary>

快捷入口只选择模型文件，资源配置沿用 Ferrum 的通用默认值和自动容量管理。
规划依据是所选设备的可用内存，以及编译后的权重、序列状态和算子工作区，
不会为 Bonsai 单独固定上下文、批处理、并发或内存预算。显式配置及 CLI
参数优先。保留模型原本的思考行为；需要关闭时，追加 `--disable-thinking`。

Ferrum 直接使用官方 **Ternary Bonsai 2 27B GGUF PQ2_0** 压缩权重，
并按模型声明执行 Hadamard 变换。已验证 Metal 文本 `run`、`serve` 和 FP16 KV，
包括 Orchestral 真实工具执行、会话续接和前缀状态复用。在 M1 Max / 32 GB 上
的测试包括 8K 上下文请求，更长上下文尚未在这台机器上验收。生效的上下文上限
不超过模型声明，并可能根据机器容量缩减。并发请求共享运行时容量；并发上限
不意味着为每个请求都预留完整上下文。

快捷入口自动下载[官方 PQ2_0 文件](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf/tree/6ed5e12bf84b7a63069882c91dd9e9218647d17b)，
以及[固定源模型版本](https://huggingface.co/Qwen/Qwen3.8-27B/tree/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0)
中的 `config.json`、`generation_config.json`、`tokenizer.json`、
`tokenizer_config.json` 和 `chat_template.jinja`，无需手动整理。

CUDA PQ2_0/Hadamard 算子和小型混合状态 checkpoint 测试已在真实 GPU 上通过，
**完整 27B 模型的 CUDA 验收仍未完成**。此 Bonsai 路径不支持 PTQ1_0、旧代
Q1_0/Q2_0 编码、MLX 包、视觉或完整 CPU 推理；Bonsai 与 INT8 KV 的组合尚未验收。
模型声明的上下文上限不代表超过上述长度的范围已经实测。

</details>

## 看它如何工作

在 M1 Max / 32 GB 上运行 Qwen3.5-9B，三个 Orchestral 终端同时阅读代码、
修复问题并运行测试。

[![观看 Ferrum + Orchestral 演示](https://ferrum-downloads.pandaailabs.com/v0.3.1/ferrum-orch-three-agents.png)](https://ferrum-downloads.pandaailabs.com/v0.3.1/ferrum-orch-three-agents.mp4)

[观看英文演示](https://ferrum-downloads.pandaailabs.com/v0.3.1/ferrum-orch-three-agents.mp4) · 50 秒 · 8 倍速。

<details>
<summary><strong>快速试用</strong></summary>

在 macOS Apple Silicon 或 Linux x86_64 上安装 Ferrum 和 Orchestral：

```sh
curl -fsSL https://ferrum.pandaailabs.com/install.sh | sh
curl -fsSL https://orch.pandaailabs.com/install.sh | sh
```

Windows x64 请使用 PowerShell：

```powershell
irm https://ferrum.pandaailabs.com/install.ps1 | iex
irm https://orch.pandaailabs.com/install.ps1 | iex
```

安装后打开新终端，一行启动模型：

```sh
ferrum serve --model unsloth/Qwen3.5-9B-GGUF
```

Ferrum 会自动选择可用后端、解析 GGUF 文件并下载缺失的权重和元数据，后续启动复用缓存。
默认配置下，API 地址为 `http://127.0.0.1:8000/v1`。

保持 Ferrum 运行，在另一个终端进入你的项目目录后执行：

```sh
orchestral --base-url http://127.0.0.1:8000/v1 --no-auth
```

模型就绪后，输入任务并按 Enter 即可，无需 JSON 配置或 API key。
内存需求和速度取决于硬件；这些默认设置用于快速试用，不等于录像的并发和性能配置。

</details>

<details>
<summary><strong>高级：复现录像配置</strong></summary>

视频使用 **M1 Max、32 GB 统一内存的 Mac**，通过 Metal 运行 **Qwen3.5-9B Q4_K_M**。
以下命令采用相同的服务参数：每个上下文 24,576 token、三个活跃序列、20 GiB 运行时内存预算，
并保留模型默认的思考行为。这些可选设置不是试用 Ferrum 的前提。
使用 **Ferrum 0.10.0** 和 **Orchestral 0.3.1**。

先安装两个程序，再打开四个终端格子：

```sh
curl -fsSL https://ferrum.pandaailabs.com/install.sh | sh -s -- --version 0.10.0 --backend metal
curl -fsSL https://orch.pandaailabs.com/install.sh | sh -s -- --version 0.3.1
export PATH="$HOME/.local/bin:$PATH"
ferrum --version
orchestral --version
```

**终端 1（左上）：启动 Ferrum。** 首次启动会从 Hugging Face 下载指定的 GGUF 和模型、分词器元数据，
以后启动复用缓存。仓库版本和文件名对应视频使用的模型权重。

```sh
ferrum serve \
  --model unsloth/Qwen3.5-9B-GGUF@3885219b6810b007914f3a7950a8d1b469d598a5 \
  --gguf-file Qwen3.5-9B-Q4_K_M.gguf \
  --served-model-name Qwen3.5-9B \
  --backend metal \
  --numerical-profile qwen3_5.f32-master \
  --host 127.0.0.1 --port 8001 \
  --max-model-len 24576 \
  --max-num-seqs 3 \
  --max-num-batched-tokens 3072 \
  --scheduler-prefill-step-chunk 1024 \
  --scheduler-active-decode-prefill-chunk 256 \
  --enable-prefix-cache \
  --runtime-memory-budget-bytes 21474836480 \
  --prefix-rendezvous-max-wait-ms 180000
```

保持 Ferrum 运行。在另一个终端检查服务是否就绪，再启动 Agent。
这条命令会发现服务中的模型，不会生成回答：

```sh
orchestral --base-url http://127.0.0.1:8001/v1 --no-auth doctor --check-connection
```

**终端 2（右上）：** 将路径替换为你的第一个项目目录。

```sh
cd /path/to/project-a
orchestral --base-url http://127.0.0.1:8001/v1 --no-auth
```

**终端 3（左下）：** 打开第二个项目。

```sh
cd /path/to/project-b
orchestral --base-url http://127.0.0.1:8001/v1 --no-auth
```

**终端 4（右下）：** 打开第三个项目。

```sh
cd /path/to/project-c
orchestral --base-url http://127.0.0.1:8001/v1 --no-auth
```

在每个 Orchestral 终端输入任务并按 Enter，三个会话共用同一个 Ferrum 服务。
视频使用三个独立的 Rust 项目，机器上已安装 Cargo；每个 Agent 的任务都是修复失败测试、
保留公开接口、运行 `cargo test`，并用英文解释修改。

</details>

## 愿景

让高性能大模型服务的部署与运维更简单。

## 快速开始

在 macOS Apple Silicon 或 Linux x86_64 上安装 Ferrum 最新正式版：

```bash
curl -fsSL https://ferrum.pandaailabs.com/install.sh | sh
```

安装脚本会校验发布包，并将 `~/.local/bin` 加入 shell 的 PATH；完成后打开新终端。
也可以使用 [Homebrew 或手动安装](#安装)。

Windows x64 支持 CPU 推理及兼容的 NVIDIA sm89 显卡。在 PowerShell 中执行：

```powershell
irm https://ferrum.pandaailabs.com/install.ps1 | iex
```

脚本校验 setup 的 SHA256 后为当前用户安装，并自动添加 PATH；当前 PowerShell 也会立即生效。

没有受支持的 GPU 时，安装脚本会选择 CPU 版本。安装包优先通过 Cloudflare CDN 下载，
保留 SHA256 校验，下载失败时回退到 GitHub。再次执行同一命令即可安装最新正式版。

下载权重前先检查安装的二进制：

```bash
ferrum --version
ferrum --help
ferrum doctor
```

### 运行模型

在 **Ferrum 0.9.0 及以上版本**中，macOS、Linux 和 Windows 使用同一个 GGUF 模型、
同一条命令。Ferrum 自动选择可用后端；Metal 和 CUDA 均支持此 Q4_K_M 示例。

```bash
ferrum run qwen3.5:4b-q4_k_m --disable-thinking
```

首次运行会下载约 **2.55 GiB**。下载耗时取决于本机到 Hugging Face 的网络链路，
CLI 会显示下载进度。Ferrum 会在启动时根据可用内存适配上下文、批处理和并发。

### 启动 API 服务

三个平台的服务启动命令也相同：

```bash
ferrum serve --model qwen3.5:4b-q4_k_m --served-model-name ferrum --disable-thinking --port 8000
```

然后从另一个终端发送请求。macOS 或 Linux：

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"ferrum","messages":[{"role":"user","content":"Reply with a short hello from Ferrum."}],"max_tokens":32}'
```

Windows PowerShell：

```powershell
$body = @{ model = 'ferrum'; messages = @(@{ role = 'user'; content = 'Reply with a short hello from Ferrum.' }); max_tokens = 32 } | ConvertTo-Json -Depth 4
Invoke-RestMethod http://localhost:8000/v1/chat/completions -Method Post -ContentType 'application/json' -Body $body
```

Ferrum 不会静默选择模型。`run` 必须提供 MODEL；`serve` 必须提供 `--model`，
或者在 `ferrum.toml` 中有意设置 `default_model`。

正常时请求会返回 HTTP 200 和非空的 assistant 回答。原生 `run` 和 `serve`
从模型声明的上下文上限出发，根据设备当前可用内存，自动适配未指定的上下文、
批次和并发设置，并在启动时显示生效值。显式设置会被保留；容量不足时明确报错。
上下文上限包含渲染后的输入与输出。

示例使用 `--disable-thinking`，让首次回答简短直接。删除该参数即可恢复模型模板默认的
推理行为；HTTP 请求也可以通过 `chat_template_kwargs.enable_thinking`、Chat 的
`reasoning_effort` 或 Responses 的 `reasoning.effort` 覆盖服务端默认值。
模型支持范围及兼容行为见 [API 说明](docs/openai-api-compatibility.md#chat-fields)。

`GET /v1/models` 还提供可选的 `reasoning` 元数据：`thinking.default_enabled`
表示受支持思考开关的服务端实际默认值，`supported_efforts` 列出模型明确声明的强度档位。
支持思考开关不代表支持 low/medium/high；省略档位字段表示支持情况未知。

`ferrum doctor <MODEL>` 会解析模型来源并打印下一条 `run`、`serve` 命令，
不会下载模型或启动推理引擎。

vNext 的 `run` 和 `serve` 共用工作目录下 `ferrum.toml` 中的可选设置：

```toml
[runtime]
reusable_execution_preparation = "auto" # auto、startup、on_demand
```

`auto` 在声明支持的运行时（目前为 CUDA）使用有容量上限的按需准备：
新形状先普通执行，后续再次出现时才准备并复用设备程序，因此首次使用的延迟可能高于稳定运行时。
`startup` 在服务就绪前预备配置指定的形状；其他后端保持原有行为，显式指定不支持的
`on_demand` 会报错。设置 `reusable_execution = false` 可关闭设备程序准备。
这些选项不改变请求准入、排队或模型数值策略。

### KV 缓存精度

Ferrum v0.11.0 的 `run` 和 `serve` 均可使用 `--kv-dtype int8`，默认仍为 FP16。
此选项需要支持标准 causal attention 的 vNext 模型，以及 Metal 或 portable CUDA
执行路径；不支持的组合会明确报错。

```sh
ferrum run unsloth/Qwen3.5-9B-GGUF --kv-dtype int8 --disable-thinking
ferrum serve --model unsloth/Qwen3.5-9B-GGUF --kv-dtype int8 --disable-thinking
```

此选项减少注意力 KV 的存储占用，包含量化所需的 scale；模型权重和固定大小的循环状态
保持原有大小。通过 `/health` 的 `kv_storage` 可确认实际生效格式。
整个模型的 checkpoint 恢复需要所有模型状态均支持恢复。关闭 prefix cache 或没有兼容的
checkpoint 时，重新发送历史会重新计算输入；使用 `--enable-prefix-cache` 并命中后，
可恢复模型状态并计算剩余后缀。Session cache 保存聊天消息，与 GPU 前缀状态复用是两回事。

## 功能

- 一个 Rust 二进制同时提供 `ferrum run` 和 `ferrum serve`。
- 支持 OpenAI 兼容的 Chat Completions 与无状态 Responses API、流式输出、
  tools 和 structured output。
- 同一 runtime 覆盖 Apple Silicon Metal 与 NVIDIA CUDA。
- 支持 continuous batching、paged KV cache、prefix cache 和 typed admission。
- 支持在兼容的 vNext Metal、portable CUDA 路径选择 8-bit KV，默认保持 FP16。
- Metal 与 CUDA 均支持 GGUF；CUDA 还支持 GPTQ/safetensors。
- Ferrum 只覆盖语言模型推理。支持的模型包括 Qwen3.5 4B、Qwen3.5 35B-A3B、
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

Windows **0.8.9 及以后版本**也可以从
[Releases](https://github.com/sizzlecar/ferrum-infer-rs/releases) 下载
`ferrum-<version>-windows-x86_64-cuda-sm89-setup.exe` 与对应 `.sha256` 文件，
核对校验和后运行 setup。安装位置为 `%LOCALAPPDATA%\Programs\Ferrum`，自动添加当前用户 PATH；
手动运行 setup 后请打开新终端。安装包包含 CUDA 与 VC 运行库，需要兼容的 NVIDIA sm89 显卡和
551.78 或更新的驱动，不安装系统驱动，也不包含模型。用户不需要安装 CUDA Toolkit、Rust 或
编译工具。Ferrum 通过 `run`、`serve` 命令使用，不提供图形界面或后台服务。

Windows 升级时重新执行同一条 PowerShell 安装命令，或运行新版 setup。正在运行的会话继续使用
原版本，新启动的会话使用新版；模型、配置和已有版本目录会保留。已有服务可在需要更新时自行重启。

macOS/Linux 一键安装在 Apple Silicon 上选择 Metal。在 Linux 上，兼容的 sm89 显卡能够加载驱动、
CUDA 12.4 与 NCCL 运行库时选择 CUDA，否则选择 CPU。也可以明确要求后端或指定版本：

```bash
curl -fsSL https://ferrum.pandaailabs.com/install.sh | sh -s -- --backend cuda
curl -fsSL https://ferrum.pandaailabs.com/install.sh | sh -s -- --version 0.11.0
```

使用脚本安装后，重新执行原安装命令即可升级。同版本、同后端已安装且校验通过时，
脚本只获取很小的发布校验文件，跳过安装包下载。脚本保留已有版本目录，将入口切换到
校验后的新版二进制。正在运行的会话继续使用原版本，新启动的会话使用新版本；已有服务
可在需要更新时自行重启。模型和配置会保留。

如果希望当前终端立即使用 Ferrum，可执行：

```bash
. "$HOME/.local/share/ferrum/installer/env"
```

通过 Homebrew 安装的用户使用 `brew upgrade` 升级对应公式。Homebrew 6 检查互斥安装时需要
信任两条[公式定义](https://github.com/sizzlecar/homebrew-ferrum/tree/main/Formula)，请先查看定义再运行
信任命令；旧版 Homebrew 可跳过。详见 [Homebrew 信任说明](https://docs.brew.sh/Tap-Trust)。

```bash
# Homebrew 6：信任已查看的两条公式定义
brew trust --formula sizzlecar/ferrum/ferrum sizzlecar/ferrum/ferrum-cuda

# macOS Apple Silicon Metal
brew install sizzlecar/ferrum/ferrum

# Linux x86_64 CUDA sm89
brew install sizzlecar/ferrum/ferrum-cuda
```

最新正式版的预编译 tarball：

```bash
# Linux x86_64 CUDA sm89
curl --fail --location --remote-name https://github.com/sizzlecar/ferrum-infer-rs/releases/latest/download/ferrum-linux-x86_64-cuda-sm89.tar.gz
curl --fail --location --remote-name https://github.com/sizzlecar/ferrum-infer-rs/releases/latest/download/ferrum-linux-x86_64-cuda-sm89.tar.gz.sha256
sha256sum --check ferrum-linux-x86_64-cuda-sm89.tar.gz.sha256
tar -xzf ferrum-linux-x86_64-cuda-sm89.tar.gz
LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-} ./ferrum --version

# macOS Apple Silicon Metal
curl --fail --location --remote-name https://github.com/sizzlecar/ferrum-infer-rs/releases/latest/download/ferrum-macos-aarch64.tar.gz
curl --fail --location --remote-name https://github.com/sizzlecar/ferrum-infer-rs/releases/latest/download/ferrum-macos-aarch64.tar.gz.sha256
shasum -a 256 --check ferrum-macos-aarch64.tar.gz.sha256
tar -xzf ferrum-macos-aarch64.tar.gz
./ferrum --version
```

从 crates.io 安装最新版 Metal build：

```bash
# macOS Apple Silicon Metal
cargo install ferrum-cli --locked --features metal
```

官方预编译 Linux CUDA 资产的目标为 `sm89`。Linux CUDA 安装需要兼容的 NVIDIA driver、
CUDA runtime 和 NCCL runtime。CUDA 源码构建还需要与 Ferrum 匹配的
native-operator set，因此受支持的安装路径是预编译 CUDA tarball 或 Homebrew formula。

## 架构

- 契约：`ferrum-types`、`ferrum-interfaces`
- 执行：`ferrum-engine`、`ferrum-scheduler`、`ferrum-kv`、`ferrum-sampler`
- 模型与计算：`ferrum-models`、`ferrum-kernels`、`ferrum-native-ops`、`ferrum-quantization`
- 产品入口：`ferrum-cli`、`ferrum-server`、`ferrum-tokenizer`
- 验证：`ferrum-bench-core`、`ferrum-testkit`

开发接口说明：[数值执行策略](docs/numerical-execution.zh.md)。

## License

MIT
