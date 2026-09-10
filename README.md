<p align="center">
  <a href="https://ferrum.pandaailabs.com/">
    <img src="assets/brand/ferrum-lockup.svg" alt="Ferrum — Rust-native LLM serving" width="520">
  </a>
</p>

[![Crates.io](https://img.shields.io/crates/v/ferrum-cli.svg)](https://crates.io/crates/ferrum-cli)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/sizzlecar/ferrum-infer-rs/blob/main/LICENSE)

# Serve LLMs with a single binary.

A Rust-native LLM serving engine. One binary, no Python runtime.

[中文说明](README_zh.md)

## Vision

Make high-performance LLM serving simple to deploy and operate.

## Quick Start

Install the latest stable Ferrum on macOS Apple Silicon or Linux x86_64:

```bash
curl -fsSL https://ferrum.pandaailabs.com/install.sh | sh
```

The installer verifies release checksums and adds `~/.local/bin` to your shell's
PATH. Open a new terminal afterward. [Homebrew and manual installation](#installation)
are also available.

Windows x64 supports CPU inference and compatible NVIDIA sm89 GPUs.
Install from PowerShell:

```powershell
irm https://ferrum.pandaailabs.com/install.ps1 | iex
```

The script verifies the setup checksum, installs for the current user, and adds
Ferrum to PATH, including the current PowerShell session.

Installers select CPU when a supported GPU is unavailable. Package downloads use
Cloudflare CDN, retain SHA256 verification, and fall back to GitHub if needed.
Running the same command again installs the latest formal release.

Inspect the installed binary before downloading weights:

```bash
ferrum --version
ferrum --help
ferrum doctor
```

### Run a model

With **Ferrum 0.9.0 or later**, use the same GGUF model on macOS, Linux, and
Windows. Ferrum automatically selects the available backend; both Metal and
CUDA support this Q4_K_M example.

```bash
ferrum run qwen3.5:4b-q4_k_m --disable-thinking
```

The first run downloads about **2.55 GiB**. Download time depends on your route
to Hugging Face; the CLI displays download progress. On a 6 GB GPU, append
`--max-model-len 2048 --max-num-seqs 1` to either `run` or `serve` to limit the
context and active sequences.

### Serve an API

The server command is also the same on all three platforms:

```bash
ferrum serve --model qwen3.5:4b-q4_k_m --served-model-name ferrum --disable-thinking --port 8000
```

Send a request from another terminal. On macOS or Linux:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"ferrum","messages":[{"role":"user","content":"Reply with a short hello from Ferrum."}],"max_tokens":32}'
```

In Windows PowerShell:

```powershell
$body = @{ model = 'ferrum'; messages = @(@{ role = 'user'; content = 'Reply with a short hello from Ferrum.' }); max_tokens = 32 } | ConvertTo-Json -Depth 4
Invoke-RestMethod http://localhost:8000/v1/chat/completions -Method Post -ContentType 'application/json' -Body $body
```

Ferrum does not silently select a model. `run` requires MODEL, and `serve`
requires either `--model` or an intentional `default_model` in `ferrum.toml`.

A working request returns HTTP 200 with a non-empty assistant response. Ferrum
uses the model's context limit unless `--max-model-len` is set explicitly; any
explicit limit must fit the rendered input plus the requested output budget.

The examples use `--disable-thinking` so the first response is short and
direct. Omit the flag to preserve the model template's default reasoning
behavior; an HTTP request can override the server default with
`chat_template_kwargs.enable_thinking`, Chat `reasoning_effort`, or Responses
`reasoning.effort`. See [reasoning control behavior](docs/openai-api-compatibility.md#chat-fields)
for model support and compatibility details.

`ferrum doctor <MODEL>` resolves an alias and prints the next `run` and `serve`
commands without downloading the model or starting an inference engine.

For vNext execution, `run` and `serve` share this optional `ferrum.toml` setting
in the working directory:

```toml
[runtime]
reusable_execution_preparation = "auto" # auto, startup, on_demand
```

`auto` uses bounded on-demand preparation on runtimes that declare support
(currently CUDA). A new shape first executes normally; later occurrences can
prepare and reuse a device program. First-use latency can therefore be higher
than steady-state latency. `startup` prepares the configured matrix before the
server becomes ready. Other backends retain their existing behavior; explicitly
requesting unsupported `on_demand` reports an error. Set `reusable_execution = false`
to disable device-program preparation. These options do not change request
admission, queuing, or the model's numerical profile.

## Features

- `ferrum run` and `ferrum serve` in one Rust binary.
- OpenAI-compatible Chat Completions and stateless Responses APIs, streaming,
  tools, and structured output.
- Apple Silicon Metal and NVIDIA CUDA from the same runtime.
- Continuous batching, paged KV cache, prefix cache, and typed admission control.
- GGUF on Metal and CUDA; CUDA also supports GPTQ/safetensors.
- Ferrum covers language-model inference only. Supported models include Qwen3.5 4B,
  Qwen3.5 35B-A3B, Qwen3 30B-A3B, and Llama 3.1 8B dense.

## Performance Snapshot

Latest R2 development `ferrum serve` checkpoint. The first three rows use
64-token input / 128-token output on Metal and 256 / 128 on CUDA. Values are
mean tok/s with the 95% confidence-interval half-width across three repeats.

| Model | M1 Max 32 GB Metal | RTX 4090 CUDA | L40S 48 GB CUDA |
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

`c` is active server concurrency. The first three rows completed 100 requests ×
3 repeats with zero errors.

## OpenAI-Compatible API

Ferrum supports:

- chat completions and streaming usage
- stateless Responses text, reasoning replay, streaming, usage, and
  caller-owned function/namespace tool loops
- function tools with `auto`, `none`, `required`, or a named function
- `json_object` and strict `json_schema` structured output
- multi-turn sessions, prefix cache, and session cache
- typed concurrency, memory, and scheduler controls

See [OpenAI API compatibility](docs/openai-api-compatibility.md) for the exact
request contract and [cache product controls](docs/cache-product.md) for prefix
and session caching.

## Installation

Windows **0.8.9 and later** can also be installed by downloading
`ferrum-<version>-windows-x86_64-cuda-sm89-setup.exe` and its `.sha256` file from
[Releases](https://github.com/sizzlecar/ferrum-infer-rs/releases), verifying the
checksum, and running setup. It installs under `%LOCALAPPDATA%\Programs\Ferrum`
and adds the current-user PATH; open a new terminal after a manual setup install.
The package includes CUDA and VC runtimes. It requires a compatible NVIDIA sm89
GPU and driver (551.78 or later); it does not install the system driver or include
models. CUDA Toolkit, Rust, and build tools are not needed. Ferrum remains a
command-line application with `run` and `serve`, without a GUI or background service.

To upgrade Windows, rerun the same PowerShell install command or the newer setup.
Existing sessions keep running their original version; new launches use the
updated version. Models, configuration, and existing version directories are
preserved. Restart an existing server when you want it to use the update.

The macOS/Linux one-line installer selects Metal on Apple Silicon. On Linux it selects CUDA
for compatible sm89 GPUs when the driver, CUDA 12.4 and NCCL runtimes can load,
and otherwise selects CPU. You can require a backend or install a specific version:

```bash
curl -fsSL https://ferrum.pandaailabs.com/install.sh | sh -s -- --backend cuda
curl -fsSL https://ferrum.pandaailabs.com/install.sh | sh -s -- --version 0.8.8
```

To upgrade an installation made with the script, rerun the original install
command. It keeps existing version directories and switches the entry point to
the verified new binary. Running sessions continue using their current version;
new launches use the new version. Restart an existing server when you want it to
use the update. Models and configuration are preserved.

For immediate PATH setup in the current terminal:

```bash
. "$HOME/.local/share/ferrum/installer/env"
```

For Homebrew installations, use `brew upgrade` for the installed formula.
Homebrew 6 needs both [formula definitions](https://github.com/sizzlecar/homebrew-ferrum/tree/main/Formula)
trusted for its conflict check. Review them before running the trust command;
older Homebrew versions can skip it. See [Homebrew's trust documentation](https://docs.brew.sh/Tap-Trust).

```bash
# Homebrew 6: trust the reviewed formula definitions
brew trust --formula sizzlecar/ferrum/ferrum sizzlecar/ferrum/ferrum-cuda

# macOS Apple Silicon Metal
brew install sizzlecar/ferrum/ferrum

# Linux x86_64 CUDA sm89
brew install sizzlecar/ferrum/ferrum-cuda
```

Prebuilt tarballs from the latest stable release:

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

Install the latest Metal build from crates.io:

```bash
# macOS Apple Silicon Metal
cargo install ferrum-cli --locked --features metal
```

The official prebuilt Linux CUDA asset targets `sm89`. Linux CUDA installation requires a
compatible NVIDIA driver, CUDA runtime, and NCCL runtime on the target host.
CUDA source builds also require Ferrum's matching native-operator set, so use
the prebuilt CUDA tarball or Homebrew formula for the supported install path.

## Architecture

- Contracts: `ferrum-types`, `ferrum-interfaces`
- Execution: `ferrum-engine`, `ferrum-scheduler`, `ferrum-kv`, `ferrum-sampler`
- Models and compute: `ferrum-models`, `ferrum-kernels`, `ferrum-native-ops`, `ferrum-quantization`
- Product surface: `ferrum-cli`, `ferrum-server`, `ferrum-tokenizer`
- Validation: `ferrum-bench-core`, `ferrum-testkit`

Development notes: [numerical execution profiles (中文)](docs/numerical-execution.zh.md).
The new CLI option is not available in v0.8.9 release assets.

## License

MIT
