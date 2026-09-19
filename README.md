<p align="center">
  <a href="https://ferrum.pandaailabs.com/">
    <img src="assets/brand/ferrum-lockup.svg" alt="Ferrum — Rust-native LLM serving" width="520">
  </a>
</p>

[![Crates.io](https://img.shields.io/crates/v/ferrum-cli.svg)](https://crates.io/crates/ferrum-cli)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/sizzlecar/ferrum-infer-rs/blob/main/LICENSE)

# Serve LLMs with a single binary.

Serve local LLMs to your apps and coding agents through an OpenAI-compatible API.
Ferrum is written in Rust and runs as a single binary, with Metal on Apple Silicon
and CUDA on [supported NVIDIA GPUs](#installation). The inference server needs no
Python runtime.

[中文说明](README_zh.md)

| What would you like to do? | Start here |
|---|---|
| Run a model in your terminal | [Quick Start](#quick-start) |
| Connect an app or coding agent | [Start the API server](#serve-an-api) · [API compatibility](docs/openai-api-compatibility.md) |
| Run Bonsai 2 27B PQ2_0 on Apple Silicon | [Metal setup and tested limits](#bonsai-2-pq2_0-on-metal) |

## See it in action

One local model. Three Orchestral terminals inspecting code, fixing bugs, and
running tests concurrently.

[![Watch the Ferrum + Orchestral demo](https://ferrum-downloads.pandaailabs.com/v0.3.1/ferrum-orch-three-agents.png)](https://ferrum-downloads.pandaailabs.com/v0.3.1/ferrum-orch-three-agents.mp4)

[Watch the English demo](https://ferrum-downloads.pandaailabs.com/v0.3.1/ferrum-orch-three-agents.mp4) · 50 seconds · 8× speed.

<details>
<summary><strong>Try it locally</strong></summary>

Install Ferrum and Orchestral on macOS Apple Silicon or Linux x86_64:

```sh
curl -fsSL https://ferrum.pandaailabs.com/install.sh | sh
curl -fsSL https://orch.pandaailabs.com/install.sh | sh
```

On Windows x64, use PowerShell:

```powershell
irm https://ferrum.pandaailabs.com/install.ps1 | iex
irm https://orch.pandaailabs.com/install.ps1 | iex
```

After installation, open a new terminal and start the model:

```sh
ferrum serve --model unsloth/Qwen3.5-9B-GGUF
```

Ferrum automatically selects an available backend, resolves the GGUF file, and
downloads any missing weights and metadata. Later starts reuse the cache. With
the default configuration, the API listens at `http://127.0.0.1:8000/v1`.

Leave Ferrum running. In another terminal, open your project directory and run:

```sh
orchestral --base-url http://127.0.0.1:8000/v1 --no-auth
```

Once the model is ready, type a task and press Enter. No JSON configuration or API
key is required. Memory requirements and speed depend on your hardware; these
defaults are for trying the model, not reproducing the recording's concurrency
and performance settings.

</details>

<details>
<summary><strong>Advanced: reproduce the recording configuration</strong></summary>

The recording uses an **M1 Max Mac with 32 GB unified memory**, Metal, and
**Qwen3.5-9B Q4_K_M**. The commands below reproduce its serving settings:
24,576 tokens per context, three active sequences, a 20 GiB runtime memory budget,
and the model's default thinking behavior. These optional settings are not
required to try Ferrum. Use **Ferrum 0.10.0** and **Orchestral 0.3.1**.

Install both programs once, then open four terminal panes:

```sh
curl -fsSL https://ferrum.pandaailabs.com/install.sh | sh -s -- --version 0.10.0 --backend metal
curl -fsSL https://orch.pandaailabs.com/install.sh | sh -s -- --version 0.3.1
export PATH="$HOME/.local/bin:$PATH"
ferrum --version
orchestral --version
```

**Terminal 1 — upper left: start Ferrum.** The first start downloads the selected
GGUF and its model/tokenizer metadata from Hugging Face; subsequent starts reuse
the cache. The repository revision and filename select the weights used in the video.

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

Leave Ferrum running. In another terminal, check that it is ready before starting
the agents. This discovers the served model without generating a response:

```sh
orchestral --base-url http://127.0.0.1:8001/v1 --no-auth doctor --check-connection
```

**Terminal 2 — upper right:** replace the path with your first project directory.

```sh
cd /path/to/project-a
orchestral --base-url http://127.0.0.1:8001/v1 --no-auth
```

**Terminal 3 — lower left:** open your second project.

```sh
cd /path/to/project-b
orchestral --base-url http://127.0.0.1:8001/v1 --no-auth
```

**Terminal 4 — lower right:** open your third project.

```sh
cd /path/to/project-c
orchestral --base-url http://127.0.0.1:8001/v1 --no-auth
```

Type a task in each Orchestral terminal and press Enter. Each session uses the
same Ferrum server. The video uses three separate Rust projects with Cargo
installed, and asks each agent to fix failing tests, preserve the public API,
run `cargo test`, and explain the fix in English.

</details>

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

`GET /v1/models` also exposes optional `reasoning` metadata. A supported thinking
switch reports its effective default in `thinking.default_enabled`; explicitly
declared effort levels appear in `supported_efforts`. A thinking switch alone
does not imply low/medium/high levels. Omitted effort metadata means unknown support.

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

### KV cache precision

Ferrum v0.11.0 accepts `--kv-dtype int8` in both `run` and `serve`. FP16 remains
the default. INT8 requires supported vNext standard causal attention on Metal or
portable CUDA; unsupported combinations report an error.

```sh
ferrum run unsloth/Qwen3.5-9B-GGUF --kv-dtype int8 --disable-thinking
ferrum serve --model unsloth/Qwen3.5-9B-GGUF --kv-dtype int8 --disable-thinking
```

This reduces attention KV storage, including its quantization scales. Model
weights and fixed recurrent state retain their existing sizes. Inspect
`/health` → `kv_storage` to confirm the selected format. Whole-model checkpoint
restore requires support for every model state; resending conversation history
recomputes the input when prefix caching is disabled or no compatible checkpoint
is available. With `--enable-prefix-cache`, a compatible hit restores model state
and processes the remaining suffix. Session caching stores chat messages; it is
separate from GPU prefix-state reuse.

### Bonsai 2 PQ2_0 on Metal

On Apple Silicon, start a chat with **Bonsai 2 27B**:

```sh
ferrum run bonsai2:27b
```

Or start a local API for your apps:

```sh
ferrum serve --model bonsai2:27b
```

The first start downloads the approximately **7.2 GB PQ2_0 weights** and their
matching metadata; later starts reuse the cache. No manual file preparation is
needed. The API listens at `http://127.0.0.1:8000/v1`.

Requires Ferrum **0.12.1 or later**. For 0.12.0, use the manual setup below.

<details>
<summary><strong>Defaults, tested limits, and manual setup</strong></summary>

The shortcut selects the model files; resource configuration follows Ferrum's
normal defaults and automatic capacity handling. It does not impose a
Bonsai-specific context length, batch size, concurrency, or memory budget.
Explicit configuration and CLI options take precedence. Model reasoning
behavior is preserved; append `--disable-thinking` if you want it off.

Ferrum keeps official **Ternary Bonsai 2 27B GGUF PQ2_0**
weights packed and applies the Hadamard transforms declared by the model.
Metal text inference has been validated through `run` and `serve` with FP16 KV,
including Orchestral tool execution, session continuation, and prefix-state reuse.
The 8K context and 10 GiB runtime budget below were tested on an M1 Max; they are
not minimum hardware requirements or guarantees for larger workloads.

Use the [official PQ2_0 file](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf/tree/6ed5e12bf84b7a63069882c91dd9e9218647d17b).
For this checkpoint, the validated metadata is the original `config.json`,
`generation_config.json`, `tokenizer.json`, `tokenizer_config.json`, and
`chat_template.jinja` from [this pinned source-model revision](https://huggingface.co/Qwen/Qwen3.8-27B/tree/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0).
Save those five files in the metadata directory below.

```sh
MODEL=/path/to/Ternary-Bonsai-2-27B-PQ2_0.gguf
METADATA=/path/to/matching-source-metadata

ferrum run "$MODEL" --semantic-source "$METADATA" --tokenizer-source "$METADATA" \
  --backend metal --numerical-profile qwen3_5.f32-master --kv-dtype fp16 \
  --max-model-len 8192 --kv-capacity 8192 --max-num-seqs 1 \
  --max-num-batched-tokens 128 --runtime-memory-budget-bytes 10737418240 \
  --disable-thinking --prompt "Explain what a hash table does." --max-tokens 128

ferrum serve --model "$MODEL" --semantic-source "$METADATA" --tokenizer-source "$METADATA" \
  --backend metal --numerical-profile qwen3_5.f32-master --kv-dtype fp16 \
  --max-model-len 8192 --kv-capacity 8192 --max-num-seqs 1 \
  --max-num-batched-tokens 128 --runtime-memory-budget-bytes 10737418240 \
  --disable-thinking --served-model-name bonsai --enable-prefix-cache --session-cache off
```

The serving example keeps session-message storage off for clients that already
send full conversation history. Check `/health` → `cache.prefix_cache`: native
reuse reports `source: "vnext-native-sequence-checkpoint-cache"`, with increasing
`hits` and `saved_prefill_tokens` after a compatible request.

CUDA PQ2_0/Hadamard operators and small mixed-state checkpoint tests have passed
on a real GPU; **the complete 27B CUDA model remains unvalidated**. This Bonsai
path does not support PTQ1_0, earlier Bonsai Q1_0/Q2_0 encodings, MLX packages,
vision, or complete CPU inference. Bonsai combined with INT8 KV is not yet
validated. The model's declared context limit does not establish tested coverage
beyond the context above.

</details>

## Features

- `ferrum run` and `ferrum serve` in one Rust binary.
- OpenAI-compatible Chat Completions and stateless Responses APIs, streaming,
  tools, and structured output.
- Apple Silicon Metal and NVIDIA CUDA from the same runtime.
- Continuous batching, paged KV cache, prefix cache, and typed admission control.
- Optional [8-bit KV storage](#kv-cache-precision) for supported vNext Metal and portable CUDA attention paths; FP16 remains the default.
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
curl -fsSL https://ferrum.pandaailabs.com/install.sh | sh -s -- --version 0.11.0
```

To upgrade an installation made with the script, rerun the original install
command. If the selected version and backend are already installed and verify
successfully, the script checks the small release checksum files and skips the
package download. It keeps existing version directories and switches the entry point to
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

## License

MIT
