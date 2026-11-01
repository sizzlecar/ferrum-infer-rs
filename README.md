<p align="center">
  <a href="https://ferrum.pandaailabs.com/">
    <img src="assets/brand/ferrum-lockup.svg" alt="Ferrum" width="520">
  </a>
</p>

[中文](README_zh.md) · [Releases](https://github.com/sizzlecar/ferrum-infer-rs/releases) · [MIT](LICENSE)

# Ferrum

Text and image inference in one Rust binary, with CLI and OpenAI-compatible APIs. Runs on Metal, CUDA and CPU.

The goal: maximize output throughput while meeting latency SLOs for time to first token (TTFT), time per output token (TPOT) and inter-token latency (ITL).

Fix hardware, model and output policy; optimize execution policy $\pi\in\Pi$:

$$
\begin{gathered}
\max_{\pi\in\Pi}\;\frac{N_{\mathrm{out}}(\pi;W,T)}{T} \\
\text{s.t.} \\
Q_{p_k}\!\left(L_k(\pi;W,T)\right)\le S_k \\
\forall k\in\{\mathrm{TTFT},\mathrm{TPOT},\mathrm{ITL}\} \\
C(\pi;W,T)\ge\rho
\end{gathered}
$$

- $W$: fixed requests and arrival schedule; $T>0$: fixed window and common cutoff. $N_{\mathrm{out}}$: valid output tokens from requests successfully completed by that cutoff.
- $L_k$: latency samples; $Q_{p_k}$: quantile; $S_k$: SLO limit. TTFT includes queueing; missing latency evidence does not establish compliance.
- $C$: fraction of $W$ successfully completed by the cutoff; rejected, failed, timed-out and unfinished requests remain in the denominator. Declare $p_k$, $S_k$ and minimum completion rate $0<\rho\le1$ before measuring. [Sampling rules](docs/performance-popular-models-p0.zh.md).

## Install

macOS Apple Silicon / Linux x86_64:

```sh
curl -fsSL https://ferrum.pandaailabs.com/install.sh | sh
```

Windows x64, in PowerShell:

```powershell
irm https://ferrum.pandaailabs.com/install.ps1 | iex
```

Installers verify checksums. Open a new terminal after installation; rerun to upgrade.

| Platform | Backend |
|---|---|
| macOS Apple Silicon | Metal |
| Linux / WSL x86_64 | CUDA SM89 or SM120; CPU fallback |
| Windows x64 | CUDA SM89; CPU fallback |

Automatic installation falls back to CPU for unsupported or mixed GPU architectures. CUDA requires a compatible NVIDIA driver. [Runtime requirements](docs/release-regression.md#prepare-the-candidate).

## Text generation

```sh
# Terminal chat
ferrum run qwen3.5:4b-q4_k_m --disable-thinking

# OpenAI-compatible service
ferrum serve --model qwen3.5:4b-q4_k_m \
  --served-model-name ferrum --disable-thinking --port 8000
```

Weights download on first use. Set your client's base URL to **`http://127.0.0.1:8000/v1`** and model to **`ferrum`**. Omit `--disable-thinking` to use the model template's default.

Execution and capacity defaults adapt to hardware and available memory; explicit settings take precedence. Supports Chat Completions, stateless Responses, SSE, tools and structured output. See [API compatibility](docs/openai-api-compatibility.md) for supported fields.

## Image generation and editing

Use a local [Qwen-Image-2.1-Turbo](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo) model directory. Linux CUDA example with FP8 text encoder/DiT:

```sh
ferrum serve /path/to/Qwen-Image-2.1-Turbo --backend cuda \
  --image-precision fp8_linear_experimental_v1 \
  --served-model-name qwen-image --port 8000
```

| API | Input |
|---|---|
| `POST /v1/images/generations` | JSON text prompt |
| `POST /v1/images/edits` | Multipart prompt + one 8-bit RGB/RGBA PNG |

Both return PNG in `b64_json`. CUDA batches compatible requests automatically within capacity limits. Validated on L20 (SM89) and RTX 5090 (SM120); cuDNN 9 is required. Metal supports generation and editing with a GGUF DiT, validated at 1024×1024 on an M4 Mac mini with 16 GiB memory.

See the [image guide](docs/image-generation.md) for CLI examples, Metal setup and CUDA LoRA.

## Models and documentation

Text architectures include Llama, Mistral, Qwen2, Qwen3 dense/MoE, Qwen3.5 dense/MoE and Gemma 3. Supports GGUF and SafeTensors; quantization and model support vary by backend. See the [validation catalog](docs/release-regression-catalog.md) for tested combinations.

- [API compatibility](docs/openai-api-compatibility.md) · [Agent serving](docs/agent-serving.md)
- [Numerical profiles](docs/numerical-execution.zh.md) · [Caching](docs/cache-product.md)
- Historical measurements: [text, v0.13 candidate](docs/performance-popular-models-p0.zh.md) · [images, v0.15 candidate](https://github.com/sizzlecar/ferrum-infer-rs/releases/tag/v0.15.0#qwen-image-performance). Each records its workload, latency and memory; results are specific to those versions and configurations.

## Project

Ferrum uses AI-assisted development and is evolving; internal APIs may change. Contributions and hardware testing are welcome.

Thanks to [vLLM](https://github.com/vllm-project/vllm) and [llama.cpp](https://github.com/ggml-org/llama.cpp) for code and implementation references. Component licenses are preserved.
