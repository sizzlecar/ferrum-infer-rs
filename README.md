<p align="center">
  <a href="https://ferrum.pandaailabs.com/">
    <img src="assets/brand/ferrum-lockup.svg" alt="Ferrum" width="520">
  </a>
</p>

[中文](README_zh.md) · [Releases](https://github.com/sizzlecar/ferrum-infer-rs/releases) · [Performance](#performance) · [MIT](LICENSE)

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
- $C$: fraction of $W$ successfully completed by the cutoff; rejected, failed, timed-out and unfinished requests remain in the denominator. Declare $p_k$, $S_k$ and minimum completion rate $0<\rho\le1$ before measuring. [Sampling rules](docs/performance-rtx5090-gpt-oss-20261102.md).

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

## Supported scope

| Scope | Models, formats and backend limits |
|---|---|
| Text architectures | Llama, Mistral, Qwen2, Qwen3 dense/MoE, Qwen3.5 hybrid dense/MoE, Gemma 3 text |
| Weights | GGUF, including Q4_K/Q5_K/Q6_K/Q8_0; SafeTensors |
| CUDA quantization | Selected model paths: GPTQ INT4, compressed-tensors INT4, block-FP8 |
| Specific CUDA models | Gemma 4 compressed-tensors W4A16; GPT-OSS MXFP4 |
| PQ2_0 | Complete-model validation on Metal; CUDA operator coverage only |
| Images | Qwen-Image-2.1-Turbo generation and single-PNG editing; Linux CUDA SM89/SM120 and Metal. Validated at 1024×1024 on M4 Metal |

Support depends on the architecture, format and backend combination; text support does not imply vision support. CPU is a compatibility backend. See the [validation catalog](docs/release-regression-catalog.md) for tested combinations and coverage.

Embedding, ASR, TTS and legacy execution APIs are retired. See [migration notes](https://github.com/sizzlecar/ferrum-infer-rs/pull/441).

## Performance

**RTX 5090 · GPT-OSS 20B · MXFP4:** the optimization shipped in **0.18.4** reaches **108.7–108.9% of vLLM throughput at C8** and **96.8–96.9% at C16**, meeting the predefined latency SLOs. Measurements use its R11 candidate, not a fresh benchmark of the final release archive.

Same GPU, source weights and pinned ShareGPT samples; natural EOS, at most 512 output tokens. Ferrum uses F16 outer activations/KV and BF16 experts; vLLM uses BF16 activations/KV. Latencies are **P50 / P99, milliseconds**; ITL measures non-empty visible SSE text events.

| Concurrency | Engine / fresh process | Output tok/s | TTFT | TPOT | SSE ITL |
|---:|---|---:|---:|---:|---:|
| 8 | Ferrum R11 #1 | 890.93 | 155.70 / 227.27 | 8.17 / 8.62 | 8.00 / 14.38 |
| 8 | Ferrum R11 #2 | 892.30 | 157.81 / 226.31 | 8.21 / 8.61 | 7.95 / 13.02 |
| 8 | vLLM 0.31.0 | 819.77 | 94.75 / 387.13 | 8.71 / 8.96 | 8.16 / 33.00 |
| 16 | Ferrum R11 #1 | 1383.53 | 346.93 / 462.27 | 10.00 / 10.29 | 9.88 / 11.73 |
| 16 | Ferrum R11 #2 | 1385.86 | 340.35 / 444.62 | 10.10 / 10.39 | 9.89 / 12.27 |
| 16 | vLLM 0.31.0 | 1429.75 | 174.08 / 183.29 | 10.14 / 10.38 | 9.47 / 34.24 |

All rows meet P99 TTFT ≤2000 ms, TPOT ≤50 ms and SSE ITL ≤100 ms, with 32/32 successful requests and zero errors/rejections per cell. Ferrum ran two fresh processes, one C8→C16 sweep each (128/128 requests); the vLLM baseline was retained. C16 first-text latency remains higher than vLLM.

Whole-sweep peak device VRAM / window process RSS: Ferrum #1 **15.978 / 1.177 GiB**, #2 **16.165 / 1.192 GiB**; vLLM **29.937 / 3.145 GiB**. These are not per-concurrency peaks; vLLM preallocates KV, and its RSS covers EngineCore only, so these figures do not establish a memory-efficiency ratio.

[Workload, commands, output validity and full evidence](docs/performance-rtx5090-gpt-oss-20261102.md). These measurements cover this model and configuration only.

## Documentation

- [API compatibility](docs/openai-api-compatibility.md) · [Agent serving](docs/agent-serving.md)
- [Numerical profiles](docs/numerical-execution.zh.md) · [Caching](docs/cache-product.md)

## Project

Ferrum uses AI-assisted development and is evolving; internal APIs may change. Contributions and hardware testing are welcome.

Thanks to [vLLM](https://github.com/vllm-project/vllm) and [llama.cpp](https://github.com/ggml-org/llama.cpp) for code and implementation references. Component licenses are preserved.
