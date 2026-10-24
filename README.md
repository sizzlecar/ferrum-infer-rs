<p align="center">
  <a href="https://ferrum.pandaailabs.com/">
    <img src="assets/brand/ferrum-lockup.svg" alt="Ferrum" width="520">
  </a>
</p>

[中文](README_zh.md) · [MIT](LICENSE) · [Releases](https://github.com/sizzlecar/ferrum-infer-rs/releases)

# Ferrum

Text and image inference in one Rust binary, with CLI and OpenAI-compatible APIs.

**[v0.15.0](https://github.com/sizzlecar/ferrum-infer-rs/releases/tag/v0.15.0)** adds Qwen-Image generation and editing.

**Goal: maximize output throughput while meeting TTFT, TPOT and ITL SLOs.**

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

Maximize successful output token throughput under a fixed workload and measurement window, subject to tail-latency SLOs and a minimum request completion rate.

- $W$: fixed request cohort and arrival schedule; $T>0$: fixed window, whose end is the common cutoff. $N_{\mathrm{out}}$: valid output tokens from requests successfully completed within that window.
- $L_k$: latency samples; $Q_{p_k}$: quantile; $S_k$: limit. TTFT includes queueing; missing latency evidence does not establish compliance. [Sampling rules](docs/performance-popular-models-p0.zh.md).
- $C$: fraction of $W$ successfully completed by the cutoff. Rejected, failed, timed-out and unfinished requests remain in the denominator. Declare $p_k$, $S_k$ and $0<\rho\le1$ before measuring.

## Install and run

macOS Apple Silicon / Linux x86_64:

```sh
curl -fsSL https://ferrum.pandaailabs.com/install.sh | sh
```

Windows x64, from PowerShell:

```powershell
irm https://ferrum.pandaailabs.com/install.ps1 | iex
```

Installers verify checksums. macOS uses Metal; Windows CUDA targets sm89. Linux/WSL selects CUDA when every visible GPU reports 8.9 or every GPU reports 12.0; sm120 requires v0.13.0 or newer. Unsupported/mixed GPUs select CPU automatically; requiring CUDA there, or requesting a release without matching assets, fails. [Runtime requirements](docs/release-regression.md#prepare-the-candidate).

```sh
ferrum run qwen3.5:4b-q4_k_m --disable-thinking
ferrum serve --model qwen3.5:4b-q4_k_m --served-model-name ferrum --disable-thinking --port 8000
```

Weights download automatically. API URL: **`http://127.0.0.1:8000/v1`**, model `ferrum`. Omit `--disable-thinking` for the template default. [Streaming, tools and structured output](docs/openai-api-compatibility.md).

Defaults adapt execution and capacity to capabilities, hardware and memory; explicit settings prevail. Qualified SM120 GGUF paths can use 8-bit activation quantization and integer dot products. `--numerical-profile qwen3_5.f32-master` retains the original mixed-precision reference. Batch layouts can change output. [Numerical profiles](docs/numerical-execution.zh.md) · [Caching](docs/cache-product.md).

## Image generation and editing

Use a complete local copy of [Qwen-Image-2.1-Turbo](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo). Validated on RTX 5090 with Linux CUDA 12.8, cuDNN 9, FP8 text encoder/DiT and BF16 vision/VAE.

~~~sh
ferrum run /path/to/Qwen-Image-2.1-Turbo \
  --image-precision fp8_linear_experimental_v1 \
  --prompt "A red fox in a snowy forest" --width 1024 --height 1024 --output fox.png

ferrum run /path/to/Qwen-Image-2.1-Turbo --image fox.png \
  --image-precision fp8_linear_experimental_v1 \
  --prompt "Make the fox blue" --width 1024 --height 1024 --output edited.png

ferrum serve --model /path/to/Qwen-Image-2.1-Turbo --served-model-name qwen-image \
  --image-precision fp8_linear_experimental_v1 --port 8000
~~~

HTTP: `POST /v1/images/generations` (JSON) and `POST /v1/images/edits` (multipart). Edits accept one 8-bit RGB/RGBA PNG; responses contain PNG as `b64_json`. Editing can redraw background/details.

The local-directory preset supports up to 1024×1024. 2K and the resident service configurations measured below use `--image-config`.

### Throughput vs vLLM-Omni

RTX 5090 · Qwen-Image-2.1-Turbo · 8 steps · FP8 text encoder/DiT, BF16 VAE. Four paired pre-release runs per workload, 640 measured requests, zero errors. Mean images/s; C is client concurrency.

| Workload | C | Ferrum | vLLM-Omni | Ferrum / Omni |
|---|---:|---:|---:|---:|
| 1024² generation | 8 | 0.5193 | 0.4428 | 1.173× |
| 1024² editing | 1 | 0.2943 | 0.2631 | 1.119× |
| 2048² generation | 2 | 0.0799 | 0.0772 | 1.035× |
| 2048² editing | 2 | 0.0686 | 0.0685 | ≈1.00×* |

Both implementations met the declared P99 limits: 20 / 5 / 30 / 30 seconds, in row order. *2K editing's 95% ratio interval is 0.9976–1.0077; the difference is inconclusive. These are resident-service measurements, not cold CLI defaults. [Configuration, latency and memory](https://github.com/sizzlecar/ferrum-infer-rs/releases/tag/v0.15.0#qwen-image-performance).

## Supported scope

| Scope | Supported paths |
|---|---|
| Text architectures | Llama, Mistral, Qwen2, Qwen3 dense/MoE, Qwen3.5 hybrid dense/MoE, Gemma 3 text |
| Weights | GGUF, including Q4_K/Q5_K/Q6_K/Q8_0; SafeTensors. Selected CUDA paths: GPTQ INT4, compressed-tensors INT4, block-FP8 |
| Specific paths | Gemma 4 compressed-tensors W4A16 and GPT-OSS MXFP4: CUDA. PQ2_0: complete-model validation on Metal; CUDA operator coverage only |
| Images | Qwen-Image-2.1-Turbo generation and single-PNG editing; validated on Linux CUDA SM120 |

Support varies by combination; CPU is a compatibility backend. See [validation scope](docs/release-regression-catalog.md).

This architecture update retires embedding, ASR, TTS and legacy execution APIs. See [migration notes](https://github.com/sizzlecar/ferrum-infer-rs/pull/441).

## Text throughput (0.13 candidate)

**0.13 candidate · RTX 5090 · Qwen3.8-27B · ShareGPT.** Fixed 32 slots, context/batch 2048. Each cell: 200 requests × 2 repeats, 8 warmups per repeat, zero protocol errors. Throughput = total output tokens / total measured time; C is client concurrency.

These fixed-request measurements use each run's actual elapsed time; the objective above specifies a fixed measurement window.

GGUF Q4_K_M, same weights; output tok/s:

| C | Ferrum | llama.cpp b11429 | F / llama |
|---:|---:|---:|---:|
| 4 | 179.25 | 170.13 | 105.36% |
| 8 | 356.42 | 291.60 | 122.23% |
| 16 | 535.51 | 377.41 | 141.89% |
| 32 | 666.24 | 435.57 | 152.96% |

Compressed-tensors INT4, group size 32; output tok/s:

| C | Ferrum | vLLM 0.31 | F / vLLM |
|---:|---:|---:|---:|
| 4 | 210.32 | 233.45 | 90.09% |
| 8 | 374.32 | 447.06 | 83.73% |
| 16 | 654.04 | 733.86 | 89.12% |
| 32 | 836.79 | 1035.09 | 80.84% |

Several cells fail latency SLOs. CT C32's second repeat reached **78.91%**: not every repeat met 80%. CT uses FP16 activations/KV in Ferrum versus BF16 in vLLM. [Full results](docs/performance-popular-models-p0.zh.md) include latency percentiles, memory, numerical differences and workload policy. This refactor provides execution and control interfaces for future joint throughput/latency-SLO optimization.

## Project

Ferrum uses AI-assisted development and iterates quickly. Internal APIs may change; model coverage and performance evolve. Quality remains a priority through correctness checks and model validation. Personal GPU resources are limited; hardware support and contributions are welcome.

Thanks to [vLLM](https://github.com/vllm-project/vllm) and [llama.cpp](https://github.com/ggml-org/llama.cpp) for code and implementation references. Component notices are preserved.
