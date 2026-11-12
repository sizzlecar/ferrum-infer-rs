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

To require authentication, set `FERRUM_API_KEY` before starting `ferrum serve`
and send `Authorization: Bearer $FERRUM_API_KEY` from clients. The same setting
protects text and image APIs; `/health` and static Playground assets remain
public. `--api-key KEY` and `[server] api_key = "KEY"` are also supported,
with CLI/environment taking precedence over the config file. Without a key,
serving retains its existing unauthenticated behavior. Available from v0.19.1;
v0.19.0 binaries do not include API key authentication.

## Image generation and editing

Supports **Qwen-Image-2.1-Turbo** and **P-H3-Image** with ordered multi-reference editing: up to ten images for Qwen from v0.19.1 and eight for P-H3. v0.19.0 accepts only one Qwen reference. P-H3 also supports continuous batching on Linux CUDA SM89/SM120.

Use a local [Qwen-Image-2.1-Turbo](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo) model directory. Linux CUDA example with FP8 text encoder/DiT:

```sh
ferrum serve /path/to/Qwen-Image-2.1-Turbo --backend cuda \
  --image-precision fp8_linear_experimental_v1 \
  --served-model-name qwen-image --port 8000
```

| API | Input |
|---|---|
| `POST /v1/images/generations` | JSON text prompt |
| `POST /v1/images/edits` | Multipart prompt + ordered PNG references: up to ten for Qwen from v0.19.1 (one in v0.19.0); up to eight for P-H3 |

Both return PNG in `b64_json`. CUDA batches compatible requests automatically within capacity limits. Validated on L20 (SM89) and RTX 5090 (SM120); cuDNN 9 is required. Qwen also supports Metal generation and editing with a GGUF DiT, validated at 1024×1024 on an M4 Mac mini with 16 GiB memory.

For P-H3, copy the [CUDA configuration](crates/ferrum-cli/examples/config/p-h3-image-cuda.json), set the model and CUDA/cuDNN paths, then start:

```sh
ferrum serve --image-config p-h3-image-cuda.json --served-model-name P-H3-Image --port 8000
```

P-H3 uses the Comfy W4A8 text encoder/DiT, INT8 VAE and native prompt expansion. For either model, repeat multipart `image[]` fields or CLI `--image` arguments in the intended reference order (for example, a character followed by a pose). P-H3 accepts `size=2720x1536` for a landscape edit. See the [image guide](docs/image-generation.md) for examples, capacity limits, Metal and Qwen CUDA LoRA.

## Supported scope

| Scope | Models, formats and backend limits |
|---|---|
| Text architectures | Llama, Mistral, Qwen2, Qwen3 dense/MoE, Qwen3.5 hybrid dense/MoE, Gemma 3 text |
| Weights | GGUF, including Q4_K/Q5_K/Q6_K/Q8_0; SafeTensors |
| CUDA quantization | Selected model paths: GPTQ INT4, compressed-tensors INT4, block-FP8 |
| Specific CUDA models | Gemma 4 compressed-tensors W4A16; GPT-OSS MXFP4 |
| PQ2_0 | Complete-model validation on Metal; CUDA operator coverage only |
| Images | Qwen-Image-2.1-Turbo generation/editing on Linux CUDA and Metal (v0.19.1: up to ten references; v0.19.0: one); P-H3-Image generation/editing with up to eight references on Linux CUDA SM89/SM120. P-H3 Metal is not supported |

Support depends on the architecture, format and backend combination; text support does not imply vision support. CPU is a compatibility backend. See the [validation catalog](docs/release-regression-catalog.md) for tested combinations and coverage.

Embedding, ASR, TTS and legacy execution APIs are retired. See [migration notes](https://github.com/sizzlecar/ferrum-infer-rs/pull/441).

## Performance

**P-H3-Image on RTX 5090 · 2720×1536 editing:**

| Workload | End-to-end time, two runs | Output throughput | Request latency P50 / P99 |
|---|---:|---:|---:|
| One reference, C1 | 26.94 / 23.65 s | — | Same as request time (one sample/run) |
| Eight requests with distinct references, C8 | 119.13 / 118.40 s | 4.03 / 4.05 images/min | 87.24 / 118.76 s; 86.38 / 118.08 s |

Linux/WSL, 48 GB guest-memory limit, W4A8 DiT/text encoder, INT8 VAE, F16 prompt expansion enabled, 8 steps, fixed server capacity 8. Both runs completed 8/8 requests without errors and met the measured P99 ≤120 s threshold; two small runs do not establish a stable production SLO. Candidate measurements precede the final release package. C8 sampled peak RSS: 38.71 / 39.33 GiB; peak VRAM was not collected. [Configuration, workload and evidence](docs/performance-rtx5090-p-h3-20261111.md).

**Qwen3.5-4B Q4_K_M · RTX 5090 · v0.19.1 development candidate:** Q6 prefill acceleration is selected automatically when the native operators qualify. Same-session ShareGPT comparison with llama.cpp b12069:

| Concurrency | TTFT P50: Ferrum / llama.cpp | TTFT P99: Ferrum / llama.cpp | Output tok/s: Ferrum / llama.cpp |
|---|---:|---:|---:|
| 8 | 47.23 / 56.70 ms | 153.78 / 196.50 ms | 1044.50 / 713.66 |
| 16 | 50.31 / 71.05 ms | 217.37 / 378.88 ms | 1524.61 / 824.84 |

64 requests × 2 repetitions per cell, fixed server capacity 32, context/batch 2048, F16 KV, thinking and prefix caching disabled. Both engines had zero request/protocol errors and passed P99 limits of TTFT 500 ms, TPOT 50 ms and client-visible SSE text-event ITL 100 ms. Values average the per-repeat metrics; the small sample does not establish a production P99 guarantee. These measurements precede the final archive and do not establish an RTX 4070 Ti result. [Full TPOT/ITL, memory, commands, output checks and limitations](https://github.com/sizzlecar/ferrum-infer-rs/pull/514).

**Earlier RTX 5090 model comparisons.** Throughput is tok/s, shown as **Ferrum / reference**. C8/C16 are client concurrency; each row identifies the measured Ferrum version.

| Model / source weights | Measured version | Reference | C8 tok/s | C16 tok/s | SLO: C8 / C16 |
|---|---|---|---:|---:|---|
| Qwen3.5-4B · Q4_K_M | 0.18.3 | llama.cpp | 960.8 / 634.7 | 1294.8 / 729.4 | pass / Ferrum ITL fail |
| Qwen3.5-9B · Q4_K_M | 0.18.3 | llama.cpp | 731.6 / 550.5 | 942.0 / 654.7 | pass / Ferrum ITL fail |
| Qwen3.5-4B · BF16 | 0.18.3 | vLLM | 840.0 / 684.6 | 1306.8 / 1153.9 | pass / pass |
| Qwen3.5-9B · BF16 | 0.18.3 | vLLM | 563.4 / 508.2 | 875.0 / 864.3 | pass / pass |
| Qwen3-30B-A3B · GPTQ INT4 | 0.18.3 | vLLM | 630.5 / 824.0 | 848.8 / 1263.7 | pass / pass |
| Qwen3.5-35B-A3B · GPTQ INT4 | 0.18.3 | vLLM | 717.6 / 625.6 | 978.3 / 947.4 | pass / vLLM ITL fail |
| GPT-OSS-20B · MXFP4 | 0.18.4 R11 | vLLM | 890.9–892.3 / 819.8 | 1383.5–1385.9 / 1429.8 | pass / pass |
| Qwen3.8-27B · UD-Q4_K_M | 0.13 candidate | llama.cpp | 356.42 / 291.60 | 535.51 / 377.41 | both fail / both fail |
| Qwen3.8-27B · CT INT4 | 0.13 candidate | vLLM | 374.32 / 447.06 | 654.04 / 733.86 | Ferrum fails / both fail |

“Pass” means both engines pass both repetitions; the noted ITL failure occurs in repetition 1. The 4B/9B/30B/35B cells contain 64 requests × 2 repetitions per engine, all successful with zero errors/rejections. GPT-OSS uses 32 requests × 2 fresh Ferrum processes and one retained reference run. The 27B cells use 200 requests × 2 repetitions, zero errors, and separate stricter SLOs; do not rank across workloads.

For 4B/9B/30B/35B/GPT-OSS, P99 limits are TTFT 2000 ms, TPOT 50 ms and visible SSE ITL 100 ms. Each pair uses the same source weights and workload. In BF16-source and 35B comparisons, Ferrum uses F16 activations/KV, while vLLM uses BF16. The 30B gap remains; small pooled leads for 9B BF16 and 35B reverse between repetitions and do not establish a stable advantage.

Full TTFT/TPOT/ITL P50/P99, peak VRAM/RSS, commands and sampling: [4B / 9B / 30B / 35B](docs/performance-rtx5090-popular-models-20261101.md) · [GPT-OSS](docs/performance-rtx5090-gpt-oss-20261102.md) · [27B](docs/performance-popular-models-p0.zh.md).

<details>
<summary>Latest GPT-OSS optimization: latency and resource details</summary>

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

</details>

## Documentation

- [API compatibility](docs/openai-api-compatibility.md) · [Agent serving](docs/agent-serving.md)
- [Numerical profiles](docs/numerical-execution.zh.md) · [Caching](docs/cache-product.md)

## Project

Ferrum uses AI-assisted development and is evolving; internal APIs may change. Contributions and hardware testing are welcome.

Thanks to [vLLM](https://github.com/vllm-project/vllm) and [llama.cpp](https://github.com/ggml-org/llama.cpp) for code and implementation references. Component licenses are preserved.
