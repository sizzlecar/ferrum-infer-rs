<p align="center">
  <a href="https://ferrum.pandaailabs.com/">
    <img src="assets/brand/ferrum-lockup.svg" alt="Ferrum" width="520">
  </a>
</p>

[中文](README_zh.md) · [MIT](LICENSE) · [Releases](https://github.com/sizzlecar/ferrum-infer-rs/releases)

# Ferrum

Text and image inference in one Rust binary, with CLI and OpenAI-compatible APIs.

**[v0.16.0](https://github.com/sizzlecar/ferrum-infer-rs/releases/tag/v0.16.0)** adds Metal Qwen-Image generation and editing with a GGUF DiT.

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

CUDA uses a complete local copy of [Qwen-Image-2.1-Turbo](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo). Validated on RTX 5090 with Linux CUDA 12.8, cuDNN 9, FP8 text encoder/DiT and BF16 vision/VAE.

Metal supports Qwen-Image generation and single-PNG editing with a Q4_K_M GGUF DiT, FP8 text encoder and BF16 vision/VAE. Validated at 1024×1024 on an M4 Mac mini with 16 GiB unified memory. Use `--image-config` to select the backend, assets and memory budget.

**Metal configuration:** copy [qwen-image-2.1-turbo-metal.json](crates/ferrum-cli/examples/config/qwen-image-2.1-turbo-metal.json) to `image-metal.json` beside your `models/` directory. Place the original model's text encoder, VAE and metadata under `models/Qwen-Image-2.1-Turbo/`, and [the Q4_K_M DiT](https://huggingface.co/m9ter1a/Qwen-Image-2.1-Turbo-GGUF/blob/a94df596a422bc8dfe406655154c0364f79ba042/qwen-image-2.1-turbo-Q4_K_M.gguf) under `models/Qwen-Image-2.1-Turbo-GGUF/`. Alternatively, edit the JSON paths; relative paths resolve from the JSON file's directory. The example keeps one active image, cold component residency, request-scoped edit caching, a 2 GiB Host budget and 90% of Metal's recommended working set, subject to capacity checks. It does not need the original BF16 DiT weights.

The preset enables fused attention and text-encoder weight prefetch in ranges of up to 128 MiB; set `backend.attention_policy` to `reference` to use the original attention path.

<details>
<summary>If the model snapshot omits transformer/config.json</summary>

Save this matching Qwen-Image-2.1-Turbo architecture as `models/Qwen-Image-2.1-Turbo/transformer/config.json`; keep an existing matching file:

```json
{
  "_class_name": "QwenImage21Transformer2DModel",
  "_diffusers_version": "0.41.0.dev0",
  "attention_head_dim": 128, "axes_dims_rope": [16, 56, 56],
  "causal_condition": true, "context_in_dim": 4096, "eps": 0.000001,
  "in_channels": 64, "mlp_ratio": 3, "num_attention_heads": 32,
  "num_layers": 32, "out_channels": 64, "patch_size": 1
}
```

</details>

```sh
ferrum run --image-config image-metal.json \
  --prompt "A red ceramic mug on a white table" --width 1024 --height 1024 --seed 42 --output mug.png
ferrum run --image-config image-metal.json --image mug.png \
  --prompt "Make the mug blue" --width 1024 --height 1024 --seed 43 --output edited.png
ferrum serve --image-config image-metal.json --host 127.0.0.1 --port 8000
```

The API model name is `qwen-image-turbo-metal`. Set precision in this JSON; do not combine `--image-config` with a positional model or `--image-precision`.

CUDA examples:

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

CUDA image LoRA loads one ordinary DiT adapter with a fixed strength at startup.
Both `run` and `serve` accept `--image-lora /path/to/peft-directory` and
`--image-lora-strength 0.8`. A PEFT directory must contain
`adapter_config.json` and `adapter_model.safetensors`. A single Diffusers A/B
file requires an explicit base binding:

```sh
ferrum serve /path/to/Qwen-Image-2.1-Turbo --backend cuda \
  --image-precision fp8_linear_experimental_v1 \
  --image-lora /path/to/style.safetensors \
  --image-lora-base-model Qwen/Qwen-Image-2.1 --image-lora-strength 0.8 \
  --served-model-name qwen-style
```

Single files use unit export scaling unless `--image-lora-alpha` declares
alpha/r; PEFT uses its config's rank and alpha. Missing PEFT base metadata can
be supplied with `--image-lora-base-model`, but it cannot override a conflicting
declaration. Every A/B pair and target is checked; unsupported variants,
unknown targets, incompatible shapes and nonfinite strengths fail before device
execution. Ordinary attention, MLP and time/modulation linear targets are
supported, including fused gate/up exports. Declared structural compatibility
with Qwen-Image-2.1 does not establish an adapter's image quality on Turbo.

The base remains immutable. Composition produces bounded BF16 tensors before
whole-group FP8 quantization, including shared Q/K/V scales. CUDA directory
entrypoints automatically budget the mapped adapter, complete composed DiT
cache and temporary workspaces; insufficient Host headroom is reported before
CUDA construction. This retains composed weights across device eviction.
Explicit JSON uses a top-level `lora` object with `source` and `strength`, and
must budget these owners in its Host limits. Startup receipts record adapter
identity, strength and cache requirements. Without an explicit served alias,
the model name ends in `:lora:<identity-prefix>`; generation and editing use
that identity. Auto batches share the same fixed adapter and strength.
Request-level switching, multiple adapters, Metal/GGUF LoRA and special
distillation sampling contracts are not supported. Turbo keeps its existing
eight-step schedule.

CUDA directory serving uses automatic image batching: an idle GPU starts available work immediately; compatible requests accumulated during execution form the next batch. No batch or gathering-delay option is required. `--max-num-seqs` optionally tightens the model/operator batch ceiling; it does not reserve that many images or select a fixed batch size. Capacity is checked before execution, and oversized candidates are reduced only on an explicit capacity rejection. Scheduling occurs between complete batches, so a large image can delay a later small image.

CUDA directory defaults enable cuDNN SDPA, request-scoped edit-prefix reuse and a bounded multi-shape plan cache: up to 16 entries per stage and 128 MiB of serialized compilation metadata in total. Both `run` and `serve` grow this cache on demand; these limits do not preallocate memory.

CUDA directory entrypoints also select convolution algorithms automatically. The selection ceiling comes from the device budget after declared opaque allowances and the largest component's execution weights. Compilation records each chosen deterministic algorithm and its actual workspace; full-plan admission still checks weights, activations and scratch together before execution. The ceiling is not preallocated. Attention keeps its separate workspace policy. Existing image JSON preserves `cudnn.workspace_limit_bytes`; setting `cudnn.convolution_workspace` to `"auto"` opts into the same automatic selection. The startup receipt records the resolved policy and selection ceiling.

For a CUDA directory with FP8 text encoding, both entrypoints also retain the complete converted text encoder when observed available Host memory covers its model-derived cache size plus conversion, input, transfer and output allowances. Insufficient or unknown headroom keeps this cache off; no partial cache is selected. The startup composition receipt records the decision, observed headroom and effective Host limits. The availability snapshot accounts for process/cgroup limits but reserves no OS memory; actual allocations still require the shared Host account. Explicit image JSON and Metal keep their configured cache behavior.

The CUDA directory also derives its Host output allowance from the declared image geometry and batch ceiling, including retained results and HTTP encoding backing from the previous batch alongside the next batch's readback and encoding requirements. It uses the same available-memory snapshot as the text encoder cache and limits expansion to the remaining headroom. The receipt distinguishes complete overlap capacity, a partial allowance, and the bounded fallback when headroom is unavailable. This budget does not preallocate image buffers; `ferrum run` derives it for one request, while serving still admits each batch against actual live Host bytes.

The CUDA directory preset admits output sizes up to 2048×2048 subject to managed capacity; the Metal preset retains its 1024×1024 bound and configured scheduling. `ferrum run` directly executes one request. Pending requests are bounded by count and Host bytes (one half of the Host input allowance by default); exceeding queue admission limits returns HTTP 429. Queued requests retain small output metadata; Host encoding backing is reserved only for the next executable batch. Encoded transport credit is acquired after inference completes, so waiting for unread response bodies does not occupy the inference worker. Host capacity still bounds retained outputs, and a single output exceeding an immutable limit is rejected. The effective policy, limits, rejection count and actual batch counters are exposed in `/health` under `images.scheduler`.

CUDA Auto derives its finite pending-request count from the 384 MiB pending-byte budget divided by the minimum queued-request metadata cost, independently of the batch ceiling. Actual metadata, prompt capacity, edit backing and prepared inputs consume the byte budget; queue storage grows on demand. Output session capacity also allows for completed responses still held by clients, using the Host output budget divided by the smallest encoded PNG as a conservative count bound. These counts do not preallocate request or response storage; actual Host and encoded-byte limits determine admission. Higher client concurrency can still reach those resource limits; report admission rejections separately from evidence of GPU saturation.

Existing `--image-config` JSON keeps configured batching when `serving.batching` is omitted. Explicit JSON can select `"batching":"auto"` with `"request_batch_max_wait_ms":0`, and optionally set a positive `maximum_pending_bytes`. The historical measurements below used explicit resident service configurations; they do not establish the performance of automatic batching.

`bench-images` can observe either local Ferrum or vLLM-Omni without changing server capacity. Supply an existing workload JSON and the actual server PID; the report path must be new:

```sh
ferrum bench-images --endpoint http://127.0.0.1:8000/v1/images/generations \
  --model qwen-image --server-config "fixed service configuration and version" \
  --dataset image-workload.json --output image-measurement.json \
  --concurrency 8 --repetitions 3 --warmup-repetitions 1 \
  --resource-server-pid 12345 --resource-process-scope process-tree \
  --resource-gpu-index 0 --resource-sample-interval-ms 200
```

Resource monitoring is optional. Each warmup and measured round records its own process identity, physical GPU identity, sampling coverage, errors and observed peaks. Only complete measured windows contribute to aggregate peaks; every sampled peak remains a lower bound on the instantaneous peak. Linux process-tree RSS can count shared pages repeatedly. NVIDIA whole-device used memory includes other processes and driver allocations. Missing observations remain missing and make the command fail after saving the report. Boundary probes are outside request timers; periodic probes compete for CPU and driver resources during requests and can affect latency and throughput. Keep sampling settings identical in comparisons. For multiple sequential endpoints, provide one server PID per endpoint; repeat the GPU flag to sum explicitly selected physical devices.

For local Ferrum lifecycle evidence, build the independent Rust example `probe_image_lifecycle`. It reads the same strict image dataset schema and a fixed prefix of samples; edit samples require `--edit-endpoint`. Keep datasets, reference PNGs and new report files outside the repository:

```sh
probe_image_lifecycle --endpoint http://127.0.0.1:8000/v1/images/generations \
  --health-endpoint http://127.0.0.1:8000/health --model qwen-image \
  --dataset /outside-repo/images-2k.json --output /outside-repo/hold-body.json \
  --mode hold-body --requests 4 --hold-body-ms 15000 \
  --observation-ms 60000 --request-timeout-ms 600000
```

`hold-body` obtains complete headers, waits without polling the body, then validates the PNG. `drop-body` instead drops that unread response after the wait. `cancel` drops the client request future after `--cancel-after-ms`; `observe-shutdown` records an externally controlled stop/restart, with optional `--recovery-delay-ms`. The probe never starts, stops or signals a service. Every mode records bounded `/health` scheduler/output/Host observations and issues normal recovery requests. Reports distinguish observed, not observed and failed findings; a client action alone does not prove server cancellation, disconnect handling or backpressure. Transport buffering can consume bytes while the application holds a body. HTTP 429, expected client actions and observed shutdown transport are counted separately; unexpected protocol, PNG or observation errors produce a failing exit after saving the report. Recovery success must be checked explicitly in the findings.

### Throughput vs vLLM-Omni

v0.15.0 candidate measurements · RTX 5090 · Qwen-Image-2.1-Turbo · 8 steps · FP8 text encoder/DiT, BF16 VAE. Four paired pre-release runs per workload, 640 measured requests, zero errors. Mean images/s; C is client concurrency.

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
| Images | Qwen-Image-2.1-Turbo generation and single-PNG editing; validated on Linux CUDA SM120 and at 1024×1024 on M4 Metal with the configuration above |

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
