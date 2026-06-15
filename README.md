# ferrum-infer-rs

[![Crates.io](https://img.shields.io/crates/v/ferrum-cli.svg)](https://crates.io/crates/ferrum-cli)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/sizzlecar/ferrum-infer-rs/blob/main/LICENSE)

> Rust-native LLM inference for OpenAI-compatible local and private serving.

**One binary. No Python runtime. Hardware-accelerated on Apple Silicon and NVIDIA CUDA.**

Ferrum is a lightweight inference engine for running and serving transformer LLMs with an OpenAI-compatible API.
It is built for developers and teams who want predictable deployment, practical serving performance, and a clean Rust-native runtime for local, edge, workstation, and production inference.

[中文说明](README_zh.md)

## Quick Start

Install a prebuilt binary:

```bash
brew tap sizzlecar/ferrum
brew install ferrum        # macOS Apple Silicon Metal / Linux x86_64 CPU
brew install ferrum-cuda   # Linux x86_64 CUDA sm89 build
ferrum --version
```

Run a model directly:

```bash
export HF_TOKEN=hf_your_token_here   # only needed for gated models
ferrum run qwen3:4b
```

Serve the same model through an OpenAI-compatible API:

```bash
ferrum serve --model qwen3:4b --port 8000

curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3:4b","messages":[{"role":"user","content":"Hello"}]}'
```

Use the larger functional CUDA lane when you want a 27B-class agent model on a
24 GB RTX 4090. This lane has correctness and concurrency evidence, but it has
a known performance gap and is not release-grade yet:

```bash
ferrum serve --model gemma3:27b-gptq --max-num-seqs 16 --kv-capacity 400 --port 8000
```

## Why Ferrum?

- **One binary:** ship `ferrum run` and `ferrum serve` without a Python service in the runtime path.
- **OpenAI-compatible API:** reuse existing OpenAI-shaped clients, SDKs, and HTTP tooling.
- **Hardware accelerated:** use Apple Silicon Metal or NVIDIA CUDA from the same project.
- **Rust-native runtime:** fewer moving parts, simpler deployment, and a runtime that is easy to embed or package.
- **Practical serving performance:** continuous batching, paged KV cache, INT4 GPTQ/Marlin paths, CUDA Graphs, and release-tested concurrency gates.

## What Ferrum is good at

Ferrum is built for developers and teams building:

- local AI agents
- private OpenAI-compatible inference services
- Apple Silicon LLM applications
- CUDA-accelerated inference servers
- edge and workstation deployments
- Rust-native AI infrastructure

## Performance Snapshot

Ferrum is designed for practical high-throughput serving on modern accelerators, with raw benchmark logs checked into the repository instead of only summary claims.

Historical CUDA same-pod throughput on RTX 4090 with
`Qwen3-30B-A3B-GPTQ-Int4` for the opt-in FA2 direct-FFI path:

| Concurrency | Ferrum tok/s | vLLM 0.20.2 tok/s | Ferrum / vLLM |
| ---: | ---: | ---: | ---: |
| 1 | `160.4 +/- 0.2` | `183.9 +/- 0.2` | `0.872x` |
| 4 | `446.3 +/- 7.0` | `512.5 +/- 2.8` | `0.871x` |
| 16 | `1185.1 +/- 12.3` | `1331.9 +/- 5.7` | `0.890x` |
| 32 | `1641.9 +/- 4.8` | `1972.9 +/- 18.6` | `0.832x` |

Full CUDA methodology and raw artifacts for that historical comparison are in
[`docs/bench/cuda-rtx4090-2026-05-30-m3-80pct-confirmed/`](docs/bench/cuda-rtx4090-2026-05-30-m3-80pct-confirmed/).
Do not treat this table as a current source-linked/default release-gate claim;
release candidates must use the current G0/G1-G4 CUDA artifacts for their exact
binary, git SHA, runtime config, and same-hardware results.

Current 0.7.7 source release gates:

| Target | Model / workload | Result | Evidence |
| --- | --- | --- | --- |
| Apple Silicon Metal source gate | Llama-3.1-8B, Qwen3-8B, and Qwen3-30B-A3B GGUF; `ferrum run`, `ferrum serve`, tool calls, stream, stateful loop, and `16/64` throughput cells | `FERRUM GATE metal PASS`; Qwen3-30B-A3B c=16 current `68.5 tok/s`, `32/32` completed, `0` errors | [`docs/release/g0/0.7.7/metal/metal-readme/summary.md`](docs/release/g0/0.7.7/metal/metal-readme/summary.md) |
| CUDA RTX 4090 source gate | `Qwen/Qwen3-30B-A3B-GPTQ-Int4`, random `256/128`, c=1/4/16/32, `n_repeats=3` | `FERRUM GATE cuda-full PASS`; c=1/4/16/32 candidate `164.2` / `353.3` / `636.9` / `706.0 tok/s`; every cell `384/384` completed, `0` errors | [`docs/release/g0/0.7.7/cuda-full/summary.json`](docs/release/g0/0.7.7/cuda-full/summary.json) |
| CUDA RTX 4090 dense source gate | `hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4`, random `256/128`, c=1/4/16/32, `n_repeats=3` | `FERRUM GATE cuda-llama-dense PASS`; c=1/4/16/32 output `122.9` / `324.3` / `640.2` / `745.6 tok/s`; every cell `288/288` completed, `0` errors | [`docs/release/g0/0.7.7/cuda-llama-dense/bench-serve.json`](docs/release/g0/0.7.7/cuda-llama-dense/bench-serve.json) |

## API Compatibility

Ferrum exposes OpenAI-shaped chat completions for local and private deployments. The endpoint contract, explicit rejections, tool-field status, usage accounting, and structured-output limits are documented in [`docs/openai-api-compatibility.md`](docs/openai-api-compatibility.md).

### Tool Calling

`/v1/chat/completions` accepts OpenAI-style function tools. Tool execution is
caller-owned: Ferrum renders tool definitions into the model prompt and returns
model-emitted calls as OpenAI-shaped `tool_calls` in non-streaming responses and
streaming deltas.

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3:4b",
    "messages": [{"role": "user", "content": "Call calc for 123+456"}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "calc",
        "description": "Evaluate an arithmetic expression.",
        "parameters": {
          "type": "object",
          "properties": {"expression": {"type": "string"}},
          "required": ["expression"]
        }
      }
    }],
    "tool_choice": "required",
    "max_tokens": 128
  }'
```

Supported tool choices include `auto`, `none`, `required`, and a specific
function selector. Non-function tool types and undeclared forced tools are
rejected with OpenAI-style 400 errors instead of being silently ignored.

### Structured Output

Ferrum supports OpenAI `response_format` for text, best-effort `json_object`,
and strict `json_schema` validation for a conservative schema subset.

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3:4b",
    "messages": [{"role": "user", "content": "Return the sum of 123+456."}],
    "response_format": {
      "type": "json_schema",
      "json_schema": {
        "name": "Answer",
        "strict": true,
        "schema": {
          "type": "object",
          "additionalProperties": false,
          "properties": {"answer": {"type": "integer"}},
          "required": ["answer"]
        }
      }
    },
    "max_tokens": 128
  }'
```

Strict schemas are validated before non-streaming responses return. Strict
schema streaming buffers generated content until validation passes, then emits
valid content and one final `[DONE]`. Unsupported schema constructs fail fast
with `param=response_format.json_schema`.

### Prefix and Session Cache

Prefix cache is an explicit serving option for repeated or shared-prefix
workloads. Session cache is also opt-in and uses caller-provided session ids.

```bash
ferrum serve qwen3:4b \
  --enable-prefix-cache \
  --session-cache memory \
  --session-cache-max-entries 128 \
  --session-cache-max-tokens 4096

curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-Ferrum-Session: agent-session-1" \
  -d '{"model":"qwen3:4b","messages":[{"role":"user","content":"Remember ferrum-blue."}]}'
```

`/health` reports whether prefix cache is active and whether the path is real
KV reuse. `/metrics` exposes prefix hit/miss, saved prefill token, entry, byte,
and session-cache counters. Details are in [`docs/cache-product.md`](docs/cache-product.md).

## Installation

Homebrew:

```bash
brew tap sizzlecar/ferrum
brew install ferrum        # macOS Metal / Linux CPU
brew install ferrum-cuda   # Linux x86_64 CUDA sm89 build
```

Prebuilt release tarballs:

```bash
# Linux x86_64 CPU
curl -L https://github.com/sizzlecar/ferrum-infer-rs/releases/latest/download/ferrum-linux-x86_64.tar.gz | tar xz
./ferrum --help

# Linux x86_64 CUDA, sm89 build
curl -L https://github.com/sizzlecar/ferrum-infer-rs/releases/latest/download/ferrum-linux-x86_64-cuda-sm89.tar.gz | tar xz
LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-} ./ferrum --help

# macOS Apple Silicon Metal
curl -L https://github.com/sizzlecar/ferrum-infer-rs/releases/latest/download/ferrum-macos-aarch64.tar.gz | tar xz
./ferrum --help
```

Linux x86_64 is the CPU build. Linux x86_64 CUDA is built for `sm89` and requires a compatible NVIDIA driver plus CUDA runtime libraries on the target host. macOS aarch64 is the Metal build.

From source:

```bash
cargo install ferrum-cli
cargo build --release -p ferrum-cli --bin ferrum
```

## Benchmarks / Docs

- CUDA vLLM comparison: [`docs/bench/cuda-rtx4090-2026-05-30-m3-80pct-confirmed/`](docs/bench/cuda-rtx4090-2026-05-30-m3-80pct-confirmed/)
- Current 0.7.7 G0 source artifacts: [`docs/release/g0/0.7.7/`](docs/release/g0/0.7.7/)
- 2026-06 model coverage evidence: [`docs/goals/model-coverage-2026-06-12/`](docs/goals/model-coverage-2026-06-12/)
- Apple Silicon regression gate: `scripts/metal_readme_regression.py` and `scripts/release/validate_metal_readme_regression.py`
- OpenAI API compatibility: [`docs/openai-api-compatibility.md`](docs/openai-api-compatibility.md)
- Prefix/session cache product surface: [`docs/cache-product.md`](docs/cache-product.md)
- Module status notes: [`docs/status/`](docs/status/)

## Supported Models

| Architecture | Apple Silicon | CUDA | INT4 (GPTQ) | Tensor Parallel |
|---|:---:|:---:|:---:|:---:|
| LLaMA (3.x, TinyLlama, Vicuna, Mistral) | ✓ | ✓ | ✓ | ✓ |
| Qwen3 dense (0.6B – 32B) | ✓ | ✓ | ✓ | ✓ |
| Qwen3-MoE (30B-A3B, Coder-30B-A3B) | ✓ | ✓ | ✓ | — |
| Gemma 3 text (1B, 27B) | 1B GGUF | functional | 27B GPTQ known-gap | — |
| Qwen2 / Qwen2.5 | ✓ | ✓ | ✓ | — |
| DeepSeek-R1 (0528-Qwen3-8B; Distill-Qwen-14B/32B; Distill-Llama-70B) | ✓ | ✓ | ✓ | 70B layer-split |
| Mistral Small 3.2 / Magistral Small (24B) | ✓ | — | — | — |
| BERT (embeddings) | ✓ | — | — | — |
| Whisper ASR (tiny → large-v3-turbo) | ✓ | — | — | — |
| Qwen3-TTS (0.6B / 1.7B) | ✓ | — | — | — |
| CLIP / Chinese-CLIP / SigLIP (text + image) | ✓ | — | — | — |

### Model coverage certification (2026-06)

Rows below are model lanes with saved gate evidence, not marketing-only
compatibility claims. W1 rows passed the ladder for the marked lanes:
chat-template golden byte-equality vs `transformers` (L0), known-answer 10/10
at temp 0, multi-turn KV reuse, stream==non-stream, natural EOS + custom stop +
max_tokens mechanics (L2/L3), and the agent gate — required tool-call 10/10 +
strict `json_schema` 20/20 (L4, "agent-grade"). W2 adds Gemma 3 27B on CUDA as
functional coverage, with L5 concurrency and a same-card llama.cpp sanity
floor. It does not yet meet the release-grade 80% baseline target. Artifacts:
[`docs/goals/model-coverage-2026-06-12/artifacts/`](docs/goals/model-coverage-2026-06-12/artifacts/).
Lane split: **GGUF serves on the Metal lane; CUDA serves GPTQ/safetensors**
(GGUF decoding is not wired into the CUDA engine yet).

| Model | Aliases | Metal (GGUF Q4_K_M) | CUDA | Agent-grade |
|---|---|:---:|:---:|:---:|
| Qwen3-Coder-30B-A3B-Instruct | `qwen3-coder:30b[-q4_k_m/-gptq]` | ✓ | GPTQ: known issue¹ | ✓ (Metal) |
| Gemma 3 27B (text) | `gemma3:27b`, `gemma3:27b-gptq` | 1B GGUF smoke; 27B waived⁵ | GPTQ functional + L5⁵ | ✓ (correctness) |
| DeepSeek-R1-0528-Qwen3-8B | `deepseek-r1:8b[-q4_k_m]` | ✓ | ✓ BF16 | ✓ |
| DeepSeek-R1-Distill-Qwen-32B | `deepseek-r1:32b[-q4_k_m/-gptq]` | 32 GB Mac: not practical² | ✓ GPTQ | tools ✓ / schema³ |
| Qwen3-14B / Qwen3-32B | `qwen3:14b/32b[-q4_k_m/-gptq]` | 14B ✓ / 32B² | 32B ✓ GPTQ | 14B ✓ / 32B³ |
| Qwen2.5-Coder-32B-Instruct | `qwen2.5-coder:32b[-q4_k_m/-gptq]` | —² | ✓ GPTQ | ✓ (CUDA) |
| Mistral Small 3.2 (24B) | `mistral-small:24b-q4_k_m` | ✓ | — | ✓ |
| Magistral Small (24B, reasoning) | `magistral:24b-q4_k_m` | ✓ | — | ✓ |
| DeepSeek-R1-Distill-Llama-70B | `OPEA/...-70B-int4-gptq-sym-inc` | — | ✓ 2×4090 layer-split | chat/reasoning grade⁴ |
| Devstral Small 2 (24B) | — | **not supported** (`mistral3` arch: YaRN-from-GGUF + attention temperature scaling; the loader refuses it loudly) | | |

¹ jart25 GPTQ chat emits empty answers on CUDA (open issue; weights fine —
Metal GGUF and CUDA random-context benches are clean).
² 32B-dense on a 32 GB Mac re-reads evicted weights from SSD every token
(~0.14 tok/s) — no practical deployment; use the CUDA lane.
³ strict `json_schema` intermittently returns 500 on 32B-GPTQ (open issue);
required tool-calls are 10/10.
⁴ R1-distill templates force `<think>` and the 70B writes tool-call JSON
inside the think block — reliable for chat/reasoning, not tool calling.
DeepSeek-R1-Distill-Qwen-14B shares the 32B's template/config byte-for-byte
and rides the same lanes.
⁵ Gemma 3 W2 evidence has final validator line
`MODEL_COVERAGE_W2 GOAL PASS: docs/goals/model-coverage-2026-06-12`.
CUDA L5 covers random `256/128` at c=1/4/16/32, 100 prompts × 3 repeats per
cell, zero errors, and usage-counted output tokens. On a 24 GB RTX 4090, the
c=32 client lane uses product CLI admission `--max-num-seqs 16` with
`--kv-capacity 400`. Same-card llama.cpp comparison is `0.500260x`, just above
the coverage floor but below the release-grade `0.8x` target, so this is
functional correctness/concurrency evidence with a known performance gap, not
release-grade support or an optimization-complete claim. 27B GGUF Metal was
waived on the degraded 32 GB Mac; Gemma 3 GGUF architecture coverage is pinned
by the 1B Q4_K_M Metal smoke artifact.

Use any HuggingFace model ID:

```bash
ferrum run Qwen/Qwen3-4B
ferrum run meta-llama/Llama-3.2-3B-Instruct
ferrum run JunHowie/Qwen3-4B-GPTQ-Int4    # INT4 auto-detected
```

### Multi-modal

```bash
# Speech-to-text (WAV/M4A/MP3/FLAC, auto ffmpeg conversion)
ferrum transcribe whisper-turbo recording.m4a -l zh
ferrum serve whisper-turbo

# Text-to-speech (basic synthesis; optional reference-audio cloning)
ferrum tts qwen3-tts "Hello, welcome to Ferrum TTS" -o output.wav
ferrum serve qwen3-tts

# Embeddings (text + image)
ferrum embed OFA-Sys/chinese-clip-vit-base-patch16 --text "sunset at the beach"
ferrum embed google/siglip-base-patch16-224 --image photo.jpg
```

## Build options

```bash
# CPU only (default)
cargo install ferrum-cli

# Metal acceleration (macOS)
cargo install ferrum-cli --features metal

# CUDA acceleration from source (NVIDIA, requires CUDA toolkit + nvcc)
cargo install ferrum-cli --features cuda,vllm-moe-marlin,vllm-paged-attn-v2,fa2-source
```

## Architecture

```
crates/
├── ferrum-types          # Shared types
├── ferrum-interfaces     # Trait contracts (Backend<B>, ModelExecutor, ...)
├── ferrum-runtime        # Backend registry
├── ferrum-engine         # Continuous-batch engine, Metal shader pipeline
├── ferrum-models         # Model architectures (LlamaFamilyModel<B>, MoE, ...)
├── ferrum-kernels        # Custom CUDA + Metal kernels, decode runner
├── ferrum-attention      # Fused-transformer prototype (Metal/CPU)
├── ferrum-quantization   # GPTQ loader, Marlin, native safetensors
├── ferrum-tokenizer      # Tokenization
├── ferrum-sampler        # Top-k/p, temperature, repetition penalty, JSON-mode
├── ferrum-scheduler      # Continuous batching, paged-KV scheduling
├── ferrum-kv             # Paged KV cache (CUDA + Metal pools)
├── ferrum-server         # HTTP API
├── ferrum-cli            # Binary entry point
└── ferrum-testkit        # Test infrastructure
```

Architecture v2 (Model-as-Code) means the model layer is an explicit Rust generic over a `Backend<B>` trait, not a config-driven runner. Adding a backend = implementing the trait, not editing models. See [docs/architecture-v2.md](docs/architecture-v2.md).

## Status

What works today:
- CLI chat, OpenAI-compatible HTTP server with streaming
- Continuous batching, PagedAttention (CUDA + Metal pools), prefix caching, preemption
- OpenAI-style function tool calling, including required tool calls and streaming `tool_calls` deltas
- Custom CUDA decode runner (Qwen3, LLaMA): 2× over Candle baseline
- Gemma 3 27B GPTQ functional on CUDA, including tool-call, strict-schema, streaming, multi-turn, and c=32 client pressure gates; release-grade performance is still a known gap
- Apple Silicon MoE inference (Qwen3-30B-A3B) — correctness, multi-turn, default serve startup, and multi-sequence serving gates covered by the Metal README gate
- INT4 GPTQ with Marlin fused kernel (Blackwell + Ampere); also Triton w4a16
- Tensor parallelism (multi-GPU NCCL, persistent per-rank threads)
- Speculative decoding (`--spec-draft <MODEL>` DeepMind accept/reject)
- Structured output (`json_object` best-effort plus strict `json_schema` validation for the supported schema subset)
- Whisper ASR (Metal-accelerated forward pass) + Qwen3-TTS
- Top-k / top-p / temperature / repetition penalty

Known regressions / in-progress:
- Apple Silicon dense at c = 4 underperforms c = 1 on small models (paged-batched is below crossover). Per-token mode remains the default for c ≤ 4 until the small-m path catches up.
- FP8 (Hopper / Blackwell) — INT4 path is at 24% peak DRAM bandwidth, so there's headroom before FP8 becomes the bottleneck.

## Roadmap

See [docs/ROADMAP.md](docs/ROADMAP.md) for the full picture.

Near-term:
- v0.1: CUDA + Apple Silicon production release with concurrent serving benchmarks
- v0.2: Broader release matrix and long-context serving benchmarks
- v0.3: Long-context tuning (32k+), more architectures and wider Gemma coverage

## License

MIT
