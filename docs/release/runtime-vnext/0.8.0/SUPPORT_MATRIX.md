# Ferrum v0.8.0 support matrix

This file defines the intended release scope. A row becomes a published support
claim only after the full R2 development baseline remains source-closed to the
merged release commit and the approved final sample validates the exact staged
and published binary for that backend. The sample is explicitly not represented
as a complete staged rerun of every R2 cell.

| Lane | Metal (macOS aarch64) | CUDA (Linux x86_64 sm89) | Weight contract | Required product coverage |
|---|---|---|---|---|
| M1 `Qwen/Qwen3.5-4B` | pinned Q4_K_M GGUF | pinned safetensors | exact revision and file SHA lock | run, serve, stream, stateful, tools, schema, concurrency, performance |
| M2 `Qwen/Qwen3.5-35B-A3B` | pinned Q4_K_S GGUF | pinned GPTQ Int4 | exact revision and file SHA lock | run, serve, stream, stateful, tools, schema, concurrency, performance |
| M3 `Qwen/Qwen3-30B-A3B` | pinned Q4_K_M GGUF | pinned GPTQ Int4 | exact revision and file SHA lock | run, serve, stream, stateful, tools, schema, concurrency, performance |
| Dense control `meta-llama/Llama-3.1-8B-Instruct` compatible weights | pinned Q4_K_M GGUF | pinned GPTQ Int4 | exact release-gate lock | run, serve, stream usage, correctness, performance |

## Post-v0.8 source-qualified adoption

This row is intentionally separate from the v0.8 release matrix and does not
qualify a published tarball, Homebrew artifact, Metal path, or other
compressed-tensors checkpoints.

| Goal | Qualified backend | Exact weight contract | Measured product boundary |
|---|---|---|---|
| `cyankiwi/Qwen3.8-27B-AWQ-INT4@63768c10df38c0395e12ef49edac1bd539eaeeea` | CUDA text-only; exactly one sm89 RTX 4090 | `compressed-tensors` pack-quantized W4, group32, asymmetric, no activation quantization; mixed dense projections; full-F16 fallback forbidden | `run` known-answer/two-turn; `serve` non-stream/stream/required-tool/strict-schema; c=1 × 3 requests with usage token counts, ≥5 tok/s median output throughput and ≤30s p50 TTFT |

The exact completion authority is the read-only
`scripts/release/model_adoption_goal_gate.py` PASS artifact defined by the
[Qwen3.8 adoption goal](../../../goals/model-adoption-2026-08-26/GOAL.md).

## Platform assets

| Target | Asset | Contract |
|---|---|---|
| macOS aarch64 | Metal tarball | Apple Silicon; exact staged binary; release Metal source/tarball and sampled three-model gates |
| Linux x86_64 | CUDA sm89 tarball | one RTX 4090 release lane; compatible NVIDIA driver, CUDA runtime, and NCCL runtime required; no Python/Torch/vLLM runtime linkage |
| Linux x86_64 | CPU tarball | CLI and CPU product path; not evidence for accelerator performance |
| crates.io | workspace crates at 0.8.0 | dependency-topological publication and clean `cargo install ferrum-cli --version 0.8.0 --locked` |
| Homebrew | Metal formula and CUDA fetch path | Metal install is executable validation; CUDA fetch validates URL/checksum while runtime validation remains the CUDA tarball lane |

## API scope

The supported product entries are `ferrum run` and `ferrum serve`, including
OpenAI-compatible chat completions, streaming usage, tools, strict structured
output, deterministic diagnostic generation, and typed concurrency/admission
behavior covered by the release matrix.

v0.8.0 does not claim vision, video, or multimodal model execution; distributed
or multi-node serving; full vLLM CLI/internal API compatibility; every legacy
model previously mentioned in historical documentation; or an official Docker
distribution. Unknown or unsupported paths must fail closed.

## Post-release backlog

The following remain v0.8.1/0.9 hardening work: non-release-model migration,
exhaustive provider and historical mutation coverage, full support disposition
for historical long-tail models, physical removal of all non-release legacy
source paths, extended build-matrix sampling, and automated optimization
dashboards. These omissions do not relax any M1/M2/M3 or Llama release gate.
