# Ferrum v0.8.0 migration guide

Ferrum v0.8.0 moves the release model path to the typed vNext runtime. The
public `ferrum run` and `ferrum serve` entrypoints remain the supported product
interfaces; clients should not depend on internal executor, scheduler, or cache
implementation details.

## Structured output

`response_format: {"type":"json_object"}` and strict `json_schema` now use
tokenizer-aware constrained decoding. In 0.7.x these modes could rely on prompt
steering or repair. In 0.8.0:

- `json_object` must produce exactly one JSON object, without Markdown fences or
  surrounding prose.
- strict `json_schema` rejects unsupported grammar before request admission and
  never silently falls back to ordinary sampling.
- streaming hard-structured output is held until validation succeeds. A failed
  generation returns an OpenAI-shaped error and exactly one terminal `[DONE]`.
- non-strict `json_schema` remains best effort for compatibility.

Applications that stripped fences or extracted the first brace-delimited region
must instead handle an explicit request or generation error.

## Runtime configuration

Release behavior is selected by typed CLI/config values. Do not use undocumented
environment-variable combinations to select a scheduler, cache, memory policy,
model implementation, or profiling mode. The stable public controls include
`--max-model-len`, `--max-num-seqs`, `--max-num-batched-tokens`,
`--gpu-memory-utilization`, prefix-cache controls, and the documented profile
detail options. Effective configuration is available in release evidence and
diagnostic output.

Temporary resource pressure is handled by typed admission/defer/resume behavior
before kernel submission. An allocator or kernel OOM is an error, not capacity
discovery. Operators should set documented admission and memory limits rather
than relying on oversubscription.

## Model and platform scope

The v0.8.0 release matrix targets the three language-model lanes listed in the
[support matrix](SUPPORT_MATRIX.md), plus a Llama 3.1 8B-class dense accelerator
control. GGUF is the production Metal weight path; pinned GPTQ/safetensors are
the production CUDA paths for the release models.

Unknown architectures and unsupported weight layouts fail closed before model
execution. Vision, video, audio input, and other multimodal model execution are
not part of the v0.8.0 release claim. Text content arrays remain covered by the
OpenAI-compatible API contract.

## Distribution

The supported v0.8.0 binary assets are Linux x86_64 CPU, Linux x86_64 CUDA sm89,
and macOS aarch64 Metal tarballs with adjacent checksums and dependency/ABI
manifests. There is no official or maintained Ferrum v0.8.0 Docker distribution.
Do not treat an older container image or a locally built image as a v0.8.0
release asset.

## Post-release hardening scope

The release does not claim completion of the historical exhaustive G00-G10
roadmap. Migration of non-release models, exhaustive provider conformance,
physical removal of every non-release legacy source path, and the broader
historical mutation platform remain v0.8.1/0.9 hardening work. No new legacy
production path may be added meanwhile.
