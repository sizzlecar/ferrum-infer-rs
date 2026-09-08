# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- Honored Chat Completions `reasoning_effort` and Responses `reasoning.effort`, preserving model defaults when omitted and respecting declared model capabilities.
- Allowed automatic tool calls alongside strict final JSON schemas in Chat Completions and Responses, including streaming, while giving schema-valid final answers precedence over ambiguous tool-call JSON.
- Recovered incomplete model downloads in `run` and `serve` by fetching missing weights, tokenizer, and chat-template files while reusing completed cache entries; `list` now identifies incomplete snapshots.

### Added

- Added one-command installation on Apple Silicon macOS, Linux x86_64, and Windows x64, with PATH setup and upgrades that keep existing processes running while new launches use the updated version.
- Added a native Windows EXE installer and portable CUDA package for a single NVIDIA sm89 GPU, bundling CUDA and VC runtimes and using the installed NVIDIA driver.

## [0.8.8] - 2026-09-07

### Fixed

- Preserved literal thinking tags in final text after a prompt-opened reasoning block closes.
- Preserved native Harmony tool-call identity when validating named tool choices, and rejected native tool calls when `tool_choice` is `none`.
- Selected the root checkpoint's indexed SafeTensors shards for fresh `run` and `serve` downloads, while retaining full-repository downloads for other consumers.
- Kept cold-download progress on stderr so `run --output-format jsonl` emits only JSONL on stdout.
- Loaded standalone Hugging Face chat templates in production Qwen packages, beyond downloading the template file.
- Accepted omitted GPTQ v1 format metadata and materialized supported symmetric GPTQ MoE routers as dense execution weights.

### Added

- Added explicit `HF_REPO@<40-hex-commit>` source selection for `run`, `serve`, and `pull`.
- Added CPU contract tests across HTTP, engine, and CLI boundaries, required real Metal/CUDA numerical checks, and selected real-model acceptance checks. These are validation improvements, not additional model-support claims.
- Automated candidate staging, evidence verification, and publication to crates.io, GitHub Releases, and Homebrew, with installation startup checks.

## [0.8.7] - 2026-09-05

### Fixed
- Included standalone Jinja chat templates in fresh Hugging Face downloads, allowing models such as GPT-OSS to load their required template.
- Kept active requests producing output while a later request processes a long prompt.
- Removed a redundant single-pass loop in runtime resource allocation.
- Preserved stop-sequence boundaries across streamed tokens, retained legal text before a stop, and flushed buffered text consistently through `run` and `serve`.
- Accepted `reasoning_content` in assistant history without losing it during multi-turn tool requests, while preserving model-template behavior and the existing `reasoning` response field.
- Preserved parsed GPT-OSS Harmony assistant history across `run` turns and applied JSON constraints within the final response channel.
- Parsed Gemma native thought channels in `run` and `serve`, including empty thought frames, tool-result continuations, and structured output.
- Accepted GPT-OSS text truncated by explicit user stops through `run` and synchronous or streaming chat requests. The engine distinguishes user stops from model EOS so malformed natural completions and incomplete tool calls remain errors.

## [0.8.6] - 2026-09-04

### Added
- Added caller-owned OpenAI Responses history with reasoning replay and namespace-aware tool loops.

### Fixed
- Preserved typed and whitespace-sensitive Qwen XML tool arguments.
- Retried model templates that reject interleaved system messages using ordered coalescing, with a typed opt-out.
- Restored concurrent CUDA greedy-decode throughput by keeping repetition penalties and row-wise token selection on device, and corrected paged-attention shared-memory sizing above 16K tokens.

## [0.8.5] - 2026-09-03

### Changed
- Optimized the vNext Metal causal-attention prefill and gated-delta execution paths used by Qwen3.5 long-context requests.
- Clarified model-derived context defaults and explicit context-limit behavior for OpenAI-compatible clients.

### Fixed
- Fixed Metal device-timing attribution when a submission contains commands without GPU counter samples.
- Added long-context admission coverage so an intentionally configured 4,096-token limit fails clearly while supported larger contexts are admitted.

## [0.8.0] - 2026-08-14

### Added
- Added the typed vNext execution path for the three release models across Apple Metal and NVIDIA CUDA, with shared `ferrum run` and OpenAI-compatible `ferrum serve` contracts.
- Added tokenizer-aware constrained decoding for `json_object` and strict `json_schema`, request replay/profile modes, and release-gated resource admission evidence.
- Added reproducible release-candidate correctness, performance, staged-asset, and publication gates.

### Changed
- Runtime memory, admission, scheduling, model resolution, and profiling policies are represented by typed CLI/config values and recorded in release artifacts.
- The v0.8.0 release ships CPU, Apple Silicon Metal, and Linux CUDA sm89 tarballs. It does not ship or promise an officially maintained Docker distribution.

### Fixed
- Fixed streaming text-byte sampling around incomplete UTF-8 fragments, request-local forbidden-token handling, and multiple scheduler/resource cleanup regressions.
- Fixed performance evidence portability while retaining binary, source, hardware, command, and raw-artifact identity checks.

### Known scope
- v0.8.0 is a language-only release. It does not claim vision or multimodal model support.
- Non-release model migration and exhaustive legacy physical removal remain post-release hardening work; unsupported paths continue to fail closed.

## [0.7.6] - 2026-06-05

### Added
- G0 release validation now requires an accelerator model matrix that covers both Qwen3-30B-A3B MoE/GPTQ and a Llama 8B-class dense model.
- Added a supplemental CUDA Llama dense release gate for `ferrum run`, `ferrum serve`, streaming usage, and `bench-serve` performance evidence.

### Fixed
- Fixed Qwen3 OpenAI-compatible API regressions for structured output, tool-call fallback behavior, streaming, and REPL input handling.
- Hardened release validators and binary gates to scan response bodies and logs for release-blocking patterns.
- Fixed runtime environment registry CI coverage so source gates account for documented runtime knobs.

### Changed
- Metal G0 validation is documented and enforced as both a correctness gate and README performance gate.
- Final G0 release summary now treats CUDA Qwen full and CUDA Llama dense evidence as required official release inputs.

## [0.1.0] - 2024-01-01

### Added
- Initial release
- Basic project structure
- Core functionality implementation
- Test framework setup
- CI/CD pipeline configuration
- Documentation and guides

---

## Release Types

- **Major version (X.y.z)**: Breaking changes that are not backward compatible
- **Minor version (x.Y.z)**: New features that are backward compatible
- **Patch version (x.y.Z)**: Bug fixes and minor improvements

## Changelog Categories

- **Added**: for new features
- **Changed**: for changes in existing functionality
- **Deprecated**: for soon-to-be removed features
- **Removed**: for now removed features
- **Fixed**: for any bug fixes
- **Security**: in case of vulnerabilities
