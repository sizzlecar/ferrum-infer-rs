# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

No unreleased changes.

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
