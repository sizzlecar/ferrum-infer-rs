# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

- N/A

## [0.7.3] - 2026-06-01

### Added
- CUDA release binary workflow for Linux x86_64, including the native source FA2 path and an ldd guard against Torch/Python/vLLM runtime links.
- Homebrew `ferrum-cuda` tap generation for Linux x86_64 CUDA downloads, alongside the existing `ferrum` CPU/Metal package.
- Qwen3-8B and LLaMA-3.1-8B GGUF CUDA serve smoke coverage and saved Ferrum/vLLM GGUF benchmark packets.
- Release documentation for the user-adjusted M3 `0.75x vLLM` threshold.

### Changed
- Native source FA2 is release-supported as an opt-in CUDA path for the M3 release evidence packet.
- CUDA GGUF serving now uses the eager-dequant/fp16 dense fallback compatibility path.
- Release packaging now separates CPU/Metal and CUDA artifacts instead of requiring CUDA users to build from source.

### Fixed
- Qwen3-MoE Metal prefill now synchronizes logits before host readback, fixing the garbage-output and command-encoder assertion smoke failure.
- Qwen3 alias serving now carries typed model paths into tokenizer/model factories instead of relying on process `FERRUM_MODEL_PATH`.

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
