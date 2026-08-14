# Ferrum v0.8.0 release notes

Ferrum v0.8.0 introduces the typed vNext language-model runtime for the release
matrix on Apple Silicon Metal and NVIDIA CUDA while preserving the `ferrum run`
and OpenAI-compatible `ferrum serve` product entrypoints.

## Highlights

- Typed model resolution, immutable execution plans, scheduler admission,
  runtime memory policy, and resource lifecycle accounting.
- Shared run/serve template, sampling, structured-output, tool-call, streaming,
  and multi-turn behavior for the three release models.
- Tokenizer-aware `json_object` and strict `json_schema` constrained decoding.
- User-visible profile detail modes with raw request/replay evidence and a
  profile-off performance contract.
- Reproducible correctness, performance, build, staged-binary, publication, and
  clean-install gates bound to source, binary, model, configuration, hardware,
  command, and raw artifact identities.

## Release assets

The release process stages exactly one build of each supported tarball and then
publishes the same bytes after validation:

- `ferrum-linux-x86_64.tar.gz` — CPU product binary.
- `ferrum-linux-x86_64-cuda-sm89.tar.gz` — NVIDIA CUDA sm89 product binary.
- `ferrum-macos-aarch64.tar.gz` — Apple Silicon Metal product binary.

Each tarball has an adjacent SHA256 plus version and dependency/ABI manifests.
After the release source is merged, a manually created RC tag drives the
production staging workflows. The GitHub prerelease is promoted only after the
approved exact staged-binary sampled regression, crates.io, Homebrew,
clean-install, and published-asset gates pass. The sampled regression does not
claim that every historical development cell was rerun on the staged bytes.

Ferrum v0.8.0 does **not** publish or promise an officially maintained Docker
distribution. No `0.8.0`, `stable`, `latest`, or candidate Docker image is part
of this release.

## Scope boundaries

v0.8.0 is a language-only release. It does not claim vision or multimodal model
support. Unsupported media-bearing requests and unknown model architectures
fail closed; this is not evidence of multimodal execution.

The release covers the matrix in [SUPPORT_MATRIX.md](SUPPORT_MATRIX.md). Other
historically advertised models and exhaustive legacy removal remain explicitly
tracked post-release hardening work, not an implied v0.8.0 support claim.

See [MIGRATION.md](MIGRATION.md) for behavioral changes and
[PERFORMANCE_REPORT.md](PERFORMANCE_REPORT.md) for measurement provenance and
the distinction between development R2 measurements and final staged R3 data.
