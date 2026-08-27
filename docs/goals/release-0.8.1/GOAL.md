# Ferrum v0.8.1 focused release goal

## Completion authority

This goal prepares and publishes Ferrum v0.8.1. It deliberately reuses the
active G0 gates instead of reopening the historical v0.8.0 runtime-vNext audit.

Completion requires both exact lines from the final artifact root:

```text
G0 RELEASE PASS: <out_dir>
FERRUM V0.8.1 RELEASE GOAL PASS: <out_dir>
```

Neither a clean build, an RC tag, an Actions artifact, nor a prerelease alone
completes this goal.

## Fixed scope

- Version: `0.8.1`.
- Release candidate tag: annotated `v0.8.1-rc.N`.
- Final tag: annotated `v0.8.1`, peeling to the same commit as the accepted RC.
- Assets: macOS aarch64 Metal, Linux x86_64 CPU, and Linux x86_64 CUDA sm89.
- New support claim: the exact Qwen3.8 CUDA checkpoint already qualified by the
  [model-adoption goal](../model-adoption-2026-08-26/GOAL.md), shown in the
  existing README model table.
- Distribution: GitHub Release, all publishable workspace crates, Homebrew
  Metal install, and Homebrew CUDA fetch.
- No Docker, Qwen3.8 Metal, multimodal, generalized compressed-tensors,
  multi-GPU, or new API surface.

## Minimal gates

### R1 — clean release candidate

- Workspace and internal dependency versions are exactly `0.8.1` and
  `Cargo.lock` is current.
- README and Chinese README add Qwen3.8 CUDA only to the existing model table.
- `FERRUM GATE unit PASS` succeeds on the clean candidate SHA.
- Metal correctness/performance covers Qwen3 30B-A3B and a Llama 8B-class dense
  model through both `run` and `serve`.
- CUDA correctness/performance passes `cuda-full` for Qwen3 30B-A3B and
  `cuda-llama-dense` on exactly one RTX 4090.

### R2 — build once, validate staged bytes

- Create an annotated RC tag only after R1 source preparation is frozen.
- Dispatch `.github/workflows/release.yml` and `release-cuda.yml` with the exact
  RC SHA/tag and `publish_release=false`.
- Save workflow run ids, commands, manifests, tarball SHA256, internal binary
  SHA256, target/runtime dependency evidence, and the clean source identity.
- Run the Metal and CUDA tarball gates against the downloaded Actions artifacts.
- Use the exact staged Metal binary for a bounded Qwen3 MoE + Llama dense
  `run`/`serve`/performance sample.
- Use the exact staged CUDA binary for the required Qwen3 MoE + Llama dense
  release lanes and a bounded Qwen3.8 `run`/`serve` plus c=1 usability sample.
- Any correctness failure stops publication. A single failed case is rerun only
  through its focused reproducer before a broader gate is reconsidered.

### R3 — publish immutable bytes

- Create annotated `v0.8.1` at the accepted RC commit and publish a GitHub
  prerelease using the exact staged tarballs and adjacent manifests; do not
  rebuild after validation.
- Verify published asset SHA256 values equal the staged values.
- Package all workspace crates, run locked dry-runs in dependency order, then
  publish serially. Poll crates.io visibility and clean dependency resolution
  before publishing the next dependent crate.
- From a clean directory, install `ferrum-cli 0.8.1` and verify version/help.
- Update Homebrew formulae with the exact published asset URLs and SHA256; run
  the Metal install gate and CUDA fetch gate.
- Only after all gates pass, promote the GitHub prerelease without changing
  tags or asset bytes, then run the final summary and completion validator.

## Time and cost controls

- Local source/unit work: expected 10–45 minutes per full milestone, hard
  deadline 60 minutes, progress from bounded logs and receipts.
- Metal staged sample: expected 20–60 minutes, hard deadline 90 minutes, progress
  from per-model artifacts.
- Paid CUDA work must reuse a retained one-RTX-4090 instance when viable. Before
  starting it, record inventory, rate, expected runtime/cost, exact correctness
  and performance commands, progress signal, and stop condition.
- Keep at most one potentially billable instance. Copy artifacts back and stop
  it after PASS, focused failure triage, or the declared deadline.
- Do not add c=4/16/32 Qwen3.8 performance, external-engine comparisons, or
  another model to the release denominator.
