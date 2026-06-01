# Ferrum 0.7.3 Pre-Tag Checklist - 2026-06-01

Scope: final operator checklist before creating or moving a `v0.7.3` release
tag.

Hard rule:

- Do not create, move, or push `v0.7.3` until the user explicitly says to do
  the tag/release step.

Current evidence checkpoint before this checklist:

- `2ecd2bb docs: add metal qwen3 0.6b server smoke`
- GPU checkout `/workspace/ferrum-release-bench` has been fast-forwarded to
  `2ecd2bb` through the git bare remote at
  `/workspace/ferrum-git-remotes/ferrum-infer-rs.git`.

Runtime-affecting checkpoint:

- `1e3ce42 fix: sync qwen3 moe prefill logits before readback`

Release package checkpoints after the runtime checkpoint:

- `cd1702c ci: add cuda release binary workflow`
- `0f4b74c release: add cuda homebrew packaging`
- `b88fc13 ci: guard release packaging metadata`
- `0fecec8 release: pin cuda brew asset to sm89`
- `2ecd2bb docs: add metal qwen3 0.6b server smoke`

Must be true before tagging:

- The worktree intended for tagging is clean.
- The tag target is the final release checkpoint selected by the user.
- Existing local/remote `v0.7.3` tag state is inspected before any tag action;
  do not overwrite an existing tag without explicit user approval.
- `release.yml` is expected to publish:
  - `ferrum-linux-x86_64.tar.gz`
  - `ferrum-linux-x86_64.tar.gz.sha256`
  - `ferrum-macos-aarch64.tar.gz`
  - `ferrum-macos-aarch64.tar.gz.sha256`
- `release-cuda.yml` is expected to publish:
  - `ferrum-linux-x86_64-cuda-sm89.tar.gz`
  - `ferrum-linux-x86_64-cuda-sm89.tar.gz.sha256`
  - `ferrum-linux-x86_64-cuda-sm89.ldd.txt`
- The CUDA binary must not link Torch, Python, or vLLM.
- `scripts/release.sh` must wait for both `release.yml` and
  `release-cuda.yml` before updating Homebrew.
- Homebrew tap update must write both formulas:
  - `Formula/ferrum.rb`
  - `Formula/ferrum-cuda.rb`
- `brew install ferrum` remains the normal package:
  - macOS Apple Silicon: Metal
  - Linux x86_64: CPU
- `brew install ferrum-cuda` is the Linux x86_64 CUDA package and downloads
  `ferrum-linux-x86_64-cuda-sm89.tar.gz`.
- Release notes must state that the CUDA Homebrew package does not bundle
  NVIDIA driver, CUDA runtime, or NCCL runtime libraries.

Evidence packets ready for release notes:

- M3 release threshold:
  `docs/bench/dev-loop-product-api-goal-progress-20260601/m3-fa2-source-current-allcells-n3-20260601.md`
- Real-model OpenAI API smoke:
  `docs/bench/dev-loop-product-api-goal-progress-20260601/m3-real-model-api-direct-smoke-20260601.md`
- Qwen3-8B and LLaMA-3.1-8B GGUF-vs-GGUF benchmark:
  `docs/bench/dev-loop-product-api-goal-progress-20260601/gguf-8b-release-benchmarks-20260601.md`
- Metal Qwen3-MoE prefill sync fix:
  `docs/bench/dev-loop-product-api-goal-progress-20260601/metal-qwen3-30b-a3b-prefill-syncfix-20260601.md`
- Metal Qwen3-8B GGUF smoke:
  `docs/bench/dev-loop-product-api-goal-progress-20260601/metal-qwen3-8b-q4km-smoke-20260601/summary.md`
- Metal Qwen3-0.6B CLI/API/concurrency/multi-turn smoke:
  `docs/bench/dev-loop-product-api-goal-progress-20260601/metal-qwen3-06b-smoke-20260601/summary.md`
- Post-`1e3ce42` GPU quick regression:
  `docs/bench/dev-loop-product-api-goal-progress-20260601/m3-quick-regress-1e3ce42-c32-20260601.md`

Release-note constraints:

- It is valid to claim the user-adjusted formal release threshold:
  Ferrum source FA2 reaches at least `0.75x vLLM` for Qwen3-30B-A3B GPTQ Int4
  on RTX 4090.
- Do not claim the previous `0.80x vLLM` stretch goal is complete.
- Source FA2 is release-supported as an opt-in path unless the user explicitly
  asks for a default selector change before tagging.
- 8B tables must be labelled `GGUF-vs-GGUF`.
- Ferrum CUDA GGUF is an eager-dequant/fp16 dense compatibility fallback, not
  a native CUDA k-quant performance path.
- vLLM GGUF should be described as experimental/under-optimized.
- Metal numbers collected on the local Mac are smoke/regression evidence only
  because swap was active.

Final tag/release sequence after explicit user approval:

1. Inspect local and remote `v0.7.3` tag state.
2. Confirm final release checkpoint.
3. Create or move the tag only if explicitly approved for that exact target.
4. Push the tag.
5. Wait for `release.yml` and `release-cuda.yml`.
6. Confirm all CPU, Metal, and CUDA assets plus sha256 files are attached.
7. Run the release script tap update stage or equivalent controlled tap update.
8. Verify `brew install ferrum` and `brew install ferrum-cuda` resolve the
   expected packages.
