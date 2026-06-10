# Test Architecture Regression Repair Goal

## Status

Drafted on 2026-06-11 from
`docs/goals/test-architecture-2026-06-10/HANDOFF.md` at commit `ed4c8d87`.

This goal exists because the previous test-architecture work claimed broad
verification, but ad hoc product testing still found regressions. Completion is
not defined by reusing stale artifacts from the previous goal. Completion
requires fresh post-fix Metal and CUDA evidence and a final validator line:

```text
TEST_ARCH_REGRESSION_REPAIR PASS: <out_dir>
```

Final validator to land in phase 0:

```text
scripts/release/test_arch_regression_repair_gate.py
```

Until that validator exists and prints the exact PASS line above, this goal is
open.

## Source Handoff

The handoff says the previous agent shipped:

- `TEST_ARCH GOAL PASS` for the original test-architecture goal.
- CUDA Qwen3-30B-A3B GPTQ MoE fast-path fixes.
- `ferrum run` default `repeat_penalty = 1.1` to prevent greedy loops.
- Matrix perf floors and a substring-repeat degeneration detector.

The same handoff also admits these gaps:

- CUDA numbers were collected from a binary built with
  `cuda,vllm-moe-marlin`, missing `vllm-paged-attn-v2`.
- `FERRUM_USE_VLLM_PAGED_ATTN=1` errored on that binary because VPA was not
  compiled.
- Multi-turn decode slowed down sharply as context grew, and the data is not
  trustworthy until rerun on a full-feature CUDA build.
- Metal matrix LLM cells were not freshly rerun after the `repeat_penalty=1.1`
  fix and the new degeneration detector.
- The matrix runner's tier-aware length and detector were not exercised
  end-to-end on a fresh matrix.

These gaps are treated as blockers, not caveats.

## Non-Negotiable Rules

- Do not claim this repair complete unless the final validator prints
  `TEST_ARCH_REGRESSION_REPAIR PASS: <out_dir>`.
- Do not claim Metal or CUDA behavior from stale `docs/goals/test-architecture`
  artifacts. Every required artifact must be regenerated after the repair.
- Do not claim CUDA performance from a partial-feature binary. CUDA regression
  evidence must use:

```bash
cargo build --release -p ferrum-cli --bin ferrum \
  --features cuda,vllm-moe-marlin,vllm-paged-attn-v2,fa2-source
```

- Validate both product entrypoints whenever a bug touches user-visible
  generation: `ferrum run` and `ferrum serve`.
- Default product behavior must be validated through typed config, CLI/config
  options, or documented presets. Hidden env combinations may be used only for
  diagnostic A/B, not as the passing product evidence.
- Correctness gates must pass before performance numbers are treated as
  evidence.
- Do not describe decode-over-context as "fixed" unless the same-hardware
  multi-turn evidence shows the regression is gone. If VPA only gives modest
  improvement, report it as such and leave vLLM-parity work to the M3 goal.

## Target Base

Before coding, make the target base explicit in the artifact notes:

- local branch name and `git rev-parse HEAD`;
- whether `HEAD` contains `origin/main` commit `0db64121`;
- whether `HEAD` contains `ed4c8d87` and the handoff file;
- `git status --short`, with unrelated user changes listed and not reverted.

If the current worktree is behind `origin/main`, repair work should start from a
branch based on `origin/main` or must explicitly cherry-pick the missing
handoff/fix commits. Running gates on the older local `main` is not acceptable
evidence for the shipped regression state.

## Known Repair Scope

### R1. Reproduce the User-Visible Failures

Create `docs/goals/test-architecture-regression-repair-2026-06-11/evidence/repro/`.
For each issue found by quick manual testing, save:

- command line;
- model id/path;
- backend and compiled features;
- `ferrum --version` or binary SHA256;
- sanitized env and effective runtime config;
- stdout/stderr/logs;
- reason the output is wrong.

At minimum, reproduction must cover:

- Metal `ferrum run` on at least one Llama-family or Qwen LLM cell from the
  README matrix;
- Metal `ferrum serve` smoke on the same model family;
- CUDA `ferrum run` on `Qwen/Qwen3-30B-A3B-GPTQ-Int4`;
- CUDA `ferrum serve` on the same MoE/GPTQ model;
- a multi-turn CUDA decode-over-context probe with at least 6 turns and about
  120 generated tokens per turn.

### R2. Fix the Regressions

Every code change must map to a reproduced failure or a hard handoff gap.
Mandatory repair checks:

- `ferrum run` default repeat penalty is `1.1`, with a CLI default test and a
  fresh matrix run that the degeneration detector accepts.
- CUDA auto-config materializes effective runtime entries for MoE fast path and
  VPA when the model, backend, and compiled features support them.
- `ferrum run` and `ferrum serve` apply the same resolved startup auto-config
  for the relevant runtime knobs.
- Full-feature CUDA binary does not emit "vLLM paged attention is not compiled"
  when VPA is selected.
- VPA on/off diagnostic A/B is recorded, but the passing product path is the
  default resolved config, not a private env-only mode.

### R3. Refresh Metal Regression Evidence

Run on the local Apple Silicon host after rebuilding the binary:

```bash
export CARGO_TARGET_DIR=/Users/chejinxuan/rust_ws/ferrum-infer-rs/target
export CARGO_INCREMENTAL=0
cargo build --release -p ferrum-cli --bin ferrum --features metal
bash scripts/release/lane_l1_metal.sh \
  docs/goals/test-architecture-regression-repair-2026-06-11/evidence/l1-metal
python3 scripts/release/readme_model_matrix.py --run \
  docs/goals/test-architecture-regression-repair-2026-06-11/evidence/metal-matrix \
  --platform metal \
  --ferrum-bin target/release/ferrum
```

Required Metal evidence:

- `TEST_ARCH L1_METAL PASS: <out_dir>`;
- README matrix Metal LLM cells freshly generated after the repair;
- no degeneration detector failures;
- both `run` and `serve` logs for LLM cells;
- git SHA, dirty status, binary SHA256, compiled features, and sanitized env.

### R4. Refresh CUDA Regression Evidence

CUDA uses a paid Vast RTX 4090 lane.

Paid GPU contract before starting:

- Lane: `test-architecture-regression-repair-cuda`.
- Hardware: one RTX 4090 unless the user explicitly approves broader hardware.
- Expected runtime/cost: warm pod 30-90 minutes, cold build/model-cache path up
  to 3 hours; at the handoff's stated rate of about `$0.46/hr`, expected cost is
  about `$0.25-$1.50`.
- Stop condition: stop after first unrecoverable correctness failure, after the
  final CUDA PASS artifacts are copied back, or after 3 hours without progress.
- Correctness gate before perf evidence: CUDA op parity, `ferrum run` coherent
  multi-turn, `ferrum serve` smoke with streaming usage, no panic/log blocker.
- Performance command: the multi-turn decode-over-context probe plus
  `ferrum bench-serve --fail-on-error --require-ci --seed 9271 --n-repeats 3`
  only after correctness passes.

Remote build command:

```bash
export CARGO_TARGET_DIR=$PWD/target
export CARGO_INCREMENTAL=0
cargo build --release -p ferrum-cli --bin ferrum \
  --features cuda,vllm-moe-marlin,vllm-paged-attn-v2,fa2-source
```

Required CUDA commands/artifacts:

```bash
bash scripts/release/lane_l1_cuda.sh \
  docs/goals/test-architecture-regression-repair-2026-06-11/evidence/l1-cuda
L1_CUDA_MODEL=Qwen/Qwen3-30B-A3B-GPTQ-Int4 bash scripts/release/lane_l1_cuda.sh \
  docs/goals/test-architecture-regression-repair-2026-06-11/evidence/l1-cuda-qwen3-moe
python3 scripts/release/readme_model_matrix.py --run \
  docs/goals/test-architecture-regression-repair-2026-06-11/evidence/cuda-matrix \
  --platform cuda \
  --ferrum-bin target/release/ferrum
```

The existing `lane_l1_cuda.sh` has placeholder verify-live notes. For this goal,
placeholder PASS is not sufficient. The artifact must include real hb-09/hb-11
verify-live logs or the lane must be tightened before it can satisfy this goal.

Required CUDA evidence:

- `TEST_ARCH L1_CUDA PASS: <out_dir>` from a tightened or supplemented lane;
- full-feature binary SHA256 and compiled feature list;
- `nvidia-smi`, `nvcc --version`, driver/CUDA versions, GPU name and memory;
- `ferrum run` and `ferrum serve` Qwen3 MoE/GPTQ logs;
- effective config showing selected MoE fast path and VPA status;
- 6+ turn decode-over-context table with per-turn input tokens, output tokens,
  tok/s, selected attention implementation, and coherence verdict;
- VPA default vs VPA-disabled diagnostic A/B, clearly labeled;
- benchmark report only after correctness passes.

### R5. Final Aggregation

Create:

```text
docs/goals/test-architecture-regression-repair-2026-06-11/evidence/final/
```

It must contain:

- `repro.json` with all reproduced failures and fixed status;
- `source.json` with git SHA, dirty status, target base, and binary hashes;
- `metal.json` with L1-metal and Metal matrix artifact paths and PASS lines;
- `cuda.json` with L1-cuda, CUDA matrix, decode probe, and benchmark artifacts;
- `product_entrypoints.json` proving both `ferrum run` and `ferrum serve` were
  validated on Metal and CUDA;
- `known_caveats.md` for any remaining M3/vLLM-parity limitations that are not
  regressions fixed by this goal.

The final validator must reject:

- missing fresh Metal matrix logs;
- missing full-feature CUDA build evidence;
- CUDA VPA claims from env-only or partial-feature runs;
- any matrix cell with degeneration detector failure;
- L1-cuda artifacts where hb-09/hb-11 are only "operator note" placeholders;
- absent `ferrum run` or absent `ferrum serve` coverage on either backend;
- dirty worktree claims that do not list dirty files.

Final command:

```bash
python3 scripts/release/test_arch_regression_repair_gate.py \
  --validate docs/goals/test-architecture-regression-repair-2026-06-11/evidence/final
```

Required final line:

```text
TEST_ARCH_REGRESSION_REPAIR PASS: docs/goals/test-architecture-regression-repair-2026-06-11/evidence/final
```

## Completion Checklist

- Phase 0: final validator and schema self-test landed.
- Phase 1: user's quick-test failures reproduced and archived.
- Phase 2: fixes implemented with focused tests.
- Phase 3: local source gates pass:

```bash
cargo fmt --all -- --check
cargo check --workspace --all-targets
cargo test --workspace --all-targets
python3 scripts/release/test_arch_goal_gate.py --self-test
python3 scripts/release/test_arch_regression_repair_gate.py --self-test
python3 scripts/release/readme_model_matrix.py --self-test
python3 scripts/release/readme_model_matrix.py --check-readme
```

- Phase 4: fresh Metal L1 + matrix pass.
- Phase 5: fresh CUDA full-feature L1 + matrix + decode probe pass.
- Phase 6: final validator prints the required PASS line.
