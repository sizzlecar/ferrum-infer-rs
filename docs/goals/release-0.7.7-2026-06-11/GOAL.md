# Ferrum 0.7.7 Release Goal

## Status

Drafted 2026-06-11 for the formal `v0.7.7` release.

This goal covers the full release path requested by the user:

1. finish the test/env regression repair on top of `origin/main`;
2. update README release evidence;
3. run full Metal and CUDA release regression;
4. manually trigger pre-release binary workflows and validate downloaded assets;
5. create the formal tag/release;
6. update and validate Homebrew formulae;
7. publish/update Cargo crates.

Completion requires the G0 release validator and release-completion validator to
print their exact PASS lines from fresh `v0.7.7` artifacts.

## Version

- Previous released version: `0.7.6`.
- Target release version: `0.7.7`.
- Target tag: `v0.7.7`.

## Non-Negotiable Gates

Release-ready claims are valid only after these PASS lines exist under
`docs/release/g0/0.7.7/`:

```text
FERRUM GATE unit PASS: docs/release/g0/0.7.7/unit
FERRUM GATE metal PASS: docs/release/g0/0.7.7/metal
FERRUM GATE cuda-full PASS: docs/release/g0/0.7.7/cuda-full
FERRUM GATE cuda-llama-dense PASS: docs/release/g0/0.7.7/cuda-llama-dense
METAL TARBALL GATE PASS: docs/release/g0/0.7.7/metal-tarball
CUDA TARBALL GATE PASS: docs/release/g0/0.7.7/cuda-tarball
HOMEBREW METAL GATE PASS: docs/release/g0/0.7.7/homebrew-metal
HOMEBREW CUDA FETCH GATE PASS: docs/release/g0/0.7.7/homebrew-cuda-fetch
G0 RELEASE PASS: docs/release/g0/0.7.7
FERRUM RELEASE COMPLETION PASS: docs/release/g0/0.7.7/release-complete
```

The CUDA release evidence must include both:

- `Qwen/Qwen3-30B-A3B-GPTQ-Int4` in the `cuda-full` lane;
- `hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4` in the
  `cuda-llama-dense` lane.

## Paid GPU Contract

- Lane: `g0-release-0.7.7-cuda`.
- Hardware: one RTX 4090 for G0 CUDA Qwen full and Llama dense lanes.
- Expected runtime/cost: warm pod 1-3 hours; cold pod/model-cache path up to 5
  hours. At about `$0.46/hr`, expected cost is about `$0.50-$2.50`.
- Stop condition: stop after first unrecoverable correctness failure, after CUDA
  artifacts are copied back, or after 5 hours without progress.
- Correctness before performance: CUDA source gates and product-path checks must
  pass before any CUDA performance evidence is treated as release evidence.
- Performance commands: only the configured G0 CUDA release lanes and
  `bench-serve --fail-on-error --require-ci --seed 9271 --n-repeats 3` from the
  lane configs count as release performance evidence.

## Work Plan

1. Commit the regression repair and version bump on a release branch.
2. Run source gates:

```bash
python3 scripts/release/run_gate.py unit --out docs/release/g0/0.7.7/unit
python3 scripts/release/run_gate.py metal --out docs/release/g0/0.7.7/metal
python3 scripts/release/run_gate.py cuda-full --out docs/release/g0/0.7.7/cuda-full
python3 scripts/release/run_gate.py cuda-llama-dense --out docs/release/g0/0.7.7/cuda-llama-dense
```

3. Update README/README_zh to point at `0.7.7` evidence.
4. Push the release branch and manually trigger binary workflow dry-runs:

```bash
gh workflow run release.yml --ref <release-branch> -f tag=v0.7.7 -f publish_release=false
gh workflow run release-cuda.yml --ref <release-branch> -f tag=v0.7.7 -f cuda_compute_cap=89 -f publish_release=false
```

5. Download workflow artifacts and validate:

```bash
python3 scripts/release/run_gate.py metal-tarball --version 0.7.7 \
  --asset-path <ferrum-macos-aarch64.tar.gz> --out docs/release/g0/0.7.7/metal-tarball
python3 scripts/release/run_gate.py cuda-tarball --version 0.7.7 \
  --asset-path <ferrum-linux-x86_64-cuda-sm89.tar.gz> --out docs/release/g0/0.7.7/cuda-tarball
```

6. Tag and push `v0.7.7`; verify release assets.
7. Update `sizzlecar/homebrew-ferrum` formulae to `0.7.7`, validate
   `homebrew-metal` and `homebrew-cuda-fetch`, and push the tap.
8. Publish Cargo crates in dependency order, then verify crates.io has
   `ferrum-cli 0.7.7`.
9. Run `release-summary` and `release-complete` final validators.

## Completion Evidence

Final artifacts live under:

```text
docs/release/g0/0.7.7/
```

The release is complete only after the exact final completion PASS line exists:

```text
FERRUM RELEASE COMPLETION PASS: docs/release/g0/0.7.7/release-complete
```
