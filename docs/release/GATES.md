# Ferrum G0 release gates

G0 gates harden release validation for source, official release assets, and Homebrew packaging. Every gate must print an explicit `PASS` line and return non-zero on failure.

Before a release-gate script change is trusted, run the validator selftest:

```bash
python3 scripts/release/selftest_g0_validators.py
```

It uses synthetic artifacts to verify both the positive path and the failure path. It does not run models or GPUs.

## Source gates

```bash
scripts/release/g0_source_gate.sh unit docs/release/g0/<version>/source
scripts/release/g0_source_gate.sh metal docs/release/g0/<version>/source
scripts/release/g0_source_gate.sh cuda-smoke /workspace/g0/<version>/source-cuda-smoke
scripts/release/g0_source_gate.sh cuda-full /workspace/g0/<version>/source-cuda-full
```

- `unit`: runs `cargo test --workspace --all-targets`.
- `metal`: builds Metal, runs `scripts/metal_readme_regression.py`, then validates the artifact with `scripts/release/validate_metal_readme_regression.py`.
- `cuda-smoke`: builds CUDA features, runs `scripts/m3_ab_runner.py` through `scripts/release/configs/g0_cuda4090_smoke.json`, then runs `scripts/m3_validate_runner_artifact.py`.
- `cuda-full`: same as smoke, but covers c=1/4/16/32 with repeats >= 3 using `scripts/release/configs/g0_cuda4090_full.json`.

## Release binary gates

```bash
python3 scripts/release/release_binary_gate.py metal-tarball --version <version> --out docs/release/g0/<version>/metal-tarball
python3 scripts/release/release_binary_gate.py cuda-tarball --version <version> --out docs/release/g0/<version>/cuda-tarball
python3 scripts/release/release_binary_gate.py homebrew-metal --version <version> --out docs/release/g0/<version>/homebrew-metal
python3 scripts/release/release_binary_gate.py homebrew-cuda-fetch --version <version> --out docs/release/g0/<version>/homebrew-cuda-fetch
```

The tarball gates download official GitHub release assets, verify checksums, run CLI multi-turn/math, run serve math/multi-turn/boundary checks, and scan server logs for release-blocking patterns. CUDA tarball also checks `ldd`, strict JSON, tool call, and streaming `[DONE]` behavior.

## Summary gate

```bash
python3 scripts/release/g0_release_summary.py docs/release/g0/<version>
```

The summary gate verifies required child `gate.json` files and prints:

```text
G0 RELEASE PASS: docs/release/g0/<version>
```

## Canonical performance client

`ferrum bench-serve` is the only canonical release HTTP performance client for `/v1/chat/completions`. Release performance invocations must use:

```bash
ferrum bench-serve ... --fail-on-error --require-ci --seed 9271
```

Smoke invocations may omit `--require-ci`, but must still use `--fail-on-error`.
