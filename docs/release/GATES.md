# Ferrum G0 release gates

G0 gates harden release validation for source, official release assets, and Homebrew packaging. Every gate must print an explicit `PASS` line and return non-zero on failure.

Before a release-gate script change is trusted, run the validator selftest:

```bash
python3 scripts/release/selftest_g0_validators.py
```

It uses synthetic artifacts to verify both the positive path and the failure path. It does not run models or GPUs.

## Source gates

```bash
scripts/release/g0_source_gate.sh unit docs/release/g0/<version>/source-unit
scripts/release/g0_source_gate.sh metal docs/release/g0/<version>/source-metal
scripts/release/g0_source_gate.sh cuda-smoke /workspace/g0/<version>/source-cuda-smoke
scripts/release/g0_source_gate.sh cuda-full /workspace/g0/<version>/source-cuda-full
scripts/release/g0_source_gate.sh cuda-llama-dense /workspace/g0/<version>/source-cuda-llama-dense
```

- `unit`: runs `cargo test --workspace --all-targets`, release-script Python compile checks, shell syntax checks, and `scripts/release/selftest_g0_validators.py`.
- `metal`: builds Metal, runs `scripts/metal_readme_regression.py`, then validates the artifact with `scripts/release/validate_metal_readme_regression.py`. The artifact must include default `serve` startup config evidence, benchmark-profile startup config evidence, and concurrent marker/square content-quality probes before throughput rows are accepted. Qwen3-30B-A3B Metal GGUF is correctness-safe single-sequence evidence until multi-sequence MoE decode passes those probes.
- `cuda-smoke`: builds CUDA features, runs `scripts/m3_ab_runner.py` through `scripts/release/configs/g0_cuda4090_smoke.json`, then runs `scripts/m3_validate_runner_artifact.py`.
- `cuda-full`: same as smoke, but covers c=1/4/16/32 with repeats >= 3 using `scripts/release/configs/g0_cuda4090_full.json`.
- `cuda-llama-dense`: supplemental CUDA release gate for a Llama 8B-class dense model. It runs `ferrum run`, `ferrum serve`, streaming usage checks, and `ferrum bench-serve --fail-on-error --require-ci --seed 9271 --n-repeats 3`.

## Release binary gates

```bash
python3 scripts/release/release_binary_gate.py metal-tarball --version <version> --out docs/release/g0/<version>/metal-tarball
python3 scripts/release/release_binary_gate.py cuda-tarball --version <version> --out docs/release/g0/<version>/cuda-tarball
python3 scripts/release/release_binary_gate.py homebrew-metal --version <version> --out docs/release/g0/<version>/homebrew-metal
python3 scripts/release/release_binary_gate.py homebrew-cuda-fetch --version <version> --out docs/release/g0/<version>/homebrew-cuda-fetch
```

The tarball gates download official GitHub release assets, verify checksums, run CLI multi-turn/math, run serve math/multi-turn/boundary checks, run serve strict JSON/tool-call/streaming `[DONE]` checks, and scan server logs for release-blocking patterns. CUDA tarball also checks `ldd`.

## Summary gate

```bash
python3 scripts/release/g0_release_summary.py docs/release/g0/<version>
```

The summary gate verifies required child `gate.json` files and prints:

```text
G0 RELEASE PASS: docs/release/g0/<version>
```

Required release children are `unit`, `metal-source`, `cuda-qwen-full`, `cuda-llama-dense`, `metal-tarball`, `cuda-tarball`, `homebrew-metal`, and `homebrew-cuda-fetch`. CUDA smoke may be attached as additional diagnostic evidence.

## Canonical performance client

`ferrum bench-serve` is the only canonical release HTTP performance client for `/v1/chat/completions`. Release performance invocations must use:

```bash
ferrum bench-serve ... --fail-on-error --require-ci --seed 9271
```

Smoke invocations may omit `--require-ci`, but must still use `--fail-on-error`.
