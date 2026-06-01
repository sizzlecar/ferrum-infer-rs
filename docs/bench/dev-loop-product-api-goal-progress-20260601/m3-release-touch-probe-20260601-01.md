# m3_cuda_build_boundary_probe run attempt (2026-06-01)

## Command

```bash
python3 scripts/m3_cuda_build_boundary_probe.py --iterations 5 --out /tmp/m3-release-touch-probe-20260601-01 --fail-on-limit --no-cargo-verbose
```

## Result

- Exit: non-zero (failed on run 1)
- Failure reason: CUDA toolchain not available in this environment.

## Evidence

- Log path: `docs/bench/dev-loop-product-api-goal-progress-20260601/m3-release-touch-probe-20260601-01-run1-build.log`

## Key error snippets

- `
  failed to run custom build command for `cudarc v0.19.4`
  ...
  `nvcc --version` failed.
  Err(Os { code: 2, kind: NotFound, message: "No such file or directory" })
  `
- `
  nvidia-smi failed while detecting CUDA compute capability: Os { code: 2, kind: NotFound, message: "No such file or directory" }
  `

## What to run next

This probe is fully gated by GPU tooling. Re-run this exact command on a restored RTX-4090 pod with CUDA + nvidia-smi available to produce:
- `build_boundary_manifest.json`
- 5-run p50/p95 timing values vs limits
- validated pass/fail against `--fail-on-limit`
