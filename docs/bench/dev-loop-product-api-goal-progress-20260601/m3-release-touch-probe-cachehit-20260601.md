# M3 release touch probe cache-hit timing - 2026-06-01

## Artifact

- Host: Vast RTX 4090 instance `38872161`
- Checkout: `/workspace/ferrum-fa2-native-restore-git-ac3dfab`
- Git head: `d93790393c04232e265abecfb32fdd81b3547e01`
- Artifact root: `/workspace/m3-release-touch-probe-cachehit-20260601-20260601_043825`
- Manifest: `/workspace/m3-release-touch-probe-cachehit-20260601-20260601_043825/build_boundary_manifest.json`

## Result

The probe ran 5 release touch iterations after the cold CUDA/static-library
prewarm. Every iteration exited successfully and every CUDA build summary row
was a cache hit, but the Milestone A timing target still failed.

| Run | elapsed_sec | exit_code | CUDA status_counts |
|---:|---:|---:|---|
| 1 | 231.517 | 0 | `cache_hit=39` |
| 2 | 229.133 | 0 | `cache_hit=39` |
| 3 | 230.500 | 0 | `cache_hit=39` |
| 4 | 234.102 | 0 | `cache_hit=39` |
| 5 | 234.608 | 0 | `cache_hit=39` |

Timing summary from the manifest:

- `p50_sec_nearest_rank=231.517`
- `p95_sec_nearest_rank=234.608`
- `p50_limit_sec=75.0`
- `p95_limit_sec=90.0`
- `limits_pass=false`

## Interpretation

This supersedes the earlier "not run yet" Milestone A blocker. The 5-run
CUDA-hosted artifact now exists and proves that narrow attention-kernel touches
do not rebuild CUDA artifacts in the cache-hit state. The remaining blocker is
the Rust/Cargo release dirtying and downstream release/link tail, not broad
nvcc rebuild.

Milestone A remains incomplete until the release touch probe is brought under
`p50 <= 75s` and `p95 <= 90s`, then rerun with the same manifest validator.
