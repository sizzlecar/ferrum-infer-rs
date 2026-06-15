# W2 dense Marlin native CUDA probe summary

- Artifact: `docs/goals/model-coverage-2026-06-12/artifacts/w2_dense_marlin_native_probe_2026-06-15/`
- Instance: Vast `40826362`, 1x RTX 4090, cache-retained.
- Scope: native CUDA kernel-ceiling diagnostic only. This did not run `ferrum run`, `ferrum serve`, or a release-grade validator.
- Result: both native probe runs returned `0` and printed `VERDICT: dense Marlin native CUDA probe complete`.
- Shutdown: API poll verified `cur_state=stopped`, `actual_status=exited` at `2026-06-15T06:52:02Z`.

Key m=16 auto-tile timings:

| shape | hot event kernel us | host-sync kernel us | cold-cache kernel us |
|---|---:|---:|---:|
| qkv | 17.207 | 18.887 | 39.929 |
| o_proj | 12.058 | 13.695 | 24.447 |
| gate_up | 133.650 | 135.924 | 177.144 |
| down | 30.395 | 32.049 | 93.558 |

Interpretation:

- Tile override evidence is weak: `64x256` only marginally improved `gate_up` in the hot-cache probe and regressed `down`, while `128x128` did not materially beat auto for m=16.
- Product profile for smaller projections sits between repeated-hot and forced-cold native timings, so the initial hot-cache probe was too optimistic for `qkv`, `o_proj`, and `down`.
- `gate_up` remains the largest dense Marlin target, but this checkpoint does not support a safe default tile change.
- No new product correctness issue was found. W2 is still not release-grade because no `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>` line exists.
