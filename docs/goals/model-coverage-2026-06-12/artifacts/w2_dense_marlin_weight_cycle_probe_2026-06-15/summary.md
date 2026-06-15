# W2 Dense Marlin Weight-Cycle Native CUDA Probe

Date: 2026-06-15

Artifact:
`docs/goals/model-coverage-2026-06-12/artifacts/w2_dense_marlin_weight_cycle_probe_2026-06-15/`

Lane:
W2 Gemma3 CUDA dense Marlin weight-cycle native probe.

Hardware:
Vast instance `40826362`, 1x RTX 4090. `vast_shutdown/cleanup_check.txt` confirms
`cur_state=stopped actual_status=exited` after artifact copyback.

Command:

```bash
timeout 1800 bash scripts/microbenches/build_and_run_dense_marlin_gemma3_perf.sh
```

Validation:

- remote HEAD: `82fb3272451083bc7f79c7aeca4610793ef579aa`
- `probe/dense_marlin_gemma3_perf.rc`: `0`
- stdout verdict: `VERDICT: dense Marlin native CUDA probe complete`
- summary JSON: `probe/weight_cycle_summary.json`

Diagnostic caveat:
remote `git_status_short.txt` is dirty because rsync excluded local artifact
directories and the remote checkout reported artifact deletes. This is acceptable
for this native CUDA diagnostic probe and is not release evidence.

Key auto-tile kernel-only results in microseconds:

| shape | m | hot | weight-cycle | cold-cache | weight/hot | cold/hot |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| qkv | 16 | 17.005 | 30.278 | 40.099 | 1.78x | 2.36x |
| qkv | 23 | 25.375 | 33.786 | 42.947 | 1.33x | 1.69x |
| qkv | 32 | 25.732 | 34.250 | 43.241 | 1.33x | 1.68x |
| o_proj | 16 | 11.819 | 21.418 | 24.793 | 1.81x | 2.10x |
| o_proj | 23 | 19.154 | 23.073 | 27.881 | 1.20x | 1.46x |
| o_proj | 32 | 20.039 | 23.448 | 28.247 | 1.17x | 1.41x |
| gate_up | 16 | 133.715 | 133.985 | 176.844 | 1.00x | 1.32x |
| gate_up | 23 | 137.396 | 136.962 | 181.151 | 1.00x | 1.32x |
| gate_up | 32 | 138.025 | 138.386 | 181.254 | 1.00x | 1.31x |
| down | 16 | 30.356 | 68.651 | 93.560 | 2.26x | 3.08x |
| down | 23 | 52.520 | 72.835 | 98.045 | 1.39x | 1.87x |
| down | 32 | 53.017 | 73.524 | 99.045 | 1.39x | 1.87x |

Interpretation:

- `gate_up` is stable under multi-weight cycling, so the dominant Gemma3 dense GPTQ
  gate/up bucket is compute/path-bound rather than a simple weight-cache artifact.
- `down` is cache sensitive, so product-side down-proj timing should be compared
  against weight-cycle/cold-cache brackets rather than repeated-hot microbench rows.
- This is a diagnostic native CUDA probe only. It does not prove product correctness,
  `ferrum run`, `ferrum serve`, or release-grade performance.

Release-grade status:
no `model_release_grade_manifest.json` and no
`MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`.
