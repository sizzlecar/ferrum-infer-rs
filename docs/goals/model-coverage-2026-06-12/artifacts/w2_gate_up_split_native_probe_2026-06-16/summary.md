# W2 gate_up split native CUDA probe

Artifact:
`docs/goals/model-coverage-2026-06-12/artifacts/w2_gate_up_split_native_probe_2026-06-16/`.

## Scope

- Lane: `W2 Gemma3 gate_up split-vs-fused native probe`.
- Source head: `50abea26c005c3115a7deb931434f53d0803de51`.
- Source status before remote sync: `clean-tracked-before-remote-sync`.
- Command: `bash scripts/microbenches/build_and_run_gemma3_gate_up_split_perf.sh`.
- Exit code: `0`.
- Verdict: `VERDICT: gemma3 gate_up split native CUDA probe complete`.
- Cleanup: `poll=00 cur_state=stopped actual_status=exited intended_status=stopped`.
- Final Vast API status:
  `cur_state=stopped actual_status=exited intended_status=stopped`.

## Runtime

- GPU: `NVIDIA GeForce RTX 4090`, 24564 MiB.
- Driver: `565.77`.
- CUDA visible through `nvidia-smi`: `12.7`.
- `nvcc`: `12.4.131`.
- Binary SHA256:
  `f0939e6164e17e6d24b18dc127ff567f5a464913bbcd36b6cfea925caf1140e5`.

## Result

| m | product fused us | split serial speedup | split overlap speedup |
|---:|---:|---:|---:|
| 1 | 135.473 | 0.9615 | 1.0005 |
| 10 | 139.265 | 0.9714 | 1.0116 |
| 16 | 139.507 | 0.9771 | 1.0136 |
| 23 | 143.324 | 0.9416 | 0.9828 |
| 32 | 145.377 | 0.9473 | 0.9899 |

## Conclusion

Serial split `gate`/`up` is slower at every tested batch size. Two-stream split is
only neutral to slightly faster around `m=10` and `m=16`, with a maximum isolated
segment speedup of `1.0136x`, and regresses at `m=23` and `m=32`.

This is not a material W2 lever. Productizing split `gate`/`up` would add loader,
runtime, stream, and correctness risk for at most about `1.4%` isolated segment
gain in the local shape where W2 is still short by much more. The branch should
be rejected for W2, and the next work should target another tail-MLP
work-reduction/fusion or prefill wall-time lever.

This artifact is diagnostic native CUDA evidence only. It is not a
`MODEL_RELEASE_GRADE_W2 PASS: <out_dir>` artifact.
