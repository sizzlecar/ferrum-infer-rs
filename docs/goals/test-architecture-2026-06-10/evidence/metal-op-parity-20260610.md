# Metal op-parity evidence (local, Apple Silicon)

Date: 2026-06-10. Host: this Apple Silicon Mac (Metal is the local GPU).

Command:
```
cargo test -p ferrum-testkit --features metal --test op_diff
```

Result: **11/11 op_diff tests passed** — CPU-reference vs Metal NMSE within
`NMSE_FP16_TOL = 1e-6` for every covered op:

- rms_norm (small + llama shapes)
- gemm (small + qkv shapes)
- qk_norm_rope (mode 0/1 + llama-offset)
- silu_mul (small + llama shapes)
- residual_add (new)
- fused_add_rms_norm (new)

Significance:
- Gate A4 Metal column is validated for the 6 covered ops (parity RUNS green,
  not just scaffolding).
- Confirms the stage-3 decoupling (`supports_native_unified_decode`) did not
  perturb Metal kernels: every Metal kernel still matches its CPU reference.

CUDA column for these ops still needs a pod (L1-cuda batch).

## L1-metal stability (Gate C5)

`lane_l1_metal.sh` steps (op-parity metal + tiny_stack metal) run 10×
consecutively: **10/10 green**. Deterministic by construction (greedy
tiny-model decode + fixed-seed op inputs). Satisfies the Gate C5
l1_metal 10/10 requirement.
