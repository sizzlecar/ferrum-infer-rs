# FastH3 raw attention projections

This component consumes already modulated BF16 `[rows, hidden]` and produces
four separate BF16 `[rows, heads * head_dim]` tensors: Q, K, V and the VSA
compression gate. For the selected checkpoint these widths are 5376 and 7168.
No bias, QK normalization, RoPE, attention, output projection or gate-activity
decision is folded into this component. Even a structurally zero gate is
returned as a raw projection.

The fixed unit-scale profile follows FastVideo commit
`866a3cdb91a85aef8561847365fc8564add9e4c8`:

- `layers/quantization/nvfp4_config.py`: `uses_unit_activation_scale`,
  `quantize_input`, and the stored F32 alpha division in `apply`.
- `models/dits/minimax_h3_vsa_fp4.py`: `_shared_input_projections`.
- `models/dits/minimax_h3.py`: `MiniMaxH3Attention` projection dimensions and
  the separate compression-gate activity decision.

The graph has one explicitly profiled NVFP4 quantizer and four NVFP4 GEMMs.
All four GEMMs reuse the same packed activation and scale values. A projection
with a checkpoint input-global-scale component is rejected; there is no hidden
amax rule or environment override.

Original packed weights and scales retain their checked physical component
views, source format and mmap owner. Invocation alpha is derived once from the
original F32 alpha divided by F32 one. Its source component, file, exact bits
and arithmetic profile are included in its derived component identity. The
BF16 weight-global-scale value is not used to reconstruct alpha. Only four
F32 alpha values and one ordinary numerical unit constant allocate managed
Host weight backing, for 20 bytes total before runtime alignment/staging.

CPU tests cover the actual package source, full operation binding contracts,
non-square output geometry, shared quantization, recipe substitution, source
capacity rollback and an independent exact-subset oracle. The ignored CUDA
test exercises the production component initialization, execution and checked
close with the same package fixture:

```text
cargo test -p ferrum-models --features cuda-cutlass-nvfp4 --lib \
  actual_attention_package_executes_shared_quantization_four_projections_and_closes \
  -- --ignored --nocapture
```

The GPU test requires the configured SM120 host and its pinned native operator
lock. Its independent FP64 dot oracle uses E2M1-exact inputs with amax six in
every 16-element group, unit E4M3 scale, and dyadic weights and alphas. This
deliberately narrow domain does not establish general activation-quantization
accuracy, complete attention equivalence, or video quality.
