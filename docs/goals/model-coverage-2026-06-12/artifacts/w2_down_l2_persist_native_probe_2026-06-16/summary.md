# W2 Gemma3 down L2 persistence native CUDA probe

- Artifact: `docs/goals/model-coverage-2026-06-12/artifacts/w2_down_l2_persist_native_probe_2026-06-16`
- Lane: W2 Gemma3 down L2 persistence native probe
- Remote git HEAD: `6cf26ca99f1958d2e326245bbe55fd8ed22c7e4a`
- Probe rc: `0`
- Binary SHA256: `c3fafa5657c5dbc1496f6a9790ffc4440cb4f17ddf01014f55df1212226826f3`
- PASS line: `VERDICT: gemma3 down L2 persistence native CUDA probe complete`

## Key Rows

- Device L2: `75,497,472` bytes
- Persisting L2 max: `51,904,512` bytes
- Access window max: `134,213,632` bytes
- Down qweight policy window: `57802752` bytes
- m16 warm repeated baseline: `35.135` us
- m16 no policy after gate_up+GeGLU: `70.342` us
- m16 persist full window hit 100: `35.088` us
- m16 persist half window hit 100: `40.936` us
- m16 persist full window hit 60: `33.158` us
- m32 warm repeated baseline: `55.127` us
- m32 no policy after gate_up+GeGLU: `75.148` us
- m32 persist full window hit 100: `55.545` us
- m32 persist full window hit 60: `54.434` us

## Interpretation

CUDA stream access-policy for down qweight materially restores `down_proj` performance after the product-shaped `gate_up+GeGLU` producer. On m16, no policy is `70.342us` while full-window persisting is `35.088us` and `hit_ratio=0.6` is `33.158us`. On m32, no policy is `75.148us` while full-window persisting is `55.545us`.

This confirms a concrete W2 lever: productize a typed CUDA L2 residency policy for dense Marlin `down_proj`, then validate correctness before endpoint performance claims.

This is diagnostic evidence, not W2 release evidence. The final W2 validator has not produced `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`.
