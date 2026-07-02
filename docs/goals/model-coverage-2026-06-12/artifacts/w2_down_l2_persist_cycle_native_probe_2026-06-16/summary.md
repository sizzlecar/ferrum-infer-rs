# W2 Gemma3 down L2 persistence cycle native CUDA probe

- Artifact: `docs/goals/model-coverage-2026-06-12/artifacts/w2_down_l2_persist_cycle_native_probe_2026-06-16`
- Lane: W2 Gemma3 down L2 persistence cycle native probe
- Remote git HEAD: `357a4b98a2eb80744b8beacf256b91bbff8ae0f2`
- Probe rc: `0`
- Binary SHA256: `f9c3e69f4407c4b4bd42b7f28593efcc7eb1c2bc81dff7c10ba98baf10b510f1`
- PASS line: `VERDICT: gemma3 down L2 persistence cycle native CUDA probe complete`

## Key Rows

- Policy window: `57802752` bytes
- m16 single-layer no-policy: `69.832` us
- m16 single-layer persist hit60: `34.493` us
- m16 cycle8 no-policy: `69.736` us
- m16 cycle8 persist hit60: `69.743` us
- m16 cycle8 persist with explicit down-warm: `34.634` us
- m32 single-layer no-policy: `75.903` us
- m32 single-layer persist hit60: `58.984` us
- m32 cycle8 no-policy: `75.745` us
- m32 cycle8 persist hit60: `75.117` us
- m32 cycle8 persist with explicit down-warm: `54.067` us

## Interpretation

Single-layer L2 persistence reproduces the prior win, but simple per-layer access-policy does not survive 8-layer weight rotation. At m16, single-layer persist improves `69.832us` to `34.493us`, while cycle8 no-policy is `69.736us` and cycle8 persist is `69.743us`. At m32, cycle8 no-policy is `75.745us` and cycle8 persist is `75.117us`.

Explicit down-warm remains an upper bound (`34.634us` at m16, `54.067us` at m32), but it adds an extra down-weight read. Therefore simple access-policy alone should not be productized as the W2 performance fix; any viable lever needs an overlap/prefetch/reuse strategy that survives layer rotation.

This is diagnostic evidence, not W2 release evidence. The final W2 validator has not produced `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`.
