# W2 Gemma3 down prefetch-overlap native CUDA probe

- Artifact: `docs/goals/model-coverage-2026-06-12/artifacts/w2_down_prefetch_overlap_native_probe_2026-06-16`
- Lane: W2 Gemma3 down prefetch-overlap native probe
- Remote git HEAD: `432e6588bac59902b7488484934494c751534221`
- Probe rc: `0`
- Binary SHA256: `58491a34483c8c4ba0ccbd4b1d9c9b127676b1f520a6ba42a0409daae5cc64bc`
- PASS line: `VERDICT: gemma3 down prefetch-overlap native CUDA probe complete`

## Key Rows

- m16 no prefetch: down `69.744` us, segment `216.072` us
- m16 overlap qweight: down `34.063` us, segment `239.437` us
- m16 overlap qweight+scales: down `32.56` us, segment `238.979` us
- m16 serial qweight+scales segment: `243.142` us
- m32 no prefetch: down `74.849` us, segment `227.628` us
- m32 overlap qweight: down `53.836` us, segment `268.3` us
- m32 overlap qweight+scales: down `53.79` us, segment `269.916` us
- m32 serial qweight+scales segment: `273.308` us

## Interpretation

The overlap warm kernel restores down kernel time under 8-layer rotation, but total segment wall time gets worse. At m16, overlap qweight+scales reduces down from `69.744us` to `32.560us` but increases segment wall time from `216.072us` to `238.979us`. At m32, down drops from `74.849us` to `53.790us` while segment wall time increases from `227.628us` to `269.916us`.

Current explicit warm/prefetch is therefore not a productizable W2 performance fix; it shifts cost rather than lowering wall time. This closes the simple cache-warm branch unless a cheaper producer-integrated prefetch design is identified.

This is diagnostic evidence, not W2 release evidence. The final W2 validator has not produced `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`.
