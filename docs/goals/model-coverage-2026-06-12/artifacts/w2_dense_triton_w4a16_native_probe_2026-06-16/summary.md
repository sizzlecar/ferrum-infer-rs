# W2 Gemma3 dense Triton W4A16 native CUDA probe

- Artifact: `docs/goals/model-coverage-2026-06-12/artifacts/w2_dense_triton_w4a16_native_probe_2026-06-16`
- Lane: W2 Gemma3 dense Triton W4A16 vs Marlin native probe
- Remote git HEAD: `2847822395e857cbe23196b9590b88479eadeb60`
- Remote git status: clean
- Probe rc: `0`
- Binary SHA256: `83a8112f31951e930b90736fcc7a7a99db69936fdebfa1f92b17449159a6e77c`
- PASS line: `VERDICT: dense Triton W4A16 Gemma3 native CUDA probe complete`

## Key m16 Rows

- gate_up Marlin product ws-zero: `137.111` us
- gate_up Triton W4A16: `618.924` us
- gate_up Triton/Marlin: `4.51x` slower
- down Marlin product ws-zero: `32.527` us
- down Triton W4A16: `609.813` us
- down Triton/Marlin: `18.75x` slower

## Interpretation

The existing Triton W4A16 dense GPTQ PTX is much slower than Marlin on the exact Gemma3 tail-MLP shapes. This rejects direct productization of the current Triton dense GPTQ path as the W2 performance lever. Any Triton direction would need a new kernel/tile design, not the existing PTX.

This is diagnostic evidence, not W2 release evidence. The final W2 validator has not produced `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`.
