# W2 Gemma3 down input-source native CUDA probe

- Artifact: `docs/goals/model-coverage-2026-06-12/artifacts/w2_down_input_source_native_probe_2026-06-16`
- Lane: W2 Gemma3 down input-source native probe
- Remote git HEAD: `cea63ef0b5c933a2a39802a82010b34eaa1a9d45`
- Probe rc: `0`
- Binary SHA256: `dd1a5ba3cd0f244603bc1fbebe8f2a6a224004f98943a5c31a21001e9aa7bfb0`
- PASS line: `VERDICT: gemma3 down input-source native CUDA probe complete`

## Key m16 Rows

- const baseline: `32.606` us
- small const baseline: `32.67` us
- const after gate_up+GeGLU immediate: `69.793` us
- const after L2 flush: `90.343` us
- GeGLU immediate: `68.356` us
- GeGLU after sync: `70.2` us
- copied GeGLU after sync: `70.098` us

## Interpretation

The isolated down_proj row is fast when repeated on constant input, but it slows to the same band as the full tail-MLP chain whenever a gate_up+GeGLU producer runs immediately before it, even if down reads a separate constant input. Small constant input is not slower, so the gap is not explained by GeGLU numeric magnitude.

This points to cache/producer-state or weight-residency around the `gate_up -> down` sequence. The next W2 lever should focus on that sequence rather than current Triton W4A16 or GeGLU data sensitivity.

This is diagnostic evidence, not W2 release evidence. The final W2 validator has not produced `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`.
