# W2 Gemma3 tail-MLP native CUDA chain probe

- Artifact: `docs/goals/model-coverage-2026-06-12/artifacts/w2_tail_mlp_chain_native_probe_2026-06-16`
- Lane: W2 Gemma3 native tail-MLP chain probe
- Remote git HEAD: `2c281e56557c11486cbdec5da9dae1234dcae78d`
- Remote git status: clean
- Probe rc: `0`
- Binary SHA256: `7dd82cd65a02958533c65b45d018e0b49600b1a30d394c7fa567a41f0d4ccca7`
- PASS line: `VERDICT: gemma3 tail MLP chain native CUDA probe complete`

## Key m16 Product Row

- `product_ws_zero` m16 `chain_event_us`: `215.75`
- `product_ws_zero` m16 `chain_host_sync_us`: `217.782`
- `gate_up_us`: `139.671`
- `down_us`: `70.903`
- `kernel_only_ws_prezero_diagnostic` m16 `chain_event_us`: `212.986`

## Interpretation

Single-layer Gemma3 tail-MLP chain timing at m16 is about 216us. Multiplying by 62 layers gives about 13.4ms, which matches the earlier product-side `tail_mlp` profile band. This rejects a hidden multi-ms launch-chain hypothesis and keeps the bottleneck on dense GPTQ MLP compute across layers, especially `gate_up` and `down`.

This is diagnostic evidence, not W2 release evidence. The final W2 validator has not produced `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`.
