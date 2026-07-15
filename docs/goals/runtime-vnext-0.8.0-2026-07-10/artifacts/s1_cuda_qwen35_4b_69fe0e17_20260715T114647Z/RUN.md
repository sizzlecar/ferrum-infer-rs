# S1 CUDA Qwen3.5-4B logical growth diagnostic

- Lane: `S1 CUDA / Qwen3.5-4B / ferrum run correctness diagnostic`
- Git SHA: `69fe0e17e3d973fd25f63ae693caa9dfa44041f1`
- Vast instance: `44965014` (reused warm model/build cache)
- Hardware: `1x NVIDIA GeForce RTX 4090, 24564 MiB`
- Hourly rate: `$0.4166666667`
- Correctness prerequisite: dynamic pool tests `38/38 PASS`; executor capacity ownership test `1/1 PASS`
- Stop condition: build failure, exact runtime failure, first valid token, or 90 seconds without a first token
- Model: `Qwen/Qwen3.5-4B`, cached HF snapshot `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
- Remote artifact: `/workspace/artifacts/s1_cuda_qwen35_4b_69fe0e17_20260715T114647Z`

## Prediction

The `CapacityShortfall` with `AwaitBackingGrowth` must be translated through the typed domain-to-pool map and grow the live domain from 1 MiB to the requested 1.5 MiB. The next accepted signal is a first output token or a new failure after sequence admission.

## Result

- Status: `REJECT`
- Build: `PASS` in 4m15s
- Binary SHA256: `efda9486752f832ab89110b167e6144752fecab0c9203b9ebe7aec239bc9e7d4`
- Admission prediction: `CONFIRMED`; the prior logical capacity failure disappeared.
- New exact failure: `resource resource/activation/sha256/ec38d0719b0d6ae9a455969982209971c071b65cd1df61852ab342c2f491368f differs from its value binding`
- Resource identity: `value.input.token_ids`
- Trace boundary: request slot and executor prefill workspace reserve/commit/release completed; dispatch failed before an operation profile event was emitted.
- Output validator: `REJECT`, assistant content was empty.
