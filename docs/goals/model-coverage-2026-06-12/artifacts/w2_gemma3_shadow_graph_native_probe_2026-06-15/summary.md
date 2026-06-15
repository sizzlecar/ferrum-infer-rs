# W2 Gemma3 Shadow Graph Native Probe

This is a native CUDA diagnostic. It does not load a model, run `ferrum run`, run
`ferrum serve`, or produce release-grade performance evidence.

## Result

- Status: PASS.
- Remote HEAD: `c46d95408d12c8c1e177145f7c4c217a34080e62`.
- GPU: `NVIDIA GeForce RTX 4090`, 24564 MiB, driver `565.77`.
- CUDA compiler: `Build cuda_12.4.r12.4/compiler.34097967_0`.
- Command: `bash scripts/microbenches/build_and_run_gemma3_shadow_graph_bench.sh`.
- Probe rc: `0`.
- Required verdict:
  `VERDICT: Gemma3 shadow graph native CUDA probe complete`.
- Vast cleanup: `cur_state=stopped actual_status=exited`.

## Measurements

The probe simulates a Gemma3-style device F32 residual shadow decode step:
62 layers, batch 16, hidden size 5376, 498 kernel launches per step.

- eager ordered state upload: `1.143 ms/step`.
- eager pre-sync state upload: `1.137 ms/step`.
- graph ordered state upload: `0.565 ms/step`, `2.02x` speedup.
- graph pre-sync state upload: `0.568 ms/step`, `2.00x` speedup.
- checksum16: `127.94618988`.

## Interpretation

The native CUDA result shows that a persistent-buffer, device-shadow-like decode
step can be captured and replayed stably, and that graph replay has meaningful
launch-overhead headroom for the Gemma3 decode shape.

This does not mean the Ferrum product path is fixed. Current product code still
keeps legacy batched CUDA graph disabled when host/device residual shadow is
active. The next product change should be a narrow, shadow-safe graph eligibility
guard plus `ferrum run` and `ferrum serve` correctness smoke before any
performance measurement.
