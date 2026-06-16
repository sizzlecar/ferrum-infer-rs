# W2 Unified Argmax C16 CUDA Diagnostic - GPU Contract

- Lane: W2 dense unified argmax c16 CUDA diagnostic.
- Hardware target: 1x RTX 4090 Vast instance, preferably cached instance `41187356`.
- Expected runtime/cost: 20-40 minutes, stop cap 45 minutes before a useful smoke/profile result; current cached offer is about USD 0.38/hour when running.
- Stop condition: instance start/SSH/CUDA/source sync/build first failure, `ferrum run` smoke failure, `ferrum serve` streaming smoke failure, `bench-serve --fail-on-error` failure, artifact collected, or profile shows the logits readback fix is not active.
- Correctness gate:
  - CUDA release build for `ferrum`;
  - product `ferrum run` smoke for `gemma3:27b-gptq`;
  - product `ferrum serve` streaming smoke with `stream_options.include_usage=true`;
  - c16 `bench-serve --fail-on-error` with completed requests full, errored requests zero, and usage token counts.
- Performance command: diagnostic-only c16 random 64/16, `--seed 9271`, `--n-repeats 1`, decode op profile and Marlin profile enabled through saved `ferrum.toml` runtime entries.
- Baseline comparison: previous same-shape Ferrum diagnostic in `w2_unified_op_profile_c16_rerun_2026-06-16` measured `readback=22039us` in a mixed c16 unified frame and `158.877 tok/s`. This run is not release evidence.
- Cleanup: copy artifacts back, then stop or destroy the instance and save final Vast status.

## Outcome

- Cached instances `41187356` and `41178475` could not be restarted because
  Vast queued the state change with resources unavailable.
- Instance `41218189` was created but SSH never became usable; it was
  destroyed before any benchmark work.
- Actual diagnostic ran on instance `41218739`, 1x RTX 4090, quoted
  `0.4696296296296296` USD/hour.
- The first CUDA release build on this fresh instance took `43m 52s`. The run
  exceeded the original 45-minute soft cap because the build was visibly
  progressing and re-creating the machine would have repeated environment,
  model-cache, and compile setup.
- Artifacts were copied back and instance `41218739` was stopped; final Vast
  status was `cur_state=stopped`, `actual_status=exited`.
