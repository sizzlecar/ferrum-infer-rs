# W2 typed-config decode integration profile

- Artifact: `docs/goals/model-coverage-2026-06-12/artifacts/w2_typed_decode_profile_2026-06-16/`.
- Scope: diagnostic only. This is not W2 release-grade evidence and did not run
  the final `model_release_grade_goal_gate.py w2` validator.
- Remote HEAD: `4fea56ec79d0c8a9edcf99dd90b3889d422869e9`.
- Non-artifact source status on the remote was clean.
- CUDA release binary SHA256:
  `23f04a49e361c836ab6a8afb125d68e771df361219013bee0d32ecf630a2559d`.
- Vast instance `40826362` was stopped after artifact copyback; sanitized status
  JSON records `cur_state=stopped` and `actual_status=exited`.

Correctness smoke:

- `ferrum serve` reached `/v1/models` after poll `62`.
- Chat smoke returned content `5` with `completion_tokens=3`.
- `bench-serve --fail-on-error` returned rc `0`.
- Bench report: `completed_per_run=[16]`, `errored_per_run=[0]`,
  `bad_output_per_run=[0]`, `zero_output_tokens_per_run=[0]`.

Typed runtime config evidence:

- `FERRUM_BATCH_DECODE_PROF` source: `config_file`.
- `FERRUM_NEXT_BATCH_PROF` source: `config_file`.
- `FERRUM_UNIFIED_POST_PROF` source: `config_file`.
- Selected graph mode: `legacy_batched_decode_graph`.

Performance diagnostic:

- c16, `n_repeats=1`, `num_prompts=16`, random 32 input / 32 output.
- Output throughput mean: `380.4915241647801 tok/s`.
- TTFT p50: `587.8543155 ms`.
- TPOT p50: `23.82219966129032 ms`.
- Output token source: `usage`.
- Output tokens per request:
  `[[32,32,32,28,32,32,32,31,32,32,32,32,32,30,32,32]]`.

Profile interpretation:

- Full `decode=16` iterations: `n=27`.
- Mean total per full decode iteration: `23679.2 us`.
- Mean model time per full decode iteration: `23311.3 us`.
- Mean decode postprocess time per full decode iteration: `347.9 us`.
- Mean model share: `98.44%`.
- Mean decode postprocess share: `1.47%`.
- Background loop gaps are mostly single-digit microseconds.

Conclusion: the current c16 bottleneck is not ordinary engine scheduling,
postprocess, stream emission, or host loop gap. The high-return path remains
Gemma3 model-side decode work, especially tail/Marlin/projection behavior and
weight-residency style effects. This result should not be treated as a release
performance claim because it is a single-run diagnostic without CI.
