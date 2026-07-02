lane: W2 Gemma3 dense Triton W4A16 vs Marlin native probe
instance: Vast 40826362 if reusable
expected_runtime_cost: 10-20 minutes, approx USD 0.07-0.15 at prior USD 0.425/hr
stop_condition: startup/SSH/CUDA failure, nvcc compile failure, probe nonzero/timeout, or VERDICT line with artifacts copied back
correctness_gate: native probe exit 0 and VERDICT line
performance_command: bash scripts/microbenches/build_and_run_dense_triton_w4a16_gemma3_perf.sh
local_head: 2847822395e857cbe23196b9590b88479eadeb60
local_status_short:
