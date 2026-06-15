lane: W2 Gemma3 native tail-MLP chain probe
instance: Vast 40826362 if reusable
expected_runtime_cost: 10-20 minutes, approx USD 0.07-0.15 at prior USD 0.425/hr
stop_condition: nvcc compile failure, probe nonzero/timeout, or VERDICT line with artifacts copied back
correctness_gate: native probe exit 0 and VERDICT line
performance_command: bash scripts/microbenches/build_and_run_gemma3_tail_mlp_chain_perf.sh
local_head: 2c281e56557c11486cbdec5da9dae1234dcae78d
local_status_short:
