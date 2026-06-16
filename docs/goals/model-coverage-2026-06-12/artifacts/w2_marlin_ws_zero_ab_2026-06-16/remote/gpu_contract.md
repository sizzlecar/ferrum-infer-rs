lane: W2 Gemma3 Marlin workspace-zero native A/B
instance_id: 41171821
git_head: 38dd65c40f5d2a1feef2fdda8291101c4e1fde90
git_status:
correctness_gate: native probe exit 0 and VERDICT line
performance_command: timeout 20m bash scripts/microbenches/build_and_run_gemma3_tail_mlp_chain_perf.sh
diagnostic_only: true
