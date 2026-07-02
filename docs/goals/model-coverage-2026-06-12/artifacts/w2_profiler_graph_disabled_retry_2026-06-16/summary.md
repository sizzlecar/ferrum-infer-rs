# W2 profiler graph-disabled retry

- status: `PASS`
- bench_rc: `0`
- profile_outer_rc: `0`
- run_profile_rc: `0`
- chat_pass: `True`
- completed_per_run: `[16]`
- errored_per_run: `[0]`
- bad_output_per_run: `[0]`
- zero_output_tokens_per_run: `[0]`
- output_token_count_source: `usage`
- output_throughput_tps_mean: `312.22668693855985`
- ferrum_vs_vllm_percent: `60.18294525342617`
- prefill_profile_line_count: `297`
- batched_op_profile_line_count: `128`
- unified_prof_line_count: `67`
- graph_capture_line_count: `0`
- capture_unsupported_panic: `False`
- vast_stopped_actual_status: `exited`
- git_head: `f7612c3a2a17c7e051f326ed7bac54484b25eb3a`
- binary_sha256: `2a2ed419f3e80ede06ceaf54ba4495b66265c5ce2ba14b66dc39a35257cb6844  /workspace/ferrum-infer-rs/target/release/ferrum`
- release-grade: no `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`; this is diagnostic evidence only.

## Interpretation

diagnostic profiler path passes after source fix disables graph capture while syncing diagnostics are active; this is not release performance evidence.

## Tail profile evidence
- `[unified-prof] iter#63 items=10 prefill=0 decode=10 total=29245us model=28600us decode_post=630us | sample=253 sched=1 stream=376 stop=0 complete=0 (us)`
- `[unified-prof] iter#64 items=10 prefill=0 decode=10 total=29276us model=28619us decode_post=642us | sample=268 sched=1 stream=373 stop=0 complete=0 (us)`
- `[unified-prof] iter#96 items=16 prefill=0 decode=16 total=29025us model=28427us decode_post=578us | sample=206 sched=0 stream=372 stop=0 complete=0 (us)`
- `[unified-prof] iter#128 items=16 prefill=0 decode=16 total=29490us model=28684us decode_post=785us | sample=320 sched=0 stream=419 stop=0 complete=46 (us)`
- `[batched-op-profile] m=15 total=28018us  matmul=6982us(125) attn=2674us(62) qkr=621us(62) norm=450us(63) other=1372us(124) tail_norm=689us(62) tail_mlp=13662us(124) tail_gate_up=8983us(62) tail_down=4679us(62) marlin_ws_zero=0us(0) marlin_kernel=0us(0) marlin_qkv_ws=0us(0) marlin_qkv_kernel=0us(0) marlin_o_ws=0us(0) marlin_o_kernel=0us(0) marlin_gate_up_ws=0us(0) marlin_gate_up_kernel=0us(0) marlin_down_ws=0us(0) marlin_down_kernel=0us(0) marlin_lm_head_ws=0us(0) marlin_lm_head_kernel=0us(0) marlin_other_ws=0us(0) marlin_other_kernel=0us(0) tail_act=422us(62) tail_resid=494us(62)  unwrapped=652us`
- `[batched-op-profile] m=15 total=28005us  matmul=6950us(125) attn=2686us(62) qkr=625us(62) norm=445us(63) other=1369us(124) tail_norm=690us(62) tail_mlp=13663us(124) tail_gate_up=8987us(62) tail_down=4676us(62) marlin_ws_zero=0us(0) marlin_kernel=0us(0) marlin_qkv_ws=0us(0) marlin_qkv_kernel=0us(0) marlin_o_ws=0us(0) marlin_o_kernel=0us(0) marlin_gate_up_ws=0us(0) marlin_gate_up_kernel=0us(0) marlin_down_ws=0us(0) marlin_down_kernel=0us(0) marlin_lm_head_ws=0us(0) marlin_lm_head_kernel=0us(0) marlin_other_ws=0us(0) marlin_other_kernel=0us(0) tail_act=429us(62) tail_resid=496us(62)  unwrapped=652us`
- `[batched-op-profile] m=15 total=28039us  matmul=6972us(125) attn=2673us(62) qkr=633us(62) norm=452us(63) other=1369us(124) tail_norm=682us(62) tail_mlp=13679us(124) tail_gate_up=8993us(62) tail_down=4686us(62) marlin_ws_zero=0us(0) marlin_kernel=0us(0) marlin_qkv_ws=0us(0) marlin_qkv_kernel=0us(0) marlin_o_ws=0us(0) marlin_o_kernel=0us(0) marlin_gate_up_ws=0us(0) marlin_gate_up_kernel=0us(0) marlin_down_ws=0us(0) marlin_down_kernel=0us(0) marlin_lm_head_ws=0us(0) marlin_lm_head_kernel=0us(0) marlin_other_ws=0us(0) marlin_other_kernel=0us(0) tail_act=429us(62) tail_resid=496us(62)  unwrapped=654us`
- `[batched-op-profile] m=14 total=27900us  matmul=6946us(125) attn=2627us(62) qkr=622us(62) norm=446us(63) other=1369us(124) tail_norm=687us(62) tail_mlp=13633us(124) tail_gate_up=8975us(62) tail_down=4658us(62) marlin_ws_zero=0us(0) marlin_kernel=0us(0) marlin_qkv_ws=0us(0) marlin_qkv_kernel=0us(0) marlin_o_ws=0us(0) marlin_o_kernel=0us(0) marlin_gate_up_ws=0us(0) marlin_gate_up_kernel=0us(0) marlin_down_ws=0us(0) marlin_down_kernel=0us(0) marlin_lm_head_ws=0us(0) marlin_lm_head_kernel=0us(0) marlin_other_ws=0us(0) marlin_other_kernel=0us(0) tail_act=414us(62) tail_resid=496us(62)  unwrapped=660us`
- `[prefill-profile] matmuls: 62 calls 6 ms (avg 97 us)`
- `[prefill-profile] norms: 62 calls 0 ms (avg 8 us)`
- `[prefill-profile] tail_norm: 62 calls 0 ms (avg 12 us)`
- `[prefill-profile] tail_gate_up: 62 calls 23 ms (avg 374 us)`
- `[prefill-profile] tail_act: 62 calls 0 ms (avg 14 us)`
- `[prefill-profile] tail_down: 62 calls 13 ms (avg 223 us)`
- `[prefill-profile] tail_mlp: 62 calls 38 ms (avg 613 us)`
- `[prefill-profile] tail_resid: 62 calls 0 ms (avg 11 us)`
