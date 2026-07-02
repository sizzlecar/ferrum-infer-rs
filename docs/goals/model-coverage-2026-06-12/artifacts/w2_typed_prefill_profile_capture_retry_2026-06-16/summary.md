# W2 typed prefill profile capture retry
- status: FAIL
- git_head: `a9d8b439097f89011fb02dc78e1046ddb07d73e6`
- build_rc/profile_rc/run_profile_rc/bench_rc: `0/143/143/143`
- binary_sha256: `e111e6ec9653fd141ad5eb8ed504f18997a7d29244dbbe685be9955d2277a350  /workspace/ferrum-infer-rs/target/release/ferrum`
- chat_smoke_pass: `True` content `5` completion_tokens `3`
- server_ready_poll_count: `58`
- panic: `thread 'tokio-runtime-worker' (766) panicked at crates/ferrum-kernels/src/backend/cuda/mod.rs:843:34:`
- profile_lines: prefill `121`, batched_op `3`, unified `7`
- vast cleanup: cur_state `stopped`, actual_status `exited`
- release-grade: no `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`; this is diagnostic failure evidence.

## Tail profile evidence
- `[unified-prof] iter#3 items=10 prefill=10 decode=0 total=892506us model=890117us decode_post=0us | sample=0 sched=0 stream=0 stop=0 complete=0 (us)`
- `[unified-prof] iter#4 items=10 prefill=0 decode=10 total=28681us model=28527us decode_post=141us | sample=8 sched=0 stream=133 stop=0 complete=0 (us)`
- `[unified-prof] iter#5 items=10 prefill=0 decode=10 total=27727us model=27577us decode_post=140us | sample=12 sched=0 stream=128 stop=0 complete=0 (us)`
- `[unified-prof] iter#6 items=10 prefill=0 decode=10 total=27673us model=27477us decode_post=182us | sample=14 sched=0 stream=168 stop=0 complete=0 (us)`
- `[batched-op-profile] m=10 total=27836us  matmul=7117us(125) attn=2013us(62) qkr=910us(62) norm=462us(63) other=1496us(124) tail_norm=685us(62) tail_mlp=13577us(124) tail_gate_up=8924us(62) tail_down=4653us(62) marlin_ws_zero=0us(0) marlin_kernel=0us(0) marlin_qkv_ws=0us(0) marlin_qkv_kernel=0us(0) marlin_o_ws=0us(0) marlin_o_kernel=0us(0) marlin_gate_up_ws=0us(0) marlin_gate_up_kernel=0us(0) marlin_down_ws=0us(0) marlin_down_kernel=0us(0) marlin_lm_head_ws=0us(0) marlin_lm_head_kernel=0us(0) marlin_other_ws=0us(0) marlin_other_kernel=0us(0) tail_act=382us(62) tail_resid=499us(62)  unwrapped=695us`
- `[batched-op-profile] m=10 total=26973us  matmul=6929us(125) attn=1868us(62) qkr=626us(62) norm=446us(63) other=1362us(124) tail_norm=716us(62) tail_mlp=13511us(124) tail_gate_up=8896us(62) tail_down=4615us(62) marlin_ws_zero=0us(0) marlin_kernel=0us(0) marlin_qkv_ws=0us(0) marlin_qkv_kernel=0us(0) marlin_o_ws=0us(0) marlin_o_kernel=0us(0) marlin_gate_up_ws=0us(0) marlin_gate_up_kernel=0us(0) marlin_down_ws=0us(0) marlin_down_kernel=0us(0) marlin_lm_head_ws=0us(0) marlin_lm_head_kernel=0us(0) marlin_other_ws=0us(0) marlin_other_kernel=0us(0) tail_act=372us(62) tail_resid=494us(62)  unwrapped=649us`
- `[batched-op-profile] m=10 total=26876us  matmul=6890us(125) attn=1880us(62) qkr=622us(62) norm=459us(63) other=1334us(124) tail_norm=688us(62) tail_mlp=13480us(124) tail_gate_up=8867us(62) tail_down=4613us(62) marlin_ws_zero=0us(0) marlin_kernel=0us(0) marlin_qkv_ws=0us(0) marlin_qkv_kernel=0us(0) marlin_o_ws=0us(0) marlin_o_kernel=0us(0) marlin_gate_up_ws=0us(0) marlin_gate_up_kernel=0us(0) marlin_down_ws=0us(0) marlin_down_kernel=0us(0) marlin_lm_head_ws=0us(0) marlin_lm_head_kernel=0us(0) marlin_other_ws=0us(0) marlin_other_kernel=0us(0) tail_act=374us(62) tail_resid=491us(62)  unwrapped=658us`
- `[prefill-profile] matmuls: 62 calls 6 ms (avg 97 us)`
- `[prefill-profile] norms: 62 calls 0 ms (avg 8 us)`
- `[prefill-profile] tail_norm: 62 calls 0 ms (avg 12 us)`
- `[prefill-profile] tail_gate_up: 62 calls 23 ms (avg 374 us)`
- `[prefill-profile] tail_act: 62 calls 0 ms (avg 14 us)`
- `[prefill-profile] tail_down: 62 calls 13 ms (avg 223 us)`
- `[prefill-profile] tail_mlp: 62 calls 37 ms (avg 612 us)`
- `[prefill-profile] tail_resid: 62 calls 0 ms (avg 11 us)`
