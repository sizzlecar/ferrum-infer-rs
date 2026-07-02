# W2 Batched Graph Replay Observability Smoke

- git_head: 22f92677b34bab932407215fcb8c11dd0b372faf
- binary_sha256: f6d6828290c330749f1523c191c3e4034759f97d7c53e0ad4948d7e786995b1b
- model_path: /root/.cache/huggingface/hub/models--circulus--gemma-3-27b-it-gptq/snapshots/70d89a3a6b401b5f56558cb5d4c0f1fd158980b2
- serve_selected_graph_mode: legacy_batched_decode_graph
- serve_selected_max_sequences: 16
- chat_content: 5  
- bench_rc: 0
- bench_completed: 16
- bench_errored: 0
- bench_c16_output_throughput_tok_s: {
  "mean": 348.0037999403131
}
- replay_evidence:
  - [batched-graph-capture] key=4611686018427387920 m=15 m_padded=16 device_shadow=true
  - [batched-graph-replay] origin=post_capture count=1 key=4611686018427387920 m=15 m_padded=16 device_shadow=true
  - [batched-graph-replay] origin=pure count=2 key=4611686018427387920 m=15 m_padded=16 device_shadow=true
  - [batched-graph-replay] origin=pure count=4 key=4611686018427387920 m=15 m_padded=16 device_shadow=true
  - [batched-graph-replay] origin=pure count=8 key=4611686018427387920 m=15 m_padded=16 device_shadow=true
  - [batched-graph-replay] origin=pure count=16 key=4611686018427387920 m=15 m_padded=16 device_shadow=true
  - [batched-graph-capture] key=4611686018427387912 m=7 m_padded=8 device_shadow=true
- log_scan: no actual panic/CUDA error/illegal address/OOM/unk/PAD matches; bench schema field panic_per_run=0 matched by name

Diagnostic only: n_repeats=1, no CI; not release performance evidence.
