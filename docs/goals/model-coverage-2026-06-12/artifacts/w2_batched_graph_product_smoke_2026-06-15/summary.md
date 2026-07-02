# W2 Batched Graph Product Smoke

- git_head: 2b3b5891ff94d6a4d793bb39bd6cab148af49588
- binary_sha256: c31d8b4af03f4669f7fac4fc49035adff97ca4d680d80775703aff99474b3d33
- model_path: /root/.cache/huggingface/hub/models--circulus--gemma-3-27b-it-gptq/snapshots/70d89a3a6b401b5f56558cb5d4c0f1fd158980b2
- run_rc: 0
- run_content: 5 
- run_selected_graph_mode: legacy_batched_decode_graph
- serve_selected_graph_mode: legacy_batched_decode_graph
- serve_selected_max_sequences: 16
- chat_content: 5 
- bench_rc: 0
- bench_completed: 16 completed / 0 errored
- bench_c16_throughput_tok_s: 372.3
- log_scan: no actual panic/CUDA error/illegal address/OOM/unk/PAD matches; bench schema field panic_per_run=0 matched by name

Diagnostic only: n_repeats=1, no CI; not release performance evidence.
