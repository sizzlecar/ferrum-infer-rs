# W2 unified_decode fallback_reason diagnostic

- status: PASS
- bench_rc: 0
- chat_pass: True
- completed_per_run: [16]
- errored_per_run: [0]
- output_token_count_source: usage
- output_throughput_tps_mean: 315.39451845233344
- ferrum_vs_vllm_ratio_orientation: 0.6079355747378077
- unified_decode_line_count: 131
- unified_decode_prefill_line_count: 3
- unified_decode_first_prefill_line: [unified-decode] call#0 items=1 prefill=1 decode=0 total_q=23 attempted_unified=true fallback=true fallback_reason=paged_kv_required elapsed=126897us
- unified_decode_fallback_reasons: {'paged_kv_required': 131}
- prefill_profile_line_count: 297
