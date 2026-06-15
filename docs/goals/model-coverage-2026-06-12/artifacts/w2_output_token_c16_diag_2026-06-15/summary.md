# W2 c16 output-token/batch-shape diagnostic

- remote_head: `25c32dac9305eb62acd733bd491b2d1294a3ba64`
- binary_sha256: `a7f561a5f49a6858e8a63040a595143395f369ab7da1a5983d94927469b3861a`
- build_rc: `0`
- serve: ready on `127.0.0.1:18084`
- chat_smoke: content `5`, usage completion_tokens `3`
- bench_rc: `0`
- completed/errored: `[16]` / `[0]`
- output_token_count_source: `usage`
- output_tokens_per_request: `[[32,32,32,28,32,32,32,32,32,30,32,32,32,32,32,32]]`
- output_throughput_tps_mean: `363.0868797206277`
- ttft_p50_ms: `496.0399835`
- tpot_p50_ms: `27.226174516129035`
- Vast shutdown: `cur_state=stopped`, `actual_status=exited`

## Interpretation

- The new per-request output-token field is present and shows only 6 total
  output tokens short of the 16x32 cap. Short outputs do not explain the W2
  performance gap.
- The first batched decode trace saw `m=13`, but the captured/replayed main
  graph shape was `m=16 m_padded=16`; the previous `m=15` capture is not a
  stable sustained-decode bottleneck.
- The drain shape captured at `m=3 m_padded=4`, consistent with two shorter
  requests finishing before the rest.
- This run is diagnostic-only (`n_repeats=1`, no CI, no `--require-ci`) and is
  not release performance evidence.

## Graph Lines

- `[batched-graph-capture] key=4611686018427387920 m=16 m_padded=16 device_shadow=true`
- `[batched-graph-replay] origin=post_capture count=1 key=4611686018427387920 m=16 m_padded=16 device_shadow=true`
- `[batched-graph-replay] origin=pure count=2 key=4611686018427387920 m=16 m_padded=16 device_shadow=true`
- `[batched-graph-replay] origin=pure count=4 key=4611686018427387920 m=16 m_padded=16 device_shadow=true`
- `[batched-graph-replay] origin=pure count=8 key=4611686018427387920 m=16 m_padded=16 device_shadow=true`
- `[batched-graph-replay] origin=pure count=16 key=4611686018427387920 m=16 m_padded=16 device_shadow=true`
- `[batched-graph-capture] key=4611686018427387908 m=3 m_padded=4 device_shadow=true`
