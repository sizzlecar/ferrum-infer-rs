# W2 Tail Profile c16 Diagnostic

- Status: `diagnostic_pass`
- Release gate: no
- Profile formats: `{'unified-op-profile': 176}`
- Profile rows: `176` total, `21` decode-only
- Bench completed/errors: `[100]` / `[0]`
- Output token count source: `usage`
- Profile-run output throughput mean: `372.00153904953766` tok/s
- Profile-run p95 ITL mean: `57.69527414999998` ms

## Decode-Only Aggregate

- total_us mean/p95/max: `27052.52380952381` / `27978` / `28153`
- generic_matmul_us mean/p95/max: `20944.190476190477` / `21075` / `21355`
- gate_up_us mean/p95/max: `8838.857142857143` / `8997` / `9005`
- down_us mean/p95/max: `4888.047619047619` / `4954` / `5011`
- qkv_us mean/p95/max: `2445.095238095238` / `2458` / `2631`
- attn_us mean/p95/max: `2220.5714285714284` / `3043` / `3133`
- lm_head_us mean/p95/max: `3009.714285714286` / `3028` / `3030`
- generic_matmul share: `0.7742046776728868`
- gate_up + down share: `0.5074167888569503`
- attention share: `0.08208370665178674`
- lm_head share: `0.11125447322052515`

This artifact is diagnostic only because profile logging changes runtime cost.
