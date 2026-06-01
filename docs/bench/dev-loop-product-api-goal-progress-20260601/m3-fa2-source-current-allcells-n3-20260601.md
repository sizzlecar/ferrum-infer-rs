# Native Source FA2 All-Cell N=3 - 2026-06-01

Purpose: provide same-pod all-cell evidence for the restored in-repo native
`FERRUM_FA2_SOURCE=1` path after removing the external FlashAttention/CUTLASS
product build dependency.

Remote artifact:

- `/workspace/m3-fa2-source-current-allcells-n3-20260601`

Run configuration:

- run code head: `451a97f`
- `REPEATS=3`
- `NUM_PROMPTS=64`
- `WARMUP_REQUESTS=10`
- concurrencies: `1,4,16,32`
- baseline env: `FERRUM_FA_LAYOUT_VARLEN=1`
- candidate env: `FERRUM_FA_LAYOUT_VARLEN=1,FERRUM_FA2_SOURCE=1`
- correctness gates: Paris, multi-turn Paris, and three-user-turn
  `Paris/basalt` recall passed for every case.

Artifact validation:

```bash
cd /workspace/ferrum-fa2-native-restore-git-ac3dfab
python3 scripts/m3_validate_runner_artifact.py \
  --require-bench /workspace/m3-fa2-source-current-allcells-n3-20260601
```

Validator result:

- `ok=true`
- `summary_rows=8`
- all 8 cases have `bench=true`

Results:

| c | source FA2 tok/s | FA-layout tok/s | source delta vs FA-layout | vLLM 0.20.2 tok/s | source/vLLM |
|---:|---:|---:|---:|---:|---:|
| 1 | `157.18 ± 0.21` | `157.04 ± 1.09` | `+0.09%` | `183.9` | `0.855×` |
| 4 | `448.36 ± 44.73` | `436.34 ± 45.62` | `+2.75%` | `512.5` | `0.875×` |
| 16 | `1115.58 ± 39.16` | `1042.56 ± 63.58` | `+7.00%` | `1331.9` | `0.838×` |
| 32 | `1488.08 ± 205.49` | `1324.57 ± 100.63` | `+12.34%` | `1972.9` | `0.754×` |

Conclusion:

- The restored in-repo native source FA2 path is correct on all four cells and
  improves over FA-layout at c4/c16/c32.
- It passes the 2026-06-01 formal release performance threshold of
  `0.75x vLLM` on all cells.
- It does not meet the stretch M3 80% target because c32 remains below the
  required `0.80 * 1972.9 = 1578.3 tok/s`.
- c32 needs about `+90 tok/s` over this result, roughly `+6.1%`, to clear the
  stretch target.
- This is opt-in source-FA2 evidence. It is not a final default-path completion
  packet for the dev-loop/product-API goal.
