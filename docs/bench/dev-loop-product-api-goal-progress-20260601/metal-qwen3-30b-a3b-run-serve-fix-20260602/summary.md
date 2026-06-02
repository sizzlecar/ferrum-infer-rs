# Metal README Regression - 2026-06-02

Scope: README Apple Silicon rows with correctness, multi-turn, concurrency throughput, and swap evidence.

Ferrum: `ferrum 0.7.4`
Swap at start: `vm.swapusage: total = 4096.00M  used = 2885.19M  free = 1210.81M  (encrypted)`
Swap at end: `vm.swapusage: total = 4096.00M  used = 2885.19M  free = 1210.81M  (encrypted)`

| Model | Serve correctness | Serve multi-turn | Run REPL multi-turn | c | README tok/s | Current tok/s | Ratio | Completed | Gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Qwen3-30B-A3B | pass | pass | pass | 16 | 72.5 | 67.5 | 0.931 | 32/32 | pass |

Correctness prompts:
- `Qwen3-30B-A3B` Paris: `

Paris`; serve multi-turn: `

蓝色月亮`; run multi-turn: `['已记住。', '蓝色月亮']`

Notes:
- Performance gate is `current >= 0.90 * README baseline`, plus all requests completed.
- Throughput cells request `ignore_eos=true` and streaming usage so runs are max-token and token-count comparable to the README harness.
- This runner records correctness failures and still collects performance data.
- Active swap means results are release-regression evidence, not clean marketing numbers.
