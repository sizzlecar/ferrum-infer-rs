# Metal README Regression - 2026-06-01

Scope: README Apple Silicon rows with correctness, multi-turn, concurrency throughput, and swap evidence.

Ferrum: `ferrum 0.7.3`
Swap at start: `vm.swapusage: total = 2048.00M  used = 1636.06M  free = 411.94M  (encrypted)`
Swap at end: `vm.swapusage: total = 2048.00M  used = 1636.06M  free = 411.94M  (encrypted)`

| Model | Correctness | Multi-turn | c | README tok/s | Current tok/s | Ratio | Completed | Gate |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Llama-3.1-8B | FAIL | FAIL | 1 | 29.1 | 26.4 | 0.906 | 8/8 | pass |
| Llama-3.1-8B | FAIL | FAIL | 8 | 51.3 | 50.6 | 0.987 | 24/24 | pass |
| Llama-3.1-8B | FAIL | FAIL | 16 | 96.7 | 86.1 | 0.890 | 32/32 | FAIL |
| Qwen3-8B | pass | pass | 16 | 93.2 | 81.3 | 0.873 | 32/32 | FAIL |
| Qwen3-30B-A3B | pass | pass | 16 | 79.2 | 19.6 | 0.248 | 32/32 | FAIL |

Correctness prompts:
- `Llama-3.1-8B` Paris: `It looks like you are you referring to?`; multi-turn: `It seems to be a word.`
- `Qwen3-8B` Paris: `Paris`; multi-turn: `basalt`
- `Qwen3-30B-A3B` Paris: `Paris`; multi-turn: `basalt`

Notes:
- Performance gate is `current >= 0.90 * README baseline`, plus all requests completed.
- This runner records correctness failures and still collects performance data.
- Active swap means results are release-regression evidence, not clean marketing numbers.
