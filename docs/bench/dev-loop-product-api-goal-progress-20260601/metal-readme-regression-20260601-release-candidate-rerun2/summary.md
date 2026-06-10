# Metal README Regression - 2026-06-01

Scope: README Apple Silicon rows with correctness, multi-turn, concurrency throughput, and swap evidence.

Ferrum: `ferrum 0.7.3`
Swap at start: `vm.swapusage: total = 2048.00M  used = 1620.06M  free = 427.94M  (encrypted)`
Swap at end: `vm.swapusage: total = 2048.00M  used = 1620.06M  free = 427.94M  (encrypted)`

| Model | Correctness | Multi-turn | c | README tok/s | Current tok/s | Ratio | Completed | Gate |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Llama-3.1-8B | pass | FAIL | 1 | 29.1 | 32.0 | 1.098 | 8/8 | pass |
| Llama-3.1-8B | pass | FAIL | 8 | 51.3 | 50.6 | 0.987 | 24/24 | pass |
| Llama-3.1-8B | pass | FAIL | 16 | 96.7 | 86.7 | 0.897 | 32/32 | FAIL |
| Qwen3-8B | pass | pass | 16 | 93.2 | 84.2 | 0.903 | 32/32 | pass |
| Qwen3-30B-A3B | pass | pass | 16 | 79.2 | 71.2 | 0.899 | 32/32 | FAIL |

Correctness prompts:
- `Llama-3.1-8B` Paris: `The capital of France is Paris.`; multi-turn: `You can be a memory.`
- `Qwen3-8B` Paris: `Paris`; multi-turn: `basalt`
- `Qwen3-30B-A3B` Paris: `Paris`; multi-turn: `basalt`

Notes:
- Performance gate is `current >= 0.90 * README baseline`, plus all requests completed.
- Throughput cells request `ignore_eos=true` and streaming usage so runs are max-token and token-count comparable to the README harness.
- This runner records correctness failures and still collects performance data.
- Active swap means results are release-regression evidence, not clean marketing numbers.
