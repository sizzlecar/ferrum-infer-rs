# Metal README Regression - 2026-06-01

Scope: README Apple Silicon rows with correctness, multi-turn, concurrency throughput, and swap evidence.

Ferrum: `ferrum 0.7.3`
Swap at start: `vm.swapusage: total = 2048.00M  used = 1620.06M  free = 427.94M  (encrypted)`
Swap at end: `vm.swapusage: total = 2048.00M  used = 1620.06M  free = 427.94M  (encrypted)`

| Model | Correctness | Multi-turn | c | README tok/s | Current tok/s | Ratio | Completed | Gate |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Llama-3.1-8B | pass | pass | 1 | 29.1 | 31.7 | 1.090 | 8/8 | pass |
| Llama-3.1-8B | pass | pass | 8 | 51.3 | 51.7 | 1.007 | 24/24 | pass |
| Llama-3.1-8B | pass | pass | 16 | 96.7 | 89.4 | 0.925 | 32/32 | pass |
| Qwen3-8B | pass | pass | 16 | 93.2 | 86.0 | 0.922 | 32/32 | pass |
| Qwen3-30B-A3B | pass | pass | 16 | 79.2 | 72.5 | 0.916 | 32/32 | pass |

Correctness prompts:
- `Llama-3.1-8B` Paris: `The capital of France is Paris.`; multi-turn: `Paris`
- `Qwen3-8B` Paris: `Paris`; multi-turn: `The assistant named Paris.`
- `Qwen3-30B-A3B` Paris: `Paris`; multi-turn: `The assistant just named Paris.`

Notes:
- Performance gate is `current >= 0.90 * README baseline`, plus all requests completed.
- Throughput cells request `ignore_eos=true` and streaming usage so runs are max-token and token-count comparable to the README harness.
- This runner records correctness failures and still collects performance data.
- Active swap means results are release-regression evidence, not clean marketing numbers.
