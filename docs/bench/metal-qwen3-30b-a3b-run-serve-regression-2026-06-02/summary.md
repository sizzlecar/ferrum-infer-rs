# Metal README Regression - 2026-06-02

Scope: README Apple Silicon rows with correctness, multi-turn, concurrency throughput, and swap evidence.

Ferrum: `ferrum 0.7.4`
Swap at start: `vm.swapusage: total = 4096.00M  used = 3315.94M  free = 780.06M  (encrypted)`
Swap at end: `vm.swapusage: total = 4096.00M  used = 3307.94M  free = 788.06M  (encrypted)`

| Model | Serve correctness | Serve multi-turn | Serve stream | Run REPL multi-turn | c | README tok/s | Current tok/s | Ratio | Completed | Gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Qwen3-30B-A3B | pass | pass | pass | pass | 16 | 72.5 | n/a | n/a | None/32 | FAIL |

Correctness prompts:
- `Qwen3-30B-A3B` Paris: `</think>

Paris`; serve multi-turn: `</think>

蓝色月亮。`; serve stream chunks: `9`; run multi-turn: `['</think>\n\n已记住。', '</think>\n\n蓝色月亮。']`; run tok/s: `7.835940650185882`; run text long: `{'passed': True, 'rc': 0, 'no_abort': True, 'rust_ok': True, 'turn_stat_lines': 4, 'contains_think': False, 'contains_unk': False, 'contains_pad': False}`

Notes:
- Performance gate is `current >= 0.90 * README baseline`, plus all requests completed.
- Throughput cells request `ignore_eos=true` and streaming usage so runs are max-token and token-count comparable to the README harness.
- `run` coverage includes JSONL multi-turn plus default text-mode long multi-turn to catch streaming and KV overflow regressions.
- This runner records correctness failures and still collects performance data.
- Active swap means results are release-regression evidence, not clean marketing numbers.
