# Metal README Regression - 2026-06-11

Scope: README Apple Silicon rows with correctness, multi-turn, concurrency throughput, and swap evidence.

Ferrum: `ferrum 0.7.7`
Swap at start: `vm.swapusage: total = 11264.00M  used = 10251.50M  free = 1012.50M  (encrypted)`
Swap at end: `vm.swapusage: total = 17408.00M  used = 16322.38M  free = 1085.62M  (encrypted)`

| Model | Default max seqs | Bench max seqs | Serve correctness | Serve multi-turn | Serve stream | Stateful loop | Tool call | Run REPL multi-turn | c | Quality | in/out | README tok/s | Current tok/s | Ratio | Completed | Gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Llama-3.1-8B | 16 | 16 | pass | pass | pass | pass | pass | pass | 1 | pass | 16/64 | 29.1 | 26.8 | 0.922 | 8/8 | pass |
| Llama-3.1-8B | 16 | 16 | pass | pass | pass | pass | pass | pass | 8 | pass | 16/64 | 46.4 | 45.6 | 0.983 | 24/24 | pass |
| Llama-3.1-8B | 16 | 16 | pass | pass | pass | pass | pass | pass | 16 | pass | 16/64 | 80.9 | 75.4 | 0.932 | 32/32 | pass |
| Qwen3-8B | 16 | 16 | pass | pass | pass | pass | pass | pass | 16 | pass | 16/64 | 57.1 | 84.4 | 1.477 | 32/32 | pass |
| Qwen3-30B-A3B | 16 | 16 | pass | pass | pass | pass | pass | pass | 16 | pass | 16/64 | 72.5 | 68.5 | 0.944 | 32/32 | pass |

Correctness prompts:
- `Llama-3.1-8B` Paris: `Paris`; serve multi-turn: `蓝色月亮`; serve stream chunks: `5`; run multi-turn: `['已记住。', '蓝色月亮']`; run tok/s: `10.477394167129056`; run text long: `{'passed': True, 'rc': 0, 'no_abort': True, 'rust_ok': True, 'turn_stat_lines': 4, 'tok_s_values': [22.5, 27.0, 26.8, 15.0], 'perf_ok': True, 'contains_think': False, 'orphan_think_close': False, 'contains_unk': False, 'contains_pad': False}`
- `Qwen3-8B` Paris: `Paris`; serve multi-turn: `蓝色月亮`; serve stream chunks: `8`; run multi-turn: `['已记住。', '蓝色月亮']`; run tok/s: `8.051250423198164`; run text long: `{'passed': True, 'rc': 0, 'no_abort': True, 'rust_ok': True, 'turn_stat_lines': 4, 'tok_s_values': [15.5, 25.3, 25.4, 15.4], 'perf_ok': True, 'contains_think': False, 'orphan_think_close': False, 'contains_unk': False, 'contains_pad': False}`
- `Qwen3-30B-A3B` Paris: `Paris`; serve multi-turn: `蓝色月亮`; serve stream chunks: `8`; run multi-turn: `['已记住。', '蓝色月亮']`; run tok/s: `4.716931464439984`; run text long: `{'passed': True, 'rc': 0, 'no_abort': True, 'rust_ok': True, 'turn_stat_lines': 4, 'tok_s_values': [12.0, 40.9, 45.8, 21.3], 'perf_ok': True, 'contains_think': False, 'orphan_think_close': False, 'contains_unk': False, 'contains_pad': False}`

Notes:
- Performance gate is `current >= 0.90 * README baseline`, plus all requests completed.
- Default startup config must be captured without benchmark CLI overrides and must expose enough sequence slots for the release cell.
- Throughput-profile startup config must expose enough sequence slots for the measured concurrency cell.
- The stateful loop probe sends multiple short prompts through one server process and rejects repeated-prefix/length regressions.
- Every throughput cell first runs a marker/square concurrent quality probe; HTTP 200 and zero request errors are not sufficient correctness evidence.
- The quality probe hard gate is marker/checksum isolation with no crosstalk or length finish; exact two-line formatting is recorded as diagnostic.
- Metal MoE release evidence requires a multi-sequence content-quality and throughput cell.
- Throughput cells use canonical `ferrum bench-serve` with streaming usage token accounting.
- Throughput cells record their input/output token workload and the server CLI profile in the artifact directory.
- `run` coverage includes JSONL multi-turn plus default text-mode long multi-turn to catch streaming and KV overflow regressions.
- This runner records correctness failures and still collects performance data.
- Active swap means results are release-regression evidence, not clean marketing numbers.
