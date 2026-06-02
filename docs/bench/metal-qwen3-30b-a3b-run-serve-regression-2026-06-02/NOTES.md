# Metal Qwen3-30B-A3B run/serve regression - 2026-06-02

Evidence from product-default Metal GGUF Qwen3-30B-A3B after the sampling-token mask and paged single-slot reset fixes.

- `ferrum run` JSONL multi-turn: passed, secret recall correct, no `<unk>`, no `[PAD...]`, no panic/KV overflow.
- `ferrum run` default text REPL four-turn: passed, four tok/s lines recorded, no `<unk>`, no `[PAD...]`, no panic/KV overflow.
- `ferrum serve` default sequence: Paris, multi-turn, and stream passed.
- Runner now treats `<unk>` and `[PAD...]` as failures and records run tok/s.
- Qwen3-30B-A3B README c16 serving performance did not pass under the current correctness-safe default; the c16 bench timed out after 420s. Do not use the old c16 README row as release evidence until Metal multi-slot GGUF MoE paged-KV correctness is fixed.

Artifacts:

- `summary.md` / `summary.json`: targeted runner output.
- `run-summary.json`: direct `ferrum run` JSONL/text evidence.
- `serve-sequence-summary.json`: direct default serve Paris/multi-turn/stream evidence.
