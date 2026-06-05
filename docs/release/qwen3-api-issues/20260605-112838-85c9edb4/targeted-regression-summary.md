# Targeted Qwen3 API + Metal run regression

- Release gate: not a G0 release gate; no G0 PASS claimed
- API final: passed=True, cases=5, dir=docs/release/qwen3-api-issues/20260605-112838-85c9edb4/api-after-tool-fallback
- Run multiturn final: passed=True, assistant_count=4, tok_s=30.869, dir=docs/release/qwen3-api-issues/20260605-112838-85c9edb4/run-multiturn-final
- Run REPL UX targeted: passed=True, cases=2, dir=docs/release/qwen3-api-issues/20260605-112838-85c9edb4/repl-tty-ux
  - tty_unicode_backspace: Qwen/Qwen3-0.6B Metal PTY, input `会变` + Backspace + Backspace + `ok`, JSONL user content=`ok`
  - tty_first_token_indicator_compact: Qwen/Qwen3-0.6B Metal PTY text mode, saw compact `Working (` indicator and no old long `waiting for first token...` text
