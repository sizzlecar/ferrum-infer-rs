# W2 dynamic active-decode prefill source checkpoint

This checkpoint replaces the diagnostic fixed two-chunk active-decode prefill guard with a dynamic mixed-prefill token budget.

- Status: source checkpoint only; this did not produce `MODEL_RELEASE_GRADE_W2 PASS`.
- Main change: active-decode prefill budget now scales by same-iteration batch slot headroom instead of `active_decode_prefill_chunk * 2`.
- Rationale: vLLM schedules from a per-step token budget rather than a fixed number of prefill chunks; this keeps Ferrum's Gemma3 guard conservative while avoiding a hard-coded c16-specific multiplier.
- Validation:
  - `cargo test -p ferrum-scheduler active_decode_prefill -- --nocapture`
  - `cargo fmt --all -- --check`
  - `cargo test -p ferrum-scheduler`
- Result:
  - targeted scheduler tests: 3 passed
  - ferrum-scheduler crate tests: 53 passed
  - rustfmt check: passed

GPU performance evidence must be regenerated for this dynamic policy before any W2 performance conclusion.
