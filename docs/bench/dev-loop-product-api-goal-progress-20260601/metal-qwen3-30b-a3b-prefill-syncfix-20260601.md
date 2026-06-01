# Metal Qwen3-30B-A3B Prefill Readback Sync Fix - 2026-06-01

Purpose: document the local Metal regression discovered while preparing the release.

Code fix:

- Commit: `1e3ce42 fix: sync qwen3 moe prefill logits before readback`
- File: `crates/ferrum-models/src/models/qwen3_moe/prefill_decode.rs`
- Change: call `B::sync(&mut ctx)` before reading `self.scratch.logits` in Qwen3-MoE prefill.

Observed failure before the fix:

- Artifact: `docs/bench/dev-loop-product-api-goal-progress-20260601/metal-qwen3-30b-a3b-20260601/`
- Build: Metal release build succeeded.
- Correctness output was garbage: `褫褫 l l L L ...`.
- Process exited with Metal assertion: `Command encoder released without endEncoding`.

Observed behavior after the fix:

- Artifact: `docs/bench/dev-loop-product-api-goal-progress-20260601/metal-qwen3-30b-a3b-syncfix-20260601/`
- Output: `The capital of France is **Paris**.`
- The Metal encoder assertion did not reproduce.
- The old exact expected sentence did not match, so this is recorded as semantic correctness pass, not exact-string pass.

KV-capacity note:

- Artifact: `docs/bench/dev-loop-product-api-goal-progress-20260601/metal-qwen3-30b-a3b-syncfix-kv1024-20260601/`
- `FERRUM_KV_CAPACITY=512` overflowed for the current tokenizer/prompt shape: `would write tokens [0..513) but capacity is 512`.
- `FERRUM_KV_CAPACITY=1024` avoided the overflow for the smoke.

Performance-note caveat:

- The local Mac had active swap during this check: about `1.1GB / 2GB` swap used.
- `ferrum run --bench-mode` stopped early on EOS for the attempted tg128 prompts, so this local run is not a clean pp512/tg128 performance packet.
- Do not use these local Metal numbers as release performance claims.

Conclusion:

- The release-relevant outcome is the correctness fix: Metal no longer reads uncommitted Qwen3-MoE prefill logits.
- Because this is a runtime code change, GPU quick regression was run and passed in `m3-quick-regress-1e3ce42-c32-20260601`.
