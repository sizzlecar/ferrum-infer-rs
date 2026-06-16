# W2 Scheduler Same-Iteration Admit Source Checkpoint

- Date: 2026-06-17
- Scope: source checkpoint only; no paid GPU run was started for this patch.
- Previous CUDA evidence:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_active_chunk_sharegpt_c16_ci_2026-06-17/`
- vLLM comparison:
  - `vllm/v1/core/sched/scheduler.py` schedules running requests first,
    then spends remaining `token_budget` on waiting requests in the same
    scheduler step.
  - Ferrum previously admitted waiting requests after collecting decode and
    existing prefill work, but only scheduled newly admitted prefills in the
    same iteration when the batch was otherwise empty.
- Source change:
  - factor prefill collection into `add_prefill_requests_to_batch`;
  - track scheduled request IDs to avoid duplicate same-iteration scheduling;
  - after waiting admission, schedule newly admitted prefills with remaining
    batch slot and token budget even when decode work is already present.
- Expected effect:
  - reduce one-iteration delay for closed-loop replacement requests;
  - target the remaining c16 TTFT/ITL tail and the roughly 6.46 tok/s gap to
    the historical 80% vLLM LCB threshold.
- Validation:
  - `cargo fmt --all -- --check` PASS
  - `cargo test -p ferrum-scheduler newly_admitted_prefill_uses_remaining_budget_with_decode -- --nocapture` PASS
  - `cargo test -p ferrum-scheduler active_decode_prefill_chunk_only_caps_when_decode_is_active -- --nocapture` PASS
  - `cargo test -p ferrum-scheduler continuous -- --nocapture` PASS
  - `cargo test -p ferrum-scheduler --lib` PASS
  - `cargo check -q -p ferrum-scheduler -p ferrum-engine -p ferrum-cli` PASS
  - `git diff --check` PASS
- Status:
  - no `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>` was produced;
  - next required checkpoint is a 1x4090 c16 ShareGPT CI diagnostic with
    product `ferrum run` and `ferrum serve` correctness before performance.
