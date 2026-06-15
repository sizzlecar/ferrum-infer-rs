# W2 Prompt-Token Admission c16 A/B Diagnostic

- Artifact: `docs/goals/model-coverage-2026-06-12/artifacts/w2_prompt_token_admission_c16_ab_2026-06-15/`.
- Instance: Vast `40826362`, 1x RTX 4090. Shutdown verified with `cur_state=stopped`, `actual_status=exited`.
- Remote worktree: clean checkout of `2f73213181475ba4bdff3e907e45182c24981a0e`.
- Binary SHA256: `551f83921ea1fb6eb0cfb75170fc2325e31d887530ba084ab72ef77b238ebaf0`.
- Scope: diagnostic only. This is not release-grade evidence and did not produce `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`.

Product correctness:

- CUDA release build rc `0`.
- `ferrum run` rc `0`, answer content `5`, n_tokens `3`.
- `ferrum serve` chat smoke rc `0`, content `5`, usage `prompt_tokens=23`, `completion_tokens=3`.
- `bench-serve --fail-on-error` rc `0`, c16 completed `16 / 0 errored`, `0 bad_output`, output token count source `usage`.
- Both `run` and `serve` decision traces selected `scheduler_admission_policy=prompt_token_estimate` from `default`.

Diagnostic performance:

- Ferrum c16 baseline from `w2_vllm_sharegpt_baseline_probe_2026-06-15`: `340.003 tok/s`, p50 TTFT `887.683ms`, p50 TPOT `32.817ms`.
- New c16 result: `344.714 tok/s`, p50 TTFT `931.776ms`, p50 TPOT `31.592ms`.
- Change vs Ferrum baseline: `+1.39%` output throughput. This is below the threshold for a meaningful performance claim and was single-run diagnostic only.
- Ratio vs same-host vLLM c16 baseline `518.796 tok/s`: `66.4%`, still well below the W2 80% release-grade line.

Conclusion:

Default prompt-token admission is now wired through the product path and validated by decision traces, but it is not the main c16 bottleneck. Continue on the decode/Marlin tail MLP side unless a later profiler shows a different first-token issue.

Note:

- The remote driver had a postprocess-only schema bug after the benchmark finished: it expected `bench-serve` JSON to be a list for the single-c case, but the file was a dict. Build, run, serve smoke, and bench rc files were already `0`; `summary.json` was regenerated from the benchmark JSON and records `PASS_CORE_WITH_POSTPROCESS_WARNING`.
