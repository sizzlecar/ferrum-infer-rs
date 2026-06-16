# W2 Marlin Cache-Policy ShareGPT Diagnostic

- Artifact: `docs/goals/model-coverage-2026-06-12/artifacts/w2_marlin_cache_policy_sharegpt_diag_2026-06-16/`.
- Lane: W2 current Ferrum ShareGPT same-dataset diagnostic after Marlin
  evict-first default.
- Instance: Vast `40826362`, 1x RTX 4090.
- Scope: diagnostic only. This used `n_repeats=1` and did not run
  `--require-ci` or the final W2 release-grade validator.
- Final diagnostic line: `DIAGNOSTIC PASS: /workspace/w2_marlin_cache_policy_sharegpt_diag_2026-06-16`.

## Contract

- Expected runtime/cost: 20-40 minutes, about USD 0.14-0.28 at
  USD 0.42488888888888887/h.
- Stop condition: startup/SSH/CUDA/sync/build/serve/bench first failure, or
  diagnostic artifact collected, then stop the instance.
- Correctness gate: prior product `ferrum run`/`ferrum serve` correctness
  artifact plus this run's server readiness, chat smoke, bench rc 0,
  completed requests, zero request errors, and clean server log scan.
- Performance command:
  `ferrum bench-serve --dataset sharegpt --sharegpt-path ascii_sharegpt.jsonl --random-output-len 64 --concurrency-sweep 16,32 --num-prompts 16 --n-repeats 1 --fail-on-error --seed 9271`.

## Evidence

- Remote HEAD: `7d93c2b481cc3a4d9ae794e2d6a66c3e05a55784`.
- Clean remote worktree: `remote/git_status_short.txt` has `0` lines.
- Binary SHA256:
  `d38caf704f252045c29bdfe02795606937f400ab00edef05647da74179b215d5`.
- Server ready: `ready_at_poll=31`.
- Chat smoke: response content `"5"`, usage present.
- Bench rc: `0`.
- Server error scan: `0` lines.
- Vast cleanup: `cur_state=stopped`, `actual_status=exited`.

## Results

Same ShareGPT ASCII dataset and same vLLM baseline as
`w2_vllm_sharegpt_baseline_probe_2026-06-15`.

| Cell | Completed | Errors | Bad output | Ferrum tok/s | vLLM tok/s | Ratio | Delta vs previous Ferrum |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| c16 | 16 | 0 | 0 | 339.9306 | 518.796 | 0.6552 | -0.0723 tok/s (-0.02%) |
| c32 | 16 | 0 | 0 | 340.5554 | 524.128 | 0.6498 | -1.7285 tok/s (-0.51%) |

## Interpretation

This run found no new correctness issue on the product `serve` path, but it
also showed no material endpoint-level performance gain from the Marlin
B-weight evict-first default. The native CUDA microbench improvement was real
inside the tail-MLP segment, but it does not move the full ShareGPT c16/c32
throughput ratio.

W2 remains not release-grade. The final validator has not produced
`MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`, and this diagnostic remains roughly
14-15 percentage points below the 80% same-hardware mainstream baseline target.

Next high-return work should move away from this cache-policy lever and focus on
dense MLP `gate_up`, launch count, and batched decode graph/integration behavior
under product c16/c32.
