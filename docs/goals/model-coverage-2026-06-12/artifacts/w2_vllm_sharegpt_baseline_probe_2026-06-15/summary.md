# W2 vLLM ShareGPT baseline-cleanliness probe

- Artifact: `docs/goals/model-coverage-2026-06-12/artifacts/w2_vllm_sharegpt_baseline_probe_2026-06-15/`.
- Instance: Vast `40826362`, 1x RTX 4090, cache-retained native CUDA machine.
- Scope: diagnostic only. This tested whether vLLM is baseline-clean on natural ASCII ShareGPT prompts after previous random-prompt invalid-UTF8 failures.
- Release status: not release-grade evidence. This did not run `--require-ci`, did not assemble `model_release_grade_manifest.json`, and did not produce `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`.

vLLM baseline:

- Engine: `vllm 0.10.1.1`, `torch 2.7.1+cu126`, GPTQ Marlin.
- Server smoke PASS: `/v1/models` ready, chat smoke returned content `5` with usage.
- Dataset: `ascii_sharegpt.jsonl`, `num_prompts=16`, `n_repeats=1`, `--fail-on-error`, seed `9271`, output len `64`.
- c=16: `16 completed / 0 errored`, `0 bad_output`, `518.796 tok/s`.
- c=32: `16 completed / 0 errored`, `0 bad_output`, `524.128 tok/s`.

Ferrum same-dataset compare:

- Binary SHA256: `3e28a4cf37b2e25b127dbd591e8891b141863d8082d1757486707c785e6869ce`.
- Server readiness PASS, no profile env enabled.
- c=16: `16 completed / 0 errored`, `0 bad_output`, `340.003 tok/s`.
- c=32: `16 completed / 0 errored`, `0 bad_output`, `342.284 tok/s`.

Diagnostic ratio:

- c=16: `340.003 / 518.796 = 0.655`.
- c=32: `342.284 / 524.128 = 0.653`.
- Ferrum is about `34.5%` to `34.7%` behind vLLM on this natural-prompt diagnostic, and about `14.5` percentage points below the 80% release-grade line.

Conclusion:

- No new product correctness issue was found for either Ferrum or vLLM on the natural ShareGPT diagnostic.
- Unlike the earlier random-prompt vLLM c16 baseline, this natural-prompt vLLM probe was zero-error, so it is a plausible baseline route to formalize with N>=3 if the goal accepts this dataset.
- W2 remains not release-grade because c16/c32 ratio is still below 80% and no final validator PASS exists.
