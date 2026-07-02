# W2 Producer-Touch ShareGPT Endpoint Diagnostic

- Artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_producer_touch_sharegpt_diag_2026-06-16`
- Lane: W2 producer-touch ShareGPT endpoint diagnostic
- Instance: cached Vast 1x RTX 4090 instance `40826362`
- GPU: NVIDIA GeForce RTX 4090, 24564 MiB, driver 565.77
- Binary SHA256:
  `5078ea014ee5299a936de62f34475456f9a3c0500d34ab41a96ebcaf9c69fbd8`
- Dataset:
  `/workspace/w2_natural_prompt_baseline_probe_2026-06-15/dataset/ascii_sharegpt.jsonl`
- Bench command shape:
  `ferrum bench-serve --dataset sharegpt --sharegpt-path ascii_sharegpt.jsonl --random-output-len 64 --concurrency-sweep 16,32 --num-prompts 16 --n-repeats 1 --fail-on-error --seed 9271`
- Bench rc: `0`
- Vast cleanup: `cur_state=stopped actual_status=exited`

## Correctness

- server readiness: PASS
- chat smoke: rc `0`, response content `5`, usage present
- bench c16: `16 completed / 0 errored / 0 bad_output / 0 zero_output`
- bench c32: `16 completed / 0 errored / 0 bad_output / 0 zero_output`
- `output_token_count_source=usage`
- server error scan: `0` lines

## Results

Against the current default Marlin ShareGPT diagnostic:

- c16 producer-touch: `313.3996 tok/s`
- c16 current default: `339.9306 tok/s`
- c16 delta: `-7.80%`

- c32 producer-touch: `348.5895 tok/s`
- c32 current default: `340.5554 tok/s`
- c32 delta: `+2.36%`

Against the existing clean vLLM ShareGPT baseline:

- c16 ratio: `313.3996 / 518.7960 = 0.6041`
- c32 ratio: `348.5895 / 524.1279 = 0.6651`

## Interpretation

The producer-integrated qweight touch does not survive product endpoint
validation as a safe default. It improves c32 slightly, but it regresses c16
materially. Because W2 release-grade requires all relevant cells to move toward
the 80% baseline line, this product prototype should not remain enabled by
default.

This validates the native probe as a useful bottleneck clue, but not as a
default product optimization. The next work should either tune a narrower
variant with direct product c16 evidence, or move to another tail-MLP
work-reduction/fusion lever.

This is diagnostic evidence only: `n_repeats=1`, no `--require-ci`, and no final
release-grade manifest. No `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>` was
produced.
