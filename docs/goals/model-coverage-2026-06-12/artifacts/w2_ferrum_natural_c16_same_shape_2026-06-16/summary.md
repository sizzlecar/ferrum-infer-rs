# W2 Ferrum Natural ShareGPT c16 Same-Shape Validation

- Artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_ferrum_natural_c16_same_shape_2026-06-16/`
- Lane: W2 Gemma3 CUDA c16 same-dataset, same-shape minimal validation.
- Hardware: existing Vast instance `40826362`, 1x RTX 4090.
- Source checkpoint: local source head `a45e3caaeb94af5451c64f7542014e580ea613e6`.
- Local tracked source status: `tracked_dirty_lines=0`.
- Binary SHA256:
  `79379516dc90c958ae03f65aeaa36b706156b5ec1f6e15e14092815f4d62a110`.
- Dataset SHA256:
  `58d5721d8389d7ed9ec4b8b2dbd8797faa61641c6ba023dd150a1a9d93c0a01e`.
- Vast cleanup: `actual_status=exited`.

## Correctness Gate

- `ferrum run` returned assistant content `5`, finish_reason `stop`,
  `n_tokens=3`.
- `scripts/model_coverage_smoke.sh gemma3:27b-gptq` printed:
  `FERRUM W1 SMOKE PASS: gemma3:27b-gptq`.

## Performance Command

```text
target/release/ferrum bench-serve \
  --base-url http://127.0.0.1:8461 \
  --model gemma3:27b-gptq \
  --tokenizer /root/.cache/huggingface/hub/models--circulus--gemma-3-27b-it-gptq/snapshots/70d89a3a6b401b5f56558cb5d4c0f1fd158980b2 \
  --dataset sharegpt \
  --sharegpt-path /workspace/ferrum-release-c16-a45e3caa-min/docs/goals/model-coverage-2026-06-12/artifacts/w2_natural_prompt_baseline_probe_2026-06-15/dataset/ascii_sharegpt.jsonl \
  --random-output-len 128 \
  --concurrency-sweep 16 \
  --num-prompts 100 \
  --n-repeats 3 \
  --fail-on-error \
  --require-ci \
  --seed 9271
```

## Result

- Ferrum c16 completed_per_run: `[100, 100, 100]`.
- Ferrum c16 errored_per_run: `[0, 0, 0]`.
- Ferrum c16 quality counts: all zero.
- Output token count source: `usage`.
- Ferrum c16 throughput: `332.005 +/- 6.821 tok/s`.
- Ferrum c16 LCB: `325.184 tok/s`.
- vLLM c16 same-dataset baseline mean: `530.829 tok/s`.
- vLLM c16 same-dataset baseline LCB: `491.150 tok/s`.
- Required 80% of vLLM LCB: `392.920 tok/s`.
- Ferrum LCB / vLLM LCB: `66.21%`.
- Gap to 80% LCB threshold: `67.736 tok/s`.
- Ferrum p95 ITL: `83.979 ms`.
- vLLM p95 ITL: `28.130 ms`.
- p95 ITL multiple: `2.99x`.

## Interpretation

This artifact closes the earlier evidence gap between 32-prompt diagnostics and
the release-shaped vLLM baseline. Correctness is clean, but c16 still fails both
the release throughput threshold and the p95 ITL threshold. W2 remains not
release-grade; no `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>` was produced.
