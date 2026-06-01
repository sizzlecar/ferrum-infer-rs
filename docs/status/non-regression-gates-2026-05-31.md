# Correctness and Performance Non-Regression Status - 2026-05-31

Milestone I is not complete. This checkpoint tightens the artifact schema so
API-only changes cannot skip M3 performance evidence without a
machine-readable reason.

## Added

- `scripts/m3_ab_runner.py` now accepts optional
  `validation.benchmark_impact` metadata and copies it into each generated
  `validation_checklist`.
- `scripts/m3_validate_runner_artifact.py` validates the
  `benchmark_impact` object:
  - `m3_benchmark_exercised` must be boolean;
  - `reason` and `evidence` must be non-empty strings.
- Publishable `change_type=api_only` artifacts with
  `performance_regression_required=false` now require `benchmark_impact`.
  If `m3_benchmark_exercised=true`, the validator rejects the artifact and
  requires a real performance-regression gate.
- `scripts/m3_collect_allcell_runner_artifacts.py` aggregates child
  `benchmark_impact` entries into the all-cell manifest so full-sweep packets
  keep the same checklist evidence.

## Validation

```bash
python3 -m py_compile \
  scripts/m3_ab_runner.py \
  scripts/m3_validate_runner_artifact.py \
  scripts/m3_collect_allcell_runner_artifacts.py
python3 scripts/m3_ab_runner.py --self-test
python3 scripts/m3_validate_runner_artifact.py --self-test
python3 scripts/m3_collect_allcell_runner_artifacts.py --self-test
```

## Remaining I Gaps

- No new GPU performance claim is made by this checkpoint.
- 2026-06-01 source-FA2 opt-in all-cell evidence exists at
  `/workspace/m3-fa2-source-current-allcells-n3-20260601`: c=1/4/16/32,
  `n_repeats=3`, Paris, multi-turn, three-turn recall, bench completion, and
  artifact validation all passed. This closes the missing source-FA2
  confirmation packet but does not close final default-path/non-regression
  completion.
- Default-path changes still need a same-pod c=1/4/16/32 full sweep with
  `n_repeats >= 3` and the existing correctness gates before a default-path
  completion claim.
- Multi-turn correctness coverage now includes a three-user-turn recall gate
  for runner artifacts that enable `multi_turn=true`; the gate requires the
  third response to preserve both `Paris` and the checkpoint token `basalt`.
- Native source FA2 still needs either more performance work or a different
  attention path before it can support an M3 80% completion claim; c32 remains
  `1488.08 tok/s`, about `0.754×` of vLLM.
- The final goal packet still needs one committed correctness and performance
  non-regression report tying local gates, GPU gates, artifact validator
  output, and exact commands together.
