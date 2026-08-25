# First Success bounded cleanup

Inventory:

```text
INVENTORY PASS: /Users/chejinxuan/rust_ws/ferrum-infer-rs/docs/release/cleanup/20260825-first-success-inventory.md
```

## Selection rule

This cleanup removes only process material from completed goals and one-off
diagnostic/generator scripts with no active invocation from GitHub workflows,
`AGENTS.md`, `run_gate.py`, `g0_source_gate.sh`, validator self-tests, Cargo
targets, or current product documentation.

Release evidence, benchmark artifacts, support matrices, final reports, active
G0 scripts, and any script named by a current historical regression contract
remain in place. Reference counts below are active references outside the file
or its completed goal directory; historical artifact command strings are not
treated as active invocations.

## Removed process documents

All entries have active reference count `0` and category
`completed-goal-process-material`:

- `docs/goals/model-coverage-2026-06-12/HANDOFF_W3_QWEN35_20260621.md`
- `docs/goals/model-coverage-2026-06-12/HANDOFF_W3_QWEN35_20260622.md`
- `docs/goals/model-coverage-2026-06-12/HANDOFF_W3_QWEN35_20260622_2H.md`
- `docs/goals/model-coverage-2026-06-12/HANDOFF_W3_QWEN35_CANCELLED_20260622.md`
- `docs/goals/model-coverage-2026-06-12/ACTIVE_PODS.md`
- `docs/goals/model-coverage-2026-06-12/W3_QWEN35_RETROSPECTIVE_20260626.md`
- `docs/goals/model-coverage-2026-06-12/W3_QWEN35_DEEP_REVIEW_ADDENDUM_20260626.md`
- `docs/goals/runtime-vnext-0.8.0-2026-07-10/HANDOFF_2026-08-13.md`
- `docs/goals/runtime-vnext-0.8.0-2026-07-10/S1_STATUS_2026-07-17.md`
- `docs/goals/test-architecture-2026-06-10/HANDOFF.md`

Reason: these are transient handoff, pod-state, review, or status snapshots.
Their durable code, goal contracts, matrices, and evidence remain tracked.

## Removed standalone scripts

All entries have active reference count `0` and category
`superseded-standalone-diagnostic`:

- `scripts/analyze_layer_dump.py`
- `scripts/gemma3_l1_reference.py`
- `scripts/inspect_hf_gptq_tensor.py`

Reason: each is an unreferenced, manually invoked investigation utility from a
completed model-debugging phase.

## Removed W3 one-off generators

All entries have active invocation count `0` and category
`completed-w3-goal-generator`:

- `scripts/release/w3_delta_rule_s0_microbench.py`
- `scripts/release/w3_hf_config_probe.py`
- `scripts/release/w3_l0_template_gate.py`
- `scripts/release/w3_l2_quantized_gate.py`
- `scripts/release/w3_l4_agent_gate.py`
- `scripts/release/w3_l5_concurrency_gate.py`
- `scripts/release/w3_qwen35_cuda_release_lane.py`
- `scripts/release/w3_qwen35_hf_layer_dump.py`
- `scripts/release/w3_qwen35_layer_compare.py`
- `scripts/release/w3_qwen35_real_product_report.py`
- `scripts/release/w3_qwen35_weight_index_probe.py`
- `scripts/release/w3_s0_design_gate.py`

Reason: these scripts generated evidence for the closed W3 model-coverage
goal, but are not called by the current source/release gates. Existing W3
artifacts and the generic manifest validators remain available.

The following similarly named scripts are deliberately retained:

- `w3_deltanet_s1_layer_compare.py`: referenced by the current DeltaNet dump
  implementation contract.
- `w3_qwen35_vast_c32_diagnostic.py` and
  `w3_qwen35_c32_diagnostic.sh`: retained by the historical regression catalog.

## Deferred cleanup

The large historical goal artifact trees remain because current tests and
runtime-vNext validators still reference selected files inside them. Removing
or relocating that evidence requires a separate manifest migration, not a
quick productization patch.
