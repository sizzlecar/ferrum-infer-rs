# S1 CUDA capacity diagnostic: REJECT

- Lane: `vnext-s1-cuda-capacity`
- Source: `47b34c78d2166f618ee025c107640474d37e02c2` (clean)
- Binary SHA256: `0a155dcb1b540cb18505d9f110baffcfd8d6dd8b8bb5d00dbf31a0fee5695586`
- Hardware: 1x RTX 4090, 23028 MiB, driver 595.45.04
- Model: `Qwen/Qwen3.5-4B` at revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
- Result: `REJECT exact_budget_replay_prompt_mismatch`

The preceding staged-authority fix passed a real CUDA `ferrum run`: it returned `Paris`,
submitted 131 vNext operations, and completed the request. The old self-stale epoch
livelock did not recur.

The full collector then rejected the `serve` warmup. Calibration used model-visible
labels `calibration-A/C` and consumed 32 prompt tokens, while exact-budget replay used
`warmup-A/C` and consumed 33. The extra token made dynamic pool domain 2 retain 110,992
additional bytes. The later 496,640-byte domain 3 growth therefore had only 385,648
bytes left under the calibrated 8,474,209,616-byte ceiling and returned a typed budget
error. This was a mismatched-workload gate defect, not CUDA OOM or allocator failure.

The fix must keep model-visible A/B/C workload text stable across calibration, warmup,
and pressure phases while retaining distinct artifact labels outside the prompt. It must
not add arbitrary budget slack. The next-run signal and reject threshold are recorded in
`reject-summary.json`.

The canonical external artifact is
`/Users/chejinxuan/ferrum-bench/artifacts/runtime-vnext-s1/vnext_capacity_47b34c78_20260715T225200Z.tar.gz`
(`393c1ad615e2f5d3ad5fdf46de74972d45019ec1446cc930cef92c7aa2b10685`, 135,198 bytes).
The repository retains only this conclusion and compact machine-readable summary.
