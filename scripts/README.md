# Scripts inventory and gate ownership

This directory keeps release gates, CUDA/M3 runners, benchmark utilities, and microbenches. Do not replace the canonical runners with a new all-in-one harness.

## Release gates

- `scripts/release/g0_source_gate.sh`: source gate wrapper for unit, Metal, CUDA smoke, CUDA full, and all-source lanes.
- `scripts/release/validate_metal_readme_regression.py`: hard validator for `scripts/metal_readme_regression.py` artifacts.
- `scripts/release/release_binary_gate.py`: official release tarball and Homebrew binary/fetch gates.
- `scripts/release/g0_release_summary.py`: release gate summary aggregator.
- `scripts/release/inventory_tree.py`: crates/docs/scripts inventory generator required before cleanup.
- `scripts/release/selftest_g0_validators.py`: tiny positive/negative selftest for the release validators. It does not run models or GPUs.

## Canonical runners

- `scripts/metal_readme_regression.py`: Metal source regression runner for README models, correctness, multi-turn, stream, concurrency throughput, and swap evidence.
- `scripts/m3_ab_runner.py`: CUDA M3/Qwen3-30B-A3B runner. Reuse this for CUDA source correctness/performance gates.
- `scripts/m3_validate_runner_artifact.py`: validator for M3 runner artifacts.
- `ferrum bench-serve`: canonical HTTP performance client for `/v1/chat/completions`.

## FA2 special gates

- `scripts/m3_fa2_direct_ffi_ab.sh`: FA2 direct/source A/B wrapper. Use only for FA2/direct/source path changes.
- `scripts/m3_fa2_source_allcells_ab.sh`: retired FA2-source wrapper. It now fails clearly until a source-owned FA2 kernel has release evidence.
- `scripts/microbenches/build_fa2_ferrum_source_shim.sh` and related `fa2_*` probes: microbench/build aids, not release gates by themselves.

## Benchmark/report utilities

- `scripts/bench_chat_completions.sh`: lightweight HTTP smoke/throughput utility.
- `scripts/bench_vs_vllm.sh`: apples-to-apples ferrum/vLLM comparison helper.
- `scripts/aggregate_m3_80pct.py`, `scripts/m3_collect_allcell_runner_artifacts.py`, `scripts/compare_bench.py`: report aggregation/validation helpers.

## GPU setup/pod helpers

- `scripts/pod_session_m3_80pct.sh`, `scripts/pod_bench.sh`, `scripts/pod_collect_m3_80pct.sh`: pod/session helpers.
- `scripts/validate_cuda_build_summary.py`, `scripts/validate_cuda_build_boundary_manifest.py`, `scripts/m3_cuda_build_boundary_probe.py`: CUDA build/cache validation aids.

## Microbenches

Files under `scripts/microbenches/` are experiment or kernel-probe utilities. They may support a performance investigation, but they are not release gates unless a release document explicitly references their output.

## Cleanup rule

Before moving, deleting, or archiving files under `scripts/`, run:

```bash
python3 scripts/release/inventory_tree.py --out docs/release/cleanup/<YYYYMMDD>-inventory.md
```

Commit the inventory with the cleanup PR. Do not delete evidence without a reference-count audit.
