# 2026-05-25 session — vllm-moe-marlin CUDA 13 attack

Goal: lever #1 from `../GOAL.md`.

## Findings

### Diagnosis confirmed

CUDA 13's nvcc emits templated `__global__` kernel host stubs with
hidden ELF visibility by default. Two distinct manifestations:

1. **vllm-moe-marlin** — `ops.cu` does implicit instantiation only;
   even with `__attribute__((visibility("default")))` on the template,
   implicit instantiations triggered inside the dispatcher's macro
   chain end up hidden. **Fix**: a sibling `kernel_instantiations.cu`
   that explicitly instantiates the same set at namespace scope (in a
   separate TU). External linkage forces the symbol to be exported.

2. **vllm-marlin (non-MoE)** — already uses upstream-aligned
   `generate_kernels.py` per-instantiation `sm80_kernel_*.cu` files
   with explicit `template __global__ void Marlin<...>` declarations.
   The Marlin template definition (in `marlin_template.h`) was
   MISSING the `visibility("default")` attribute that the MoE version
   carried. And `compile_vllm_marlin()` in build.rs was MISSING the
   `-Xcompiler -fvisibility=default` flag that the MoE compile path
   had. Both gaps combined: explicit instantiations still got hidden
   visibility. **Fix**: add attribute + flag (mirroring the MoE side).

These two bugs co-located but unrelated.

## Sequence (2026-05-25 UTC)

| time | event |
|---|---|
| 17:30 | Rent Vast offer 36913047 (RTX 4090 / CUDA 13.0.48 / Denmark / 1766 Mb/s) |
| 17:39 | Bootstrap apt + Rust + git clone + lock GPU + Qwen3-30B-A3B-Int4 pull (16 GB) |
| 17:40 | cargo build base CUDA → green |
| 17:47 | cargo build + vllm-moe-marlin → starts |
| 18:12 | libvllm_moe_marlin.a built (21 MB) — my kernel_instantiations.cu compile OK |
| 18:14 | rustc link of ferrum-cli fails: 11× `undefined hidden symbol marlin::Marlin<...>` — vllm-MARLIN (non-MoE) bug surfaces |
| 18:19 | Push fix #2 (visibility on non-MoE marlin_template.h + -fvisibility=default in build.rs) |
| 18:20 | Rebuild start |
| _T+15m_ | Expected build green |
| _T+30m_ | Bench A/B sweep complete |

## Commits

| sha | message |
|---|---|
| `dd311d5` | wip(vllm-moe-marlin): explicit instantiations for CUDA-13 link |
| `2d9f7ed` | build(vllm-moe-marlin): compile kernel_instantiations.cu |
| `9703fef` | fix(vllm-marlin): CUDA 13 visibility on Marlin template + build flag |

## Bench (TODO once build green)

(append numbers here after `pod_bench_critical.sh` runs)
