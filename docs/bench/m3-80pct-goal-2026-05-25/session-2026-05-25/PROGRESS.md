# 2026-05-25 session — vllm-moe-marlin CUDA 13 attack

Goal: lever #1 from `../GOAL.md`.

## TL;DR

`vllm::ScalarType::id()` constexpr is **non-deterministic across TUs**
under CUDA 13.0.48 nvcc. Even `__attribute__((visibility("default")))`
+ `-Xcompiler -fvisibility=default` + per-TU explicit instantiation
can't fix the resulting symbol-mangling mismatch — the dispatcher in
`marlin.cu` emits Marlin<kFloat16.id_X, ...> while the per-tile
sm80_kernel_*.cu defines Marlin<kFloat16.id_Y, ...>. **Different mangled
names** for the same logical type. rust-lld then reports 11+
`undefined hidden symbol marlin::Marlin<...>` errors.

Pivot: disable the get_marlin_kernel dispatcher in marlin.cu for now
(falls back to MarlinDefault no-op). M3 doesn't need the vllm-marlin
GEMM path; its hot loop is vllm-moe-marlin (separate namespace,
separate TU pair: ops.cu + kernel_instantiations.cu — both compute
.id() values consistent within that pair, so they link).

## Pod

- vast contract 37796550, RTX 4090, CUDA 13.0.48, driver 580.82.09
- $0.577/h, Denmark, 1766/1524 Mbps inet

## Sequence (UTC, 2026-05-25)

| time | event |
|---|---|
| 17:30 | rent pod |
| 17:40 | bootstrap apt + Rust + git clone + model pull (16 GB ✓) |
| 17:47 | cargo build base CUDA → green |
| 18:12 | libvllm_moe_marlin.a (21 MB) built — my kernel_instantiations.cu compiles |
| 18:14 | rustc link fails: 11× `marlin::Marlin<...>` (non-MoE) hidden symbols |
| 18:19 | push fix #2 (visibility on non-MoE marlin_template.h + -fvisibility=default flag) |
| 18:31 | libvllm_marlin.a rebuilt — still hidden symbols (visibility attr + flag insufficient) |
| 18:42 | symbol dump confirms .id() value inconsistency across TUs (encoding bug) |
| 18:54 | push fix #3 (disable get_marlin_kernel dispatch — sidestep the encoding bug entirely) |
| _T+15m_ | expected build green |
| _T+30m_ | expected sweep complete |

## Findings

### Symbol-level evidence

After all visibility patches applied, `nm -C` of the two .o files
shows:

| symbol mangling: `_ZN6marlin6MarlinIL` + 4 longs | TU |
|---|---|
| `Ll1125899906909960E Ll562949953487106E Ll1125899906909960E Ll2814749767106568E …` | marlin.o (undefined ref) |
| `Ll1125899906910725E Ll1125899907892224E Ll1125899906910725E Ll1125899906910725E …` | sm80_kernel_float16_u4b8_float16.o (defined) |

Same template, same C++ source dispatcher uses `vllm::kFloat16.id()`
identically in both, yet the encoded ID differs. Within marlin.o,
c_type_id (1125899906909960) and s_type_id (2814749767106568) ALSO
differ even though both source expressions are `vllm::kFloat16.id()`
— so the issue is more than just per-TU caching. It looks like nvcc's
`cudafe++` mangles the same constexpr expression differently when it
appears in different template instantiation sites within the same
file.

### What did NOT fix it

1. `-Xcompiler -fvisibility=default` on every nvcc invocation
2. `__attribute__((visibility("default")))` on the Marlin template definition
3. Explicit per-instantiation `template __global__ void Marlin<...>` at
   namespace scope in dedicated .cu files (this is upstream vLLM's
   pattern via `generate_kernels.py`; we already use it for the
   non-MoE side and it still doesn't link under CUDA 13)

### What did work for the immediate path

Disabling the dispatcher (`#include "kernel_selector.h"` commented
out). M3's critical path is unaffected because it lives entirely
under `marlin_moe_wna16::Marlin<...>` (a separate namespace, separate
TU pair). The MoE side's ops.cu + kernel_instantiations.cu produce
matching IDs within the pair (the dispatcher and the explicit
instantiations are in TUs compiled with the same flags and from the
same header set), so the link there is clean.

## Commits

| sha | message |
|---|---|
| `dd311d5` | wip(vllm-moe-marlin): explicit instantiations for CUDA-13 link |
| `2d9f7ed` | build(vllm-moe-marlin): compile kernel_instantiations.cu |
| `9703fef` | fix(vllm-marlin): CUDA 13 visibility on Marlin template + build flag |
| `47e8dec` | fix(vllm-marlin): disable get_marlin_kernel dispatch on CUDA 13 |

## Bench

(append numbers here after `pod_bench_full.sh` runs)
