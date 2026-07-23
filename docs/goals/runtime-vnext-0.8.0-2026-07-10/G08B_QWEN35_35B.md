# G08B: Qwen3.5-35B-A3B Hybrid-MoE 迁移

## 依赖与目标

- 依赖：G08A
- 下游：G08C
- 目标：在 M1 hybrid 基础上加入 256-expert/shared-expert MoE、GPTQ/Marlin 和高压资源路径。

## 必需交付

- CUDA official GPTQ-Int4 full product path。
- Metal Q4_K_S full product path，固定 32GB M1 Max，>=2 GiB 实测 headroom、swap growth 0，无 waiver。
- requested/effective concurrency 分离；CUDA client c32、typed active cap至少 16，并记录
  observed max-active；Metal required client c1/4/16、typed active floor `4`。CUDA/Metal 最高
  cell 的 eligible interval active duty-cycle 均须 `>=0.80`。
- recurrent + KV + scratch 多资源事务 fault grid。
- G00 legacy CUDA binary parity；Metal 使用 HF/CPU + same-GGUF llama.cpp new-lane reference。
- 删除全部 Qwen3.5 family legacy runner/factory/arch-named adapter，包括 G08A test-only adapter。

## 验收

- M2 CUDA/Metal C01-C21 `2/2 PASS`。
- Metal op/layer/full-vocab-logit/token reference 全部满足 MODEL_MATRIX 固定数值门，并绑定 checked-in
  tolerance blob/row；missing/post-hoc-widened tolerance 数量 `0`。
- G02 Qwen3.5 resource/output historical mutations kill `100%`。
- Qwen3.5 family legacy production/test adapter 数量 `0`。
- Qwen3.5 架构专属执行脚手架相对 G00 减少 `>=60%`。
- CUDA client c32/admission-cap 路径资源终态正确，OOM/livelock/leak `0`。
- G08 统一 performance smoke：CUDA `>=0.90x` G00 legacy，Metal `>=0.70x` same-host
  llama.cpp；两者都只作 diagnostic，完整正式门留给 G09。

```text
FERRUM RUNTIME VNEXT G08B QWEN35 35B A3B PASS: <out_dir>
FERRUM GATE vnext-g08b PASS: <out_dir>
```

## CUDA Correctness Checkpoint - 2026-07-23

Clean frozen source `6fa8e21514bcb602e5d21aa2fa367c55159c6d8e` and source tree
`2011052b234ff313fd98eed1c7cf3187172014bb` completed the M2 CUDA C01-C21 matrix on
one RTX 4090. The bound release binary SHA256 was
`4a580c6b3513716c22ae57fc3268728bedfa9d250515b202723260698d17b12b`.

- cases: `703/703 pass`, unexpected/error/known-fail/blocked `0`;
- product command groups: five resident `ferrum run` groups and two isolated
  `ferrum serve` sessions;
- bounded duration: `4817.444734s`;
- bounded peaks: processes `4`, process-group threads `102`, per-process threads `67`;
- driver stderr bytes: `0`.

The canonical runner and unified checkpoint printed:

```text
FERRUM RUNTIME VNEXT G08 MODEL MATRIX SCENARIOS PASS: /workspace/ferrum-artifacts/runtime-vnext-g08b-cuda-matrix-6fa8e215-20260723T033143Z/correctness/m2-qwen35-35b-a3b/cuda/scenario-report.json
FERRUM RUNTIME VNEXT G08B CUDA MODEL MATRIX PASS: /workspace/ferrum-artifacts/runtime-vnext-g08b-cuda-matrix-6fa8e215-20260723T033143Z/gate-vnext-g08b-cuda
FERRUM GATE vnext-g08b-cuda PASS: /workspace/ferrum-artifacts/runtime-vnext-g08b-cuda-matrix-6fa8e215-20260723T033143Z/gate-vnext-g08b-cuda
```

The complete compressed artifact is stored in the temporary
[GitHub transfer release](https://github.com/sizzlecar/ferrum-infer-rs/releases/download/untagged-711d3e8abdfcbe0c8b41/runtime-vnext-g08b-cuda-matrix-6fa8e215-20260723T033143Z.tar.gz)
with SHA256 `3816f1ea3f696bb3595bd8319cf070d02cabaf7490381a10708211c9df50b2ea`.
The verified local compressed copy is
`/Users/chejinxuan/ferrum-bench/artifacts/runtime-vnext-20260723/g08b-cuda-matrix-6fa8e215-github/`;
it was not expanded because local disk space is constrained.

Vast instance `45319871` was stopped only after GitHub upload, local download, and
SHA256 verification. Reconciled state is `cur_state=stopped`,
`actual_status=exited`, and potentially billable sibling count `0`; the stopped
instance retains model/build caches.

This checkpoint is historical/intermediate after later runner and test-policy changes.
It proves M2 CUDA product correctness at `6fa8e215`; it does not prove current-HEAD
freshness or complete G08B. Metal Q4_K_S correctness, CUDA/Metal performance smoke,
legacy/reference parity, historical mutation plus legacy-deletion acceptance, and the
final G08B aggregate remain open.
