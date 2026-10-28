# SM89 and SM120 W8A8 native boundary

This optional ABI accepts BF16 row-major A[M,K], E4M3FN row-major
weights W[N,K], one positive finite F32 weight scale, and BF16 row-major
output Y[M,N]. K and N are positive multiples of 16; M can have a tail.
On SM120 it uses a fixed CUTLASS 128x128x128 tile, cluster 1x1x1, FP32 accumulation
and persistent scheduling without split-K. The device is physical SM120;
the separately typed build target is `sm_120a`, requiring CUDA 12.8 or later.
It is an independent numerical profile, not deployment bitwise parity or a
measured performance claim.

SM89 (including L20) uses Ada FP8 Tensor Cores with a 128x64x128 tile,
64x32x128 warp tile and three stages, compiled for `sm_89` with CUDA 12.4.
It does not use the SM120 collective or emulate FP8 with BF16 GEMM.
Its FP32 accumulator matrix is explicitly included in the caller's scratch
reservation; a following kernel applies the scalar weight scale and per-row
activation scale, in that order, before rounding to BF16. No split-K or
atomic output reduction is used. Architecture-specific kernel IDs and the
linked artifact distinguish the prepared implementations.

Each row is quantized with 256 threads and a fixed max tree. Its F32 scale is
`max(RN32(amax / 448), RN32(1 / (448 * 512)))`; a zero row uses that positive
floor, not one. Each FP32 quotient is clamped with CUDA `fminf`/`fmaxf`, then
converted with E4M3 RNE/SATFINITE. The epilogue multiplies the FP32 accumulator
by the scalar weight scale, then by the row scale, and finally rounds to BF16.
Finite inputs and positive finite weight scales define the accuracy test
domain. Nonfinite inputs follow these actual CUDA expressions and are not
given an idealized NaN-propagation or accuracy guarantee.

All five exact byte spans, stream, current context and native opaque capacity
belong to the caller. Scratch contains A's E4M3 bytes, aligned F32 row scales
and the queried CUTLASS workspace. Nothing allocates CUDA payload storage here.
The scale payload is not copied or inspected by host validation; the typed
weight source is responsible for its positive-finite contract. The raw native
boundary confers no Ferrum admission authority. Every run error may follow
enqueue and therefore requires the usual fence/drain/quarantine ownership.
Static CUDA registration can remain loaded until context destruction, so its
opaque claim must outlive a provider and follow the context. Capture and the
implicit default stream are rejected. There is no `cudaSetDevice` fallback.

## Source and build evidence

CUTLASS is fixed to `cb4247394dd82148787aed73e5dc7cef33cbf862` (v4.7.1),
under BSD-3-Clause. The official codeload archive SHA256 is
`7a37329b43a47aaf9ef142be99baff3e71dbeea007b975891350cc0e50bc598b`.
Keep the downloaded tree and all generated artifacts outside the repository.
The build source root contains the Ferrum headers/source under `kernels/`, and
that exact CUTLASS revision under `cutlass/`.

`crates/ferrum-kernels/native/cutlass-fp8/cutlass-fp8.seed.json` is a **seed**, not an observed
dependency closure or a build receipt. On the intended CUDA host, obtain an
NVCC `-M` depfile for `kernels/boundary.cu`, using the seed's includes, C++17,
O3, relaxed constexpr, extended lambda, PIC and `-arch=sm_120a`, without fast
math. Then run the Rust builder's `lock-source-dependencies` subcommand with
explicit `--definition`, `--source-root`, `--depfile`, `--out-definition` and
`--out-plan` paths. Outputs must be new. It imports only compiler-observed
headers inside the declared source roots, while retaining all required seed
headers; symlink escapes and undeclared source directories are rejected.

Use the resulting locked plan with the existing source-build, toolkit/host
manifest, package and operator-set workflows. The normal build must reproduce
the dependency closure and validate every source/toolchain hash. Discovery
alone does not authenticate external compiler headers or prove a successful
build. Package with `../package-definitions/cutlass-fp8.json` and the live
provider catalog. For the first package, the CPU-only Rust example
`cutlass_fp8_catalog_declaration <opaque-bytes> <maximum-plans> <opaque-profile>
<catalog-output>` emits canonical JSON to a new file without loading CUDA.
Its declaration hashes the real operation contract and shared provider/native
source bytes plus the explicit configuration. This source provenance is not
an installed-artifact or execution certificate; the running provider separately
binds the actual compiled artifact into its prepared implementation identity.
Enable Cargo feature `cuda-cutlass-fp8` only with that resolved
artifact set; no legacy GEMM artifact is substituted. Actual CUDA compilation,
workspace extent, repeated-result determinism and numerical accuracy require
separate hardware evidence.

For SM89, use `cutlass-fp8-sm89.seed.json`, `-arch=sm_89` and
`-DFERRUM_CUTLASS_FP8_SM89=1`, and package with
`../package-definitions/cutlass-fp8-sm89.json`. The shared
`release_cuda_native` Rust example preserves the historical text-operator
set and adds FP8 using an explicit `--architecture sm89` or `sm120` target.
Linux and Windows SM89 distributions enable `cuda-cutlass-fp8`; runtime
CUDA/cuDNN requirements still apply. Compilation alone does not establish
full-image correctness or throughput on L20.
