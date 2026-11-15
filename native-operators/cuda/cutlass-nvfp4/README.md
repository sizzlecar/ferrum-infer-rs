# SM120 packed NVFP4 matrix multiplication

Ferrum-owned C ABI source lives in `crates/ferrum-kernels/native/cutlass-nvfp4`.
It implements E2M1 packed A/W, group-16 E4M3FN scales in 128x4 storage, an
explicit device F32 alpha, and BF16 output. There is no Python, Torch, TVM,
model-specific key lookup, allocation or implicit stream in the GEMM boundary.

The numerical profile is `CUTLASS_NVFP4_NUMERICAL_IDENTITY`. Its fixed tile is
128x128x128 with static persistent scheduling and no Stream-K, AB swap or PDL.
Qualification against another implementation is not established by compilation.
The first native geometry requires K divisible by 128 and N divisible by 8;
the generic operation permits group-16 widths. K tails remain unqualified.
The explicit runtime provider uses the shared bounded static module owner and
ordinary tensor submission/completion path. Its estimator admits exact native
workspace, and commands retain every operand, output and module claim. No
plan cache is created: core-admitted commands own their small prepared records.
Enabling the Cargo feature alone does not route model execution to this provider.

The separate `NVFP4_QUANTIZE_BF16` operation consumes BF16 activations and an
explicit positive finite F32 global scale. It produces packed values and padded
E4M3 scales, without weight access or alpha recomputation. Its required
`CudaRcpApproxFtzV1` profile uses explicit `rcp.approx.ftz.f32` and
`mul.rn.ftz.f32` and `setp.eq.ftz.f32` for the zero-amax branch; production
compilation still disables global fast math. A BF16-subnormal amax therefore
takes the zero branch, as in the independently compiled upstream helper.
Quantization supports K divisible by 16, including scale-column padding, and
requires no scratch or plan cache. The grouped factory gives quantization and
GEMM one shared module owner. The independent diagnostic artifact and its
managed reference provider are described in
[`native/diagnostics`](../../../crates/ferrum-kernels/native/diagnostics/README.md).

The ignored Rust test `cuda_nvfp4_matmul` exercises the managed provider on
SM120 with an independently decoded dyadic scalar oracle, scale-tile row
boundaries, repeated submissions and output retention after session close.
Run it with the locked artifact set and `cargo test -p ferrum-kernels --test
cuda_nvfp4_matmul --features cuda-cutlass-nvfp4 -- --ignored`. Passing this test
qualifies its selected arithmetic and lifecycle cases; it does not establish
activation-quantizer equivalence, model output quality or comparative speed.

`cuda_nvfp4_quantize` uses the independent, SHA-pinned FlashInfer artifact with
the same managed runtime. Its explicit test configuration is a serialized
`CudaNvfp4ReferenceConfig`, selected by `FERRUM_NVFP4_REFERENCE_CONFIG`. Run
with features `cuda-cutlass-nvfp4,cuda-nvfp4-diagnostics` and `-- --ignored`.
It compares every packed/scaling byte, independently checks exact E2M1 ties,
and covers every finite BF16 input code, every positive finite BF16 group
maximum, signed zeros, K tails, scale padding and extreme positive F32 globals.
Repeated executions and Host output ownership are checked in the same test.
This is quantizer qualification, not independent full-model output validation.

Use CUTLASS v4.7.1 revision `cb4247394dd82148787aed73e5dc7cef33cbf862`, as for
the FP8 artifact. The upstream archive SHA256 is
`7a37329b43a47aaf9ef142be99baff3e71dbeea007b975891350cc0e50bc598b`.
Keep the source root and all artifacts outside this repository. Materialize
Ferrum sources under `kernels/` and the pinned upstream tree under `cutlass/`.
Compile with CUDA 12.8 or newer, C++17, `sm_120a`, no fast math. The source seed
records the required flags and include roots. Obtain the compiler-observed
dependency closure with NVCC `-M`, then use the existing native operator builder
`lock-source-dependencies` and `source-build` commands. Do not treat the seed as
a fully locked closure or compile vendored code in product Cargo builds.

The native boundary validates current device, plan identity, exact byte spans,
alignment and output/workspace disjointness before enqueue. A zero workspace
is exactly `{address:0,bytes:0}`. Nonzero workspace is caller-admitted scratch.
The caller must establish readiness on the supplied stream and retain every
input/output/workspace and native module resource through completion, including
error paths. Run rejects stream capture and implicit/default streams. Scale
and alpha value-domain validation belongs to their producer/source; this API
does not synchronously inspect device payloads or create an internal allocator.
