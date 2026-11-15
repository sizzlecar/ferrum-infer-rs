# Independent NVFP4 qualification artifact

`flashinfer_nvfp4.cu` calls the unmodified FlashInfer / TensorRT-LLM
`cvt_warp_fp16_to_fp4<__nv_bfloat16,16,16,false,false>` and scale-layout helper.
It does not call Ferrum's quantization implementation. The wrapper accepts only
caller-owned device spans and an explicit stream; it allocates and synchronizes
nothing. Its padding loop is wrapper code and must be checked separately.

Build the shared library outside the repository against a pinned FlashInfer
header tree, CUDA and host compiler. Enable BF16/FP8, target `sm_120a`, and use
FlashInfer's `--use_fast_math` compilation profile. This profile includes FTZ
multiplication and zero comparison as well as approximate FTZ reciprocal. Disabling fast math is a
different numerical profile. Record the full compiler invocation, compiler
binary hashes, versions and the compiler-observed transitive header closure in
`Nvfp4ReferenceManifest`; hash the library and the complete manifest separately.
The build is independent of the production static native-operator archive.

Enable `ferrum-kernels/cuda-nvfp4-diagnostics` and pass an explicit
`CudaNvfp4ReferenceConfig` to `cuda_vnext_nvfp4_reference_provider`. The config
requires both artifact paths, both SHA256 pins, and a positive opaque module
allowance. Keep these files immutable during validation and loading. The factory
checks pins before loading code and reserves the same runtime's device capacity
before `dlopen`. A dedicated finite module slot prevents duplicate live owners;
the command retains the library, context, inputs and outputs through completion
or quarantine. Normal product factories never register this provider.

Compare packed values and every padded scale byte for the explicit
`CudaRcpApproxFtzV1` contract, including signed zero, BF16 subnormals, E4M3 scale
underflow, saturation and rounding boundaries. Matching a second graph that
shares Ferrum's GEMM only establishes the quantizer comparison; it is not an
independent full-model numerical oracle or performance evidence.
