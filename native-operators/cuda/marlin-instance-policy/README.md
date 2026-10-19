# Marlin instance policy

`marlin_cuda_kernel.cu` is the pinned a9 source with both `FERRUM_MARLIN_TILE`
overrides removed. The original automatic tile selection and explicit FFI tile
parameters are unchanged, as are the numerical kernels, stream and workspace
contracts. Operator ABI **2** rejects ABI 1 artifacts with ambient overrides;
Ferrum's descriptor layout and the two mathematical signatures remain unchanged.

`../source-overlays/marlin-instance-policy.json` binds the original bundle member
and this override. Materialize it with the existing native builder, generate the
source lock, and use the ordinary source-build/package/assemble-set pipeline.
This repository overlay is a build input, not a published binary. Actual CUDA
math and child-process environment isolation must pass with the newly built
artifact before product pins are updated.
