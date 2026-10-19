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

The source definition uses `compute80_ptx_with_sm120_native`: the exact
`sm_120` target compiles with `-arch=sm_120`; every other target retains the
original `-arch=compute_80` policy. This does not change the CUDA source or FFI
ABI and does not qualify additional targets. Regenerate the source plan with
`lock-source`; the resolved argument is recorded and validated in source-build
receipts and object-cache identities. A new SM120 artifact requires its own
numerical qualification and package/set receipts; existing SM89 artifacts
retain their original build provenance.
