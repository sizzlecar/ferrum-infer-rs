# Active compression and output projection

`FastH3AttentionOutputSource::prepare` selects one block's original
`attn.to_out` NVFP4 projection. Its three numerical inputs are fine attention,
pooled attention already expanded to the same packed rows, and the original
raw BF16 compression gate. Pooling, pooled attention and tile/untile mapping
are separate producers. The output is BF16 `[rows, hidden]`, suitable for the
existing residual, second norm/modulation and FFN consumers.

The fixed FastVideo source `866a3cdb91a85aef8561847365fc8564add9e4c8`
(`models/dits/minimax_h3_vsa_fp4.py:415–430` and
`attention/backends/video_sparse_attn_h3.py:1174–1196`) converts pooled
attention to the fine-output dtype and applies `addcmul` with the raw gate.
There is no sigmoid or clamp. The deployed Torch package identifies commit
`7661cd9c6b841b62b7f411aa52ec51f05457263b`; its
[pointwise helper](https://github.com/pytorch/pytorch/blob/7661cd9c6b841b62b7f411aa52ec51f05457263b/aten/src/ATen/native/cuda/DeviceAddCmulCdiv.cuh)
explicitly uses F32 FMA for BF16 addcmul with alpha one.

The graph therefore has three operations: `ADDCMUL_FMA_BF16`, explicit unit
activation quantization and the original packed NVFP4 GEMM. FMA retains the
product through its F32 add and rounds only the result to BF16. The existing
separate-product addcmul operation keeps its original contract. No new
executor or allocation owner is introduced. Packed weight/scales preserve
their original physical roles; authoritative stored F32 alpha and unit scale
use eight admitted Host bytes. The shared unit-alpha recipe is the same one
used by the Q/K/V/gate projection consumer.

The initial family explicitly represents an active compression branch. It
does not infer inactivity from a runtime zero gate or claim that an unverified
checkpoint permits skipping the pooled branch. A structurally inactive route
needs its own checked checkpoint proof.

CPU tests cover real package sources, shape and physical-component bindings,
source mixing, capacity rollback and an independent exact dyadic full-graph
oracle. The ignored CUDA package test executes all three operations with the
same Stored source; the independent pointwise CUDA test covers signed zero,
intermediate overflow, NaN classification and raw gates outside `[0,1]`.
Those GPU tests require a separately recorded run. They do not establish
full FastVideo output parity, video quality or performance.
