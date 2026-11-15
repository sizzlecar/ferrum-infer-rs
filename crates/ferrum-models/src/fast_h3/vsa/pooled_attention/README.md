# Shared pooled attention scores

The fixed FastVideo revision `866a3cdb91a85aef8561847365fc8564add9e4c8`
uses `models/dits/minimax_h3_vsa_fp4.py:397–420` to compute F32 tile Q/K scores,
scale by `1/sqrt(head_dim)`, select sparse regions, and reuse those same scores
for softmax times pooled V. `FastH3PooledAttentionFamily` implements the four
arithmetic nodes with three explicit F32 inputs `[batch, head, tile, channel]`.
It returns both scaled scores and weighted V, retaining the head axis and the
two distinct F32 matrix products. It does not recompute scores inside a fused
attention call. Tile pooling and the final head permutation/BF16 cast remain
explicit adjacent components.

The scalar boundary follows the installed Torch source revision
`7661cd9c6b841b62b7f411aa52ec51f05457263b`:
[BinaryDivTrueKernel.cu](https://github.com/pytorch/pytorch/blob/7661cd9c6b841b62b7f411aa52ec51f05457263b/aten/src/ATen/native/cuda/BinaryDivTrueKernel.cu#L32)
computes a CPU double reciprocal, casts it to F32, then multiplies. Consequently
the separate scale node uses `F32(1.0_f64/sqrt(F64(head_dim)))`; multiplying inside
cuBLAS alpha would remove a declared F32 boundary. The config reconstructs and
checks these scalar bits after deserialization. The fixed model has head width
128; this source analysis is not a binary parity measurement of arbitrary
Python libm implementations.

The CUDA provider uses the existing managed lane cuBLAS handle with
`CUBLAS_COMPUTE_32F_PEDANTIC`, atomics disabled, and null/zero vendor workspace.
It never enables TF32 or falls back to an implicit workspace. Each batch and
head remains independently indexed. cuBLAS reduction order is vendor-defined;
the independent Reference profile uses ascending F32 FMA. Both store the full
matrix result before scaling. Softmax uses F32 max/subtraction/exp/sum/division;
CUDA uses libdevice `expf` and a fixed 256-thread reduction, Reference uses
ascending scalar reduction. These profiles do not assert Torch bit equality.
Negative infinity is a masked entry; NaN, positive infinity, and all-negative-
infinity rows yield all NaNs. No BF16 probabilities or hidden score allocation
exist. Real scores, probabilities, outputs and live readback all participate in
the original Core lifetime and capacity accounting.

The first qualification uses exact dyadic products, a separate FP64 softmax/PV
oracle, non-square matrix tests, independent batch/head values, IEEE boundary
tests, invalid storage/alias/runtime rejection, and retained output retirement.
Compilation, backend execution and full-model quality/performance evidence must
be reported separately; source presence alone establishes none of them.
