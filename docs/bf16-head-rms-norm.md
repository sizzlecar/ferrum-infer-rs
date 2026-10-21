# BF16 head-wise RMS normalization

`operation.head_rms_norm.bf16` normalizes independent contiguous groups in a
BF16 `[rows, columns]` matrix. The required `head_dim=D` attribute divides
`columns`; gamma is a contiguous BF16 `[D]` vector shared across every group.
Output preserves the matrix shape. No reshape copy, token count, KV geometry,
bias, or zero-centered gamma adjustment is implied.

For each group, the operation computes FP32 mean square and inverse root, then
`B(decode(B(x * inverse)) * gamma)`, where `B` is BF16 round-to-nearest-even.
The first BF16 conversion is part of the operation and must survive fusion.
The canonical rational epsilon lies in `[1e-12, 1]`; its effective value is the
F64 quotient converted to FP32. Providers declare their reduction and inverse
root arithmetic separately. An operation-defined oracle does not make different
provider profiles bitwise equivalent.

The semantic contract allows `D` up to `2^24`, so its divisor is exactly
representable in FP32. `HeadRmsNormBf16Geometry` validates rank, dtype, contiguous
layout, divisibility, gamma extent and epsilon. It proves shape consistency;
it grants no allocation or execution authority.

The Reference profile is
`reference.head_rms_norm.bf16.pairwise_f32.sqrt_div.v1`. It uses a bounded
recursive pairwise reduction, with floor-half left partitions for odd lengths,
then FP32 square root and separate reciprocal division. It reads admitted
regions directly, uses bounded scalar stack storage, rejects output overlap,
checks exact byte extents and runtime ownership, and participates in batch
preflight before any output is written.

The CUDA profile is `cuda.ieee.head_rms_norm.bf16.pairwise_sqrt_div.v1`.
Its first implementation accepts power-of-two `D` up to 4096, including the
Qwen Image 2.1 head dimension 128. It uses one bounded shared-memory reduction
per group, with explicit round-to-nearest FP32 operations. Gamma may come from
a managed program input or stored dense BF16 weights. Commands retain the
shared native module and all admitted buffer owners until retirement.
Zero global scratch does not establish the size of opaque context/library
memory; those experimental allowances remain separately declared.

Logical arithmetic work is `groups * (4D + 3)` and logical element count is
`rows * columns`. These counts describe the operation rather than measured
hardware FLOPs. IEEE nonfinite/overflow propagation remains numerical behavior;
structural preflight does not scan activation contents to prove a finite domain.

A strict first-rounding test uses `D=128`, every input and gamma equal to
`129/128`, and epsilon `112375/147456`. The declared FP32 sqrt/div profile
produces reciprocal `3/4`. Keeping both BF16 boundaries gives `0x3f44`;
omitting the first gives `0x3f43`. This targeted counterexample does not qualify
all numerical inputs or establish PyTorch/vLLM-Omni equivalence.
