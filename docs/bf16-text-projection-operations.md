# BF16 text projection operations

`operation.zero_centered_rms_norm.bf16` consumes BF16 `x[R,C]` and the original
BF16 weight `[C]`. It computes the whole-row mean of squared inputs, adds the
declared rational epsilon, and forms an FP32 reciprocal square root using
separate square-root and division operations. Both `x * inverse` and `weight + 1`
remain FP32; only their final product is rounded to BF16. This differs from the
existing head RMS operation, which has an intermediate BF16 boundary.

`operation.gelu_tanh.bf16` is a unary same-shape BF16 operation. Its numerical
profile fixes the FP32 constants and multiplication/addition order for the tanh
approximation, with one final BF16 rounding and no implicit FMA. Computed FP32
tanh arguments require their own validation; prior qualification of the separate
BF16-input tanh operation does not qualify GELU. Nonfinite values follow the
declared IEEE arithmetic; providers do not reject tensor contents after enqueue.

Both use fixed tensor execution, existing resource admission and no global
scratch. Reference supports whole-row RMS widths through 2^24; the initial CUDA
provider supports power-of-two widths through 4096 with bounded CTA shared
memory. CPU and CUDA profiles are separate, and neither claims bitwise Torch
equivalence, whose reduction and reciprocal-square-root choices may differ.

`cuda_vnext_tensor_providers(runtime, selected)` selects typed operations from
one native module. It rejects empty, duplicate or unsupported selections before
module loading and shares one retained module permit across the resulting
providers. Existing convenience factories retain their previous operation sets.
Dense remains separate because it uses the already admitted lane BLAS handle.
