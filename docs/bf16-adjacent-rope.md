# Adjacent BF16 rotary operation

`operation.adjacent_rope.bf16` accepts contiguous BF16 input/output `[R,C]`,
an explicit contiguous F32 frequency tensor `[S,D/2,2]`, and the unsigned
`head_dim=D` attribute. Dimensions are positive, D is even, R is divisible
by S, and C is divisible by D. Frequencies are shared across heads and batches;
row `r` uses frequency row `r % S`. Output has NoAlias semantics and must not
overlap either input. This operation neither generates positions/frequencies
nor requires frequencies to have unit magnitude.

For adjacent input `(a,b)` and frequency `(c,s)`, the version 1 contract is:

```text
real = BF16_RNE(FMA_RN(a, c, RN32((-b) * s)))
imag = BF16_RNE(FMA_RN(b, c, RN32(a * s)))
```

The independent products round before the explicit fused multiply/add. Rust
Reference uses `f32::mul_add`; CUDA uses `__fmul_rn` and `__fmaf_rn` with gradual
underflow. The CUDA `--fmad=false` setting prevents implicit contraction and
does not remove explicit FMA. NaN classification is specified, payload/sign is
not. Logical work is seven scalar arithmetic operations per pair, counting a
FMA as two operations and negation as one; this is not a hardware FLOP counter.

Both providers validate exact scalar widths, byte ranges and geometry before
submission. They operate directly on admitted buffers without a temporary
tensor or global scratch allocation. CUDA shares the existing native module
and its capacity permit through `cuda_vnext_tensor_component_providers`.
The older five-operation and seven-operation factories retain their API sets.

The profiles are `reference.adjacent_rope.bf16.f32_fma_rne.v1` and
`cuda.ieee.adjacent_rope.bf16.fma.v1`. Implementation identity alone does not
establish whole-domain CUDA or deployed-model numerical equivalence.

The Qwen block-zero verifier reuses its existing Q/K normalization path.
`--frontend rotary --rotary-probe identity|quadrants` selects a distinct typed
family/profile, admits a managed F32 frequency input, and retains all six
Q/K pre-normalization, post-normalization and post-rotation boundaries.
The quadrant oracle propagates every precomputed normalization possibility;
it does not inspect candidate intermediate values to construct expectations.
These explicit probes do not validate production position synthesis or
arbitrary trigonometric frequencies. The default frontend remains `norm`.

Tests cover batch/head indexing, explicit FMA versus split multiply/add,
signed-zero behavior, mixed scalar storage, exact ranges, NoAlias and whole
batch preflight. Real CUDA tests are explicitly ignored unless an exclusive
GPU window is available; run them with `--test-threads=1`.
