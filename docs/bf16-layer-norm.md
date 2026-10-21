# BF16 LayerNorm without affine weights

`operation.layer_norm.bf16` version 1.0 normalizes each row of a contiguous
BF16 tensor `[rows, hidden_size]`. Its one output has the same shape and dtype.
It subtracts the row mean and uses the population variance, dividing by
`hidden_size`, not `hidden_size - 1`. There are no gamma, beta, bias, residual,
or modulation operands. RMSNorm is a different operation.

The required `hidden_size` attribute is an unsigned integer in `1..=2^24`.
Every integer in this interval is exactly representable as an FP32 divisor.
The required `epsilon` attribute is a canonical rational in `[1e-12, 1]`.
The effective numerical parameter is defined by these conversions, in order:

```text
epsilon32 = (numerator as f64 / denominator as f64) as f32
```

This expression includes the integer-to-F64 conversions. It does not claim
correct rounding of arbitrary rational numbers directly to FP32. Both the
provider and an independent mathematical comparator use this same effective
positive FP32 epsilon. The original canonical rational remains in the plan
identity; two different rationals need not become the same semantic identity
merely because they round to the same FP32 value.

For exact real values decoded from a BF16 input row `x` of width `n`, the
mathematical normalization is:

```text
mu = sum(x[i]) / n
v  = sum((x[i] - mu)^2) / n
y[i] = (x[i] - mu) / sqrt(v + epsilon32)
```

Input/output storage is BF16. The arithmetic profile uses FP32, followed by
BF16 round to nearest, ties to even. A provider must establish its numerical
error against this mathematical operation; the storage types alone do not
prove equivalence. The contract selects `OracleSpec::OperationDefined` rather
than a uniform output-relative tolerance, which is unsuitable for constant
or nearly constant rows.

## Reference arithmetic order

The Reference provider uses the following specific FP32 sequence. This is
its reproducibility definition, not a mandatory reduction tree for every
future backend. `RN32` denotes FP32 round to nearest, ties to even; `B`
denotes the final BF16 round to nearest, ties to even.

1. Decode each BF16 element exactly to FP32. Reduce the row with a balanced
   binary tree. A single-element range returns its element; otherwise split
   at `floor(length / 2)`, recursively reduce both nonempty halves, then add
   their results in FP32. There is no padding to a larger width and no initial
   extra addition to zero. The maximum depth is `ceil(log2(n))`.
2. Compute `mean32 = RN32(sum32 / n32)`.
3. For each element compute `d32 = RN32(x32 - mean32)` and
   `square32 = RN32(d32 * d32)`. Reduce these squares with the same binary
   tree, then compute `variance32 = RN32(square_sum32 / n32)`.
4. Compute `q32 = RN32(variance32 + epsilon32)`, then `root32 = sqrt(q32)`
   and `inverse32 = RN32(1 / root32)` as separate operations. This is not an
   approximate reciprocal-square-root instruction.
5. Recompute each `RN32(x32 - mean32)`, multiply by `inverse32` in FP32, and
   round that product once to BF16. There is no affine multiplication or
   extra BF16 rounding between centering and the output product.

Multiplication and addition have separate rounding points. The numerical
evidence for this Reference path assumes IEEE gradual underflow; a backend
using FTZ/DAZ, Welford reduction, approximate `rsqrt`, or different contraction
must provide its own numerical evidence. It cannot inherit a bitwise claim
from this implementation. A mathematical comparison may establish acceptable
error for different algorithms without establishing equal output bits.

The logical arithmetic count for this particular Reference sequence is
`rows * (6 * hidden_size + 3)`: each scalar add, subtract, multiply, divide,
and square root counts once; storage conversions do not count. It is not a
hardware FLOP count or a throughput measurement. Telemetry must use the
tensor work domain, never fabricate a token count.

## Independent numerical evidence

The following small fixtures have analytically derived expectations. They
can be encoded directly as BF16 bits without calling the provider to generate
the expected output. They test precise local behavior, not a complete model
or a deployed PyTorch/vLLM backend.

| BF16 input row | Effective epsilon | Independent reasoning | Expected BF16 bits |
|---|---|---|---|
| `[3]` | `1` | Mean equals the input; centered value is zero | `[0x0000]` |
| `[3, 3, 3, 3]` | `1` | Exact mean is 3; every centered value is zero | `[0x0000, 0x0000, 0x0000, 0x0000]` |
| `[-1, -1, -1, 3]` | `1` | Mean 0, variance 3, square root 2; output `[-1/2, -1/2, -1/2, 3/2]` | `[0xbf00, 0xbf00, 0xbf00, 0x3fc0]` |
| `[1, 1, 1, 5]` | `1` | Mean 2; same centered row and result as the preceding fixture | `[0xbf00, 0xbf00, 0xbf00, 0x3fc0]` |
| `[-1, -1, 2]` | `1/4` | Mean 0, variance 2, square root `3/2`; FP32 inverse is `RN32(2/3)` and the final values round to the listed BF16 numbers | `[0xbf2b, 0xbf2b, 0x3fab]` |
| 4095 ones followed by `1+2^-7` | `258049/274877906944` | Mean `1+2^-19`, variance `4095*2^-38`, inverse square root 1024; the declared FP32 statistics are exact | 4095 copies of `0xbb00`, then `0x4100` |

Mixed rows should retain their independent results. Wrong dtype, rank,
same-element-count but different shape, attribute/width mismatch, illegal
epsilon, unsupported width, overlapping output, and out-of-range storage are
separate structural cases. Expected bits must not be generated with a second
call to the same normalization implementation.

The independent [Rust test oracle](../crates/ferrum-models/tests/vnext_tensor_reference_runtime/layer_norm_oracle.rs)
prepares a Candle CPU F64 reference and error bounds before candidate execution.
It explicitly uses sum/divide, two-pass population variance, sqrt/divide,
and multiplication. It rejects Accelerate/MKL dependency features: the scalar
F64 rounding analysis does not certify their vector math implementations.
The [managed integration tests](../crates/ferrum-models/tests/vnext_tensor_reference_runtime/layer_norm.rs)
execute real BF16 Dense-to-LayerNorm programs, including a half-ULP projection
that detects omission of the intermediate BF16 boundary.

For the Reference profile, let `u=2^-24`, `eta=2^-150`,
`h=ceil(log2(n))`, `gamma(k)=k*u/(1-k*u)`, and `A=sum(abs(x[i]))`.
The sum and mean error bounds are:

```text
Es = gamma(h)*A + (n-1)*eta/(1-h*u)
Em = ((1+u)*Es + u*A)/n + eta
```

Let `D[i]` conservatively bound `abs(x[i]-mu)` using the independent F64
mean enclosed by its own theoretical error. Define:

```text
c[i] = Em + u*(D[i]+Em) + eta
H[i] = (D[i]+c[i])^2
B[i] = 2*D[i]*c[i] + c[i]^2
Ev = sum(B)/n + gamma(h+2)*sum(H)/n + 2*n*eta/(1-(h+2)*u)
```

The absolute variance term covers `n` squares, `n-1` additions, and one
division. It conservatively forgoes division by `n` for earlier absolute
errors. With a true variance enclosure `[vLower,vUpper]`, propagate
`Eq=Ev+u*(vUpper+Ev+epsilon32)+eta`. The computed `q` enclosure has lower
bound at least `epsilon32`, since computed squares and their reductions
are nonnegative. Apply inverse square root to this enclosure and multiply
by the outward ratio interval `[(1-u)/(1+u), (1+u)/(1-u)]` for the two
separate normal square-root and reciprocal roundings. This yields an
upper bound `R` on the computed reciprocal and an error bound `Er` against
the true reciprocal. Each output's FP32 error is bounded by:

```text
E32[i] = D[i]*Er + c[i]*R + u*(D[i]+c[i])*R + eta
```

The same propagation with `u=2^-53`, `eta=0`, and `h=n-1` bounds the scalar
Candle F64 sequential reduction and produces `E64[i]`. F64 mean/variance
are used as mathematical enclosures only after adding their theoretical
errors. Finite BF16 inputs, width at most `2^24`, and the epsilon domain
keep the F64 operations away from underflow. The final comparison is:

```text
abs(actual_bf16-y64) <= E32 + E64 + 2^-8*(abs(y64)+E32+E64) + 2^-134
```

All nonnegative bound operations and interval endpoints round outward in
F64; the measured comparison distance rounds upward. Before execution the
oracle sufficiently excludes overflow at every FP32 stage and at the final
BF16 conversion. Inputs outside that domain are rejected, never accepted
using a tolerance inferred from the candidate. These conservative bounds
do not replace strict analytic fixtures or establish bitwise parity with
another normalization algorithm.

For an IEEE gradual-underflow analysis, each FP32 operation can require an
absolute rounding term in addition to its relative term; the half-minimum
subnormal terms are `2^-150` for FP32 and `2^-134` for BF16. A tree's additive
error must account for its internal additions and their propagation, not
just multiply this term by tree depth. Any sufficient no-overflow domain
must cover intermediate sums, differences, squares, variance accumulation,
epsilon addition, and final normalization. A finite final mathematical mean
or variance alone does not establish this domain.

The finite/no-overflow restrictions belong to numerical validation, not a
promise that providers scan tensor contents before submission. Ordinary
IEEE nonfinite propagation remains possible. A numerical oracle must not
count nonfinite candidate results or inputs outside its proven domain as a
successful finite-error comparison. Width-one and constant-row examples
above deliberately avoid overflowing intermediate sums.

## Managed execution boundaries

The descriptor binds one BF16 input and output, requires their shared shape
symbols, prohibits output aliasing, and closes the attribute vocabulary.
`LayerNormBf16Geometry` additionally checks that the actual width equals the
`hidden_size` attribute and that the exact rational epsilon is in range.
Providers call this semantic check during estimation and encoding. The
geometry object is only a descriptor; it grants no memory or submission
authority and does not validate physical buffer identity or retention.

The initial Reference provider supports fixed tensor work. Planning,
admission, retained input/output regions, and ordinary submission use the
existing runtime authority. Input/output alignment is at least 16 bytes.
The contract permits explicitly planned scratch for another implementation;
binding and persistent auxiliary resources are forbidden. The balanced
Reference implementation can read managed backing directly with bounded
recursion, without a whole-row or whole-tensor payload copy.

Shape, region, runtime identity, alignment, and overlap validation precede
device effects. An input may be produced by an earlier command in the same
submission, so content-dependent numerical failure cannot be promised to
occur during this structural preflight. A failure after execution starts
must preserve the existing submission/completion semantics; it cannot be
reported as definitely not submitted.
