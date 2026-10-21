# BF16 modulation and gated residual

Two model-independent version 1.0 operations express pointwise arithmetic
with explicit intermediate BF16 boundaries:

| Operation | Contiguous BF16 inputs | Output | Arithmetic |
| --- | --- | --- | --- |
| `operation.modulation_scale.bf16` | `x[R,H]`, `s[P,H]` | `y[R,H]` | `B(x * B(1+s))` |
| `operation.gated_residual.bf16` | `x[R,H]`, `g[P,H]`, `u[R,H]` | `y[R,H]` | `B(x + B(g*u))` |

`B` means conversion to BF16 using round to nearest, ties to even. Each
addition and multiplication first operates in FP32, with its own FP32
rounding. Operands are decoded exactly from BF16; each internal `B` is
decoded back to FP32 before the next arithmetic operation. The gate `g` is
already activated. The second operation neither computes tanh nor constrains
the gate to a model-specific interval. The first has no shift operand.

These boundaries are semantic requirements. A provider may keep the
intermediate in a register, but cannot drop its BF16 round-trip, contract
the residual into an FMA, or evaluate the complete expression in F64 followed
by a single BF16 conversion. A future fusion must preserve the same steps.
Both descriptors use an operation-defined oracle.

## Explicit row broadcasting

Both operations require the text attribute `broadcast`:

- `elementwise`: `P == R`; element `(r,h)` reads parameter `(r,h)`.
- `row`: `P == 1`; every output row reads parameter `(0,h)`.

The mode is never inferred from the element count or from `P == 1`. For
`R == 1`, both explicitly selected modes have valid geometry; the distinct
attribute values still participate in plan identity. Scalar, column,
arbitrary strided and segmented broadcasts are outside version 1.0.

The parameter's first dimension has its own `parameter_rows` symbol. Sharing
the base tensor's `rows` symbol would incorrectly exclude row broadcasting
before provider estimation. `Bf16RowBroadcastGeometry` checks the actual
resolved tensor shapes, dtypes and layouts, the output/base equality, and
the selected relation between `P` and `R`. Gated residual also requires
`u` to have exactly the base tensor's shape and layout. Both estimation and
encoding perform these checks. Equal element counts alone are insufficient.

Dimensions must be positive and tensor byte extents must fit checked
arithmetic. Providers additionally check their addressable range and logical
work counts. There is no LayerNorm-style `H <= 2^24` restriction: pointwise
operations never convert the width to an FP32 divisor.

## Reference execution and resources

The initial Reference providers accept fixed tensor work with one execution
participant and ordinary eager submission. A participant may contain many
mathematical rows. Token invocation is unsupported. Each operation has an
independent capability, provider identity and estimator identity.

Inputs and outputs remain subject to the existing compiler, runtime plan,
admission, host/device ownership and completion rules. Typed plan bindings
require at least 16-byte alignment. Output aliasing is prohibited against
every input, including partial overlap. Read-only inputs may share storage
where core permits reuse of the same logical read-only value; this does not
relax core's rejection of overlapping bindings for distinct logical values.
The private backend can safely read overlapping input regions, and tests
exercise that storage behavior separately. Nonoverlapping regions of the
same allocation are also legal. Region
identity, bounds, scalar alignment and output overlap are checked for the
entire command batch before any command writes or obtains a completion
fence.

The Reference implementation locks the actual allocations in address order
and deduplicates aliases with fixed-size `AllocationLocks<3>` or
`AllocationLocks<4>`. It indexes row parameters directly and retains only
scalar arithmetic intermediates. It does not expand a `[1,H]` parameter into
an `[R,H]` payload. This statement concerns the arithmetic kernel; it is
not a claim that compilation or the whole process performs no heap
allocation. The contracts permit explicitly planned scratch for another
provider; the Reference estimators require none. Neither operation has
weight inputs or auxiliary persistent resources.

Both Reference commands report `R*H` tensor elements and `2*R*H` scalar
arithmetic operations. Each addition or multiplication counts once; storage
conversions do not. This is a logical work measure, not a hardware FLOP
estimate or a token count.

Structural preflight cannot scan numerical inputs that an earlier command
in the same batch has yet to produce. IEEE infinities, NaNs, gradual
underflow and signed zeros therefore follow the specified arithmetic at
execution. Numerical validation compares NaN classification rather than a
particular NaN payload. Reference results do not certify a backend using
FTZ/DAZ or a different rounding mode.

## Independent numerical checks

Two small analytic fixtures distinguish the internal BF16 rounding points:

| Operation and operands | Correct output bits | Output if internal `B` is omitted |
| --- | --- | --- |
| Modulation: `x=3/2`, `s=2^-8` | `0x3fc0` (`3/2`) | `0x3fc1` |
| Residual: `x=-1`, `g=127/128`, `u=129/128` | `0x0000` (`+0`) | `0xb880` (`-2^-14`) |

For modulation, `1+2^-8` is exactly halfway between BF16 `1` and
`1+2^-7`; the even result is `1`. For residual, the product is exactly
`1-2^-14`, which rounds to BF16 `1` before adding `-1`. These expected
values follow from rational arithmetic and BF16 rounding cells, independently
of the provider.

BF16's maximum finite value is `(2 - 2^-7) * 2^127`; `65504` is the
maximum of FP16, a different format. In particular, `256 * 256 = 65536`
is exactly representable in BF16 (`0x4780`). A genuine intermediate
overflow fixture uses `g=127/64` (`0x3ffe`) and
`u=(129/128)*2^127` (`0x7f01`). Their exact product is
`(2 - 2^-13)*2^127`, finite in FP32 but above the BF16 overflow rounding
threshold `(2 - 2^-8)*2^127`. The internal `B(g*u)` becomes infinity.
With `x=-BF16_MAX`, the correct residual remains positive infinity, while
omitting the internal conversion would yield a finite cancellation result.

The [managed integration tests](../crates/ferrum-models/tests/vnext_tensor_reference_runtime/pointwise.rs)
compose actual BF16 projection, normalization, modulation and residual
nodes. The projection weights are real inputs to the tested computation;
there is no placeholder weight to bypass the current family schema. Tests
also exercise the two broadcast modes, exact host capacity, rejection before
compute when capacity is insufficient, and output ownership after the plan
root closes.

The [test oracle](../crates/ferrum-models/tests/vnext_tensor_reference_runtime/pointwise_oracle.rs)
prepares expected results before candidate execution. It decodes BF16 bits
independently, evaluates each local binary operation in F64, explicitly
rounds that result to FP32, and applies an independent integer BF16 RNE
conversion at each specified boundary. This is a comparison with the
declared FP32/BF16 sequence, not an ideal-real-valued fused expression.

A product of finite BF16 operands has at most 16 significant binary digits
and a magnitude representable by F64, including the product of two minimum
subnormals. Thus this local product is exact in F64 before its FP32 cast.
For addition, nearby exponents fit the exact sum in F64; when the exponent
gap is too large to retain the smaller term, that term is far below a
half-ULP of FP32 at the larger BF16 operand. The larger operand has only
eight significant bits and cannot itself sit at an FP32 rounding midpoint.
Consequently the F64 intermediate does not introduce a double-rounding
difference for these binary BF16-operand operations. This reasoning does
not justify arbitrary FP32-operand expressions or transcendental functions.

## Model adaptation and remaining scope

Qwen-Image-2.1 uses scale-only modulation and already activated gates, but
these operation contracts contain no model-name selection. Models compose
them with their actual parameter layout and activation operations.

The two broadcast modes do not yet represent a mixed sequence whose
conditioning rows use one timestep and target rows use another. A later
explicit, validated segmented-row mapping must preserve that distinction
and enter plan identity; it must not silently copy or reinterpret a full
parameter tensor. Column slices must also satisfy their actual layout
contracts before they can be consumed as contiguous tensors.

Unary SiLU, SiLU-mul and tanh are separate planned operations. They require
provider-specific transcendental arithmetic and numerical evidence. For
example, FP32 `g / (1 + exp(-g))` at `g=-90` can produce negative zero after
the exponential overflows, while a full F64 evaluation gives a nonzero
BF16 result. Neither a stable rewrite nor a higher precision expression is
automatically equivalent to the deployed expression.

This slice does not implement a complete DiT block, conditioning encoder,
VAE, CUDA pointwise kernels, CLI/HTTP image generation or end-to-end
vLLM-Omni quality and throughput parity.
