# Reference BF16 activation profile

The three tensor providers are registered and exercised through managed graphs.
The independent CPU qualification includes all BF16 input encodings; backend
and managed component tests also pass. This is not deployed GPU parity or an
image-generation result. Workspace and backend compile checks are recorded
separately from these component tests.

## Three distinct operations

| Operation | Inputs | Output | Scalar logical work per element |
| --- | --- | --- | --- |
| `operation.silu.bf16` | `x[R,H]` | `y[R,H]` | 4 |
| `operation.silu_mul.bf16` | `gate[R,H], up[R,H]` | `y[R,H]` | 5 |
| `operation.tanh.bf16` | `x[R,H]` | `y[R,H]` | 1 |

All values are contiguous BF16, rank two, with positive dimensions and exactly
the same shape. Attributes are empty. There is no broadcasting or implicit
all-one `up`. Checked tensor extents and logical-work multiplication must fit
their runtime representations. These elementwise operations have no LayerNorm
width-to-FP32 conversion and consequently do not inherit its `2^24` width limit.

The output is NoAlias. Core permits repeated immutable read ports for the same
logical program value; distinct logical inputs may not physically overlap.
Private backend storage tests can exercise wider read/read overlap without
relaxing that public identity rule. Each operation has its own contract,
capability, provider and resource estimator.

The Reference path uses one fixed-tensor participant and eager execution, with
no token invocation. Unary execution locks two allocation owners; SiLU-mul locks
three. Locks are sorted and deduplicated. Whole-command-batch preflight checks
runtime identity, BF16 dtype, range, alignment and all output/read overlaps
before modifying any output. Numerical NaNs and infinities are valid data, not
structural validation errors. No preflight scan assumes future device values
already exist.

Compilation and execution use the normal managed tensor path and
`PlanRuntimeResources`. Reference estimates require 16-byte value alignment,
without scratch, binding or persistent buffers. Scalar intermediates stay in
local arithmetic; there is no unaccounted expanded or FP32 tensor payload.
Logical work counts sign negation, addition, division or multiplication once
and each exp/tanh call once. Conversion operations and the library's internal
approximation are excluded. These counts are **not hardware FLOPs**.

## Fixed numerical identity

The profile is `reference.bf16.transcendentals.libm-0.2.15.v1`. Cargo pins
`libm = "=0.2.15"` and `half = "=2.7.1"`, which were already the resolved lockfile
versions. The numerical identity includes their registry checksums and audited
source identities. Providers and estimators include that identity and their
actual implementation sources in their fingerprints. Dependency upgrades need
a new identity and renewed qualification, rather than a local-file text test.

Let `RN32` mean FP32 nearest-even and `B` mean BF16 nearest-even:

```text
e = libm::expf(-decode(x))
d = RN32(1 + e)
a = B(RN32(decode(x) / d))
silu(x) = a
silu_mul(x, up) = B(RN32(decode(a) * decode(up)))
tanh(x) = B(libm::tanhf(decode(x)))
```

The environment must use nearest-even and gradual underflow, without FTZ/DAZ
or fusion across explicit boundaries. Tests probe the arithmetic assumptions.
NaN payloads are unspecified; NaN classification, infinity signs and signed
zeros are explicit. In particular, this expression gives `SiLU(-Inf)=NaN`,
`SiLU(+Inf)=+Inf`, and preserves both zero signs. A stable-sigmoid rewrite would
change the profile: `expf(90)` overflows, so `SiLU(-90)=-0`, although the ideal
real function is nonzero there.

## Independent qualification

The test-only oracle prepares expected sets before evaluating the actual scalar
helpers. It calls neither candidate exp/tanh nor host F64 exp/tanh to construct
expectations. All 65,536 BF16 encodings, including 65,280 finite encodings, are
covered for both primitives and both unary BF16 compositions. SiLU-mul propagates
every permitted activation through an independently rounded BF16 product, with
selected multipliers covering finite rounding, underflow, overflow, signed zero
and nonfinite behavior. This selected pair domain is not all `2^32` input pairs.

The predeclared primitive bound is four adjacent **finite FP32 representable
steps** from ideal mathematical RN32. Classification takes priority: an ideal
zero or infinity must match its sign and class; a finite nonzero reference must
remain finite and nonzero. Normal/subnormal transitions are permitted, but
neighborhoods cannot cross zero or infinity. This is not a relative-error bound
and is not a claim of correctly rounded exp/tanh.

For a real enclosure spanning possible rounded cells `R`, an accepted candidate
must be within four steps of **every** `r` in `R`. The allowed set is the
intersection of those neighborhoods and the class constraints. An empty set or
unresolved class fails qualification. Typed sets retain NaN, infinity and both
zeros through composition, including `0 * Inf`. Their maximum capacity of nine
branches follows from one four-step neighborhood; there is no assumed two-value
BF16 output limit. A passing set-membership check does not establish unique bits.

The independent enclosures and local arithmetic are:

- Exp uses exact binary scaling `a=abs(t)/512`, positive Taylor terms through
  degree 24, and tail bound `term25/(1-a/26)`, all with F64 outward rounding.
  Nine interval squares undo the scaling; negative arguments invert and reverse
  endpoints. Analytic tails at `t>=128` and `t<=-256` determine FP32 infinity and
  zero. Intermediate positive underflow rounds outward to a nonzero upper bound.
- For `0<=a=abs(x)<=2^-12`, the inequality `a-a^3/3 <= tanh(a) <= a` and odd symmetry
  fix the ideal RN32 cell at the BF16 input. The candidate still has four-step
  latitude. Above `abs(x)>=16`, an analytic saturation bound fixes the ideal
  cell at signed one. Middle inputs use interval
  `sign(x)*(1-exp(-2a))/(1+exp(-2a))`.
- Exp has a stronger, predeclared profile rule: `abs(t)<=2^-25` returns exactly
  FP32 one. The mathematical rounding cell and the pinned implementation's
  `1+t` branch independently satisfy this rule.
- `RN32(1+e)` is independently decided by exact F64 addition in a bounded
  significand domain, and analytic tiny/large branches outside it.
- Division uses integer significands, exponents, quotient and remainder to
  decide RN-even for finite BF16 `x` divided by positive FP32 `d>=1` or infinity.
  Binade and underflow classification bound the integer shifts. It does not rely
  on a possibly doubly rounded F64 quotient.
- BF16 conversion uses independent integer tie-to-even rounding. Finite BF16
  products have at most 16 significand bits and fit exactly in F64 before the
  specified FP32 and BF16 rounding steps.

No unresolved input is silently omitted, and no measured result changes the
predeclared tolerance. Timing, step-error histograms and allowed-set sizes are
observations, not pass-ratio or fixed-count gates.

In the initial CPU run, the observed primitive distance bound against every
possible reference cell was at most one FP32 step. There were 61 ambiguous exp
reference intervals and 48 tanh intervals; this bound is consequently not a
claim about an exact observed error. SiLU allowed sets were singleton for 65,534
encodings and had two values for two encodings. Tanh BF16 sets were singleton
for all encodings. The 720,896 selected SiLU-mul pairs had 720,886 singleton sets
and ten two-value sets. These observations do not narrow the declared four-step
contract or generalize to untested providers.

## Rounding counterexamples

For `x=1`, independent bounds on the exponential fix `B(SiLU(1))=187/256`.
Multiplication by `129/128` then produces BF16 `188/256` (`3f3c`); omitting the
activation's BF16 boundary gives `189/256` (`3f3d`). Similarly,
`B(tanh(1))=195/256`; multiplying by `129/128` gives `197/256` (`3f45`), whereas
omitting the tanh BF16 boundary gives `196/256` (`3f44`).

At the smallest positive BF16 input `x=2^-133` (`0001`), the fixed expression
divides by exactly two. FP32 represents `2^-134`, and BF16 ties-to-even yields
`0000`. The ideal real SiLU is slightly larger than that midpoint and would
round directly to `0001`. The two numerical functions are different.

That smallest input is **not** a counterexample to the generic four-step exp
allowance: its FP32 subnormal quotient absorbs all those perturbations before
the final BF16 conversion. A sensitive input is the largest BF16 subnormal,
`x=127*2^-133` (`007f`). Exact exp one gives quotient FP32 `003f8000`, a BF16 tie
rounding to `0040`. The otherwise permitted exp value `1+4*2^-23` gives divisor
`2+2^-21`; independent integer division rounds to FP32 `003f7fff`, then BF16
`003f`. The stronger exp tiny rule removes this alternative. Tests preserve
both examples and the intermediate FP32 division.

## Managed and deployment evidence boundaries

Managed graph fixtures use real BF16 safetensors projections followed by SiLU
or SiLU-mul. The residual graph joins a Dense skip branch with a separate
host-gate-to-tanh branch at gated residual. They check the numerical boundaries,
repeated read-only values, Dense skip lifetime, actual uploads/readback, output
capacity rejection before device admission, and output ownership after the
device root closes. These are component tests, not a complete Qwen block.

The CPU profile does not establish CUDA libdevice/Triton/ATen parity. Deployment
requires separate evidence for actual versions, eager/compiled paths, division,
fusion and FTZ settings. Full conditioning, denoising, VAE, CLI/HTTP behavior,
image quality and same-SLO throughput remain separate acceptance gates.
