# BF16 dense linear contract

`operation.dense_linear.bf16` version 1.0 computes
`Y[r,o] = sum_i X[r,i] * W[o,i]`. Input `[R,K]`, weight `[O,K]`, and
output `[R,O]` are contiguous BF16. Multiplication and accumulation use FP32;
the output is rounded once to BF16 using round to nearest, ties to even.
It has an independent provider capability. The existing F16 operation's
signature, version, and numerical tolerance remain unchanged.

The Reference implementation decodes each BF16 bit pattern directly to FP32,
accumulates in increasing K order, and converts the final accumulator to BF16.
Its repeatability guarantee applies to the same Reference implementation.
Other backends may have different reduction trees; their results need an
explicit numerical comparison, not a claim of bitwise equality.

## Input-dependent numerical oracle

The contract uses `OracleSpec::OperationDefined`. A single relative tolerance
does not handle cancellation correctly. For one output, define the exact real
dot product `y`, expanded absolute sum `S = sum_i abs(X_i * W_i)`,
`u32 = 2^-24`, `ub = 2^-8`, and `gamma(n,u) = n*u/(1-n*u)`.
Within the domain below, let `E32 = gamma(2K,u32) * S`. Then the absolute
error bound is:

```text
abs(actual_bf16 - y) <= E32 + ub * (abs(y) + E32) + 2^-134
```

This allows FP32 arithmetic error and the final BF16 rounding, including the
BF16 subnormal absolute rounding term. The declared domain requires finite
inputs, no FP32 product or intermediate sum overflow/underflow, no final BF16
overflow, and `2K*u32 < 1`. Exact zeros are allowed. Outside this domain the
bound cannot establish correctness; those cases require separate evidence.
The provider does not silently sanitize nonfinite values.

If the independent reference is an F64 computation `y64`, its own error must
be included: obtain an upper bound `S_upper`, set
`E64 = gamma(2K,2^-53) * S_upper`, and compare against
`E32 + E64 + ub * (abs(y64) + E64 + E32) + 2^-134`.
Computing `S_upper` and the bound must account for floating point rounding;
simply evaluating a mathematical bound with downward rounding is insufficient.
Report the maximum absolute/relative error, RMSE, worst coordinate, and
per-element bound failures. Relative error near zero is diagnostic only.

Small typed fixtures with independent expected BF16 bit patterns test exact
rounding, negative values, cancellation, and BF16 interpretation. Real weight
basis probes establish weight direction and binding. An independent Candle
F64 dense reference establishes a mathematical comparison for the sampled
inputs. None of these alone establishes parity with a deployed vLLM backend.

## Execution and memory boundaries

The operation uses ordinary `ModelProgram` and `ExecutionPlan` contracts. A
fixed tensor program declares `ProgramNodeWorkSpec::Fixed` and obtains its
input and output shapes at compilation; rows are not token counts.
Static weights, input upload, output retention, and readback use the managed
runtime and its resource ledger.

Reference dense execution locks at most three distinct backing allocations in
address order. It reads scalar values and writes into the admitted output
region without copying whole inputs, weights, or outputs to temporary vectors.
Disjoint regions of one allocation are supported. Output overlap with either
input is rejected during launch validation and whole-batch submission preflight.
Other Reference operations and preparation/oracle allocations are separate
accounting scopes; this change does not establish an overall process RSS bound.
