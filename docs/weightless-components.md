# Managed components without stored weights

Tensor components such as activation functions and diffusion schedulers can
declare `ModelWeights::None`. They still compile through `ModelFamilyDefinition`,
`PreparedModelFamily`, and `ExecutionPlan`, and execute with a device resource
ledger, dynamic pools, and a host memory account.

`ModelWeights::Stored(WeightSchema)` represents actual stored weights. Its
existing nonempty schema and required program-reference rules remain enforced.
Causal language families require `Stored`. A `None` family must declare
`ModelSemanticMetadata::TensorComponent` and no program weight references.
Numerical profile selection calls `specialize_weight_schema` only for `Stored`.

## Compilation and authority

`ModelFamilyProvider::weights` declares presence explicitly. Consumers that
require weights use `stored_weight_schema()`, which returns a typed error for
`None`; consumers that handle both cases match `weights()`.

`ProgramPlanCompileOptions` defaults to `WeightPreparation::Auto`. For `None`,
compilation skips materializer selection entirely. For `Stored`, automatic
preparation uses the identity materializer. Each explicit
`require_weight_materializer*` setter records `WeightPreparation::Require`;
every such demand is rejected for `None`, including an explicit identity
materializer or a selection carrying a numeric quality artifact. Direct registry
selection also rejects a family without stored weights before invoking a
materializer.

Plans carry `ExecutionWeights::None` or
`ExecutionWeights::Stored(ExecutionWeightPlan)`. Absence contains no schema,
source map, materializer, or quality receipt. The process-local
`TrustedExecutionWeights` authority is separate from this serializable
declaration. Core creates its absence proof from the complete prepared-family
fingerprint; callers cannot deserialize that proof or reuse it for a different
prepared family.

The plan summary derives `weight_format: None` and an empty quantization-format
set from absence. Semantic bindings and memory plans cannot introduce weight
storage under that declaration. Plan hashes still include family, program,
numerical profile, provider, runtime, and policy identities.

## Resource lifetime

Weight presence does not determine whether a plan has static resources.
`provision_static` examines the actual static allocation list:

- `NoStatic` directly creates the managed plan root without a fabricated
  initialization receipt. Dynamic and host allocations still require admission.
- `Required` commits real static allocations. A weightless caller invokes
  `initialize_static_without_weights(family, plan, policy)` on the committed
  transaction. This shares the ordinary preflight, zero commands, submission,
  fence, recovery, and retirement path. It returns an initialized transaction
  suitable for plan-root handoff.

The source-taking initialization entrypoint requires `Stored`; it rejects a
source passed to a `None` plan. An unknown completion fence retains allocation
ownership and device capacity until quiescence is established. Output owners
and their slices retain the full admitted host backing until the last owner is
dropped, including after closing the plan root.

## Wire compatibility

Prepared-family packages use wire version **4** and execution plans use schema
**10.0**. Older packages must be rebuilt from their model configuration. Strict
decoding rejects stored-only fields attached to `{"kind":"none"}`. Stored
model declarations contain `schema`; stored execution declarations contain
`plan`. Decoding a declaration does not confer execution authority: registry
reconstruction and semantic plan validation remain required.

## Verification scope

The Reference integration fixture executes a real BF16 SiLU-only component with
no checkpoint, source, or materializer. It checks independent expected values,
host and dynamic capacity boundaries, plan/root ownership, output lifetime,
wire tampering, explicit materializer rejection, and pending-fence recovery.

A separate Reference initialization fixture requests a real plan-persistent
buffer, poisons its admitted bytes, and observes their zeroing. It tests
capacity contention and initialization failure/recovery. Its metadata provider
does not implement an executable cached-copy operation; these are initialization
tests. Neither fixture establishes CUDA execution, complete image generation,
or image-serving throughput.
