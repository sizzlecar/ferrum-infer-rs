# Image generation runtime

Image stages compile and execute through the shared `DeviceRuntime` and
`PlanRuntimeResources`. The default `cold` residency policy initializes and
closes each stage per request batch. Both `ferrum run` and `ferrum serve` use
the same typed policy; the local-directory preset remains cold.

The pipeline's `plan_cache` setting defaults to `last_per_stage`. It retains
at most one compiled plan for each of Text Encoder, Noise, Denoiser and Decoder.
Only an identical prepared family fingerprint, stage batch size and compile
options can reuse an entry. This includes the exact attention mask, tensor
geometry and selected numerical profile. The registry, runtime and memory
policy belong to the immutable pipeline instance; caches are never shared
between different instances or devices.

For explicit image configuration, set `backend.pipeline.plan_cache` to
`"disabled"` to compile each batch again. The local-directory preset uses
`"last_per_stage"`. Each stage receipt reports `plan_cache_hit` and the current
lookup/compile wall duration. A compilation cache hit alone does not imply
resident weights.

The cache contains graph metadata and immutable provider bindings. It retains
no tensor payloads, dynamic pool backing or initialized component sessions.
Metadata memory is bounded by four entries; it is not charged to the Host
tensor-payload account. Provider/native-library owners follow their existing
opaque admission and lifetime rules. Cache eviction does not invalidate a
plan already owned by an in-flight request.

`CompiledTensorComponent::initialize` can create multiple independent sessions
from one compilation. Each session uses fresh transactions and passes the
same device capacity checks. Closing one session cannot close another session
or release resources that it owns.

Explicit image JSON configuration can opt into retaining DiT and VAE sessions:

```json
{
  "execution_profile": "experimental_resident_v1",
  "backend": {
    "pipeline": { "residency": "denoiser_and_decoder" }
  }
}
```

These are fields to add to the complete configuration, not a standalone
configuration. The cold profiles require `residency: "cold"` (the default).
Resident mode is experimental and carries no throughput or maximum-shape
qualification. It retains at most one initialized Denoiser and one Decoder.
TE and noise still initialize and close every batch. It uses the existing
device capacity authority without another allocator or memory allowance.
Consequently, retained static weights also consume capacity during TE; an
insufficient device ceiling remains an admission error, not an implicit offload.

Only an identical complete plan can reuse a session. Different geometry,
mask, profile, or provider implementation closes the old session before
initializing its replacement. Historical prompts with different padded text
lengths or validity masks can therefore miss the DiT session cache: this does
not promise that an arbitrary eight-prompt workload avoids uploads. Sources,
runtime, profiles and policy remain fixed within the pipeline instance.

After a resident stage completes, the runtime must release its idle dynamic
backing before the session enters the slot. Its static resources stay charged.
`retirement.kind: "resident"` includes the actual idle-release receipt and
omits `close`; `retirement.kind: "closed"` includes the ordinary `close`
object. `session_reused: true` reports no initialization for this acquisition,
and `initialization_wall_duration_us` is zero. The retained `initialization`
object identifies the session's original initialization; do not count its
uploaded bytes again for reused sessions. Plan-change evictions and shutdown
produce explicit zero-execution close receipts rather than counting previous
execution timings twice.

`QwenImage21Pipeline::shutdown` stops admission and explicitly closes retained
sessions. Returned image Host backing remains usable after that close. CLI run
calls shutdown before PNG encoding. The HTTP worker calls the same hook after
normal shutdown, and its joined shutdown owner retains any original failure.
A fatal generation error bypasses the hook: the original error plus pipeline
remain available for recovery instead of closing around unrecovered work.
Idle-release, execution and close failures stop successful response processing;
they do not publish a fake closed receipt or silently retry another policy.
