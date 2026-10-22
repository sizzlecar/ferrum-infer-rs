# Image generation runtime

Image stages compile and execute through the shared `DeviceRuntime` and
`PlanRuntimeResources`. Each stage currently initializes and closes its own
session, so weights are loaded for each request batch. This behavior applies
to both `ferrum run` and `ferrum serve`.

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
lookup/compile wall duration. A cache hit still performs ordinary session
initialization and resource admission; it does not imply resident weights.

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
