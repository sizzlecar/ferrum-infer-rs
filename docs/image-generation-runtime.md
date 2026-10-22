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
qualification. `denoiser_and_decoder` retains at most one initialized Denoiser
and one Decoder; TE and noise still initialize and close every batch.
Alternatively, explicitly select `residency: "all_model_components"` with the
same `experimental_resident_v1` execution profile to retain the Text Encoder
as well. Noise remains cold. The precision selection does not change residency:
BF16 and experimental FP8 both require this independent opt-in.

Both resident policies use the existing device capacity authority without
another allocator or memory allowance. Every retained component's static
weights stay charged while another component initializes or executes. The
active stage's dynamic backing and native opaque claims must also fit the
same device ceiling. An insufficient ceiling remains an admission error;
there is no implicit offload, quantization or fallback to cold execution.
Neither policy guarantees that three components fit a particular device.

Before execution, resident batches compare the compiled retained static total
plus the largest active-stage requirement and one composition-supplied native
allowance against that ceiling. A nonresident stage includes its own static
weights in the active-stage requirement. Provider scratch, including cuDNN
workspace, is already part of each compiled dynamic plan and is not added
again. A batch that fails this check is rejected before payload uploads and
does not stop the service. This planning check does not reserve memory or
promise availability against other users of the same device; ordinary runtime
admission remains authoritative.

Only an identical complete plan can reuse a session. Different geometry,
mask, profile, or provider implementation closes the old session before
binding its replacement. Resident mode also keeps one explicitly initialized
readonly weight snapshot per retained stage. A replacement with the same
execution weight plan, physical component placements and static layout can
bind that snapshot without source reads, materialization or weight uploads.
The original physical allocation and capacity claim remain single-owned;
each new plan has separate dynamic pools and logical admission. A mismatch
in that physical contract, runtime instance or admission ceiling fails before
attachment, with no silent cold fallback. Sources, runtime, profiles and
policy remain fixed within the pipeline instance. This mechanism does not
infer equal checkpoint contents from paths or schema names, and supports no
mutable static state or device static transforms.

With `all_model_components`, a text sequence or mask change closes the previous
text session and applies the same snapshot compatibility checks to the new
text plan. A successful attachment skips source reads, CPU materialization and
device weight upload. Exact text-plan hits retain the existing session. A
different weight contract is rejected; arbitrary prompts or shapes are not
assumed to produce compatible plans.

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

`weights_reused: true` distinguishes shared physical weights from a new cold
upload. A changed-plan attachment has `session_reused: false`,
`weights_reused: true` and an `initialization.kind: "shared_weights"` receipt
identifying its original sealed installation. Exact-session reuse has both
flags true. Closing a session can release zero static resources because the
snapshot still owns them; this does not mean that its physical bytes are free.

`QwenImage21Pipeline::shutdown` stops admission and explicitly closes retained
sessions, including a retained Text Encoder, then closes the final weight
snapshots. No snapshot is released before all resident sessions have closed.
Snapshot retirement produces
separate `retirement.kind: "shared_weights_closed"` receipts. A referenced or
failed snapshot close retains its typed physical owner for recovery. Returned
image Host backing remains usable after that close. CLI run
calls shutdown before PNG encoding. The HTTP worker calls the same hook after
normal shutdown, and its joined shutdown owner retains any original failure.
A fatal generation error bypasses the hook: the original error plus pipeline
remain available for recovery instead of closing around unrecovered work.
Idle-release, execution and close failures stop successful response processing;
they do not publish a fake closed receipt or silently retry another policy.

The CUDA configuration accepts `backend.cudnn.dependency_libraries`, an ordered
list of up to 16 absolute native library file paths. Empty preserves the system
loader behavior. On Linux GNU, an explicit entry is loaded before cuDNN and
checked against the actual mapped path and SONAME resolution. File contents,
canonical paths and order enter the provider identity. The selected deployment
files must remain immutable for the runtime's lifetime.

Declare compiler dependencies as well as the compiler library when required by
the selected cuDNN installation. For example, the tested CUDA 13 installation
needed `libnvrtc-builtins.so.13.0`, `libnvJitLink.so.13`, and `libnvrtc.so.13`, in
that order, as absolute files from the same installation. Selecting only the
last two loaded successfully but left cuDNN JIT compilation failing. This list
is deployment-specific; it is not an automatic search path or a claim to pin
every transitive dependency. No environment variable is changed by the loader.
Native owners retain these libraries under the declared cuDNN opaque allowance;
that allowance is not a measurement of the native library's peak memory usage.

The independent precision selection and its conversion/storage boundaries are
documented in [image FP8 precision](image-fp8-precision.md).
