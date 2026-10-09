# Locked upstream linear artifact

This optional `cuda-upstream-linear` unit uses llama.cpp
`d81235049384534c167caea52b85a694f6103d14` (MIT). Cargo only links a
validated native operator-set artifact. Build the archive separately with
`ferrum-native-ops-builder lock-source` and `source-build`; the source definition
pins each translation unit and its compiler-observed external header closure.
No upstream checkout, downloads or compiler invocation occur in product Cargo.
The external bundle has `kernels/` (these small adapter files), `llama/` (the pinned
external header tree), and `llama-MIT.txt`. Do not vendor the latter tree here.

The package definition describes the SM89/CUDA12.6 qualification target. A
package for another target needs its actual source-build receipt and matching
explicit package metadata. Provider bindings must match the real live G03
catalog; a successful raw ABI test is not provider qualification.

`plan_v1` verifies current device facts, reconstructs all geometry and configures
MMQ shared memory outside capture. Every launch re-derives the plan and rejects
changes. Rust checks byte extents, alignment and aliasing. The caller retains
same-device leases and stream ordering; the ABI performs no device allocation
or synchronization. Positive status is the CUDA runtime error, negative status
is a typed contract/invariant rejection. Upstream host assertions are caught at
the C boundary; they never abort the service or trigger a fallback. Asynchronous
device faults remain observable through the caller's normal completion fence.

V1 preserves the measured upstream fast-math expression and remains numerically
unqualified unless its declared dynamic-domain requirement is implemented.
MarkerV2 adds distinct pack/cast/weight-check symbols. Zero activation groups,
including padding, produce positive-zero metadata/codes. Nonfinite input or
consumed metadata produces zero codes, canonical NaN scale and a row flag.
MMVQ's unused original sum may overflow. Half-scale underflow remains zero.
A preparation-time scan checks all consumed weight coefficients once, writing
a retained per-leaf flag (not a host claim). Final cast observes both flags,
nonfinite F32 output and F16 overflow and writes canonical F16 `0x7e00`.
Flags describe asynchronous propagation, not rejection before dot. The finite
qualified expression, dot kernels, layouts and precision are otherwise retained.

The CUDA-independent ABI tests run by default. The explicit
`ferrum-native-ops/cuda-upstream-linear-tests` feature enables Rust hardware
conformance against a source-builder-produced `libferrum_upstream_linear.a`
provided using the standard linker search path. It has no libloading/env source
selection or test catalog masquerading as a product artifact. Tests separate
actual-pack F64 error, V1-qualified bits and marker/extreme behavior.
