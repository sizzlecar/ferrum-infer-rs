# ADR: Runtime vNext Core Contract Boundary

- Status: Historical design input. The 2026-07-13 isolated-contract checkpoint passed, but this ADR
  no longer defines the active G01A completion contract. S0A/S0B and production acceptance are
  superseded by
  [`EXECUTION_STRATEGY_AMENDMENT_2026-07-14.md`](EXECUTION_STRATEGY_AMENDMENT_2026-07-14.md) and
  [`G01_CORE_CONTRACTS.md`](G01_CORE_CONTRACTS.md).
- Scope: contract design only
- Depends on: historical canonical G00a checkpoint `G00a`
- Decision owner: G01
- Migration owner: G03 and G08
- Legacy deletion owner: G08D

## Context

The frozen G00a coupling inventory contains 82 methods on the legacy
`ferrum_kernels::backend::Backend` trait. That trait combines device lifecycle,
buffer transfer, operation capability discovery, mathematical kernels, model-family
policy, profiling, and unsupported defaults. A method added for one model or one
backend therefore changes a shared trait, expands the compile surface, and can move a
failure from planning into a late runtime call.

The historical G01A checkpoint defines an isolated contract. It does not route
`ferrum run` or `ferrum serve`, change runtime defaults, migrate a production model,
or establish a performance result. Its types are candidates for S0A ownership-preserving
split and S0B production-driven rewrite, not frozen public architecture. The authoritative
legacy classification is
[`G01A_LEGACY_CONTRACT_MAP.json`](G01A_LEGACY_CONTRACT_MAP.json).

## Decision Drivers

The boundary must:

1. keep device ownership and failure semantics small and stable;
2. add an existing operation or model family without editing a shared backend trait;
3. resolve capabilities once while building an immutable execution plan;
4. reject unknown architecture, operation version, shape, dtype, layout, or weight
   layout before allocating weights;
5. retain request, plan, node, operation, provider, resource, and event identity;
6. remain auditable and deterministic without `Any`, `TypeId`, string downcasts, or
   product-entry routing; and
7. leave real overhead measurement, the reference runtime, and all five extension
   drills to G01B.

## Alternatives

### Aggregate small capability traits

This option splits the legacy mega-trait into traits such as memory, attention,
normalization, recurrent state, and quantization. Generic model code composes the
traits it needs.

### Typed versioned operation registry

This option describes every executable operation with an `OperationContract`, keeps
portable descriptors in a versioned `CapabilityCatalog`, and owns the exact typed
`OperationProvider<R>` objects in one `OperationRuntimeRegistry<R>`. The planner selects
providers before execution. A small `DeviceRuntime` remains below the operation layer.

## Comparison

| Dimension | Aggregate small capability traits | Typed operation registry plus small `DeviceRuntime` |
|---|---|---|
| Compile-time effect | Splitting reduces a single file, but adding a required method still changes a shared trait and recompiles every generic consumer and backend implementation. | A new operation/provider is additive. Shared contracts change only for a schema-major change; catalog and provider modules are independently type-checked. Actual incremental-build deltas are deferred to G01B. |
| Runtime overhead | Static dispatch is normally free after monomorphization. Capability checks can still leak into the token loop when expressed as `supports_*` methods. | Catalog resolution occurs once in planning. Execution consumes a preselected provider. A provider call may introduce dispatch cost; G01A makes no claim that it is negligible and G01B must measure it. |
| Object safety | Associated types, generic closure methods such as legacy `with_device_ordinal<R>`, and `Self: Sized` patterns make a composed dynamic boundary difficult. | `DeviceRuntime` and `OperationContract` are object-safe once associated runtime types are fixed. `OperationProvider<R>` retains compile-time typed buffers. Metadata and validation cross the boundary without downcasting. |
| Error localization | A missing method or permissive default can surface at the first execution of a rare branch. Separate capability booleans can disagree with the implementation. | Missing operation id/version or incompatible shape/dtype/layout is reported by the planner with model, node, operation, provider, and reject reason before execution. There is no successful unsupported default. |
| Extension cost | An existing-op family still needs a new generic trait composition and can require changes in shared bounds. A novel op edits a shared trait and all implementors. | An existing-op family adds a provider/program/registration/fixture. A novel op adds a contract/provider/oracle/catalog entry. Core runtime and product entrypoints remain unchanged. The 5/5 proof is deferred to G01B. |

## Decision

Adopt the hybrid: a small stable `DeviceRuntime` for device primitives, a portable
versioned `CapabilityCatalog`, and a typed `OperationRuntimeRegistry<R>` that owns the
exact contract/provider implementations for both planning and dispatch.

The stable contract graph is:

```text
ModelFamilyProvider::Config
  -> PreparedModelFamily(config + weights + program + metadata)
  -> ExecutionPlanner(RuntimePolicy, CapabilityCatalog, physical bindings)
  -> immutable ExecutionPlan
  -> registry-bound OperationProvider<R>
  -> DeviceRuntime

CapabilityCatalog + OperationContract
  -> OperationOracleRegistry
  -> registry-bound CPU/high-precision oracle
```

The crate-owned `ResourceTransaction<D, Stage>`, its non-cloneable committed lease,
and `ExecutionEventSink` are peers of execution rather than backend extensions.
`ResolvedModelPlan` is the validated, data-only product composition result; G01A does
not connect it to a product entrypoint.

### 2026-07-19 exact completion timing amendment

Real CUDA S1 evidence showed that a host-only completion interval cannot distinguish
device execution, fence wait, and readback. Timing policy remains selected through the
typed `ExecutionEventSink`, but the exact request is copied into `DeviceCommandBatch`
and the result is inseparable from the quiescent terminal receipt. This is a bounded
extension of the device primitive contract, not an operation-provider or model-family
branch.

The amendment has four invariants:

1. `Off` creates no backend start event, takes no host clock reading, and allocates no
   timing side channel on the submission hot path.
2. `Completion` records at most one backend start event per physical submission; the
   existing terminal fence event is the end event.
3. Device elapsed time can only be consumed with the exact terminal receipt. There is
   no second fence query and no timing owner outside completion/reaper ownership.
4. Device elapsed, host fence wait, and host readback use distinct clocks, may overlap,
   and must never be summed into a fabricated total.

## Frozen Contract Rules

### Object safety and typed dispatch

- `DeviceRuntime` is the small object-safe device boundary after its associated buffer,
  stream, command, fence, and error types are fixed. It owns allocation, typed host/device
  and device/device copy, stream creation, submit, fence query/wait and recovery drain. A
  submit error is a sealed `DefinitelyNotSubmitted<E>`; once any work may be device-visible,
  submit must return a fence and report later failure as `FailedButQuiescent` or
  `Indeterminate`. Only a quiescent terminal proves buffers release-safe. Core, not the
  backend, attaches exact execution identity and the fixed `Device` failure domain. Blocking
  fence wait and lane drain are recovery/shutdown correctness tools, never normal request
  progress. Device timing policy is selected through `ExecutionEventSink` and the
  `OperationContract::profile_phase` identity. Under the exact-completion amendment,
  `DeviceCommandBatch` carries the typed request and the terminal receipt carries the
  optional measurement or typed unavailability reason; timing never becomes a global
  backend switch or a second post-terminal query.
- `OperationContract` is object-safe metadata and validation. It contains an operation
  id and schema-major-compatible version plus shape, dtype, layout, typed attribute
  schema and constraints, resource, oracle, and profile-phase requirements.
- `OperationProvider<R>` is generic over a concrete `DeviceRuntime`, preserving typed
  buffers without `Any`. It must implement a versioned resource estimator and bind the
  estimator id, version, implementation SHA-256 and exact semantic estimator-input
  fingerprint into its selected `ProviderResourcePlan`. It does not perform catalog
  lookup during execution.
- `CapabilityCatalog` is compile-time assembled portable metadata. Every provider row
  binds the exact operation descriptor fingerprint, device, version, capabilities,
  accepted weight formats, and accepted quantization formats. Empty or mismatched rows,
  duplicate provider identities, incompatible reference-oracle signatures, reference
  cycles, oversized catalogs, and over-depth reference chains are errors.
- `OperationRuntimeRegistry<R>` owns the exact `OperationContract` and
  `OperationProvider<R>` trait objects used for planning and execution. It issues a
  process-local, non-serializable authority through `OperationPlanningHandle`; every
  `PlanNodeResolution` carries that authority and one plan may contain only one. The
  immutable plan retains it outside its serde payload and hash. A byte-identical second
  registry can independently produce the same deterministic plan bytes, but cannot bind
  or dispatch a plan built by the first registry.
- `OperationOracleRegistry` owns one callable, independently anchored oracle for every
  terminal non-reference operation and resolves compatible reference-operation chains
  before invocation. Oracle descriptors bind oracle id/version/implementation SHA-256 and
  the exact operation id/fingerprint. Requests/results use bounded canonical row-major
  little-endian host tensors; trusted descriptors, tensors, requests and results are
  Serialize-only, and untrusted decoders enforce raw byte limits before serde.
- `OperationDispatch` consumes one non-cloneable `InvocationResourceLease` for an exact
  continuous batch and one scheduler-owned `ExecutionLane`; it never borrows a per-sequence
  stream. The invocation contains a canonical non-empty participant set, a core-derived
  `BatchWorkShape`, exact per-participant Step/Sequence/Request authority, batch node/provider
  identity, Request-state
  hazards and batch scratch claimed once. `DynamicResourceShape` is an internal evaluated
  projection and has no public constructor or public admission path. A caller supplies only
  core-issued token-span, full-input and committed-page-set authorities; core validates them
  against the exact participant/frame and the plan node's `NodeWorkContract`, then derives the
  immediate/fit metric vectors and their canonical fingerprint. Dispatch verifies that the
  derived sequence count equals the participant count and that all participants bind the same
  plan/hash/device/runtime/coordinator/node/provider before privately constructing a borrowed
  `BatchedOperationInvocation`. It derives checked physical regions from immutable plan-static
  resources and exact committed dynamic leases; no ceiling-derived slot offset, copied resource
  id or canonical leader is authority. It re-reads every runtime buffer descriptor and validates
  input/output ordinal, shape, dtype, layout, access, alias, capacity, generation, scratch,
  persistent workspace and typed attributes.
- Before provider encode or device submit, dispatch reserves a durable completion-reaper record
  that owns the invocation lease, participant-flight holds, hazards and lane `Arc`. Provider
  encode may only produce a command. `DefinitelyNotSubmitted` releases the prepared record
  without emitting submitted evidence; `Ok(fence)` installs the fence without allocation and
  only then publishes `OperationSubmitted`. The returned handle is a ticket to durable
  ownership, so dropping it cannot release device-visible resources. Raw commands never escape
  for submission on another runtime or lane. Provider errors are accepted only when their
  execution identity and profile phase equal the dispatch context.
- A plan node records the selected provider identity. The execution loop neither asks
  `supports_*` questions nor branches on a concrete backend or architecture.
- Every shape-dependent plan node records a core-derived `NodeWorkContract` in the immutable plan
  and plan hash. The model program explicitly identifies the token-bearing value and axis; core
  resolves that source to the exact `ResolvedValueBinding` role/ordinal/rank. Each page metric is
  bound to an exact state/resource/pool/storage profile and page geometry derived from committed
  state and selected storage, never from a caller-supplied page count. Missing, ambiguous or
  incompatible token-axis/page-metric sources reject plan construction instead of guessing from
  tensor element count, a model name or backend type.
- Exact alias equivalence and state read/write effects are immutable plan data.
  `MustAlias` is one exact storage range; `MayAlias` accepts only exact equality or a
  disjoint range. Partial overlap and overwrite before the final input consumer are
  rejected. State effects add RAW, WAR and WAW dependencies while independent reads
  remain unordered.

### Ownership

- `DeviceRuntime` owns device handles, non-cloneable buffer/resource leases, stream
  lifetimes, copy/submit/sync, and device failures. It does not own model policy,
  provider selection, profiling policy, or product config.
- `CapabilityCatalog` owns immutable operation/provider descriptions. A provider owns
  one operation implementation, not model routing or resource admission.
- `ModelFamilyProvider` owns typed config, weight schema, `ModelProgram`, chat template,
  EOS, and other model semantics. Physical weight schemas explicitly model packed
  values, scales, zero points, indices, permutations, codebooks, repacking, and expert
  stacks without naming an architecture. It cannot inspect a concrete backend type.
  Every registration declares its exact external metadata ids. Core, not a registry
  implementation, resolves both internal family id and external metadata id from one
  registration set, rejects unknown or duplicate matches, and requires both identities to
  select the same trait object. Typed parsing must preserve every supplied raw config field
  and value; explicit typed defaults may add fields, but a parser cannot silently consume,
  rename, coerce, or discard input. The provider separately validates architecture-specific
  identity fields such as `model_type` before a prepared family is signed.
- `StateSpec` contains only a logical tensor, Request/Sequence/Step lifetime and fixed-per-scope
  or token-scaled capacity demand. Physical pages are an allocator quantum, not model semantics;
  semantic `PageScaled` is rejected. The model never names allocator, physical view, pool,
  provider, backend or device. Selected provider binding requirements, device storage offers and
  the backend-neutral ordered runtime policy are jointly solved by core, so the same program can
  use fixed-block+paged CUDA storage or linear+contiguous Metal storage without changing its
  program fingerprint.
- `ExecutionPlanner` owns capability resolution, memory planning, structured reject
  reasons, and construction of an immutable `ExecutionPlan`. The core builder consumes
  `PreparedModelFamily + CapabilityCatalog + RuntimePolicy + PlanNodeResolution`; callers
  cannot supply a `MemoryPlan`, plan id, provider compatibility result, program/catalog/
  policy fingerprint, or plan hash. Each `PlanNodeResolution` is supplied as independent
  trusted planner evidence and is checked against the selected provider's exact
  estimator. Core validates per-node weight formats plus weight/state shape, dtype,
  layout, explicit blocked-layout padding and physical components, then derives value
  allocations plus every scratch/persistent requirement, alignment and lifetime scope.
  Plan-resident resources become exact static allocations. Sequence-, request-, step- and
  invocation-scoped resources remain one `O(graph)` descriptor per graph requirement with
  a core-validated actual-shape formula over admitted sequences, tokens, pages or bounded
  shape buckets. Provider estimators cannot observe or multiply the global concurrency
  ceiling. The planner derives minimum-runnable and wide-integer theoretical-ceiling
  evidence, but neither value is a preallocated slot table or a normal admission claim.
  Every dynamic descriptor also carries a core-derived storage contract and pool id. The
  canonical pool compatibility key binds allocator kind, contiguous/paged view, usage,
  element type, layout fingerprint and alignment; provider code cannot choose the pool id.
  `MemoryPlan` owns the canonical pool specs, per-pool resident minimums and invocation
  liveness/reuse proof, and signs them into the plan hash. The resource layer consumes these
  specs directly and may not rediscover pools from descriptors or divide leftover bytes.
- `ResourceTransaction<D, Stage>` exclusively owns reserve/commit/rollback/release
  transitions for plan-static backing only. `ExecutionPlan` issues a non-cloneable static
  provisioning permit that binds plan id/hash, device id, runtime implementation SHA-256,
  raw/usable capacity, exact static bytes, a unique resource-pool id and a monotonic
  provisioning generation. A zero-static plan uses an explicit no-op provisioning outcome;
  neither static nor no-op provisioning can create a sequence, execution stream or dispatch
  authority. The plan also derives the exact logical-admission coordinator from its dynamic
  descriptors and typed backing-domain catalog. The committed pool and every logical lease
  share that coordinator through the same core-owned `Arc`; equal copied descriptors or an
  independently constructed coordinator are not authority. The process-global device account
  rejects live static pools that reuse a device id with a different runtime implementation or
  raw capacity. It sums static backing claims against the minimum usable-capacity ceiling
  declared by every live pool, so a second plan cannot consume another live plan's typed
  reserve. Transaction begin verifies the driver's exact device/runtime/capacity. A driver
  cannot mint a commit
  outcome: it asks `ResourceTransactionContext::allocate` to consume the exact
  one-shot allocation authority and call the matching `DeviceRuntime`. Any successful
  allocation is installed in core-owned pending storage before control returns to the
  driver; the call-scoped receipt contains metadata only and cannot escape the commit.
  A driver error, dropped/forgotten receipt, or unwind therefore leaves the actual buffer
  in the pool ledger for reconcile, quarantine, or abandon. Reconcile receives only a
  borrowed view and never takes buffer ownership. Core owns actual-state progress records
  and buffers, including the backend's actual descriptor on poisoned returns. It validates
  resource, generation, size, alignment, usage and element type
  before creating the non-cloneable committed static lease. Partial reserve/commit effects are
  compensated incrementally in strict reverse order. Rollback and release are irreversible
  cleanup and therefore resume only forward; completed cleanup is never "undone". Invalid
  or ambiguous ownership transfers retain their device-capacity claim and must reconcile,
  compensate or quarantine. Quarantine and abandon transfer the whole pool, its static capacity
  claim, exact coordinator `Arc` with logical claims/listeners, abandon signal and
  sequence-recovery registry by value. Dropping any nonterminal transaction invokes one
  non-blocking abandon transfer; no transaction or permit destructor calls a device
  synchronization API. The external abandon callback is panic-contained. If it panics during
  another unwind, the pool owner conservatively retains every buffer, static and logical claim,
  runtime, coordinator/listener and recovery-registry lifetime instead of allowing a second
  panic to abort the process or treating uncertain cleanup as success.
- Plan lifetime is rooted in one owning `Arc<PlanRuntimeResources<R>>`, never a borrow of
  `NoStatic<R>` or `StaticProvisioningLease<R>`. `NoStatic::into_plan_runtime(self)` and
  `ResourceTransaction<D, TransactionCommitted>::into_plan_runtime(self)` are the only handoffs;
  the latter retains the driver, ledger, buffers, capacity claim and recovery authority behind the
  root. A static lease alone cannot mint a trusted binding. `TrustedPlanRuntimeBinding<R>` holds the
  root `Arc`, and the owning Request -> Sequence -> `SequenceSession` -> Step -> Invocation chain has
  no `'plan` parameter and is `Send + 'static` (with every `Arc` target also `Sync`). A borrowed
  operation view may remain call-scoped, but cannot enter durable completion ownership.
- Plan close first changes the root from Active to Closing, then returns typed
  `Referenced { resources, strong_count }` while any request, invocation or reaper record retains
  it. After unique ownership is recovered, dynamic pools close before static buffers and the static
  capacity claim; each resident chunk drops its buffer before its `DeviceCapacityGrant`. Partial
  release retains a retry/quarantine owner, and fallback Drop reuses transaction abandon semantics.
  The root never strongly owns the completion reaper: the engine owns them as siblings or uses
  `Weak`, so `root -> reaper -> record -> invocation -> root` is impossible. The reaper is
  `CompletionReaper<R>`; a temporary `CompletionReaper<'plan, ...>` contract is prohibited.
- Runtime policy construction rejects only zero concurrency; `1`, `32`, `4096` and `u32::MAX`
  ceilings produce the same `O(graph)` descriptor and build-allocation counts. The immutable
  plan separates plan-resident allocations from dynamic demand formulas and may retain a
  wide-integer theoretical ceiling as evidence, but never iterates, reserves, allocates or
  claims resources from that ceiling. `MemoryPlan` declares per-pool minimum/maximum and a typed
  elastic provisioning policy, not a fabricated current-free byte count. The global device account
  grants actual initial chunks at transaction time from then-live capacity; the runtime binding and
  artifact record each exact buffer identity, generation, ordinal and granted size. Published domain
  capacity is exactly installed usable chunk bytes.
  Per-request or per-batch admission atomically claims actual KV pages, recurrent slots and
  workspace slices from non-interchangeable typed capacity domains. Explicit grow appends real
  chunks through global claim -> allocate -> validate/install -> publish; steady admission and
  decode cannot perform per-request device allocation.
- One admission demand keeps four meanings separate. `immediate_claim` is the actual-shape
  capacity vector deducted now. `fit_requirement` proves that the selected frontier or full
  input fits current backing under the typed fit policy, but does not claim future units.
  Max-context and concurrency ceilings are policy limits and consume no capacity. An optional
  `future_reservation`, if a later product policy enables it, is a fourth independently accounted
  vector that must be returned explicitly; absent that policy its value is exactly zero. A
  full-input fit therefore must not be reported as a future reservation or used to hide
  overcommit policy.
- Dynamic admission has three typed outcomes: `Admitted` with a committed logical lease,
  temporary `Deferred` with exact immediate/fit vectors, per-domain requested/available/maximum
  shortfalls, retry action, global audit epochs and a canonical exact-source `CapacityWaitCondition`,
  or permanent `Impossible` when the request cannot fit even exclusive maximum backing or violates
  a typed ceiling. A deferred or impossible value cannot construct the sealed authority accepted by
  provider encode, device submit or prefill. A large deferred request does not mutate capacity and
  cannot prevent a later eligible request from admission.
- Active-sequence authority is a sparse coordinator-issued id plus a monotonically changing
  generation; callers cannot choose an index. Coordinator storage grows only with actual live or
  reusable records and never allocates a `Vec` from the configured ceiling. Releasing a logical
  claim returns its exact capacity-domain units, increments that domain's availability generation
  and advances global audit `release_epoch`; sequence release also advances the active-slot
  generation. A backing grow, extent availability change or other increase to effective capacity
  advances the relevant domain generation and global audit `capacity_epoch`. A waiter installs the
  shared listener, snapshots the exact sources, rechecks its own generations, then parks. Global
  epoch changes may notify all listeners, but only a relevant source or policy change permits an
  admission probe. This closes the check/register/park lost-wakeup window without polling, a busy
  loop, unrelated-domain retry amplification or a global head-of-line block.
- `AdmittedRequestResources` owns Request-scoped physical/logical state once. Each
  `AdmittedSequenceResources` owns an `Arc` parent and a coordinator-issued sparse id/generation.
  Each admitted sequence also owns one fail-closed execution-authority source selector. The first
  successful `SequenceSession` open or legacy stream activation permanently selects that source;
  completion, abort, synchronization, abandonment and recovery never reset it. Session open and
  legacy activation linearize on the same selector before publishing authority, and selector or
  source-state poisoning rejects both paths. A single sequence therefore cannot mint byte-equal
  active receipts from both the session and legacy authority systems, even across terminal epochs.
  A live `SequenceSession` freezes the exact run/request/plan/coordinator/sequence generation and
  active-session fingerprint for its lifetime. Step and invocation authorities retain owning
  session/sequence parents, while session completion or abort requires its own participant-flight
  count to reach zero; it cannot mutate identity, borrow another session's resource authority or
  drain an unrelated shared execution lane to manufacture quiescence. A participant flight is
  `Prepared` at invocation admission and transitions atomically to `InFlight` across the complete
  canonical participant set before provider encode. That transition and `request_cancel` lock the
  same exact session state: if cancellation linearizes first, provider encode and device submit are
  unreachable; if dispatch linearizes first, the already-started work reaches a typed terminal and
  the cancelled Step discards its result. Partial participant transitions are fail-closed.
  `ExecutionBatchParticipants` canonicalizes a non-empty actual participant set without cloning
  logical leases; all entries must share exact plan/hash/device/runtime/coordinator identity.
  A batch child capacity lease charges one demand vector and installs a release edge on every
  parent sequence. Dropping any parent early fails closed; using a canonical leader as the only
  capacity parent is forbidden.
- `StepResourceLease` owns the exact participant set plus one Step guard/state hold per participant.
  `InvocationResourceLease` owns an exact non-empty participant subset, the Step `Arc`, one batch
  child capacity lease, one batch scratch extent set, and Request-state hazards. The actual shape's
  sequence count equals the exact participant count. Core-minted `BatchStepId` and
  `BatchInvocationId` identify the physical scheduler step/submission while each participant
  retains its own continuous `ExecutionFrameId`, request-journal node identity and active-session
  fingerprint; participant-local ids need not be equal. Any Nth-parent/extent/hazard failure rolls
  back every earlier effect before provider encode/submit. Field ownership and terminalization
  ensure physical extents are retired before logical claims and all parent `Arc`s remain live.
- Shape evaluation produces one private `ClaimedBackingTransaction` that owns the immutable
  `BatchWorkShape` authority/fingerprint together with its exact evaluated demand, logical batch
  capacity claim and physical backing extents. Step, invocation and operation construction carry
  that transaction rather than copying aggregate counts. Even an empty demand retains the work
  authority. Dispatch must prove that every dynamic extent and provider workspace equals the bytes
  evaluated from the same immediate shape; checking only a minimum size or substituting work after
  claim is invalid.
- `ExecutionLane` is scheduler/device authority backed by `Arc<ExecutionLaneInner<R>>`; its stream
  is not bound to a request or sequence. Host submit locks the lane only long enough to enqueue;
  fence query does not lock or synchronize the stream. Every in-flight record holds the lane `Arc`,
  so the raw stream cannot be destroyed before all associated fences become quiescent or the lane
  is quarantined. Sequence terminalization waits only for that sequence's participant-flight count
  and never drains work for unrelated requests on the same lane.
- The completion reaper owns every submitted invocation before `OperationSubmitted` becomes
  observable. Normal progress uses non-blocking fence queries. Terminal success commits Request
  state generations; terminal failed-but-quiescent releases scratch but poisons/discards affected
  writes. An indeterminate fence moves to bounded blocking wait in the reaper, then an independent
  lane-drain recovery worker; drain failure quarantines the lane, extents and write state. An
  external in-flight handle Drop only detaches or requests cancellation. Final reaper/sequence
  owner Drop performs no backend call and transfers each exact non-quiescent owner to a
  process-reachable, plan-domain cleanup registry. The scheduler invokes bounded maintenance from
  an explicit recovery thread; every selected task is attempted at most once per pass, the global
  registry lock is released before backend code, and one blocked task remains available separately
  from sibling lanes. Retry, panic and quarantine leave the same owner reachable. Cleanup pressure
  `64` is an independent recovery-backlog threshold, not a request/model/concurrency limit: the
  registry accepts every additional owner without loss, then rejects new execution-authority
  derivation until maintenance reduces pressure. Plan close reports the exact pending cleanup
  status and can converge after retry; there is no hidden cleanup thread or destructor-side wait.
- A temporary admission defer owns no logical claim and cannot enter dispatch. Once admitted,
  cancel/release returns only exact dynamic capacity after that sequence has no participant-flight
  holds. Failure quarantines the smallest authority whose quiescence or state validity is unknown:
  exact invocation extents/write state first, then execution lane if fence state is indeterminate.
  It must not poison an unrelated sibling sequence or the entire shared backing pool merely because
  one request was cancelled. Whole-pool quarantine is reserved for unprovable physical buffer,
  allocator, static-state or runtime ownership contamination. Any quarantine transfer is typed,
  retry-owning on failure, preserves every global/logical claim and remains observable until a
  trusted close receipt proves cleanup.
- `ExecutionEventSink` owns typed events only. Request events use a typed participant identity
  envelope with run/request/plan/frame/node-invocation/node/operation/provider/device,
  resource-pool, logical coordinator, sparse sequence id/generation, runtime implementation,
  batch fingerprint, participant index, completion id, span and async-link identities. A separate
  `BatchOperationIdentity` binds the canonical participant set and exact plan/node/provider/lane;
  one physical submission creates one batch receipt and linked participant receipts. A typed emitter
  transitions from pre-plan to a plan-bound state using
  `TrustedExecutionTopology::from_plan`. Every execution frame starts at one and advances
  contiguously; each frame executes every trusted plan node exactly once, node invocation
  ids are globally monotonic, and dependencies must complete in the same frame. A pristine
  Step that proves `RollbackUnsubmitted` restores each participant's unexecuted frame; its
  retry uses a fresh `BatchStepId`/physical attempt id but the same participant-local
  `ExecutionFrameId`. Abort, arbitrary Drop, prepared/in-flight work, or an indeterminate
  submission cannot rewind a frame. Repeating
  a plan node in later decode frames is valid; repeating an invocation is not. Node and
  operation events require a live typed sequence session, whose active fingerprint cannot
  be substituted with a resource-journal hash. Active operation invocation and failure
  identities reject both completed- and aborted-sequence fingerprints, so terminal evidence
  cannot be relabeled as in-flight work. `OperationSubmitted` consumes the exact sealed dispatch
  receipt only after the fence is durable. `OperationCompleted` or terminal failure consumes the
  exact fence completion receipt. Identity schema `3.0` replaces the ambiguous `NodeCompleted`
  transition with `NodeRetired`; it consumes the matching participant completion outcome, and
  submission alone is insufficient. `SequenceCompleted` requires that
  sequence's participant-flight count to be zero, not a shared-lane synchronization receipt.
- A separate physical batch ledger owns exactly one `Prepared -> InFlight -> Retired` path, or
  `Prepared -> NotSubmitted`, for each trusted batch invocation. Its collision key is the canonical
  `ParticipantNodeKey = (sequence authority, request authority, ExecutionFrameId, NodeId)`, not
  `BatchInvocationId` or participant-local node invocation id. Preparing a batch atomically reserves
  every key; overlap with any `Prepared`, `InFlight` or `Retired` key rejects the whole attempt and
  retired tombstones remain until Step terminalization. The `Prepared -> InFlight` transition first
  installs the durable fence and only then publishes the external Submitted receipt.
  `DefinitelyNotSubmitted` is the only
  transition that mints a sealed retry authority; retry requires the exact topology and work-shape
  fingerprints and a fresh attempt id. A whole pristine Step rollback follows the same distinction:
  it changes the physical attempt identity without creating a gap in the request journal's frame
  identity. Provider failure, arbitrary Drop, a changed shape or a
  possibly-submitted error cannot relabel a key as retryable. Request journals contain only
  participant projections linked to that shared submission/completion fingerprint. They cannot
  construct, duplicate or count a physical command/fence. Replay requires closure over every
  referenced physical batch record; projections without that ledger are invalid. Bare submission
  receipts are not replay terminal evidence. Replay consumes exact completion, recovery-drain or
  quarantine receipts under one cross-type completion-slot uniqueness set, and every receipt must
  be used exactly once. A receipt with no installed submission fence cannot justify an
  `OperationSubmitted` event. Successful completion cannot coexist with an operation failure;
  failed-but-quiescent completion must equal the observed `IdentifiedFailure`; contract failure,
  drain and quarantine require a failed request and bind their reason or recovery cause into the
  replay fingerprint. A quarantine receipt carries process-local freshness shared by all clones;
  a later successful drain invalidates it before publishing the drain receipt. Current quarantine
  evidence is pending ownership and is accepted only with `AllowPending` plus pending plan cleanup;
  it cannot be combined with clean/closed replay evidence.
- Failure observation and cleanup are distinct. `FailureObserved` records the first exact
  identified failure and freezes further node execution. An active failed request must then
  produce either `SequenceCompleted` or `SequenceAborted`; the latter consumes the exact
  poison receipt. `RequireClean` replay additionally consumes a core-issued plan close or
  quarantine receipt bound to the exact owning root; an abort receipt alone proves stream
  quiescence but never proves logical/static cleanup. `AllowPending` without that root receipt
  is recorded as `CleanupPending`, not relabeled as clean sequence quiescence.
  `RequestFailed` references the first-failure fingerprint from a new terminal event identity,
  rather than replaying an old sequence/span. A pre-plan failure may terminate directly.
- Long-lived resource-pool events use a separate pool journal/cursor with their own
  open/close lifecycle and provisioning identity. It validates transition and lease
  receipts, global generation ownership, and failure/recovery continuation by exact
  failure id, failure point, completed prefix, strategy and ledger. Transition, lease, failure
  and recovery construction and untrusted revalidation bind the supplied transaction identity
  and static provisioning exactly to `ResourcePoolEvidence`; sequence-linked events also bind
  the exact logical coordinator, sparse id/generation and dynamic lease. Failure events require
  the resource failure domain. `ResourceFailed` accepts only an incomplete recovery anchor, while
  `ResourceRecoveryCompleted` accepts only completed recovery evidence; relabeling one kind as
  the other is rejected before a trusted event exists. Pool close accepts only a trusted
  `ResourceLedgerSnapshot` bound to
  that same transaction and static-provisioning identity, with a non-empty,
  generation-matched, buffer-free terminal ledger.
  The cursor separately proves journal continuity and absence of pending recovery. A user request links to
  a pool through its active binding but does not own or close that pool; request completion
  therefore never pretends shared plan resources were released. Both cursors reject
  non-contiguous sequence, non-increasing timestamp, invalid span ancestry and events after
  terminal state. Static replay binds both its complete pool journal and exact root cleanup
  receipt. No-static replay carries no fabricated pool id or journal fingerprint and requires a
  matching zero-static root close receipt when clean replay is requested. A quarantine without a
  core-issued root quarantine receipt is pending or rejected according to replay policy. Failures
  use exact domain-specific `IdentifiedFailure` shapes; a bare
  message cannot enter either stream. `DisabledEventSink` is explicit; event identity is
  never reconstructed from stderr.
- `ResolvedModelPlan` contains exactly one validated `PreparedModelFamily`, source file
  hashes, tokenizer semantics, full device/catalog, runtime, engine, generation policy,
  execution plan, and fingerprinted decision evidence. Config, program, weight schema and
  template are not duplicated. Initial construction and wire revalidation use the same
  external evidence context. A typed family config is serialized once; the same canonical
  value that passes raw-field preservation is signed into the prepared package. The provider
  also signs the exact external metadata identity represented by that config, so two aliases
  registered to one family cannot be substituted without explicit provider evidence. Raw
  source bytes are parsed twice by an identified parser with a version and implementation
  SHA-256. Parser descriptors can only deserialize through their validated constructor and
  every descriptor read is revalidated; identity/document drift is rejected, and core alone
  signs artifacts and chosen-value fingerprints. Provenance is either one exact
  locked model file (path, size and SHA-256) or a structured upstream producer identity;
  there is no free-form self-attested source. Evidence construction performs bounded
  source/provenance/path preflight only; initial plan construction and every wire
  revalidation each parse identical bytes exactly twice. Before the built-in JSON parser
  constructs a `Value`, a streaming `DeserializeSeed`/visitor enforces decoded depth, node,
  key/string-byte, single-root and trailing-input budgets. Source, resolved-wire, JSON
  depth/node/text, provenance and field-path limits are therefore enforced without first
  allocating an over-budget tree or recursively canonicalizing it. Every declared field must have exactly one
  used artifact/path binding and unused evidence is rejected. The result validates token
  bounds and explicit role-collision policy, resolves the engine through the same catalog,
  and cross-validates every family, device/runtime implementation, per-node format,
  provider, policy, external node resolution and plan link.
  CLI and server adapters may consume it only in later goals through one composition root.

### Determinism and plan hashing

`ExecutionPlan` hashes a deterministic semantic serde JSON payload with SHA-256. The
payload uses ordered vectors, sets, and maps; it includes the exact schema, family and
device/runtime-implementation identities, core-computed program/catalog/runtime-policy fingerprints, model
weight and quantization formats, ordered operation nodes and descriptor fingerprints,
typed attributes, selected provider and typed fallback/reject decisions, estimator
identity/input/output, resolved value bindings, static allocations, dynamic actual-shape
formulas, typed capacity domains, policy ceilings and stable node identities. Runtime
admission decisions, epochs and live claims are excluded. Plan id is derived from the hash and
is intentionally excluded from its own hash input. Pointers, process ids, timestamps,
allocation addresses, caller-chosen fingerprints, mutable counters, and the process-local
registry authority are excluded.

Deserialization is not trusted. `ExecutionPlan` and `PreparedModelFamily` are
Serialize-only trusted types. JSON first becomes an explicit `Unvalidated*` wire;
  `PreparedModelFamily` is regenerated through the typed family registry and compared
field-for-field, while `ExecutionPlan` is rebuilt from that family, the current catalog,
typed runtime policy and independently supplied physical node resolutions. The entire
rebuilt payload and hash must match. Thus an attacker cannot mutate memory/provider/model/
estimator fields and merely recompute SHA-256. `ResolvedModelPlan` performs both
reconstructions using externally verified source artifacts and node resolutions before
checking its own fingerprint and decision provenance. Trusted failure, resource receipt,
event and replay types are likewise Serialize-only or require an explicit external
validation context. Public top-level wire-entry `Unvalidated*` values do not implement
  `Deserialize`; private wire structs are reachable only through decoders that check raw
  payload size before serde. Execution-plan, resolved-plan, execution-event and resource-event
  decoders also serialize the complete unvalidated value back to canonical JSON and require
  exact equality with the parsed raw value, closing unknown fields inside nested enum/receipt
  payloads even when a nested serde type would otherwise ignore them. Equal normalized inputs
  must produce the same bytes and hash; a
breaking schema major is rejected before reconstruction.

### Fail-closed behavior

- Unknown architecture/config/weight layout does not fall back to Llama, ChatML, a
  default provider, or a host reference path.
- Missing operation/version, incompatible shape/dtype/layout/attribute/weight format,
  incompatible provider, and unavailable resource contract return a structured planning
  error before weight allocation. Numeric absolute/relative oracle tolerance is rejected when
  any possible output dtype is boolean, because boolean tolerance has no comparison semantics.
- Unsupported operations have no success default. A reference provider, when one is
  intentionally registered, is an explicit catalog entry with its own identity and
  oracle.
- Backend/provider/resource errors retain versioned run, request, plan, frame, node
  invocation, operation, provider, device/runtime, pool, transaction, resource, generation,
  activation epoch and span identity; they are not
  flattened to a log string. Resource states and compensation are typed records rather
  than free-form `from`/`to` strings.
- Validated transition/event/failure receipts are Serialize-only. Deserialize is limited
  to explicit unvalidated wire structures whose constructors recheck identity, order,
  batch fingerprints, generation and lifecycle closure; unknown JSON fields are rejected.

### Prohibited escape hatches

The vNext contract must contain no `std::any::Any`, `TypeId`, string-based downcast,
architecture-named generic symbol, hidden-environment product decision, or `run`/`serve`
routing. Opaque typed ids are allowed; converting an opaque id into an unvalidated
concrete type is not.

## Legacy Disposition

Every one of the 82 G00a `backend_trait_method` findings is classified as a stable
device primitive, versioned operation, model semantic, or dead code. There are no
unmapped entries and no special-case bucket. `owner` in the map identifies the target
contract owner; G08D owns removal of every mapped legacy method after its provider or
semantic migration has passed the applicable model/backend gates.

Capability booleans do not survive as independently mutable runtime facts. Device
capabilities become typed `DeviceRuntime` facts; operation availability is derived from
`CapabilityCatalog`; family policy becomes `ModelProgram` requirements. The map does
not claim that migration has happened.

## G01B Measurement And Proof Plan

G01B, after full G00 PASS, must produce the evidence that G01A deliberately does not:

1. Implement the minimum reference runtime/provider/catalog/planner path and bind it to
   the same contract blob accepted by G01A.
2. Run all five extension drills from `G01_CORE_CONTRACTS.md`; each drill records files
   changed by ownership area and proves the required shared-runtime/product-entry zero
   change counts.
3. Measure dispatch with the same release binary on one host: direct typed call versus a
   preselected `OperationProvider<R>` call. Registry lookup is measured separately and
   is forbidden inside the timed token loop. Use at least 30 independent samples after
   warmup, save raw samples and commands, and report paired median ratio plus a 95% CI.
4. Measure `DisabledEventSink` and enabled basic sink against the same no-event baseline.
   The G01 gates remain `<=1%` and `<=2%` overhead respectively; a noisy or incomplete
   run is REJECT, not PASS. The current validation cursor copies its candidate state before
   commit; profiling must explicitly attribute that cost and test journal-length scaling. If
   meeting the bound requires changing the sink/cursor trait or serialized contract, G01A must
   be rerun and its checkpoint refreshed rather than reusing stale contract evidence.
5. Use G00 build-timing inputs to compare an added provider/family fixture with a shared
   contract edit. Save invalidation sets and five independent samples per build scenario;
   G01A's qualitative compile-time comparison is not performance evidence.
6. Run deterministic plan construction and schema round-trip `100/100`, including a
   forged but self-hashed memory payload, incompatible major, unknown operation,
   ambiguous provider, typed-family drift and pre-allocation fail-closed mutations.

Until those artifacts pass the G01B validator, this ADR proves only that the contract
decision and legacy mapping are complete. It does not prove reference-runtime
correctness, extension cost, runtime overhead, model migration, or performance.

## Consequences

The design adds explicit operation descriptors, catalog assembly, validation, and plan
serialization. This is more contract code than splitting the legacy trait, but it moves
failure detection to planning and makes operation and family extensions additive. The
remaining risk is dispatch and metadata overhead; it is bounded by the G01B measurement
plan rather than assumed away in G01A.
