# S0A Contract Split Map

## Status

- Work package: S0A, semantics-preserving structural split.
- Current stage: the `resource`, `execution`, and `event` production monoliths are split into real
  owner modules with explicit imports and zero multi-module dependency SCCs. The four oversized
  resource/event/core/device contract targets are replaced by 24 invariant-owner targets and 10
  explicit reusable support owners. The structured public owner audit maps all `1,490/1,490`
  externally reachable API units with zero loss, ambiguity, inaccessible item, or added item.
- This map records the implemented ownership structure. Canonical G01A/S0A completion still
  requires a clean-source `vnext-g00f` binding and bounded aggregate `vnext-g01a` artifact; neither
  this document nor historical focused test output substitutes for those PASS lines.

## Source Evidence

- Pre-split source: `crates/ferrum-interfaces/src/vnext/resource.rs`
- Pre-split logical lines: `13,220`
- Pre-split SHA256:
  `26b3e035010111b0d1da2f1133b665c207c5802e689ab02b5f3bc35c9933a97d`
- Pre-split execution source: `crates/ferrum-interfaces/src/vnext/execution.rs`
- Pre-split execution logical lines: `6,651`
- Pre-split execution SHA256:
  `276711236b000f35633df1662751a6acc1182af8e2b98bfe43aa546d18a37f18`
- Pre-split event source: `crates/ferrum-interfaces/src/vnext/event.rs`
- Pre-split event logical lines: `4,893`
- Pre-split event SHA256:
  `aac28b3bdadf16f15ebcab71ec72d3bab62c3cc28f9b18893c8b8b053c50edcb`
- Repository inventory: `docs/release/cleanup/20260714-inventory.md`
- Inventory SHA256:
  `6cce246fc3f62ec058498bdb0a825613d47abaf327fcf8a698c59ece41c79190`
- Inventory validator result:
  `INVENTORY PASS: /Users/chejinxuan/rust_ws/ferrum-infer-rs/docs/release/cleanup/20260714-inventory.md`

Before module visibility changed, the 13 initial physical fragments concatenated byte-for-byte with the
original seven-line test-module tail to the pre-split SHA256. The validated implementation now
uses real child `mod` declarations and facade `pub use` exports; it does not use `include!`.
Existing public paths remain unchanged. Cross-owner implementation details that were implicitly
shared by the giant module are now explicitly limited to the `vnext::resource` parent with
`pub(super)`, so they do not become crate-public API. All 13 production fragments now use explicit
symbol imports; only the two existing in-module test files retain `use super::*`. Owner
normalization added a fourteenth production module, `provisioning`, and moved shared contracts to
their lowest valid owner. The complete symbol graph now has zero multi-module strongly connected
components. The transition is recorded in `S0A_RESOURCE_DEPENDENCY_AUDIT.md`.

As a second mechanical-equivalence check, removing only the added `use super::*` lines and
`pub(super)` visibility qualifiers, concatenating the production fragments, and applying the same
`rustfmt` produced byte-identical old/new files with SHA256
`3fac9f1b587513d77fc796538ae40444cfe9be08992bdf0d8b36f1f88168560b`.

The first execution checkpoint applies the same stronger proof. Concatenating the eight production
fragments in source order, removing only their explicit import preludes and `pub(super)` qualifiers,
restoring the original facade imports, and applying the same formatter produces identical old/new
production SHA256 values:
`70899c1ef6365b65e1df0a34e7b052a1b605d7228b343d17859cac137eb8cac1`.
The extracted white-box test module independently produces identical old/new SHA256 values:
`452bdff62ee00cd91473fb61cfcc5758f98c2d407fce24b3317d56ad30e2712e`.
The subsequent ownership normalization preserves the parent facade paths while moving misplaced
helpers and implementations to their lowest valid owner. Its complete dependency transition is
recorded in `S0A_EXECUTION_DEPENDENCY_AUDIT.md`.

## Resource Ownership

| Current owner | Lines | Primary responsibility |
|---|---:|---|
| `resource/contracts.rs` | 564 | Base identifiers, descriptors, shared error/state encodings and reservation contracts |
| `resource/capacity.rs` | 360 | Device capacity authority, accounting, epochs and process-wide claims |
| `resource/provisioning.rs` | 355 | Static/elastic plan provisioning and admission construction |
| `resource/allocation.rs` | 555 | Allocation ownership and resource driver contracts |
| `resource/ledger.rs` | 1,668 | Lease state, transition receipts, allocation ledger and receipt validation |
| `resource/recovery.rs` | 348 | Abandoned-sequence recovery registry and terminal abort evidence |
| `resource/dynamic_pool.rs` | 2,132 | Dynamic backing pools, growth, extent/view ownership and quarantine |
| `resource/static_lease.rs` | 371 | Plan-static lease, owned slots, borrowed buffer views and typed admission requests |
| `resource/work.rs` | 363 | Step/invocation work-shape admission requests and checked demand derivation |
| `resource/plan_runtime.rs` | 952 | Plan runtime root, close state, capacity waits and pool coordination |
| `resource/sequence.rs` | 1,105 | Request, sequence and session resource lifetime authorities |
| `resource/batch.rs` | 1,045 | Batch participants, step ownership, physical invocation ledger and retirement |
| `resource/invocation.rs` | 1,627 | Invocation leases, bound streams, retry authority and active-sequence permits |
| `resource/transaction.rs` | 1,920 | Sealed transaction typestate, commit/rollback/release and compensation |

`resource.rs` is now a 69-line facade. Every production fragment is below the S0A `2,500`
logical-line limit and the facade is below `500` lines.

## Execution Ownership Checkpoint

| Current owner | Lines | Primary responsibility |
|---|---:|---|
| `execution/foundation.rs` | 84 | Shared validation, canonical fingerprints, alignment and storage arithmetic |
| `execution/binding.rs` | 248 | Semantic value/weight binding validation and estimator-input identity |
| `execution/work.rs` | 615 | Token/page work evidence and bounded dynamic demand formulas |
| `execution/workspace.rs` | 144 | Provider workspace formula, scope and storage requirement contracts |
| `execution/provider_resource.rs` | 174 | Bound provider resource estimate evidence and validation |
| `execution/contracts.rs` | 292 | Plan identity, provider selection evidence and immutable node contracts |
| `execution/storage.rs` | 788 | Storage compatibility, dynamic pool specifications and descriptors |
| `execution/allocation.rs` | 121 | Static/dynamic resource allocation contract and validation |
| `execution/solver.rs` | 416 | Joint provider/storage solver and checked selection helpers |
| `execution/memory.rs` | 775 | Core-derived memory plan and pool/liveness accounting |
| `execution/provider.rs` | 242 | Provider selection and serialized execution-plan payload contracts |
| `execution/policy.rs` | 74 | Typed runtime policy and validated plan-build request boundary |
| `execution/plan.rs` | 1,898 | Semantic plan construction and deterministic provider/storage selection |
| `execution/resolution.rs` | 289 | Provider registry resolution into typed per-node evidence |
| `execution/validation.rs` | 56 | Untrusted execution-plan revalidation boundary |
| `execution/planner.rs` | 12 | Pure execution planner trait boundary |

`execution.rs` is now a 67-line facade. Every production fragment is below `2,500` lines. The
existing 14 white-box tests are isolated in a 506-line `execution/tests.rs` module and pass with
`--test-threads=2`.

The first eight-module split made former same-module coupling observable as one SCC containing all
eight production owners. Low-level canonical/allocation helpers, provider resource evidence,
serialized payload validation, resolution, and the planner API have now moved to distinct owners.
The complete sixteen-module production graph has zero multi-module SCCs. Public execution paths
remain re-exported by the facade and no production fragment uses `use super::*`.

## Event Ownership

| Current owner | Lines | Primary responsibility |
|---|---:|---|
| `event/foundation.rs` | 127 | Event IDs, phases, timestamps, canonical fingerprints and shared validation |
| `event/identity.rs` | 243 | Validated and unvalidated execution identity envelopes |
| `event/topology.rs` | 90 | Trusted immutable execution topology derived from a plan |
| `event/sequence_binding.rs` | 490 | Active, completed and aborted sequence evidence |
| `event/execution_event.rs` | 1,596 | Execution event wire boundary, context validation and transactional cursor |
| `event/resource_pool.rs` | 1,125 | Resource-pool events, receipt validation and pool cursor |
| `event/replay.rs` | 1,128 | Replay evidence closure, cleanup requirements and replay identity |
| `event/sink.rs` | 156 | Event sink capability, transactional emitter and disabled sink |

`event.rs` is now a 48-line facade. Public paths remain unchanged, every production import is
symbol-explicit, and no production fragment uses `use super::*`. The complete eight-module graph
has zero multi-module SCCs. Replay and sink use typed getters or narrowly scoped parent-private
proof methods instead of reading another owner's private fields.

## Resource Test Ownership

The former 4,289-line `vnext_resource_contract_tests` target is replaced by one shared 1,474-line
fixture and seven owner targets. The limit below counts the shared fixture against every target;
it is not hidden from the logical target size.

| Target | Owner lines | With shared fixture | Frozen proof cases |
|---|---:|---:|---:|
| `vnext_resource_capacity_contract_tests` | 402 | 1,876 | 33 |
| `vnext_resource_transaction_lifecycle_tests` | 486 | 1,960 | 70 |
| `vnext_resource_transaction_evidence_tests` | 503 | 1,977 | 69 |
| `vnext_resource_sequence_activation_tests` | 376 | 1,850 | 53 |
| `vnext_resource_sequence_recovery_tests` | 389 | 1,863 | 48 |
| `vnext_resource_recovery_authority_tests` | 404 | 1,878 | 38 |
| `vnext_resource_runtime_close_tests` | 424 | 1,898 | standalone close/recovery assertions |

The frozen aggregated resource proof remains `311` cases:
`13 + 20 + 70 + 69 + 53 + 48 + 38 = 311`. The panic-isolation child stays only in the
transaction-evidence target. G01A checkpoint consumers now validate the exact seven-target test
matrix and sum the owner proof lines instead of accepting one monolithic `311/311` line.

## Event Test Ownership

The former 6,210-line event/replay target is replaced by five invariant-owner targets and six
normal Rust fixture modules. No `include!` source assembly is used.

| Target | Owner responsibility | Frozen proof cases |
|---|---|---:|
| `vnext_event_execution_contract_tests` | execution identity, wire validation and cursor state | 54 |
| `vnext_event_sink_contract_tests` | transactional emission, live witness and sink failure | 13 |
| `vnext_event_resource_pool_contract_tests` | pool event identity, transition and lease cursors | 27 |
| `vnext_event_recovery_contract_tests` | failure/recovery continuation and root close evidence | 20 |
| `vnext_event_replay_contract_tests` | replay closure, terminal evidence and no-static cleanup | 47 |

The frozen counted total remains `54 + 13 + 27 + 20 + 47 = 161`. The no-static replay helper also
retains its direct assertions. Every target root and every reusable fixture owner is below 2,000
lines. S0A LOC accounting counts each source owner once: a target and each reusable fixture module
are independently bounded, while a shared fixture is not duplicated into every consumer's LOC.
Counting the complete crate dependency graph as target LOC would charge all production contracts
to every test and is not the ownership/reviewability metric defined by this gate.

## Core Contract Test Ownership

The former 5,445-line `vnext_contract_tests` target is replaced by a 1,648-line shared contract
fixture and seven invariant-owner targets:

| Target | Owner lines | Test count | Primary invariant owner |
|---|---:|---:|---|
| `vnext_planning_resource_contract_tests` | 573 | 8 | resource demand, memory and capacity planning |
| `vnext_plan_wire_contract_tests` | 257 | 9 | deterministic plan build and validated wire reconstruction |
| `vnext_provider_selection_contract_tests` | 310 | 5 | provider identity, fallback and registry rejection |
| `vnext_weight_layout_contract_tests` | 659 | 8 | physical weight layout, padding and model program shape |
| `vnext_resolution_contract_tests` | 1,429 | 11 | typed model resolution, provenance and fail-closed input |
| `vnext_execution_graph_contract_tests` | 530 | 7 | alias and state-effect dependency graph |
| `vnext_source_audit_contract_tests` | 59 | 3 | architecture-neutral source and wire-size audit |

The exact test-name union is still `51/51`, with no duplicate owner. The plan proof lines remain
`100/100` for determinism, round trip and breaking-version rejection; resolution retains
`VNEXT FAIL CLOSED PASS: 62/62` and `VNEXT MODEL IDENTITY PASS: 5/5`. Both release validators now
require the seven-target matrix. The target roots and shared core fixture are separate reusable
owners and are each independently below the 2,000-line hard limit.

## Device Operation Test Ownership

The former 3,799-line device/operation target is replaced by a 1,672-line shared fixture and five
owner targets:

| Target | Owner lines | Tests | Frozen proof cases |
|---|---:|---:|---:|
| `vnext_device_operation_dispatch_contract_tests` | 652 | 1 | 70 |
| `vnext_device_operation_cancel_contract_tests` | 168 | 1 | 16 |
| `vnext_device_operation_legacy_authority_contract_tests` | 115 | 1 | 13 |
| `vnext_device_operation_completion_contract_tests` | 1,063 | 2 | 200 plus bounded drop |
| `vnext_device_operation_batch_contract_tests` | 169 | 1 | standalone 32-participant batch |

The exact counted proof remains `70 + 16 + 13 + 200 = 299`. The old aggregate test is gone; the
checkpoint and outer gate require the six exact test names and sum four machine proof lines. Every
target and the shared device fixture are explicit owners below 2,000 lines; shared code is neither
hidden nor multiplied by its consumer count.

## Preserved Dynamic Resource Invariants

This split is not permission to simplify the resource model. The following owners and behavior
must remain represented after S0B:

- `capacity.rs` and `dynamic_pool.rs`: effective capacity is published only from installed,
  committed backing; growth and release advance monotonic epochs.
- `work.rs`, `plan_runtime.rs`, and `transaction.rs`: no provider encode, prefill, or device submit
  is reachable before a committed lease; temporary pressure remains typed defer and permanent
  over-capacity remains typed impossible/reject.
- `sequence.rs`, `batch.rs`, and `invocation.rs`: request/sequence/session/step/invocation
  authorities retain exact lifetimes and non-empty participant identity.
- `ledger.rs`, `recovery.rs`, and `invocation.rs`: possibly-submitted work retains ownership until
  a typed fence terminal; retry, compensation, release, recovery, and quarantine remain explicit.
- Capacity waiting must preserve register/recheck lost-wakeup protection and must not introduce
  global head-of-line blocking.

## Dependency Direction

The final resource graph is acyclic. One valid dependencies-first topological order is:

```text
contracts -> ledger -> capacity -> allocation -> dynamic_pool -> provisioning -> static_lease
-> plan_runtime -> transaction -> work -> recovery -> sequence -> batch -> invocation
```

Rust module privacy now separates the owners, all newly shared internals are restricted to the
resource parent, and production imports are symbol-explicit. The SCC audit reports `0` cycles;
pairwise and the previously hidden three-module cycle are both eliminated. Execution production
imports are also symbol-explicit and its complete SCC audit reports `0` cycles. A valid
dependencies-first execution order is:

```text
foundation -> binding -> work -> workspace -> provider_resource -> contracts -> storage
-> allocation -> solver -> memory -> provider -> policy -> plan -> resolution -> validation
-> planner
```

The event graph is also acyclic. One valid dependencies-first order is:

```text
foundation -> identity -> topology -> sequence_binding -> execution_event -> resource_pool
-> replay -> sink
```

S0B may later shrink or break these contracts only against the real Qwen3.5-4B production consumer.

## Validation For This Stage

1. The resource and execution initial reconstructed source SHA256 values equal their pre-split
   source SHA256 values.
2. Their normalized old/new production source SHA256 values are identical after stripping only
   the module visibility/prelude additions and applying the same formatter.
3. `cargo fmt --all -- --check` accepts the fragments and facade.
4. `CARGO_BUILD_JOBS=4 cargo check -p ferrum-interfaces --all-targets` passes.
5. `RUST_TEST_THREADS=2 cargo test -p ferrum-interfaces --lib resource:: -- --test-threads=2`
   passes `47/47` tests.
6. One bounded cargo invocation with the seven `vnext_resource_*` owner targets and
   `-- --test-threads=2` passes `12/12` parent tests, including the expected isolated panic-child
   fault case and all `311` frozen proof cases.
7. No paid GPU, model download, performance run, or product migration claim is part of this stage.

For the normalized execution graph, `cargo check -p ferrum-interfaces --all-targets` passes, the
bounded execution white-box target passes `14/14`, the focused external execution contracts pass
`51/51 + 12/12`, and all `80` vNext compile-time UI fixtures pass in normal trybuild mode. These
results establish focused compile, contract, and structural evidence. The canonical S0A gate also
runs the complete crate through one bounded aggregate and binds its public owner map and split
inventory to the same clean Git SHA.

For the normalized event graph, `cargo check -p ferrum-interfaces --all-targets` passes and all `80`
vNext compile-time UI fixtures pass without snapshot changes. The former aggregate is now five
owner targets whose exact proof lines pass `54/54`, `13/13`, `27/27`, `20/20`, and `47/47`.
`runtime_vnext_g01a_checkpoint.py` preserves the historical exact five-target matrix and sums it
back to `161`; the current S0A gate consumes the same proof lines from the bounded aggregate.

After owner normalization and the zero-SCC audit, the bounded all-target check, `47/47` resource
library tests, and all seven external resource owner targets pass. The transaction-evidence
target's isolated panic-child output is expected fault injection; the parent test exits
successfully. `runtime_vnext_g01a_checkpoint.py --self-test` and `run_gate.py --self-test` also
pass with the split matrix.

The canonical clean-source execution is:

```text
python3 scripts/release/run_gate.py vnext-g00f --g00a <fresh-g00a-gate-manifest> \
  --out <external-g00f-out>
python3 scripts/release/run_gate.py vnext-g01a --g00f <fresh-g00f-gate-manifest> \
  --out <external-g01a-out>
```

The second command must create `g01a-contract-split/` with the six required core artifacts and
print both the child contract-split PASS line and `FERRUM GATE vnext-g01a PASS` before S0A is
complete.
