# S0A Resource Dependency Audit

## Scope

This audit covers the production modules under
`crates/ferrum-interfaces/src/vnext/resource/`. Test modules are excluded because they are
consumers of the production facade, not production dependency owners.

The audit discovers each top-level type, trait, function, constant, and static owner, resolves the
symbol-explicit `use super::{...}` imports, builds importer-to-owner edges, and runs a strongly
connected component check over the complete graph. Checking only pairs of modules is insufficient:
the audit also rejects cycles spanning three or more modules.

## Findings Before Normalization

The first explicit-import graph exposed ten bidirectional owner pairs that had been hidden inside
the original monolith:

1. `allocation <-> ledger`
2. `batch <-> invocation`
3. `capacity <-> dynamic_pool`
4. `dynamic_pool <-> plan_runtime`
5. `invocation <-> recovery`
6. `invocation <-> sequence`
7. `ledger <-> transaction`
8. `plan_runtime <-> sequence`
9. `plan_runtime <-> transaction`
10. `recovery <-> sequence`

Extracting elastic plan provisioning from `capacity` temporarily exposed an additional
`dynamic_pool <-> provisioning` pair. After all pairwise edges were removed, a full SCC check found
one remaining three-module cycle:

```text
plan_runtime -> static_lease -> recovery -> plan_runtime
```

This cycle was not visible in a pairwise-only audit.

## Ownership Corrections

| Contract or behavior | Final owner | Reason |
|---|---|---|
| Transaction identity and driver failure envelope | `contracts` | Lower-level identity/error contract shared by allocation and ledger |
| Context/lease receipt validation | `ledger` | Validation is defined by ledger records, not transaction orchestration |
| Elastic plan provisioning | `provisioning` | Separates admission/provision construction from capacity arithmetic |
| Backing chunk identity, segment projection, and free-extent index | `backing_extent` | This is the leaf owner for physical range translation and allocation journal rollback |
| Logical backing evidence, authority, and views | `dynamic_pool` | These contracts bind allocated extents to request and invocation lifetimes |
| Dynamic pool domain specification | `dynamic_pool` | Pool identity and membership are pool-owned facts |
| Cross-pool growth, claim, reclaim and lane-stable arena orchestration | `dynamic_pool_set` | Depends on backing/pool contracts without making the lower-level pool owner depend on orchestration |
| Program binding layout | `program_binding` | Immutable reusable-execution binding layout is compiled separately from pool lifecycle |
| Dynamic pool maintenance API | `dynamic_pool_maintenance` | Pressure handling and retry policy consume the pool-set boundary without owning allocation state |
| Core resource failure constructor and dispatch poison bit | `contracts` | Shared wire/state encoding with no higher-level owner dependency |
| Step lease storage | `batch` | A step owns one exact continuous-batch frame and its participants |
| Sequence slot/dispatch state machine | `sequence` | Sequence lifecycle owns the encoded slot and dispatch gate |
| Sequence abort evidence and recovery registry | `recovery` | Recovery owns terminal abort evidence and abandoned stream state |
| Request admission implementation | `sequence` | It creates the request root and is a consumer of the plan runtime binding |
| Bound execution stream | `invocation` | Invocation dispatch owns stream activation and transfer to recovery |
| Step admission and bound stream activation/synchronization | `execution_session` | Execution-session typestate consumes batch/invocation owners and remains the terminal dependency |
| Static owned slots and borrowed buffer views | `static_lease` | These types are used exclusively by the static provisioning lease |

No public resource path changed: the parent facade continues to re-export the public contracts.
Sibling-only implementation access uses `pub(super)` and does not widen the crate's public API.

## Final Result

The final production graph contains twenty-one modules and zero strongly connected components with
more than one member:

```text
resource_dependency_scc_count=0
```

One valid dependencies-first topological order is:

```text
contracts -> backing_extent -> capacity -> ledger -> work -> allocation -> dynamic_pool
-> program_binding -> dynamic_pool_set -> dynamic_pool_maintenance -> runtime_driver
-> static_lease -> provisioning -> plan_runtime -> recovery -> transaction -> sequence
-> static_initialization -> batch -> invocation -> execution_session
```

This order is evidence that the graph is acyclic, not a requirement that unrelated modules share
one linear architectural layer.

The 2026-07-31 split deliberately uses peer `mod` owners rather than nested files. Logical backing
evidence remains in `dynamic_pool`; `dynamic_pool_set` imports it and never exports a dependency
back into the lower-level owner. Likewise, `execution_session` consumes `invocation`; invocation
does not consume session typestate. This keeps both new edges one-way and makes every source owner
visible to the canonical inventory.

## Bounded Validation

The following validations passed after the ownership corrections:

```text
CARGO_BUILD_JOBS=4 cargo check -p ferrum-interfaces --all-targets
RUST_TEST_THREADS=1 CARGO_BUILD_JOBS=4 cargo test -p ferrum-interfaces --lib \
  vnext::resource:: -- --test-threads=1
  92 passed; 0 failed
RUST_TEST_THREADS=1 CARGO_BUILD_JOBS=4 cargo test -p ferrum-interfaces \
  --test vnext_resource_capacity_contract_tests \
  --test vnext_resource_transaction_lifecycle_tests \
  --test vnext_resource_transaction_evidence_tests \
  --test vnext_resource_sequence_activation_tests \
  --test vnext_resource_sequence_recovery_tests \
  --test vnext_resource_recovery_authority_tests \
  --test vnext_resource_runtime_close_tests -- --test-threads=1
  12 parent tests passed; 0 failed; 311 frozen proof cases preserved
```

The external test target includes an isolated panic-child fault injection. Its child panic output
is expected; the parent test and target both exited successfully. This audit does not claim S0A
completion: execution normalization is recorded separately in
`S0A_EXECUTION_DEPENDENCY_AUDIT.md`; `event.rs`, remaining oversized test targets, the public owner
map, and the final artifact validator remain open.
