# S0A Event Dependency Audit

## Scope

This audit covers the production modules under `crates/ferrum-interfaces/src/vnext/event/`.
The parent facade and external contract tests are excluded from the production dependency graph.

The audit resolves each symbol-explicit `use super::{...}` import to its defining child module,
builds importer-to-owner edges, computes transitive reachability, and rejects every strongly
connected component containing more than one module.

## Monolith Findings

The 4,893-line source mixed eight responsibilities in one privacy and dependency scope:

1. execution identity and common event primitives;
2. trusted plan topology;
3. active/completed/aborted sequence evidence;
4. execution event wire validation and cursor state;
5. resource-pool event validation and cursor state;
6. replay closure and cleanup evidence;
7. sink capability and transactional emission;
8. shared fingerprint and validation helpers.

The monolith allowed replay and sink to read private event, context and pool fields directly. It
also made two tests assert source shape through the old physical `event.rs` path rather than the
contract owner.

## Ownership Corrections

| Contract or behavior | Final owner | Boundary correction |
|---|---|---|
| IDs, phases, timestamp and fingerprint validation | `foundation` | Leaf dependencies shared by every event family |
| Execution identity envelope | `identity` | Independent validated/unvalidated identity boundary |
| Plan-derived topology | `topology` | Immutable execution graph evidence, independent of event cursors |
| Sequence disposition evidence | `sequence_binding` | Resource/session authority remains below event validation |
| Resource maintenance evidence | `resource_maintenance` | Typed maintenance outcomes are independent from execution-event cursors |
| Execution event and cursor | `execution_event` | Owns event shape, context validation and request lifecycle state |
| Resource pool event and cursor | `resource_pool` | Owns receipt validation and pool lifecycle state |
| Replay identity | `replay` | Consumes completed execution/resource evidence without owning it |
| Sink and emitter | `sink` | Consumes validated events and transactional cursor updates |

Existing public paths are preserved by the 51-line facade. Cross-owner reads now use existing typed
getters where available. Replay-only context constructors, sequence liveness checks and pool proof
queries are parent-private methods; fields were not broadly widened.

## Final Graph

The complete importer-to-owner edge set is:

```text
execution_event: foundation, identity, sequence_binding, topology
foundation:
identity: foundation
replay: execution_event, foundation, identity, resource_pool, sequence_binding, topology
resource_maintenance: foundation, sequence_binding
resource_pool: foundation, sequence_binding, topology
sequence_binding: foundation
sink: execution_event, foundation, resource_maintenance
topology: foundation
```

The SCC result is:

```text
event_dependency_multi_module_scc_count=0
```

One valid dependencies-first topological order is:

```text
foundation -> identity -> sequence_binding -> resource_maintenance -> topology
-> execution_event -> resource_pool -> replay -> sink
```

The linear spelling is only a proof of acyclicity; `execution_event` and `resource_pool`, and
`replay` and `sink`, remain independent branches where the edge set permits it.

## Bounded Validation Matrix

The current event contract targets are `vnext_event_execution_contract_tests`,
`vnext_event_sink_contract_tests`, `vnext_event_resource_pool_contract_tests`,
`vnext_event_recovery_contract_tests`, and `vnext_event_replay_contract_tests`. The canonical S0A
aggregate discovers and runs them with `RUST_TEST_THREADS=1` and `--test-threads=1`; its artifact,
not a copied count in this review, is the source of truth for the exact results.

The event aggregate originally failed only because two source-shape assertions still read the old
monolithic path. They now inspect `event/sink.rs` and `event/resource_pool.rs` while preserving the
same invariants. `#[track_caller]` on the aggregate assertion helper now reports the actual failing
invariant line rather than the shared helper line.

The former aggregate is split into five invariant-owner targets and explicit reusable fixtures.
This review does not claim S0A completion: the clean-source bounded aggregate and final
`vnext-g01a` validator remain authoritative.
