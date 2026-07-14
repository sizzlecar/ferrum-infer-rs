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
| Execution event and cursor | `execution_event` | Owns event shape, context validation and request lifecycle state |
| Resource pool event and cursor | `resource_pool` | Owns receipt validation and pool lifecycle state |
| Replay identity | `replay` | Consumes completed execution/resource evidence without owning it |
| Sink and emitter | `sink` | Consumes validated events and transactional cursor updates |

Existing public paths are preserved by the 48-line facade. Cross-owner reads now use existing typed
getters where available. Replay-only context constructors, sequence liveness checks and pool proof
queries are parent-private methods; fields were not broadly widened.

## Final Graph

The complete importer-to-owner edge set is:

```text
execution_event: foundation, identity, sequence_binding, topology
foundation:
identity: foundation
replay: execution_event, foundation, identity, resource_pool, sequence_binding, topology
resource_pool: foundation, sequence_binding, topology
sequence_binding: foundation
sink: execution_event, foundation
topology: foundation
```

The SCC result is:

```text
event_dependency_multi_module_scc_count=0
```

One valid dependencies-first topological order is:

```text
foundation -> identity -> topology -> sequence_binding -> execution_event -> resource_pool
-> replay -> sink
```

The linear spelling is only a proof of acyclicity; `execution_event` and `resource_pool`, and
`replay` and `sink`, remain independent branches where the edge set permits it.

## Bounded Validation

```text
CARGO_BUILD_JOBS=4 cargo check -p ferrum-interfaces --all-targets
  PASS
RUST_TEST_THREADS=1 CARGO_BUILD_JOBS=4 cargo test -p ferrum-interfaces \
  --test vnext_event_contract_tests -- --test-threads=1
  1 passed; 0 failed
RUST_TEST_THREADS=1 CARGO_BUILD_JOBS=4 cargo test -p ferrum-interfaces \
  --test vnext_compile -- --test-threads=1
  80 UI fixtures passed; 0 mismatches
```

The event aggregate originally failed only because two source-shape assertions still read the old
monolithic path. They now inspect `event/sink.rs` and `event/resource_pool.rs` while preserving the
same invariants. `#[track_caller]` on the aggregate assertion helper now reports the actual failing
invariant line rather than the shared helper line.

This audit does not claim S0A completion. The 6,208-line event aggregate test still requires
owner-aligned splitting, and the public owner map, bounded aggregate, and final S0A artifact
validator remain open.
