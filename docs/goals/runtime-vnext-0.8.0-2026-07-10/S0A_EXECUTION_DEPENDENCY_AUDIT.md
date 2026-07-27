# S0A Execution Dependency Audit

## Scope

This audit covers the production modules under
`crates/ferrum-interfaces/src/vnext/execution/`. The parent facade and white-box test module are
excluded: the facade preserves public paths and tests are consumers rather than production owners.

The audit resolves every symbol-explicit production import to its defining module, builds
importer-to-owner edges, and checks strongly connected components across the complete graph. It
does not accept a pairwise-only cycle check.

## Findings Before Normalization

The first semantics-preserving physical split exposed one strongly connected component containing
all eight initial production modules:

```text
contracts, work, storage, memory, provider, plan, solver, policy
```

The SCC was architectural rather than an artifact of Rust declaration order. Examples included
contracts calling solver arithmetic while solver consumed contract types, storage calling solver
quantization while solver consumed storage types, policy owning planner/result types while plan
consumed policy, and provider mixing resource evidence, resolution, wire payloads, and plan
revalidation. The monolith had allowed these responsibilities to call sideways without an explicit
dependency boundary.

## Ownership Corrections

| Contract or behavior | Final owner | Reason |
|---|---|---|
| Invalid-plan construction, canonical fingerprints, alignment and quantization | `foundation` | Leaf behavior shared by most owners without importing domain contracts |
| Semantic value/weight binding and estimator-input identity | `binding` | Model semantics are independent of plan assembly and provider registry resolution |
| Provider workspace formula and requirement | `workspace` | Workspace shape is a lower-level resource contract consumed by provider evidence and memory planning |
| Bound provider resource estimate | `provider_resource` | Estimate provenance and validation precede provider selection and plan assembly |
| Static/dynamic resource allocation | `allocation` | Allocation validation depends on storage contracts but not on memory-plan construction |
| Provider selection and execution-plan payload | `provider` | Wire/data ownership remains separate from resolution and revalidation behavior |
| Per-node provider resolution | `resolution` | Registry lookup is an input-producing boundary, not provider payload data |
| Untrusted execution-plan revalidation | `validation` | Revalidation consumes a completed plan instead of being mixed into provider data |
| Pure planner trait | `planner` | The public orchestration boundary depends on plan and policy, not vice versa |
| Semantic-program compiler | `compiler` | Terminal consumer that turns model semantics and provider evidence into an immutable executable plan |

No public execution path changed. The parent facade continues to re-export public contracts;
sibling-only implementation access remains private to the execution parent.

## Final Graph

The final production graph has seventeen modules and zero strongly connected components with more
than one member:

```text
execution_dependency_scc_count=0
```

The complete importer-to-owner edge set is:

```text
allocation: contracts, foundation, storage
binding: foundation
compiler: binding, contracts, foundation, plan, policy, provider_resource, resolution, storage
contracts: foundation, provider_resource
foundation:
memory: allocation, contracts, foundation, solver, storage
plan: allocation, binding, contracts, foundation, memory, policy, provider,
      provider_resource, solver, storage, workspace
planner: plan, policy
policy: foundation, provider
provider: contracts, memory, provider_resource, workspace
provider_resource: foundation, workspace
resolution: binding, contracts, foundation, policy, provider, provider_resource
solver: contracts, foundation, provider_resource, storage, work
storage: contracts, foundation, work
validation: contracts, foundation, plan, policy, provider
work: foundation
workspace: foundation, work
```

One valid dependencies-first topological order is:

```text
foundation -> binding -> work -> workspace -> provider_resource -> contracts -> storage
-> allocation -> solver -> memory -> provider -> policy -> plan -> resolution -> validation
-> planner -> compiler
```

This order proves acyclicity; it does not force unrelated branches into one runtime layer.

## Bounded Validation

The following validations passed after normalization:

```text
CARGO_BUILD_JOBS=4 cargo check -p ferrum-interfaces --all-targets
RUST_TEST_THREADS=2 CARGO_BUILD_JOBS=4 cargo test -p ferrum-interfaces \
  --lib vnext::execution::tests:: -- --test-threads=2
  14 passed; 0 failed
RUST_TEST_THREADS=2 CARGO_BUILD_JOBS=4 cargo test -p ferrum-interfaces \
  --test vnext_contract_tests \
  --test vnext_resolution_limits_contract_tests -- --test-threads=2
  51 + 12 passed; 0 failed
RUST_TEST_THREADS=1 CARGO_BUILD_JOBS=4 cargo test -p ferrum-interfaces \
  --test vnext_compile -- --test-threads=1
  80 UI fixtures passed; 0 mismatches
```

Seven trybuild snapshots were refreshed because private definition diagnostics now name their real
resource/execution child-module paths; one affected compiler help block also normalized its
indentation. A normal, non-overwrite trybuild run accepted all updated snapshots. This audit does
not claim S0A completion: `event.rs`, remaining oversized test targets, the public owner map,
bounded aggregate, and final artifact validator remain open.
