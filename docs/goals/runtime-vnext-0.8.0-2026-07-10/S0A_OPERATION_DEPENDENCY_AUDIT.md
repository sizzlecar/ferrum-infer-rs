# S0A Operation Dependency Audit

## Scope

This audit covers the production owners under
`crates/ferrum-interfaces/src/vnext/operation/`. The `operation.rs` facade is a public-path
compatibility boundary and is not a semantic owner. Test modules are consumers and are excluded
from the production graph.

The canonical graph is generated from the Rust syntax tree. It resolves symbol-explicit imports,
rejects hidden production modules and wildcard parent imports, computes the complete strongly
connected component partition, and records source evidence for every importer-to-owner edge. The
same scan rejects references from operation owners to model-owned root symbols.

## Ownership

The former 4,734-line operation monolith is split into 22 production owners:

| Layer | Owners | Responsibility |
|---|---|---|
| Leaf contracts | `foundation`, `semantic`, `attribute`, `storage_profile`, `tensor_contract` | Stable identifiers, values, attributes, storage geometry, and tensor access contracts |
| Weight ABI | `weight_contract`, `resolved_value`, `buffer_view` | Provider-visible physical weight identity, resolved storage, and checked physical views |
| Provider planning | `descriptor`, `provider`, `catalog`, `registry` | Operation contracts, provider semantics, capability catalog, selection, and runtime binding |
| Submission identity | `compiled_identity`, `identity`, `invocation`, `dispatch_contract` | Immutable compiled identity, batch participants, invocations, and submission policy |
| Execution evidence | `determinism`, `determinism_artifact`, `backing_upload`, `workspace_encoding`, `dispatch`, `compiled_submission_wave` | Replay witnesses, logical-backing uploads, workspace encoding, device dispatch, and compiled-wave evidence |

The physical weight ABI belongs to `weight_contract`, not `model`. Model-owned `WeightSchema`
performs schema-to-binding construction and implements the operation-owned logical-validation
capability. Operation code therefore does not import model types, while model families can still
validate and construct the same public `ResolvedWeightBinding` API.

## Boundary Rules

- `operation` may depend on root identities and device/resource-neutral primitives, but not on
  model-family schemas, architecture names, scheduler policy, product entrypoints, or backend
  implementations.
- Provider-visible bindings contain explicit component IDs, physical shapes, encodings, layout,
  and schema identity. They do not infer an ABI from a model name or quantization nickname.
- Planning resolves providers and capabilities before dispatch. The dispatch path consumes bound
  identities and must not repeat architecture selection.
- Recursive physical layouts are bounded in depth and node count. Binding wire order is canonical,
  every referenced component is present exactly once, and model-side logical validation remains
  mandatory before a binding enters an execution plan.
- Public paths continue through the facade. Cross-owner helpers use crate- or parent-private
  visibility and are not promoted to public API to make the split compile.

## Current Graph

The current graph contains 22 owners and 84 directed dependency edges. Every strongly connected
component is a singleton; multi-owner SCC count and diagnostic count are both zero. A valid
dependencies-first order is:

```text
foundation -> semantic -> attribute -> storage_profile -> tensor_contract -> weight_contract
-> resolved_value -> buffer_view -> descriptor -> provider -> catalog -> compiled_identity
-> identity -> dispatch_contract -> backing_upload -> invocation -> registry -> determinism
-> determinism_artifact -> workspace_encoding -> dispatch -> compiled_submission_wave
```

`operation.rs` is 102 physical lines. The largest production owner is `dispatch.rs` at 2,409
physical lines, below the S0A 2,500-line limit. No production owner uses `include!` or a wildcard
parent import.

`backing_upload` owns the complete checked translation from one logical backing range to one or
more physical upload commands. `dispatch` consumes that private helper; the helper depends only on
device/resource contracts plus `dispatch_contract`, `foundation`, and `storage_profile`, so it does
not depend back on the dispatch orchestrator.

## Validation

The focused pre-checkpoint evidence for the final ownership change is:

```text
VNEXT OWNER DEPENDENCY GRAPH PASS: groups=4 owners=78 edges=300 scc=0 diagnostics=0
backing-upload determinism focused tests: 4 passed; 0 failed
owner dependency graph unit tests: 6 passed; 0 failed
weight/model/provider/execution/static-initialization focused tests: 46 passed; 0 failed
resolved-weight boundary negative test: 1 passed; 0 failed
```

These focused results do not claim G01A completion. The clean-source `vnext-g01a` gate must bind
the generated graph, public owner map, inventory, this review document, and the bounded
`ferrum-interfaces --all-targets` result to one Git SHA before emitting the canonical PASS line.
