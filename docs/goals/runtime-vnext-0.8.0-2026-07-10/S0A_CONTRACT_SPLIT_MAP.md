# S0A Contract Split Map

## Status

S0A now covers four production contract groups: `resource`, `execution`, `event`, and
`operation`, plus the root `static_initialization.rs` composition owner. The four domain groups
contain 78 production owners and 300 importer-to-owner edges. The syntax-tree graph reports zero
multi-owner strongly connected components and zero diagnostics; the composition owner is bounded
and identity-checked separately because it intentionally joins otherwise independent domains.

The public owner audit accounts for every baseline API unit:

```text
mapped=1852/1866
migrated=14
lost=0
ambiguous=0
inaccessible=0
added=1157
added_sha256=557dec231957692d7e9e727ad4da20d33441b00c788eaee76fdf2de398a0ef5a
unsupported=0
```

The 14 migrations are not omissions. Each entry in `S0A_PUBLIC_API_MIGRATIONS.json` names its old
path and kind, public replacement targets, rationale, and introducing commit. Any unexplained
loss, redundant migration, inaccessible target, unsupported syntax, or added-item digest drift is
a gate failure.

The added-item ledger includes 46 request-state hazard contract items introduced by `5626fd2b`
and 43 determinism, replay, and packed-coordinate contract items introduced afterward. This
refresh changes no public paths or runtime semantics; it binds the gate to the public owner
surface already present at `fd694d4a`.

This document records the implemented ownership design. It is not a G01A PASS artifact. Canonical
completion requires a clean-source `vnext-g01a` run and both exact PASS lines documented below.

## Baseline

The fixed baseline is commit `b5377b12464b60203a3fe57a6de4c9952ed2474b`.

| Monolith | Physical lines | Logical lines | SHA256 |
|---|---:|---:|---|
| `resource.rs` | 13,220 | 11,991 | `26b3e035010111b0d1da2f1133b665c207c5802e689ab02b5f3bc35c9933a97d` |
| `execution.rs` | 6,651 | 6,145 | `276711236b000f35633df1662751a6acc1182af8e2b98bfe43aa546d18a37f18` |
| `event.rs` | 4,893 | 4,584 | `aac28b3bdadf16f15ebcab71ec72d3bab62c3cc28f9b18893c8b8b053c50edcb` |
| `operation.rs` | 4,734 | 4,350 | `76e73a93ac091bcab3cf1d4e47145feb011d50b6ade7403c2161b727bd29ae9d` |

The required pre-move inventory is
`docs/release/cleanup/20260802-operation-s0a-inventory.md`, with SHA256
`7ec20f1dc708553e1206bb3ad962b66e1344f2be39910cff971e2420b6357785`.

## Size Policy

- Production facades: at most 500 physical lines.
- Production owners, including root composition owners: at most 2,500 physical lines.
- Contract test targets and reusable support owners: at most 2,000 physical lines each.
- `include!` source assembly: zero.
- Wildcard parent imports in production owners: zero, except explicitly excluded test-only modules.

Current maxima before the clean aggregate gate are:

| Category | Largest file | Physical lines | Limit |
|---|---|---:|---:|
| Facade | `operation.rs` | 102 | 500 |
| Production owner | `operation/dispatch.rs` | 2,409 | 2,500 |
| Test/support owner | `tests/vnext_device_operation_contract/mod.rs` | 1,971 | 2,000 |

The final gate recomputes these values from the committed tree. These are ownership and
reviewability limits, not a target to delete useful dynamic-resource or determinism logic.

The only root composition owner is
`crates/ferrum-interfaces/src/vnext/static_initialization.rs` (1,141 physical lines). The gate
requires this exact path and binds its line count, byte size, SHA256, and Git blob to the clean
checkout so it cannot escape the production-owner limit.

## Production Owners

### Resource

Resource has 25 owners and 106 edges. It owns capacity publication, physical backing, pool growth
and reclaim, request/sequence/session/step/invocation lifetimes, transactions, fences, recovery,
and runtime close. Model-aware static initialization was lifted to the vNext composition root;
`StateInitialization` remains a low-level resource contract.

Valid dependencies-first order:

```text
contracts -> backing_extent -> capacity -> dynamic_pool -> lane_stable_identity
-> lane_stable_arena -> ledger -> allocation -> program_binding -> request_state_hazard
-> dynamic_pool_set -> dynamic_pool_maintenance -> provisioning -> runtime_driver
-> sequence_state -> static_lease -> plan_runtime -> recovery -> transaction -> work
-> sequence -> batch -> backing_initialization -> invocation -> execution_session
```

Detailed rationale is in `S0A_RESOURCE_DEPENDENCY_AUDIT.md`.

### Execution

Execution has 22 owners and 89 edges. It owns semantic bindings, work and workspace formulas,
provider/resource evidence, storage selection, reusable-memory policy, immutable plans,
determinism coverage, resolution, validation, and program compilation.

Valid dependencies-first order:

```text
foundation -> binding -> weight -> work -> workspace -> provider_resource -> contracts
-> checkpoint -> storage -> allocation -> reusable -> solver -> memory -> provider -> policy
-> plan -> determinism -> determinism_coverage -> planner -> compiler -> resolution -> validation
```

Detailed rationale is in `S0A_EXECUTION_DEPENDENCY_AUDIT.md`.

### Event

Event has 9 owners and 21 edges. It owns event identity, topology, sequence evidence, resource
maintenance evidence, execution/resource cursors, replay closure, and sinks.

Valid dependencies-first order:

```text
foundation -> identity -> sequence_binding -> resource_maintenance -> topology
-> execution_event -> resource_pool -> replay -> sink
```

Detailed rationale is in `S0A_EVENT_DEPENDENCY_AUDIT.md`.

### Operation

Operation has 22 owners and 84 edges. It owns semantic operation contracts, storage and tensor
geometry, provider-visible physical weight ABI, provider planning, compiled identity, invocation,
dispatch, replay/determinism evidence, and compiled submission waves.

Valid dependencies-first order:

```text
foundation -> semantic -> attribute -> storage_profile -> tensor_contract -> weight_contract
-> resolved_value -> buffer_view -> descriptor -> provider -> catalog -> compiled_identity
-> identity -> dispatch_contract -> backing_upload -> invocation -> registry -> determinism
-> determinism_artifact -> workspace_encoding -> dispatch -> compiled_submission_wave
```

Detailed rationale is in `S0A_OPERATION_DEPENDENCY_AUDIT.md`.

## Dependency Policy

The canonical graph is generated with `syn`; handwritten dependency lists are review notes only.
For every group the gate verifies the exact owner set, all edge endpoints and evidence, a
dependency-first topological order, the complete SCC partition, recomputed summaries, and artifact
SHA bindings.

`resource` and `operation` are lower-level boundaries and may not reference model-owned root
symbols. This rule forced two ownership corrections instead of allowlisting violations:

1. `StateInitialization` moved from model schema code to resource contracts.
2. Provider-visible physical weight encoding/layout/binding moved from model to
   `operation/weight_contract.rs`; model retains `WeightSchema` and schema-aware construction.

`static_initialization.rs` is a root composition owner because it legitimately joins model weight
sources with resource transactions. Keeping it under `resource` would invert that dependency.

## Test Ownership

The S0A structural matrix has 28 invariant-owner integration targets and 13 explicit reusable
support owners. The removed oversized aggregate roots may not reappear. Shared fixtures are counted
once as real owners rather than hidden through `include!` or multiplied into every consumer.

The device-operation transaction driver fixture is a peer support owner. The parent support module
re-exports it for unchanged test call sites, while both files remain independently size-bounded.

The canonical aggregate does not run a curated subset. It executes:

```text
CARGO_BUILD_JOBS=4 RUST_TEST_THREADS=1 \
  cargo test -p ferrum-interfaces --all-targets -- --test-threads=1 --nocapture
```

The command is contained by `scripts/release/bounded_command.py`, records a receipt, and must
execute every integration target discovered in `crates/ferrum-interfaces/tests/`. Machine proof
lines for plan determinism, version rejection, fail-closed resolution, resource lifecycle, event
replay, and device operation behavior must each appear exactly once.

## Preserved Invariants

- Dynamic capacity is published only from committed physical backing.
- Admission is derived from live capacity and distinguishes `Deferred` from `Impossible`.
- Register-then-recheck waiting prevents lost wakeups; temporary pressure does not globally block
  active decode or unrelated eligible work.
- Request, sequence, session, step, and invocation remain distinct owning lifetimes.
- Possibly submitted device work retains allocations and authority until a typed fence reaches a
  terminal state; recovery and quarantine remain explicit.
- Physical weight identity carries component ID, shape, encoding, layout, and schema provenance;
  providers do not infer layouts from model names.
- Provider selection is resolved before dispatch. Architecture or backend feature branching is not
  reintroduced into the operation hot path.
- Event, replay, and determinism evidence preserve exact request/plan/node/operation/resource
  attribution.

S0A preserves these semantics while changing ownership. Breaking semantic changes belong to
S0B/S1 and must be driven by the real CUDA `run` and `serve` consumer.

## Canonical Gate

After committing the split, produce fresh clean-source G00A/G00F evidence for that same SHA, then
run:

```text
python3 scripts/release/run_gate.py vnext-g01a \
  --g00f <fresh-g00f-gate-manifest> \
  --out <external-g01a-out>
```

The run must create `g01a-contract-split/` containing the inventory, ADR, migration manifest,
public owner map, owner dependency graph, contract map, bounded receipt/logs, aggregate evidence,
and child manifest. S0A is complete only when both exact lines exist:

```text
FERRUM RUNTIME VNEXT G01A CONTRACT SPLIT PASS: <out_dir>
FERRUM GATE vnext-g01a PASS: <out_dir>
```

This gate unlocks G01B/S1 work. It does not prove model migration, product wiring, performance,
G01 aggregate completion, or release readiness.
