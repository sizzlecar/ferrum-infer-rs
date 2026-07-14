# S0A Contract Split Map

## Status

- Work package: S0A, semantics-preserving structural split.
- Current stage: `resource.rs` production contract split, symbol-level imports, and complete
  dependency-cycle removal validated; the external resource contract target is split by invariant
  owner. `execution.rs` now has a semantics-preserving real-module split with explicit production
  imports; execution owner normalization and cycle removal remain open. `event.rs` and the remaining
  oversized contract targets also remain open.
- This map records file ownership, not G01A completion. Public item inventory and the final
  `contract-map.json` remain required before the S0A PASS artifact can be issued.

## Source Evidence

- Pre-split source: `crates/ferrum-interfaces/src/vnext/resource.rs`
- Pre-split logical lines: `13,220`
- Pre-split SHA256:
  `26b3e035010111b0d1da2f1133b665c207c5802e689ab02b5f3bc35c9933a97d`
- Pre-split execution source: `crates/ferrum-interfaces/src/vnext/execution.rs`
- Pre-split execution logical lines: `6,651`
- Pre-split execution SHA256:
  `276711236b000f35633df1662751a6acc1182af8e2b98bfe43aa546d18a37f18`
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
| `execution/contracts.rs` | 466 | Plan identity, provider selection evidence, nodes and allocation contracts |
| `execution/work.rs` | 614 | Token/page work evidence and bounded dynamic demand formulas |
| `execution/storage.rs` | 789 | Storage compatibility, dynamic pool specifications and descriptors |
| `execution/memory.rs` | 928 | Core-derived memory plan, pool/liveness accounting and workspace requirements |
| `execution/provider.rs` | 789 | Provider resource evidence, node resolution and serialized plan payloads |
| `execution/plan.rs` | 2,088 | Semantic plan construction, validation and deterministic provider selection |
| `execution/solver.rs` | 442 | Joint provider/storage solver and checked allocation helpers |
| `execution/policy.rs` | 86 | Typed runtime policy and planner request boundary |

`execution.rs` is now a 45-line facade. Every production fragment is below `2,500` lines. The
existing 14 white-box tests are isolated in a 506-line `execution/tests.rs` module and pass with
`--test-threads=2`.

This is deliberately a structural checkpoint, not the final execution ownership graph. The split
made former same-module coupling observable: low-level canonical/allocation helpers, provider
resource evidence, serialized payload validation, and the planner API still form misplaced
cross-owner dependencies. The next execution change must move those symbols to their lowest valid
owner and produce a zero-SCC symbol graph before execution splitting is marked complete.

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
imports are also symbol-explicit, but its SCC audit remains an open checkpoint as described above.
S0B may later shrink or break these contracts only against the real Qwen3.5-4B production consumer.

## Validation For This Stage

1. Initial reconstructed source SHA256 equals the pre-split source SHA256.
2. Normalized old/new production source SHA256 values are identical after stripping only the
   module visibility/prelude additions and applying the same formatter.
3. `cargo fmt --all -- --check` accepts the fragments and facade.
4. `CARGO_BUILD_JOBS=4 cargo check -p ferrum-interfaces --all-targets` passes.
5. `RUST_TEST_THREADS=2 cargo test -p ferrum-interfaces --lib resource:: -- --test-threads=2`
   passes `47/47` tests.
6. One bounded cargo invocation with the seven `vnext_resource_*` owner targets and
   `-- --test-threads=2` passes `12/12` parent tests, including the expected isolated panic-child
   fault case and all `311` frozen proof cases.
7. No paid GPU, model download, performance run, or product migration claim is part of this stage.

For the execution structural checkpoint, `cargo check -p ferrum-interfaces --all-targets` passes,
and the bounded execution white-box target passes `14/14`. These results establish compile and
mechanical-equivalence evidence only; the execution zero-SCC and final S0A PASS artifacts remain
open.

After owner normalization and the zero-SCC audit, the bounded all-target check, `47/47` resource
library tests, and all seven external resource owner targets pass. The transaction-evidence
target's isolated panic-child output is expected fault injection; the parent test exits
successfully. `runtime_vnext_g01a_checkpoint.py --self-test` and `run_gate.py --self-test` also
pass with the split matrix.
