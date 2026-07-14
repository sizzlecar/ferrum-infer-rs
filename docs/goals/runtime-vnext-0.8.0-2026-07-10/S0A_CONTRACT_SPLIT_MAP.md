# S0A Contract Split Map

## Status

- Work package: S0A, semantics-preserving structural split.
- Current stage: `resource.rs` production contract split, symbol-level imports, and complete
  dependency-cycle removal validated; `execution.rs`, `event.rs`, and owner-aligned external test
  targets remain open.
- This map records file ownership, not G01A completion. Public item inventory and the final
  `contract-map.json` remain required before the S0A PASS artifact can be issued.

## Source Evidence

- Pre-split source: `crates/ferrum-interfaces/src/vnext/resource.rs`
- Pre-split logical lines: `13,220`
- Pre-split SHA256:
  `26b3e035010111b0d1da2f1133b665c207c5802e689ab02b5f3bc35c9933a97d`
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

The final graph is acyclic. One valid dependencies-first topological order is:

```text
contracts -> ledger -> capacity -> allocation -> dynamic_pool -> provisioning -> static_lease
-> plan_runtime -> transaction -> work -> recovery -> sequence -> batch -> invocation
```

Rust module privacy now separates the owners, all newly shared internals are restricted to the
resource parent, and production imports are symbol-explicit. The SCC audit reports `0` cycles;
pairwise and the previously hidden three-module cycle are both eliminated. S0B may later shrink or
break these contracts only against the real Qwen3.5-4B production consumer.

## Validation For This Stage

1. Initial reconstructed source SHA256 equals the pre-split source SHA256.
2. Normalized old/new production source SHA256 values are identical after stripping only the
   module visibility/prelude additions and applying the same formatter.
3. `cargo fmt --all -- --check` accepts the fragments and facade.
4. `CARGO_BUILD_JOBS=4 cargo check -p ferrum-interfaces --all-targets` passes.
5. `RUST_TEST_THREADS=2 cargo test -p ferrum-interfaces --lib resource:: -- --test-threads=2`
   passes `47/47` tests.
6. `RUST_TEST_THREADS=2 cargo test -p ferrum-interfaces --test vnext_resource_contract_tests`
   with `-- --test-threads=2` passes `7/7` tests, including the expected isolated panic-child
   fault case.
7. No paid GPU, model download, performance run, or product migration claim is part of this stage.

After owner normalization and the zero-SCC audit, the bounded all-target check, `47/47` resource
library tests, and `7/7` external resource contract tests pass. The external target's isolated
panic-child output is expected fault injection; the parent test exits successfully.
