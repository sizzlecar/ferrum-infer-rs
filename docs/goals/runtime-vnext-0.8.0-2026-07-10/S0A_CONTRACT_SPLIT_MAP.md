# S0A Contract Split Map

## Status

- Work package: S0A, semantics-preserving structural split.
- Current stage: `resource.rs` production contract split and symbol-level imports validated;
  dependency-cycle removal, `execution.rs`, `event.rs`, and owner-aligned external test targets
  remain open.
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

Before module visibility changed, the 13 physical fragments concatenated byte-for-byte with the
original seven-line test-module tail to the pre-split SHA256. The validated implementation now
uses real child `mod` declarations and facade `pub use` exports; it does not use `include!`.
Existing public paths remain unchanged. Cross-owner implementation details that were implicitly
shared by the giant module are now explicitly limited to the `vnext::resource` parent with
`pub(super)`, so they do not become crate-public API. All 13 production fragments now use explicit
symbol imports; only the two existing in-module test files retain `use super::*`. The resulting
symbol graph exposes the monolith's previously hidden bidirectional owner dependencies. S0A must
move those shared contracts to their lowest valid owner and produce a zero-cycle audit before it
can pass.

As a second mechanical-equivalence check, removing only the added `use super::*` lines and
`pub(super)` visibility qualifiers, concatenating the production fragments, and applying the same
`rustfmt` produced byte-identical old/new files with SHA256
`3fac9f1b587513d77fc796538ae40444cfe9be08992bdf0d8b36f1f88168560b`.

## Resource Ownership

| Original lines | New owner | Lines | Primary responsibility |
|---:|---|---:|---|
| 1-513 | `resource/contracts.rs` | 490 | Base identifiers, descriptors, reservation contracts, shared validation |
| 514-1226 | `resource/capacity.rs` | 728 | Device capacity authority, epochs, static provisioning and admission |
| 1227-1838 | `resource/allocation.rs` | 620 | Transaction identity, allocation ownership and resource driver contracts |
| 1839-3307 | `resource/ledger.rs` | 1,485 | Lease state, transition receipts, allocation ledger and failure evidence |
| 3308-3748 | `resource/recovery.rs` | 450 | Owned lease slots, abandoned-sequence recovery and recovery stream state |
| 3749-5725 | `resource/dynamic_pool.rs` | 1,988 | Dynamic backing pools, growth, extent/view ownership and quarantine |
| 5726-6005 | `resource/static_lease.rs` | 287 | Plan-static provisioning lease and typed admission request construction |
| 6006-6361 | `resource/work.rs` | 363 | Step/invocation work-shape admission requests and checked demand derivation |
| 6362-7481 | `resource/plan_runtime.rs` | 1,145 | Plan runtime root, close state, capacity wait and logical backing ownership |
| 7482-8448 | `resource/sequence.rs` | 983 | Request, sequence and session resource lifetime authorities |
| 8449-9459 | `resource/batch.rs` | 1,028 | Batch participants, physical invocation ledger and step retirement |
| 9460-11117 | `resource/invocation.rs` | 1,680 | Step/invocation leases, retry authority and active-sequence permits |
| 11118-13213 | `resource/transaction.rs` | 2,114 | Sealed transaction typestate, commit/rollback/release and receipt validation |

`resource.rs` is now a 67-line facade. Every production fragment is below the S0A `2,500`
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

The intended lower-to-higher direction is:

```text
contracts
  -> capacity / allocation
  -> ledger / dynamic_pool / recovery
  -> static_lease / work
  -> plan_runtime
  -> sequence
  -> batch
  -> invocation
  -> transaction facade and validation
```

Rust module privacy now separates the owners, all newly shared internals are restricted to the
resource parent, and production imports are symbol-explicit. The exposed graph still contains
bidirectional edges, including capacity/pool orchestration, ledger/transaction validation, and
sequence/batch/invocation lifecycle helpers. Those are recorded defects of the old ownership
layout, not accepted final dependencies. S0A must relocate them and emit the cycle audit; S0B may
then shrink or break the resulting contracts against the real Qwen3.5-4B production consumer.

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

After replacing all production glob imports with explicit symbol imports, the same bounded
`cargo check -p ferrum-interfaces --all-targets` passes. The behavioral tests were not repeated
for this import-only follow-up.
