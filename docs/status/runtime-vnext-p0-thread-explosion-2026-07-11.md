# Runtime vNext P0: Resource Test Thread Explosion

Status: the crash path is fixed. Its bounded exact regression and the broader
`vnext_resource_contract_tests` contract pass locally. These are development artifacts, not a
formal Runtime vNext Goal PASS.

## Incident Evidence

Two macOS 15.1.1 kernel panics identify `vnext_resource_c` as the panicked task with exactly
8192 threads. Both backtraces include `com.apple.kec.pthread`; memory compression and swap were
reported healthy.

| Panic artifact | PID | Threads | SHA256 |
|---|---:|---:|---|
| `/Library/Logs/DiagnosticReports/Retired/panic-base+socd-2026-07-11-104310.000.panic` | 89326 | 8192 | `2ac1b2c165c21c2606958327b9827f8a9c1c2b2d682c57eae6f1f14b48896036` |
| `/Library/Logs/DiagnosticReports/Retired/panic-full-2026-07-11-153143.0002.panic` | 1733 | 8192 | `a7e53ed76c5ec5fea86c60c4598137bc75bf3ffec25eac7630c5272ca7c4a765` |

## Root Cause

The capacity test used a plan-static claim of 48 bytes, a minimum usable device capacity of
650,000 bytes, and therefore calculated `maximum_claims = 13,541`. It then treated capacity
cardinality as native concurrency and attempted to create `maximum_claims + 5 = 13,546` scoped
OS threads. Every worker waited on a 13,546-party barrier, so no worker could exit while thread
creation was still in progress. When macOS refused another thread, scope unwinding attempted to
join workers that could never reach the barrier.

At incident time, production `resource`, `admission`, `device`, and `operation` code did not create
native threads. The crash was a test-construction defect, not a production runtime thread leak.
Subsequent fail-closed cleanup hardening also creates no internal native thread. Final sequence and
completion owner Drop only transfers ownership into a process-reachable, plan-domain registry;
the scheduler must invoke bounded maintenance on an explicit recovery thread. The `64` threshold
limits unresolved cleanup pressure and maintenance batch size, not user/model concurrency, and the
registry retains additional already-existing owners instead of dropping or forgetting them.

## Fix Invariants

- Test capacity is derived from the real plan-static peak and is exactly 16 claims.
- `maximum_claims == 16` is asserted before any capacity-sized loop.
- Capacity saturation is prepared serially.
- The final atomic race uses the calling test thread plus exactly one named scoped worker.
- A compile-time assertion fixes that worker count at one and below the independent test cap.
- Worker spawn failure occurs before the caller enters the two-party barrier.
- The nested panic-isolation test explicitly uses `--test-threads=1`.
- Repository rules now prohibit deriving native workers from capacity/request/model cardinality.
- Backend recovery is never performed synchronously from `CompletionReaper::drop` or the final
  `AdmittedSequenceResources::drop`; destructor-side backend calls and spawned workers are both
  zero. Each exact owner transfers to the plan-domain registry. Retry, panic and quarantine retain
  the same reachable owner; one blocked task does not hold the registry lock or remove sibling
  tasks from other recovery callers. Close reports pending cleanup and converges after a successful
  retry.

## Verification

Source identity at collection time:

- Git HEAD: `430e5947cfaede13a190c24151fbdc0ddae137d0`
- Worktree: dirty; this is P0 development evidence, not release evidence.
- Test source SHA256: `ba8c79ed78d7853b2880c00bc305b3c738ba8403100d3bb6e728727dd1d35ac1`
- Test binary SHA256: `8cd48076ce961e8d0935b459c6966fbd3f0bb6317c2a65677de3dd71176fcf94`

Static and compile checks:

```text
rustfmt --edition 2021 --check crates/ferrum-interfaces/tests/vnext_resource_contract_tests.rs
PASS

CARGO_BUILD_JOBS=2 cargo test -p ferrum-interfaces --no-default-features \
  --test vnext_resource_contract_tests --no-run
PASS

python3 scripts/release/bounded_command.py --self-test
BOUNDED COMMAND SELFTEST PASS
```

Final bounded exact artifact:

- Root: `/private/tmp/ferrum-p0-thread-explosion-20260711/capacity-exact-final`
- Receipt SHA256: `abc998edd62b069eb21bf784f89dcdd1570fb6d3723575ff7ae6b237f11acac8`
- Limits: 2 processes, 8 group threads, 4 threads per process, 30 seconds.
- Observed peak: 1 process, 2 group threads, 2 threads in one process.
- Sampling errors: 0.
- Cleanup: process group gone.
- Exit code: 0.
- Exact test line: `VNEXT RESOURCE CAPACITY THREAD BOUND PASS: 20/20`.

The final bounded exhaustive resource contract also passes:

- Root: `/private/tmp/ferrum-vnext-final-20260711/resource-311`
- Receipt SHA256: `f5623c300881a58afd723247c12617aa900ba5c4bc5dcfa477926f5282e45c12`
- Limits: 4 processes, 12 group threads, 8 threads per process, 60 seconds.
- Observed peak: 2 processes, 4 group threads, 2 threads in one process.
- Sampling errors: 0.
- Cleanup: process group gone.
- Exit code: 0.
- Exact contract line: `VNEXT RESOURCE TRANSACTION PASS: 311/311`.

The final bounded `ferrum-interfaces --all-targets` run passes as well:

- Root: `/private/tmp/ferrum-vnext-owned-root-20260711/all-targets-tests-3`
- Receipt SHA256: `6dd9ed43bb0782aa8bca0294c93e2a2b9171535b0089e09cbde801b1ea9592fb`
- Limits: 16 processes, 96 group threads, 48 threads per process, 300 seconds.
- Observed peak: 6 processes, 30 group threads, 12 threads in one process.
- Sampling errors: 0.
- Cleanup: process group gone.
- Exit code: 0.

The unified workspace source gate now passes with Criterion benches included:

- Root: `/private/tmp/ferrum-g01a-checkpoint-20260711/unit-gate-env-1`
- Gate line: `FERRUM GATE unit PASS: /private/tmp/ferrum-g01a-checkpoint-20260711/unit-gate-env-1`
- Child line: `G0 SOURCE unit PASS: /private/tmp/ferrum-g01a-checkpoint-20260711/unit-gate-env-1`
- Receipt SHA256: `7f7f4bfb420ec4864c2cc38442c709e27c428c135dfad2a1a219868454346cae`
- Command: `env PYTHONDONTWRITEBYTECODE=1 CARGO_BUILD_JOBS=2 RUST_TEST_THREADS=1 cargo test --workspace --all-targets`
- Duration: 234.31772 seconds.
- Observed peak: 5 processes, 22 group threads, 12 threads in one process.
- Sampling errors: 0.
- Cleanup: process group gone.
- Criterion witness: `engine_bench`, 12/12 cases observed as `Testing ... Success`.
- Validator self-tests: `G0 VALIDATOR SELFTEST PASS` and
  `G1/G3/G4 RELEASE REGRESSION SELFTEST PASS`.

This artifact closes the bounded workspace source-gate failure class. It does not prove G00A,
G01A, accelerator correctness, performance, product migration, or release readiness. The current
G01A validator intentionally rejects the incomplete contract at execution identity version 2.0.

Post-gate P1 review changed completion/resource ownership code, so the preceding unified unit
artifact is stale for the current worktree and must be rerun before commit validation. Focused
evidence for the superseded first implementation was:

- Selective lane drain: `/private/tmp/ferrum-completion-selective-drain-20260711/focused-3`,
  `VNEXT DEVICE OPERATION PASS: 299/299`; peak 2 processes / 5 group threads / 3 per-process.
- Sequence owner non-blocking Drop:
  `/private/tmp/ferrum-blocking-drop-20260711/resource-exact-2`; 1/1 PASS, peak 4 processes / 20
  group threads / 11 per-process.
- Completion reaper non-blocking Drop:
  `/private/tmp/ferrum-blocking-drop-20260711/completion-exact-1`; 1/1 PASS, peak 4 processes / 21
  group threads / 15 per-process.
- All three receipts record zero sampling errors and successful process-group cleanup.

The first implementation was rejected because queue/spawn/backend failure could make an owner
permanently unreachable and one hidden worker could block unrelated recovery. The replacement has
no hidden worker or lossy queue. Current bounded evidence is:

- Registry retry/panic/saturation/sibling-progress unit contract:
  `/private/tmp/ferrum-deferred-cleanup-registry-20260711/registry-unit-2`, 4/4 PASS, receipt SHA256
  `4b136e095345024ba44a828f04f543734f550466ee0740fd7cb267d33fd0b9d4`.
- Exact last-sequence failure then retry:
  `/private/tmp/ferrum-deferred-cleanup-registry-20260711/resource-owner-retry`, 1/1 PASS, receipt
  SHA256 `c73bdd217ddfb302be94aba93447aab5defad4add143ed4021d8d1ef230e2364`.
- Exact completion Drop plus quarantined retry:
  `/private/tmp/ferrum-deferred-cleanup-registry-20260711/completion-retry-2`, 1/1 PASS, receipt
  SHA256 `a01b102c171689c39675d7b6cb31eaa29dd9af7707de052fd3efbd1fc64308ea`.
- Full resource contract:
  `/private/tmp/ferrum-deferred-cleanup-registry-20260711/resource-full`, 7/7 PASS, peak 4 processes /
  21 group threads / 11 per-process, receipt SHA256
  `fec125897de2b9a706b0ff08bd074a392b54eb2320525210964c72fa96f98505`.
- Full device/operation contract:
  `/private/tmp/ferrum-deferred-cleanup-registry-20260711/device-full`, 2/2 PASS, peak 2 processes /
  5 group threads / 4 per-process, receipt SHA256
  `58543475ca0a3fbce74da8ab45848dee82a46bc2a7d25ac1ab6275b86867c45c`.
- `ferrum-interfaces --all-targets`:
  `/private/tmp/ferrum-deferred-cleanup-registry-20260711/interfaces-all-targets-2`, PASS in
  98.277453 seconds, peak 7 processes / 38 group threads / 12 per-process, receipt SHA256
  `da6fcb03884f99e1794aa9cfce2362c73863c84d90c1ba9e31e9ad779e11b0b8`.
- Interface clippy:
  `/private/tmp/ferrum-deferred-cleanup-registry-20260711/interfaces-clippy`, PASS, receipt SHA256
  `f693c126e203de7a7f2a3f0151eb11a3a50f59a6102cbd63fa3542bf823889f4`.
- Every current receipt above records zero sampling errors and confirms the process group is gone.

The current-worktree unified workspace unit gate now passes:

- Root: `/private/tmp/ferrum-deferred-cleanup-registry-20260711/unit-gate-1`.
- Final lines:
  `G0 SOURCE unit PASS: /private/tmp/ferrum-deferred-cleanup-registry-20260711/unit-gate-1` and
  `FERRUM GATE unit PASS: /private/tmp/ferrum-deferred-cleanup-registry-20260711/unit-gate-1`.
- Exact child command:
  `env PYTHONDONTWRITEBYTECODE=1 CARGO_BUILD_JOBS=2 RUST_TEST_THREADS=1 cargo test --workspace --all-targets`.
- Duration: 374.609227 seconds; peak 7 processes / 39 group threads / 12 per-process; sampling
  errors `0`; process group gone.
- Criterion witness: `engine_bench`, 12/12 cases recorded by the final manifest.
- Gate manifest SHA256:
  `921e576c2d3a73600edae98152ae621eb4bdba87404886ff19c20e7c4053d759`.
- Bounded receipt SHA256:
  `dbb730ea288923d9e31adad97763084b11f9c4ddf93e1d312717399958ed34b6`.

These remain dirty-worktree development artifacts. They close the P0/P1 source-gate failure class,
but no formal G00A/G01A, accelerator, performance, product-migration or release PASS is claimed.

## Residual Audit Findings

The repository-wide static audit found similar partial-start barrier risks in CUDA TP worker
startup and an unbounded legacy Python concurrency benchmark. They did not cause these panics and
must be fixed as separate, reviewable hardening changes before their affected lanes are called
release-ready.
