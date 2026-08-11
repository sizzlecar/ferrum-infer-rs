# Reusable Execution Capture Policy

Status: v0.8.0 implementation decision; focused CUDA validation pending.

## Problem

Reusable workspace capacity and reusable device-program identity are different
concepts:

- a workspace bucket may cover several smaller logical work shapes;
- a captured device program is reusable only for the complete replay-affecting
  physical identity observed for a logical startup case.

The vNext physical program identity already includes exact immediate sequence,
token, page, topology, plan, provider, lane, and stable-layout identity. A
wider workspace bucket therefore does not authorize replaying a program
captured for another width. Conversely, one logical width is only a startup
capture case: it does not claim coverage for every context/page/provider
topology variant at that width.

The former startup matrix captured only power-of-two decode widths. Real
continuous batches such as width 10 consequently missed the sealed catalog and
fell back to eager execution. The local M3 CUDA artifact on source `02616ff4`
measured this directly: preserving the catalog removed epoch misses and raised
C32 throughput from `137.11` to `209.27 tok/s`, but exact-shape catalog misses
remained `639/1908` waves.

Ferrum must not solve those misses by aliasing exact widths to capacity buckets.
Historical case H07.1 did that for legacy MoE graphs: actual widths `17..32`
shared one padded key, stale slots corrupted logits, and requests terminated
early. Commit `db7e5293` restored actual-width graph identity.

## vLLM reference and boundary

The locally installed vLLM sources were inspected at:

- current `33c4f3551c` (`v0.23.1rc0-1467`);
- stable `bc150f5029` (`v0.20.2`).

vLLM exposes typed capture sizes and a maximum capture size. Its balanced and
throughput modes use sparse sizes and dispatch an actual size to the next
captured size. Interactivity mode captures every width through 32 to reduce
padding overhead. The sparse policy is safe only because vLLM separately owns
logical and padded token/request counts, invalid padding slots, KV/state write
suppression, attention metadata, output slicing, sampling exclusion, DP shape
agreement, and eager fallback.

Ferrum does not yet have that padding ABI. Its submission participants are real
active sequences with resource authority, and attention, KV/recurrent state,
sampling, readback, and program identities all use the exact participant set.
Sparse capture plus padding is therefore out of scope for this fix.

## v0.8.0 decision

### One resolved authority

Add a backend-neutral resolved device-program policy to the vNext reusable
execution contract. It is embedded in `ResolvedRuntimePolicy`, validated before
plan compilation, serialized in effective evidence, and included in the policy
fingerprint.

The resolved policy owns:

- exact logical decode startup cases;
- exact logical prefill startup cases, including prerequisite prefix cases;
- shape semantics (`exact` for v0.8.0);
- catalog miss behavior (`eager_fallback`);
- catalog lifetime (`startup_sealed`).

The following values must all derive from that same resolved shape matrix:

- immutable runtime-policy fingerprint;
- reusable memory plan `maximum_device_executables`;
- backend `DeviceReusableExecutionPlan` capacity;
- startup capture order and logical-case-to-physical-ProgramId receipts;
- effective config and executor startup report.

Workspace buckets remain independent power-of-two capacity chains. They may
cover an exact work shape but may never substitute their capacity for the
device-program identity.

### Product intent and resolution

The product-facing config is typed. `ferrum run` has an entrypoint default of
admission `1` and exact decode list `[1]`; both values appear in its effective
config and can be overridden by normal config/env/CLI precedence. `ferrum
serve` with an omitted list requests every exact width through
`min(admission, configured automatic ceiling, hard startup ceiling)`. M3
therefore resolves to `1..32` because its typed admission is 32.

Users may provide an explicit bounded exact-size list when they deliberately
prefer lower startup cost and accept observable eager fallback on omitted
widths. Both automatic and explicit lists are subject to the independent hard
startup width bound `32`; this is not a runtime concurrency limit. Admission
may exceed 32, but widths above the hard startup bound use eager fallback.
Lists are validated, deduplicated, canonically sorted, and bounded before plan
compilation. No user concurrency value directly controls synthetic worker or
sequence creation.

If the synthetic startup state budget cannot execute the requested automatic
matrix, the resolved effective matrix and the reduction reason must be recorded
before compilation. An explicit user list must not be silently reduced.

### Runtime behavior

- Capture largest shapes first.
- Seal the catalog before product requests.
- Keep the exact logical shape in policy and capture receipts, and keep every
  replay-affecting physical dimension in the ProgramId/provider topology key.
- Require every logical startup case to observe at most one physical ProgramId
  under the v0.8.0 budget; multiple variants fail closed rather than silently
  exceeding residency accounting.
- Permit multiple logical prefill cases to share one ProgramId only when each
  case independently observed that exact complete physical identity. This is a
  legitimate many-to-one physical reuse; a case is never marked covered merely
  because another case observed the program.
- Record every sealed physical ProgramId, pages/topology identity, resident
  segments, eager boundaries, and gaps. A logical case is reported prepared
  only when its observed physical program has a resident segment.
- On a catalog miss, run eager and record the exact sequences/tokens/pages,
  topology, and reason (`program_identity_unavailable`, `catalog_empty`,
  `program_absent`, `program_non_resident`, or `epoch_mismatch`). The miss
  ledger is capped at 64 unique keys with explicit per-reason overflow counts;
  catalog hits take no new lock and allocate nothing.
- When reusable execution is disabled, unsupported, or otherwise resolves no
  startup plan, eager dispatch is intentional and is not classified as a
  catalog miss.
- Never reopen, evict, or replace a sealed catalog while a product request can
  reference it.
- Do not infer policy from a model name, GPU name, request count, or hidden
  environment-variable combination.

## Ferrum advantage

Ferrum's default is correctness-verifiable reuse, not optimistic padded reuse:

- exact work identity is hash-bound from policy through replay;
- eager/replay equivalence includes declared outputs and KV/recurrent post-state;
- unsupported or missing shapes fail over explicitly instead of silently
  aliasing another graph;
- capture policy, startup cost, resident capacity, hit/miss shape distribution,
  and physical memory headroom are release evidence;
- both `ferrum run` and `ferrum serve` use the same typed policy mechanism,
  while their intentional single-request/server defaults remain visible and
  independently overridable.

This costs more startup work than a sparse padded matrix, but avoids repeating
H07 and gives a safe base for later optimization.

## Follow-on design

1. Support next-start adaptation: an explicitly supplied, artifact-derived
   exact matrix for the next process, without live catalog mutation.
2. Add quiescent adaptive exact capture only with catalog generations, atomic
   publication, old-generation fence drain, bounded residency, and rollback.
3. Design padding as a separate goal. It requires distinct logical/physical
   shapes, inert dummy backing, KV/recurrent write suppression, attention masks,
   sampling/readback exclusion, provider padding-equivalence capabilities, and
   H07 plus per-tail eager/replay bitwise tests.

Adaptive capture and padding must not land in one patch.

## Focused acceptance before resuming R2

After the typed-policy change, run only the affected M3 CUDA focused lane:

1. product correctness for the existing focused `run` and `serve` cases;
2. same-server C16/C32 benchmark;
3. exact startup report and resolved policy fingerprint;
4. catalog misses by exact width and direct replay ratio;
5. startup elapsed time, resident executable count, peak VRAM, and at least
   512 MiB physical headroom.

This focused run is KEEP only if correctness remains clean, epoch misses remain
zero, ordinary decode-width misses are eliminated or fully classified, direct
replay meets the focused gate, and the memory headroom hard gate passes. It does
not rerun R0 or R1 and is not an R2 PASS by itself.
