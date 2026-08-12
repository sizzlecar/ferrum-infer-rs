# Runtime vNext v0.8.0 host-suspend evidence amendment (2026-08-12)

## Status and precedence

- Status: Active, one-time, fail-closed.
- This amendment applies only to the M1 Metal `c19-006` artifact and the single
  validation-only Git hop defined below. It does not establish a reusable host-suspend policy.
- It overrides the ordinary wall-clock/duration consistency check only for this exact
  `c19-006`. It overrides the final-source-SHA clause in
  [`CORRECTNESS_ACCEPTANCE_AMENDMENT_2026-08-07.md`](CORRECTNESS_ACCEPTANCE_AMENDMENT_2026-08-07.md)
  only for the exact `a609cac8 -> final` direct-child control-plane commit below.
- It does not waive a case, change an oracle, edit a timestamp, edit any original artifact byte,
  authorize a model rerun, or authorize a performance claim. It does not alter R2/R3, G09/G10,
  staged-binary, final-RC same-SHA, or release-asset requirements.

## Exact source artifact

The only eligible artifact root is the existing M1 Metal root named
`runtime-vnext-r1-metal-a609cac8-20260812-r1/m1-metal`, with all of these identities:

| Field | Required value |
|---|---|
| source Git SHA | `a609cac8099e0190004a7f6523166f281c6b9ad2` |
| source tree | `d2fa2ffd22d322cf4a7188562f121a3a8babc0c7` |
| execution contract | `g08-model-matrix-v1` |
| model/backend | `m1-qwen35-4b` / `metal` |
| hardware | `metal-m1max-32gb-24core` |
| model revision | `e87f176479d0855a907a41277aca2f8ee7a09523` |
| model file SHA256 | `Qwen3.5-4B-Q4_K_M.gguf` = `00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4` |
| binary SHA256 | `0c80d38bd53909178d048ef2b72ace5e367ed23ad0dbde931c319ac9fbbf4d04` |
| original scenario runner SHA256 | `c1fe55826770b34426eec21eff0554cf2849e52abfa8715488e9e1eb95d72e00` |
| models lock SHA256 | `107805513950ed9dd9d0dc064f85252ba8999bffdb2daf216d3477d8eb824039` |
| execution manifest SHA256 | `9378a7dce1225cd3b3c6716ae5ee656d365b2148692fbfea468da3d107149d8d` |
| expectations catalog SHA256 | `13707b0be6e3645700f7572a5735371206ac6debdab1c7523bf8c40dbc3833d7` |

The sealed evidence directory is exactly
`control/diagnostic-only-host-sleep-c19-006`. It contains one directory,
`original-case/`, and exactly the following 16 regular files; missing, extra, symlinked, differently
sized, or differently hashed entries are a hard reject:

| Relative path | Bytes | SHA256 |
|---|---:|---|
| `full-702.receipt.json` | 1986 | `523e5645c50c1701a261d21847b05863e9f8ace127366b90ecc50961418b0884` |
| `full-702.stderr.log` | 212 | `29c0855c7536dd5bae1280ad1a2ae80082ba235ffa054a1c472095a05ae0ea4a` |
| `hashes.receipt.json` | 1847 | `1fcb30b6aeb3f1bd7f7a5c10b3bec367427b4a0938621a56413e6f506b6438bc` |
| `hashes.stderr.log` | 273 | `fb25e3c091eb2fab45e9a6dc5f67012148222e5492b840d4851050ec55dd541a` |
| `hashes.stdout.log` | 644 | `7bef53ad1d49926050d6520142e739d174cd171f4c210492a9bb1b9a49e5f047` |
| `original-case/case.json` | 4213 | `2fde53ce63f360610b4d6bf6f2d9d386393a9cbe4cebc0c5c24645653337d2b8` |
| `original-case/checker.log` | 95 | `440747fd851a55d5227a928da567e54d55ab3f7629f8593c58e668075c301090` |
| `original-case/command-spec.json` | 1102 | `eb4f389e67d7cb44b901408fbfc7276bc4600f10bb8dfd8218ec03b245fdd2ad` |
| `original-case/execution-envelope.json` | 7776 | `66925d45f5dd7d5038b32104ce55d0e9a5a2d32e0a544e58f09dd7dcdaa918cb` |
| `original-case/http-transcript.json` | 19736 | `d0034d2e33bf044a2a1288f06ad45eb831710bd7f8c739a93fa4dbab3f72514f` |
| `original-case/input.json` | 670 | `ed932f66a93bf4467a06a4342d499fecf0e904f4c2e2a6127e6c3ee04487bee8` |
| `original-case/stderr.log` | 65 | `23d9cd0107f60ee5398c48ec0f37e24e5045012b0d39d33e096b84418fd9bc30` |
| `original-case/stdout.log` | 6455 | `8004936d04da855fbac10c18dcf0c3fef1ba91d647b0f4eeff51b4ba86aad683` |
| `pmset.receipt.json` | 1527 | `8528143e86326404aa22a03b5b6b2e31441ec381be2441f13dda6786648e3966` |
| `pmset.stderr.log` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `pmset.stdout.log` | 9401806 | `2b88f8475fc97b6f507f095139cfdc6a6246bd54b34efc72c7cddec714fff371` |

The copied `original-case/*` files must be byte-identical to the canonical
`correctness/m1-qwen35-4b/metal/scenarios/C19/cases/c19-006/*` files.
The copied `full-702.receipt.json` and `full-702.stderr.log` must be byte-identical to their files
under `control/`; the receipt's stdout must bind the empty
`control/full-702.stdout.log` with SHA256
`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.
The bounded run must remain the original `rc=1`, `reason=command_exit_nonzero` receipt whose sole
reported failure is `case c19-006 duration does not match timestamps`; it is not rewritten as a
successful process receipt.

## Exact host-suspend proof

The assembler must parse LF-terminated raw bytes from the sealed output of the bounded command
`pmset -g log`. Inside the `c19-006` wall window it must find exactly one qualifying pair: the
following `Sleep` line and its first following `DarkWake` line. The text, byte count, SHA256,
timestamp, reason, and advertised sleep duration are all fixed:

| Event | Exact facts |
|---|---|
| Sleep | `2026-08-12 10:11:42 +0800`, `Entering Sleep state due to 'Clamshell Sleep'`, `83 secs`, 148 LF-terminated bytes, SHA256 `96ecb23ed53e19546eacb093200cbc53e5cf21383dad2c81b9d31e595299be78` |
| first following DarkWake | `2026-08-12 10:13:05 +0800`, `DarkWake from Deep Idle [CDNP]`, 162 LF-terminated bytes, SHA256 `1c0e9abe6cf5cbb5344f361c54d6094cb49de9888fdb12b922b201971f16fe13` |

The parsed timestamp delta and the Sleep line's duration must both equal exactly `83` seconds.
Both events must fall inside each applicable wall window below. No fuzzy text match, different
sleep reason, later wake, adjacent sleep episode, or hand-entered timestamp is acceptable.

The typed discontinuity receipt must contain exactly these three process windows and recompute
their values from the original bytes:

| Window | Required process and timing facts |
|---|---|
| `c19-006-spawn` | PID/PGID `58900/58900`; wall `2026-08-12T02:11:35.159184Z` to `2026-08-12T02:14:32.144726Z` = `176.985542s`; monotonic `2597678276620375` to `2597773927392791` = recorded `95.650772416s`; wall-minus-monotonic `81.334769584s`; residual `83 - 81.334769584 = 1.665230416s` |
| `scenario-executor-invocation` | PID/PGID `58900/58900`; wall `2026-08-12T01:06:38.080553Z` to `2026-08-12T02:31:30.022205Z` = `5091.941652s`; monotonic `2593781121646375` to `2598791825307916` = recorded `5010.703661541s`; discontinuity `81.237990459s`; residual `1.762009541s` |
| `bounded-full-702` | PID/PGID `58900/58900`; wall `2026-08-12T01:06:34.734Z` to `2026-08-12T02:31:41.093Z` = `5106.359s`; receipt duration `5025.120741s`; discontinuity `81.238259s`; residual `1.761741s` |

For every window the mechanical rule is:

```text
discontinuity_sec = wall_duration_sec - monotonic_or_receipt_duration_sec
residual_sec = 83.0 - discontinuity_sec
0.0 <= residual_sec <= 2.0
```

The fixed two-second ceiling is only the bound for `pmset`'s integer-second event accounting. It
is not configurable. The assembler must also bind the `c19-006` case execution and envelope spawn
fields byte-for-byte, the scenario-executor receipt/invocation, server PID `84281`, common PGID
`58900`, and the original bounded limits (`max_processes=8`, `max_group_threads=128`,
`max_per_process_threads=64`, `wall_timeout_seconds=7200`) and observed peaks (`3`, `35`, `18`).
Cleanup must remain `process_group_gone=true`.

Across all 702 canonical `case.json` files and their 702 execution envelopes, exactly one case may
fail the ordinary `max(0.05s, 5%)` wall/duration comparison: `C19/c19-006`. Every other timing,
identity, artifact, product, oracle, resource, and scenario check remains unchanged and must pass.
Every canonical case, execution, envelope, receipt, and wire timing field must retain its original
`started_at`, `finished_at`, monotonic value, and `duration_sec`; the assembly records the external
clock suspension rather than "correcting" any timestamp.

## Artifact-only assembly

The only authorized command is:

```text
python3 scripts/release/runtime_vnext_baseline_scenarios.py \
  --manifest <artifact-root>/execution-manifest.json \
  --artifact-root <artifact-root> \
  --host-suspend-evidence \
    <artifact-root>/control/diagnostic-only-host-sleep-c19-006 \
  --assemble-existing
```

`--execute`, `--discover`, `--out`, a model path override, or any product/model subprocess is
forbidden. The fixed output directory is
`<artifact-root>/derived/host-suspend-c19-006/`; it must not exist at start and must be published
atomically. It contains exactly:

| Output | Required typed identity |
|---|---|
| `original-artifact-inventory.json` | `runtime-vnext-original-artifact-inventory-v1` |
| `host-suspend-provenance.json` | `runtime-vnext-host-suspend-c19-006-provenance-v1` |
| `scenario-report.json` | normal scenario-report schema plus an `assembly` object of kind `runtime-vnext-host-suspend-c19-006-assembly-v1` |
| `assembly-manifest.json` | `runtime-vnext-host-suspend-c19-006-manifest-v1` |

The original bounded run failed before constructing its top-level scenario report, and the seven
top-level commands do not persist a product-process termination wall timestamp. The assembly must
not claim that such a timestamp was recorded or reconstructed. Only in this exact assembled report,
each command records a mechanically derived observation interval with
`window_semantics="observed-live-window"`; this is neither a process-lifetime interval nor evidence
of its wall-clock termination. Ordinary scenario reports must not contain any of the derived fields
below and retain their existing validation behavior.

Every one of `actual-run-01` through `actual-run-05`, `actual-serve-01`, and `actual-serve-02` must
carry these exact derivation bindings:

```text
started_at_evidence = {
  kind: "product-process-started-at",
  case_id: <command-owned case>,
  execution_envelope: {kind: "raw-json", path: <path>, sha256: <sha256>}
}
started_at = execution_envelope.product_process.started_at

finished_at_evidence = {
  kind: "last-case-finished-at",
  case_id: <command-owned case with greatest execution.finished_at>,
  case_json: {kind: "raw-json", path: <path>, sha256: <sha256>}
}
finished_at = case_json.execution.finished_at
duration_sec = finished_at - started_at
```

The validator must resolve both typed artifact references, verify their hashes and command/case
ownership, select and recompute the greatest command-owned case `execution.finished_at`, and
recompute the wall difference. It must reject a value copied without its derivation or any attempt
to label the derived `finished_at` as process termination. The command-specific terminal binding is:

| Command | Exact `terminal_evidence` and source |
|---|---|
| `actual-run-01` | `{kind: "resident-jsonl-exit-event-monotonic", wire_receipt: {kind: "raw-json", path, sha256}, exit_received_monotonic_ns, controlled_stop: true}` from that command's hash-bound `wire_receipt.exit_event` |
| `actual-run-02` | same structure, recomputed from `actual-run-02`'s hash-bound `wire_receipt.exit_event` |
| `actual-run-03` | same structure, recomputed from `actual-run-03`'s hash-bound `wire_receipt.exit_event` |
| `actual-run-04` | same structure, recomputed from `actual-run-04`'s hash-bound `wire_receipt.exit_event` |
| `actual-run-05` | same structure, recomputed from `actual-run-05`'s hash-bound `wire_receipt.exit_event` |
| `actual-serve-01` | `{kind: "wall-terminal-not-persisted", last_observed_case_id: <finished_at_evidence.case_id>}` |
| `actual-serve-02` | `{kind: "wall-terminal-not-persisted", last_observed_case_id: <finished_at_evidence.case_id>}` |

For each run command, `exit_received_monotonic_ns` must equal the bound exit event's
`received_monotonic_ns`, and `controlled_stop` must bind the wire receipt's true controlled-stop
fact. This monotonic terminal observation is not converted into or used to fabricate a wall
timestamp. For each serve command, `wall-terminal-not-persisted` is an explicit absence marker;
the last observed case remains an observation boundary, not a termination claim.

Before writing, the assembler recursively indexes every regular file below the artifact root,
excluding only the fixed derived directory, as `(relative path, size, SHA256)`. It rejects every
symlink and every entry other than a regular file or directory. After atomic publication it
recomputes that original-file inventory; the before/after rows must be identical. The canonical
inventory has exactly `5772` files, `1739606140` bytes, and canonical rows SHA256
`f297460de16f9a9ea83681a618b40d722d805c5652f77146f2924ad622972c4a`. The provenance and
manifest bind both the original a609 artifact source and the final validation source/tree,
assembler Git blob, and direct-child bridge receipt. The manifest also binds the SHA256 and size of
the other three derived outputs and records `product_processes_started=0` and
`model_processes_started=0`.

The derived scenario report is a fresh validation/assembly result, not edited raw evidence. It may
report `702/702 PASS` only after revalidating all original cases and the exact discontinuity receipt;
known-fail, blocked, skip, waiver, error, and unexpected remain `0`. Any original byte change,
additional timing mismatch, ambiguous/missing `pmset` pair, residual outside `[0,2]`, failed oracle,
extra output, or attempted process launch must reject the assembly.

## One direct-child validation bridge

The source history is ordered and must not be flattened, squashed, rebased, or replaced by an
aggregate diff:

```text
05a5d2f8611ed3a3fedb5c69ff3ba11e533bc4c7
  -> a609cac8099e0190004a7f6523166f281c6b9ad2
  -> <final validation-only direct child>
```

The first hop is the unique direct child of `05a5d2f8` and remains exactly the following five-path
G02 roster bridge already sealed in the correctness amendment:

```text
docs/goals/runtime-vnext-0.8.0-2026-07-10/CORRECTNESS_ACCEPTANCE_AMENDMENT_2026-08-07.md
scripts/release/runtime_vnext_g02_core.py
scripts/release/runtime_vnext_r0_core_closure.py
scripts/release/runtime_vnext_r1_product_correctness.py
scripts/release/runtime_vnext_s2_cuda_product_contract.py
```

In particular, `runtime_vnext_g02_core.py` changes Git blob
`38b832c95ecee833240a1477678fb5ce350f52fb` to
`fa369a3ee52535ead59aefb4b3f675844feb09b8`; no additional interpretation of that hop is allowed.

The final commit must have exactly one parent, exact SHA `a609cac8099e0190004a7f6523166f281c6b9ad2`,
and exactly this ordered raw mode/status set:

```text
000000 100644 A docs/goals/runtime-vnext-0.8.0-2026-07-10/HOST_SUSPEND_EVIDENCE_AMENDMENT_2026-08-12.md
100644 100644 M scripts/release/runtime_vnext_baseline_scenarios.py
100644 100644 M scripts/release/runtime_vnext_r0_core_closure.py
100755 100755 M scripts/release/runtime_vnext_r1_product_correctness.py
```

At `a609cac8`, the three modified files must have Git blobs
`e667cef1b2bad37d439be472abd09d2203bd42c1`,
`d86d3cb9719b5d669802bbf22b81cafc9d060360`, and
`e23b242414afee16b0435099900bf78a4e832d12`, respectively, and the amendment path must be absent.
The final validators must bind the exact base/final SHA and tree, all four final Git blobs, and the
raw name-status/diff. The amendment and assembler final blobs must be sealed constants; the R0/R1
validator blobs must be read from the exact final commit to avoid a self-hash exception. Any fifth
path, different status, merge parent, intermediate commit, blob drift, `crates/`, Cargo, model lock,
runtime config, scenario manifest, product implementation, or release-lane change fails closed.

This hop may only rerun artifact assembly and R0/R1 validators. It may not run a model and may not
turn old failed product behavior into PASS.

## Evidence-consumption boundary

- Evidence already recorded at `a609cac8` may cross only the exact `a609cac8 -> final` hop.
  This includes the a609 R0 checkpoint, eligible a609 matrices, and Llama CUDA/Metal supplemental
  evidence. Llama is accepted only when it is exact-final evidence or exact-a609 evidence crossing
  this one hop.
- Evidence recorded at `05a5d2f8` may cross only the already sealed `05a5d2f8 -> a609cac8` G02
  roster hop followed by this exact `a609cac8 -> final` hop. Validators must preserve and report
  both receipts in order; they may not validate a flattened `05a5d2f8..final` allowlist.
- Llama evidence has no `05a5d2f8 -> final` or two-hop exception. The current final-SHA clause is
  overridden for Llama only when its source is exactly `a609cac8` and the only later commit is this
  final direct child.
- S2 receives no new standalone bridge. No other ancestor, artifact family, model, backend, goal,
  performance row, binary, or future commit may cite this amendment as freshness authority.

The resulting artifact and R0/R1 PASS lines are correctness/validation evidence only. They are not
performance evidence, a performance waiver, release readiness, or permission to skip the final
same-SHA staged-binary correctness and performance matrices.
