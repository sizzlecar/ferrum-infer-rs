# Shared-cohort structured V2 calibration

`ferrum calibrate-slo` supports `validation_model.kind =
"structured_whole_wave_group_v2"`. The existing schema-1 rendered inputs or
schema-2 pinned selection and Chat conversion remain unchanged. This mode trains
several explicitly declared owners from one complete request schedule. It does
not discover membership during training or certify all future planner branches.

The validation model has `capture` and `residual` fields. `capture` contains:

- `warmup`: shared complete cohorts, run before any source opens;
- `catalog`: the output catalog path;
- `children`: one ordinary V2 capture declaration per distinct owner, with its
  own `scope`, `membership_rule`, `settings`, `phase_members`, `source`, `profile`,
  file bound and clock-error declaration. Child `warmup` must be empty;
- `limits`: `maximum_children`, `maximum_total_file_bytes`,
  `maximum_retained_numeric_bytes`, and `maximum_retained_coordinates`.

Use independently observed owner identities and freeze all windows and expected
counts before capture. `training`, `residual` and `validation` are the common fit,
residual and qualification complete cohorts. Every child sees the original FIFO,
including outside attempts and failures. Each wave's membership is assigned before
execution; its original wall observation is validated once and delivered only to
the already attached children. A child reaching its count cannot end a cohort or
advance the common phase early. The source3 and profile10 formats are unchanged.

Use the shared SLO policy with Observe, CompleteRequests, credited output,
`cost_observation.predictor = "structured_whole_wave_v2"`, and
`structured_capture = "host_settled_v1"`. Do not import a cost profile or set a
legacy export during capture. Declare the actual local clock accuracy in
`cost_observation.profile_import.declared_local_clock_max_error_ns`.

The import budget applies to the whole catalog: metadata, all profile/source
bytes, all children’s reserved populations and all source shape rows, including
outside rows. The existing hard file limit is 256 MiB. Declare enough capacity
before the run; increasing a bound does not establish statistical support.
Capture separately enforces aggregate source bytes, retained numeric storage and
coordinate bounds. Output paths resolve against the manifest directory; their
parents must exist, and existing files are never overwritten.

Run with the existing command and backend/model/resource arguments:

```text
ferrum calibrate-slo MODEL --manifest MANIFEST --slo-config POLICY --out REPORT --observations RAW --startup-usage serve
```

All children must finish qualified before the first profile export. Each export
replays its real source; the final catalog is checked by the original product
loader using the live runtime identity and clock. This inspection installs no
model. Success appears in `summary.structured_calibration_group_v2.verified_catalog`.
Failures retain available raw/source/child artifacts and an error; partial files
are not a verified catalog. Original TTLs are never refreshed. The default sample
TTL remains 300 seconds, so a declared schedule plus export/import work must fit
the original lifetime. A timeout still attempts normal session cleanup.

Serving can then explicitly import the verified catalog with the V2 predictor in
either `run` or `serve`. Missing owners, pending/Length combinations, numeric
support, or expired samples remain Unknown. Qualification is an empirical
coverage check, not a p99 guarantee or a throughput/SLO acceptance result.
