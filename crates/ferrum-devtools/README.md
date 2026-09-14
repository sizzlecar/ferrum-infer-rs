# Internal developer tools

## Native operation profiles

Capture a completed `ferrum run` or `ferrum serve` trace with
`--profile-detail kernel --profile-jsonl /path/to/profile.jsonl`, then summarize it:

```console
cargo run -p ferrum-devtools --bin native_profile -- \
  --input /path/to/profile.jsonl --output /path/to/new-summary.json
```

The output must be a new file. Shared commands are counted once by physical
submission and command index; conflicting copies are errors. Operation totals
and subwork intervals are separate views of the same work and must not be added
together. Unavailable or invalid measurements, missing identities, and command
index gaps remain visible in the report.

Observed token shapes do not establish whether a submission is prefill, decode,
or mixed. Missing final commands and entire missing submissions cannot be
reconstructed from native records alone. Kernel instrumentation changes execution
boundaries: these interval sums diagnose operator cost, and do not establish
production latency, output correctness, or an end-to-end performance advantage.
