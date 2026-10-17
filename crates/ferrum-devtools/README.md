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

## Standard-answer reference

`reference_quality` collects a report-only llama.cpp reference from a completed
strict quality capture. It sends the original actual prompt token IDs to the
native `/completion` endpoint and preserves the frozen cases, concurrency,
generation limits and exact-answer scoring. It does not change or rerun the
paired quality gate. The same GGUF and server capacity must be pinned in an
external lifecycle guard; this tool does not start a server.

After binding the reference and collector binaries, obtain the runtime pin with
`reference_quality guard-pin --config /path/to/guard.json`. Pass that value to
`reference_quality run --guard-runtime-sha256 ...`; `run --help` lists the
required source-capture and output paths. The pin excludes client arguments to
avoid self-reference and includes the server configuration and asset identities.
Without `--execute`, the command validates inputs and writes requests only.

All raw replies, terminal token IDs and truncated answers remain in the output.
Invalid evidence makes the reference incomplete; a valid length truncation is
scored wrong. Submitted prompt IDs are distinguished from reference-side token
observations. Reports do not establish a noninferiority or performance result.
