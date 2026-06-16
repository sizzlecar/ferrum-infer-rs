# W2 Batched Graph ShareGPT Diagnostic

## Scope

This artifact isolates product CLI `--batched-graph` on the current HEAD using
the same ShareGPT dataset and c16/c32 cells as the previous graph-disabled
Ferrum diagnostic. It is not release-grade evidence because it uses
`n_repeats=1`, omits `--require-ci`, and does not run the final W2 validator.

## Evidence

- Remote clean worktree HEAD:
  `017300426514d62e8e50ac1546ff77d4d54fd6ce`.
- Remote worktree status: `0` tracked changes in
  `local/remote_clean_worktree.txt`.
- Binary SHA256:
  `d38caf704f252045c29bdfe02795606937f400ab00edef05647da74179b215d5`.
- Server command: `server/serve.command.json`.
- Bench command: `bench/bench-serve.command.json`.
- Effective graph mode: `legacy_batched_decode_graph`.
- Server ready: `ready_at_poll=29`.
- Chat smoke: assistant content `5`, usage present.
- Bench rc: `0`.
- Server error scan: `0` lines.
- Vast cleanup: `vast_shutdown/shutdown_complete.txt` confirms
  `cur_state=stopped` and `actual_status=exited` at `stop_poll_32`.

The local shutdown helper initially parsed the Vast response as `instance`
instead of `instances`; the committed shutdown evidence keeps the correctly
parsed final poll and `shutdown_complete.txt`.

## Results

Against the existing clean vLLM ShareGPT baseline:

- c16: `16 completed / 0 errored / 0 bad_output`,
  `337.6359 tok/s`; ratio `337.6359 / 518.796 = 0.6508`.
- c32: `16 completed / 0 errored / 0 bad_output`,
  `340.1011 tok/s`; ratio `340.1011 / 524.128 = 0.6489`.

Against the current graph-disabled Ferrum same-dataset diagnostic:

- c16: `-2.2947 tok/s`, `-0.675%`.
- c32: `-0.4543 tok/s`, `-0.133%`.

## Interpretation

No new product `serve` correctness problem was found. Product
`--batched-graph` selects `legacy_batched_decode_graph`, but it does not improve
the current ShareGPT endpoint throughput and is slightly below graph-disabled
for this diagnostic. The W2 performance gap remains about 15 percentage points
below the 80% same-hardware mainstream baseline target.

The next high-return work should stop treating graph enablement as the primary
missing lever and focus on the model-step dominant path, especially dense MLP
`gate_up`, work reduction, and launch/graph integration where profiler evidence
shows most decode time is spent.
