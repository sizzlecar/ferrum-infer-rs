# Performance evaluation

Compare output throughput only after checking the workload's declared latency and
completion requirements. Freeze hardware, model and precision, server capacity,
output policy, arrival schedule, metric definitions, thresholds, repetitions and
acceptable regression before measuring. Client concurrency may vary while server
capacity remains fixed. Use the same selected requests in each implementation and
concurrency cell; use llama.cpp on the same hardware/model as the first baseline
where that exact format is supported. Disclose unsupported artifacts and intentional
differences instead of silently substituting precision or output policies.

## Freeze a ShareGPT workload

`ferrum bench-serve --dataset sharegpt` selects first human/assistant pairs without
truncating them. Length filters use the specified tokenizer without special tokens;
the template reserve is an explicit estimate. By default the reference answer's
token length determines the output budget. `--sharegpt-fixed-output-tokens` changes
that policy explicitly. Natural stopping remains enabled unless `--ignore-eos` is
requested. Save the source and tokenizer checksums with the selection evidence.

For an independent holdout, exclude every known development or training pool with
one or more `--sharegpt-exclude-path FILE` arguments. Each file is a ShareGPT JSON
array or JSONL source. A matching original ID **or** first-human prompt removes the
candidate before length filtering and reservoir sampling. Prompt matching trims
and collapses Unicode whitespace to ASCII spaces, preserving case and other bytes;
the served prompt itself is unchanged. ID-only records are accepted when only an
ID is available, but cannot exclude a differently named copy of its prompt. Use
complete original records when possible. Malformed or unusable history records
cause an error. The exclusion sources and counts appear in the evidence and bind
the prepared cache, so changed history cannot reuse a previous selection.

Export a new pool before starting any server requests:

```sh
ferrum bench-serve \
  --base-url http://127.0.0.1:1 --model offline-export \
  --tokenizer /models/tokenizer-directory \
  --dataset sharegpt --sharegpt-path /data/parent.json \
  --sharegpt-source-sha256 PARENT_SHA256 \
  --sharegpt-exclude-path /data/development-pool.json \
  --export-sharegpt-pool /data/new-holdout \
  --sharegpt-selection-pool-size 384 --seed 20261021 --n-repeats 1 \
  --sharegpt-min-input-tokens 32 --sharegpt-max-input-tokens 1536 \
  --sharegpt-fixed-output-tokens 64 \
  --sharegpt-chat-template-reserve-tokens 128
```

The destination must be new. Export verifies the full parent checksum, preserves
selected source records, and records their parent indices, IDs, hashes, tokenizer,
filters and output policy in a receipt. It returns before constructing an HTTP
client. The numbers above illustrate a selection, not a recommended SLO or sample
size. Pin the exported pool and use its receipt to reproduce the same samples.
Selection cannot detect unlisted training data or semantically paraphrased prompts.
History loading is bounded to 256 sources, one million distinct IDs and prompts
each, and 64 MiB of ID text; source readers have separate file/record bounds.

## Measure the client-visible result

Report TTFT, TPOT and ITL P50/P99 and the target percentile for every run, with
output throughput, Peak VRAM / Memory Usage, errors, rejected and timed-out
requests, completion rate, sample counts and repetitions.

- TTFT ends at the first nonempty visible text update and includes server queueing.
  For scheduled open arrivals, include dispatch lag from the original arrival and
  distinguish it from HTTP-dispatch-only timing.
- TPOT uses the interval from first to last visible text divided by actual server
  usage output tokens minus one. Fewer than two usage tokens is undefined; report
  that count instead of substituting zero or removing those requests from completion.
- Serving ITL is the client-observed interval between successive nonempty visible
  SSE text events. Exclude role-only, empty and finish-only events; whitespace text
  still counts. Report transport coalescing and strict single-token timing
  eligibility separately, retaining visible-text stalls in the primary metric.
- Throughput uses successful output usage tokens, not event counts. Report missing
  usage and errors. Finite-drain and fixed-window throughput are different measures;
  low-arrival throughput includes idle time and does not establish peak capacity.

Keep every scheduled arrival in the completion denominator. Distinguish protocol
validity from semantic output checks. Internal execution timing, channel delivery
clocks and model risk estimates are diagnostics; none replaces client SSE timing.

## Interpret comparisons

Retain per-run percentiles and the raw observations needed to recompute them. An
average of repeated P95/P99 values cannot establish that every run meets its SLO.
Requests and events within one server run are correlated: account for runs and
workload episodes when reporting uncertainty instead of treating every event as
independent. State when there is insufficient precision to establish a margin.

Learning comparisons start from the declared model, residual and cache states.
Online updates may use only observations completed before the prediction. Choose
models and parameters using development data, then freeze a fresh holdout; reusing
holdout outputs to choose a winner turns them into development evidence. Report
collection, eligible learning and decision coverage with separate denominators,
plus actual interventions and planning/feedback overhead. Unknown or expired
predictions that fall back to ordinary scheduling are not evidence of SLO compliance.
