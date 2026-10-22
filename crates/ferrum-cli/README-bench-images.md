# Image HTTP benchmarks

`ferrum bench-images` sends a fixed, ordered dataset to image-generation HTTP
endpoints. It does not load models, change server capacity, retry requests, or
require a CUDA-enabled client. Each request asks for one `b64_json` image.

Create a JSON dataset outside the repository:

```json
{
  "schema_version": 1,
  "samples": [
    {"id": "kite", "prompt": "A red kite over a green field", "seed": 17, "width": 512, "height": 512},
    {"id": "boat", "prompt": "A wooden boat on a quiet lake", "seed": 29, "width": 512, "height": 512}
  ]
}
```

Use at least eight samples to sustain eight concurrent requests. Every measured
repetition and warmup repetition submits the entire dataset. Optional sample
`num_inference_steps` is sent only when supplied; otherwise record the server's
fixed step count in its configuration declaration. A server rejecting that
optional field produces a recorded protocol failure.

```text
ferrum bench-images \
  --endpoint http://127.0.0.1:8000/v1/images/generations \
  --endpoint http://127.0.0.1:8091/v1/images/generations \
  --model Qwen-Image-2.1-Turbo \
  --server-config "Ferrum: exact command/version, hardware, precision and fixed server capacity" \
  --server-config "Omni: exact command/version, hardware, precision and fixed server capacity" \
  --dataset /path/to/images.json --output /path/to/new-report.json \
  --concurrency 8 --repetitions 3 --warmup-repetitions 1 \
  --initial-server-state unknown
```

Endpoints run sequentially. The two servers must have comparable hardware,
configuration and isolation; the client cannot enforce that. To avoid overlapping
GPU residency, run the same dataset/configuration in separate invocations with
one endpoint each and compare the recorded dataset hashes. One model alias can
be shared, or repeat `--model` once per endpoint in endpoint order. Each endpoint
requires its own `--server-config` declaration. The client uses direct connections
without environment proxy settings and rejects redirects.

The output path is reserved with `create_new` before requests start. Completed
runs write all errors and request records; any request failure returns a failing
CLI exit status after saving the report. Ctrl-C or cancellation does not persist
partial request records: an interrupted run may leave an empty or incomplete
file, which is not a valid complete benchmark report and is never silently
overwritten. Cancellation recovery is not supported by this version.

The report includes per-request HTTP E2E latency, validation time, response size,
decoded RGBA8 hash and PNG file hash, plus per-repetition and pooled success/error
counts, E2E P50/P99 and validated images/second. PNG bytes/base64 are not written
to the report or logs. The dataset itself is included for reproducibility.

To retain the validated images for inspection, provide both
`--capture-dir /path/to/new-directory` and
`--maximum-capture-bytes BYTES`, with a nonzero aggregate encoded-PNG byte limit.
The directory must not exist. Each image, including warmup images, uses an
exclusive file with numeric endpoint/phase/repetition/sample indices; prompts
and sample IDs never become path components. The client writes the validated
PNG buffer directly, without a second PNG copy or an unbounded writer queue.
The report links capture records to sample IDs and records independent capture
failures. Incomplete capture makes the command exit nonzero after the report is
saved, while retaining the HTTP/validation result; a failed write may leave a
partial file and is never reported as a complete capture.

The primary throughput denominator includes client PNG validation and all failed
requests, plus local capture IO when enabled. Request E2E ends at complete HTTP
body reception and excludes both validation and capture. With capture enabled,
validation time includes blocking-worker queue time through decode completion,
but excludes saving; capture records contain separate save timing. Without
capture, the original validation cutoff after worker join is retained.
Capture can delay the next paced request, so compare capture-enabled and
capture-disabled throughput only with that difference disclosed.
A separate first-send to last-body-reception span is recorded; it can
overlap validation of earlier responses and is not server-only throughput.
Latency percentiles include successful, validated images only; failures remain
explicit. Client admission follows dataset order, while concurrent network/server
arrival order is not guaranteed. The initial cold/warm state is a declaration,
not a client reset; later repetitions may naturally warm the server.

Response bytes, PNG bytes, dimensions, pixels, decoder workspace, dataset bytes,
request count and HTTP duration have explicit limits shown by `--help`. Decode
jobs share the HTTP concurrency bound. These payload/codec bounds are not a total
process RSS measurement. Validation checks PNG framing/checksums, complete decode
and exact dimensions; supported color forms expand to RGBA8 without color-profile
or gamma conversion. RGBA8 integers are finite by definition, which cannot prove
that pre-encoding model floats were finite or that image quality is correct.

`--latency-slo-p99-ms MS` declares an optional threshold before measurement. It
requires zero errors and P99 within the limit; without it, latency SLO status is
`not_evaluated`. Quality, throughput parity and G3 readiness are always separate
evaluations. GPU/host peak memory is not sampled by this client and remains null;
collect matching external evidence before making a full performance claim.
