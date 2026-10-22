# Experimental image entrypoints

The CUDA build exposes the existing in-process image pipeline through both
entrypoints. It uses one device runtime and one Host account, with the registered
architecture from `model_index.json`. The public model name is an alias only.

For a complete local pipeline installation, use its directory directly:

```text
ferrum run /models/image-pipeline --prompt "A red kite above the sea" --seed 42 --output kite.png
ferrum serve /models/image-pipeline --max-num-seqs 8 --gpu-memory-utilization 0.9 --host 127.0.0.1 --port 8000
```

Both commands recognize the registered pipeline declaration and validate its
component metadata before creating CUDA. A directory name does not select an
architecture. A normal text-model directory without `model_index.json` retains
the existing text route. An unknown or incomplete pipeline is an error, not a
text fallback. In particular, a missing `transformer/config.json` must be
supplied by a complete model installation; Ferrum does not reconstruct it from
weight shapes or invent it. A successful run using separately supplied metadata
through `--image-config` does not establish that the ordinary directory route
works for an incomplete installation.

The directory route selects the named **`experimental_bf16_cold_v1`** preset.
It uses BF16 components and reloads weights between stages and request batches;
it does not provide cross-batch residency. Its explicit defaults and bounds are:

| Control | Preset behavior |
| --- | --- |
| Dimensions | 512×512 by default, at most 1024 per axis; model divisibility is checked without rounding |
| `--max-num-seqs` | Defaults to 1; larger values enable real compatible TE/DiT batches up to the supported attention capacity |
| Batch wait | 0 ms for a ceiling of 1, 200 ms for larger batch ceilings |
| VAE | One B1 decoder session reused for the images in each batch |
| User tokens | At most 1024 per prompt |
| Host tensor account | 1 GiB total, with 768 MiB input, 256 MiB output and 16 MiB transfer-staging class ceilings |
| CUDA declarations | 1 GiB bootstrap, four streams at 64 MiB each, 64 MiB cuDNN opaque allowance, 16 MiB convolution workspace limit |

These are bounded configuration choices, not measured optimal settings or proof
that every resolution/batch combination fits. All plans still pass actual
capacity admission. `run` accepts one prompt and writes one image even when a
larger batch ceiling is allowed; concurrent compatible HTTP requests use the
real batching path. Cold execution and these limits do not establish G3
readiness, latency-SLO compliance or vLLM-Omni throughput parity.

`--gpu-memory-utilization` must be finite and in `(0, 1]` (default 0.9). The
automatic factory creates the sole managed CUDA context and then selects
`min(observed_free_bytes, floor(total_bytes * utilization))` as a conservative
**total** ceiling. Bootstrap, streams, library claims and tensor allocations all
share it; declared opaque bytes are not added back to observed free memory.
`--runtime-memory-budget-bytes`, when supplied, only lowers this ceiling.
It is not an independent plan budget. Free memory may change after observation;
the selected ceiling is not a promise to reach a particular measured GPU usage.
The composition receipt separates this observed snapshot from declared profiles.

The directory route supports `--backend auto` or `cuda`, with one optional CUDA
ordinal in `--gpu-devices`. cuDNN 9 discovery checks `LD_LIBRARY_PATH`, bounded
system `ldconfig -p` output on Linux, then standard Linux/CUDA library directories.
To override discovery, pass `--image-cudnn-library /absolute/path/libcudnn.so.9`.
Discovery only resolves files; the native library loads after its opaque
reservation. The canonical path is recorded in the composition receipt and
the existing provider records its library fingerprint. There is no search in a
model-specific virtual environment. Missing cuDNN produces an installation or
explicit-path error. The cuDNN deployment and its sibling libraries must remain
trusted and immutable while running.

For diagnostics or nonstandard metadata/archive locations, the existing explicit
configuration route remains available:

```text
ferrum run --image-config image.json --prompt "A red kite above the sea" --width 512 --height 512 --seed 42 --output kite.png
ferrum serve --image-config image.json --host 127.0.0.1 --port 8000
```

Choose one explicit experimental profile in that JSON:

- `experimental_serial_cold` retains one request per execution. Its pipeline
  `maximum_batch_size` and serving `max_num_seqs` must both be `1`.
- `experimental_batched_cold` enables compatible requests to share one real
  TE/DiT forward per stage and denoise step. Set pipeline
  `maximum_batch_size` explicitly and keep serving `max_num_seqs` at or below it.
  The implementation ceiling is `STRUCTURED_ATTENTION_BF16_MAX_BATCH` (currently
  16). Larger or mixed-size batches are rejected, not serialized into fake batches.
- `experimental_resident_v1` explicitly enables cross-batch residency with
  `backend.pipeline.residency` set to `denoiser_and_decoder` or
  `all_model_components`. The latter also retains text-encoder weights.
  Residency is independent of precision and decoder batching; it does not
  change the default cold preset. See the [runtime and retirement contract](../../docs/image-generation-runtime.md)
  for exact-plan reuse, compatible weight snapshots, admission and shutdown.

The pipeline maximum defaults to `1` when omitted. The service collects compatible
FIFO requests up to `max_num_seqs`, bounded by `request_batch_max_wait_ms`.
The configured `max_num_seqs * maximum_prompt_bytes` must fit the T2I tokenizer's
1 MiB combined raw-prompt bound, so individually admitted requests can compose.
The independent request field `n=1` still selects one returned image per request;
it does not restrict the number of requests in a batch.

`backend.pipeline.decoder_batch_policy` independently selects VAE execution:

- `single_image` (default) compiles a B1 decoder and reuses one session for each
  image in a request batch. Cold execution initializes that session per batch;
  an explicit resident policy can reuse it across batches. Inputs are exact
  NCHW views of the same accounted Host backing. Each output retains its own
  accounted single-image allocation. TE and DiT still execute the admitted B.
- `full_batch` explicitly selects the experimental diagnostic B-wide decoder.
  Its output slices retain the entire B-wide allocation until the last owner
  drops. This may require much more device activation capacity.

Receipts record the selected policy, each stage's compiled `batch_size`, and
`completed_executions`: decoding B requests uses B executions under `single_image`
and one under `full_batch`. Cold execution closes its decoder session before
returning images. Resident execution reports `retirement.kind="resident"` and retains
the session after trimming idle dynamic storage; explicit shutdown retires it.
Sample-independent mathematics does not establish bitwise equality:
backend convolution algorithms can change with batch geometry. Neither policy
promises a fixed VRAM peak or fit without actual plan admission and measurement;
Host input/output storage still grows with request count.

In explicit JSON, `backend.usable_memory_bytes` is the single product device-memory ceiling.
The internal plan admission limit is derived from it automatically for both
`run` and `serve`; there is no second plan budget to fill in. The compiler still
calculates each stage's resource requirements and admission checks those
requirements against the same capacity account. Bootstrap, stream and cuDNN
opaque reservations are charged there alongside tensor storage, without another
subtraction during configuration resolution. Choose this one ceiling with
measured free-device headroom. Older configurations containing
`pipeline.device_capacity_bytes` are rejected: remove that field and retain the
intended total ceiling in `backend.usable_memory_bytes`.

The cold profiles reload component weights for each batch. Compiled-plan caches
can avoid recompilation when their complete typed keys match; plan caching alone
does not retain device weights. All profiles remain experimental: selecting one
does not establish G3 readiness, throughput parity or capacity for every shape.
Actual Host/device admission still decides whether a requested batch fits.

The HTTP path is `POST /v1/images/generations` with `model`, `prompt`, optional
`size` (`WIDTHxHEIGHT`), `seed`, `n=1` and `response_format="b64_json"`.
Unknown fields and unsupported controls are rejected. Text generation flags such
as `--temperature`, `--max-tokens` and text compatibility controls are rejected
for either image route. `--image-config` also rejects simultaneous MODEL,
backend/budget flags and `--image-cudnn-library`: change its explicit JSON instead.
Image defaults for `run` are 512×512 pixels and seed zero. HTTP defaults come from
the named preset or the explicit `serving` configuration.
Only text-to-image generation is exposed. Image editing, image-to-image inputs
and `/v1/images/edits` are not supported by this product path.

`run` creates its PNG destination exclusively, preserving RGBA and alpha. It
emits a JSON receipt to stdout after the PNG is finished and synced. A failed
write may leave a partial destination; it is never reported as a completed PNG.
The output encoder streams rows with bounded codec storage and conversion
scratch. Host accounting covers managed tensor buffers, staging and conversion
scratch, not metadata, mmap page cache, codec internals or total process RSS.

The following example gives explicit experimental allowances, not measured
capacity guarantees. Adjust them for the actual device and workload. Every path
is relative to the JSON file's directory unless absolute. Weight files and index
names are exact: a missing source does not fall back to another checkpoint.
Metadata may live separately from the checkpoint directories.

```json
{
  "model": "local-image",
  "execution_profile": "experimental_serial_cold",
  "metadata": {
    "model_index": "metadata/model_index.json",
    "text_encoder": "metadata/text_encoder/config.json",
    "transformer": "metadata/transformer/config.json",
    "vae": "metadata/vae/config.json",
    "scheduler": "metadata/scheduler/scheduler_config.json",
    "tokenizer": "metadata/tokenizer/tokenizer.json",
    "tokenizer_config": "metadata/tokenizer/tokenizer_config.json",
    "chat_template": "metadata/tokenizer/chat_template.jinja"
  },
  "weights": {
    "text_encoder": {"kind":"index", "directory":"weights/text_encoder", "file":"model.safetensors.index.json"},
    "transformer": {"kind":"index", "directory":"weights/transformer", "file":"diffusion_pytorch_model.safetensors.index.json"},
    "vae": {"kind":"single", "directory":"weights/vae", "file":"diffusion_pytorch_model.safetensors"}
  },
  "backend": {
    "kind": "cuda",
    "ordinal": 0,
    "usable_memory_bytes": 32212254720,
    "bootstrap_memory_bytes": 1073741824,
    "stream_memory_bytes": 67108864,
    "maximum_streams": 4,
    "maximum_in_flight_per_stream": 1,
    "maximum_commands_per_submission": 2048,
    "opaque_memory_profile": "experimental-unqualified-runtime-v1",
    "cudnn": {
      "library_path": "/absolute/path/to/libcudnn.so.9",
      "opaque_memory_bytes": 67108864,
      "workspace_limit_bytes": 16777216,
      "opaque_memory_profile": "experimental-unqualified-cudnn-v1"
    },
    "host": {
      "total_bytes": 536870912,
      "input_preprocess_bytes": 402653184,
      "transfer_staging_bytes": 16777216,
      "output_bytes": 134217728,
      "weight_backing_bytes": 0
    },
    "pipeline": {
      "maximum_user_tokens": 1024,
      "maximum_width": 512,
      "maximum_height": 512,
      "maximum_batch_size": 1,
      "decoder_batch_policy": "single_image",
      "maximum_staging_bytes": 16777216,
      "maximum_initialization_commands": 64,
      "maximum_backing_maintenance_passes": 8
    }
  },
  "serving": {
    "maximum_pending": 4,
    "maximum_prompt_bytes": 65536,
    "max_num_seqs": 1,
    "request_batch_max_wait_ms": 0,
    "default_width": 512,
    "default_height": 512,
    "default_seed": 0,
    "rgba_scratch_bytes": 16384,
    "output": {
      "max_sessions": 4,
      "max_event_bytes": 16777216,
      "encoded_bytes_total": 67108864
    }
  }
}
```

`serving` is optional for `run` and required for `serve`. Output configuration
fields omitted inside `serving.output` use the shared typed `OutputConfig`
defaults. The account must cover the source RGBA, conversion buffers and encoded
HTTP body while they coexist. Dimension divisibility and real plan admission
remain enforced; dimensions are never rounded to fit a model.

For an explicit B2 functional experiment, change `execution_profile` to
`experimental_batched_cold`, `backend.pipeline.maximum_batch_size` to `2`,
`serving.max_num_seqs` to `2`, and choose a finite
`serving.request_batch_max_wait_ms` (for example 25). Keep two simultaneous
requests at the same dimensions with independently selected seeds. This enables
the batched path but does not guarantee every arrival is batched, that existing
memory limits fit it, or that a latency/throughput objective is met. `run` remains
a one-request invocation even when its configuration permits a larger batch.
Keep `decoder_batch_policy: "single_image"` for the bounded decoder experiment;
select `"full_batch"` explicitly when measuring the B-wide decoder diagnostic.
