# Experimental image entrypoints

The CUDA build exposes the existing in-process image pipeline through both
entrypoints. It uses one device runtime and one Host account, with the registered
architecture from `model_index.json`. The public model name is an alias only.

```text
ferrum run --image-config image.json --prompt "A red kite above the sea" --width 512 --height 512 --seed 42 --output kite.png
ferrum serve --image-config image.json --host 127.0.0.1 --port 8000
```

Choose one explicit experimental profile:

- `experimental_serial_cold` retains one request per execution. Its pipeline
  `maximum_batch_size` and serving `max_num_seqs` must both be `1`.
- `experimental_batched_cold` enables compatible requests to share one real
  TE/DiT/VAE forward per stage and denoise step. Set pipeline
  `maximum_batch_size` explicitly and keep serving `max_num_seqs` at or below it.
  The implementation ceiling is `STRUCTURED_ATTENTION_BF16_MAX_BATCH` (currently
  16). Larger or mixed-size batches are rejected, not serialized into fake batches.

The pipeline maximum defaults to `1` when omitted. The service collects compatible
FIFO requests up to `max_num_seqs`, bounded by `request_batch_max_wait_ms`.
The configured `max_num_seqs * maximum_prompt_bytes` must fit the T2I tokenizer's
1 MiB combined raw-prompt bound, so individually admitted requests can compose.
The independent request field `n=1` still selects one returned image per request;
it does not restrict the number of requests in a batch.

Both profiles recompile and reload component weights for each batch. The batched
profile is experimental and unqualified until actual GPU tests establish its
functional scope; neither profile establishes G3 readiness or throughput parity.
Actual Host/device admission still decides whether a requested batch fits. There
is no cross-batch component residency in this implementation.

The HTTP path is `POST /v1/images/generations` with `model`, `prompt`, optional
`size` (`WIDTHxHEIGHT`), `seed`, `n=1` and `response_format="b64_json"`.
Unknown fields and unsupported controls are rejected. Text flags such as
`--temperature`, `--max-tokens`, `--backend`, `--model` and compatibility controls
are rejected when supplied with `--image-config`; without it, text behavior is
unchanged. Image defaults for `run` are 512×512 pixels and seed zero. HTTP defaults
come from the explicit `serving` configuration.

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
      "device_capacity_bytes": 25769803776,
      "maximum_user_tokens": 1024,
      "maximum_width": 512,
      "maximum_height": 512,
      "maximum_batch_size": 1,
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
