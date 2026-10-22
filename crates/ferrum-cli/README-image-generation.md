# Experimental image entrypoints

The CUDA build exposes the existing in-process image pipeline through both
entrypoints. It uses one device runtime and one Host account, with the registered
architecture from `model_index.json`. The public model name is an alias only.

```text
ferrum run --image-config image.json --prompt "A red kite above the sea" --width 512 --height 512 --seed 42 --output kite.png
ferrum serve --image-config image.json --host 127.0.0.1 --port 8000
```

The current required profile is `experimental_serial_cold`. It recompiles and
reloads component weights for each request and executes one request at a time.
It qualifies functional integration only; it is not the final concurrent serving
architecture or evidence of throughput parity. `max_num_seqs` must be `1`.
Pending HTTP requests can queue, but the queue does not execute them as a batch.
The independent request field `n=1` selects one returned image per request.

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
