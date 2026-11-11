# Image generation and editing

Ferrum supports Qwen-Image-2.1-Turbo on Linux CUDA and Apple Silicon Metal, and P-H3-Image on Linux CUDA SM89/SM120. On main/development, Qwen editing accepts up to ten ordered references; P-H3 accepts up to eight. **Released v0.19.0 still supports only one Qwen reference.** Each reference must be a static 8-bit RGB/RGBA PNG. Masks are unsupported. Editing can redraw background and details.

## P-H3 setup

Copy the [P-H3 CUDA configuration](../crates/ferrum-cli/examples/config/p-h3-image-cuda.json)
beside your `models/` directory. It expects P-H3 metadata and prompt-upsample
sources in `models/P-H3-Image/`, and the Comfy W4A8 text encoder/DiT and INT8 VAE
archives in `models/P-H3-Image-Comfy/`. Set the CUDA/cuDNN absolute library paths
to the installed libraries. The example uses the tested CUDA 13 cuDNN dependency
family alongside the CUDA release binary; these libraries are not bundled.

The example targets a 32 GB GPU with a 29 GiB managed device ceiling and 48 GB
system-memory limit. Its opaque module allowances are explicit reservations,
not certified measurements for every driver/library version. Smaller GPUs need
their own capacity configuration. P-H3 currently requires this typed JSON;
passing only a model-directory name does not select the P-H3 pipeline.

```sh
ferrum serve --image-config p-h3-image-cuda.json --served-model-name P-H3-Image --port 8000
ferrum run --image-config p-h3-image-cuda.json --prompt "A red teapot on a wooden table" --width 2720 --height 1536 --output teapot.png
```

Prompt expansion is enabled, MTP and reference-result reuse are off, and compatible
requests batch automatically within the configured capacity. The image sampler
retains the model's eight-step schedule. [RTX 5090 measurements](performance-rtx5090-p-h3-20261111.md)
describe the measured configuration and its limits.

## CUDA

Use a complete local [model directory](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo), a Ferrum CUDA release matching your GPU, and cuDNN 9. The FP8 profile quantizes the text encoder and DiT; vision and VAE stay BF16.

```sh
ferrum run /path/to/Qwen-Image-2.1-Turbo --backend cuda \
  --image-precision fp8_linear_experimental_v1 \
  --prompt "A red fox in a snowy forest" --width 1024 --height 1024 --output fox.png

ferrum run /path/to/Qwen-Image-2.1-Turbo --backend cuda --image fox.png \
  --image-precision fp8_linear_experimental_v1 \
  --prompt "Make the fox blue" --width 1024 --height 1024 --output edited.png

ferrum serve /path/to/Qwen-Image-2.1-Turbo --backend cuda \
  --image-precision fp8_linear_experimental_v1 \
  --served-model-name qwen-image --port 8000
```

The directory preset supports outputs up to 2048×2048, subject to capacity checks. Serving automatically batches compatible requests; `--max-num-seqs` is an optional upper bound, not a fixed batch size. Scheduling occurs between complete batches, so large requests can delay smaller ones. `/health` exposes image scheduling and resource limits.

For generators that support continuous denoising, `serving.batching: "auto"` also considers auxiliary model initialization. When every resident request has exactly one denoising step left, it completes that step before starting another auxiliary model. Already initialized requests can still join, and pending initialization and cancellation continue to advance. The generator must declare both the auxiliary model and the remaining steps; unknown progress keeps ordinary admission. `"configured"` preserves immediate admission at step boundaries. This scheduling policy applies to `serve`; a single `run` request has no competing request to defer.

## Metal

Validated at 1024×1024 on an M4 Mac mini with 16 GiB unified memory. Uses a Q4_K_M GGUF DiT, FP8 text encoder and BF16 vision/VAE.

1. Copy [the Metal configuration](../crates/ferrum-cli/examples/config/qwen-image-2.1-turbo-metal.json) to `image-metal.json` beside your `models/` directory.
2. Put the original model's text encoder, VAE and metadata in `models/Qwen-Image-2.1-Turbo/`. The original BF16 DiT weights are not needed.
3. Put [the Q4_K_M DiT](https://huggingface.co/m9ter1a/Qwen-Image-2.1-Turbo-GGUF/blob/a94df596a422bc8dfe406655154c0364f79ba042/qwen-image-2.1-turbo-Q4_K_M.gguf) in `models/Qwen-Image-2.1-Turbo-GGUF/`. Alternatively, edit the JSON paths; relative paths resolve from the configuration directory.

```sh
ferrum run --image-config image-metal.json \
  --prompt "A red ceramic mug on a white table" --width 1024 --height 1024 --output mug.png

ferrum run --image-config image-metal.json --image mug.png \
  --prompt "Make the mug blue" --width 1024 --height 1024 --output edited.png

ferrum serve --image-config image-metal.json --host 127.0.0.1 --port 8000
```

The API model name is `qwen-image-turbo-metal`. Set precision in the JSON; do not combine `--image-config` with a positional model or `--image-precision`.

<details>
<summary>If the model snapshot omits transformer/config.json</summary>

Save this Qwen-Image-2.1-Turbo architecture as `models/Qwen-Image-2.1-Turbo/transformer/config.json`. Keep an existing matching file.

```json
{
  "_class_name": "QwenImage21Transformer2DModel",
  "_diffusers_version": "0.41.0.dev0",
  "attention_head_dim": 128, "axes_dims_rope": [16, 56, 56],
  "causal_condition": true, "context_in_dim": 4096, "eps": 0.000001,
  "in_channels": 64, "mlp_ratio": 3, "num_attention_heads": 32,
  "num_layers": 32, "out_channels": 64, "patch_size": 1
}
```

</details>

## HTTP API

`POST /v1/images/generations` accepts a JSON text prompt. `POST /v1/images/edits` accepts multipart PNG inputs. Both support `n=1` and return PNG in `b64_json`.

```sh
curl http://127.0.0.1:8000/v1/images/generations \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen-image","prompt":"A red fox in a snowy forest","size":"1024x1024"}'

curl http://127.0.0.1:8000/v1/images/edits \
  -F model=qwen-image -F image=@fox.png \
  -F 'prompt=Make the fox blue' -F size=1024x1024
```

For Metal, replace `qwen-image` with `qwen-image-turbo-metal`.

On main/development, Qwen accepts up to ten images in one edit, with the same
reference order in `run` and `serve`. Repeat multipart `image[]` fields (`image`
is also accepted), or repeat CLI `--image`; refer to image 1, image 2, and so on
in the prompt. These Qwen multi-reference examples require a development build,
not the v0.19.0 release:

```sh
curl http://127.0.0.1:8000/v1/images/edits \
  -F model=qwen-image \
  -F 'image[]=@person.png' -F 'image[]=@pose.png' \
  -F 'prompt=Keep the person, face and clothing from image 1. Use the pose from image 2.' \
  -F size=1024x1024

ferrum run /path/to/Qwen-Image-2.1-Turbo --backend cuda \
  --image-precision fp8_linear_experimental_v1 \
  --image person.png --image pose.png \
  --prompt 'Keep the person from image 1 and adopt the pose from image 2.' \
  --width 1024 --height 1024 --output edited.png
```

P-H3 accepts up to eight ordered input images per edit. The first is Picture 1,
the image being edited; subsequent inputs are Picture 2 through Picture 8.
Repeat `image[]` in the OpenAI-compatible multipart request (`image` is also
accepted). State each image's purpose in the prompt, for example:

```sh
curl http://127.0.0.1:8000/v1/images/edits \
  -F model=P-H3-Image \
  -F 'image[]=@person.png' -F 'image[]=@pose.png' \
  -F 'prompt=Keep the face, hair and clothing of the person in Picture 1. Change their body pose to match Picture 2.' \
  -F size=1536x1024

ferrum run --image-config ph3.json \
  --image person.png --image pose.png \
  --prompt 'Keep the person from Picture 1 and adopt the pose from Picture 2.' \
  --width 1536 --height 1024 --output edited.png
```

The `/health` image capabilities report `maximum_reference_images` separately
from the concurrent edit request capacity; use the running server's declaration
when choosing how many images to send. Each file must be a static RGB/RGBA8 PNG within the encoded image
limit; all input backings, decoded pixels and visual outputs must also fit the
configured Host and device budgets. Multiple images increase those requirements.
The optional H3 main-vision cohort cache currently applies to single-image edits;
multi-image edits execute each Picture's visual encoder. Masks remain unsupported.

With P-H3 prompt expansion enabled, both `run` and `serve` accept complete edit
descriptions as well as structured briefs. Conditioning preserves briefs and
explicit Picture instructions; ordinary descriptions receive the model's
`Edit <Picture 1>. ` prefix. Missing EOS, truncated or cancelled generation,
malformed thinking delimiters and capacity violations still fail the request.

## P-H3 causal precision and bounded weight uploads

`backend.f32_attention_policy` selects the shared F32 attention route for prompt
vision, reference-image vision and the VAE decoder in both `run` and `serve`.
Its default `auto` uses the declared fused capability when compiled with
`cuda-cutlass-fused-f32-attention` and its locked native artifact; otherwise it
keeps the original sequential F32 profiles. `sequential` explicitly selects
the original profiles, while `cutlass_fused` requires the native capability.

The fused profiles retain F32 inputs and outputs and use CUTLASS FastF32 tensor
operations with a different softmax reduction. They do not promise bitwise
equivalence to the sequential implementation. Full bidirectional, all-valid
MHA with head dimensions 64 or 72 uses the fused kernel; other supported shapes
and masks use the declared strict-kernel fallback. The native module has a
separate 64 MiB allowance charged inside the existing device ceiling, with no
global payload workspace. This route is independent of the DiT
`backend.attention_policy`. A native-kernel qualification alone does not establish
image quality or end-to-end latency and throughput.

Both `run --image-config` and `serve --image-config` use the same P-H3
configuration. The optional `backend.prompt_expansion_causal.precision` selects
`f16` (the existing default) or `bf16` for the prompt upsampler's dense
weights, activations, convolution history and KV; DeltaNet recurrent state and
small decay/normalization parameters stay F32. BF16 is a separate model schema
and native operation profile, available for dense Qwen3.5 safetensors on CUDA
SM80 or newer. It requires `prompt_expansion.mtp: "off"`; ordinary single-token
`decode_replay` uses the same explicit graph-count and per-stream memory limits
as F16.
This selection does not change the H3 W4A8 denoiser/text encoder or INT8 VAE.

With `prompt_expansion.source_backing: "bounded_streaming"`, original BF16
weight ranges copy directly into the existing bounded Host slot. F16 execution
continues to convert those ranges. Affine normalization parameters retain their
explicit conversion recipe. This policy does not cache completed weights or
reference-image results across requests.

An optional `backend.pinned_weight_upload` object declares
`maximum_allocation_bytes` and `maximum_registered_bytes`, both positive
physical byte counts including Host alignment padding. For a 64 MiB streaming
slot on a host with 4 KiB pages, the following allowance includes that padding:

```json
"pinned_weight_upload": {
  "maximum_allocation_bytes": 67112960,
  "maximum_registered_bytes": 268435456
}
```

CUDA registers eligible managed static-weight allocations in place; registration
survives slot reuse and retires before the original allocation. The total must
cover concurrently retained eligible owners, including fallback components.
Owners above the declared allocation limit and immutable archive mappings keep
their ordinary upload path. Eligible registration or capacity failures are
errors. Omission retains ordinary uploads. The original Host limits still apply,
with native page alignment accounted before allocation. These settings do not
by themselves establish output quality, latency or concurrency performance.

## CUDA LoRA

`run` and `serve` support one ordinary DiT adapter with a fixed strength at startup. A PEFT directory needs `adapter_config.json` and `adapter_model.safetensors`:

```sh
ferrum serve /path/to/Qwen-Image-2.1-Turbo --backend cuda \
  --image-precision fp8_linear_experimental_v1 \
  --image-lora /path/to/peft-directory --image-lora-strength 0.8 \
  --served-model-name qwen-style --port 8000
```

For a single Diffusers A/B `.safetensors` file, set `--image-lora` to that file and declare `--image-lora-base-model Qwen/Qwen-Image-2.1`. Single files use unit scaling unless `--image-lora-alpha` supplies alpha (applied as alpha/r); PEFT reads rank and alpha from its configuration. The base-model flag can fill missing metadata but cannot override a conflicting declaration.

Per-request switching, multiple adapters, Metal/GGUF LoRA and special distillation schedules are unsupported. Turbo keeps its eight-step schedule; adapter compatibility does not establish image quality. Explicit image JSON uses a top-level `lora` object with `source` and `strength`, and must budget the adapter and composed weights in its Host limits.
