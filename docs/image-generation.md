# Image generation and editing

Ferrum supports Qwen-Image-2.1-Turbo on Linux CUDA and Apple Silicon Metal. Generation uses text; editing accepts one static 8-bit RGB/RGBA PNG without a mask. Editing can redraw background and details.

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

`POST /v1/images/generations` accepts a JSON text prompt. `POST /v1/images/edits` accepts multipart with one PNG. Both support `n=1` and return PNG in `b64_json`.

```sh
curl http://127.0.0.1:8000/v1/images/generations \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen-image","prompt":"A red fox in a snowy forest","size":"1024x1024"}'

curl http://127.0.0.1:8000/v1/images/edits \
  -F model=qwen-image -F image=@fox.png \
  -F 'prompt=Make the fox blue' -F size=1024x1024
```

For Metal, replace `qwen-image` with `qwen-image-turbo-metal`.

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
