# Experimental image FP8 precision

The default image precision remains BF16. A build containing the locked
`cuda-cutlass-fp8` native package can explicitly select:

```text
ferrum run MODEL_DIR --image-precision fp8_linear_experimental_v1 --prompt "A teapot" --output image.png
ferrum serve MODEL_DIR --image-precision fp8_linear_experimental_v1 --max-num-seqs 8
```

Both commands use the same in-process composition. This changes text-encoder
Linear projections and DiT block Linear projections to E4M3FN weights and
dynamically quantized E4M3FN activations, with BF16 tensor boundaries. Each
text Linear has its own weight scale. DiT Q/K/V share one scale per block;
attention output and the three MLP matrices have individual scales. VAE,
embedding lookup, norms, and DiT entrance/time/modulation/final projections
remain BF16. The MLP activation is SiLU-mul; the text entrance uses GELU-tanh.

This is an explicit experimental precision scope. It does not certify the
deployed Omni ignored-layer matcher, fused GEMM numerical equality, full-model
quality, throughput, or every requested batch fitting GPU memory. The source
BF16 checkpoint headers remain checked; conversion uses exact BF16 mappings
and the same managed Host account as inference. The directory preset reserves
a 128 MiB Host WeightBacking class within its existing 1 GiB total ceiling;
group conversion and transfer staging overlap under that shared total.
The device precheck uses the selected physical execution weights, including
every required physical scale scalar, as a lower bound; plan alignment, activations, and workspace
still undergo normal admission. The preset bounds the native FP8 plan cache at
64 distinct matrix geometries and rejects additional shapes when it is full.

For diagnostic `--image-config`, set backend `precision_policy` to
`"fp8_linear_experimental_v1"` and provide a nonzero
`backend.host.weight_backing_bytes` sufficient for the largest conversion group.
Do not also pass `--image-precision`. Omission keeps `"bf16"`.

Precision never changes attention or residency implicitly. The directory
preset remains cold. A diagnostic configuration can separately select
`execution_profile: "experimental_resident_v1"` and
`backend.pipeline.residency: "all_model_components"` to retain TE/DiT/VAE
weights. Exact-plan reuse and compatible physical snapshot reuse remain subject
to ordinary admission and identity checks. HTTP requests use the configured
service precision; there is no per-request precision switch.

The CUTLASS module has a separate declared 128 MiB opaque allowance. This first
CUDART bridge cannot prove native unload, so that claim is conservatively
retained until process exit. Tensor payloads and Host conversion buffers have
their ordinary completion/retirement lifetimes. These allowances are not
measurements of native peaks.
