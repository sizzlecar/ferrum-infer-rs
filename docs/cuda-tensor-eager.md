# Experimental managed CUDA tensors

`CudaDeviceRuntime::new_tensor_eager(CudaTensorRuntimeConfig)` is the first
bounded CUDA execution path for fixed tensor components. Its providers implement
BF16 SiLU, SiLU-mul, tanh, modulation scale, gated residual, LayerNorm,
head-wise RMSNorm, and BF16 Dense with FP32 accumulation. Full image generation
is still in development.

The path uses the ordinary program compiler, `PlanRuntimeResources`, and
`PreparedTensorExecutor`. SiLU uses `ModelWeights::None`; Dense can bind stored
weights uploaded with `initialize_static_managed`. It carries tensor logical
work through dispatch without token counts, paged KV, or continuous batching.
The ordinary CUDA constructor does not advertise managed tensor IO.

The typed configuration declares usable device capacity, bootstrap and per-stream
opaque allowances, stream and in-flight limits, and an opaque-memory profile
identity. The physical device ID is `device.cuda.{ordinal}`. Device queries occur
before context creation; the existing process capacity ledger reserves bootstrap
memory before native construction. Stream creation reserves a lane and memory
before creating its CUDA stream and cuBLAS handle. Tensor allocations request
their exact charged bytes and accept alignments up to 256 bytes.

These opaque allowances are caller-declared experimental budgets. They are not
measurements of CUDA context, module, cuBLAS, or driver staging consumption.
Successful admission establishes accounting under the declared budgets; actual
memory qualification needs device measurements. External headroom remains
separate and must not charge the same allowance twice.

One native module provider bundle may be live per runtime. Its lifetime holds the
bootstrap authority; encoded commands retain the same owner. Commands, streams,
fences and quarantine retain allocations and transfer owners until retirement.
Unknown native creation or abandoned in-flight work conservatively retains
reservations. This can require process restart after an unrecoverable error.
The pointwise factory returns five providers sharing one module owner.
SiLU-mul rounds SiLU to BF16 before multiplication;
modulation rounds `1 + scale`, and gated residual rounds `gate * update`, before
their final operation. Each operation has its own CUDA numerical profile.
The `cuda_vnext_tensor_bf16_providers` factory also returns LayerNorm
and head-wise RMSNorm. The `cuda_vnext_tensor_component_providers` factory adds
[adjacent BF16 RoPE with explicit F32 frequencies](bf16-adjacent-rope.md).
All eight functions share that same module and permit; the older factories
still return their original five or seven providers.

CUDA normalization currently accepts power-of-two widths up to 4096. Both
providers check this during planning and encoding. They use a bounded
adjacent-pair FP32 reduction, explicit square root and reciprocal division, and
at most 16 KiB of dynamic CTA shared memory plus scalar shared storage. They
allocate no global scratch. LayerNorm computes population variance in two
passes without affine parameters. Head RMSNorm preserves its BF16 normalized
boundary before multiplying the shared gamma vector; see
[the operation contract](bf16-head-rms-norm.md). Their numerical profiles are
independent of deployment kernels using `rsqrt` or another reduction order.

Managed upload retains the original `ManagedHostBytes` through submission. It
does not copy payloads into the token upload packet or a cached executable.
Accounted host storage is pageable: CUDA may stage or synchronize H2D internally.
Managed readback synchronizes and copies directly into reserved output backing.
Stored weight uploads reserve `TransferStaging` bytes before copying each chunk.
Initialization retains the chunk until completion and flushes pending uploads
when they hold the staging quota needed by the next chunk. This entrypoint does
not silently fall back to raw uploads or native weight import; source mmap and
materializer storage remain outside its staging allowance.

The initial path accepts eager submissions with timing disabled and a bounded
number of commands. It does not support graph capture/replay, token commands,
or terminal token readback. These restrictions belong to this experimental
runtime profile, not to the shared `DeviceRuntime` interface.

The independent numerical profile is
`cuda.libdevice.precise.silu.bf16.v1`: FP32 `expf`, separate round-to-nearest
addition/division, then integer BF16 round-to-nearest-even. Its source is compiled
without fast math and with FTZ disabled. It does not inherit the Reference libm
profile or claim PyTorch/vLLM-Omni equality. Dense has a separate profile,
`cuda.cublas.gemmex.bf16_f32.no_workspace.v1`, and records the loaded cuBLAS version
in its provider fingerprint. It disables reduced-precision reductions and atomics,
and sets a null, zero-byte workspace after selecting the stream. Unsupported
zero-workspace execution fails without relaxing the resource policy. Native handle
and library overhead still require the declared opaque allowances.

On a configured CUDA host with the native operator lock and an exclusive GPU
window, run the actual managed execution smoke test:

```console
cargo test --locked -p ferrum-models --features cuda \
  --test vnext_tensor_cuda_runtime \
  managed_cuda_silu_upload_execute_readback_and_retirement -- --ignored --nocapture
```

The test uses selected independently known BF16 cases, exercises lane retirement
and slot reuse (with new native streams), rejects duplicate provider installation,
and retains output after closing the plan. It is not full-domain libdevice qualification, an image quality
comparison, or a performance benchmark. Reuse the host's Cargo target cache and
set `CARGO_INCREMENTAL=0`; do not create another target for this test.

The existing `verify_qwen_image21_projection` example also accepts
`--backend cuda` when built with the CUDA feature. It requires explicit
`--cuda-usable-memory-bytes`, `--cuda-bootstrap-memory-bytes`,
`--cuda-stream-memory-bytes`, and `--cuda-opaque-memory-profile` values.
`--cuda-transfer-staging-bytes` defaults to 65536. It loads the pinned BF16
`img_in.weight` fragment through the real partial family and checks analytic
probes and dense inputs against an independent CPU F64 mathematical oracle.
Artifacts record the configured budgets, provider identity, full output tensors,
and comparisons. This is a partial projection check, not a full DiT block or
deployed FP8/PyTorch equivalence. Reference remains the example's default backend.

`verify_qwen_image21_mlp` accepts the same backend and budget options, plus
`--config`, `--model-dir`, and `--artifact-dir`. Its optional `--index-file`
defaults to `diffusion_pytorch_model.safetensors.index.json`; the weight archive
selects this index explicitly without falling back to another file. It maps the
existing shards and uploads only block 0's three MLP weights. Use an explicit
staging budget such as `--cuda-transfer-staging-bytes 16777216` to avoid thousands
of small synchronous upload chunks for those 288 MiB of BF16 weights.

The MLP example takes an already normalized/modulated basis input. It checks
every gate, up, and SiLU-mul channel and a predeclared subset of final output
coordinates against expectations prepared from weights before execution. It
preserves all four output tensors in external artifacts. This is a partial MLP
check, not attention, a complete DiT block, image generation, or throughput
qualification. Source mmap and oracle allocations are outside the managed host
staging account.
