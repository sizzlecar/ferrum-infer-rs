# KV cache precision

KV cache holds attention keys and values for tokens the model has already processed. Ferrum uses FP16 by default. On supported vNext GPU attention paths, `int8` stores those keys and values as signed 8-bit values with separate FP32 scales for each token and KV head.

```console
ferrum run MODEL --kv-dtype int8
ferrum serve MODEL --kv-dtype int8
```

The same setting is available in `ferrum.toml`:

```toml
[runtime]
kv_dtype = "int8"
```

An explicit CLI option takes precedence over runtime environment and configuration. Use `--kv-dtype fp16` to select the default precision again. An unsupported INT8 combination fails during planning; it does not silently use FP16.

## What changes

Attention KV takes approximately half the unpaged storage. Including scales, the ratio to FP16 is `1/2 + 2/head_dimension`: a head dimension of 128 saves 48.4375% of the KV payload and scales. Page rounding affects physical allocations, especially at short contexts.

This is a reduction in KV storage. Weights, recurrent state, activations, workspaces, and allocator overhead still consume memory. A model whose memory is dominated by weights or recurrent state can see a much smaller reduction in total memory. The saved capacity may permit longer contexts or more concurrent requests within the model's own limits and the configured memory budget.

INT8 quantization changes numerical values and adds quantization/dequantization work. It does not guarantee faster inference or identical generated text. Backend and model measurements are tracked in the [validation record](vnext-8bit-kv-validation.zh.md); unmeasured combinations have no quality or performance claim.

## Compatibility and evidence

- The vNext standard causal-attention operations have INT8 implementations for Metal and portable CUDA, including their F16 and F32-master activation profiles.
- CUDA `auto` attention chooses portable execution for INT8. Explicit `native-adaptive` attention currently requires FP16 KV and rejects INT8. Record this provider difference when comparing performance.
- Specialized attention operations that have not declared an INT8 profile, including the current Gemma 4 and GPT-OSS paths, reject the request. CPU vNext attention has no INT8 implementation.
- Legacy model loaders and non-language-model endpoints do not implement this KV format and reject an INT8 request. A supported GPU alone does not establish model support.
- Prefix/checkpoint reuse requires the same resolved profile and both the payload and scale states. FP16 and INT8 checkpoints cannot be interchanged. Whole-model checkpoint support still depends on every stateful operation in that model.

For an executing model, `/health` exposes `kv_storage.requested`, `kv_storage.selected`, and `kv_storage.numerical_profile`. `--effective-config-json PATH`, available on both `run` and `serve`, records the same resolved plan after initialization. The INT8 storage identifier is `int8_per_token_head_f32_scale_v1`; seeing an input option alone is not proof that the model is executing that format.

`kv_storage.logical_sequence_state` reports complete-model KV bytes per token and fixed recurrent state separately. It excludes allocator rounding, workspace and checkpoint copies. Actual pool residency is reported under `cache.prefix_cache.dynamic_pools.pools` even when prefix reuse is unavailable. The compatibility field `auto_config.admission.memory_estimate` remains a labeled legacy FP16 estimate; use the resolved state layout and pool measurements when assessing INT8 memory use.

KV precision does not remove conversation messages or summarize a prompt. Client context management still applies. It also does not add SSD KV offload or disk checkpoint loading.
