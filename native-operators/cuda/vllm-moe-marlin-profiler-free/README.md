# MoE native profiler retirement

`ops.cu` is the pinned a9 MoE source with its process-global profile config,
setter/clearer, file writer, environment fallback and diagnostic counter removed.
The three mathematical entrypoint signatures, launch selection, error status,
stream and workspace contracts are unchanged. The existing launch-configuration
overrides are retained; this patch does not change numerical policy.

`../source-overlays/vllm-moe-marlin-profiler-free.json` pins the unmodified a9
bundle manifest, all nine baseline source members and this repository override.
The upstream revision and Apache-2.0 notice remain those in the source definition.
This is a reproducible repository source overlay, not a replacement release asset.

Use the ordinary native builder with a fresh external output directory:

```text
ferrum-native-ops-builder materialize-source \
  --spec <repo>/native-operators/cuda/source-overlays/vllm-moe-marlin-profiler-free.json \
  --base-source-root <unpacked-a9> --override-source-root <repo> --out <source>
ferrum-native-ops-builder lock-source \
  --definition <repo>/native-operators/cuda/source-definitions/vllm-moe-marlin.json \
  --source-root <source> --out <plan>
```

Then run the existing `source-build`, `materialize-package-spec`, `package` and
`assemble-set` steps with the actual CUDA target and existing object cache.
The plan binds the materialized bytes; receipts and final artifact-set lock must
be generated from the actual build. Other unchanged operator packages can be
reused with their original verified receipts and compatible provider catalog.

The MoE operator package ABI is **2**, while Ferrum's descriptor-v2 layout and
the three compute function signatures are retained. The resolver and build-unit
coverage reject ABI 1 and both retired profile exports. Actual archive symbols
are checked as well as manifest claims, so omitting a writer symbol from the
manifest cannot hide an old binary. Product Cargo builds require the new pinned
artifact set; they do not compile this override or download sources themselves.

Before updating product/CI artifact pins, run the existing FP8 and MXFP4 numerical
FFI tests and `native_moe_profile_retirement_ignores_environment_after_real_math`
with the new linked artifact on CUDA. The latter isolates the obsolete profile
environment in child processes and verifies that valid numerical execution
neither creates a native profile file nor changes one owned by another writer.
