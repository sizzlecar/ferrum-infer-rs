# Indexed addressed paged attention

These two modified sources extend the pinned vLLM source bundle with four
additive `addressed_indexed_v1` entrypoints. Existing entrypoints and their
per-row arithmetic are retained. The upstream revision and Apache-2.0 license
remain those in the source definition; the header retains its copyright notice.

`../source-overlays/vllm-paged-attention-indexed.json` pins the unchanged a9
bundle manifest, all twelve baseline members, and these two repository
overrides. It is the reproducible source recipe, not a new release asset.

Materialize into a new directory outside both inputs using the ordinary native
builder:

```text
ferrum-native-ops-builder materialize-source \
  --spec <repo>/native-operators/cuda/source-overlays/vllm-paged-attention-indexed.json \
  --base-source-root <unpacked-a9> --override-source-root <repo> --out <source>
```

Then use the existing `lock-source`, `source-build` and `package` commands with
`<source>` and the paged-attention source/package definitions. The source
revision pins this recipe; the plan pins the actual materialized source bytes.
No source is fetched by materialization. The original a9 bundle is unchanged.

The device `i32 row_map` maps group ordinal to an original query/output,
address-table and length row. The native entrypoints receive both group size
and original row count and reject out-of-range mapped rows before accessing
them. V2 partition scratch is compact in group order; reduce writes the
original output row. The caller must provide a unique, immutable map, retain
every addressed resource, and select V1/V2 according to each row's existing
numerical contract. This address mapping grants no resource authority.
