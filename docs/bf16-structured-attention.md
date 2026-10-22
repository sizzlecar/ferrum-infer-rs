# Managed structured BF16 attention

`operation.structured_attention.bf16` consumes contiguous BF16 Q, K and V and
writes an independent NoAlias BF16 output of the same `[B*S,H*D]` shape. It is
fixed tensor work, with no paged state, token batching or implicit model lookup.
The positive effective FP32 scale is explicit, represented by a canonical
rational attribute. No provider synthesizes a square-root scale.

The first version supports B from 1 to 16, positive S fitting u32, and
power-of-two D from 2 to 256. A checked layout supplies these attributes:

| Attribute | Meaning |
| --- | --- |
| `sequence_length`, `head_dim` | S and D |
| `scale` | Canonical rational evaluated in F64 then converted to F32 |
| `segment_lengths` | Up to 32 positive lengths covering S |
| `segment_bidirectional` | One 0/1 flag per segment |
| `key_run_counts` | Number of validity runs for each batch, up to 32 each |
| `key_run_lengths` | Batch-major flattened positive lengths; each batch covers S |
| `key_run_valid` | Batch-major flattened 0/1 flags, alternating within each batch |

Key k is visible to query q if it is valid and precedes `visible_end(q)`.
The end is q+1 for a causal segment and the segment's exclusive end for a
bidirectional segment. Earlier segments remain visible through causality.
Adjacent bidirectional segments retain separate boundaries. Padding excludes
keys; it does not suppress query outputs. An empty visible set produces +0
without reading Q/K/V. Masked keys and values are skipped before loading.

Geometry, byte extents, closed attributes and metadata bounds are validated
during planning and encoding. All attributes participate in plan identity.
Changing mask layout or padding requires a new plan in this version; this is
an explicit fixed-layout limitation. A future dynamic-mask operation must
preserve typed validation and resource accounting.

Both initial providers use FP32 products and a balanced adjacent-pair dot
reduction, followed by explicit scale multiplication. Keys are visited in
ascending order. The first visible key initializes maximum=score,
denominator=1 and accumulator=V. Later keys update an online softmax with
separate FP32 multiplication/addition; probabilities are not rounded to BF16.
The final FP32 reciprocal and accumulator multiplication precede BF16 RNE.
A nonfinite visible score makes the whole query/head output NaN. Visible V
values otherwise propagate IEEE arithmetic per channel. NaN payload/sign is
not specified. These exceptional paths do not return a structural preflight
error or claim that input data were validated on the host.

Reference uses libm 0.2.15 expf and a bounded 256-element stack accumulator.
CUDA uses precise libdevice expf, one CTA per query/head, exactly D threads,
one Q scalar and one accumulator per thread. Its mask is an initialized
2,312-byte by-value C ABI structure; together with pointer/scalar parameters,
the launch remains below 4 KiB. It declares 4*D dynamic shared bytes and nine
static 4-byte scalars. Compiler padding is additional: the pinned sm_120
offline assembler reports 48 static shared bytes, a zero-byte stack frame
and no spill loads/stores. This is not a measurement of the driver's JIT
allocation. There is no S*S score tensor or global scratch allocation.
Native compiler register spills still require inspecting the compiled result;
the source structure alone is not a physical VRAM qualification.

The CUDA factory `cuda_vnext_tensor_attention_providers` returns nine providers
sharing the existing native module/permit. The five-, seven- and eight-provider
factories retain their original API sets. The runtime admits all tensor storage
and reserves its experimental opaque native budgets before context/module work.

Logical arithmetic counts describe the nominal finite path, not measured
hardware FLOPs: per head, `(5*D+6)*P - (2*D+5)*N`, where P is the number of
visible query/key pairs and N is the number of nonempty queries. Mask work is
computed in O(batch*segments*runs), without expanding S*S or iterating S during
plan preparation. Comparisons, indexing and conversions are excluded.

Profiles are `reference.structured_attention.bf16.online_f32.libm-0.2.15.v1`
and `cuda.ieee.structured_attention.bf16.online_exp.v1`. General FP32 expf
qualification is pending; earlier BF16-input SiLU qualification does not apply
to arbitrary attention scores. Analytic and hardware probes do not establish
FlashAttention/Torch bitwise equality, full-image accuracy or throughput parity.
