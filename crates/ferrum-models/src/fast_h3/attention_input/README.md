# FastH3 attention inputs

`FastH3AttentionInputSource::prepare` binds one checked VSA layout and one
transformer block to a stateless Stored graph. Inputs are modulated BF16 hidden
rows `[B, packed_rows, H]` and explicit BF16 cosine/sine tables
`[B, packed_rows, rotary_dim]`. The position producer remains responsible for
the table's coordinate and trigonometric profile.

The fixed `minimax_h3_vsa_fp4.py` path gathers hidden rows before sharing one
unit-scale quantization across Q/K/V. It computes `to_gate_compress` on the
original hidden rows separately. This graph preserves those two quantization
invocations. It does not compute gate on tiled rows and assume equivalence.
The original F32 checkpoint alpha remains authoritative. `AllFour` remains
available as the projection component's diagnostic/default group.

The graph gathers hidden/cosine/sine through one shared I32 tile map, copies
between explicit batched and matrix ranks, projects Q/K/V and gate, applies
the single-round F32-intermediate QK/RoPE profile, and pools unsmoothed Q/K and
V through one shared positive-prefix count vector. It returns tiled BF16
Q/K/V `[B, padded_rows, heads*head_dim]`, packed BF16 gate
`[B, packed_rows, heads*head_dim]`, and three F32 means
`[B, heads, logical_tiles, head_dim]`. There is no Host readback between nodes.
This is not the sparse-attention operation: smoothing, region selection,
compression, attention and output projection are separate consumers.

One projection source owns four derived F32 alpha scalars and one unit scalar
(20 Host bytes). All three gathers share one original map allocation; all
three means share one count allocation. Identical schema interpretations
converge on one logical weight/value in `StoredComposition`, preserving Core's
single physical-layout owner rule. Neither config nor source retains duplicate
layout maps, and no private allocator or cache is introduced.

CPU tests validate the real package source, full graph contracts, routing,
physical/logical sharing, retained Host ownership and rollback. The Reference
backend deliberately cannot compile the NVFP4 graph without its provider; these
tests do not claim full CPU execution or GPU qualification. Pooling explicitly
uses ascending F32 reduction; it does not claim Torch parallel-sum bit equality.
