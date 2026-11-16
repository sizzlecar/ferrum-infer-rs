# Logical-tile pooling

`FastH3VsaPoolConfig` consumes row-major BF16 `[B,T*128,H*D]`, directly matching
the checked `FastH3VsaRowTransport::Tile` result. It publishes contiguous F32
`[B,H,T,D]`. Counts come from the immutable layout's positive valid prefixes;
native pair-padding tiles are excluded. The same ordinary Stored I32 vector
can serve Q, K and V, all batches and heads. It is not a Q/K smoothing step.

The source admits exactly `T*4` bytes through the caller's Host WeightBacking
account. Its component identity hashes the recipe, tile width, count extent
and little-endian I32 count values. Configuration stores that digest and shape,
not another count vector. Source construction reconstructs the identity from
the checked layout before allocating; malformed configuration alone cannot
provide count provenance. Existing Core static initialization uploads it.

The fixed FastVideo `_pool_tiles` implementation promotes its zero-padded BF16
tile to an F32 sum and divides by the positive valid count, then permutes the
output heads. The Ferrum `f32_ascending_masked_rn_v1` profile visits every tile
row in increasing order. Valid rows are promoted exactly; invalid suffix rows
contribute positive zero without reading input storage. Each addition and the
final division round F32, with gradual underflow. A count outside `1..=R`
produces canonical positive quiet NaN and no source reads. No count is clamped.

The concrete reduction order is explicit and independently testable. It is not
a claim that the deployed Torch parallel sum chooses the same reduction tree.
The model test uses an admitted Host copy between two sessions to qualify the
individual gather and pool components; production composition can wire them
within one `ModelProgram` through the shared stateless fragment mechanism.
