// Included after dense_core.metal; packet matches the strict 64-byte ABI.
struct FerrumDenseGuardOriginalParams {
    ulong jobs, grid_groups, query_tiles, query_sequence, query_heads, key_value_heads, key_sequence;
    uint head_dim;
    float scale;
};

// Sufficient overflow envelope, distinct from the original exp-based profile:
// abs(Q/K/V)<=2^20, D128, scale<=2^20 give |scaled log2 scores|<2^69,
// including ample F32 rounding margin. Max subtraction feeds exp2 nonpositive
// arguments; underflow to zero is allowed by this independent fused profile.
// For an ordinary finite exp2 result near [0,1], K<=65536 keeps the numerator
// far below F32 overflow. This is NOT an imported ULP guarantee for fast::exp2:
// the actual fast CTA additionally checks positive finite denominator and
// bounded finite normalized output before storing; otherwise strict reruns it.
// Every consumed flag is overwritten, even prefix/rejected jobs. No Host read.
kernel void ferrum_dense_attention_prepass(
    device const ushort* query [[buffer(0)]],
    device const ushort* key [[buffer(1)]],
    device const ushort* value [[buffer(2)]],
    constant FerrumDenseGuardOriginalParams& p [[buffer(4)]],
    constant FerrumDenseGateParams& gate [[buffer(5)]],
    device uint* flags [[buffer(6)]],
    threadgroup uchar* workspace [[threadgroup(0)]],
    uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    threadgroup atomic_uint* rejected = reinterpret_cast<threadgroup atomic_uint*>(workspace);
    const ulong heads = p.jobs / p.query_tiles;
    const ulong groups = min(heads, 65535ul);
    const ulong first_query = gate.first_fast_tile * 32;
    for (ulong head = group; head < heads; head += groups) {
        if (tid == 0) atomic_store_explicit(rejected, 0, memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const ulong batch = head / p.query_heads;
        const ulong h = head % p.query_heads;
        const ulong q_base = (batch * p.query_sequence * p.query_heads + h) * 128;
        const ulong kv_base = (batch * p.key_sequence * p.key_value_heads + h) * 128;
        bool bad = false;
        for (ulong i = first_query * 128 + tid; i < p.query_sequence * 128; i += 128) {
            const ulong offset = q_base + (i / 128) * p.query_heads * 128 + i % 128;
            bad = bad || (query[offset] & ushort(0x7fff)) > ushort(0x4980);
        }
        for (ulong i = tid; i < p.key_sequence * 128; i += 128) {
            const ulong offset = kv_base + (i / 128) * p.key_value_heads * 128 + i % 128;
            bad = bad || (key[offset] & ushort(0x7fff)) > ushort(0x4980)
                || (value[offset] & ushort(0x7fff)) > ushort(0x4980);
        }
        if (bad) atomic_fetch_or_explicit(rejected, 1, memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const bool admitted = atomic_load_explicit(rejected, memory_order_relaxed) == 0;
        for (ulong tile = tid; tile < p.query_tiles; tile += 128)
            flags[head * p.query_tiles + tile] = admitted && tile >= gate.first_fast_tile ? 1u : 0u;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
