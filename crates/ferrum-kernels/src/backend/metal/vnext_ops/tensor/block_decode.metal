struct BlockDecodeParams { ulong elements; ulong grid_width; uint format; uint values; uint bytes; uint reserved; };

kernel void ferrum_tensor_block_decode_bf16(
    device const uchar* packed [[buffer(0)]],
    device ushort* decoded [[buffer(1)]],
    constant BlockDecodeParams& p [[buffer(2)]],
    uint tid [[thread_position_in_grid]]) {
    for (ulong i = tid; i < p.elements; i += p.grid_width) {
        device const uchar* block = packed + (i / p.values) * p.bytes;
        decoded[i] = encode_bf16(native_block_value(block, uint(i % p.values), p.format));
    }
}
