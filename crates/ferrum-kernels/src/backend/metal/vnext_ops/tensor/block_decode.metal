struct BlockDecodeParams { ulong elements; ulong grid_width; uint format; uint values; uint bytes; uint reserved; };

template <uint FORMAT, uint VALUES, uint BYTES>
inline void ferrum_decode_native_block_bf16(
    device const uchar* packed,
    device ushort* decoded,
    constant BlockDecodeParams& p,
    uint tid) {
    for (ulong i = tid; i < p.elements; i += p.grid_width) {
        // The typed block ABI makes both divisions power-of-two constants and
        // lets the original decoder discard unrelated format branches. Its
        // floating arithmetic and the final BF16 boundary stay unchanged.
        device const uchar* block = packed + (i / VALUES) * BYTES;
        decoded[i] = encode_bf16(native_block_value(block, uint(i % VALUES), FORMAT));
    }
}

kernel void ferrum_tensor_block_decode_bf16(
    device const uchar* packed [[buffer(0)]],
    device ushort* decoded [[buffer(1)]],
    constant BlockDecodeParams& p [[buffer(2)]],
    uint tid [[thread_position_in_grid]]) {
    // One uniform format dispatch per thread, outside the grid-stride loop.
    // The Rust geometry validates this GGUF format and its exact block ABI
    // before admission; keep the existing single PSO and invocation scratch.
    switch (p.format) {
        case 8:  ferrum_decode_native_block_bf16<8, 32, 34>(packed, decoded, p, tid); return;
        case 12: ferrum_decode_native_block_bf16<12, 256, 144>(packed, decoded, p, tid); return;
        case 13: ferrum_decode_native_block_bf16<13, 256, 176>(packed, decoded, p, tid); return;
        case 14: ferrum_decode_native_block_bf16<14, 256, 210>(packed, decoded, p, tid); return;
    }
    // Preserve the original generic behavior for an un-specialized format;
    // this does not expand the provider's declared supported-format set.
    for (ulong i = tid; i < p.elements; i += p.grid_width) {
        device const uchar* block = packed + (i / p.values) * p.bytes;
        decoded[i] = encode_bf16(native_block_value(block, uint(i % p.values), p.format));
    }
}
