struct ScatterParams {
    ulong elements, grid_threads, columns;
    ulong starts[32], ends[32], update_starts[32];
    uint run_count, reserved;
};
kernel void ferrum_tensor_row_scatter_add_bf16(
    device const ushort* input [[buffer(0)]],
    device const ushort* updates [[buffer(1)]],
    device ushort* output [[buffer(2)]],
    constant ScatterParams& p [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
    for (ulong i = tid; i < p.elements; i += p.grid_threads) {
        ulong row = i / p.columns;
        ushort result = input[i];
        for (uint run = 0; run < p.run_count; ++run) {
            if (row < p.starts[run]) break;
            if (row < p.ends[run]) {
                ulong update = (p.update_starts[run] + row - p.starts[run]) * p.columns
                    + i % p.columns;
                result = encode_bf16(decode_bf16(result) + decode_bf16(updates[update]));
                break;
            }
        }
        output[i] = result;
    }
}
