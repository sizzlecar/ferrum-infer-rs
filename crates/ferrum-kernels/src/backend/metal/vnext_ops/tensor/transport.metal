struct TransportParams {
    ulong elements, grid_threads, rows_per_batch, source_width, column_start, columns, left_rows, right_rows;
    ulong run_ends[32], source_starts[32], batch_strides[32], row_strides[32];
    uint run_count, kind;
};
kernel void ferrum_tensor_transport_bf16(
    device const ushort* left [[buffer(0)]],
    device const ushort* right [[buffer(1)]],
    device ushort* output [[buffer(2)]],
    constant TransportParams& p [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
    for (ulong i = tid; i < p.elements; i += p.grid_threads) {
        ulong row = i / p.columns;
        ulong column = i % p.columns;
        ulong batch = row / p.rows_per_batch;
        ulong local_row = row % p.rows_per_batch;
        if (p.kind == 0) {
            ulong start = 0;
            for (uint run = 0; run < p.run_count; ++run) {
                if (local_row < p.run_ends[run]) {
                    ulong source_row = p.source_starts[run] + batch * p.batch_strides[run]
                        + (local_row - start) * p.row_strides[run];
                    output[i] = left[source_row * p.source_width + p.column_start + column];
                    break;
                }
                start = p.run_ends[run];
            }
        } else if (local_row < p.left_rows) {
            output[i] = left[(batch * p.left_rows + local_row) * p.columns + column];
        } else {
            output[i] = right[(batch * p.right_rows + local_row - p.left_rows) * p.columns + column];
        }
    }
}
