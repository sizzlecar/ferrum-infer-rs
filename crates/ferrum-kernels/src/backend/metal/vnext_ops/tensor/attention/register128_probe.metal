// Ordinary U32 output of a diagnostic operation, never an attention output.
kernel void ferrum_attention_coordinate_probe(
    device const float* tag [[buffer(0)]],
    device uint* output [[buffer(1)]],
    device float* tag_echo [[buffer(2)]],
    threadgroup uchar* workspace [[threadgroup(0)]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sid [[simdgroup_index_in_threadgroup]]) {
    // Match the production calibration location and 128-thread launch.
    threadgroup float* coordinates = reinterpret_cast<threadgroup float*>(workspace + 16 + 32 * 136 * 2);
    uint2 rows, columns;
    uint peers[4], peer_count, column_mask, peer_mask;
    bool four_peers, xor_peers;
    attention_register_calibrate(coordinates, tid, lane, rows, columns, peers,
        peer_count, column_mask, peer_mask, four_peers, xor_peers);
    uint fields[13] = {as_type<uint>(tag[0]), lane, sid, rows.x, columns.x, rows.y, columns.y,
        peer_count, peer_mask, column_mask, uint(four_peers), uint(xor_peers), tid};
    for (uint field = 0; field < 13; ++field) output[tid * 13 + field] = fields[field];
    if (tid == 0) tag_echo[0] = tag[0];
}
