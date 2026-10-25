// Shared by the attention kernel and its independently typed managed probe.
// Every thread participates; the caller supplies at least 64 scratch floats.
inline void attention_register_calibrate(
    threadgroup float* coordinates, uint tid, uint lane,
    thread uint2& rows, thread uint2& columns, thread uint* peers,
    thread uint& peer_count, thread uint& column_mask, thread uint& peer_mask,
    thread bool& four_peers, thread bool& xor_peers) {
    if (tid < 64) coordinates[tid] = float(tid);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_float8x8 coordinate_fragment;
    simdgroup_load(coordinate_fragment, coordinates, 8, 0, false);
    for (uint e = 0; e < 2; ++e) {
        uint coordinate = uint(coordinate_fragment.thread_elements()[e]);
        rows[e] = coordinate / 8;
        columns[e] = coordinate % 8;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    peer_count = 0;
    column_mask = 0;
    peer_mask = 0;
    bool same_row = simd_all(rows.x == rows.y);
    for (ushort source_lane = 0; source_lane < 32; ++source_lane) {
        uint source_row = simd_shuffle(rows.x, source_lane);
        uint2 source_columns = uint2(simd_shuffle(columns.x, source_lane), simd_shuffle(columns.y, source_lane));
        if (source_row == rows.x) {
            if (peer_count < 4) peers[peer_count] = source_lane;
            ++peer_count;
            column_mask |= (1u << source_columns.x) | (1u << source_columns.y);
            peer_mask |= 1u << source_lane;
        }
    }
    four_peers = simd_all(same_row && peer_count == 4 && column_mask == 255);
    const uint xor_peer_mask = (1u << lane) | (1u << (lane ^ 1u))
        | (1u << (lane ^ 8u)) | (1u << (lane ^ 9u));
    xor_peers = simd_all(four_peers && peer_mask == xor_peer_mask);
}
