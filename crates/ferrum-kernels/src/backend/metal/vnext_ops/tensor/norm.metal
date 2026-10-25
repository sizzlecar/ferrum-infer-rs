struct NormParams {
    ulong groups; ulong grid_groups;
    uint width; uint padded; uint kind; uint threads;
    float epsilon; uint reserved;
};

inline void tensor_pairwise_sum(threadgroup float* values, uint padded, uint tid, uint threads) {
    for (uint stride = 1; stride < padded; stride *= 2) {
        for (uint left = tid * (2 * stride); left < padded; left += threads * (2 * stride))
            values[left] = values[left] + values[left + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// All reduction storage is declared on-chip threadgroup memory. Padding is
// initialized on every group; non-power-of-two widths divide by real width.
kernel void ferrum_tensor_norm_bf16(
    device const ushort* input [[buffer(0)]],
    device const ushort* gamma [[buffer(1)]],
    device ushort* output [[buffer(2)]],
    constant NormParams& p [[buffer(3)]],
    threadgroup float* values [[threadgroup(0)]],
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    for (ulong group = gid; group < p.groups; group += p.grid_groups) {
        ulong start = group * p.width;
        float mean = 0.0f;
        if (p.kind == 0) {
            for (uint i = tid; i < p.padded; i += p.threads)
                values[i] = i < p.width ? decode_bf16(input[start + i]) : 0.0f;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            tensor_pairwise_sum(values, p.padded, tid, p.threads);
            mean = values[0] / float(p.width);
            // Every reader must capture the mean before reusing the tree.
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        for (uint i = tid; i < p.padded; i += p.threads) {
            float x = i < p.width ? decode_bf16(input[start + i]) - mean : 0.0f;
            values[i] = x * x;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        tensor_pairwise_sum(values, p.padded, tid, p.threads);
        float variance = values[0] / float(p.width);
        float root = sqrt(variance + p.epsilon);
        float inverse = 1.0f / root;
        for (uint i = tid; i < p.width; i += p.threads) {
            float x = decode_bf16(input[start + i]) - mean;
            float normalized = x * inverse;
            if (p.kind == 1) normalized = rounded_bf16(normalized) * decode_bf16(gamma[i]);
            if (p.kind == 2) normalized = normalized * (1.0f + decode_bf16(gamma[i]));
            output[start + i] = encode_bf16(normalized);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
