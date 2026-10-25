struct ChannelParams {
    ulong spatial, pixels, grid_groups;
    uint channels, padded, threads, reserved;
    float epsilon, scale;
};
kernel void ferrum_tensor_channel_l2_bf16(
    device const ushort* input [[buffer(0)]], device const ushort* gamma [[buffer(1)]],
    device ushort* output [[buffer(2)]], constant ChannelParams& p [[buffer(3)]],
    threadgroup float* sums [[threadgroup(0)]], uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    for (ulong pixel = group; pixel < p.pixels; pixel += p.grid_groups) {
        ulong base = (pixel / p.spatial) * p.channels * p.spatial + pixel % p.spatial;
        for (uint c = tid; c < p.padded; c += p.threads) {
            float x = c < p.channels ? decode_bf16(input[base + ulong(c) * p.spatial]) : 0.0f;
            sums[c] = x * x;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = 1; stride < p.padded; stride *= 2) {
            for (uint pair = tid; pair < p.padded / (2 * stride); pair += p.threads) {
                uint c = pair * stride * 2;
                sums[c] = sums[c] + sums[c + stride];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        float length = sqrt(sums[0]);
        float denominator = isnan(length) ? length : max(length, p.epsilon);
        for (uint c = tid; c < p.channels; c += p.threads) {
            ulong i = base + ulong(c) * p.spatial;
            float value = rounded_bf16(decode_bf16(input[i]) / denominator);
            value = rounded_bf16(value * p.scale);
            value = rounded_bf16(value * decode_bf16(gamma[c]));
            output[i] = encode_bf16(value + 0.0f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
