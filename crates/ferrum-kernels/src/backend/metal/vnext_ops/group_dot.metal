// IQ4_XS grouped dot. Four lanes cover each 32-value quant group; a SIMD
// covers all eight groups of one block. Packed codes are shared by B
// independent activation rows. Factoring the scale changes FP32 rounding;
// no decoded coefficient or accumulator is narrowed to half.
template<ushort B>
static inline void iq4_group_dot(
    device const half * input, device const uchar * weight, device half * output,
    constant NativeLinearParams & p, uint group, uint lane, uint subgroup) {
    const uint first = group * 4 + subgroup * 2;
    const uint blocks = p.in_features / 256;
    const uint quant_group = lane / 4;
    const uint lane_fragment = lane % 4;
    // Finish each output before starting the next to bound register lifetime.
    // Every activation row retains its own accumulator.
    #pragma clang loop unroll(disable)
    for (uint part = 0; part < 2; ++part) {
        const uint column = first + part;
        if (column >= p.out_features) continue;
        float sums[B] = {};
        for (uint block = 0; block < blocks; ++block) {
            device const uchar * w = weight + (ulong(column) * blocks + block) * 136;
            const ushort d_bits = ushort(w[0]) | (ushort(w[1]) << 8);
            const uint high = uint(w[2]) | (uint(w[3]) << 8);
            const uint lo = (w[4 + quant_group / 2] >> (4 * (quant_group % 2))) & 15;
            const uint hi = (high >> (2 * quant_group)) & 3;
            const float scale = float(as_type<half>(d_bits)) * float(int(lo | (hi << 4)) - 32);
            // 136-byte blocks and the byte offset preserve uint alignment.
            const uint packed = *reinterpret_cast<device const uint *>(
                w + 8 + quant_group * 16 + lane_fragment * 4);
            float4 low, upper;
            for (uint j = 0; j < 4; ++j) {
                low[j] = float(iq4_nl_values[(packed >> (8 * j)) & 15]);
                upper[j] = float(iq4_nl_values[(packed >> (8 * j + 4)) & 15]);
            }
            const ulong k = ulong(block) * 256 + quant_group * 32 + lane_fragment * 4;
            #pragma clang loop unroll(full)
            for (ushort batch = 0; batch < B; ++batch) {
                device const half * x = input + ulong(batch) * p.in_features + k;
                const float4 xl = float4(x[0], x[1], x[2], x[3]);
                const float4 xh = float4(x[16], x[17], x[18], x[19]);
                const float partial = dot(xl, low) + dot(xh, upper);
                sums[batch] += scale * partial;
            }
        }
        #pragma clang loop unroll(full)
        for (ushort batch = 0; batch < B; ++batch) {
            const float value = simd_sum(sums[batch]);
            if (lane == 0) {
                output[ulong(batch) * p.output_stride + p.output_column_offset + column] = half(value);
            }
        }
    }
}

#define GROUP_DOT_ENTRY(B) \
kernel void vnext_iq4_group_dot_b##B( \
    device const half * x [[buffer(0)]], device const uchar * w [[buffer(1)]], \
    device half * y [[buffer(2)]], constant NativeLinearParams & p [[buffer(3)]], \
    uint3 group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]], \
    uint subgroup [[simdgroup_index_in_threadgroup]]) { \
    if (p.rows != B) return; \
    iq4_group_dot<B>(x, w, y, p, group.x, lane, subgroup); \
}
GROUP_DOT_ENTRY(1)
GROUP_DOT_ENTRY(2)
GROUP_DOT_ENTRY(3)
GROUP_DOT_ENTRY(4)
#undef GROUP_DOT_ENTRY
