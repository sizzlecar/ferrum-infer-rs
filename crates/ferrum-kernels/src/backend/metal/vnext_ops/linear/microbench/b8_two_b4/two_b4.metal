// Appended ONLY by the Rust test to the unchanged production shader source.
// Two independent four-row tiles preserve each B4 dot-product order and live
// accumulators. This is NOT bitwise equivalent to half-coefficient B8 MMA.
#define TWO_B4_ENTRY(FORMAT, NAME) \
kernel void NAME( \
    device const half * input [[buffer(0)]], \
    device const FORMAT##Block * weights [[buffer(1)]], \
    device half * output [[buffer(2)]], \
    constant Params & p [[buffer(3)]], \
    uint3 group [[threadgroup_position_in_grid]], \
    ushort lane [[thread_index_in_simdgroup]], \
    ushort simdgroup [[simdgroup_index_in_threadgroup]]) { \
    if (p.rows != 8 || group.y >= 2) return; \
    const ulong row = ulong(group.y) * 4; \
    FORMAT##_shared<4, half>(input + row * p.in_features, weights, \
        output + row * p.output_stride, p, group.x, lane, simdgroup); \
}
TWO_B4_ENTRY(q4, q4_shared_b8_two_b4_experiment)
TWO_B4_ENTRY(q5, q5_shared_b8_two_b4_experiment)
TWO_B4_ENTRY(q6, q6_shared_b8_two_b4_experiment)
