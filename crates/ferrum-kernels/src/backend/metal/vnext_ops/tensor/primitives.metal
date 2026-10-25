struct PrimitiveParams {
    ulong elements, grid_width, columns, head_dim;
    uint kind, reserved;
};
kernel void ferrum_tensor_primitive_bf16(
    device const ushort* input [[buffer(0)]],
    device const float* cosine [[buffer(1)]],
    device const float* sine [[buffer(2)]],
    device ushort* output [[buffer(3)]],
    constant PrimitiveParams& p [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
    for (ulong i = gid; i < p.elements; i += p.grid_width) {
        if (p.kind == 2) { output[i] = input[i]; continue; }
        float x = decode_bf16(input[i]);
        float result;
        if (p.kind == 1) {
            ulong c = i % p.head_dim, half_width = p.head_dim / 2;
            ulong partner = c < half_width ? i + half_width : i - half_width;
            float rotated = decode_bf16(input[partner]);
            if (c < half_width) rotated = -rotated;
            ulong table = (i / p.columns) * p.head_dim + c;
            result = x * cosine[table] + rotated * sine[table];
        } else {
            float e = ferrum_erf_f32(x * 0.7071067811865475244f);
            result = (0.5f * x) * (1.0f + e);
        }
        output[i] = encode_bf16(result);
    }
}
