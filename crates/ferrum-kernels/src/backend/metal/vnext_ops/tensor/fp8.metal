#include <metal_simdgroup_matrix>

struct Fp8Params { uint rows; uint inputs; uint outputs; uint reserved; ulong scale_offset; };

inline float ferrum_e4m3fn_decode(uchar bits) {
    uint e = (bits >> 3) & 15, m = bits & 7;
    if (e == 15 && m == 7) return NAN;
    float value = e == 0 ? float(m) * (1.0f / 512.0f)
        : (1.0f + float(m) * 0.125f) * as_type<float>((e + 120) << 23);
    return (bits & 128) ? -value : value;
}
inline uchar ferrum_e4m3fn_encode(float value) {
    uint sign = (as_type<uint>(value) >> 24) & 128;
    if (!isfinite(value)) return uchar(sign | 127);
    float a = min(abs(value), 448.0f);
    if (a < 0.015625f) return uchar(sign | uint(rint(a * 512.0f)));
    int exponent = int((as_type<uint>(a) >> 23) & 255) - 127;
    float step = as_type<float>(uint(exponent + 124) << 23);
    uint code = uint((exponent + 6) * 8 + int(rint(a / step)));
    return uchar(sign | min(code, 126u));
}

// Finite row max; any non-finite input poisons the row instead of hiding it.
// Scale floor keeps division normal even for zero/subnormal BF16 rows.
kernel void ferrum_tensor_fp8_quantize(
    device const ushort* input [[buffer(0)]], device uchar* scratch [[buffer(1)]],
    constant Fp8Params& p [[buffer(2)]], uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float maxima[256];
    float maximum = 0.0f;
    for (ulong k = tid; k < p.inputs; k += 256) {
        float x = decode_bf16(input[ulong(row) * p.inputs + k]);
        maximum = max(maximum, isfinite(x) ? abs(x) : INFINITY);
    }
    maxima[tid] = maximum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride != 0; stride >>= 1) {
        if (tid < stride) maxima[tid] = max(maxima[tid], maxima[tid + stride]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float scale = isfinite(maxima[0]) ? max(maxima[0] / 448.0f, 1.0f / (448.0f * 512.0f)) : NAN;
    if (tid == 0) reinterpret_cast<device float*>(scratch + p.scale_offset)[row] = scale;
    for (ulong k = tid; k < p.inputs; k += 256)
        scratch[ulong(row) * p.inputs + k] = ferrum_e4m3fn_encode(decode_bf16(input[ulong(row) * p.inputs + k]) / scale);
}

// Decode packed operands directly into on-chip tiles. The device allocation
// contains no dense copy of either the complete model or a weight matrix.
kernel void ferrum_tensor_linear_fp8_bf16(
    device const uchar* scratch [[buffer(0)]], device const uchar* weight [[buffer(1)]],
    device ushort* output [[buffer(2)]], constant Fp8Params& p [[buffer(3)]],
    device const float* weight_scale [[buffer(4)]],
    uint3 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint sid [[simdgroup_index_in_threadgroup]]) {
    threadgroup float workspace[3072];
    threadgroup float* a = workspace;
    threadgroup float* b = workspace + 1024;
    ulong row_start = ulong(group.x) * 32, col_start = ulong(group.y) * 64;
    uint local_row = (sid / 2) * 16, local_col = (sid % 2) * 32;
    simdgroup_float8x8 sums[8];
    for (uint i = 0; i < 8; ++i) sums[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    for (ulong k = 0; k < p.inputs; k += 32) {
        for (uint i = tid; i < 1024; i += 128) {
            ulong row = row_start + i / 32, col = k + i % 32;
            a[i] = row < p.rows && col < p.inputs ? ferrum_e4m3fn_decode(scratch[row * p.inputs + col]) : 0.0f;
        }
        for (uint i = tid; i < 2048; i += 128) {
            ulong row = col_start + i % 64, col = k + i / 64;
            b[i] = row < p.outputs && col < p.inputs ? ferrum_e4m3fn_decode(weight[row * p.inputs + col]) : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint chunk = 0; chunk < 32; chunk += 8) {
            simdgroup_float8x8 av[2], bv[4];
            for (uint m = 0; m < 2; ++m) simdgroup_load(av[m], a + (local_row + m * 8) * 32 + chunk, 32, 0, false);
            for (uint n = 0; n < 4; ++n) simdgroup_load(bv[n], b + chunk * 64 + local_col + n * 8, 64, 0, false);
            for (uint m = 0; m < 2; ++m)
                for (uint n = 0; n < 4; ++n) simdgroup_multiply_accumulate(sums[m * 4 + n], av[m], bv[n], sums[m * 4 + n]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint m = 0; m < 2; ++m)
        for (uint n = 0; n < 4; ++n) simdgroup_store(sums[m * 4 + n], b + (local_row + m * 8) * 64 + local_col + n * 8, 64, 0, false);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device const float* scales = reinterpret_cast<device const float*>(scratch + p.scale_offset);
    float ws = weight_scale[0];
    for (uint i = tid; i < 2048; i += 128) {
        ulong row = row_start + i / 64, col = col_start + i % 64;
        if (row < p.rows && col < p.outputs) {
            float result = isfinite(ws) && ws > 0.0f ? (b[i] * scales[row]) * ws : NAN;
            output[row * p.outputs + col] = encode_bf16(result);
        }
    }
}
