// Independent normalization/rotation ABI in the existing precise CUDA TU.
// No fast math, FMA contraction or FTZ; only the square reduction uses explicit
// FMA. Storage/module/context ownership comes from the ordinary managed lane.
#pragma once
extern "C" __global__ void ferrum_vnext_fused_head_rms_partial_rope_bf16(
    const uint16_t* input, const uint16_t* gamma, const uint16_t* cosine,
    const uint16_t* sine, uint16_t* output, uint64_t heads,
    uint64_t columns, uint64_t head_dim, uint64_t rotary_dim, float epsilon) {
    const unsigned lane = threadIdx.x & 31;
    const unsigned warp = threadIdx.x >> 5;
    const uint64_t first = uint64_t(blockIdx.x) * 4 + warp;
    const uint64_t stride = uint64_t(gridDim.x) * 4;
    for (uint64_t head = first; head < heads; head += stride) {
        const uint64_t start = head * head_dim;
        const uint64_t row = start / columns;
        float sum = 0.0f;
        for (uint64_t d = lane; d < head_dim; d += 32) {
            const float x = ferrum_bf16_decode(input[start + d]);
            sum = __fmaf_rn(x, x, sum);
        }
        for (unsigned offset = 16; offset; offset >>= 1)
            sum = __fadd_rn(sum, __shfl_down_sync(0xffffffffu, sum, offset));
        sum = __shfl_sync(0xffffffffu, sum, 0);
        const float mean = __fdiv_rn(sum, float(head_dim));
        const float inverse = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(mean, epsilon)));
        const uint64_t pairs = rotary_dim / 2;
        for (uint64_t d = lane; d < pairs; d += 32) {
            const uint64_t j = d + pairs;
            const float x0 = ferrum_bf16_decode(input[start + d]);
            const float x1 = ferrum_bf16_decode(input[start + j]);
            const float n0 = ferrum_bf16_decode(ferrum_bf16_rne(__fmul_rn(
                __fmul_rn(x0, inverse), ferrum_bf16_decode(gamma[d]))));
            const float n1 = ferrum_bf16_decode(ferrum_bf16_rne(__fmul_rn(
                __fmul_rn(x1, inverse), ferrum_bf16_decode(gamma[j]))));
            const float c0 = ferrum_bf16_decode(cosine[row * rotary_dim + d]);
            const float c1 = ferrum_bf16_decode(cosine[row * rotary_dim + j]);
            const float s0 = ferrum_bf16_decode(sine[row * rotary_dim + d]);
            const float s1 = ferrum_bf16_decode(sine[row * rotary_dim + j]);
            output[start + d] = ferrum_bf16_rne(__fadd_rn(__fmul_rn(n0,c0),__fmul_rn(-n1,s0)));
            output[start + j] = ferrum_bf16_rne(__fadd_rn(__fmul_rn(n1,c1),__fmul_rn(n0,s1)));
        }
        for (uint64_t d = rotary_dim + lane; d < head_dim; d += 32) {
            const float x = ferrum_bf16_decode(input[start + d]);
            output[start + d] = ferrum_bf16_rne(__fmul_rn(__fmul_rn(x,inverse),ferrum_bf16_decode(gamma[d])));
        }
    }
}
