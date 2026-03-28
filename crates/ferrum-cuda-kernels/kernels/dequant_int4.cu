// INT4 dequantization kernel (GPTQ format).
//
// Unpacks 4-bit weights from int32 words and converts to FP16 using
// per-group scales and zero-points.
//
// GPTQ packing: 8 x INT4 values per int32 word
//   int4_val[i] = (packed >> (i * 4)) & 0xF
//
// Dequantization: fp16_val = (int4_val - zero_point) * scale
//
// Grid:  (ceil(N/256), K/8)
// Block: (256,)

#include <cuda_fp16.h>

extern "C" __global__ void dequant_int4_to_fp16(
    const int*    __restrict__ qweight,   // [K/8, N] packed int32
    const __half* __restrict__ scales,    // [K/group_size, N] fp16
    const int*    __restrict__ qzeros,    // [K/group_size, N/8] packed int32
    __half*       __restrict__ output,    // [K, N] fp16
    const int K,
    const int N,
    const int group_size
) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;  // N dimension
    const int packed_row = blockIdx.y;                        // K/8 dimension

    if (col >= N) return;

    const int base_k = packed_row * 8;
    const int packed = qweight[packed_row * N + col];

    const int group = base_k / group_size;
    const float s = __half2float(scales[group * N + col]);

    // Extract zero-point for this (group, col)
    const int zp_packed = qzeros[group * (N / 8) + col / 8];
    const int zp_shift = (col % 8) * 4;
    const int zero = (zp_packed >> zp_shift) & 0xF;

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        const int val = (packed >> (i * 4)) & 0xF;
        const float dequantized = (float)(val - zero) * s;
        output[(base_k + i) * N + col] = __float2half(dequantized);
    }
}

// Symmetric variant (zero_point fixed at 8, no qzeros tensor needed).
// Used when quantize_config.json has "sym": true.
extern "C" __global__ void dequant_int4_sym_to_fp16(
    const int*    __restrict__ qweight,   // [K/8, N] packed int32
    const __half* __restrict__ scales,    // [K/group_size, N] fp16
    __half*       __restrict__ output,    // [K, N] fp16
    const int K,
    const int N,
    const int group_size
) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int packed_row = blockIdx.y;

    if (col >= N) return;

    const int base_k = packed_row * 8;
    const int packed = qweight[packed_row * N + col];

    const int group = base_k / group_size;
    const float s = __half2float(scales[group * N + col]);

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        const int val = (packed >> (i * 4)) & 0xF;
        const float dequantized = (float)(val - 8) * s;
        output[(base_k + i) * N + col] = __float2half(dequantized);
    }
}
