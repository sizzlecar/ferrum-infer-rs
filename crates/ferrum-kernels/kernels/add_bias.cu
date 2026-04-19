// Broadcast bias add: data[r, c] += bias[c] for every row.
// Used by Bert / Clip / Whisper linear projections (LLM path uses bias-free DenseLinear).
//
// Launch: grid = (rows, 1, 1), block = (min(cols, 1024), 1, 1).

#include <cuda_fp16.h>

extern "C" __global__ void add_bias_f16(
    __half* __restrict__ data,            // [rows, cols]
    const __half* __restrict__ bias,      // [cols]
    const int rows,
    const int cols
) {
    const int r = blockIdx.x;
    if (r >= rows) return;
    __half* row = data + r * cols;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        float v = __half2float(row[c]) + __half2float(bias[c]);
        row[c] = __float2half(v);
    }
}
