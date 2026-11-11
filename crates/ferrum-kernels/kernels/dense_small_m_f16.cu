// Experimental dense F16 projection for one or two rows. No production provider
// selects this module. The Rust diagnostic compares it with the existing cuBLAS
// path before any selector, numerical identity or resource-plan change.
//
// W is contiguous [outputs, inputs], X is [Rows, inputs], Y is [Rows, outputs].
// One warp owns an output column. For Rows=2 each loaded weight pair feeds both
// independent F32 dot products. This is not an activation/weight quantizer.
// Preconditions: positive extents, even inputs, 4-byte aligned X/W, disjoint Y,
// and exactly four warps per block. All extents/owners are checked by the caller.
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

template <int Rows>
__device__ __forceinline__ void dense_small_m_f16(
    const __half* __restrict__ x,
    const __half* __restrict__ w,
    __half* __restrict__ y,
    const int outputs,
    const int inputs
) {
    const int lane = threadIdx.x & 31;
    const int output = blockIdx.x * 4 + (threadIdx.x >> 5);
    if (output >= outputs) return;  // Uniform within this warp.

    const int pairs = inputs / 2;
    const __half2* wr = reinterpret_cast<const __half2*>(
        w + static_cast<int64_t>(output) * inputs);
    const __half2* x0 = reinterpret_cast<const __half2*>(x);
    const __half2* x1 = reinterpret_cast<const __half2*>(x + inputs);
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    for (int pair = lane; pair < pairs; pair += 32) {
        const float2 weight = __half22float2(wr[pair]);
        const float2 input0 = __half22float2(x0[pair]);
        sum0 = __fmaf_rn(weight.x, input0.x, sum0);
        sum0 = __fmaf_rn(weight.y, input0.y, sum0);
        if (Rows == 2) {
            const float2 input1 = __half22float2(x1[pair]);
            sum1 = __fmaf_rn(weight.x, input1.x, sum1);
            sum1 = __fmaf_rn(weight.y, input1.y, sum1);
        }
    }
    // This explicit F32 reduction tree may differ from cuBLAS. The operation's
    // F16 stores are retained; bit equality for arbitrary sums is not promised.
    for (int offset = 16; offset != 0; offset >>= 1) {
        sum0 = __fadd_rn(sum0, __shfl_down_sync(0xffffffffu, sum0, offset));
        if (Rows == 2) {
            sum1 = __fadd_rn(sum1, __shfl_down_sync(0xffffffffu, sum1, offset));
        }
    }
    if (lane == 0) {
        y[output] = __float2half_rn(sum0);
        if (Rows == 2) y[static_cast<int64_t>(outputs) + output] = __float2half_rn(sum1);
    }
}

extern "C" __global__ void dense_small_m_f16_m1(
    const __half* x, const __half* w, __half* y, int outputs, int inputs
) {
    dense_small_m_f16<1>(x, w, y, outputs, inputs);
}

extern "C" __global__ void dense_small_m_f16_m2(
    const __half* x, const __half* w, __half* y, int outputs, int inputs
) {
    dense_small_m_f16<2>(x, w, y, outputs, inputs);
}
