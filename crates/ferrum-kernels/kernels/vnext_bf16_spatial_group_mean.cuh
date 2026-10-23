#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>

// Conceptual [C, temporal phase, dy, dx] grouping. The real single frame
// occupies the last temporal phase; preceding phases are positive zero.
struct FerrumSpatialGroupMean {
    unsigned long long input_channels, input_height, input_width;
    unsigned long long output_channels, output_height, output_width;
    unsigned long long scale_h, scale_w, frame_factor, group_size;
};
static_assert(sizeof(FerrumSpatialGroupMean) == 80, "group mean ABI");
__device__ __forceinline__ float ferrum_group_mean_read(
    const __nv_bfloat16* input, FerrumSpatialGroupMean g,
    unsigned long long n, unsigned long long o, unsigned long long y,
    unsigned long long x, unsigned long long j) {
    const auto spatial = g.scale_h * g.scale_w;
    const auto index = o * g.group_size + j;
    const auto c = index / (g.frame_factor * spatial);
    const auto phase = index % (g.frame_factor * spatial);
    if (phase / spatial != g.frame_factor - 1) return 0.0f;
    const auto dy = (phase % spatial) / g.scale_w;
    const auto dx = phase % g.scale_w;
    return __bfloat162float(input[((n * g.input_channels + c) * g.input_height
        + y * g.scale_h + dy) * g.input_width + x * g.scale_w + dx]);
}
extern "C" __global__ void ferrum_vnext_spatial_group_mean_bf16(
    const __nv_bfloat16* input, __nv_bfloat16* output,
    unsigned long long elements, FerrumSpatialGroupMean g) {
    const auto step = static_cast<unsigned long long>(blockDim.x) * gridDim.x;
    for (auto i = static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < elements; i += step) {
        const auto x = i % g.output_width;
        const auto y = (i / g.output_width) % g.output_height;
        const auto o = (i / (g.output_width * g.output_height)) % g.output_channels;
        const auto n = i / (g.output_width * g.output_height * g.output_channels);
        float sum = ferrum_group_mean_read(input, g, n, o, y, x, 0);
        for (unsigned long long j = 1; j < g.group_size; ++j)
            sum = __fadd_rn(sum, ferrum_group_mean_read(input, g, n, o, y, x, j));
        output[i] = __float2bfloat16_rn(__fdiv_rn(sum, static_cast<float>(g.group_size)));
    }
}
