// Direct native cuRAND F32 draws. This deliberately shares the established
// BF16 kernel's launch/state mapping, but never passes values through BF16.
// The numerical profile is native CUDA/cuRAND, not Torch CPU's generator.
extern "C" __global__ void ferrum_vnext_philox_normal_f32(
    const uint32_t* words, float* output, uint64_t elements, uint64_t rounds) {
    const uint64_t seed = uint64_t(words[0]) | (uint64_t(words[1]) << 32);
    const uint64_t offset = uint64_t(words[2]) | (uint64_t(words[3]) << 32);
    const uint64_t thread = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t stride = uint64_t(gridDim.x) * blockDim.x;
    curandStatePhilox4_32_10_t state;
    curand_init(seed, thread, offset, &state);
    for (uint64_t round = 0; round < rounds; ++round) {
        const float4 values = curand_normal4(&state);
        const uint64_t base = thread + round * stride * 4;
        if (base < elements) output[base] = __fadd_rn(__fmul_rn(values.x, 1.0f), 0.0f);
        if (base + stride < elements) output[base + stride] = __fadd_rn(__fmul_rn(values.y, 1.0f), 0.0f);
        if (base + 2 * stride < elements) output[base + 2 * stride] = __fadd_rn(__fmul_rn(values.z, 1.0f), 0.0f);
        if (base + 3 * stride < elements) output[base + 3 * stride] = __fadd_rn(__fmul_rn(values.w, 1.0f), 0.0f);
    }
}
