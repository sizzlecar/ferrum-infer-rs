// Row-major input, head-major pooled output. No allocations or synchronization.
extern "C" __global__ void ferrum_vnext_masked_tile_mean_bf16_f32(
    const uint16_t* x, const int* counts, float* y,
    unsigned long long elements, unsigned long long source_rows,
    unsigned long long columns, unsigned long long heads,
    unsigned long long tiles, unsigned long long tile_rows,
    unsigned long long head_dim) {
    for (unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
         i < elements; i += (unsigned long long)blockDim.x * gridDim.x) {
        const unsigned long long d = i % head_dim;
        const unsigned long long tile = (i / head_dim) % tiles;
        const unsigned long long head = (i / (head_dim * tiles)) % heads;
        const unsigned long long batch = i / (head_dim * tiles * heads);
        const int count = counts[tile];
        if (count <= 0 || (unsigned long long)count > tile_rows) {
            y[i] = __uint_as_float(0x7fc00000u);
            continue;
        }
        float sum = 0.0f;
        for (unsigned long long row = 0; row < tile_rows; ++row) {
            float value = 0.0f;
            if (row < (unsigned long long)count) {
                const unsigned long long offset =
                    (batch * source_rows + tile * tile_rows + row) * columns + head * head_dim + d;
                value = __uint_as_float(uint32_t(x[offset]) << 16);
            }
            asm("add.rn.f32 %0, %1, %2;" : "=f"(sum) : "f"(sum), "f"(value));
        }
        const float divisor = (float)count;
        float mean;
        asm("div.rn.f32 %0, %1, %2;" : "=f"(mean) : "f"(sum), "f"(divisor));
        y[i] = mean;
    }
}
