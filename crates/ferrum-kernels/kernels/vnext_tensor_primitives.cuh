// Included by the precise BF16 bundle; no device-side allocation or global state.
struct FerrumTensorPrimitive {
    uint64_t rows, columns, head_dim;
    uint64_t starts[32], lengths[32];
    float scalar;
    uint32_t kind, count, padding;
};
static_assert(sizeof(FerrumTensorPrimitive) == 552, "tensor primitive ABI");

__device__ float ferrum_primitive_sum(float value, float* shared) {
    shared[threadIdx.x] = value;
    __syncthreads();
    for (unsigned step = blockDim.x / 2; step; step >>= 1) {
        if (threadIdx.x < step) shared[threadIdx.x] = __fadd_rn(shared[threadIdx.x], shared[threadIdx.x + step]);
        __syncthreads();
    }
    return shared[0];
}

extern "C" __global__ void ferrum_vnext_tensor_primitive_bf16(
    const uint16_t* x, const void* second, const void* third, uint16_t* y,
    const float* scratch, FerrumTensorPrimitive p) {
    __shared__ float shared[256];
    const uint16_t* a = static_cast<const uint16_t*>(second);
    const uint16_t* b = static_cast<const uint16_t*>(third);
    if (p.kind == 2) { // affine population layer norm, FP32 through affine
        for (uint64_t row = blockIdx.x; row < p.rows; row += gridDim.x) {
            float total = 0.0f;
            for (uint64_t c = threadIdx.x; c < p.columns; c += blockDim.x)
                total = __fadd_rn(total, ferrum_bf16_decode(x[row * p.columns + c]));
            const float mean = __fdiv_rn(ferrum_primitive_sum(total, shared), float(p.columns));
            __syncthreads();
            float squares = 0.0f;
            for (uint64_t c = threadIdx.x; c < p.columns; c += blockDim.x) {
                const float v = __fsub_rn(ferrum_bf16_decode(x[row * p.columns + c]), mean);
                squares = __fadd_rn(squares, __fmul_rn(v, v));
            }
            const float variance = __fdiv_rn(ferrum_primitive_sum(squares, shared), float(p.columns));
            const float inv = __fdiv_rn(1.0f, sqrtf(__fadd_rn(variance, p.scalar)));
            __syncthreads();
            for (uint64_t c = threadIdx.x; c < p.columns; c += blockDim.x) {
                const uint64_t i = row * p.columns + c;
                const float norm = __fmul_rn(__fsub_rn(ferrum_bf16_decode(x[i]), mean), inv);
                y[i] = ferrum_bf16_rne(__fadd_rn(__fmul_rn(norm, ferrum_bf16_decode(a[c])), ferrum_bf16_decode(b[c])));
            }
            __syncthreads();
        }
        return;
    }
    if (p.kind == 7) { // isolated full attention, one CTA per query/head
        const uint64_t heads = p.columns / p.head_dim;
        for (uint64_t task = blockIdx.x; task < p.rows * heads; task += gridDim.x) {
            const uint64_t row = task / heads, head = task % heads, d = threadIdx.x;
            uint32_t seg = 0;
            while (seg + 1 < p.count && row >= p.starts[seg] + p.lengths[seg]) ++seg;
            const uint64_t base = row * p.columns + head * p.head_dim;
            const float q = d < p.head_dim ? ferrum_bf16_decode(x[base + d]) : 0.0f;
            float maximum = -INFINITY, denominator = 0.0f, accumulator = 0.0f;
            bool nonfinite = false;
            for (uint64_t key = p.starts[seg]; key < p.starts[seg] + p.lengths[seg]; ++key) {
                const uint64_t kbase = key * p.columns + head * p.head_dim;
                const float product = d < p.head_dim ? __fmul_rn(q, ferrum_bf16_decode(a[kbase + d])) : 0.0f;
                const float score = __fmul_rn(ferrum_primitive_sum(product, shared), p.scalar);
                __syncthreads();
                nonfinite |= !isfinite(score);
                const float value = d < p.head_dim ? ferrum_bf16_decode(b[kbase + d]) : 0.0f;
                if (key == p.starts[seg]) {
                    maximum = score; denominator = 1.0f; accumulator = value;
                } else {
                    const float next = fmaxf(maximum, score);
                    const float old_weight = expf(__fsub_rn(maximum, next));
                    const float weight = expf(__fsub_rn(score, next));
                    denominator = __fadd_rn(__fmul_rn(denominator, old_weight), weight);
                    accumulator = __fadd_rn(__fmul_rn(accumulator, old_weight), __fmul_rn(weight, value));
                    maximum = next;
                }
            }
            if (d < p.head_dim) y[base + d] = ferrum_bf16_rne(nonfinite ? NAN : __fmul_rn(accumulator, __fdiv_rn(1.0f, denominator)));
            __syncthreads();
        }
        return;
    }
    const uint64_t first = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t stride = uint64_t(gridDim.x) * blockDim.x;
    for (uint64_t i = first; i < p.rows * p.columns; i += stride) {
        if (p.kind <= 1) {
            // nn.Linear and Conv3d have different observable bias boundaries.
            const float dot = p.kind == 1 ? ferrum_bf16_decode(ferrum_bf16_rne(scratch[i])) : scratch[i];
            y[i] = ferrum_bf16_rne(__fadd_rn(dot, ferrum_bf16_decode(b[i % p.columns])));
        } else if (p.kind == 3) {
            const float v = ferrum_bf16_decode(x[i]);
            y[i] = ferrum_bf16_rne(__fmul_rn(__fmul_rn(0.5f, v), __fadd_rn(1.0f, erff(__fmul_rn(v, 0.70710678118654752440f)))));
        } else if (p.kind == 4) {
            const uint64_t d = i % p.head_dim, half = p.head_dim / 2;
            const uint64_t partner = d < half ? i + half : i - half;
            const float rotated = d < half ? -ferrum_bf16_decode(x[partner]) : ferrum_bf16_decode(x[partner]);
            const uint64_t phase = (i / p.columns) * p.head_dim + d;
            const float cos = static_cast<const float*>(second)[phase], sin = static_cast<const float*>(third)[phase];
            y[i] = ferrum_bf16_rne(__fadd_rn(__fmul_rn(ferrum_bf16_decode(x[i]), cos), __fmul_rn(rotated, sin)));
        } else if (p.kind == 5) {
            const uint64_t row = i / p.columns;
            uint64_t source = 0;
            bool updated = false;
            for (uint32_t run = 0; run < p.count; ++run) {
                if (row >= p.starts[run] && row < p.starts[run] + p.lengths[run]) {
                    source += row - p.starts[run];
                    y[i] = ferrum_bf16_rne(__fadd_rn(ferrum_bf16_decode(x[i]), ferrum_bf16_decode(a[source * p.columns + i % p.columns])));
                    updated = true; break;
                }
                source += p.lengths[run];
            }
            if (!updated) y[i] = x[i];
        } else { // reshape is exact bit transport
            y[i] = x[i];
        }
    }
}
