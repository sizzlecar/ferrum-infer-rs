#pragma once
// Contiguous N,C,1,L signals. No allocation, stream creation or model state.
extern "C" __global__ void ferrum_vnext_fir_resample1d_f32(
    const float* x, const float* taps, float* y,
    unsigned long long elements, unsigned long long input_width,
    unsigned long long output_width, unsigned long long kernel,
    unsigned long long stride, unsigned long long pad_left,
    unsigned long long pad_right, unsigned long long crop_left, int transpose) {
    const unsigned long long padded = input_width + pad_left + pad_right;
    for (unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
         i < elements; i += (unsigned long long)blockDim.x * gridDim.x) {
        const unsigned long long position = i % output_width + crop_left;
        const unsigned long long base = i / output_width * input_width;
        float sum = 0.0f;
        for (unsigned long long k = 0; k < kernel; ++k) {
            unsigned long long j;
            if (transpose) {
                if (position < k || (position - k) % stride != 0) continue;
                j = (position - k) / stride;
                if (j >= padded) continue;
            } else {
                j = position * stride + k;
            }
            j = j < pad_left ? 0 : j - pad_left;
            j = j < input_width ? j : input_width - 1;
            sum = __fmaf_rn(x[base + j], taps[k], sum);
        }
        y[i] = sum;
    }
}
extern "C" __global__ void ferrum_vnext_snake_beta_log_nchw_f32(
    const float* x, const float* log_alpha, const float* log_beta, float* y,
    unsigned long long elements, unsigned long long channels,
    unsigned long long spatial) {
    for (unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
         i < elements; i += (unsigned long long)blockDim.x * gridDim.x) {
        const unsigned long long c = i / spatial % channels;
        const float alpha = expf(log_alpha[c]);
        const float beta = expf(log_beta[c]);
        const float sine = sinf(__fmul_rn(alpha, x[i]));
        const float reciprocal = __fdiv_rn(1.0f, __fadd_rn(beta, 1.0e-9f));
        y[i] = __fadd_rn(x[i], __fmul_rn(reciprocal, __fmul_rn(sine, sine)));
    }
}
extern "C" __global__ void ferrum_vnext_clamp_rank4_f32(
    const float* x, float* y, unsigned long long elements, float lower, float upper) {
    for (unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
         i < elements; i += (unsigned long long)blockDim.x * gridDim.x) {
        const float v = x[i];
        y[i] = v < lower ? lower : v > upper ? upper : v;
    }
}
extern "C" __global__ void ferrum_vnext_snake_alpha_nchw_f32(
    const float* x, const float* alphas, float* y,
    unsigned long long elements, unsigned long long channels,
    unsigned long long spatial) {
    for (unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
         i < elements; i += (unsigned long long)blockDim.x * gridDim.x) {
        const float alpha = alphas[i / spatial % channels];
        const float sine = sinf(__fmul_rn(alpha, x[i]));
        const float reciprocal = __fdiv_rn(1.0f, __fadd_rn(alpha, 1.0e-9f));
        y[i] = __fadd_rn(x[i], __fmul_rn(reciprocal, __fmul_rn(sine, sine)));
    }
}
