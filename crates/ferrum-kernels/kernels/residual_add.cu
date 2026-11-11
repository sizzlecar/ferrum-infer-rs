// Element-wise residual add: output[i] = a[i] + b[i]

#include <cuda_fp16.h>
#include <cuda_bf16.h>

extern "C" __global__ void residual_add_bf16(
    const __nv_bfloat16* a, const __nv_bfloat16* __restrict__ b,
    __nv_bfloat16* output, const int n) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < static_cast<unsigned int>(n)) {
        output[idx] = __float2bfloat16_rn(__bfloat162float(a[idx]) + __bfloat162float(b[idx]));
    }
}

extern "C" __global__ void residual_add_inplace_bf16(
    __nv_bfloat16* __restrict__ a, const __nv_bfloat16* __restrict__ b, const int n) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < static_cast<unsigned int>(n)) {
        a[idx] = __float2bfloat16_rn(__bfloat162float(a[idx]) + __bfloat162float(b[idx]));
    }
}

extern "C" __global__ void vnext_visual_row_select_f16(
    const unsigned short* __restrict__ input,
    const unsigned short* __restrict__ visual,
    const unsigned char* __restrict__ mask,
    unsigned short* __restrict__ output,
    const unsigned long long elements,
    const unsigned long long columns) {
    const unsigned long long index = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (index < elements) {
        // Mask=0 never touches visual memory, including poison/unused slots.
        // Every nonzero U8 predicate selects visual; no binary-only precondition.
        output[index] = mask[index / columns] == 0 ? input[index] : visual[index];
    }
}

extern "C" __global__ void vnext_visual_row_select_bf16(
    const unsigned short* __restrict__ input,
    const unsigned short* __restrict__ visual,
    const unsigned char* __restrict__ mask,
    unsigned short* __restrict__ output,
    const unsigned long long elements,
    const unsigned long long columns) {
    const unsigned long long index = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (index < elements) {
        // Mask=0 never touches visual memory, including poison/unused slots.
        // Every nonzero U8 predicate selects visual; no binary-only precondition.
        output[index] = mask[index / columns] == 0 ? input[index] : visual[index];
    }
}

extern "C" __global__ void residual_add_f16(
    // The vNext residual and recurrent providers permit output to alias a.
    // Do not promise the compiler that these two pointers are disjoint.
    const __half* a,
    const __half* __restrict__ b,
    __half* output,
    const int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = __float2half(__half2float(a[idx]) + __half2float(b[idx]));
    }
}

// In-place variant: a += b (a is in/out).
// Exists to avoid Rust borrow-checker conflicts when aliasing `a` as
// both read-only and writable argument.
extern "C" __global__ void residual_add_inplace_f16(
    __half* __restrict__ a,
    const __half* __restrict__ b,
    const int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        a[idx] = __float2half(__half2float(a[idx]) + __half2float(b[idx]));
    }
}

extern "C" __global__ void residual_add_inplace_f32(
    float* __restrict__ a,
    const float* __restrict__ b,
    const int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        a[idx] += b[idx];
    }
}

extern "C" __global__ void residual_add_f32(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ output,
    const int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = a[idx] + b[idx];
    }
}

// The output may alias the F32 residual input, as declared by the vNext
// contract. Do not apply restrict to either pointer in this variant.
extern "C" __global__ void vnext_residual_add_f32_f16(
    const float* residual,
    const __half* update,
    float* output,
    const int n
) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < static_cast<unsigned int>(n)) {
        output[idx] = residual[idx] + __half2float(update[idx]);
    }
}
