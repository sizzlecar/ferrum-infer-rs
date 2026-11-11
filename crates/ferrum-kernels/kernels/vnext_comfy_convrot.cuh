// Regular ConvRot + canonical codebook W4A8. Included only by the precise
// vnext_bf16_silu translation unit (no fast math/FMA/FTZ). All pointers are
// admitted by the runtime; no kernel allocates storage or creates a context.
#pragma once

__device__ __forceinline__ float ferrum_comfy_h4(
    unsigned digit, float a, float b, float c, float d) {
    switch (digit) {
    case 0: return __fsub_rn(__fadd_rn(__fadd_rn(a, b), c), d);
    case 1: return __fadd_rn(__fsub_rn(__fadd_rn(a, b), c), d);
    case 2: return __fadd_rn(__fadd_rn(__fsub_rn(a, b), c), d);
    default: return __fadd_rn(__fadd_rn(__fadd_rn(-a, b), c), d);
    }
}

// The first two H4 stages never cross a 16-lane group. Preserve the original
// RN32 arithmetic and half scaling while exchanging values in registers.
// Every caller launches complete 256-thread blocks and all lanes participate.
__device__ __forceinline__ float ferrum_comfy_h4_warp_prefix(float value, unsigned i) {
    for (unsigned stride = 1; stride < 16; stride *= 4) {
        const unsigned digit = (i / stride) & 3;
        const unsigned base = (i - digit * stride) & 31;
        const float a = __shfl_sync(0xffffffffu, value, base);
        const float b = __shfl_sync(0xffffffffu, value, base + stride);
        const float c = __shfl_sync(0xffffffffu, value, base + 2 * stride);
        const float d = __shfl_sync(0xffffffffu, value, base + 3 * stride);
        value = __fmul_rn(0.5f, ferrum_comfy_h4(digit, a, b, c, d));
    }
    return value;
}

// Each block rotates a complete 256-element group, with a 1/2 multiplication
// after every H4 stage. This is the regular Kronecker H4, not Sylvester H256.
extern "C" __global__ void ferrum_comfy_convrot_bf16_f32(
    const uint16_t* input, float* rotated, uint64_t groups) {
    __shared__ float left[256];
    __shared__ float right[256];
    const unsigned i = threadIdx.x;
    for (uint64_t group = blockIdx.x; group < groups; group += gridDim.x) {
        left[i] = ferrum_comfy_h4_warp_prefix(
            ferrum_bf16_decode(input[group * 256 + i]), i);
        __syncthreads();
        float* source = left;
        float* destination = right;
        for (unsigned stride = 16; stride < 256; stride *= 4) {
            const unsigned digit = (i / stride) & 3;
            const unsigned base = i - digit * stride;
            destination[i] = __fmul_rn(0.5f, ferrum_comfy_h4(
                digit, source[base], source[base + stride],
                source[base + 2 * stride], source[base + 3 * stride]));
            __syncthreads();
            float* swap = source; source = destination; destination = swap;
        }
        rotated[group * 256 + i] = source[i];
        __syncthreads();
    }
}

// A separately declared activation profile: libdevice expf with precise RN32
// add/divide/multiply, no FMA/FTZ. Gate|up are BF16 but SiLU and the product
// remain F32 through H256. This arithmetic never switches at a row threshold.
// One block uses the same two 256-float tiles (2 KiB) as raw ConvRot; it does
// not require upstream's full-row dynamic shared-memory reservation.
extern "C" __global__ void ferrum_comfy_swiglu_convrot_bf16_f32(
    const uint16_t* gate_up, float* rotated, uint64_t groups, uint64_t k) {
    __shared__ float left[256];
    __shared__ float right[256];
    const unsigned i = threadIdx.x;
    for (uint64_t group = blockIdx.x; group < groups; group += gridDim.x) {
        const uint64_t index = group * 256 + i;
        const uint64_t row = index / k;
        const uint64_t column = index % k;
        const float gate = ferrum_bf16_decode(gate_up[row * (2 * k) + column]);
        const float up = ferrum_bf16_decode(gate_up[row * (2 * k) + k + column]);
        const float silu = __fdiv_rn(gate, __fadd_rn(1.0f, expf(-gate)));
        left[i] = ferrum_comfy_h4_warp_prefix(__fmul_rn(silu, up), i);
        __syncthreads();
        float* source = left;
        float* destination = right;
        for (unsigned stride = 16; stride < 256; stride *= 4) {
            const unsigned digit = (i / stride) & 3;
            const unsigned base = i - digit * stride;
            destination[i] = __fmul_rn(0.5f, ferrum_comfy_h4(
                digit, source[base], source[base + stride],
                source[base + 2 * stride], source[base + 3 * stride]));
            __syncthreads();
            float* swap = source; source = destination; destination = swap;
        }
        rotated[index] = source[i];
        __syncthreads();
    }
}

// Full-F32 input route. Same small H256 tile, with no input downcast.
extern "C" __global__ void ferrum_comfy_convrot_f32_f32(
    const float* input,float* rotated,uint64_t groups) {
    __shared__ float left[256];
    __shared__ float right[256];
    const unsigned i=threadIdx.x;
    for(uint64_t group=blockIdx.x;group<groups;group+=gridDim.x) {
        left[i]=ferrum_comfy_h4_warp_prefix(input[group*256+i],i);
        __syncthreads();
        float* source=left;float* destination=right;
        for(unsigned stride=16;stride<256;stride*=4) {
            const unsigned digit=(i/stride)&3;
            const unsigned base=i-digit*stride;
            destination[i]=__fmul_rn(0.5f,ferrum_comfy_h4(digit,source[base],source[base+stride],source[base+2*stride],source[base+3*stride]));
            __syncthreads();
            float* swap=source;source=destination;destination=swap;
        }
        rotated[group*256+i]=source[i];
        __syncthreads();
    }
}

// F32 specialization of upstream's quant_div_float_to_float<InputType>:
// neither numerator nor denominator nor quotient passes through BF16.
extern "C" __global__ void ferrum_comfy_quantize_activation_f32_i8(
    const float* rotated,int8_t* quantized,float* scales,uint64_t rows,uint64_t k) {
    __shared__ float maxima[256];
    const unsigned lane=threadIdx.x;
    for(uint64_t row=blockIdx.x;row<rows;row+=gridDim.x) {
        float maximum=0.0f;
        for(uint64_t col=lane;col<k;col+=256)maximum=fmaxf(maximum,fabsf(rotated[row*k+col]));
        maxima[lane]=maximum;
        __syncthreads();
        for(unsigned stride=128;stride>0;stride/=2) {
            if(lane<stride)maxima[lane]=fmaxf(maxima[lane],maxima[lane+stride]);
            __syncthreads();
        }
        const float maximum_finite=__int_as_float(0x7f7fffff);
        const float scale=fmaxf(__fmul_rn(fminf(maxima[0],maximum_finite),1.0f/127.0f),1.0e-30f);
        if(lane==0)scales[row]=scale;
        for(uint64_t col=lane;col<k;col+=256) {
            const uint64_t index=row*k+col;
            const float quotient=__fdiv_rn(rotated[index],scale);
            quantized[index]=static_cast<int8_t>(fminf(127.0f,fmaxf(-128.0f,nearbyintf(quotient))));
        }
        __syncthreads();
    }
}

// CUDA BF16 activation contract from the pinned comfy-kitchen int8_linear.cu:
// scale=max(min(amax,BF16_MAX)*(1/127),1e-30); q=RNE(B(B(x)/B(scale))).
// amax uses the F32 rotated values, before their explicit BF16 conversion.
extern "C" __global__ void ferrum_comfy_quantize_activation_bf16_i8(
    const float* rotated, int8_t* quantized, float* scales, uint64_t rows, uint64_t k) {
    __shared__ float maxima[256];
    const unsigned lane = threadIdx.x;
    for (uint64_t row = blockIdx.x; row < rows; row += gridDim.x) {
        float maximum = 0.0f;
        for (uint64_t col = lane; col < k; col += 256) {
            maximum = fmaxf(maximum, fabsf(rotated[row * k + col]));
        }
        maxima[lane] = maximum;
        __syncthreads();
        for (unsigned stride = 128; stride > 0; stride /= 2) {
            if (lane < stride) maxima[lane] = fmaxf(maxima[lane], maxima[lane + stride]);
            __syncthreads();
        }
        const float maximum_finite = ferrum_bf16_decode(0x7f7f);
        const float scale = fmaxf(__fmul_rn(fminf(maxima[0], maximum_finite), 1.0f / 127.0f), 1.0e-30f);
        if (lane == 0) scales[row] = scale;
        const float denominator = ferrum_bf16_decode(ferrum_bf16_rne(scale));
        for (uint64_t col = lane; col < k; col += 256) {
            const uint64_t index = row * k + col;
            const float numerator = ferrum_bf16_decode(ferrum_bf16_rne(rotated[index]));
            const float quotient = ferrum_bf16_decode(ferrum_bf16_rne(__fdiv_rn(numerator, denominator)));
            quantized[index] = static_cast<int8_t>(fminf(127.0f, fmaxf(-128.0f, nearbyintf(quotient))));
        }
        __syncthreads();
    }
}

__device__ __forceinline__ float ferrum_comfy_e4m3fn(uint8_t bits) {
    const unsigned exponent = (bits >> 3) & 15;
    const unsigned mantissa = bits & 7;
    float value;
    if (exponent == 0) value = float(mantissa) * (1.0f / 512.0f);
    else if (exponent == 15 && mantissa == 7) value = __int_as_float(0x7fffffff);
    else value = ldexpf(1.0f + float(mantissa) * 0.125f, int(exponent) - 7);
    return (bits & 128) ? -value : value;
}

// Even K is the low nibble, odd K the high nibble. Decode to the INT8 grid
// BEFORE channel scaling; floating codebook * scales alone is not this format.
extern "C" __global__ void ferrum_comfy_decode_codebook_w4_i8(
    const uint8_t* packed, const uint8_t* relative, const float* codebook,
    int8_t* decoded, uint64_t elements) {
    const uint64_t stride = uint64_t(gridDim.x) * blockDim.x;
    for (uint64_t index = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
         index < elements; index += stride) {
        const uint8_t word = packed[index / 2];
        const unsigned code = (index & 1) ? (word >> 4) : (word & 15);
        const float value = __fmul_rn(codebook[code], ferrum_comfy_e4m3fn(relative[index / 16]));
        decoded[index] = static_cast<int8_t>(fminf(127.0f, fmaxf(-127.0f, nearbyintf(value))));
    }
}

extern "C" __global__ void ferrum_comfy_int8_epilogue_bf16(
    const int32_t* accumulator, const float* row_scales, const float* channel_scales,
    uint16_t* output, uint64_t elements, uint64_t n) {
    const uint64_t stride = uint64_t(gridDim.x) * blockDim.x;
    for (uint64_t index = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
         index < elements; index += stride) {
        const float row_scaled = __fmul_rn(__int2float_rn(accumulator[index]), row_scales[index / n]);
        output[index] = ferrum_bf16_rne(__fmul_rn(row_scaled, channel_scales[index % n]));
    }
}

extern "C" __global__ void ferrum_comfy_int8_epilogue_f32(
    const int32_t* accumulator,const float* row_scales,const float* channel_scales,
    float* output,uint64_t elements,uint64_t n) {
    const uint64_t stride=uint64_t(gridDim.x)*blockDim.x;
    for(uint64_t index=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;index<elements;index+=stride) {
        const float row_scaled=__fmul_rn(__int2float_rn(accumulator[index]),row_scales[index/n]);
        output[index]=__fmul_rn(row_scaled,channel_scales[index%n]);
    }
}
