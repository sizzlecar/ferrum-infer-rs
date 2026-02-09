//! Core Metal definitions for Ferrum inference kernels
//! Optimized for Apple Silicon unified memory architecture

#include <metal_stdlib>
using namespace metal;

// Data type definitions
typedef float4 float4_t;
typedef half4 half4_t;

// Maximum supported dimensions
constant uint MAX_HEAD_DIM = 128;
constant uint MAX_SEQ_LEN = 4096;
constant uint WARP_SIZE = 32;

// Thread group sizes optimized for Apple GPU
constant uint ATTENTION_BLOCK_SIZE = 32;
constant uint ROPE_BLOCK_SIZE = 256;
constant uint SAMPLING_BLOCK_SIZE = 256;

// Math constants
constant float FERRUM_INFINITY = 3.40282347e+38F;
constant float FERRUM_NEG_INFINITY = -3.40282347e+38F;
constant float FERRUM_EPSILON = 1e-6;

// Utility functions
inline float safe_exp(float x) {
    return exp(clamp(x, -88.0f, 88.0f));
}

inline float safe_log(float x) {
    return log(max(x, FERRUM_EPSILON));
}

// SIMD group reduction for attention softmax
inline float simdgroup_sum(float value) {
    return simd_sum(value);
}

inline float simdgroup_max(float value) {
    return simd_max(value);
}
