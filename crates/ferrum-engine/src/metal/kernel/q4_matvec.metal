//! Q4_0 Matrix-Vector Multiplication Kernel for Apple GPU
//! 
//! High-performance SIMD implementation optimized for Apple Silicon.
//! Based on analysis of llama.cpp's ggml-metal Q4_0 kernels with 
//! simdgroup reductions and optimized memory access patterns.

#include "definitions.metal"

// Q4_0 quantization constants
constant uint QK4_0 = 32;                    // Elements per block  
constant uint WORDS_PER_BLOCK = 5;           // GPU format: scale + 4×u32 nibbles
constant uint N_SIMDWIDTH = 32;              // Apple GPU SIMD width

// Kernel parameters matching Metal argument buffer layout
typedef struct {
    uint32_t nrows;
    uint32_t ncols; 
    uint32_t blocks_per_row;
    uint32_t _pad;
} MatvecParams;

// Dequantize 4-bit nibble to float  
inline float dequantize_q4_0_nibble(uint nibble, float scale) {
    // Map [0, 15] back to [-8, 7] range, then scale
    return (float(nibble) - 8.0f) * scale;
}

// Main Q4_0 matvec kernel  
// Each threadgroup processes one output row using 32 threads
// Threads stride across blocks for coalesced memory access
kernel void q4_0_matvec_main(
    constant MatvecParams& params [[buffer(0)]],
    device const uint32_t* weights [[buffer(1)]],        // Q4_0 quantized weights
    device const float* input [[buffer(2)]],             // Input vector
    device float* output [[buffer(3)]],                  // Output vector
    uint3 tgpig [[threadgroup_position_in_grid]],        // threadgroup position
    uint tiitg [[thread_index_in_threadgroup]],          // thread index in threadgroup
    uint sgitg [[simdgroup_index_in_threadgroup]]        // simdgroup index in threadgroup  
) {
    const uint row = tgpig.y;
    if (row >= params.nrows) { return; }
    
    const uint thread_id = tiitg;
    const uint blocks_per_row = params.blocks_per_row;
    
    float accumulator = 0.0f;
    
    // Stride across blocks with N_SIMDWIDTH step for coalesced access
    for (uint block_idx = thread_id; block_idx < blocks_per_row; block_idx += N_SIMDWIDTH) {
        // Calculate base index for this block  
        const uint block_base = (row * blocks_per_row + block_idx) * WORDS_PER_BLOCK;
        
        // Load scale factor
        const uint scale_bits = weights[block_base];
        const float scale = as_type<float>(scale_bits);
        
        // Load packed nibbles (4 words containing 16 bytes = 32 nibbles)
        const uint nibble_word0 = weights[block_base + 1];
        const uint nibble_word1 = weights[block_base + 2];
        const uint nibble_word2 = weights[block_base + 3];
        const uint nibble_word3 = weights[block_base + 4];
        
        // Process all 32 elements in this block
        const uint block_input_base = block_idx * QK4_0;
        
        // Unroll nibble extraction and accumulation for efficiency
        // Based on ggml-metal's approach: process 4 words containing 16 bytes (32 nibbles)
        
        // Extract and accumulate elements 0-7 from nibble_word0
        for (uint byte_idx = 0; byte_idx < 4; ++byte_idx) {
            const uint byte_val = (nibble_word0 >> (byte_idx * 8)) & 0xFF;
            const uint nibble0 = byte_val & 0xF;
            const uint nibble1 = (byte_val >> 4) & 0xF;
            
            const uint input_idx0 = block_input_base + byte_idx * 2;
            const uint input_idx1 = block_input_base + byte_idx * 2 + 1;
            
            if (input_idx0 < params.ncols) {
                accumulator += dequantize_q4_0_nibble(nibble0, scale) * input[input_idx0];
            }
            if (input_idx1 < params.ncols) {
                accumulator += dequantize_q4_0_nibble(nibble1, scale) * input[input_idx1];
            }
        }
        
        // Extract and accumulate elements 8-15 from nibble_word1
        for (uint byte_idx = 0; byte_idx < 4; ++byte_idx) {
            const uint byte_val = (nibble_word1 >> (byte_idx * 8)) & 0xFF;
            const uint nibble0 = byte_val & 0xF;
            const uint nibble1 = (byte_val >> 4) & 0xF;
            
            const uint input_idx0 = block_input_base + 8 + byte_idx * 2;
            const uint input_idx1 = block_input_base + 8 + byte_idx * 2 + 1;
            
            if (input_idx0 < params.ncols) {
                accumulator += dequantize_q4_0_nibble(nibble0, scale) * input[input_idx0];
            }
            if (input_idx1 < params.ncols) {
                accumulator += dequantize_q4_0_nibble(nibble1, scale) * input[input_idx1];
            }
        }
        
        // Extract and accumulate elements 16-23 from nibble_word2
        for (uint byte_idx = 0; byte_idx < 4; ++byte_idx) {
            const uint byte_val = (nibble_word2 >> (byte_idx * 8)) & 0xFF;
            const uint nibble0 = byte_val & 0xF;
            const uint nibble1 = (byte_val >> 4) & 0xF;
            
            const uint input_idx0 = block_input_base + 16 + byte_idx * 2;
            const uint input_idx1 = block_input_base + 16 + byte_idx * 2 + 1;
            
            if (input_idx0 < params.ncols) {
                accumulator += dequantize_q4_0_nibble(nibble0, scale) * input[input_idx0];
            }
            if (input_idx1 < params.ncols) {
                accumulator += dequantize_q4_0_nibble(nibble1, scale) * input[input_idx1];
            }
        }
        
        // Extract and accumulate elements 24-31 from nibble_word3
        for (uint byte_idx = 0; byte_idx < 4; ++byte_idx) {
            const uint byte_val = (nibble_word3 >> (byte_idx * 8)) & 0xFF;
            const uint nibble0 = byte_val & 0xF;
            const uint nibble1 = (byte_val >> 4) & 0xF;
            
            const uint input_idx0 = block_input_base + 24 + byte_idx * 2;
            const uint input_idx1 = block_input_base + 24 + byte_idx * 2 + 1;
            
            if (input_idx0 < params.ncols) {
                accumulator += dequantize_q4_0_nibble(nibble0, scale) * input[input_idx0];
            }
            if (input_idx1 < params.ncols) {
                accumulator += dequantize_q4_0_nibble(nibble1, scale) * input[input_idx1];
            }
        }
    }
    
    // SIMD group reduction - Apple Silicon optimized
    // Use simd_sum for efficient reduction within SIMD groups
    const float sum_result = simd_sum(accumulator);
    
    // Only one thread per simdgroup needs to contribute to final result
    const uint lane_id = thread_id % N_SIMDWIDTH;
    if (lane_id == 0) {
        output[row] = sum_result;
    }
}
