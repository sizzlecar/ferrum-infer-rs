//! High-performance sampling kernels for token generation
//! Optimized for Apple Silicon with SIMD operations

#include "definitions.metal"

// Sampling configuration
constant uint VOCAB_SIZE [[function_constant(0)]];
constant uint SEED [[function_constant(1)]];

// Simple linear congruential generator for deterministic sampling
inline uint lcg_next(uint state) {
    return state * 1664525u + 1013904223u;
}

inline float uint_to_float(uint x) {
    return float(x) / float(0xFFFFFFFFu);
}

// Greedy sampling - select token with highest probability
kernel void sampling_greedy(
    device const float* logits [[buffer(0)]],       // [vocab_size]
    device uint* output_token [[buffer(1)]],        // [1]
    constant uint& vocab_size [[buffer(2)]],
    uint tid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint gid [[threadgroup_position_in_grid]]
) {
    threadgroup float shared_values[SAMPLING_BLOCK_SIZE];
    threadgroup uint shared_indices[SAMPLING_BLOCK_SIZE];
    
    // Initialize shared memory
    shared_values[lid] = FERRUM_NEG_INFINITY;
    shared_indices[lid] = 0;
    
    // Find local maximum
    if (tid < vocab_size) {
        shared_values[lid] = logits[tid];
        shared_indices[lid] = tid;
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Parallel reduction to find global maximum
    for (uint stride = SAMPLING_BLOCK_SIZE / 2; stride > 0; stride /= 2) {
        if (lid < stride) {
            if (shared_values[lid + stride] > shared_values[lid]) {
                shared_values[lid] = shared_values[lid + stride];
                shared_indices[lid] = shared_indices[lid + stride];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Write result
    if (lid == 0 && gid == 0) {
        output_token[0] = shared_indices[0];
    }
}

// Top-p (nucleus) sampling with optimized prefix sum
kernel void sampling_top_p(
    device const float* logits [[buffer(0)]],       // [vocab_size]
    device uint* output_token [[buffer(1)]],        // [1]
    constant uint& vocab_size [[buffer(2)]],
    constant float& top_p [[buffer(3)]],            // Top-p threshold
    constant float& temperature [[buffer(4)]],      // Temperature for softmax
    constant uint& random_seed [[buffer(5)]],       // Random seed
    uint tid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint gid [[threadgroup_position_in_grid]]
) {
    threadgroup float shared_probs[SAMPLING_BLOCK_SIZE];
    threadgroup uint shared_indices[SAMPLING_BLOCK_SIZE];
    threadgroup float shared_cumsum[SAMPLING_BLOCK_SIZE];
    
    // Apply temperature and convert to probabilities
    float max_logit = FERRUM_NEG_INFINITY;
    
    // First pass: find maximum logit for numerical stability
    for (uint i = tid; i < vocab_size; i += SAMPLING_BLOCK_SIZE) {
        max_logit = max(max_logit, logits[i]);
    }
    
    // Reduce to find global maximum
    shared_probs[lid] = max_logit;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    for (uint stride = SAMPLING_BLOCK_SIZE / 2; stride > 0; stride /= 2) {
        if (lid < stride) {
            shared_probs[lid] = max(shared_probs[lid], shared_probs[lid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    const float global_max = shared_probs[0];
    
    // Second pass: compute probabilities and sort by value
    float thread_sum = 0.0f;
    if (tid < vocab_size) {
        const float scaled_logit = (logits[tid] - global_max) / temperature;
        const float prob = safe_exp(scaled_logit);
        shared_probs[lid] = prob;
        shared_indices[lid] = tid;
        thread_sum = prob;
    } else {
        shared_probs[lid] = 0.0f;
        shared_indices[lid] = vocab_size;
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Simple bubble sort for top-p (can be optimized with bitonic sort)
    for (uint i = 0; i < SAMPLING_BLOCK_SIZE; ++i) {
        for (uint j = lid; j < SAMPLING_BLOCK_SIZE - 1; j += 2) {
            if (shared_probs[j] < shared_probs[j + 1]) {
                // Swap probabilities and indices
                float temp_prob = shared_probs[j];
                uint temp_idx = shared_indices[j];
                shared_probs[j] = shared_probs[j + 1];
                shared_indices[j] = shared_indices[j + 1];
                shared_probs[j + 1] = temp_prob;
                shared_indices[j + 1] = temp_idx;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Compute cumulative sum
    if (lid == 0) {
        float cumsum = 0.0f;
        float total_sum = 0.0f;
        
        // First compute total sum for normalization
        for (uint i = 0; i < SAMPLING_BLOCK_SIZE && shared_indices[i] < vocab_size; ++i) {
            total_sum += shared_probs[i];
        }
        
        // Generate random number
        uint rng_state = random_seed ^ (gid * 1664525u + 1013904223u);
        rng_state = lcg_next(rng_state);
        const float random_val = uint_to_float(rng_state) * top_p;
        
        // Find token where cumulative probability exceeds random threshold
        for (uint i = 0; i < SAMPLING_BLOCK_SIZE && shared_indices[i] < vocab_size; ++i) {
            cumsum += shared_probs[i] / total_sum;
            if (cumsum >= random_val) {
                output_token[0] = shared_indices[i];
                return;
            }
        }
        
        // Fallback to most probable token
        output_token[0] = shared_indices[0];
    }
}
