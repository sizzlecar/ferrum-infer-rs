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

// Mask a single token index by setting its logit to -inf (in-place).
kernel void sampling_mask_one(
    device float* logits [[buffer(0)]],             // [vocab_size]
    constant uint& vocab_size [[buffer(1)]],
    constant uint& token_id [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid == 0 && token_id < vocab_size) {
        logits[token_id] = FERRUM_NEG_INFINITY;
    }
}

// Sample from a provided Top-K candidate list.
//
// This kernel:
// - reads logits at candidate indices (from the full vocab logits buffer)
// - applies repetition penalty (sparse: only for candidates) using (token_id, freq) list
// - applies temperature
// - softmax over K
// - applies top-p (nucleus) on the K candidates (after sorting by prob)
// - samples 1 token id and writes it to output
//
// Notes:
// - K is expected to be small (e.g. 32/64/128).
// - For correctness and simplicity, sorting is done with bitonic sort (O(K log^2 K)).
kernel void sampling_sample_from_topk(
    device const float* logits [[buffer(0)]],       // [vocab_size]
    device const uint* topk_indices [[buffer(1)]],  // [K]
    device const uint* rep_token_ids [[buffer(2)]], // [R]
    device const uint* rep_token_freqs [[buffer(3)]], // [R]
    device uint* output_token [[buffer(4)]],        // [1]
    constant uint& vocab_size [[buffer(5)]],
    constant uint& k [[buffer(6)]],
    constant uint& rep_len [[buffer(7)]],
    constant float& temperature [[buffer(8)]],
    constant float& top_p [[buffer(9)]],
    constant float& repetition_penalty [[buffer(10)]],
    constant uint& random_seed [[buffer(11)]],
    uint tid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]]
) {
    threadgroup float probs[SAMPLING_BLOCK_SIZE];
    threadgroup float logits_k[SAMPLING_BLOCK_SIZE];
    threadgroup uint idx_k[SAMPLING_BLOCK_SIZE];

    if (lid < SAMPLING_BLOCK_SIZE) {
        probs[lid] = 0.0f;
        logits_k[lid] = FERRUM_NEG_INFINITY;
        idx_k[lid] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Load candidate logits (only first K threads participate).
    if (lid < k && lid < SAMPLING_BLOCK_SIZE) {
        uint tok = topk_indices[lid];
        idx_k[lid] = tok;
        float v = (tok < vocab_size) ? logits[tok] : FERRUM_NEG_INFINITY;

        // Apply repetition penalty if requested (sparse: only over candidates).
        if (repetition_penalty != 1.0f && rep_len > 0) {
            // Linear scan over rep list (R is expected to be small-ish).
            for (uint i = 0; i < rep_len; ++i) {
                if (rep_token_ids[i] == tok) {
                    const uint freq = rep_token_freqs[i];
                    // penalty_factor = repetition_penalty ^ freq
                    float penalty_factor = 1.0f;
                    for (uint j = 0; j < freq; ++j) {
                        penalty_factor *= repetition_penalty;
                    }
                    if (v > 0.0f) {
                        v = v / penalty_factor;
                    } else {
                        v = v * penalty_factor;
                    }
                    break;
                }
            }
        }

        // Temperature scaling (temperature <= 0 treated as 1.0 here).
        const float t = (temperature > 0.0f) ? temperature : 1.0f;
        v = v / t;
        logits_k[lid] = v;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Compute softmax over K (single thread does stable softmax, K is small).
    if (lid == 0) {
        float maxv = FERRUM_NEG_INFINITY;
        for (uint i = 0; i < k && i < SAMPLING_BLOCK_SIZE; ++i) {
            maxv = max(maxv, logits_k[i]);
        }
        float sum = 0.0f;
        for (uint i = 0; i < k && i < SAMPLING_BLOCK_SIZE; ++i) {
            float p = safe_exp(logits_k[i] - maxv);
            probs[i] = p;
            sum += p;
        }
        sum = max(sum, FERRUM_EPSILON);
        for (uint i = 0; i < k && i < SAMPLING_BLOCK_SIZE; ++i) {
            probs[i] = probs[i] / sum;
        }

        // Sort by probability descending (bitonic sort) on [0, K)
        // We sort probs and idx_k together.
        uint n = k;
        if (n > SAMPLING_BLOCK_SIZE) n = SAMPLING_BLOCK_SIZE;

        // Next power of two for bitonic (cap to SAMPLING_BLOCK_SIZE)
        uint pow2 = 1;
        while (pow2 < n) pow2 <<= 1;
        if (pow2 > SAMPLING_BLOCK_SIZE) pow2 = SAMPLING_BLOCK_SIZE;

        // Pad remaining entries with -inf probs
        for (uint i = n; i < pow2; ++i) {
            probs[i] = 0.0f;
            idx_k[i] = 0;
        }

        for (uint size = 2; size <= pow2; size <<= 1) {
            for (uint stride = size >> 1; stride > 0; stride >>= 1) {
                for (uint i = 0; i < pow2; ++i) {
                    uint ixj = i ^ stride;
                    if (ixj > i) {
                        bool up = ((i & size) == 0);
                        // We want descending order => treat "up" as compare for greater.
                        float a = probs[i];
                        float b = probs[ixj];
                        bool swap = up ? (a < b) : (a > b);
                        if (swap) {
                            probs[i] = b;
                            probs[ixj] = a;
                            uint ti = idx_k[i];
                            idx_k[i] = idx_k[ixj];
                            idx_k[ixj] = ti;
                        }
                    }
                }
            }
        }

        // Apply top-p on sorted probs
        float cum = 0.0f;
        uint cutoff = n;
        float pth = clamp(top_p, 0.0f, 1.0f);
        if (pth < 1.0f) {
            for (uint i = 0; i < n; ++i) {
                cum += probs[i];
                if (cum >= pth) {
                    cutoff = i + 1;
                    break;
                }
            }
        }
        if (cutoff == 0) cutoff = 1;

        // Renormalize over [0, cutoff)
        float psum = 0.0f;
        for (uint i = 0; i < cutoff; ++i) psum += probs[i];
        psum = max(psum, FERRUM_EPSILON);
        for (uint i = 0; i < cutoff; ++i) probs[i] /= psum;

        // Draw sample
        uint rng_state = random_seed;
        rng_state = lcg_next(rng_state);
        float r = uint_to_float(rng_state);
        float c = 0.0f;
        for (uint i = 0; i < cutoff; ++i) {
            c += probs[i];
            if (c >= r) {
                output_token[0] = idx_k[i];
                return;
            }
        }

        // Fallback
        output_token[0] = idx_k[0];
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
