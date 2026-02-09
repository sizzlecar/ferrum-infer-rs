//! Optimized attention kernel for Apple GPU
//! Supports both single-pass (short sequences) and two-pass (long sequences) attention

#include "definitions.metal"

// Attention kernel configuration (set via function constants)
constant uint NUM_HEADS [[function_constant(0)]];
constant uint HEAD_DIM [[function_constant(1)]];
constant uint NUM_KV_HEADS [[function_constant(2)]];
constant uint MAX_SEQ_LENGTH [[function_constant(3)]];
constant bool USE_GQA [[function_constant(4)]];

// Single-pass attention for sequences <= 1024 tokens
// Optimized for Apple Silicon with unified memory
kernel void attention_single_pass(
    device const float* queries [[buffer(0)]],     // [num_heads, seq_len, head_dim]
    device const float* keys [[buffer(1)]],        // [num_kv_heads, seq_len, head_dim]  
    device const float* values [[buffer(2)]],      // [num_kv_heads, seq_len, head_dim]
    device float* output [[buffer(3)]],            // [num_heads, seq_len, head_dim]
    device const float* mask [[buffer(4)]],        // [seq_len, seq_len] (optional)
    constant uint& seq_len [[buffer(5)]],
    constant float& scale [[buffer(6)]],           // 1.0 / sqrt(head_dim)
    uint3 tid [[thread_position_in_grid]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]
) {
    const uint head_idx = tgid.x;
    const uint seq_idx = tgid.y;
    const uint dim_idx = tid.z;
    
    if (head_idx >= NUM_HEADS || seq_idx >= seq_len || dim_idx >= HEAD_DIM) {
        return;
    }
    
    // For Grouped Query Attention, map head to kv_head
    const uint kv_head_idx = USE_GQA ? (head_idx / (NUM_HEADS / NUM_KV_HEADS)) : head_idx;
    
    // Shared memory for intermediate results
    threadgroup float shared_scores[ATTENTION_BLOCK_SIZE];
    threadgroup float shared_max[ATTENTION_BLOCK_SIZE];
    threadgroup float shared_sum[ATTENTION_BLOCK_SIZE];
    
    const uint local_id = lid.x;
    float thread_max = FERRUM_NEG_INFINITY;
    float thread_sum = 0.0f;
    
    // Query for this head and sequence position
    const uint q_offset = head_idx * seq_len * HEAD_DIM + seq_idx * HEAD_DIM;
    
    // Compute attention scores and find max (for numerical stability)
    for (uint k_seq = local_id; k_seq < seq_len; k_seq += ATTENTION_BLOCK_SIZE) {
        const uint k_offset = kv_head_idx * seq_len * HEAD_DIM + k_seq * HEAD_DIM;
        
        // Compute Q·K^T
        float score = 0.0f;
        for (uint d = 0; d < HEAD_DIM; ++d) {
            score += queries[q_offset + d] * keys[k_offset + d];
        }
        score *= scale;
        
        // Apply causal mask
        if (k_seq > seq_idx) {
            score = FERRUM_NEG_INFINITY;
        }
        
        // Apply additional mask if provided
        if (mask != nullptr) {
            score += mask[seq_idx * seq_len + k_seq];
        }
        
        thread_max = max(thread_max, score);
        shared_scores[local_id] = score;
    }
    
    // Reduce to find global max
    shared_max[local_id] = thread_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    for (uint stride = ATTENTION_BLOCK_SIZE / 2; stride > 0; stride /= 2) {
        if (local_id < stride) {
            shared_max[local_id] = max(shared_max[local_id], shared_max[local_id + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    const float global_max = shared_max[0];
    
    // Compute softmax probabilities
    thread_sum = 0.0f;
    for (uint k_seq = local_id; k_seq < seq_len; k_seq += ATTENTION_BLOCK_SIZE) {
        const float prob = safe_exp(shared_scores[local_id] - global_max);
        shared_scores[local_id] = prob;
        thread_sum += prob;
    }
    
    // Reduce to get sum for normalization
    shared_sum[local_id] = thread_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    for (uint stride = ATTENTION_BLOCK_SIZE / 2; stride > 0; stride /= 2) {
        if (local_id < stride) {
            shared_sum[local_id] += shared_sum[local_id + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    const float global_sum = max(shared_sum[0], FERRUM_EPSILON);
    
    // Compute final attention output: sum(prob_i * V_i)
    float result = 0.0f;
    for (uint v_seq = local_id; v_seq < seq_len; v_seq += ATTENTION_BLOCK_SIZE) {
        const uint v_offset = kv_head_idx * seq_len * HEAD_DIM + v_seq * HEAD_DIM + dim_idx;
        const float prob = shared_scores[local_id] / global_sum;
        
        if (v_seq < seq_len && dim_idx < HEAD_DIM) {
            result += prob * values[v_offset];
        }
    }
    
    // Write output
    const uint out_offset = head_idx * seq_len * HEAD_DIM + seq_idx * HEAD_DIM + dim_idx;
    if (out_offset < NUM_HEADS * seq_len * HEAD_DIM) {
        output[out_offset] = result;
    }
}
