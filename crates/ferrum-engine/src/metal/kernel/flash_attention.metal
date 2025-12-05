//! Flash Attention kernel for Apple GPU
//!
//! Memory-efficient attention using online softmax algorithm.
//! Based on Flash Attention paper: https://arxiv.org/abs/2205.14135
//!
//! Key optimizations:
//! - Online softmax to avoid materializing full attention matrix
//! - Block-based processing for cache efficiency
//! - Shared memory for intermediate results
//! - Fused QKV projection when possible

#include "definitions.metal"

// Block sizes for Flash Attention
// Tuned for Apple M-series unified memory architecture
constant uint FLASH_BLOCK_M [[function_constant(10)]];  // Query block size
constant uint FLASH_BLOCK_N [[function_constant(11)]];  // Key/Value block size
constant uint FLASH_HEAD_DIM [[function_constant(12)]]; // Head dimension

// ============================================================================
// Flash Attention - Single Query (Decode Phase)
// ============================================================================
// Optimized for generating one token at a time

kernel void flash_attention_decode(
    device const float* query [[buffer(0)]],      // [batch, num_heads, 1, head_dim]
    device const float* key_cache [[buffer(1)]],  // [batch, num_kv_heads, max_seq, head_dim]
    device const float* value_cache [[buffer(2)]],// [batch, num_kv_heads, max_seq, head_dim]
    device float* output [[buffer(3)]],           // [batch, num_heads, 1, head_dim]
    constant uint& batch_size [[buffer(4)]],
    constant uint& num_heads [[buffer(5)]],
    constant uint& num_kv_heads [[buffer(6)]],
    constant uint& head_dim [[buffer(7)]],
    constant uint& seq_len [[buffer(8)]],
    constant float& scale [[buffer(9)]],
    uint3 tid [[thread_position_in_grid]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    const uint batch_idx = tgid.z;
    const uint head_idx = tgid.y;
    const uint dim_idx = tid.x;
    
    if (batch_idx >= batch_size || head_idx >= num_heads || dim_idx >= head_dim) {
        return;
    }
    
    // For GQA: map query head to KV head
    const uint kv_head_idx = head_idx / (num_heads / num_kv_heads);
    
    // Shared memory for block-wise computation
    threadgroup float shared_scores[256];  // Attention scores for current block
    threadgroup float shared_max[32];       // Maximum scores per warp
    threadgroup float shared_sum[32];       // Sum of exp(scores) per warp
    
    const uint local_id = lid.x;
    
    // Load query vector
    const uint q_offset = batch_idx * num_heads * head_dim + head_idx * head_dim + dim_idx;
    float q_val = query[q_offset];
    
    // Online softmax variables
    float m_prev = -INFINITY;
    float l_prev = 0.0f;
    float o_acc = 0.0f;
    
    // Process keys in blocks for cache efficiency
    const uint block_size = min(256u, seq_len);
    
    for (uint block_start = 0; block_start < seq_len; block_start += block_size) {
        const uint block_end = min(block_start + block_size, seq_len);
        
        // Compute attention scores for this block
        float block_max = -INFINITY;
        
        for (uint k_idx = block_start + local_id; k_idx < block_end; k_idx += 32) {
            const uint k_offset = batch_idx * num_kv_heads * seq_len * head_dim + 
                                  kv_head_idx * seq_len * head_dim + 
                                  k_idx * head_dim + dim_idx;
            
            // Compute Q·K^T for this position
            float score = q_val * key_cache[k_offset];
            
            // SIMD reduction to sum across head dimension
            score = simd_sum(score);
            score *= scale;
            
            if (local_id == 0) {
                shared_scores[k_idx - block_start] = score;
                block_max = max(block_max, score);
            }
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Reduce to find block max
        if (local_id < 32) {
            shared_max[local_id] = block_max;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        for (uint stride = 16; stride > 0; stride /= 2) {
            if (local_id < stride) {
                shared_max[local_id] = max(shared_max[local_id], shared_max[local_id + stride]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        
        float m_new = max(m_prev, shared_max[0]);
        
        // Compute exp(scores - m_new) and sum
        float block_sum = 0.0f;
        for (uint k_idx = block_start + local_id; k_idx < block_end; k_idx += 32) {
            float score = shared_scores[k_idx - block_start];
            float p = exp(score - m_new);
            shared_scores[k_idx - block_start] = p;  // Store normalized prob
            block_sum += p;
        }
        
        // Reduce to get block sum
        block_sum = simd_sum(block_sum);
        if (simd_lane_id == 0) {
            shared_sum[simd_group_id] = block_sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        if (local_id == 0) {
            float total_sum = 0.0f;
            for (uint i = 0; i < 8; ++i) {
                total_sum += shared_sum[i];
            }
            shared_sum[0] = total_sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        float l_block = shared_sum[0];
        
        // Online softmax update
        float correction = exp(m_prev - m_new);
        o_acc = o_acc * correction;
        l_prev = l_prev * correction + l_block;
        
        // Accumulate weighted values
        for (uint v_idx = block_start + local_id; v_idx < block_end; v_idx += 32) {
            const uint v_offset = batch_idx * num_kv_heads * seq_len * head_dim + 
                                  kv_head_idx * seq_len * head_dim + 
                                  v_idx * head_dim + dim_idx;
            
            float p = shared_scores[v_idx - block_start];
            o_acc += p * value_cache[v_offset];
        }
        
        m_prev = m_new;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Normalize output
    float result = o_acc / max(l_prev, 1e-6f);
    
    // Write output
    const uint out_offset = batch_idx * num_heads * head_dim + head_idx * head_dim + dim_idx;
    output[out_offset] = result;
}

// ============================================================================
// Flash Attention - Prefill (Multiple Queries)
// ============================================================================
// Optimized for processing initial prompt

kernel void flash_attention_prefill(
    device const float* queries [[buffer(0)]],   // [batch, num_heads, seq_q, head_dim]
    device const float* keys [[buffer(1)]],      // [batch, num_kv_heads, seq_k, head_dim]
    device const float* values [[buffer(2)]],    // [batch, num_kv_heads, seq_k, head_dim]
    device float* output [[buffer(3)]],          // [batch, num_heads, seq_q, head_dim]
    constant uint& batch_size [[buffer(4)]],
    constant uint& num_heads [[buffer(5)]],
    constant uint& num_kv_heads [[buffer(6)]],
    constant uint& head_dim [[buffer(7)]],
    constant uint& seq_q [[buffer(8)]],
    constant uint& seq_k [[buffer(9)]],
    constant float& scale [[buffer(10)]],
    constant bool& causal [[buffer(11)]],
    uint3 tid [[thread_position_in_grid]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]
) {
    // Block indices
    const uint batch_idx = tgid.z;
    const uint head_idx = tgid.y;
    const uint q_block_idx = tgid.x;
    
    // Thread indices within block
    const uint row_in_block = lid.y;  // Which query row within the block
    const uint dim_idx = lid.x;        // Which dimension
    
    if (batch_idx >= batch_size || head_idx >= num_heads) {
        return;
    }
    
    const uint kv_head_idx = head_idx / (num_heads / num_kv_heads);
    
    // Calculate global query index
    const uint BLOCK_M = 32;  // Query block size
    const uint BLOCK_N = 64;  // Key block size
    
    const uint q_idx = q_block_idx * BLOCK_M + row_in_block;
    if (q_idx >= seq_q) {
        return;
    }
    
    // Shared memory
    threadgroup float shared_q[32 * 128];    // Query block
    threadgroup float shared_k[64 * 128];    // Key block
    threadgroup float shared_v[64 * 128];    // Value block
    threadgroup float shared_scores[32 * 64]; // Attention scores
    
    // Load query row into shared memory
    const uint q_offset = batch_idx * num_heads * seq_q * head_dim + 
                          head_idx * seq_q * head_dim + 
                          q_idx * head_dim + dim_idx;
    
    if (dim_idx < head_dim) {
        shared_q[row_in_block * head_dim + dim_idx] = queries[q_offset];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Online softmax accumulators
    float m_i = -INFINITY;
    float l_i = 0.0f;
    float o_i[128];  // Output accumulator (max head_dim = 128)
    
    for (uint d = 0; d < head_dim; ++d) {
        o_i[d] = 0.0f;
    }
    
    // Process key blocks
    const uint num_k_blocks = (seq_k + BLOCK_N - 1) / BLOCK_N;
    
    for (uint k_block_idx = 0; k_block_idx < num_k_blocks; ++k_block_idx) {
        const uint k_start = k_block_idx * BLOCK_N;
        const uint k_end = min(k_start + BLOCK_N, seq_k);
        
        // Load key and value blocks into shared memory
        for (uint k_in_block = 0; k_in_block < BLOCK_N; ++k_in_block) {
            const uint k_idx = k_start + k_in_block;
            if (k_idx < seq_k && dim_idx < head_dim) {
                const uint kv_offset = batch_idx * num_kv_heads * seq_k * head_dim + 
                                       kv_head_idx * seq_k * head_dim + 
                                       k_idx * head_dim + dim_idx;
                shared_k[k_in_block * head_dim + dim_idx] = keys[kv_offset];
                shared_v[k_in_block * head_dim + dim_idx] = values[kv_offset];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Compute attention scores for this block
        float m_block = -INFINITY;
        
        for (uint k_in_block = 0; k_in_block < min(BLOCK_N, k_end - k_start); ++k_in_block) {
            const uint k_idx = k_start + k_in_block;
            
            // Causal mask: skip future positions
            if (causal && k_idx > q_idx) {
                continue;
            }
            
            // Compute Q[q_idx] · K[k_idx]^T
            float score = 0.0f;
            for (uint d = 0; d < head_dim; ++d) {
                score += shared_q[row_in_block * head_dim + d] * 
                         shared_k[k_in_block * head_dim + d];
            }
            score *= scale;
            
            shared_scores[row_in_block * BLOCK_N + k_in_block] = score;
            m_block = max(m_block, score);
        }
        
        // Online softmax update
        float m_new = max(m_i, m_block);
        float l_block = 0.0f;
        
        // Compute exp(scores - m_new) and update output
        for (uint k_in_block = 0; k_in_block < min(BLOCK_N, k_end - k_start); ++k_in_block) {
            const uint k_idx = k_start + k_in_block;
            
            if (causal && k_idx > q_idx) {
                continue;
            }
            
            float score = shared_scores[row_in_block * BLOCK_N + k_in_block];
            float p = exp(score - m_new);
            l_block += p;
            
            // Update output accumulator
            for (uint d = 0; d < head_dim; ++d) {
                o_i[d] = o_i[d] * exp(m_i - m_new) + p * shared_v[k_in_block * head_dim + d];
            }
        }
        
        // Update running statistics
        l_i = l_i * exp(m_i - m_new) + l_block;
        m_i = m_new;
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Normalize and write output
    if (dim_idx < head_dim) {
        float result = o_i[dim_idx] / max(l_i, 1e-6f);
        const uint out_offset = batch_idx * num_heads * seq_q * head_dim + 
                                head_idx * seq_q * head_dim + 
                                q_idx * head_dim + dim_idx;
        output[out_offset] = result;
    }
}

// ============================================================================
// Paged Attention for KV Cache
// ============================================================================
// Supports block-based KV cache for PagedAttention

kernel void paged_attention(
    device const float* query [[buffer(0)]],          // [batch, num_heads, 1, head_dim]
    device const float* key_cache [[buffer(1)]],      // [num_blocks, num_kv_heads, block_size, head_dim]
    device const float* value_cache [[buffer(2)]],    // [num_blocks, num_kv_heads, block_size, head_dim]
    device const uint* block_tables [[buffer(3)]],    // [batch, max_num_blocks_per_seq]
    device const uint* context_lens [[buffer(4)]],    // [batch]
    device float* output [[buffer(5)]],               // [batch, num_heads, 1, head_dim]
    constant uint& batch_size [[buffer(6)]],
    constant uint& num_heads [[buffer(7)]],
    constant uint& num_kv_heads [[buffer(8)]],
    constant uint& head_dim [[buffer(9)]],
    constant uint& block_size [[buffer(10)]],
    constant uint& max_num_blocks_per_seq [[buffer(11)]],
    constant float& scale [[buffer(12)]],
    uint3 tid [[thread_position_in_grid]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]
) {
    const uint batch_idx = tgid.z;
    const uint head_idx = tgid.y;
    const uint dim_idx = tid.x;
    
    if (batch_idx >= batch_size || head_idx >= num_heads || dim_idx >= head_dim) {
        return;
    }
    
    const uint kv_head_idx = head_idx / (num_heads / num_kv_heads);
    const uint context_len = context_lens[batch_idx];
    const uint num_blocks = (context_len + block_size - 1) / block_size;
    
    // Load query
    const uint q_offset = batch_idx * num_heads * head_dim + head_idx * head_dim + dim_idx;
    float q_val = query[q_offset];
    
    // Shared memory
    threadgroup float shared_scores[512];
    
    // Online softmax variables
    float m_prev = -INFINITY;
    float l_prev = 0.0f;
    float o_acc = 0.0f;
    
    const uint local_id = lid.x;
    
    // Process each block in the block table
    for (uint block_idx = 0; block_idx < num_blocks; ++block_idx) {
        // Get physical block ID from block table
        const uint physical_block_id = block_tables[batch_idx * max_num_blocks_per_seq + block_idx];
        
        // Calculate tokens in this block
        const uint block_start = block_idx * block_size;
        const uint block_end = min(block_start + block_size, context_len);
        const uint tokens_in_block = block_end - block_start;
        
        // Compute attention scores for this block
        float block_max = -INFINITY;
        
        for (uint token_in_block = local_id; token_in_block < tokens_in_block; token_in_block += 32) {
            // Calculate offset into key cache
            const uint k_offset = physical_block_id * num_kv_heads * block_size * head_dim +
                                  kv_head_idx * block_size * head_dim +
                                  token_in_block * head_dim + dim_idx;
            
            // Compute Q·K^T
            float score = q_val * key_cache[k_offset];
            score = simd_sum(score);  // Reduce across head dimension
            score *= scale;
            
            if (local_id == 0) {
                shared_scores[token_in_block] = score;
                block_max = max(block_max, score);
            }
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Find block maximum (parallel reduction)
        block_max = simd_max(block_max);
        if (local_id == 0) {
            shared_scores[block_size] = block_max;  // Store at end of shared memory
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        float m_new = max(m_prev, shared_scores[block_size]);
        
        // Compute softmax and accumulate output
        float block_sum = 0.0f;
        
        for (uint token_in_block = local_id; token_in_block < tokens_in_block; token_in_block += 32) {
            float score = shared_scores[token_in_block];
            float p = exp(score - m_new);
            
            // Load value and accumulate
            const uint v_offset = physical_block_id * num_kv_heads * block_size * head_dim +
                                  kv_head_idx * block_size * head_dim +
                                  token_in_block * head_dim + dim_idx;
            
            // Correction factor for previous accumulator
            if (token_in_block == 0) {
                o_acc *= exp(m_prev - m_new);
            }
            
            o_acc += p * value_cache[v_offset];
            block_sum += p;
        }
        
        // Update running sum
        block_sum = simd_sum(block_sum);
        l_prev = l_prev * exp(m_prev - m_new) + block_sum;
        m_prev = m_new;
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Normalize and write output
    float result = o_acc / max(l_prev, 1e-6f);
    const uint out_offset = batch_idx * num_heads * head_dim + head_idx * head_dim + dim_idx;
    output[out_offset] = result;
}

// ============================================================================
// Fused RoPE + Attention
// ============================================================================
// Combines rotary position embedding with attention computation

kernel void fused_rope_attention(
    device const float* query [[buffer(0)]],     // [batch, num_heads, seq_len, head_dim]
    device const float* key [[buffer(1)]],       // [batch, num_kv_heads, seq_len, head_dim]
    device const float* value [[buffer(2)]],     // [batch, num_kv_heads, seq_len, head_dim]
    device const float* cos_cache [[buffer(3)]], // [max_seq, head_dim/2]
    device const float* sin_cache [[buffer(4)]], // [max_seq, head_dim/2]
    device const uint* positions [[buffer(5)]],  // [seq_len]
    device float* output [[buffer(6)]],          // [batch, num_heads, seq_len, head_dim]
    constant uint& batch_size [[buffer(7)]],
    constant uint& num_heads [[buffer(8)]],
    constant uint& num_kv_heads [[buffer(9)]],
    constant uint& head_dim [[buffer(10)]],
    constant uint& seq_len [[buffer(11)]],
    constant float& scale [[buffer(12)]],
    uint3 tid [[thread_position_in_grid]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]
) {
    const uint batch_idx = tgid.z;
    const uint head_idx = tgid.y;
    const uint q_idx = tid.x;
    
    if (batch_idx >= batch_size || head_idx >= num_heads || q_idx >= seq_len) {
        return;
    }
    
    const uint kv_head_idx = head_idx / (num_heads / num_kv_heads);
    const uint half_dim = head_dim / 2;
    
    // Shared memory for rotated Q and partial K
    threadgroup float shared_q_rotated[256];
    
    // Load and rotate query
    const uint q_offset = batch_idx * num_heads * seq_len * head_dim + 
                          head_idx * seq_len * head_dim + 
                          q_idx * head_dim;
    
    const uint pos = positions[q_idx];
    
    // Apply RoPE to query in shared memory
    for (uint d = lid.x; d < half_dim; d += 32) {
        float q1 = query[q_offset + d];
        float q2 = query[q_offset + half_dim + d];
        
        float cos_val = cos_cache[pos * half_dim + d];
        float sin_val = sin_cache[pos * half_dim + d];
        
        // RoPE rotation: [x1, x2] -> [x1*cos - x2*sin, x2*cos + x1*sin]
        shared_q_rotated[d] = q1 * cos_val - q2 * sin_val;
        shared_q_rotated[half_dim + d] = q2 * cos_val + q1 * sin_val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Online softmax variables
    float m_i = -INFINITY;
    float l_i = 0.0f;
    float o_i[128];  // Output accumulator
    
    for (uint d = 0; d < head_dim; ++d) {
        o_i[d] = 0.0f;
    }
    
    // Process all key positions
    for (uint k_idx = 0; k_idx < seq_len; ++k_idx) {
        // Causal mask
        if (k_idx > q_idx) {
            continue;
        }
        
        const uint k_offset = batch_idx * num_kv_heads * seq_len * head_dim + 
                              kv_head_idx * seq_len * head_dim + 
                              k_idx * head_dim;
        
        const uint k_pos = positions[k_idx];
        
        // Apply RoPE to key and compute score
        float score = 0.0f;
        for (uint d = 0; d < half_dim; ++d) {
            float k1 = key[k_offset + d];
            float k2 = key[k_offset + half_dim + d];
            
            float cos_val = cos_cache[k_pos * half_dim + d];
            float sin_val = sin_cache[k_pos * half_dim + d];
            
            float k1_rot = k1 * cos_val - k2 * sin_val;
            float k2_rot = k2 * cos_val + k1 * sin_val;
            
            score += shared_q_rotated[d] * k1_rot;
            score += shared_q_rotated[half_dim + d] * k2_rot;
        }
        score *= scale;
        
        // Online softmax update
        float m_new = max(m_i, score);
        float p = exp(score - m_new);
        float correction = exp(m_i - m_new);
        
        // Update output accumulator
        const uint v_offset = batch_idx * num_kv_heads * seq_len * head_dim + 
                              kv_head_idx * seq_len * head_dim + 
                              k_idx * head_dim;
        
        for (uint d = 0; d < head_dim; ++d) {
            o_i[d] = o_i[d] * correction + p * value[v_offset + d];
        }
        
        l_i = l_i * correction + p;
        m_i = m_new;
    }
    
    // Normalize and write output
    const uint out_offset = batch_idx * num_heads * seq_len * head_dim + 
                            head_idx * seq_len * head_dim + 
                            q_idx * head_dim;
    
    for (uint d = lid.x; d < head_dim; d += 32) {
        output[out_offset + d] = o_i[d] / max(l_i, 1e-6f);
    }
}
