//! RoPE (Rotary Position Embedding) optimized for Apple GPU
//! Applies rotational position encoding to query and key vectors

#include "definitions.metal"

// RoPE kernel configuration
constant uint ROPE_HEAD_DIM [[function_constant(0)]];
constant float ROPE_THETA [[function_constant(1)]];

// RoPE forward pass kernel
kernel void rope_forward(
    device const float* qkv_input [[buffer(0)]],      // [seq_len, (num_heads + 2*num_kv_heads) * head_dim]
    device const uint* positions [[buffer(1)]],       // [seq_len] - actual token positions
    device float* queries_output [[buffer(2)]],       // [num_heads, seq_len, head_dim]
    device float* keys_output [[buffer(3)]],          // [num_kv_heads, seq_len, head_dim]
    constant uint& seq_len [[buffer(4)]],
    constant uint& num_heads [[buffer(5)]],
    constant uint& num_kv_heads [[buffer(6)]],
    constant uint& head_dim [[buffer(7)]],
    uint3 tid [[thread_position_in_grid]],
    uint3 tgid [[threadgroup_position_in_grid]]
) {
    const uint seq_idx = tgid.x;
    const uint head_idx = tgid.y;
    const uint dim_pair = tid.z; // Process pairs of dimensions
    
    if (seq_idx >= seq_len || dim_pair >= head_dim / 2) {
        return;
    }
    
    const uint position = positions[seq_idx];
    const float theta = ROPE_THETA;
    
    // Compute rotation frequency for this dimension pair
    const float freq = 1.0f / pow(theta, float(dim_pair * 2) / float(head_dim));
    const float angle = float(position) * freq;
    const float cos_val = cos(angle);
    const float sin_val = sin(angle);
    
    // Process queries
    if (head_idx < num_heads) {
        const uint q_input_offset = seq_idx * (num_heads + 2 * num_kv_heads) * head_dim + head_idx * head_dim;
        const uint q_output_offset = head_idx * seq_len * head_dim + seq_idx * head_dim;
        
        const uint dim_0 = dim_pair * 2;
        const uint dim_1 = dim_pair * 2 + 1;
        
        if (dim_1 < head_dim) {
            const float q0 = qkv_input[q_input_offset + dim_0];
            const float q1 = qkv_input[q_input_offset + dim_1];
            
            // Apply rotation
            queries_output[q_output_offset + dim_0] = q0 * cos_val - q1 * sin_val;
            queries_output[q_output_offset + dim_1] = q0 * sin_val + q1 * cos_val;
        }
    }
    
    // Process keys (only for the corresponding kv_head to avoid redundant work)
    const uint kv_head_for_this_head = head_idx / (num_heads / num_kv_heads);
    if (head_idx % (num_heads / num_kv_heads) == 0 && kv_head_for_this_head < num_kv_heads) {
        const uint k_input_offset = seq_idx * (num_heads + 2 * num_kv_heads) * head_dim + 
                                   (num_heads + kv_head_for_this_head) * head_dim;
        const uint k_output_offset = kv_head_for_this_head * seq_len * head_dim + seq_idx * head_dim;
        
        const uint dim_0 = dim_pair * 2;
        const uint dim_1 = dim_pair * 2 + 1;
        
        if (dim_1 < head_dim) {
            const float k0 = qkv_input[k_input_offset + dim_0];
            const float k1 = qkv_input[k_input_offset + dim_1];
            
            // Apply rotation
            keys_output[k_output_offset + dim_0] = k0 * cos_val - k1 * sin_val;
            keys_output[k_output_offset + dim_1] = k0 * sin_val + k1 * cos_val;
        }
    }
}
