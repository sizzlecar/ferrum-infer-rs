//! RMS Normalization kernel for Apple GPU
//!
//! Implements Root Mean Square Layer Normalization as used in LLaMA and other models.
//! RMSNorm(x) = x * rsqrt(mean(x^2) + eps) * weight
//!
//! Optimized for Apple Silicon unified memory architecture with:
//! - Efficient SIMD reductions for variance computation
//! - Shared memory for intermediate results
//! - Fused multiply-accumulate operations

#include "definitions.metal"

// ============================================================================
// RMS Norm - Single Vector (for decode phase)
// ============================================================================
// Processes one hidden state vector at a time

kernel void rms_norm_single(
    device const float* input [[buffer(0)]],     // [hidden_size]
    device const float* weight [[buffer(1)]],    // [hidden_size]
    device float* output [[buffer(2)]],          // [hidden_size]
    constant uint& hidden_size [[buffer(3)]],
    constant float& epsilon [[buffer(4)]],
    uint tid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    // Shared memory for reduction
    threadgroup float shared_sum[32];
    
    const uint threads_per_group = 256;
    const uint elements_per_thread = (hidden_size + threads_per_group - 1) / threads_per_group;
    
    // Step 1: Compute sum of squares (x^2)
    float local_sum = 0.0f;
    
    for (uint i = 0; i < elements_per_thread; ++i) {
        uint idx = lid + i * threads_per_group;
        if (idx < hidden_size) {
            float val = input[idx];
            local_sum += val * val;
        }
    }
    
    // SIMD reduction within warp
    local_sum = simd_sum(local_sum);
    
    // Store warp results to shared memory
    if (simd_lane_id == 0) {
        shared_sum[simd_group_id] = local_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Final reduction across warps (done by first warp)
    if (simd_group_id == 0) {
        float warp_sum = (simd_lane_id < 8) ? shared_sum[simd_lane_id] : 0.0f;
        warp_sum = simd_sum(warp_sum);
        
        if (simd_lane_id == 0) {
            // Compute RMS: rsqrt(mean(x^2) + epsilon)
            float mean_sq = warp_sum / float(hidden_size);
            shared_sum[0] = rsqrt(mean_sq + epsilon);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Step 2: Apply normalization with weight
    float rms_scale = shared_sum[0];
    
    for (uint i = 0; i < elements_per_thread; ++i) {
        uint idx = lid + i * threads_per_group;
        if (idx < hidden_size) {
            output[idx] = input[idx] * rms_scale * weight[idx];
        }
    }
}

// ============================================================================
// RMS Norm - Batched (for prefill phase)
// ============================================================================
// Processes multiple hidden states in parallel

kernel void rms_norm_batched(
    device const float* input [[buffer(0)]],     // [batch_size, hidden_size]
    device const float* weight [[buffer(1)]],    // [hidden_size]
    device float* output [[buffer(2)]],          // [batch_size, hidden_size]
    constant uint& batch_size [[buffer(3)]],
    constant uint& hidden_size [[buffer(4)]],
    constant float& epsilon [[buffer(5)]],
    uint3 tid [[thread_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    const uint batch_idx = tgid.y;
    
    if (batch_idx >= batch_size) {
        return;
    }
    
    // Shared memory for reduction
    threadgroup float shared_sum[32];
    
    const uint threads_per_group = 256;
    const uint elements_per_thread = (hidden_size + threads_per_group - 1) / threads_per_group;
    
    // Input/output offset for this batch element
    const uint offset = batch_idx * hidden_size;
    
    // Step 1: Compute sum of squares (x^2)
    float local_sum = 0.0f;
    
    for (uint i = 0; i < elements_per_thread; ++i) {
        uint idx = lid.x + i * threads_per_group;
        if (idx < hidden_size) {
            float val = input[offset + idx];
            local_sum += val * val;
        }
    }
    
    // SIMD reduction within warp
    local_sum = simd_sum(local_sum);
    
    // Store warp results to shared memory
    if (simd_lane_id == 0) {
        shared_sum[simd_group_id] = local_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Final reduction across warps
    if (simd_group_id == 0) {
        float warp_sum = (simd_lane_id < 8) ? shared_sum[simd_lane_id] : 0.0f;
        warp_sum = simd_sum(warp_sum);
        
        if (simd_lane_id == 0) {
            float mean_sq = warp_sum / float(hidden_size);
            shared_sum[0] = rsqrt(mean_sq + epsilon);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Step 2: Apply normalization with weight
    float rms_scale = shared_sum[0];
    
    for (uint i = 0; i < elements_per_thread; ++i) {
        uint idx = lid.x + i * threads_per_group;
        if (idx < hidden_size) {
            output[offset + idx] = input[offset + idx] * rms_scale * weight[idx];
        }
    }
}

// ============================================================================
// RMS Norm - Half Precision (for memory efficiency)
// ============================================================================

kernel void rms_norm_half(
    device const half* input [[buffer(0)]],      // [batch_size, hidden_size]
    device const half* weight [[buffer(1)]],     // [hidden_size]
    device half* output [[buffer(2)]],           // [batch_size, hidden_size]
    constant uint& batch_size [[buffer(3)]],
    constant uint& hidden_size [[buffer(4)]],
    constant float& epsilon [[buffer(5)]],
    uint3 tid [[thread_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    const uint batch_idx = tgid.y;
    
    if (batch_idx >= batch_size) {
        return;
    }
    
    threadgroup float shared_sum[32];
    
    const uint threads_per_group = 256;
    const uint elements_per_thread = (hidden_size + threads_per_group - 1) / threads_per_group;
    const uint offset = batch_idx * hidden_size;
    
    // Compute sum of squares (accumulate in float for precision)
    float local_sum = 0.0f;
    
    for (uint i = 0; i < elements_per_thread; ++i) {
        uint idx = lid.x + i * threads_per_group;
        if (idx < hidden_size) {
            float val = float(input[offset + idx]);
            local_sum += val * val;
        }
    }
    
    local_sum = simd_sum(local_sum);
    
    if (simd_lane_id == 0) {
        shared_sum[simd_group_id] = local_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (simd_group_id == 0) {
        float warp_sum = (simd_lane_id < 8) ? shared_sum[simd_lane_id] : 0.0f;
        warp_sum = simd_sum(warp_sum);
        
        if (simd_lane_id == 0) {
            float mean_sq = warp_sum / float(hidden_size);
            shared_sum[0] = rsqrt(mean_sq + epsilon);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    float rms_scale = shared_sum[0];
    
    for (uint i = 0; i < elements_per_thread; ++i) {
        uint idx = lid.x + i * threads_per_group;
        if (idx < hidden_size) {
            float val = float(input[offset + idx]) * rms_scale * float(weight[idx]);
            output[offset + idx] = half(val);
        }
    }
}

// ============================================================================
// Fused RMS Norm + Residual Add
// ============================================================================
// Combines: output = RMSNorm(input + residual) * weight

kernel void rms_norm_residual(
    device const float* input [[buffer(0)]],     // [batch_size, hidden_size]
    device const float* residual [[buffer(1)]],  // [batch_size, hidden_size]
    device const float* weight [[buffer(2)]],    // [hidden_size]
    device float* output [[buffer(3)]],          // [batch_size, hidden_size]
    constant uint& batch_size [[buffer(4)]],
    constant uint& hidden_size [[buffer(5)]],
    constant float& epsilon [[buffer(6)]],
    uint3 tid [[thread_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    const uint batch_idx = tgid.y;
    
    if (batch_idx >= batch_size) {
        return;
    }
    
    threadgroup float shared_sum[32];
    threadgroup float shared_values[4096];  // Temp storage for input+residual
    
    const uint threads_per_group = 256;
    const uint elements_per_thread = (hidden_size + threads_per_group - 1) / threads_per_group;
    const uint offset = batch_idx * hidden_size;
    
    // Step 1: Compute input + residual and sum of squares
    float local_sum = 0.0f;
    
    for (uint i = 0; i < elements_per_thread; ++i) {
        uint idx = lid.x + i * threads_per_group;
        if (idx < hidden_size) {
            float val = input[offset + idx] + residual[offset + idx];
            shared_values[idx] = val;  // Store for reuse
            local_sum += val * val;
        }
    }
    
    local_sum = simd_sum(local_sum);
    
    if (simd_lane_id == 0) {
        shared_sum[simd_group_id] = local_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (simd_group_id == 0) {
        float warp_sum = (simd_lane_id < 8) ? shared_sum[simd_lane_id] : 0.0f;
        warp_sum = simd_sum(warp_sum);
        
        if (simd_lane_id == 0) {
            float mean_sq = warp_sum / float(hidden_size);
            shared_sum[0] = rsqrt(mean_sq + epsilon);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Step 2: Apply normalization with weight
    float rms_scale = shared_sum[0];
    
    for (uint i = 0; i < elements_per_thread; ++i) {
        uint idx = lid.x + i * threads_per_group;
        if (idx < hidden_size) {
            output[offset + idx] = shared_values[idx] * rms_scale * weight[idx];
        }
    }
}

// ============================================================================
// In-place RMS Norm (for memory efficiency)
// ============================================================================

kernel void rms_norm_inplace(
    device float* data [[buffer(0)]],            // [batch_size, hidden_size] - in/out
    device const float* weight [[buffer(1)]],    // [hidden_size]
    constant uint& batch_size [[buffer(2)]],
    constant uint& hidden_size [[buffer(3)]],
    constant float& epsilon [[buffer(4)]],
    uint3 tid [[thread_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    const uint batch_idx = tgid.y;
    
    if (batch_idx >= batch_size) {
        return;
    }
    
    threadgroup float shared_sum[32];
    
    const uint threads_per_group = 256;
    const uint elements_per_thread = (hidden_size + threads_per_group - 1) / threads_per_group;
    const uint offset = batch_idx * hidden_size;
    
    // Compute sum of squares
    float local_sum = 0.0f;
    
    for (uint i = 0; i < elements_per_thread; ++i) {
        uint idx = lid.x + i * threads_per_group;
        if (idx < hidden_size) {
            float val = data[offset + idx];
            local_sum += val * val;
        }
    }
    
    local_sum = simd_sum(local_sum);
    
    if (simd_lane_id == 0) {
        shared_sum[simd_group_id] = local_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (simd_group_id == 0) {
        float warp_sum = (simd_lane_id < 8) ? shared_sum[simd_lane_id] : 0.0f;
        warp_sum = simd_sum(warp_sum);
        
        if (simd_lane_id == 0) {
            float mean_sq = warp_sum / float(hidden_size);
            shared_sum[0] = rsqrt(mean_sq + epsilon);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Apply normalization in-place
    float rms_scale = shared_sum[0];
    
    for (uint i = 0; i < elements_per_thread; ++i) {
        uint idx = lid.x + i * threads_per_group;
        if (idx < hidden_size) {
            data[offset + idx] = data[offset + idx] * rms_scale * weight[idx];
        }
    }
}

