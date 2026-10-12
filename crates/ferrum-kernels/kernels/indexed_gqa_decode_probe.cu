// Test-reachable mechanism probe; no provider or product route selects this ABI.
//
// One CTA shares a physical K/V block among six query heads of one KV head.
// Queries, online-softmax state, partial numerators and output remain FP32;
// the addressed combined K/V blocks retain the existing FP16 vLLM layout:
//   K: [kv_heads, 256 / 8, 16, 8], V: [kv_heads, 256, 16].
// A block-table entry is the actual device address of that block's K plane.
// Neither this pointer mapping nor either kernel grants resource authority.
// The caller validates and retains every current physical range, query/output,
// immutable unique row map and scratch allocation until completion/replay.
//
// Partial scratch: [group_rows, query_heads, max_partitions, 258] FP32, where
// fields 0..255 are unnormalized numerators, 256 is max logit and 257 is sum.
// Only live partitions are written/read. Output uses the original dense row.
// Probe envelope: head_dim=256, query_heads=6*kv_heads, lengths 1..2048.
// Compile through the ordinary PTX path without --use_fast_math.

#include <cuda_fp16.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stddef.h>

namespace {
constexpr int HEAD = 256;
constexpr int QUERY_GROUP = 6;
constexpr int BLOCK_TOKENS = 16;
constexpr int PARTITION_TOKENS = 128;
constexpr int MAX_SEQUENCE = 2048;
constexpr int MAX_PARTITIONS = MAX_SEQUENCE / PARTITION_TOKENS;
constexpr int PARTIAL_FIELDS = HEAD + 2;

__device__ __forceinline__ float warp_sum(float value) {
    for (int delta = 16; delta > 0; delta >>= 1) {
        value += __shfl_xor_sync(0xffffffffu, value, delta);
    }
    return value;
}

__device__ __forceinline__ float warp_max(float value) {
    for (int delta = 16; delta > 0; delta >>= 1) {
        value = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, delta));
    }
    return value;
}

__device__ __forceinline__ bool valid_envelope(
    int group_rows, int original_rows, int query_heads, int max_sequence_length,
    int max_partitions) {
    return group_rows > 0 && original_rows >= group_rows && query_heads > 0 &&
        max_sequence_length > 0 && max_sequence_length <= MAX_SEQUENCE &&
        max_partitions >=
            (max_sequence_length + PARTITION_TOKENS - 1) / PARTITION_TOKENS &&
        max_partitions <= MAX_PARTITIONS;
}

__device__ __forceinline__ size_t partial_offset(
    int group_row, int query_head, int partition, int query_heads,
    int max_partitions) {
    return ((static_cast<size_t>(group_row) * query_heads + query_head) *
        max_partitions + partition) * PARTIAL_FIELDS;
}
} // namespace

// Launch grid=(kv_heads,group_rows,max_partitions), block=(192,1,1).
// No dynamic shared memory. Current lengths and row map are read on each replay.
extern "C" __global__ void indexed_gqa_decode_f32_query_f16_kv_h256_gqa6_partial(
    const float* __restrict__ query,
    const uint64_t* __restrict__ block_addresses,
    const int* __restrict__ seq_lens,
    const int* __restrict__ row_map,
    float* __restrict__ partial,
    const int group_rows,
    const int original_rows,
    const int query_heads,
    const int kv_heads,
    const int table_stride,
    const int query_stride,
    const int max_sequence_length,
    const int max_partitions,
    const float scale
) {
    if (blockDim.x != QUERY_GROUP * 32 || blockDim.y != 1 || blockDim.z != 1 ||
        !valid_envelope(group_rows, original_rows, query_heads,
            max_sequence_length, max_partitions) ||
        kv_heads <= 0 || static_cast<int64_t>(kv_heads) * QUERY_GROUP != query_heads ||
        table_stride < (max_sequence_length + BLOCK_TOKENS - 1) / BLOCK_TOKENS ||
        query_stride < static_cast<int64_t>(query_heads) * HEAD ||
        !isfinite(scale) || scale <= 0.0f || query == nullptr ||
        block_addresses == nullptr || seq_lens == nullptr || row_map == nullptr ||
        partial == nullptr || blockIdx.x >= static_cast<unsigned>(kv_heads) ||
        blockIdx.y >= static_cast<unsigned>(group_rows) ||
        blockIdx.z >= static_cast<unsigned>(max_partitions)) {
        return;
    }
    const int group_row = static_cast<int>(blockIdx.y);
    const int original_row = row_map[group_row];
    if (original_row < 0 || original_row >= original_rows) {
        return;
    }
    const int length = seq_lens[original_row];
    if (length <= 0 || length > max_sequence_length) {
        return;
    }
    const int partition = static_cast<int>(blockIdx.z);
    const int begin = partition * PARTITION_TOKENS;
    if (begin >= length) {
        return;
    }
    const int end = min(begin + PARTITION_TOKENS, length);
    const int kv_head = static_cast<int>(blockIdx.x);
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int query_head = kv_head * QUERY_GROUP + warp;
    const float* q = query + static_cast<size_t>(original_row) * query_stride +
        static_cast<size_t>(query_head) * HEAD;
    const int dot_lane = lane % 8;
    const int token_group = lane / 8;
    // Four 8-thread groups compute four tokens concurrently for this head.
    // Each group retains the same query, partition and physical K/V owner.
    float q_registers[HEAD / 8];
    float numerator[HEAD / 32];
    #pragma unroll
    for (int item = 0; item < HEAD / 8; ++item) {
        q_registers[item] = q[dot_lane + item * 8];
    }
    #pragma unroll
    for (int item = 0; item < HEAD / 32; ++item) {
        numerator[item] = 0.0f;
    }
    float running_max = -FLT_MAX;
    float running_sum = 0.0f;
    // The two coalesced global layouts transpose differently into shared
    // memory. Pad K rows by one 8-element vector and V rows by one scalar to
    // avoid concentrating their respective stores on the same shared banks.
    __shared__ float tile_key[BLOCK_TOKENS][HEAD + 8];
    __shared__ float tile_value[BLOCK_TOKENS][HEAD + 1];
    __shared__ float logits[QUERY_GROUP][BLOCK_TOKENS];
    __shared__ float weights[QUERY_GROUP][BLOCK_TOKENS];
    const size_t head_block_elements = HEAD * BLOCK_TOKENS;
    const size_t key_plane_elements = static_cast<size_t>(kv_heads) * head_block_elements;
    const uint64_t* addresses = block_addresses +
        static_cast<size_t>(original_row) * table_stride;

    for (int token_start = begin; token_start < end; token_start += BLOCK_TOKENS) {
        const int live_tokens = min(BLOCK_TOKENS, end - token_start);
        const uint64_t address = addresses[token_start / BLOCK_TOKENS];
        // A null entry violates the caller's admitted-address contract. Returning
        // avoids a null access, but is not successful validation or an oracle.
        if (address == 0) {
            return;
        }
        const __half* key = reinterpret_cast<const __half*>(address) +
            static_cast<size_t>(kv_head) * head_block_elements;
        const __half* value = reinterpret_cast<const __half*>(address) +
            key_plane_elements + static_cast<size_t>(kv_head) * head_block_elements;
        // Coalesced loads follow each plane's physical layout, then publish
        // token-major FP32 tiles shared by all six query-head warps.
        for (int physical = threadIdx.x; physical < HEAD * BLOCK_TOKENS;
             physical += QUERY_GROUP * 32) {
            const int key_token = (physical / 8) % BLOCK_TOKENS;
            const int key_feature = (physical / (BLOCK_TOKENS * 8)) * 8 + physical % 8;
            tile_key[key_token][key_feature] = key_token < live_tokens
                ? __half2float(key[physical]) : 0.0f;
            const int value_token = physical % BLOCK_TOKENS;
            const int value_feature = physical / BLOCK_TOKENS;
            tile_value[value_token][value_feature] = value_token < live_tokens
                ? __half2float(value[physical]) : 0.0f;
        }
        __syncthreads();
        #pragma unroll
        for (int round = 0; round < BLOCK_TOKENS / 4; ++round) {
            const int token = round * 4 + token_group;
            // Independent chains keep the 32 features per thread from
            // becoming one long FMA dependency before the group reduction.
            float dot0 = 0.0f;
            float dot1 = 0.0f;
            float dot2 = 0.0f;
            float dot3 = 0.0f;
            #pragma unroll
            for (int item = 0; item < HEAD / 32; ++item) {
                dot0 = fmaf(q_registers[item],
                    tile_key[token][dot_lane + item * 8], dot0);
                dot1 = fmaf(q_registers[item + 8],
                    tile_key[token][dot_lane + (item + 8) * 8], dot1);
                dot2 = fmaf(q_registers[item + 16],
                    tile_key[token][dot_lane + (item + 16) * 8], dot2);
                dot3 = fmaf(q_registers[item + 24],
                    tile_key[token][dot_lane + (item + 24) * 8], dot3);
            }
            float dot = (dot0 + dot1) + (dot2 + dot3);
            #pragma unroll
            for (int delta = 4; delta > 0; delta >>= 1) {
                // All 32 lanes participate; width8 isolates the four token
                // groups while preserving the full active-warp mask.
                dot += __shfl_xor_sync(0xffffffffu, dot, delta, 8);
            }
            if (dot_lane == 0) {
                logits[warp][token] = token < live_tokens ? dot * scale : -FLT_MAX;
            }
        }
        __syncwarp();
        const float logit = lane < BLOCK_TOKENS ? logits[warp][lane] : -FLT_MAX;
        const float next_max = fmaxf(running_max, warp_max(logit));
        const float previous_weight = expf(running_max - next_max);
        const float weight = lane < live_tokens ? expf(logit - next_max) : 0.0f;
        const float tile_sum = warp_sum(weight);
        if (lane < BLOCK_TOKENS) {
            weights[warp][lane] = weight;
        }
        __syncwarp();
        #pragma unroll
        for (int item = 0; item < HEAD / 32; ++item) {
            float tile_numerator = 0.0f;
            #pragma unroll
            for (int token = 0; token < BLOCK_TOKENS; ++token) {
                tile_numerator = fmaf(weights[warp][token],
                    tile_value[token][lane + item * 32], tile_numerator);
            }
            numerator[item] = fmaf(previous_weight, numerator[item], tile_numerator);
        }
        running_sum = fmaf(previous_weight, running_sum, tile_sum);
        running_max = next_max;
        // All query warps finish consuming this tile before any thread replaces it.
        __syncthreads();
    }
    float* destination = partial + partial_offset(group_row, query_head, partition,
        query_heads, max_partitions);
    #pragma unroll
    for (int item = 0; item < HEAD / 32; ++item) {
        destination[lane + item * 32] = numerator[item];
    }
    if (lane == 0) {
        destination[HEAD] = running_max;
        destination[HEAD + 1] = running_sum;
    }
}

// Launch grid=(query_heads,group_rows,1), block=(128,1,1).
// Merge FP32 partition numerators into dense original-row output, with FP32
// max/sum/weights and no FP16 output cast. No dynamic shared memory.
extern "C" __global__ void indexed_gqa_decode_f32_query_f16_kv_h256_gqa6_reduce(
    const float* __restrict__ partial,
    const int* __restrict__ seq_lens,
    const int* __restrict__ row_map,
    float* __restrict__ out,
    const int group_rows,
    const int original_rows,
    const int query_heads,
    const int max_sequence_length,
    const int max_partitions
) {
    if (blockDim.x != 128 || blockDim.y != 1 || blockDim.z != 1 ||
        !valid_envelope(group_rows, original_rows, query_heads,
            max_sequence_length, max_partitions) ||
        query_heads % QUERY_GROUP != 0 || partial == nullptr || seq_lens == nullptr ||
        row_map == nullptr || out == nullptr ||
        blockIdx.x >= static_cast<unsigned>(query_heads) ||
        blockIdx.y >= static_cast<unsigned>(group_rows) || blockIdx.z != 0) {
        return;
    }
    const int group_row = static_cast<int>(blockIdx.y);
    const int original_row = row_map[group_row];
    if (original_row < 0 || original_row >= original_rows) {
        return;
    }
    const int length = seq_lens[original_row];
    if (length <= 0 || length > max_sequence_length) {
        return;
    }
    const int live_partitions = (length + PARTITION_TOKENS - 1) / PARTITION_TOKENS;
    const int query_head = static_cast<int>(blockIdx.x);
    __shared__ float factors[MAX_PARTITIONS];
    __shared__ float denominator;
    if (threadIdx.x == 0) {
        float maximum = -FLT_MAX;
        for (int partition = 0; partition < live_partitions; ++partition) {
            const float* source = partial + partial_offset(group_row, query_head,
                partition, query_heads, max_partitions);
            maximum = fmaxf(maximum, source[HEAD]);
        }
        float sum = 0.0f;
        for (int partition = 0; partition < live_partitions; ++partition) {
            const float* source = partial + partial_offset(group_row, query_head,
                partition, query_heads, max_partitions);
            const float factor = expf(source[HEAD] - maximum);
            factors[partition] = factor;
            sum = fmaf(factor, source[HEAD + 1], sum);
        }
        denominator = sum;
    }
    __syncthreads();
    for (int feature = threadIdx.x; feature < HEAD; feature += 128) {
        float numerator = 0.0f;
        for (int partition = 0; partition < live_partitions; ++partition) {
            const float* source = partial + partial_offset(group_row, query_head,
                partition, query_heads, max_partitions);
            numerator = fmaf(factors[partition], source[feature], numerator);
        }
        out[(static_cast<size_t>(original_row) * query_heads + query_head) *
            HEAD + feature] = numerator / denominator;
    }
}

// Explicit control-only conversion and matched FP32 final gate. These helpers
// let the Rust fixture time conversion/gating inside both complete bundles;
// they do not assert the old F16 attention has the candidate's precision.
extern "C" __global__ void indexed_gqa_probe_cast_f32_to_f16(
    const float* __restrict__ input,
    __half* __restrict__ output,
    const int elements
) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (elements > 0 && index < static_cast<size_t>(elements)) {
        output[index] = __float2half_rn(input[index]);
    }
}

extern "C" __global__ void indexed_gqa_probe_gate_f32(
    const float* __restrict__ context,
    const float* __restrict__ gate,
    float* __restrict__ output,
    const int elements
) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (elements > 0 && index < static_cast<size_t>(elements)) {
        output[index] = context[index] / (1.0f + expf(-gate[index]));
    }
}

extern "C" __global__ void indexed_gqa_probe_gate_f16_to_f32(
    const __half* __restrict__ context,
    const float* __restrict__ gate,
    float* __restrict__ output,
    const int elements
) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (elements > 0 && index < static_cast<size_t>(elements)) {
        output[index] = __half2float(context[index]) / (1.0f + expf(-gate[index]));
    }
}
