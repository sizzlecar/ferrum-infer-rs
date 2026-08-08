#include <metal_stdlib>
using namespace metal;

struct MoeRouteParams {
    uint tokens;
    uint expert_count;
    uint experts_per_token;
    uint normalize_topk;
};

kernel void vnext_moe_route_topk_f16(
    device const half *logits [[buffer(0)]],
    device int *route_ids [[buffer(1)]],
    device float *route_weights [[buffer(2)]],
    constant MoeRouteParams &p [[buffer(3)]],
    threadgroup float *scratch [[threadgroup(0)]],
    uint3 group [[threadgroup_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]]) {
    const uint token = group.x;
    if (token >= p.tokens) {
        return;
    }

    threadgroup float *probabilities = scratch;
    threadgroup float *selected_weights = probabilities + p.expert_count;
    threadgroup int *selected_ids =
        reinterpret_cast<threadgroup int *>(selected_weights + p.experts_per_token);
    threadgroup float *selected_sum =
        reinterpret_cast<threadgroup float *>(selected_ids + p.experts_per_token);

    float local_max = -INFINITY;
    for (uint expert = lane; expert < p.expert_count; expert += 32) {
        const float raw = float(logits[token * p.expert_count + expert]);
        const float value = isfinite(raw)
            ? raw
            : (raw > 0.0f ? 65504.0f : -65504.0f);
        probabilities[expert] = value;
        local_max = max(local_max, value);
    }
    const float row_max = simd_max(local_max);

    float local_sum = 0.0f;
    for (uint expert = lane; expert < p.expert_count; expert += 32) {
        const float value = exp(probabilities[expert] - row_max);
        probabilities[expert] = value;
        local_sum += value;
    }
    const float inverse_sum = 1.0f / max(simd_sum(local_sum), 0x1.0p-14f);
    for (uint expert = lane; expert < p.expert_count; expert += 32) {
        probabilities[expert] *= inverse_sum;
    }
    if (lane == 0) {
        selected_sum[0] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint slot = 0; slot < p.experts_per_token; ++slot) {
        float local_best = -INFINITY;
        int local_id = -1;
        for (uint expert = lane; expert < p.expert_count; expert += 32) {
            const float value = probabilities[expert];
            if (value > local_best) {
                local_best = value;
                local_id = int(expert);
            }
        }
        const float best = simd_max(local_best);
        const int candidate = local_best == best ? local_id : 0x7fffffff;
        const int selected = simd_min(candidate);
        if (lane == 0) {
            selected_weights[slot] = best;
            selected_ids[slot] = selected;
            selected_sum[0] += best;
            probabilities[selected] = -INFINITY;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (uint(lane) < p.experts_per_token) {
        const float scale = p.normalize_topk == 0
            ? 1.0f
            : 1.0f / max(selected_sum[0], 0x1.0p-14f);
        const uint output = token * p.experts_per_token + uint(lane);
        route_ids[output] = selected_ids[lane];
        route_weights[output] = selected_weights[lane] * scale;
    }
}

#define VNEXT_QK_K 256
#define VNEXT_MOE_ROWS_PER_SIMDGROUP 2
#define VNEXT_MOE_SIMDGROUPS 2
#define VNEXT_UNROLL _Pragma("clang loop unroll(full)")

struct VNextBlockQ4K {
    half d;
    half dmin;
    uchar scales[12];
    uchar qs[VNEXT_QK_K / 2];
};

struct MoeQ4GateUpParams {
    uint output_features;
    uint input_features;
    uint row_stride_bytes;
    uint expert_stride_bytes;
    uint expert_count;
    uint experts_per_token;
    uint pair_count;
};

kernel void vnext_moe_q4k_gate_up_silu_f16(
    device const VNextBlockQ4K *gate_weights [[buffer(0)]],
    device const VNextBlockQ4K *up_weights [[buffer(1)]],
    device const half *input [[buffer(2)]],
    device const int *route_ids [[buffer(3)]],
    device half *activation [[buffer(4)]],
    constant MoeQ4GateUpParams &p [[buffer(5)]],
    uint3 group [[threadgroup_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort simdgroup [[simdgroup_index_in_threadgroup]]) {
    const uint pair = group.z;
    if (pair >= p.pair_count) {
        return;
    }
    const int expert = route_ids[pair];
    if (expert < 0 || uint(expert) >= p.expert_count) {
        return;
    }
    const uint token = pair / p.experts_per_token;

    constexpr uint16_t scale_mask = 0x3f3f;
    constexpr uint16_t low_mask = 0x0f0f;
    constexpr uint16_t high_mask = 0xc0c0;
    const short block_lane = lane / 8;
    const short item = lane % 8;
    const short quant_group = item / 4;
    const short quant_item = item % 4;
    const uint blocks_per_row = p.input_features / VNEXT_QK_K;
    const uint first_row =
        (group.x * VNEXT_MOE_SIMDGROUPS + simdgroup) * VNEXT_MOE_ROWS_PER_SIMDGROUP;
    if (first_row >= p.output_features) {
        return;
    }

    device const VNextBlockQ4K *gate =
        reinterpret_cast<device const VNextBlockQ4K *>(
            reinterpret_cast<device const char *>(gate_weights)
            + uint(expert) * p.expert_stride_bytes
            + first_row * p.row_stride_bytes);
    device const VNextBlockQ4K *up =
        reinterpret_cast<device const VNextBlockQ4K *>(
            reinterpret_cast<device const char *>(up_weights)
            + uint(expert) * p.expert_stride_bytes
            + first_row * p.row_stride_bytes);
    device const half *input_row = input + token * p.input_features;
    device const half *input_block =
        input_row + block_lane * VNEXT_QK_K + 64 * quant_group + 8 * quant_item;

    float low[16];
    float high[16];
    float gate_sum[VNEXT_MOE_ROWS_PER_SIMDGROUP] = {0.0f};
    float up_sum[VNEXT_MOE_ROWS_PER_SIMDGROUP] = {0.0f};
    uint16_t unpacked_scales[4];
    thread const uint8_t *scales8 =
        reinterpret_cast<thread const uint8_t *>(unpacked_scales);

    for (uint block = uint(block_lane); block < blocks_per_row; block += 4) {
        float4 input_sums = 0.0f;
        for (short index = 0; index < 8; ++index) {
            low[index] = float(input_block[index]);
            input_sums[0] += low[index];
            low[index + 8] = float(input_block[index + 32]);
            input_sums[1] += low[index + 8];
            high[index] = float(input_block[index + 128]);
            input_sums[2] += high[index];
            high[index + 8] = float(input_block[index + 160]);
            input_sums[3] += high[index + 8];
        }

        device const uint16_t *gate_scales =
            reinterpret_cast<device const uint16_t *>(gate[block].scales) + quant_group;
        device const uint16_t *gate_quants =
            reinterpret_cast<device const uint16_t *>(gate[block].qs)
            + 16 * quant_group + 4 * quant_item;
        device const half *gate_delta = &gate[block].d;
        device const uint16_t *up_scales =
            reinterpret_cast<device const uint16_t *>(up[block].scales) + quant_group;
        device const uint16_t *up_quants =
            reinterpret_cast<device const uint16_t *>(up[block].qs)
            + 16 * quant_group + 4 * quant_item;
        device const half *up_delta = &up[block].d;

        for (short row = 0; row < VNEXT_MOE_ROWS_PER_SIMDGROUP; ++row) {
            unpacked_scales[0] = gate_scales[0] & scale_mask;
            unpacked_scales[1] = gate_scales[2] & scale_mask;
            unpacked_scales[2] = ((gate_scales[4] >> 0) & low_mask)
                | ((gate_scales[0] & high_mask) >> 2);
            unpacked_scales[3] = ((gate_scales[4] >> 4) & low_mask)
                | ((gate_scales[2] & high_mask) >> 2);
            device const uint16_t *gate_high_quants = gate_quants + 32;
            float4 gate_acc_low = 0.0f;
            float4 gate_acc_high = 0.0f;
            VNEXT_UNROLL for (short index = 0; index < 4; ++index) {
                gate_acc_low[0] += low[2 * index] * (gate_quants[index] & 0x000f);
                gate_acc_low[1] += low[2 * index + 1] * (gate_quants[index] & 0x0f00);
                gate_acc_low[2] += low[2 * index + 8] * (gate_quants[index] & 0x00f0);
                gate_acc_low[3] += low[2 * index + 9] * (gate_quants[index] & 0xf000);
                gate_acc_high[0] += high[2 * index] * (gate_high_quants[index] & 0x000f);
                gate_acc_high[1] += high[2 * index + 1] * (gate_high_quants[index] & 0x0f00);
                gate_acc_high[2] += high[2 * index + 8] * (gate_high_quants[index] & 0x00f0);
                gate_acc_high[3] += high[2 * index + 9] * (gate_high_quants[index] & 0xf000);
            }
            gate_sum[row] += float(gate_delta[0]) * (
                (gate_acc_low[0] + gate_acc_low[1] / 256.0f) * scales8[0]
                + (gate_acc_low[2] + gate_acc_low[3] / 256.0f) * scales8[1] / 16.0f
                + (gate_acc_high[0] + gate_acc_high[1] / 256.0f) * scales8[4]
                + (gate_acc_high[2] + gate_acc_high[3] / 256.0f) * scales8[5] / 16.0f)
                - float(gate_delta[1]) * (
                    input_sums[0] * scales8[2] + input_sums[1] * scales8[3]
                    + input_sums[2] * scales8[6] + input_sums[3] * scales8[7]);

            unpacked_scales[0] = up_scales[0] & scale_mask;
            unpacked_scales[1] = up_scales[2] & scale_mask;
            unpacked_scales[2] = ((up_scales[4] >> 0) & low_mask)
                | ((up_scales[0] & high_mask) >> 2);
            unpacked_scales[3] = ((up_scales[4] >> 4) & low_mask)
                | ((up_scales[2] & high_mask) >> 2);
            device const uint16_t *up_high_quants = up_quants + 32;
            float4 up_acc_low = 0.0f;
            float4 up_acc_high = 0.0f;
            VNEXT_UNROLL for (short index = 0; index < 4; ++index) {
                up_acc_low[0] += low[2 * index] * (up_quants[index] & 0x000f);
                up_acc_low[1] += low[2 * index + 1] * (up_quants[index] & 0x0f00);
                up_acc_low[2] += low[2 * index + 8] * (up_quants[index] & 0x00f0);
                up_acc_low[3] += low[2 * index + 9] * (up_quants[index] & 0xf000);
                up_acc_high[0] += high[2 * index] * (up_high_quants[index] & 0x000f);
                up_acc_high[1] += high[2 * index + 1] * (up_high_quants[index] & 0x0f00);
                up_acc_high[2] += high[2 * index + 8] * (up_high_quants[index] & 0x00f0);
                up_acc_high[3] += high[2 * index + 9] * (up_high_quants[index] & 0xf000);
            }
            up_sum[row] += float(up_delta[0]) * (
                (up_acc_low[0] + up_acc_low[1] / 256.0f) * scales8[0]
                + (up_acc_low[2] + up_acc_low[3] / 256.0f) * scales8[1] / 16.0f
                + (up_acc_high[0] + up_acc_high[1] / 256.0f) * scales8[4]
                + (up_acc_high[2] + up_acc_high[3] / 256.0f) * scales8[5] / 16.0f)
                - float(up_delta[1]) * (
                    input_sums[0] * scales8[2] + input_sums[1] * scales8[3]
                    + input_sums[2] * scales8[6] + input_sums[3] * scales8[7]);

            gate_quants += p.row_stride_bytes / 2;
            gate_scales += p.row_stride_bytes / 2;
            gate_delta += p.row_stride_bytes / 2;
            up_quants += p.row_stride_bytes / 2;
            up_scales += p.row_stride_bytes / 2;
            up_delta += p.row_stride_bytes / 2;
        }
        input_block += 4 * VNEXT_QK_K;
    }

    device half *output = activation + pair * p.output_features;
    for (short row = 0;
         row < VNEXT_MOE_ROWS_PER_SIMDGROUP && first_row + uint(row) < p.output_features;
         ++row) {
        const float gate_value = simd_sum(gate_sum[row]);
        const float up_value = simd_sum(up_sum[row]);
        if (lane == 0) {
            output[first_row + uint(row)] = half(
                (gate_value / (1.0f + exp(-gate_value))) * up_value);
        }
    }
}

struct MoeQ4DownParams {
    uint output_features;
    uint input_features;
    uint row_stride_bytes;
    uint expert_stride_bytes;
    uint expert_count;
    uint experts_per_token;
    uint pair_count;
};

kernel void vnext_moe_q4k_down_f16(
    device const VNextBlockQ4K *weights [[buffer(0)]],
    device const half *activation [[buffer(1)]],
    device const int *route_ids [[buffer(2)]],
    device half *down_slots [[buffer(3)]],
    constant MoeQ4DownParams &p [[buffer(4)]],
    uint3 group [[threadgroup_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort simdgroup [[simdgroup_index_in_threadgroup]]) {
    const uint pair = group.z;
    if (pair >= p.pair_count) {
        return;
    }
    const int expert = route_ids[pair];
    if (expert < 0 || uint(expert) >= p.expert_count) {
        return;
    }

    constexpr uint16_t scale_mask = 0x3f3f;
    constexpr uint16_t low_mask = 0x0f0f;
    constexpr uint16_t high_mask = 0xc0c0;
    const short block_lane = lane / 8;
    const short item = lane % 8;
    const short quant_group = item / 4;
    const short quant_item = item % 4;
    const uint blocks_per_row = p.input_features / VNEXT_QK_K;
    const uint first_row =
        (group.x * VNEXT_MOE_SIMDGROUPS + simdgroup) * VNEXT_MOE_ROWS_PER_SIMDGROUP;
    if (first_row >= p.output_features) {
        return;
    }

    device const VNextBlockQ4K *expert_weights =
        reinterpret_cast<device const VNextBlockQ4K *>(
            reinterpret_cast<device const char *>(weights)
            + uint(expert) * p.expert_stride_bytes
            + first_row * p.row_stride_bytes);
    device const half *input = activation + pair * p.input_features;
    device const half *input_block =
        input + block_lane * VNEXT_QK_K + 64 * quant_group + 8 * quant_item;
    float low[16];
    float high[16];
    float sums[VNEXT_MOE_ROWS_PER_SIMDGROUP] = {0.0f};
    uint16_t unpacked_scales[4];
    thread const uint8_t *scales8 =
        reinterpret_cast<thread const uint8_t *>(unpacked_scales);

    for (uint block = uint(block_lane); block < blocks_per_row; block += 4) {
        float4 input_sums = 0.0f;
        for (short index = 0; index < 8; ++index) {
            low[index] = float(input_block[index]);
            input_sums[0] += low[index];
            low[index + 8] = float(input_block[index + 32]);
            input_sums[1] += low[index + 8];
            high[index] = float(input_block[index + 128]);
            input_sums[2] += high[index];
            high[index + 8] = float(input_block[index + 160]);
            input_sums[3] += high[index + 8];
        }
        device const uint16_t *scales =
            reinterpret_cast<device const uint16_t *>(expert_weights[block].scales)
            + quant_group;
        device const uint16_t *quants =
            reinterpret_cast<device const uint16_t *>(expert_weights[block].qs)
            + 16 * quant_group + 4 * quant_item;
        device const half *delta = &expert_weights[block].d;
        for (short row = 0; row < VNEXT_MOE_ROWS_PER_SIMDGROUP; ++row) {
            unpacked_scales[0] = scales[0] & scale_mask;
            unpacked_scales[1] = scales[2] & scale_mask;
            unpacked_scales[2] = ((scales[4] >> 0) & low_mask)
                | ((scales[0] & high_mask) >> 2);
            unpacked_scales[3] = ((scales[4] >> 4) & low_mask)
                | ((scales[2] & high_mask) >> 2);
            device const uint16_t *high_quants = quants + 32;
            float4 acc_low = 0.0f;
            float4 acc_high = 0.0f;
            VNEXT_UNROLL for (short index = 0; index < 4; ++index) {
                acc_low[0] += low[2 * index] * (quants[index] & 0x000f);
                acc_low[1] += low[2 * index + 1] * (quants[index] & 0x0f00);
                acc_low[2] += low[2 * index + 8] * (quants[index] & 0x00f0);
                acc_low[3] += low[2 * index + 9] * (quants[index] & 0xf000);
                acc_high[0] += high[2 * index] * (high_quants[index] & 0x000f);
                acc_high[1] += high[2 * index + 1] * (high_quants[index] & 0x0f00);
                acc_high[2] += high[2 * index + 8] * (high_quants[index] & 0x00f0);
                acc_high[3] += high[2 * index + 9] * (high_quants[index] & 0xf000);
            }
            sums[row] += float(delta[0]) * (
                (acc_low[0] + acc_low[1] / 256.0f) * scales8[0]
                + (acc_low[2] + acc_low[3] / 256.0f) * scales8[1] / 16.0f
                + (acc_high[0] + acc_high[1] / 256.0f) * scales8[4]
                + (acc_high[2] + acc_high[3] / 256.0f) * scales8[5] / 16.0f)
                - float(delta[1]) * (
                    input_sums[0] * scales8[2] + input_sums[1] * scales8[3]
                    + input_sums[2] * scales8[6] + input_sums[3] * scales8[7]);
            quants += p.row_stride_bytes / 2;
            scales += p.row_stride_bytes / 2;
            delta += p.row_stride_bytes / 2;
        }
        input_block += 4 * VNEXT_QK_K;
    }

    device half *output = down_slots + pair * p.output_features;
    for (short row = 0;
         row < VNEXT_MOE_ROWS_PER_SIMDGROUP && first_row + uint(row) < p.output_features;
         ++row) {
        const float value = simd_sum(sums[row]);
        if (lane == 0) {
            output[first_row + uint(row)] = half(value);
        }
    }
}

// Q6_K routed expert down projection. This keeps both the expert stack and
// the activation/output ABI native: packed Q6_K bytes in, F16 slots out.
struct VNextBlockQ6K {
    uchar ql[VNEXT_QK_K / 2];
    uchar qh[VNEXT_QK_K / 4];
    int8_t scales[VNEXT_QK_K / 16];
    half d;
};

kernel void vnext_moe_q6k_down_f16(
    device const VNextBlockQ6K *weights [[buffer(0)]],
    device const half *activation [[buffer(1)]],
    device const int *route_ids [[buffer(2)]],
    device half *down_slots [[buffer(3)]],
    constant MoeQ4DownParams &p [[buffer(4)]],
    uint3 group [[threadgroup_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort simdgroup [[simdgroup_index_in_threadgroup]]) {
    const uint pair = group.z;
    if (pair >= p.pair_count) {
        return;
    }
    const int expert = route_ids[pair];
    if (expert < 0 || uint(expert) >= p.expert_count) {
        return;
    }

    constexpr uint8_t mask_0 = 0x03;
    constexpr uint8_t mask_1 = 0x0c;
    constexpr uint8_t mask_2 = 0x30;
    constexpr uint8_t mask_3 = 0xc0;
    const uint blocks_per_row = p.input_features / VNEXT_QK_K;
    const uint first_row =
        (group.x * VNEXT_MOE_SIMDGROUPS + simdgroup) * VNEXT_MOE_ROWS_PER_SIMDGROUP;
    if (first_row >= p.output_features) {
        return;
    }

    device const VNextBlockQ6K *expert_weights =
        reinterpret_cast<device const VNextBlockQ6K *>(
            reinterpret_cast<device const char *>(weights)
            + uint(expert) * p.expert_stride_bytes
            + first_row * p.row_stride_bytes);
    device const half *input = activation + pair * p.input_features;
    float sums[VNEXT_MOE_ROWS_PER_SIMDGROUP] = {0.0f};
    float inputs[16];

    const short lane_pair = lane / 2;
    const short block_parity = lane % 2;
    const short input_partition = lane_pair / 8;
    const short input_lane = lane_pair % 8;
    const short lane_offset = 4 * input_lane;
    const short scale_offset = 8 * input_partition + lane_offset / 16;
    const short input_offset = 128 * input_partition + lane_offset;
    const short low_offset = 64 * input_partition + lane_offset;
    const short high_offset = 32 * input_partition + lane_offset;

    for (uint block = uint(block_parity); block < blocks_per_row; block += 2) {
        device const uchar *low = expert_weights[block].ql + low_offset;
        device const uchar *low_second = low + 32;
        device const uchar *high = expert_weights[block].qh + high_offset;
        device const int8_t *scales = expert_weights[block].scales + scale_offset;
        device const half *delta = &expert_weights[block].d;
        device const half *input_block = input + block * VNEXT_QK_K + input_offset;

        VNEXT_UNROLL for (short index = 0; index < 4; ++index) {
            inputs[4 * index + 0] = float(input_block[index + 0]);
            inputs[4 * index + 1] = float(input_block[index + 32]);
            inputs[4 * index + 2] = float(input_block[index + 64]);
            inputs[4 * index + 3] = float(input_block[index + 96]);
        }

        for (short row = 0; row < VNEXT_MOE_ROWS_PER_SIMDGROUP; ++row) {
            float4 partial = 0.0f;
            VNEXT_UNROLL for (short index = 0; index < 4; ++index) {
                partial[0] += inputs[4 * index + 0]
                    * (int8_t((low[index] & 0x0f) | ((high[index] & mask_0) << 4)) - 32);
                partial[1] += inputs[4 * index + 1]
                    * (int8_t((low_second[index] & 0x0f) | ((high[index] & mask_1) << 2)) - 32);
                partial[2] += inputs[4 * index + 2]
                    * (int8_t((low[index] >> 4) | (high[index] & mask_2)) - 32);
                partial[3] += inputs[4 * index + 3]
                    * (int8_t((low_second[index] >> 4) | ((high[index] & mask_3) >> 2)) - 32);
            }
            sums[row] += float(delta[0]) * (
                partial[0] * float(scales[0])
                + partial[1] * float(scales[2])
                + partial[2] * float(scales[4])
                + partial[3] * float(scales[6]));
            low += p.row_stride_bytes;
            low_second += p.row_stride_bytes;
            high += p.row_stride_bytes;
            scales += p.row_stride_bytes;
            delta += p.row_stride_bytes / 2;
        }
    }

    device half *output = down_slots + pair * p.output_features;
    for (short row = 0;
         row < VNEXT_MOE_ROWS_PER_SIMDGROUP && first_row + uint(row) < p.output_features;
         ++row) {
        const float value = simd_sum(sums[row]);
        if (lane == 0) {
            output[first_row + uint(row)] = half(value);
        }
    }
}

struct MoeCombineParams {
    uint tokens;
    uint experts_per_token;
    uint hidden_size;
};

kernel void vnext_moe_combine_routed_f16(
    device const half *routed_slots [[buffer(0)]],
    device const float *route_weights [[buffer(1)]],
    device half *output [[buffer(2)]],
    constant MoeCombineParams &p [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
    const uint element_count = p.tokens * p.hidden_size;
    if (index >= element_count) {
        return;
    }
    const uint token = index / p.hidden_size;
    const uint column = index - token * p.hidden_size;
    float value = 0.0f;
    for (uint slot = 0; slot < p.experts_per_token; ++slot) {
        const uint pair = token * p.experts_per_token + slot;
        value += route_weights[pair]
            * float(routed_slots[pair * p.hidden_size + column]);
    }
    output[index] = half(value);
}

kernel void vnext_moe_combine_f16(
    device const half *routed_slots [[buffer(0)]],
    device const float *route_weights [[buffer(1)]],
    device const half *shared_gate [[buffer(2)]],
    device const half *shared_output [[buffer(3)]],
    device half *output [[buffer(4)]],
    constant MoeCombineParams &p [[buffer(5)]],
    uint index [[thread_position_in_grid]]) {
    const uint element_count = p.tokens * p.hidden_size;
    if (index >= element_count) {
        return;
    }
    const uint token = index / p.hidden_size;
    const uint column = index - token * p.hidden_size;
    float value = 0.0f;
    for (uint slot = 0; slot < p.experts_per_token; ++slot) {
        const uint pair = token * p.experts_per_token + slot;
        value += route_weights[pair]
            * float(routed_slots[pair * p.hidden_size + column]);
    }
    const float gate = float(shared_gate[token]);
    value += (1.0f / (1.0f + exp(-gate))) * float(shared_output[index]);
    output[index] = half(value);
}
