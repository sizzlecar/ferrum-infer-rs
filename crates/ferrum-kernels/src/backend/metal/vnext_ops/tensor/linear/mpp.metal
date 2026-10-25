// BF16 operands with a cooperative FP32 result and the original bias epilogue.
// The independent library prepends ../bf16.metal so its rounding/NaN helpers
// exactly match the existing linear epilogue. No new device buffer is created.
#if __METAL_VERSION__ < 400
#error "Ferrum MPP BF16 linear requires Metal Shading Language 4"
#endif
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp::tensor_ops;

// Same buffer(3) layout as LinearParams: six consecutive uints.
struct MppLinearParams {
    uint rows; uint inputs; uint outputs; uint bias_mode;
    uint row_offset; uint column_offset;
};

// Host-side dispatch admits only complete output tiles, positive int-range
// extents, and exactly four 32-thread SIMD groups. K is dynamic, including tails.
template<int TM>
inline void ferrum_tensor_linear_bf16_mpp_tile(
    device const bfloat* input,
    device const bfloat* weight,
    device ushort* output,
    constant MppLinearParams& p,
    device const ushort* bias,
    uint3 group,
    uint tid,
    threadgroup float* results) {
    constexpr int TN = 64;
    int row_start = int(p.row_offset) + int(group.x) * TM;
    int col_start = int(p.column_offset) + int(group.y) * TN;
    // These are borrowed views of the original admitted buffers. Weight is OI,
    // so the descriptor transposes the right operand without a global copy.
    // MPP's cooperative type checker requires an unqualified element type.
    // These views are read-only operands of run(); the borrowed inputs are
    // never stored through, and remain disjoint from the admitted output.
    auto a = tensor(const_cast<device bfloat*>(input), dextents<int, 2>{int(p.inputs), int(p.rows)},
                    array<int, 2>{1, int(p.inputs)});
    auto w = tensor(const_cast<device bfloat*>(weight), dextents<int, 2>{int(p.inputs), int(p.outputs)},
                    array<int, 2>{1, int(p.inputs)});
    auto a_tile = a.slice<dynamic_extent, TM>(0, row_start);
    auto w_tile = w.slice<dynamic_extent, TN>(0, col_start);
    constexpr auto descriptor = matmul2d_descriptor(
        TM, TN, dynamic_length_v<int>, false, true, false);
    matmul2d<descriptor, execution_simdgroups<4>> operation;
    auto accumulator = operation.template get_destination_cooperative_tensor<
        decltype(a_tile), decltype(w_tile), float>();
    operation.run(a_tile, w_tile, accumulator);

    // Preserve the existing per-column bias and integer BF16 epilogue without
    // depending on the implementation-defined cooperative element ownership.
    // All scratch here is on-chip; check the compiled PSO's actual static usage.
    auto result_tile = tensor(results, dextents<int, 2>{TN, TM},
                              array<int, 2>{1, TN});
    accumulator.store(result_tile);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < uint(TM * TN); i += 128) {
        ulong row = ulong(row_start) + i / TN;
        ulong col = ulong(col_start) + i % TN;
        if (row < p.rows && col < p.outputs) {
            float value = p.bias_mode == 2 ? rounded_bf16(results[i]) : results[i];
            if (p.bias_mode != 0) value = value + decode_bf16(bias[col]);
            output[row * p.outputs + col] = encode_bf16(value);
        }
    }
}

kernel void ferrum_tensor_linear_bf16_mpp(
    device const bfloat* input [[buffer(0)]],
    device const bfloat* weight [[buffer(1)]],
    device ushort* output [[buffer(2)]],
    constant MppLinearParams& p [[buffer(3)]],
    device const ushort* bias [[buffer(4)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float results[32 * 64];
    ferrum_tensor_linear_bf16_mpp_tile<32>(input, weight, output, p, bias, group, tid, results);
}

kernel void ferrum_tensor_linear_bf16_mpp_m64n64(
    device const bfloat* input [[buffer(0)]],
    device const bfloat* weight [[buffer(1)]],
    device ushort* output [[buffer(2)]],
    constant MppLinearParams& p [[buffer(3)]],
    device const ushort* bias [[buffer(4)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float results[64 * 64];
    ferrum_tensor_linear_bf16_mpp_tile<64>(input, weight, output, p, bias, group, tid, results);
}
