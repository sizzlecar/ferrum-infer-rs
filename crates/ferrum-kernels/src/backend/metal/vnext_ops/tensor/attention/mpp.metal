// Prepended only to the independent MSL 4 attention library. The original
// library never parses these headers and keeps its generic SIMD-group path.
#if __METAL_VERSION__ < 400
#error "Ferrum MPP BF16 QK requires Metal Shading Language 4"
#endif
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#define FERRUM_ATTENTION_MPP_QK 1
using namespace mpp::tensor_ops;

inline void ferrum_attention_mpp_qk(
    device const ushort* query,
    device const ushort* key,
    ulong query_offset,
    ulong key_offset,
    int query_stride,
    int key_stride,
    threadgroup float* scores) {
    // MPP requires unqualified element types. These borrowed views are only
    // operands of run(); neither original admitted input is written through.
    auto q = tensor(
        const_cast<device bfloat*>(reinterpret_cast<device const bfloat*>(query + query_offset)),
        dextents<int, 2>{128, 16}, array<int, 2>{1, query_stride});
    auto k = tensor(
        const_cast<device bfloat*>(reinterpret_cast<device const bfloat*>(key + key_offset)),
        dextents<int, 2>{128, 32}, array<int, 2>{1, key_stride});
    constexpr auto descriptor = matmul2d_descriptor(16, 32, 128, false, true, false);
    matmul2d<descriptor, execution_simdgroups<4>> operation;
    auto result = operation.get_destination_cooperative_tensor<decltype(q), decltype(k), float>();
    operation.run(q, k, result);
    // Reuse the original 16x32 FP32 score tile; no added shared/device storage.
    auto destination = tensor(scores, dextents<int, 2>{32, 16}, array<int, 2>{1, 32});
    result.store(destination);
}
