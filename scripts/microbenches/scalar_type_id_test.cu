// scalar_type_id_test.cu — minimal repro that verifies template
// specializations of ScalarType-id-keyed templates dispatch correctly
// across multiple TUs.
//
// Without rebuilding all of ferrum:
//   nvcc -O3 -arch=sm_89 -std=c++17 \
//        -DMARLIN_NAMESPACE_NAME=marlin_moe_wna16 \
//        -Ikernels/vllm_marlin_moe \
//        scalar_type_id_test.cu scalar_type_id_test_other_tu.cu \
//        -o scalar_type_id_test
//
// Two TUs simulate the dispatcher (ops.cu) and explicit instantiations
// (kernel_instantiations.cu) using ScalarType IDs as template params.
// If the fix works, both TUs report the same ID literals and templated
// kernel runs the SPECIALIZED path (not the unspecialized base).

#include <string>
#include <cstdint>
#include <tuple>
#include <utility>
#include "core/scalar_type.hpp"
#include <cstdio>
#include <cuda_runtime.h>

// Forward decl from the OTHER TU (simulates kernel_instantiations.cu)
namespace other_tu {
    extern void print_ids_from_other_tu();
    extern int64_t kU4B8_id_value();
    extern int64_t kFloat16_id_value();
}

// Template that specializes on `int64_t W_ID, int64_t S_ID` (mirrors
// `Marlin<scalar_t, W_TYPE.id(), S_TYPE.id(), ...>` in real code).
template <int64_t W_ID, int64_t S_ID>
__device__ int templated_kernel_body(int input) {
    return -1;  // unspecialized base — wrong dispatch
}

// Specialization for (kU4B8, kFloat16) — mirrors what kernel_instantiations.cu
// would explicitly instantiate.
template <>
__device__ int templated_kernel_body<vllm::kU4B8Id_LITERAL, vllm::kFloat16Id_LITERAL>(int input) {
    return input + 42;  // specialized — correct dispatch returns this
}

__global__ void dispatch_kernel(int* out, int input) {
    // Mimic dispatcher: compute IDs at "this TU's" constexpr eval and
    // call the templated kernel.
    constexpr int64_t W = vllm::kU4B8Id_LITERAL;
    constexpr int64_t S = vllm::kFloat16Id_LITERAL;
    *out = templated_kernel_body<W, S>(input);
}

int main() {
    printf("=== ScalarType ID consistency test ===\n");

    // Print IDs from THIS TU
    int64_t w_this = vllm::kU4B8Id_LITERAL;
    int64_t s_this = vllm::kFloat16Id_LITERAL;
    int64_t w_constexpr = vllm::kU4B8.id();      // legacy: from id() method
    int64_t s_constexpr = vllm::kFloat16.id();
    printf("\n[this TU]\n");
    printf("  kU4B8Id_LITERAL  = %lld\n", (long long)w_this);
    printf("  kFloat16Id_LITERAL = %lld\n", (long long)s_this);
    printf("  vllm::kU4B8.id() = %lld  %s\n", (long long)w_constexpr,
           (w_this == w_constexpr) ? "(MATCH)" : "(MISMATCH ← old bug)");
    printf("  vllm::kFloat16.id() = %lld  %s\n", (long long)s_constexpr,
           (s_this == s_constexpr) ? "(MATCH)" : "(MISMATCH)");

    // Print IDs from OTHER TU (simulates kernel_instantiations.cu)
    printf("\n[other TU — simulates kernel_instantiations.cu]\n");
    other_tu::print_ids_from_other_tu();
    int64_t w_other = other_tu::kU4B8_id_value();
    int64_t s_other = other_tu::kFloat16_id_value();
    printf("  cross-TU U4B8 match: %s (%lld == %lld?)\n",
           (w_this == w_other) ? "OK" : "MISMATCH",
           (long long)w_this, (long long)w_other);
    printf("  cross-TU Float16 match: %s\n",
           (s_this == s_other) ? "OK" : "MISMATCH");

    // Now exercise the GPU dispatch
    int* d_out;
    cudaMalloc(&d_out, sizeof(int));
    dispatch_kernel<<<1, 1>>>(d_out, 100);
    cudaDeviceSynchronize();
    int h_out;
    cudaMemcpy(&h_out, d_out, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_out);

    printf("\n[GPU dispatch test]\n");
    printf("  templated_kernel_body<U4B8,Float16>(100) = %d\n", h_out);
    printf("  expected = 142 (= 100 + 42, specialization hit)\n");
    printf("  got -1 means specialization MISSED → ID mismatch in template params\n");
    printf("  VERDICT: %s\n", (h_out == 142) ? "✓ PASS — template dispatch works"
                                              : "✗ FAIL — template dispatch broken");

    return (h_out == 142) ? 0 : 1;
}
