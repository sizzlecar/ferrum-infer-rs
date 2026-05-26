// scalar_type_id_test_other_tu.cu — second translation unit. Simulates
// kernel_instantiations.cu — uses the same scalar_type.hpp from a
// SEPARATE compile, so we can confirm constexpr/literal IDs match
// across TUs (the bug we're hunting).

#include <string>
#include <cstdint>
#include <tuple>
#include <utility>
#include "core/scalar_type.hpp"
#include <cstdio>

namespace other_tu {

void print_ids_from_other_tu() {
    int64_t w_lit = vllm::kU4B8Id_LITERAL;
    int64_t s_lit = vllm::kFloat16Id_LITERAL;
    int64_t w_ce  = vllm::kU4B8.id();
    int64_t s_ce  = vllm::kFloat16.id();
    printf("  kU4B8Id_LITERAL    = %lld\n", (long long)w_lit);
    printf("  kFloat16Id_LITERAL = %lld\n", (long long)s_lit);
    printf("  vllm::kU4B8.id()   = %lld  %s\n", (long long)w_ce,
           (w_lit == w_ce) ? "(MATCH)" : "(MISMATCH)");
    printf("  vllm::kFloat16.id() = %lld  %s\n", (long long)s_ce,
           (s_lit == s_ce) ? "(MATCH)" : "(MISMATCH)");
}

int64_t kU4B8_id_value() {
    return vllm::kU4B8Id_LITERAL;
}

int64_t kFloat16_id_value() {
    return vllm::kFloat16Id_LITERAL;
}

}  // namespace other_tu
