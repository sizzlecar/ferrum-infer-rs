// Minimal stubs for the torch APIs that vLLM's marlin code uses.
// Lets us compile the kernel files standalone without PyTorch.
#pragma once

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <stdexcept>

// Stream args into a string and call printf, then abort. We match TORCH_CHECK's
// usage pattern (free args after a condition).
namespace ferrum_marlin_stubs {
inline void stub_check_fail(const char* file, int line, const char* cond_text) {
    std::fprintf(stderr, "%s:%d: assertion failed: %s\n", file, line, cond_text);
    std::abort();
}

template <typename... Args>
inline void stub_print_args(std::ostream& os) { (void)os; }
template <typename T, typename... Rest>
inline void stub_print_args(std::ostream& os, const T& v, const Rest&... rest) {
    os << v;
    stub_print_args(os, rest...);
}

template <typename... Args>
inline void stub_check_fail_msg(const char* file, int line, const char* cond_text,
                                 const Args&... args) {
    std::ostringstream os;
    stub_print_args(os, args...);
    std::fprintf(stderr, "%s:%d: %s :: %s\n", file, line, cond_text, os.str().c_str());
    std::abort();
}
}  // namespace ferrum_marlin_stubs

// TORCH_CHECK(cond, ...): abort if cond false.
#define TORCH_CHECK(cond, ...)                                                          \
    do {                                                                                \
        if (!(cond)) {                                                                  \
            ::ferrum_marlin_stubs::stub_check_fail_msg(__FILE__, __LINE__, #cond,       \
                                                       ##__VA_ARGS__);                   \
        }                                                                               \
    } while (0)

// TORCH_CHECK_NOT_IMPLEMENTED — used in stubs.
#define TORCH_CHECK_NOT_IMPLEMENTED(cond, ...) TORCH_CHECK(cond, __VA_ARGS__)

// vLLM uses torch::Tensor signatures in some non-kernel entry points. We don't
// need those here — we only call marlin_mm directly. Provide a forward declare
// so compilation passes if any header references it.
namespace torch {
class Tensor;  // opaque
}
