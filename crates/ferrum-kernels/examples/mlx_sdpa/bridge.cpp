// Thin C ABI only. Fixtures, timers, statistics and oracles live in Rust.
// Link an existing official MLX wheel; do not compile a second MLX here.
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "mlx/mlx.h"
#include "mlx/version.h"

namespace mx = mlx::core;

struct Config {
  uint32_t abi, batch, queries, keys, heads, dim;
  float scale;
  uint32_t reserved;
};
static_assert(sizeof(Config) == 32);
struct Info {
  char version[128], library[1024], device[128], architecture[128], source_sha256[65];
};

template <typename F>
int guarded(char* error, size_t capacity, F&& f) noexcept {
  try {
    f();
    return 0;
  } catch (const std::exception& e) {
    if (error && capacity) std::snprintf(error, capacity, "%s", e.what());
  } catch (...) {
    if (error && capacity) std::snprintf(error, capacity, "unknown C++ exception");
  }
  return 1;
}

template <size_t N>
void text(char (&out)[N], const std::string& value) {
  if (value.size() >= N) throw std::runtime_error("bridge identity field too long");
  std::memcpy(out, value.c_str(), value.size() + 1);
}

size_t elements(const Config& c, uint32_t sequence) {
  size_t n = 1;
  for (auto x : {c.batch, sequence, c.heads, c.dim}) {
    if (!x || x > uint32_t(std::numeric_limits<int>::max()) ||
        n > size_t(std::numeric_limits<ptrdiff_t>::max()) / 2 / x)
      throw std::invalid_argument("invalid or overflowing SDPA shape");
    n *= x;
  }
  return n;
}

mx::array input(const uint16_t* data, const Config& c, uint32_t sequence,
                const mx::Stream& stream) {
  const auto n = elements(c, sequence);
  // bit_cast preserves the incoming BF16 representation, without interpreting
  // uint16_t numerically. The iterator constructor copies into MLX ownership.
  static_assert(sizeof(mx::bfloat16_t) == sizeof(uint16_t));
  std::vector<mx::bfloat16_t> words(n);
  for (size_t i = 0; i < n; ++i) words[i] = std::bit_cast<mx::bfloat16_t>(data[i]);
  mx::array backing(words.begin(),
      {int(c.batch), int(sequence), int(c.heads), int(c.dim)}, mx::bfloat16);
  return mx::transpose(backing, {0, 2, 1, 3}, stream);
}

struct State {
  Config config;
  mx::Stream stream;
  mx::array q, k, v;
  std::optional<mx::array> output;
  bool failed = false;
  State(const Config& c, const uint16_t* q_, const uint16_t* k_, const uint16_t* v_)
      : config(c), stream(mx::default_stream(mx::Device::gpu)),
        q(input(q_, c, c.queries, stream)), k(input(k_, c, c.keys, stream)),
        v(input(v_, c, c.keys, stream)) {}
};

extern "C" uint32_t ferrum_mlx_sdpa_abi_v1() noexcept { return 1; }

extern "C" int ferrum_mlx_sdpa_info_v1(Info* out, char* error, size_t capacity) noexcept {
  return guarded(error, capacity, [&] {
    if (!out) throw std::invalid_argument("null identity output");
    *out = {};
    text(out->version, mx::version());
    text(out->source_sha256, FERRUM_MLX_BRIDGE_SOURCE_SHA256);
    Dl_info library{};
    if (!dladdr(reinterpret_cast<const void*>(&mx::version), &library) || !library.dli_fname)
      throw std::runtime_error("cannot resolve the live MLX library identity");
    text(out->library, library.dli_fname);
    if (!mx::metal::is_available()) throw std::runtime_error("MLX Metal backend unavailable");
    mx::set_default_device(mx::Device::gpu);
    const auto& facts = mx::device_info(mx::Device::gpu);
    auto field = [&](const char* key) {
      auto it = facts.find(key);
      if (it == facts.end()) return std::string("unreported");
      if (auto value = std::get_if<std::string>(&it->second)) return *value;
      return std::to_string(std::get<size_t>(it->second));
    };
    text(out->device, field("device_name"));
    text(out->architecture, field("architecture"));
  });
}

extern "C" int ferrum_mlx_sdpa_create_v1(
    const Config* config, const uint16_t* q, uint64_t q_count,
    const uint16_t* k, const uint16_t* v, uint64_t kv_count,
    void** output, char* error, size_t capacity) noexcept {
  if (output) *output = nullptr;
  return guarded(error, capacity, [&] {
    if (!config || !q || !k || !v || !output || config->abi != 1 || config->reserved ||
        !std::isfinite(config->scale) || config->scale <= 0 ||
        q_count != elements(*config, config->queries) ||
        kv_count != elements(*config, config->keys))
      throw std::invalid_argument("invalid SDPA input extent/configuration");
    auto state = std::make_unique<State>(*config, q, k, v);
    // Publish ownership before any GPU submission. Even an evaluation error
    // returns this handle for synchronize/retirement, never an unowned job.
    auto* owned = state.release();
    *output = owned;
    try {
      mx::eval(owned->q, owned->k, owned->v);
      mx::synchronize(owned->stream);
    } catch (...) {
      owned->failed = true;
      throw;
    }
  });
}

extern "C" int ferrum_mlx_sdpa_evaluate_v1(void* handle, char* error, size_t capacity) noexcept {
  return guarded(error, capacity, [&] {
    if (!handle) throw std::invalid_argument("null SDPA handle");
    auto& s = *static_cast<State*>(handle);
    if (s.failed) throw std::runtime_error("SDPA session stopped after an error");
    try {
      s.output.reset();
      s.output = mx::fast::scaled_dot_product_attention(
          s.q, s.k, s.v, s.config.scale, "", {}, {}, false, s.stream);
      mx::eval(*s.output);
      s.output->wait();
    } catch (...) {
      s.failed = true;
      throw;
    }
  });
}

extern "C" int ferrum_mlx_sdpa_copy_v1(
    void* handle, uint16_t* output, uint64_t count, char* error, size_t capacity) noexcept {
  return guarded(error, capacity, [&] {
    if (!handle || !output) throw std::invalid_argument("null output/handle");
    auto& s = *static_cast<State*>(handle);
    if (s.failed || !s.output || !s.output->is_available())
      throw std::runtime_error("SDPA output is not available");
    const auto& c = s.config;
    auto& a = *s.output;
    if (count != elements(c, c.queries) || a.dtype() != mx::bfloat16 ||
        a.shape() != mx::Shape{int(c.batch), int(c.heads), int(c.queries), int(c.dim)} ||
        a.strides(3) != 1)
      throw std::runtime_error("unexpected MLX output dtype/shape/last stride");
    for (auto stride : a.strides())
      if (stride < 0) throw std::runtime_error("unexpected negative output stride");
    const auto* data = a.data<mx::bfloat16_t>();
    // Full-SDPA currently returns BSHD physical storage. Do not assume that for
    // other dispatches: the small rectangular probe also checks strided copy.
    if (a.strides(1) == c.dim && a.strides(2) == size_t(c.heads) * c.dim &&
        a.strides(0) == size_t(c.queries) * c.heads * c.dim) {
      std::memcpy(output, data, count * sizeof(uint16_t));
    } else {
      size_t dst = 0;
      for (size_t b = 0; b < c.batch; ++b)
        for (size_t q = 0; q < c.queries; ++q)
          for (size_t h = 0; h < c.heads; ++h) {
            const size_t src = b * a.strides(0) + h * a.strides(1) + q * a.strides(2);
            std::memcpy(output + dst, data + src, c.dim * sizeof(uint16_t));
            dst += c.dim;
          }
    }
  });
}

extern "C" int ferrum_mlx_sdpa_close_v1(void* handle, char* error, size_t capacity) noexcept {
  return guarded(error, capacity, [&] {
    if (!handle) return;
    auto* s = static_cast<State*>(handle);
    mx::synchronize(s->stream); // Failure leaves the handle and arrays owned.
    delete s;
  });
}
