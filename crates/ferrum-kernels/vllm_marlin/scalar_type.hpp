#pragma once

// For TORCH_CHECK — replaced with our stub.
#include "torch_stubs.h"

// ferrum-infer-rs port: vLLM's build pulls these via <torch/all.h>; we need
// them explicitly when compiling standalone with nvcc.
#include <cstdint>
#include <variant>

namespace vllm {

//
//  ScalarType can represent a wide range of floating point and integer types,
//  in particular it can be used to represent sub-byte data types (something
//  that torch.dtype currently does not support).
//
//  The type definitions on the Python side can be found in: vllm/scalar_type.py
//  these type definitions should be kept up to date with any Python API changes
//  here.
//
class ScalarType {
 public:
  enum NanRepr : uint8_t {
    NAN_NONE = 0,                // nans are not supported
    NAN_IEEE_754 = 1,            // nans are: exp all 1s, mantissa not all 0s
    NAN_EXTD_RANGE_MAX_MIN = 2,  // nans are: exp all 1s, mantissa all 1s

    NAN_REPR_ID_MAX
  };

  constexpr ScalarType(uint8_t exponent, uint8_t mantissa, bool signed_,
                       int32_t bias, bool finite_values_only = false,
                       NanRepr nan_repr = NAN_IEEE_754)
      : exponent(exponent),
        mantissa(mantissa),
        signed_(signed_),
        bias(bias),
        finite_values_only(finite_values_only),
        nan_repr(nan_repr) {};

  static constexpr ScalarType int_(uint8_t size_bits, int32_t bias = 0) {
    return ScalarType(0, size_bits - 1, true, bias);
  }

  static constexpr ScalarType uint(uint8_t size_bits, int32_t bias = 0) {
    return ScalarType(0, size_bits, false, bias);
  }

  // IEEE 754 compliant floating point type
  static constexpr ScalarType float_IEEE754(uint8_t exponent,
                                            uint8_t mantissa) {
    TORCH_CHECK(mantissa > 0 && exponent > 0);
    return ScalarType(exponent, mantissa, true, 0, false, NAN_IEEE_754);
  }

  // IEEE 754 non-compliant floating point type
  static constexpr ScalarType float_(uint8_t exponent, uint8_t mantissa,
                                     bool finite_values_only,
                                     NanRepr nan_repr) {
    TORCH_CHECK(nan_repr < NAN_REPR_ID_MAX, "Invalid NanRepr");
    TORCH_CHECK(mantissa > 0 && exponent > 0);
    TORCH_CHECK(nan_repr != NAN_IEEE_754,
                "use `float_IEEE754` constructor for floating point types that "
                "follow IEEE 754 conventions");
    return ScalarType(exponent, mantissa, true, 0, finite_values_only,
                      nan_repr);
  }

  uint8_t const exponent;  // size of the exponent field (0 for integer types)
  uint8_t const mantissa;  // size of the mantissa field (size of the integer
                           // excluding the sign bit for integer types)
  bool const signed_;  // flag if the type supports negative numbers (i.e. has a
                       // sign bit)
  int32_t const bias;  // stored values equal value + bias,
                       // used for quantized type

  // Extra Floating point info
  bool const finite_values_only;  // i.e. no +/-inf if true
  NanRepr const nan_repr;         // how NaNs are represented
                                  // (not applicable for integer types)

  using Id = int64_t;

 public:
  // unique id for this scalar type that can be computed at compile time for
  //  c++17 template specialization this is not needed once we migrate to
  //  c++20 and can pass literal classes as template parameters
  constexpr Id id() const {
    // Keep this layout explicit. CUDA 13 evaluated the former generic
    // member-reduction implementation differently across translation units,
    // which changed template symbol names and made generated kernels
    // unreachable from the dispatcher.
    return static_cast<Id>(
        uint64_t(exponent) | (uint64_t(mantissa) << 8) |
        (uint64_t(signed_) << 16) |
        (uint64_t(static_cast<uint32_t>(bias)) << 17) |
        (uint64_t(finite_values_only) << 49) |
        (uint64_t(static_cast<uint8_t>(nan_repr)) << 50));
  }

  // create a ScalarType from an id, for c++17 template specialization,
  //  this is not needed once we migrate to c++20 and can pass literal
  //  classes as template parameters
  static constexpr ScalarType from_id(Id id) {
    const auto raw = static_cast<uint64_t>(id);
    const auto encoded_bias = static_cast<uint32_t>((raw >> 17) & 0xffffffffu);
    const auto decoded_bias =
        encoded_bias >= 0x80000000u
            ? static_cast<int64_t>(encoded_bias) - (int64_t(1) << 32)
            : static_cast<int64_t>(encoded_bias);
    return ScalarType(
        static_cast<uint8_t>(raw & 0xffu),
        static_cast<uint8_t>((raw >> 8) & 0xffu),
        static_cast<bool>((raw >> 16) & 0x1u),
        static_cast<int32_t>(decoded_bias),
        static_cast<bool>((raw >> 49) & 0x1u),
        static_cast<NanRepr>((raw >> 50) & 0xffu));
  }

  constexpr int64_t size_bits() const {
    return mantissa + exponent + is_signed();
  }
  constexpr bool is_signed() const { return signed_; }
  constexpr bool is_integer() const { return exponent == 0; }
  constexpr bool is_floating_point() const { return exponent > 0; }
  constexpr bool is_ieee_754() const {
    return is_floating_point() && finite_values_only == false &&
           nan_repr == NAN_IEEE_754;
  }
  constexpr bool has_nans() const {
    return is_floating_point() && nan_repr != NAN_NONE;
  }
  constexpr bool has_infs() const {
    return is_floating_point() && finite_values_only == false;
  }
  constexpr bool has_bias() const { return bias != 0; }

 private:
  double _floating_point_max() const {
    TORCH_CHECK(mantissa <= 52 && exponent <= 11,
                "Cannot represent max/min as a double for type ", str());

    uint64_t max_mantissa = (uint64_t(1) << mantissa) - 1;
    if (nan_repr == NAN_EXTD_RANGE_MAX_MIN) {
      max_mantissa -= 1;
    }

    uint64_t max_exponent = (uint64_t(1) << exponent) - 2;
    if (nan_repr == NAN_EXTD_RANGE_MAX_MIN || nan_repr == NAN_NONE) {
      TORCH_CHECK(exponent < 11,
                  "Cannot represent max/min as a double for type ", str());
      max_exponent += 1;
    }

    // adjust the exponent to match that of a double
    //  for now we assume the exponent bias is the standard 2^(e-1) -1, (where e
    //  is the exponent bits), there is some precedent for non-standard biases,
    //  example `float8_e4m3b11fnuz` here: https://github.com/jax-ml/ml_dtypes
    //  but to avoid premature over complication we are just assuming the
    //  standard exponent bias until there is a need to support non-standard
    //  biases
    uint64_t exponent_bias = (uint64_t(1) << (exponent - 1)) - 1;
    uint64_t exponent_bias_double = (uint64_t(1) << 10) - 1;  // double e = 11

    uint64_t max_exponent_double =
        max_exponent - exponent_bias + exponent_bias_double;

    // shift the mantissa into the position for a double and
    // the exponent
    uint64_t double_raw =
        (max_mantissa << (52 - mantissa)) | (max_exponent_double << 52);

    return *reinterpret_cast<double*>(&double_raw);
  }

  constexpr std::variant<int64_t, double> _raw_max() const {
    if (is_floating_point()) {
      return {_floating_point_max()};
    } else {
      TORCH_CHECK(size_bits() < 64 || size_bits() == 64 && is_signed(),
                  "Cannot represent max as a int64_t");
      return {(int64_t(1) << mantissa) - 1};
    }
  }

  constexpr std::variant<int64_t, double> _raw_min() const {
    if (is_floating_point()) {
      TORCH_CHECK(is_signed(),
                  "We currently assume all floating point types are signed");
      constexpr uint64_t sign_bit_double = (uint64_t(1) << 63);

      double max = _floating_point_max();
      uint64_t max_raw = *reinterpret_cast<uint64_t*>(&max);
      uint64_t min_raw = max_raw | sign_bit_double;
      return {*reinterpret_cast<double*>(&min_raw)};
    } else {
      TORCH_CHECK(!is_signed() || size_bits() <= 64,
                  "Cannot represent min as a int64_t");
      if (is_signed()) {
        // set the top bit to 1 (i.e. INT64_MIN) and the rest to 0
        // then perform an arithmetic shift right to set all the bits above
        // (size_bits() - 1) to 1
        return {INT64_MIN >> (64 - size_bits())};
      } else {
        return {int64_t(0)};
      }
    }
  }

 public:
  // Max representable value for this scalar type.
  // (accounting for bias if there is one)
  constexpr std::variant<int64_t, double> max() const {
    return std::visit(
        [this](auto x) -> std::variant<int64_t, double> { return {x - bias}; },
        _raw_max());
  }

  // Min representable value for this scalar type.
  // (accounting for bias if there is one)
  constexpr std::variant<int64_t, double> min() const {
    return std::visit(
        [this](auto x) -> std::variant<int64_t, double> { return {x - bias}; },
        _raw_min());
  }

  std::string str() const {
    /* naming generally follows: https://github.com/jax-ml/ml_dtypes
     * for floating point types (leading f) the scheme is:
     *  `float<size_bits>_e<exponent_bits>m<mantissa_bits>[flags]`
     *  flags:
     *  - no-flags: means it follows IEEE 754 conventions
     *  - f: means finite values only (no infinities)
     *  - n: means nans are supported (non-standard encoding)
     * for integer types the scheme is:
     *  `[u]int<size_bits>[b<bias>]`
     *  - if bias is not present it means its zero
     */
    if (is_floating_point()) {
      auto ret = "float" + std::to_string(size_bits()) + "_e" +
                 std::to_string(exponent) + "m" + std::to_string(mantissa);
      if (!is_ieee_754()) {
        if (finite_values_only) {
          ret += "f";
        }
        if (nan_repr != NAN_NONE) {
          ret += "n";
        }
      }
      return ret;
    } else {
      auto ret = ((is_signed()) ? "int" : "uint") + std::to_string(size_bits());
      if (has_bias()) {
        ret += "b" + std::to_string(bias);
      }
      return ret;
    }
  }

  constexpr bool operator==(ScalarType const& other) const {
    return mantissa == other.mantissa && exponent == other.exponent &&
           bias == other.bias && signed_ == other.signed_ &&
           finite_values_only == other.finite_values_only &&
           nan_repr == other.nan_repr;
  }
};

using ScalarTypeId = ScalarType::Id;

// "rust style" names generally following:
//   https://github.com/pytorch/pytorch/blob/6d9f74f0af54751311f0dd71f7e5c01a93260ab3/torch/csrc/api/include/torch/types.h#L60-L70
static inline constexpr auto kS4 = ScalarType::int_(4);
static inline constexpr auto kU4 = ScalarType::uint(4);
static inline constexpr auto kU4B8 = ScalarType::uint(4, 8);
static inline constexpr auto kS8 = ScalarType::int_(8);
static inline constexpr auto kU8 = ScalarType::uint(8);
static inline constexpr auto kU8B128 = ScalarType::uint(8, 128);

static inline constexpr auto kFE2M1f =
    ScalarType::float_(2, 1, true, ScalarType::NAN_NONE);
static inline constexpr auto kFE3M2f =
    ScalarType::float_(3, 2, true, ScalarType::NAN_NONE);
static inline constexpr auto kFE4M3fn =
    ScalarType::float_(4, 3, true, ScalarType::NAN_EXTD_RANGE_MAX_MIN);
static inline constexpr auto kFE8M0fnu =
    ScalarType(8, 0, false, 0, true, ScalarType::NAN_EXTD_RANGE_MAX_MIN);
static inline constexpr auto kFE5M2 = ScalarType::float_IEEE754(5, 2);
static inline constexpr auto kFE8M7 = ScalarType::float_IEEE754(8, 7);
static inline constexpr auto kFE5M10 = ScalarType::float_IEEE754(5, 10);

// Fixed width style names, generally following:
//  https://github.com/pytorch/pytorch/blob/6d9f74f0af54751311f0dd71f7e5c01a93260ab3/torch/csrc/api/include/torch/types.h#L47-L57
static inline constexpr auto kInt4 = kS4;
static inline constexpr auto kUint4 = kU4;
static inline constexpr auto kUint4b8 = kU4B8;
static inline constexpr auto kInt8 = kS8;
static inline constexpr auto kUint8 = kU8;
static inline constexpr auto kUint8b128 = kU8B128;

static inline constexpr auto kFloat4_e2m1f = kFE2M1f;
static inline constexpr auto kFloat6_e3m2f = kFE3M2f;
static inline constexpr auto kFloat8_e4m3fn = kFE4M3fn;
static inline constexpr auto kFloat8_e5m2 = kFE5M2;
static inline constexpr auto kFloat16_e8m7 = kFE8M7;
static inline constexpr auto kFloat16_e5m10 = kFE5M10;

// colloquial names
static inline constexpr auto kHalf = kFE5M10;
static inline constexpr auto kFloat16 = kHalf;
static inline constexpr auto kBFloat16 = kFE8M7;

static inline constexpr auto kFloat16Id = kFloat16.id();

static_assert(kFloat16.id() == INT64_C(1125899906910725),
              "ScalarType FP16 ABI changed");
static_assert(kFE4M3fn.id() == INT64_C(2814749767172868),
              "ScalarType E4M3FN ABI changed");
static_assert(kU4B8.id() == INT64_C(1125899907892224),
              "ScalarType U4B8 ABI changed");
static_assert(ScalarType::from_id(kFloat16.id()) == kFloat16,
              "ScalarType FP16 ABI does not round trip");
static_assert(ScalarType::from_id(kFE4M3fn.id()) == kFE4M3fn,
              "ScalarType E4M3FN ABI does not round trip");
static_assert(ScalarType::from_id(kU4B8.id()) == kU4B8,
              "ScalarType U4B8 ABI does not round trip");
};  // namespace vllm
