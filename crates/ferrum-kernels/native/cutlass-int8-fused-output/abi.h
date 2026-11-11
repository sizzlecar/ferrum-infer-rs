#pragma once
#include <stdint.h>

// Independent qualification ABI. This does not register a product operator.
struct FerrumInt8OutputShape { uint32_t rows, outputs, inputs, reserved; };
struct FerrumInt8OutputBindings {
  uint64_t activation, weight, row_scale, channel_scale, output;
};
struct FerrumInt8OutputInfo {
  uint64_t workspace_bytes;
  uint32_t tile_m, tile_n, tile_k, threads;
};
struct FerrumInt8OutputComparison {
  uint64_t mismatches, nonfinite;
  uint32_t maximum_absolute_error_bits, reserved;
};

extern "C" int ferrum_int8_fused_output_prepare_v1(
    const FerrumInt8OutputShape*, FerrumInt8OutputInfo*);
extern "C" int ferrum_int8_fused_output_run_v1(
    const FerrumInt8OutputShape*, const FerrumInt8OutputBindings*, void* stream);
extern "C" int ferrum_int8_output_reference_epilogue_v1(
    const FerrumInt8OutputShape*, const FerrumInt8OutputBindings*,
    uint64_t accumulator, void* stream);
extern "C" int ferrum_int8_output_fill_v1(uint64_t output, uint64_t elements,
    uint16_t bits, void* stream);
extern "C" int ferrum_int8_output_compare_v1(uint64_t reference, uint64_t actual,
    uint64_t elements, uint64_t result, void* stream);
