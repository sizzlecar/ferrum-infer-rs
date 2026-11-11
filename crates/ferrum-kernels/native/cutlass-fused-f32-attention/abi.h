#pragma once
#include <stdint.h>

// ABI 1: BMHD contiguous F32 MHA, full bidirectional/all-valid,
// no bias/dropout, FastF32 tensor-op arithmetic. Caller selects an independently declared numerical profile.
struct FerrumCutlassFusedF32AttentionRequestV1 {
  uint32_t abi, size, batches, sequence, heads, dimension;
  float scale;
  uint32_t reserved;
};
struct FerrumCutlassFusedF32AttentionSpanV1 { uint64_t address, elements; };
struct FerrumCutlassFusedF32AttentionBindingsV1 { FerrumCutlassFusedF32AttentionSpanV1 query, key, value, output; };
struct FerrumCutlassFusedF32AttentionInfoV1 {
  uint32_t abi, size, shared_bytes, threads, query_tile, key_tile;
  uint64_t global_workspace_bytes;
};
static_assert(sizeof(FerrumCutlassFusedF32AttentionRequestV1) == 32, "request ABI");
static_assert(sizeof(FerrumCutlassFusedF32AttentionBindingsV1) == 64, "bindings ABI");
static_assert(sizeof(FerrumCutlassFusedF32AttentionInfoV1) == 32, "info ABI");

// Caller owns the current CUDA context, stream, and every allocation until
// completion. No allocation, synchronization or ownership transfer here.
// Positive status: CUDA error; negative: invalid input / caught exception.
// A run error does NOT prove that no work was submitted.
extern "C" int ferrum_cutlass_fused_f32_attention_prepare_v1(const FerrumCutlassFusedF32AttentionRequestV1*, FerrumCutlassFusedF32AttentionInfoV1*) noexcept;
extern "C" int ferrum_cutlass_fused_f32_attention_run_v1(const FerrumCutlassFusedF32AttentionRequestV1*, const FerrumCutlassFusedF32AttentionBindingsV1*, void*) noexcept;
