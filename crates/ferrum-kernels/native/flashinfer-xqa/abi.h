// Ferrum MIT. Allocation-free boundary for pinned FlashInfer XQA.
#pragma once
#include <cstdint>
struct FerrumXqaRequestV1 {
  uint32_t abi, size, batch, max_pages_per_sequence, binding_stride_bytes, sliding_window;
  uint32_t device_ordinal, architecture, multiprocessors, reserved;
  uint64_t maximum_dynamic_shared_bytes;
};
struct FerrumXqaPlanV1 {
  FerrumXqaRequestV1 request;
  uint32_t shared_bytes, subsequences;
  uint64_t semaphore_bytes, scratch_bytes;
};
struct FerrumXqaSpan { uint64_t address, bytes; };
struct FerrumXqaBindingsV1 {
  FerrumXqaSpan query, sinks_f32, packed_metadata, output, semaphores, scratch;
};
static_assert(sizeof(FerrumXqaRequestV1)==48);
static_assert(sizeof(FerrumXqaPlanV1)==72);
static_assert(sizeof(FerrumXqaBindingsV1)==96);
extern "C" int ferrum_flashinfer_xqa_plan_v1(FerrumXqaRequestV1 const*, FerrumXqaPlanV1*);
extern "C" int ferrum_flashinfer_xqa_validate_v1(FerrumXqaPlanV1 const*, FerrumXqaBindingsV1 const*);
extern "C" int ferrum_flashinfer_xqa_run_v1(FerrumXqaPlanV1 const*, FerrumXqaBindingsV1 const*, void*);
