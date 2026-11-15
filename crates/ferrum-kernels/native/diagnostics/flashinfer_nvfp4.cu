// Qualification-only adapter. Build against an independently pinned FlashInfer
// source tree; never include this translation unit in the production archive.
// The numerical implementation is the unmodified upstream helper below.
#include <cfloat>
#include "tensorrt_llm/kernels/quantization_utils.cuh"
#include <climits>
#include <cstdint>

namespace {
using namespace tensorrt_llm::kernels;
__global__ void reference(const __nv_bfloat16* input, const float* global,
    uint64_t* packed, uint32_t* scales, int rows, int width) {
  const uint64_t groups = width / 16;
  const uint64_t padded_groups = (groups + 3) / 4 * 4;
  const uint64_t count = ((uint64_t(rows) + 127) / 128 * 128) * padded_groups;
  for (uint64_t index = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
       index < count; index += uint64_t(gridDim.x) * blockDim.x) {
    const int row = int(index / padded_groups), group = int(index % padded_groups);
    // This is the upstream offset helper, not Ferrum's scale-index function.
    auto* sf = cvt_quant_to_fp4_get_sf_out_offset<uint32_t, 16, 1>(row, group, width, scales);
    if (row >= rows || uint64_t(group) >= groups) { *sf = 0; continue; }
    PackedVec<__nv_bfloat16, 16> value =
      *reinterpret_cast<const PackedVec<__nv_bfloat16, 16>*>(input + uint64_t(row) * width + group * 16);
    packed[uint64_t(row) * groups + group] =
      cvt_warp_fp16_to_fp4<__nv_bfloat16, 16, 16, false, false>(value, *global, sf);
  }
}
}

// All memory and the explicit non-capturing stream are owned by Ferrum's
// managed invocation. No allocation, implicit stream, synchronization or cache.
extern "C" int ferrum_reference_nvfp4_quantize_bf16_v1(
    const uint16_t* input, const float* global, uint8_t* packed, uint8_t* scales,
    uint32_t rows, uint32_t width, void* raw_stream) noexcept {
  if (!input || !global || !packed || !scales || !raw_stream || !rows ||
      rows > INT_MAX || !width || width > INT_MAX || width % 16) return -1;
  if ((reinterpret_cast<uintptr_t>(input) % 16) ||
      (reinterpret_cast<uintptr_t>(global) % 4) ||
      (reinterpret_cast<uintptr_t>(packed) % 16) ||
      (reinterpret_cast<uintptr_t>(scales) % 16)) return -2;
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(raw_stream);
  cudaStreamCaptureStatus capture;
  auto status = cudaStreamIsCapturing(stream, &capture);
  if (status != cudaSuccess) return int(status);
  if (capture != cudaStreamCaptureStatusNone) return -3;
  reference<<<128, 128, 0, stream>>>(reinterpret_cast<const __nv_bfloat16*>(input),
    global, reinterpret_cast<uint64_t*>(packed), reinterpret_cast<uint32_t*>(scales),
    int(rows), int(width));
  return int(cudaPeekAtLastError());
}
