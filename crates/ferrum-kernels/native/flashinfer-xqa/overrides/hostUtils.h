// Ferrum overlay of the pinned FlashInfer launch helper. PDL is not enabled by
// this ABI. Omitting its disabled attribute also removes mutable process-global
// host state, so independent callers/streams do not race on launch metadata.
#pragma once
#include <cuda_runtime.h>
#include <stdexcept>
inline cudaLaunchConfig_t makeLaunchConfig(dim3 const& gridDim, dim3 const& ctaDim,
                                           size_t dynShmBytes, cudaStream_t stream, bool usePDL) {
  if (usePDL) throw std::invalid_argument("Ferrum XQA ABI does not enable PDL");
  return cudaLaunchConfig_t{gridDim, ctaDim, dynShmBytes, stream, nullptr, 0};
}
