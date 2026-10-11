// Ferrum-owned host boundary; the fixed upstream header closure declares these.
#include "boundary.h"
#include <cuda_runtime.h>
#include <utility>
#include <initializer_list>
extern "C" [[noreturn]] void ferrum_upstream_extra_linear_assertion(const char *, int, const char *, ...) {
    throw FerrumUpstreamInvariant{};
}
[[noreturn]] void ferrum_upstream_extra_linear_cuda_error(const char *, const char *, const char *, int, const char *) {
    const auto error = cudaPeekAtLastError();
    if (error!=cudaSuccess) throw FerrumUpstreamCudaFailure{static_cast<int>(error)};
    throw FerrumUpstreamInvariant{};
}
int ferrum_upstream_extra_linear_get_device() {
    int device = -1;
    const auto error = cudaGetDevice(&device);
    if (error!=cudaSuccess) throw FerrumUpstreamCudaFailure{static_cast<int>(error)};
    return device;
}

int ferrum_upstream_extra_verify_device(const FerrumUpstreamLinearRequestV1 & r) {
    int device=-1, major=0, minor=0, sm=0, shared=0;
    auto status = cudaGetDevice(&device);
    if (status!=cudaSuccess) return status;
    const std::pair<cudaDeviceAttr,int *> attributes[]={{cudaDevAttrComputeCapabilityMajor,&major},
            {cudaDevAttrComputeCapabilityMinor,&minor}, {cudaDevAttrMultiProcessorCount,&sm},
            {cudaDevAttrMaxSharedMemoryPerBlockOptin,&shared}};
    for (auto entry : attributes) {
        status = cudaDeviceGetAttribute(entry.second,entry.first,device);
        if (status!=cudaSuccess) return status;
    }
    return r.cc==uint32_t(major*100+minor*10) && r.sm_count==uint32_t(sm) &&
        r.shared_limit<=uint64_t(shared) ? 0 : -2;
}
