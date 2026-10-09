// Private exception boundary: an upstream host assertion never aborts a server.
#pragma once
#include "abi.h"
#include <cstring>
struct FerrumUpstreamQ6F32Invariant {};
struct FerrumUpstreamQ6F32CudaFailure { int status; };
#define ggml_abort ferrum_upstream_q6_f32_assertion
#define ggml_cuda_error ferrum_upstream_q6_f32_cuda_error
#define ggml_cuda_get_device ferrum_upstream_q6_f32_get_device
int ferrum_upstream_q6_f32_verify_device(const FerrumUpstreamQ6F32RequestV1 & request);

template<class F> static int ferrum_upstream_q6_f32_boundary(F && f) noexcept {
    try { return f(); }
    catch (const FerrumUpstreamQ6F32CudaFailure & error) { return error.status; }
    catch (const FerrumUpstreamQ6F32Invariant &) { return -6; }
    catch (...) { return -7; }
}
static bool ferrum_upstream_q6_f32_request(const FerrumUpstreamQ6F32RequestV1 * r) {
    return r && r->abi==1 && r->size==sizeof(*r) && r->reserved==0 && r->rows && r->inputs &&
        r->outputs && r->cc>=800 && r->sm_count && r->shared_limit && r->inputs%256==0 &&
        r->format==14 && r->layout==0 && r->rows<=32;
}
static bool ferrum_upstream_q6_f32_same(const FerrumUpstreamQ6F32PlanV1 & a,
        const FerrumUpstreamQ6F32PlanV1 & b) {
    // This ABI consists only of explicit 32/64-bit fields, with no padding.
    return std::memcmp(&a,&b,sizeof(a))==0;
}
