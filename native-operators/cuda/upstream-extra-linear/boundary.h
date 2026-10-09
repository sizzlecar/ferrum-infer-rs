// Private exception boundary: an upstream host assertion never aborts a server.
#pragma once
#include "format.h"
#include "abi.h"
#include <cstring>
struct FerrumUpstreamInvariant {};
struct FerrumUpstreamCudaFailure { int status; };
#define ggml_abort ferrum_upstream_extra_linear_assertion
#define ggml_cuda_error ferrum_upstream_extra_linear_cuda_error
#define ggml_cuda_get_device ferrum_upstream_extra_linear_get_device
int ferrum_upstream_extra_verify_device(const FerrumUpstreamLinearRequestV1 & request);

template<class F> static int ferrum_upstream_extra_boundary(F && f) noexcept {
    try { return f(); }
    catch (const FerrumUpstreamCudaFailure & error) { return error.status; }
    catch (const FerrumUpstreamInvariant &) { return -6; }
    catch (...) { return -7; }
}
static bool ferrum_upstream_extra_request(const FerrumUpstreamLinearRequestV1 * r) {
    return r && r->abi==1 && r->size==sizeof(*r) && r->reserved==0 && r->rows && r->inputs &&
        r->outputs && r->cc>=800 && r->sm_count && r->shared_limit && r->inputs%256==0 &&
        ferrum_linear_format(r->format) && r->layout<=1;
}
static bool ferrum_upstream_extra_same(const FerrumUpstreamLinearPlanV1 & a,
        const FerrumUpstreamLinearPlanV1 & b) {
    // This ABI consists only of explicit 32/64-bit fields, with no padding.
    return std::memcmp(&a,&b,sizeof(a))==0;
}
