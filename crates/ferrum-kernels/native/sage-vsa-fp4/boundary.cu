#include "abi.h"
#include "boundary_checks.cuh"
#include "launch.cuh"
#include <cstring>

namespace {
using namespace ferrum_sage_vsa;
int derive(const FerrumSageVsaFp4RequestV1& r, FerrumSageVsaFp4PlanV1& p) {
  int status = request_valid(r); if (status) return status;
  status = device_valid(r); if (status) return status;
  constexpr uint64_t shared = sizeof(Traits::SharedStorage);
  if (!shared || shared > r.maximum_dynamic_shared_bytes || shared > UINT32_MAX) return -2;
  p = {}; p.request = r; p.abi = 1; p.size = sizeof(p); p.kernel_id = 1;
  p.shared_bytes = shared; p.threads = Traits::kNWarps * 32;
  p.grid_blocks = r.multiprocessors; p.upload_count = 3; p.memset_count = 1;
  status = sizes(r, p.bytes); if (status) return status;
  // The original DS atom has stride (0,1): its logical 128 query rows
  // share one physical KV row. Verify the compiled layout's exact footprint.
  const auto ds = cute::tile_to_shape(Traits::SmemLayoutAtomDS{},
      cute::make_shape(128,int(r.key_blocks)*128,int(r.heads),int(r.batches)),
      cute::Step<cute::_2,cute::_1,cute::_3,cute::_4>{});
  if (uint64_t(cute::cosize(ds)) != p.bytes.zero_delta / sizeof(float)) return -5;
  return 0;
}
int validate_bindings(const FerrumSageVsaFp4PlanV1& p, const FerrumSageVsaFp4BindingsV1& b) {
  int status;
  const FerrumSageVsaFp4SpanV1 spans[] = {b.query,b.key,b.value,b.query_scales,b.key_scales,b.value_scales,
      b.output,b.lse,b.zero_delta,b.indices,b.counts,b.half_valid};
  const auto& s = p.bytes;
  const uint64_t bytes[] = {s.query,s.key,s.value,s.query_scales,s.key_scales,s.value_scales,
      s.output,s.lse,s.zero_delta,s.indices,s.counts,s.half_valid};
  const uint64_t align[] = {16,16,16,16,16,16,16,4,16,4,4,4};
  for (int i = 0; i < 12; ++i) {
    if (!span(spans[i], bytes[i], align[i])) return -4;
    status = pointer_kind(spans[i], cudaMemoryTypeDevice, p.request.device_ordinal);
    if (status) return status;
    if (i >= 6) for (int j = 0; j < i; ++j) if (!disjoint(spans[i],spans[j])) return -4;
  }
  return 0;
}
int validate(const FerrumSageVsaFp4PlanV1& p, const FerrumSageVsaFp4BindingsV1& b,
             const FerrumSageVsaFp4HostListsV1& h) {
  FerrumSageVsaFp4PlanV1 expected{};
  int status = derive(p.request, expected); if (status) return status;
  if (std::memcmp(&p, &expected, sizeof(p))) return -5;
  status = validate_bindings(p,b); if (status) return status;
  const FerrumSageVsaFp4SpanV1 writes[] = {b.output,b.lse,b.zero_delta,b.indices,b.counts,b.half_valid};
  const uint64_t bytes[] = {p.bytes.indices,p.bytes.counts,p.bytes.half_valid};
  const FerrumSageVsaFp4SpanV1 hosts[] = {h.indices,h.counts,h.half_valid};
  for (int i = 0; i < 3; ++i) {
    if (!span(hosts[i],bytes[i],4)) return -4;
    status = pointer_kind(hosts[i],cudaMemoryTypeHost,p.request.device_ordinal);
    if (status) return status;
    // Separate CPU allocations are not required, but device writes may never
    // alias an upload source, even under a raw caller's forged bindings.
    for (auto write : writes) if (!disjoint(hosts[i],write)) return -4;
  }
  return list_contents(p.request,h) ? 0 : -6;
}
int run(const FerrumSageVsaFp4PlanV1& p, const FerrumSageVsaFp4BindingsV1& b,
        const FerrumSageVsaFp4HostListsV1& h, void* raw_stream) {
  int status = validate(p,b,h); if (status) return status;
  cudaStream_t stream;
  status = stream_valid(raw_stream,stream); if (status) return status;
  const FerrumSageVsaFp4SpanV1 src[] = {h.indices,h.counts,h.half_valid};
  const FerrumSageVsaFp4SpanV1 dst[] = {b.indices,b.counts,b.half_valid};
  // From this point every failure may follow a submitted copy. The caller's
  // managed command must retain all leases and its completion/quarantine path.
  for (int i = 0; i < 3; ++i) {
    const auto error = cudaMemcpyAsync(pointer(dst[i]),pointer(src[i]),src[i].bytes,cudaMemcpyHostToDevice,stream);
    if (error != cudaSuccess) return int(error);
  }
  const auto error = cudaMemsetAsync(pointer(b.zero_delta),0,b.zero_delta.bytes,stream);
  if (error != cudaSuccess) return int(error);
  auto params = parameters(p.request,b);
  return launch(params,p.request.multiprocessors,stream);
}
}
extern "C" int ferrum_sage_vsa_fp4_plan_v1(const FerrumSageVsaFp4RequestV1* r, FerrumSageVsaFp4PlanV1* p) noexcept {
  if (!r || !p) return -1;
  try { FerrumSageVsaFp4PlanV1 value{}; const int status = derive(*r,value); if (!status) *p = value; return status; }
  catch (...) { return -8; }
}
extern "C" int ferrum_sage_vsa_fp4_validate_v1(const FerrumSageVsaFp4PlanV1* p,
    const FerrumSageVsaFp4BindingsV1* b, const FerrumSageVsaFp4HostListsV1* h) noexcept {
  if (!p || !b || !h) return -1;
  try { return validate(*p,*b,*h); } catch (...) { return -8; }
}
extern "C" int ferrum_sage_vsa_fp4_run_v1(const FerrumSageVsaFp4PlanV1* p,
    const FerrumSageVsaFp4BindingsV1* b, const FerrumSageVsaFp4HostListsV1* h, void* stream) noexcept {
  if (!p || !b || !h) return -1;
  try { return run(*p,*b,*h,stream); } catch (...) { return -8; }
}

#include "region_boundary.cuh"
#include "region_kernel_boundary.cuh"
#include "quantize_boundary.cuh"
