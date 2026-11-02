// Ferrum MIT. Upstream source/math is pinned separately, see source-overlay.json.
#include "abi.h"
#include "mha.cu"
#include <cstring>
#include <limits>

namespace {
constexpr uint32_t kAbi=1;
constexpr uint32_t kHeads=8;
constexpr uint64_t kScratchAlignment=1024;
using Partial = Array2D<LdGrain, nbValidRows, SharedMem::XSmemBuffer::cols>;
static_assert(sizeof(SMemWarpRowMax)==128);
static_assert(sizeof(Partial)==1024);
static_assert(gemm1WarpsPerGrp*nbHeadSplits==1);
static_assert(ctaTile.x==256 && headGrpSize==8 && validElemsPerHead==64 && tokensPerPage==32);

int derive(FerrumXqaRequestV1 const& r,FerrumXqaPlanV1& p) {
  if(r.abi!=kAbi || r.size!=sizeof(r) || r.batch==0 || r.batch>65535 ||
     r.max_pages_per_sequence==0 || r.max_pages_per_sequence>uint32_t(INT32_MAX)/32 ||
     uint64_t(r.binding_stride_bytes)<16+8*uint64_t(r.max_pages_per_sequence) ||
     r.binding_stride_bytes%8 || r.reserved || !r.multiprocessors ||
     r.multiprocessors>uint32_t(INT32_MAX) || r.device_ordinal>uint32_t(INT32_MAX) ||
     !r.maximum_dynamic_shared_bytes) return 1;
  if(r.architecture!=1200 || (r.sliding_window!=0 && r.sliding_window!=128) ||
     sizeof(SharedMem)>r.maximum_dynamic_shared_bytes) return 2;
  uint32_t const splits=std::min(std::max(1U,r.multiprocessors/(r.batch*kHeads)),
                                 (r.max_pages_per_sequence+7)/8);
  uint64_t const count=uint64_t(r.batch)*kHeads*splits;
  uint64_t const bytes=splits>1 ? count*(2*sizeof(SMemWarpRowMax)+sizeof(Partial)) : 0;
  if(bytes>UINT32_MAX || uint64_t(r.batch)*r.binding_stride_bytes>INT64_MAX) return 3;
  p={r,uint32_t(sizeof(SharedMem)),splits,uint64_t(r.batch)*kHeads*4,bytes};
  return 0;
}
int plan_valid(FerrumXqaPlanV1 const* p) {
  if(!p) return 1;
  FerrumXqaPlanV1 expected{};
  int const error=derive(p->request,expected);
  if(error) return error;
  return std::memcmp(&expected,p,sizeof(expected))==0 ? 0 : 5;
}
bool span(FerrumXqaSpan s,uint64_t bytes,uint64_t alignment) {
  if(!bytes) return s.address==0 && s.bytes==0;
  return s.address && s.address%alignment==0 && s.bytes==bytes && s.address<=UINT64_MAX-s.bytes;
}
bool overlap(FerrumXqaSpan a,FerrumXqaSpan b) {
  return a.bytes && b.bytes && a.address<b.address+b.bytes && b.address<a.address+a.bytes;
}
int cuda_status(cudaError_t error) { return error==cudaSuccess ? 0 : 1000+int(error); }
}
extern "C" int ferrum_flashinfer_xqa_plan_v1(FerrumXqaRequestV1 const* r,FerrumXqaPlanV1* p) {
  if(!r || !p) return 1;
  FerrumXqaPlanV1 result{};
  int error=derive(*r,result); if(error) return error;
  int ordinal=-1,major=0,minor=0,sm=0,maximum=0;
  if((error=cuda_status(cudaGetDevice(&ordinal)))) return error;
  if(ordinal!=int(r->device_ordinal)) return 2;
  for(auto item : {std::pair{cudaDevAttrComputeCapabilityMajor,&major},
                   std::pair{cudaDevAttrComputeCapabilityMinor,&minor},
                   std::pair{cudaDevAttrMultiProcessorCount,&sm},
                   std::pair{cudaDevAttrMaxSharedMemoryPerBlockOptin,&maximum}}) {
    if((error=cuda_status(cudaDeviceGetAttribute(item.second,item.first,ordinal)))) return error;
  }
  if(major*100+minor*10!=int(r->architecture) || sm!=int(r->multiprocessors) ||
     uint64_t(maximum)<r->maximum_dynamic_shared_bytes) return 2;
  // Explicit prepare, before capture: verify host/device layout agrees and set
  // shared capacity on the current context. No load-time CUDA side effects.
  uint32_t device_shared=0;
  if((error=cuda_status(cudaMemcpyFromSymbol(&device_shared,smemSize,sizeof(device_shared))))) return error;
  if(device_shared!=result.shared_bytes) return 5;
  if((error=cuda_status(cudaFuncSetAttribute(kernel_mha,cudaFuncAttributeMaxDynamicSharedMemorySize,
                                             int(result.shared_bytes))))) return error;
  *p=result;
  return 0;
}
extern "C" int ferrum_flashinfer_xqa_validate_v1(FerrumXqaPlanV1 const* p,FerrumXqaBindingsV1 const* b) {
  int const error=plan_valid(p); if(error) return error; if(!b) return 1;
  uint64_t const qbytes=uint64_t(p->request.batch)*64*64*2;
  if(!span(b->query,qbytes,16) || !span(b->sinks_f32,256,4) ||
     !span(b->packed_metadata,uint64_t(p->request.batch)*p->request.binding_stride_bytes,8) ||
     !span(b->output,qbytes,16) || !span(b->semaphores,p->semaphore_bytes,4) ||
     !span(b->scratch,p->scratch_bytes,kScratchAlignment)) return 4;
  FerrumXqaSpan const writes[]={b->output,b->semaphores,b->scratch};
  for(int i=0;i<3;++i) {
    for(auto read : {b->query,b->sinks_f32,b->packed_metadata}) if(overlap(writes[i],read)) return 4;
    for(int j=i+1;j<3;++j) if(overlap(writes[i],writes[j])) return 4;
  }
  return 0;
}
extern "C" int ferrum_flashinfer_xqa_run_v1(FerrumXqaPlanV1 const* p,FerrumXqaBindingsV1 const* b,void* raw_stream) {
  int error=ferrum_flashinfer_xqa_validate_v1(p,b); if(error) return error;
  if(!raw_stream) return 1;
  auto stream=reinterpret_cast<cudaStream_t>(raw_stream);
  // Only multi-block execution reads/increments semaphores. Single-split
  // execution never accesses them and needs no reset node in CUDA graphs.
  // Multi-block reset remains on the same stream for every invocation.
  // A failure after this point may leave pending work; caller retains ownership.
  if(p->subsequences>1 &&
     (error=cuda_status(cudaMemsetAsync(reinterpret_cast<void*>(b->semaphores.address),0,
                                       p->semaphore_bytes,stream)))) return error;
  auto const& r=p->request;
  uint32_t const max_seq=r.max_pages_per_sequence*32;
  try {
    launchMHAFlashInfer(r.multiprocessors,8,r.sliding_window ? r.sliding_window : max_seq,
      1.0f,nullptr,reinterpret_cast<OutputHead*>(b->output.address),
      reinterpret_cast<InputHead const*>(b->query.address),reinterpret_cast<float const*>(b->sinks_f32.address),
      reinterpret_cast<GMemCacheHead*>(uintptr_t(0)),reinterpret_cast<GMemCacheHead*>(uintptr_t(1024)),
      reinterpret_cast<KVCachePageIndex const*>(b->packed_metadata.address+16),max_seq,
      reinterpret_cast<uint32_t const*>(b->packed_metadata.address),r.batch,1.0f,nullptr,
      reinterpret_cast<uint32_t*>(b->semaphores.address),reinterpret_cast<void*>(b->scratch.address),
      false,32768,1024,64,r.binding_stride_bytes,stream);
    return cuda_status(cudaPeekAtLastError());
  } catch(...) { return 6; }
}
