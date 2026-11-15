/*
 * Adapted from FastVideo attn_qat_infer/blackwell/launch.h.
 * Copyright (c) 2025 by SageAttention team.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once
#include "abi.h"
#include "cute/tensor.hpp"
#include "cutlass/cluster_launch.hpp"
#include "cutlass/version.h"
#include "sage/params.h"
#include "sage/kernel_traits.h"
#include "sage/kernel_ws.h"
#include <cstring>

namespace ferrum_sage_vsa {
static_assert(CUTLASS_MAJOR == 4 && CUTLASS_MINOR == 3 && CUTLASS_PATCH == 0,
              "This source profile requires the deployed CUTLASS 4.3.0 closure");
using Traits = Flash_fwd_kernel_traits<128, 128, 128, 3, 1, false,
    cutlass::nv_float4_t<cutlass::float_e2m1_t>, cutlass::bfloat16_t>;
using Mainloop = flash::CollectiveMainloopFwd<Traits, false>;
using Epilogue = flash::CollectiveEpilogueFwd<Traits>;
using Scheduler = flash::StaticPersistentTileScheduler;
static_assert(Traits::kNWarps * 32 == 384);
static_assert(cute::size<0>(Traits::ClusterShape_MNK{}) == 1 &&
              cute::size<1>(Traits::ClusterShape_MNK{}) == 1 &&
              cute::size<2>(Traits::ClusterShape_MNK{}) == 1);

inline void* pointer(FerrumSageVsaFp4SpanV1 s) {
  return reinterpret_cast<void*>(uintptr_t(s.address));
}
inline Flash_fwd_params parameters(const FerrumSageVsaFp4RequestV1& r,
                                  const FerrumSageVsaFp4BindingsV1& x) {
  Flash_fwd_params p{};
  const int64_t q = int64_t(r.query_blocks) * 128, k = int64_t(r.key_blocks) * 128;
  p.q_ptr = pointer(x.query); p.k_ptr = pointer(x.key); p.v_ptr = pointer(x.value);
  p.sfq_ptr = pointer(x.query_scales); p.sfk_ptr = pointer(x.key_scales); p.sfv_ptr = pointer(x.value_scales);
  p.delta_s_ptr = pointer(x.zero_delta); p.o_ptr = pointer(x.output); p.softmax_lse_ptr = pointer(x.lse);
  // Q/K/V strides count unpacked E2M1 elements. SF and BF16 strides count
  // their respective physical elements, as in the fixed Python wrapper.
  p.q_row_stride = p.k_row_stride = p.o_row_stride = 128; p.v_row_stride = k;
  p.q_head_stride = p.o_head_stride = q * 128; p.k_head_stride = p.v_head_stride = k * 128;
  p.q_batch_stride = p.o_batch_stride = r.heads * q * 128;
  p.k_batch_stride = p.v_batch_stride = r.heads * k * 128;
  p.sfq_row_stride = p.sfk_row_stride = 8; p.sfv_row_stride = k / 16;
  p.sfq_head_stride = q * 8; p.sfk_head_stride = p.sfv_head_stride = k * 8;
  p.sfq_batch_stride = r.heads * q * 8; p.sfk_batch_stride = p.sfv_batch_stride = r.heads * k * 8;
  // The nonsmoothed-Q path still reads a zero [B,H,1,KV] delta table.
  // Mainloop ignores these nominal DS strides: its SmemLayoutAtomDS uses
  // query stride 0, so logical seqlen_s=128 is exactly one physical row.
  p.ds_row_stride = p.ds_head_stride = k; p.ds_batch_stride = r.heads * k;
  p.b = r.batches; p.h = p.h_k = r.heads; p.h_h_k_ratio = 1;
  p.seqlen_q = p.seqlen_q_rounded = q;
  p.seqlen_k = p.seqlen_k_rounded = p.unpadded_seqlen_k = k;
  p.d = p.d_rounded = 128; p.seqlen_s = 128;
  p.head_divmod = cutlass::FastDivmod(p.h);
  p.m_block_divmod = cutlass::FastDivmod(r.query_blocks);
  p.total_blocks = int64_t(r.batches) * r.heads * r.query_blocks;
  std::memcpy(&p.scale_softmax, &r.softmax_scale_bits, sizeof(float));
  // Original API multiplies by the double M_LOG2E before narrowing to F32.
  p.scale_softmax_log2 = p.scale_softmax * 1.44269504088896340736;
  const __half scale = __float2half(p.scale_softmax_log2);
  const __half2 scale2(scale, scale);
  std::memcpy(&p.scale_softmax_log2_half2, &scale2, sizeof(scale2));
  p.p_dropout = p.rp_dropout = 1.f; p.p_dropout_in_uint8_t = 255;
  p.scale_softmax_rp_dropout = p.scale_softmax;
  p.window_size_left = p.window_size_right = -1;
  p.is_bf16 = p.is_seqlens_k_cumulative = true;
  p.q2k_idx = static_cast<const int*>(pointer(x.indices));
  p.q2k_num = static_cast<const int*>(pointer(x.counts));
  p.kv_valid = static_cast<const int*>(pointer(x.half_valid));
  p.q2k_max = r.key_blocks; p.num_m_blocks = r.query_blocks;
  // false: causal, block mean, single-level P; null: quadrant, semaphore.
  return p;
}

inline int launch(Flash_fwd_params& p, uint32_t sms, cudaStream_t stream) {
  auto mainloop = Mainloop::to_underlying_arguments({
    static_cast<Traits::Element const*>(p.q_ptr), {p.seqlen_q, p.d, p.h, p.b},
    {p.q_row_stride, cute::_1{}, p.q_head_stride, p.q_batch_stride},
    static_cast<Traits::Element const*>(p.k_ptr), {p.seqlen_k, p.d, p.h_k, p.b},
    {p.k_row_stride, cute::_1{}, p.k_head_stride, p.k_batch_stride},
    {p.unpadded_seqlen_k, p.d, p.h_k, p.b},
    static_cast<Traits::Element const*>(p.v_ptr), {p.d, p.seqlen_k, p.h_k, p.b},
    {p.v_row_stride, cute::_1{}, p.v_head_stride, p.v_batch_stride},
    static_cast<Traits::ElementSF const*>(p.sfq_ptr), {p.seqlen_q, p.d, p.h, p.b},
    static_cast<Traits::ElementSF const*>(p.sfk_ptr), {p.seqlen_k, p.d, p.h_k, p.b},
    static_cast<Traits::ElementSF const*>(p.sfv_ptr), {p.d, p.seqlen_k, p.h_k, p.b},
    static_cast<float const*>(p.delta_s_ptr), {p.seqlen_s, p.seqlen_k, p.h_k, p.b},
    {p.ds_row_stride, cute::_1{}, p.ds_head_stride, p.ds_batch_stride},
    p.scale_softmax_log2, p.q2k_idx, p.q2k_num, p.q2k_max,
    p.num_m_blocks, p.h, p.kv_valid, p.q2k_quad
  });
  auto epilogue = Epilogue::to_underlying_arguments({
    static_cast<Traits::ElementOut*>(p.o_ptr), {p.seqlen_q, p.d, p.h, p.b},
    {p.o_row_stride, cute::_1{}, p.o_head_stride, p.o_batch_stride},
    static_cast<float*>(p.softmax_lse_ptr), {cute::_1{}, p.seqlen_q, p.h * p.seqlen_q}
  });
  const Scheduler::Arguments args{p.num_m_blocks, p.h, p.b};
  auto scheduler = Scheduler::to_underlying_arguments(args);
  void* kernel = reinterpret_cast<void*>(flash::compute_attn_ws<Traits, false, Scheduler>);
  constexpr int shared = sizeof(Traits::SharedStorage);
  if (shared >= 48 * 1024) {
    const auto status = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared);
    if (status != cudaSuccess) return int(status);
  }
  const cutlass::ClusterLaunchParams config{
      Scheduler::get_grid_dim(args, sms), dim3(Traits::kNWarps * 32), dim3(1, 1, 1), shared, stream};
  const auto status = cutlass::launch_kernel_on_cluster(config, kernel, p, mainloop, epilogue, scheduler);
  if (status != cutlass::Status::kSuccess) return -5;
  return int(cudaPeekAtLastError());
}
} // namespace ferrum_sage_vsa
