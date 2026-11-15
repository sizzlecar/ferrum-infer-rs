/*
 * Copyright (c) 2025 by SageAttention team.
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
// Adapted from fixed FastVideo attn_qat_infer/quantization/fp4_quantization_4d.cu.
// Original complete source SHA256: f04880acf1e73c48b817c355f0290528c0cd9cad18a7e7d09e18ed8300e41a4e
// Only Torch dispatch/bindings and unused includes are removed. The original
// numerical kernels, 16 elements/thread and physical layouts are preserved.
#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdint.h>
#include <type_traits>

namespace ferrum_sage_vsa_quantization {
 constexpr int CVT_FP4_ELTS_PER_THREAD = 16;
 
 // Convert 4 float2 values into 8 e2m1 values (represented as one uint32_t).
 inline __device__ uint32_t fp32_vec_to_e2m1(float2 *array) {
 #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
   uint32_t val;
   asm volatile(
       "{\n"
       ".reg .b8 byte0;\n"
       ".reg .b8 byte1;\n"
       ".reg .b8 byte2;\n"
       ".reg .b8 byte3;\n"
       "cvt.rn.satfinite.e2m1x2.f32   byte0, %2, %1;\n"
       "cvt.rn.satfinite.e2m1x2.f32   byte1, %4, %3;\n"
       "cvt.rn.satfinite.e2m1x2.f32   byte2, %6, %5;\n"
       "cvt.rn.satfinite.e2m1x2.f32   byte3, %8, %7;\n"
       "mov.b32 %0, {byte0, byte1, byte2, byte3};\n"
       "}"
       : "=r"(val)
       : "f"(array[0].x), "f"(array[0].y), "f"(array[1].x), "f"(array[1].y),
         "f"(array[2].x), "f"(array[2].y), "f"(array[3].x), "f"(array[3].y));
   return val;
 #else
   return 0;
 #endif
 }
 
 // Get type2 from type or vice versa (applied to half and bfloat16)
 template <typename T>
 struct TypeConverter {
   using Type = half2;
 };  // keep for generality
 
 template <>
 struct TypeConverter<half2> {
   using Type = half;
 };
 
 template <>
 struct TypeConverter<half> {
   using Type = half2;
 };
 
 template <>
 struct TypeConverter<__nv_bfloat162> {
   using Type = __nv_bfloat16;
 };
 
 template <>
 struct TypeConverter<__nv_bfloat16> {
   using Type = __nv_bfloat162;
 };
 
 // Define a 32 bytes packed data type.
 template <class Type>
 struct PackedVec {
   typename TypeConverter<Type>::Type elts[8];
 };
 
 template <uint32_t head_dim, uint32_t BLOCK_SIZE, bool permute, typename T>
 __global__ void scaled_fp4_quant_kernel(
     const T* input, uint8_t* output, uint8_t* output_sf,
     int batch_size, int num_heads, int num_tokens,
     int stride_bz_input, int stride_h_input, int stride_seq_input,
     int stride_bz_output, int stride_h_output, int stride_seq_output,
     int stride_bz_output_sf, int stride_h_output_sf, int stride_seq_output_sf) {
   static_assert(std::is_same<T, half>::value || std::is_same<T, nv_bfloat16>::value, "Only half and bfloat16 input are supported");
   using PackedVec = PackedVec<T>;
 
   const int batch_id = blockIdx.y;
   const int head_id = blockIdx.z;
   const int token_block_id = blockIdx.x;
 
   static_assert(CVT_FP4_ELTS_PER_THREAD == 8 || CVT_FP4_ELTS_PER_THREAD == 16,
                 "CVT_FP4_ELTS_PER_THREAD must be 8 or 16");
   static_assert(sizeof(PackedVec) == sizeof(T) * CVT_FP4_ELTS_PER_THREAD,
                 "Vec size is not matched.");
 
   constexpr uint32_t NUM_THREADS_PER_TOKEN = head_dim / CVT_FP4_ELTS_PER_THREAD;
 
   // load input
   const int token_id = token_block_id * BLOCK_SIZE + threadIdx.x / NUM_THREADS_PER_TOKEN;
   
   int load_token_id;
   if constexpr (!permute) {
     load_token_id = token_id;
   } else {
     int local_token_id = threadIdx.x / NUM_THREADS_PER_TOKEN;
     int local_token_id_residue = local_token_id % 32;
     // [0, 1, 8, 9, 16, 17, 24, 25, 2, 3, 10, 11, 18, 19, 26, 27, 4, 5, 12, 13, 20, 21, 28, 29, 6, 7, 14, 15, 22, 23, 30, 31]
     load_token_id = token_block_id * BLOCK_SIZE + (local_token_id / 32) * 32 +
                     (local_token_id_residue / 8) * 2 + 
                     ((local_token_id_residue % 8) / 2) * 8 +
                     (local_token_id_residue % 8) % 2;
   }
 
   PackedVec in_vec;
   
   #pragma unroll
   for (int i = 0; i < CVT_FP4_ELTS_PER_THREAD / 2; i++) {
     reinterpret_cast<uint32_t&>(in_vec.elts[i]) = 0;
   }
   
   if (load_token_id < num_tokens) {
     in_vec = reinterpret_cast<PackedVec const*>(input + 
                                           batch_id * stride_bz_input + // batch dim
                                           head_id * stride_h_input +   // head dim
                                           load_token_id * stride_seq_input + // seq dim
                                           (threadIdx.x % NUM_THREADS_PER_TOKEN) * CVT_FP4_ELTS_PER_THREAD)[0]; // feature dim
   }
 
   // calculate max of every consecutive 16 elements
   auto localMax = __habs2(in_vec.elts[0]);
   #pragma unroll
   for (int i = 1; i < CVT_FP4_ELTS_PER_THREAD / 2; i++) { // local max
     localMax = __hmax2(localMax, __habs2(in_vec.elts[i]));
   }
 
   if constexpr (CVT_FP4_ELTS_PER_THREAD == 8) { // shuffle across two threads
     localMax = __hmax2(__shfl_xor_sync(0xffffffff, localMax, 1, 32), localMax);
   }
 
   float vecMax = float(__hmax(localMax.x, localMax.y));
 
   // scaling factor
   float SFValue = vecMax / 6.0f;
   uint8_t SFValueFP8;
   reinterpret_cast<__nv_fp8_e4m3&>(SFValueFP8) = __nv_fp8_e4m3(SFValue);
   SFValue = float(reinterpret_cast<__nv_fp8_e4m3&>(SFValueFP8));
 
   float SFValueInv = (SFValue == 0.0f) ? 0.0f : 1.0f / SFValue;
 
   // convert input to float2 and apply scale
   float2 fp2Vals[CVT_FP4_ELTS_PER_THREAD / 2];
 
   #pragma unroll
   for (int i = 0; i < CVT_FP4_ELTS_PER_THREAD / 2; i++) {
     if constexpr (std::is_same<T, half>::value) {
       fp2Vals[i] = __half22float2(in_vec.elts[i]);
     } else {
       fp2Vals[i] = __bfloat1622float2(in_vec.elts[i]);
     }
     fp2Vals[i].x = fp2Vals[i].x * SFValueInv;
     fp2Vals[i].y = fp2Vals[i].y * SFValueInv;
   }
 
   // convert to e2m1
   uint32_t e2m1Vals[CVT_FP4_ELTS_PER_THREAD / 8];
   #pragma unroll
   for (int i = 0; i < CVT_FP4_ELTS_PER_THREAD / 8; i++) {
     e2m1Vals[i] = fp32_vec_to_e2m1(fp2Vals + i * 4);
   }
 
   // Skip out-of-range tokens: never write past the sequence when num_tokens
   // is not a multiple of BLOCK_SIZE (out-of-bounds write fix).
   if (token_id >= num_tokens) return;

   // save
   if constexpr (CVT_FP4_ELTS_PER_THREAD == 8) {
     reinterpret_cast<uint32_t*>(output + 
                                 batch_id * stride_bz_output +
                                 head_id * stride_h_output +
                                 token_id * stride_seq_output +
                                 (threadIdx.x % NUM_THREADS_PER_TOKEN) * CVT_FP4_ELTS_PER_THREAD / 2)[0] = e2m1Vals[0];
   } else {
     reinterpret_cast<uint64_t*>(output + 
                                 batch_id * stride_bz_output +
                                 head_id * stride_h_output +
                                 token_id * stride_seq_output +
                                 (threadIdx.x % NUM_THREADS_PER_TOKEN) * CVT_FP4_ELTS_PER_THREAD / 2)[0] = reinterpret_cast<uint64_t*>(e2m1Vals)[0];
   }
   
   uint8_t* output_sf_save_base = output_sf + batch_id * stride_bz_output_sf + head_id * stride_h_output_sf + (token_id / 64) * 64 * stride_seq_output_sf;
   uint32_t token_id_local = token_id % 64;
 
   if constexpr (CVT_FP4_ELTS_PER_THREAD == 16) {
     uint32_t col_id_local = threadIdx.x % NUM_THREADS_PER_TOKEN;
     uint32_t offset_local = (col_id_local / 4) * 256 + (col_id_local % 4) + 
                             (token_id_local / 16) * 4 + (token_id_local % 16) * 16;
     reinterpret_cast<uint8_t*>(output_sf_save_base + offset_local)[0] = SFValueFP8;
   } else {
     if (threadIdx.x % 2 == 0) {
       uint32_t col_id_local = (threadIdx.x % NUM_THREADS_PER_TOKEN) / 2;
       uint32_t offset_local = (col_id_local / 4) * 256 + (col_id_local % 4) + 
                             (token_id_local / 16) * 4 + (token_id_local % 16) * 16;
       reinterpret_cast<uint8_t*>(output_sf_save_base + offset_local)[0] = SFValueFP8;
     }
   }
 }
 
 template <uint32_t head_dim, uint32_t BLOCK_SIZE, typename T>
 __global__ void scaled_fp4_quant_trans_kernel(
     const T* input, uint8_t* output, uint8_t* output_sf,
     int batch_size, int num_heads, int num_tokens,
     int stride_bz_input, int stride_h_input, int stride_seq_input,
     int stride_bz_output, int stride_h_output, int stride_d_output,
     int stride_bz_output_sf, int stride_h_output_sf, int stride_d_output_sf) {
   static_assert(std::is_same<T, half>::value || std::is_same<T, nv_bfloat16>::value, "Only half and bfloat16 input are supported");
   using PackedVec = PackedVec<T>;
 
   const int batch_id = blockIdx.y;
   const int head_id = blockIdx.z;
   const int token_block_id = blockIdx.x;
 
   static_assert(CVT_FP4_ELTS_PER_THREAD == 8 || CVT_FP4_ELTS_PER_THREAD == 16,
                 "CVT_FP4_ELTS_PER_THREAD must be 8 or 16");
   static_assert(sizeof(PackedVec) == sizeof(T) * CVT_FP4_ELTS_PER_THREAD,
                 "Vec size is not matched.");
 
   constexpr uint32_t NUM_THREADS_PER_TOKEN = head_dim / CVT_FP4_ELTS_PER_THREAD;
   constexpr uint32_t NUM_THREADS_PER_SEQ = BLOCK_SIZE / CVT_FP4_ELTS_PER_THREAD;
 
   // load input
   const int token_id = token_block_id * BLOCK_SIZE + threadIdx.x / NUM_THREADS_PER_TOKEN;

   PackedVec in_vec;

   #pragma unroll
   for (int i = 0; i < CVT_FP4_ELTS_PER_THREAD / 2; i++) {
     reinterpret_cast<uint32_t&>(in_vec.elts[i]) = 0;
   }

   if (token_id < num_tokens) {
     in_vec = reinterpret_cast<PackedVec const*>(input +
                                           batch_id * stride_bz_input + // batch dim
                                           head_id * stride_h_input +   // head dim
                                           token_id * stride_seq_input + // seq dim
                                           (threadIdx.x % NUM_THREADS_PER_TOKEN) * CVT_FP4_ELTS_PER_THREAD)[0]; // feature dim
   }
 
   // transpose
   __shared__ T shared_input[BLOCK_SIZE * head_dim];
   reinterpret_cast<PackedVec*>(shared_input)[threadIdx.x] = in_vec;
   __syncthreads();
   #pragma unroll
   for (int i = 0; i < CVT_FP4_ELTS_PER_THREAD / 2; i++) {
     in_vec.elts[i].x = shared_input[(threadIdx.x / NUM_THREADS_PER_SEQ) + ((threadIdx.x % NUM_THREADS_PER_SEQ) * CVT_FP4_ELTS_PER_THREAD + 2 * i) * head_dim];
     in_vec.elts[i].y = shared_input[(threadIdx.x / NUM_THREADS_PER_SEQ) + ((threadIdx.x % NUM_THREADS_PER_SEQ) * CVT_FP4_ELTS_PER_THREAD + 2 * i + 1) * head_dim];
   }
 
   // calculate max of every consecutive 16 elements
   auto localMax = __habs2(in_vec.elts[0]);
   #pragma unroll
   for (int i = 1; i < CVT_FP4_ELTS_PER_THREAD / 2; i++) { // local max
     localMax = __hmax2(localMax, __habs2(in_vec.elts[i]));
   }
 
   if constexpr (CVT_FP4_ELTS_PER_THREAD == 8) { // shuffle across two threads
     localMax = __hmax2(__shfl_xor_sync(0xffffffff, localMax, 1, 32), localMax);
   }
 
   float vecMax = float(__hmax(localMax.x, localMax.y));
 
   // scaling factor
   float SFValue = vecMax / 6.0f;
   uint8_t SFValueFP8;
   reinterpret_cast<__nv_fp8_e4m3&>(SFValueFP8) = __nv_fp8_e4m3(SFValue);
   SFValue = float(reinterpret_cast<__nv_fp8_e4m3&>(SFValueFP8));
 
   float SFValueInv = (SFValue == 0.0f) ? 0.0f : 1.0f / SFValue;
 
   // convert input to float2 and apply scale
   float2 fp2Vals[CVT_FP4_ELTS_PER_THREAD / 2];
 
   #pragma unroll
   for (int i = 0; i < CVT_FP4_ELTS_PER_THREAD / 2; i++) {
     if constexpr (std::is_same<T, half>::value) {
       fp2Vals[i] = __half22float2(in_vec.elts[i]);
     } else {
       fp2Vals[i] = __bfloat1622float2(in_vec.elts[i]);
     }
     fp2Vals[i].x = fp2Vals[i].x * SFValueInv;
     fp2Vals[i].y = fp2Vals[i].y * SFValueInv;
   }
 
   // convert to e2m1
   uint32_t e2m1Vals[CVT_FP4_ELTS_PER_THREAD / 8];
   #pragma unroll
   for (int i = 0; i < CVT_FP4_ELTS_PER_THREAD / 8; i++) {
     e2m1Vals[i] = fp32_vec_to_e2m1(fp2Vals + i * 4);
   }
 
   // Skip out-of-range tokens: never write past the sequence when num_tokens
   // is not a multiple of BLOCK_SIZE (out-of-bounds write fix).
   const int write_token_id = token_block_id * BLOCK_SIZE +
       (threadIdx.x % NUM_THREADS_PER_SEQ) * CVT_FP4_ELTS_PER_THREAD;
   if (write_token_id >= num_tokens) return;

   // save
   if constexpr (CVT_FP4_ELTS_PER_THREAD == 8) {
     reinterpret_cast<uint32_t*>(output + 
                                 batch_id * stride_bz_output +
                                 head_id * stride_h_output +
                                 (threadIdx.x / NUM_THREADS_PER_SEQ) * stride_d_output +
                                 (token_block_id * BLOCK_SIZE + (threadIdx.x % NUM_THREADS_PER_SEQ) * CVT_FP4_ELTS_PER_THREAD) / 2)[0] = e2m1Vals[0];
   } else {
     reinterpret_cast<uint64_t*>(output + 
                                 batch_id * stride_bz_output +
                                 head_id * stride_h_output +
                                 (threadIdx.x / NUM_THREADS_PER_SEQ) * stride_d_output +
                                 (token_block_id * BLOCK_SIZE + (threadIdx.x % NUM_THREADS_PER_SEQ) * CVT_FP4_ELTS_PER_THREAD) / 2)[0] = reinterpret_cast<uint64_t*>(e2m1Vals)[0];
   }
 
   uint8_t *output_sf_save_base = output_sf + 
                                 batch_id * stride_bz_output_sf +
                                 head_id * stride_h_output_sf +
                                 (threadIdx.x / NUM_THREADS_PER_SEQ / 64) * 64 * stride_d_output_sf;
   uint32_t row_id_local = (threadIdx.x / NUM_THREADS_PER_SEQ) % 64;
 
   if constexpr (CVT_FP4_ELTS_PER_THREAD == 16) {
     uint32_t col_id_local = token_block_id * BLOCK_SIZE / CVT_FP4_ELTS_PER_THREAD + threadIdx.x % NUM_THREADS_PER_SEQ;
     uint32_t offset_local = (col_id_local / 4) * 256 + (col_id_local % 4) + 
                             (row_id_local / 16) * 4 + (row_id_local % 16) * 16;
     reinterpret_cast<uint8_t*>(output_sf_save_base + offset_local)[0] = SFValueFP8;
   } else {
     if (threadIdx.x % 2 == 0) {
       uint32_t col_id_local = token_block_id * BLOCK_SIZE / CVT_FP4_ELTS_PER_THREAD + (threadIdx.x % NUM_THREADS_PER_SEQ) / 2;
       uint32_t offset_local = (col_id_local / 4) * 256 + (col_id_local % 4) + 
                               (row_id_local / 16) * 4 + (row_id_local % 16) * 16;
       reinterpret_cast<uint8_t*>(output_sf_save_base + offset_local)[0] = SFValueFP8;
     }
   }
 }
 

} // namespace ferrum_sage_vsa_quantization
