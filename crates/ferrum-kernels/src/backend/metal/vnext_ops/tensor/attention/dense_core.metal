/*
 * Standalone Ferrum dense-attention compute core.
 * Derived from Apple MLX Steel v0.32.2, commit
 * 1f8e74e3f12f31365464a6867c6579f0e9b29d85:
 * mlx/backend/metal/kernels/steel/attn/{loader.h,mma.h,params.h,
 * kernels/steel_attention.h}, and kernels/utils.h (finite_min).
 *
 * Preserved specialization: D128, BQ32, BK16, WM4/WN1; native bfloat IO;
 * float2 F32 8x8 fragments/MMA; ascending K blocks and original fragment
 * reduction order; scale*log2(e), fast::exp2, online rescale, final division;
 * padded shared Q and aliased transposed-K/V; final native BF16 conversion.
 * No intermediate BF16 P/O and no S^2 device allocation.
 * Removed: generic tile/dtype machinery, causal/additive/boolean masks,
 * attention sinks, GQA, and unused loader operations. The remaining loader
 * is BlockLoaderT specialized to one source row and 16/32 consecutive scalar
 * reads per thread; it adds no global-vector alignment assumption. Source
 * row addressing is widened from upstream int to long for the stride ABI.
 *
 * Caller contract, NOT checked by this compute core:
 * - explicit MSL3.1, fast_math=false; 32 lanes/SIMDgroup, exactly 128 threads;
 * - grid=(ceil(qL/32), H, B), threads=(32,4,1); B,H,qL,kL positive;
 * - D=128, gqa_factor=1; all keys visible and valid for all dispatched queries;
 * - function constants 200=(qL%32==0), 201=(kL%16==0);
 * - NQ/NK are ceil divisions, *_aligned floor divisions, *_rem remainders;
 * - positive element strides (B,H,L), unit D stride, checked extents/offsets;
 * - non-overlapping output and inputs; retained admitted device allocations;
 * - provider separately proves supported finite numerical domain or falls back.
 * Dense suffixes use offset Q/O pointers but preserve ORIGINAL batch strides.
 * qL_off is ABI padding/compatibility only; no causal interpretation is made.
 * Empty-key rows and nonfinite/overflow behavior require the fallback path.
 * Declared shared arrays total 14,848 bytes; the provider must query and admit
 * actual compiled pipeline resources before execution.
 *
 * Copyright © 2024-25 Apple Inc. (derived Steel implementation)
 * MIT License
 * Copyright © 2023 Apple Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>

using namespace metal;

#define FERRUM_DENSE_UNROLL _Pragma("clang loop unroll(full)")

constant bool ferrum_dense_align_q [[function_constant(200)]];
constant bool ferrum_dense_align_k [[function_constant(201)]];

// Identical field order and widths to the upstream 152-byte AttnParams ABI.
struct FerrumDenseAttentionParams {
  int B, H, D;
  int qL, kL;
  int gqa_factor;
  float scale;
  int NQ, NK;
  int NQ_aligned, NK_aligned;
  int qL_rem, kL_rem, qL_off;
  long Q_strides[3];
  long K_strides[3];
  long V_strides[3];
  long O_strides[3];
};
static_assert(sizeof(FerrumDenseAttentionParams) == 152, "attention ABI size");
static_assert(alignof(FerrumDenseAttentionParams) == 8, "attention ABI alignment");

namespace ferrum_dense {

// Fixed-configuration BlockLoaderT. Each thread owns one source row fragment.
// Q: 32 reads, four threads/row. K/V: 16 reads, eight threads/row.
template <short Rows, short DstRowStride, short DstColStride>
struct BlockLoader {
  static constant constexpr short Reads = Rows;
  static constant constexpr short ThreadsPerRow = 128 / Reads;
  const long src_row_stride;
  const short row;
  threadgroup bfloat* dst;
  const device bfloat* src;

  METAL_FUNC BlockLoader(const device bfloat* source, long source_row_stride,
              threadgroup bfloat* destination, ushort simd_id, ushort lane) thread
      : src_row_stride(source_row_stride),
        row((simd_id * 32 + lane) / ThreadsPerRow),
        dst(destination + row * DstRowStride +
            Reads * ((simd_id * 32 + lane) % ThreadsPerRow) * DstColStride),
        src(source + row * source_row_stride +
            Reads * ((simd_id * 32 + lane) % ThreadsPerRow)) {}

  METAL_FUNC void load_unsafe() const thread {
    FERRUM_DENSE_UNROLL
    for (short j = 0; j < Reads; ++j) {
      dst[j * DstColStride] = src[j];
    }
  }

  // D is exactly 128, so only the source row can be out of bounds.
  METAL_FUNC void load_safe(short valid_rows) const thread {
    if (row >= valid_rows) {
      FERRUM_DENSE_UNROLL
      for (short j = 0; j < Reads; ++j) {
        dst[j * DstColStride] = bfloat(0);
      }
      return;
    }
    bfloat values[Reads];
    FERRUM_DENSE_UNROLL
    for (short j = 0; j < Reads; ++j) {
      values[j] = src[j];
    }
    FERRUM_DENSE_UNROLL
    for (short j = 0; j < Reads; ++j) {
      dst[j * DstColStride] = values[j];
    }
  }

  METAL_FUNC void next() thread { src += Rows * src_row_stride; }
};

// Preserve Steel's native per-lane float2 fragment mapping and MMA call.
METAL_FUNC void mma(thread float2& d, thread float2& a, thread float2& b,
                thread float2& c) {
  simdgroup_matrix<float, 8, 8> dm, am, bm, cm;
  reinterpret_cast<thread float2&>(am.thread_elements()) = a;
  reinterpret_cast<thread float2&>(bm.thread_elements()) = b;
  reinterpret_cast<thread float2&>(cm.thread_elements()) = c;
  simdgroup_multiply_accumulate(dm, am, bm, cm);
  d = reinterpret_cast<thread float2&>(dm.thread_elements());
}

METAL_FUNC float row_max(float2 values, float previous) {
  const float thread_value = metal::max(values.x, values.y);
  const float quad_value = metal::max(
      thread_value, simd_shuffle_xor(thread_value, ushort(1)));
  const float row_value = metal::max(
      quad_value, simd_shuffle_xor(quad_value, ushort(8)));
  return metal::max(previous, row_value);
}

METAL_FUNC float row_sum(float2 values, float previous) {
  const float thread_value = values.x + values.y;
  const float quad_value = thread_value + simd_shuffle_xor(thread_value, ushort(1));
  const float row_value = quad_value + simd_shuffle_xor(quad_value, ushort(8));
  return previous + row_value;
}

METAL_FUNC float2 load_fragment(const threadgroup bfloat* source) {
  return float2(float(source[0]), float(source[1]));
}

} // namespace ferrum_dense

[[kernel, max_total_threads_per_threadgroup(128)]]
void ferrum_dense_attention_bf16_bq32_bk16_d128(
    const device bfloat* Q [[buffer(0)]],
    const device bfloat* K [[buffer(1)]],
    const device bfloat* V [[buffer(2)]],
    device bfloat* O [[buffer(3)]],
    const constant FerrumDenseAttentionParams* params [[buffer(4)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]]) {
  using namespace ferrum_dense;

  const ulong3 block{tid.x, tid.y, tid.z};
  Q += block.z * params->Q_strides[0] + block.y * params->Q_strides[1] +
      block.x * 32 * params->Q_strides[2];
  K += block.z * params->K_strides[0] + block.y * params->K_strides[1];
  V += block.z * params->V_strides[0] + block.y * params->V_strides[1];
  O += block.z * params->O_strides[0] + block.y * params->O_strides[1] +
      block.x * 32 * params->O_strides[2];

  threadgroup bfloat q_shared[32 * 136];
  threadgroup bfloat kv_shared[128 * 24];
  BlockLoader<32, 136, 1> loader_q(Q, params->Q_strides[2], q_shared, simd_id, lane);
  BlockLoader<16, 1, 24> loader_k(K, params->K_strides[2], kv_shared, simd_id, lane);
  BlockLoader<16, 136, 1> loader_v(V, params->V_strides[2], kv_shared, simd_id, lane);

  const float scale = params->scale * M_LOG2E_F;
  const short quad_id = lane / 4;
  const short sm = (quad_id & 4) + ((lane / 2) % 4);
  const short sn = (quad_id & 2) * 2 + (lane % 2) * 2;
  const short tm = 8 * simd_id;
  const short q_offset = (tm + sm) * 136 + sn;
  const short k_offset = sm * 24 + sn;
  const short v_offset = sm * 136 + sn;

  float2 output[16];
  FERRUM_DENSE_UNROLL
  for (short id = 0; id < 16; ++id) {
    output[id] = float2(0);
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (!ferrum_dense_align_q && int(tid.x) == params->NQ_aligned) {
    loader_q.load_safe(short(params->qL_rem));
  } else {
    loader_q.load_unsafe();
  }

  // Upstream Limits<float>::finite_min is -FLT_MAX, deliberately not -Inf.
  constexpr float finite_min = -metal::numeric_limits<float>::max();
  float maximum = finite_min;
  float denominator = 0;

  for (int kb = 0; kb < params->NK; ++kb) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const bool key_tail = !ferrum_dense_align_k && kb == params->NK_aligned;
    if (key_tail) {
      loader_k.load_safe(short(params->kL_rem));
    } else {
      loader_k.load_unsafe();
    }

    float2 probability[2] = {float2(0), float2(0)};
    threadgroup_barrier(mem_flags::mem_threadgroup);
    FERRUM_DENSE_UNROLL
    for (short dd = 0; dd < 16; ++dd) {
      simdgroup_barrier(mem_flags::mem_none);
      float2 query = load_fragment(q_shared + q_offset + dd * 8);
      float2 key0 = load_fragment(kv_shared + k_offset + dd * 8 * 24);
      float2 key1 = load_fragment(kv_shared + k_offset + dd * 8 * 24 + 8);
      simdgroup_barrier(mem_flags::mem_none);
      mma(probability[0], query, key0, probability[0]);
      mma(probability[1], query, key1, probability[1]);
    }

    FERRUM_DENSE_UNROLL
    for (short ik = 0; ik < 2; ++ik) {
      FERRUM_DENSE_UNROLL
      for (short j = 0; j < 2; ++j) {
        probability[ik][j] *= scale;
      }
    }
    if (key_tail) {
      FERRUM_DENSE_UNROLL
      for (short ik = 0; ik < 2; ++ik) {
        FERRUM_DENSE_UNROLL
        for (short j = 0; j < 2; ++j) {
          if (sn + ik * 8 + j >= params->kL_rem) {
            probability[ik][j] = finite_min;
          }
        }
      }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (key_tail) {
      loader_v.load_safe(short(params->kL_rem));
    } else {
      loader_v.load_unsafe();
    }

    float next_maximum = maximum;
    FERRUM_DENSE_UNROLL
    for (short ik = 0; ik < 2; ++ik) {
      next_maximum = row_max(probability[ik], next_maximum);
    }
    FERRUM_DENSE_UNROLL
    for (short ik = 0; ik < 2; ++ik) {
      FERRUM_DENSE_UNROLL
      for (short j = 0; j < 2; ++j) {
        probability[ik][j] = fast::exp2(probability[ik][j] - next_maximum);
      }
    }
    const float factor = fast::exp2(maximum - next_maximum);
    maximum = next_maximum;
    float block_sum = 0;
    FERRUM_DENSE_UNROLL
    for (short ik = 0; ik < 2; ++ik) {
      block_sum = row_sum(probability[ik], block_sum);
    }
    denominator = denominator * factor + block_sum;
    FERRUM_DENSE_UNROLL
    for (short id = 0; id < 16; ++id) {
      FERRUM_DENSE_UNROLL
      for (short j = 0; j < 2; ++j) {
        output[id][j] *= factor;
      }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    FERRUM_DENSE_UNROLL
    for (short id = 0; id < 16; ++id) {
      FERRUM_DENSE_UNROLL
      for (short ik = 0; ik < 2; ++ik) {
        simdgroup_barrier(mem_flags::mem_none);
        float2 value = load_fragment(kv_shared + v_offset + ik * 8 * 136 + id * 8);
        simdgroup_barrier(mem_flags::mem_none);
        mma(output[id], probability[ik], value, output[id]);
      }
    }
    loader_k.next();
    loader_v.next();
  }

  FERRUM_DENSE_UNROLL
  for (short id = 0; id < 16; ++id) {
    FERRUM_DENSE_UNROLL
    for (short j = 0; j < 2; ++j) {
      output[id][j] /= denominator;
    }
  }
  threadgroup_barrier(mem_flags::mem_none);

  if (!ferrum_dense_align_q && int(tid.x) == params->NQ_aligned &&
      tm + sm >= params->qL_rem) {
    return;
  }
  O += (tm + sm) * params->O_strides[2] + sn;
  FERRUM_DENSE_UNROLL
  for (short id = 0; id < 16; ++id) {
    FERRUM_DENSE_UNROLL
    for (short j = 0; j < 2; ++j) {
      O[id * 8 + j] = bfloat(output[id][j]);
    }
  }
}

#undef FERRUM_DENSE_UNROLL
