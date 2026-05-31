// C ABI bridge from Ferrum to a source-built FlashAttention-2 forward kernel.
//
// Unlike fa2_ferrum_shim.cpp, this file does not call into vLLM's packaged
// Torch extension. The build script compiles the needed FlashAttention source
// templates into this shared object directly, then Ferrum can use the same
// ferrum_fa2_paged_varlen_fwd ABI for an apples-to-apples runtime check.

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <exception>

#include <cutlass/numeric_types.h>

#include "flash.h"

namespace flash {
template <typename T, int Headdim, bool Is_causal>
void run_mha_fwd_splitkv_dispatch(Flash_fwd_params &params, cudaStream_t stream);
}  // namespace flash

namespace {

int round_multiple(int x, int multiple) {
    return ((x + multiple - 1) / multiple) * multiple;
}

void set_err(char *err_buf, size_t err_buf_len, const char *fmt, ...) {
    if (err_buf == nullptr || err_buf_len == 0) {
        return;
    }
    va_list args;
    va_start(args, fmt);
    std::vsnprintf(err_buf, err_buf_len, fmt, args);
    va_end(args);
    err_buf[err_buf_len - 1] = '\0';
}

void run_hdim128_fp16_splitkv(flash::Flash_fwd_params &params, cudaStream_t stream) {
    if (params.is_causal) {
        flash::run_mha_fwd_splitkv_dispatch<cutlass::half_t, 128, true>(params, stream);
    } else {
        flash::run_mha_fwd_splitkv_dispatch<cutlass::half_t, 128, false>(params, stream);
    }
}

}  // namespace

extern "C" __attribute__((visibility("default"))) int ferrum_fa2_paged_varlen_fwd(
    const void *q,
    const void *k,
    const void *v,
    void *out,
    void *lse,
    const void *cu_seqlens_q,
    const void *seq_lens,
    const void *block_tables,
    int num_seqs,
    int total_q_tokens,
    int max_q_len,
    int max_kv_len,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int block_size,
    int max_blocks_per_seq,
    void *stream,
    char *err_buf,
    size_t err_buf_len) {
    try {
        if (q == nullptr || k == nullptr || v == nullptr || out == nullptr || lse == nullptr ||
            cu_seqlens_q == nullptr || seq_lens == nullptr || block_tables == nullptr) {
            set_err(err_buf, err_buf_len, "null pointer argument");
            return 1;
        }
        if (num_seqs <= 0 || total_q_tokens <= 0 || max_q_len <= 0 || max_kv_len <= 0) {
            set_err(err_buf, err_buf_len,
                    "invalid sizes: num_seqs=%d total_q=%d max_q=%d max_k=%d",
                    num_seqs, total_q_tokens, max_q_len, max_kv_len);
            return 2;
        }
        if (head_dim != 128 || block_size != 16 || num_kv_heads <= 0 ||
            num_heads % num_kv_heads != 0) {
            set_err(err_buf, err_buf_len,
                    "unsupported shape: heads=%d kv_heads=%d head_dim=%d block_size=%d",
                    num_heads, num_kv_heads, head_dim, block_size);
            return 3;
        }

        flash::Flash_fwd_params params{};
        params.q_ptr = const_cast<void *>(q);
        params.k_ptr = const_cast<void *>(k);
        params.v_ptr = const_cast<void *>(v);
        params.o_ptr = out;
        params.softmax_lse_ptr = lse;

        params.q_row_stride = num_heads * head_dim;
        params.k_row_stride = num_kv_heads * head_dim;
        params.v_row_stride = num_kv_heads * head_dim;
        params.o_row_stride = num_heads * head_dim;

        params.q_head_stride = head_dim;
        params.k_head_stride = head_dim;
        params.v_head_stride = head_dim;
        params.o_head_stride = head_dim;

        params.k_batch_stride = static_cast<int64_t>(block_size) * num_kv_heads * head_dim;
        params.v_batch_stride = static_cast<int64_t>(block_size) * num_kv_heads * head_dim;

        params.h = num_heads;
        params.h_k = num_kv_heads;
        params.h_h_k_ratio = num_heads / num_kv_heads;
        params.b = num_seqs;
        params.seqlen_q = max_q_len;
        params.seqlen_k = max_kv_len;
        params.seqlen_q_rounded = round_multiple(max_q_len, 128);
        params.seqlen_k_rounded = round_multiple(max_kv_len, 128);
        params.d = head_dim;
        params.d_rounded = head_dim;
        params.total_q = total_q_tokens;

        params.scale_softmax = 1.0f / std::sqrt(static_cast<float>(head_dim));
        params.scale_softmax_log2 = params.scale_softmax * static_cast<float>(M_LOG2E);
        params.p_dropout = 1.0f;
        params.p_dropout_in_uint8_t = 255;
        params.rp_dropout = 1.0f;
        params.scale_softmax_rp_dropout = params.scale_softmax;
        params.softcap = 0.0f;

        params.cu_seqlens_q = static_cast<int *>(const_cast<void *>(cu_seqlens_q));
        params.cu_seqlens_k = nullptr;
        params.seqused_k = static_cast<int *>(const_cast<void *>(seq_lens));
        params.block_table = static_cast<int *>(const_cast<void *>(block_tables));
        params.block_table_batch_stride = max_blocks_per_seq;
        params.page_block_size = block_size;

        params.is_bf16 = false;
        params.is_causal = max_q_len > 1;
        params.is_seqlens_k_cumulative = true;
        params.unpadded_lse = true;
        params.seqlenq_ngroups_swapped = false;
        params.window_size_left = -1;
        params.window_size_right = params.is_causal ? 0 : -1;
        params.num_splits = 0;

        run_hdim128_fp16_splitkv(params, reinterpret_cast<cudaStream_t>(stream));
        const cudaError_t launch_err = cudaPeekAtLastError();
        if (launch_err != cudaSuccess) {
            set_err(err_buf, err_buf_len, "FA2 source launch failed: %s",
                    cudaGetErrorString(launch_err));
            return 4;
        }
        if (err_buf != nullptr && err_buf_len > 0) {
            err_buf[0] = '\0';
        }
        return 0;
    } catch (const std::exception &e) {
        set_err(err_buf, err_buf_len, "exception: %s", e.what());
        return 10;
    } catch (...) {
        set_err(err_buf, err_buf_len, "unknown exception");
        return 11;
    }
}
