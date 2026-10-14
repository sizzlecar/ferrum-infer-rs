// Independent TEST-only I32/two-warp specialization of the pinned MMQ body.
// The normal TU, public Plan, activation pack and cast remain unchanged.
#include "boundary.h"
#include "common.cuh"
#include "vecdotq.cuh"
#include "mma.cuh"
#include <climits>
#include <cstdint>

// In the pinned mmq.cuh these two function-like names occur only in the config
// constructor's initializers. Keep the config type private to this TU, avoid
// changing dependency headers, and immediately remove all three macros.
// J8 has one 16-output-row MMA minitile per warp: I32 therefore needs 2 warps.
#define ggml_cuda_mmq_config FerrumTestMmqI32Config
#define FERRUM_TEST_I32_DOMAIN (J == 8 && I == 128 && \
    (type == GGML_TYPE_IQ4_XS || type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K))
#define nthreads(value) nthreads(FERRUM_TEST_I32_DOMAIN ? 64 : (value))
#define I(value) I(FERRUM_TEST_I32_DOMAIN ? 32 : (value))
#include "mmq.cuh"
#undef I
#undef nthreads
#undef FERRUM_TEST_I32_DOMAIN
#undef ggml_cuda_mmq_config

extern "C" int ferrum_test_upstream_mmq_validate_i32_full_tile_plan_v1(
        const FerrumUpstreamLinearPlanV1 *) noexcept;

template<ggml_type Type, bool Fallback>
static int ferrum_test_i32_full_tile_dispatch(const FerrumUpstreamLinearPlanV1 & p,
        const void * weights, const void * packed, float * output, cudaStream_t stream) {
    // Host and device selectors derive from the same private config constructor.
    const auto config = ggml_cuda_mmq_get_config(Type, 8, Fallback, p.request.cc);
    if (config.type != Type || config.I != 32 || config.J != 8 ||
            config.nthreads != 64 || config.K_vram != 256 || !config.stream_k) return -2;
    const size_t shared = mmq_get_nbytes_shared(config, p.request.cc);
    if (shared > p.shared_bytes || shared > p.request.shared_limit || shared > 48*1024) return -3;

    const auto one = init_fastdiv_values(1);
    const auto blocks_per_k = init_fastdiv_values(p.request.inputs/256);
    const auto ntx = init_fastdiv_values(1);
    const int weight_stride = p.request.inputs/256;
    const int packed_stride = p.request.rows*p.padded_inputs*sizeof(block_q8_1_mmq)/(128*sizeof(int));
    const uint32_t tiles = (p.request.outputs + 31)/32;
    // One complete K range per output tile. With blocks == tiles there is no
    // partial sum/fixup; the original stream-K reduction tree intentionally
    // changes, and the Rust diagnostic reports complete cross-arm differences.
    mul_mat_q<Type, 8, Fallback><<<dim3(tiles, 1, 1), dim3(32, 2, 1), shared, stream>>>(
        static_cast<const char *>(weights), static_cast<const int *>(packed), nullptr, nullptr,
        output, nullptr, nullptr, blocks_per_k, p.request.outputs, p.request.rows,
        weight_stride, p.request.rows, p.request.outputs,
        one, one, p.request.outputs*weight_stride, packed_stride, p.request.rows*p.request.outputs,
        one, one, p.request.outputs*weight_stride, packed_stride, p.request.rows*p.request.outputs,
        ntx);
    return cudaGetLastError();
}

template<ggml_type Type>
static int ferrum_test_i32_full_tile_format(const FerrumUpstreamLinearPlanV1 & p,
        const void * weights, const void * packed, float * output, cudaStream_t stream) {
    // The original Plan admits padded partial output tiles with bounded stores.
    return p.request.outputs%128
        ? ferrum_test_i32_full_tile_dispatch<Type, true>(p, weights, packed, output, stream)
        : ferrum_test_i32_full_tile_dispatch<Type, false>(p, weights, packed, output, stream);
}

extern "C" int ferrum_test_upstream_mmq_i32_full_tile_dot_v1(
        const FerrumUpstreamLinearPlanV1 * plan, const void * weights,
        const void * packed, void * output, void * fixup, void * stream) noexcept {
    return ferrum_upstream_boundary([&]() -> int {
        const int status = ferrum_test_upstream_mmq_validate_i32_full_tile_plan_v1(plan);
        if (status) return status;
        // Preserve the original extent authority, including its fixup pointer
        // admission even though this candidate neither reads nor writes it.
        if (!weights || !packed || !output || (plan->fixup && !fixup) ||
                uintptr_t(weights)%4 || uintptr_t(packed)%16 ||
                uintptr_t(output)%4 || uintptr_t(fixup)%4) return -5;
        const auto cuda_stream = static_cast<cudaStream_t>(stream);
        switch (plan->request.format) {
            case GGML_TYPE_IQ4_XS:
                return ferrum_test_i32_full_tile_format<GGML_TYPE_IQ4_XS>(
                    *plan, weights, packed, static_cast<float *>(output), cuda_stream);
            case GGML_TYPE_Q4_K:
                return ferrum_test_i32_full_tile_format<GGML_TYPE_Q4_K>(
                    *plan, weights, packed, static_cast<float *>(output), cuda_stream);
            case GGML_TYPE_Q5_K:
                return ferrum_test_i32_full_tile_format<GGML_TYPE_Q5_K>(
                    *plan, weights, packed, static_cast<float *>(output), cuda_stream);
            default: return -4;
        }
    });
}
