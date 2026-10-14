// Independent TEST TU. The normal MMVQ implementation and its public exports
// are retained; only these explicit entrypoints opt quantize -> dot into PDL.
#include "mmvq.cu"
#include <vector>

#if defined(GGML_CUDA_USE_PDL)
template<typename Kernel, typename... Args>
static int test_mmvq_launch_pdl(Kernel kernel, dim3 grid, dim3 block,
        cudaStream_t stream, Args... args) {
    cudaLaunchAttribute attr{};
    attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr.val.programmaticStreamSerializationAllowed = 1;
    cudaLaunchConfig_t config{};
    config.gridDim = grid;
    config.blockDim = block;
    config.stream = stream;
    config.attrs = &attr;
    config.numAttrs = 1;
    return cudaLaunchKernelEx(&config, kernel, args...);
}

template<ggml_type T, int C, bool Small, bool Shared>
static const void * test_mmvq_pdl_kernel() {
    return reinterpret_cast<const void *>(mul_mat_vec_q<T,C,false,Small,false,Shared>);
}

template<ggml_type T>
static const void * test_mmvq_pdl_kernel(const TestMmvqPlan & n) {
    if (n.ncols == 1) return n.small_k
        ? test_mmvq_pdl_kernel<T,1,true,false>()
        : test_mmvq_pdl_kernel<T,1,false,false>();
    if constexpr (T == GGML_TYPE_Q4_K || T == GGML_TYPE_Q5_K) {
        return test_mmvq_pdl_kernel<T,8,false,true>();
    }
    return test_mmvq_pdl_kernel<T,8,false,false>();
}

static const void * test_mmvq_pdl_kernel(const TestMmvqPlan & n) {
    switch (n.format) {
        case GGML_TYPE_Q4_K: return test_mmvq_pdl_kernel<GGML_TYPE_Q4_K>(n);
        case GGML_TYPE_Q5_K: return test_mmvq_pdl_kernel<GGML_TYPE_Q5_K>(n);
        case GGML_TYPE_IQ4_XS: return test_mmvq_pdl_kernel<GGML_TYPE_IQ4_XS>(n);
        default: return nullptr;
    }
}

template<ggml_type T, int C, bool Small, bool Shared>
static int test_mmvq_pdl_dot(const TestMmvqPlan & n, const void * weights,
        const void * packed, float * output, cudaStream_t stream) {
    const dim3 grid(n.padded_outputs/n.rows_per_block,n.channels,1), block(32,n.nwarps,1);
    const auto one = init_fastdiv_values(1);
    const uint32_t qstride = n.padded_inputs/32;
    ggml_cuda_mm_fusion_args_device fusion{};
    // This exact specialization already synchronizes grid dependencies before
    // its first input/packed/weight read. No arithmetic or reduction is changed.
    return test_mmvq_launch_pdl(mul_mat_vec_q<T,C,false,Small,false,Shared>,grid,block,stream,
        weights,packed,static_cast<const int32_t *>(nullptr),fusion,output,n.inputs,
        make_uint3(0,0,0),n.inputs/256,qstride,n.outputs,init_fastdiv_values(n.channels),
        n.padded_outputs*(n.inputs/256),n.layout?qstride:n.rows*qstride,
        n.layout?n.outputs:n.rows*n.outputs,one,
        n.padded_outputs*(n.inputs/256),n.rows*qstride,n.rows*n.outputs,uint32_t(0));
}

template<ggml_type T>
static int test_mmvq_pdl_dispatch(const TestMmvqPlan & n, const void * weights,
        const void * packed, float * output, cudaStream_t stream) {
    if (n.ncols == 1) return n.small_k
        ? test_mmvq_pdl_dot<T,1,true,false>(n,weights,packed,output,stream)
        : test_mmvq_pdl_dot<T,1,false,false>(n,weights,packed,output,stream);
    if constexpr (T == GGML_TYPE_Q4_K || T == GGML_TYPE_Q5_K) {
        return test_mmvq_pdl_dot<T,8,false,true>(n,weights,packed,output,stream);
    }
    return test_mmvq_pdl_dot<T,8,false,false>(n,weights,packed,output,stream);
}
#endif

static bool test_mmvq_pdl_valid(const FerrumUpstreamLinearPlanV1 * p, TestMmvqPlan & n) {
    return upstream_mmvq_valid(p,n) && n.layout == 0 && n.channels == 1 &&
        (n.rows == 1 || n.rows == 8);
}

// Cold setup only. Rust calls this before capture and chooses the ordinary
// pack/dot functions when either the actual CC or the loaded PTX cannot use PDL.
// There is no environment toggle, attribute query, mutex or cache in hot calls.
extern "C" int ferrum_test_upstream_mmvq_pdl_support_v1(
        const FerrumUpstreamLinearPlanV1 * p, uint32_t * supported) noexcept {
    return ferrum_upstream_boundary([&]() -> int {
        TestMmvqPlan n{};
        if (!supported || !test_mmvq_pdl_valid(p,n)) return -1;
        *supported = 0;
#if defined(GGML_CUDA_USE_PDL) && CUDART_VERSION >= 12030
        if (n.cc < GGML_CUDA_CC_HOPPER) return 0;
        cudaFuncAttributes pack{}, dot{};
        auto status = cudaFuncGetAttributes(&pack,quantize_q8_1<true>);
        if (status != cudaSuccess) return status;
        status = cudaFuncGetAttributes(&dot,test_mmvq_pdl_kernel(n));
        if (status != cudaSuccess) return status;
        *supported = pack.ptxVersion >= 90 && dot.ptxVersion >= 90;
#endif
        return 0;
    });
}

extern "C" int ferrum_test_upstream_mmvq_pdl_pack_v2(const FerrumUpstreamLinearPlanV1 * p,
        const void * input, uint32_t stride, void * converted, void * packed,
        void * flags, void * s) noexcept {
    return ferrum_upstream_boundary([&]() -> int {
        TestMmvqPlan n{};
        if (!test_mmvq_pdl_valid(p,n)) return -1;
        if (!input || !converted || !packed || !flags || stride<n.inputs ||
                uint64_t(stride)*n.rows>INT_MAX || uintptr_t(input)%2 ||
                uintptr_t(converted)%16 || uintptr_t(packed)%4 || uintptr_t(flags)%4) return -5;
#if defined(GGML_CUDA_USE_PDL)
        if (n.cc < GGML_CUDA_CC_HOPPER) return -2;
        const auto stream = static_cast<cudaStream_t>(s);
        // Preserve ordinary barriers for flag clear and F16 -> F32 conversion.
        mmvq_zero_row_flags<<<(uint64_t(n.rows)+127)/128,128,0,stream>>>(
            static_cast<uint32_t *>(flags),n.rows);
        auto status = cudaGetLastError(); if (status != cudaSuccess) return status;
        test_mmvq_convert<<<(n.rows*n.padded_inputs+255)/256,256,0,stream>>>(
            static_cast<const half *>(input),static_cast<float *>(converted),
            n.rows,n.inputs,stride,n.padded_inputs);
        status = cudaGetLastError(); if (status != cudaSuccess) return status;
        const dim3 grid((n.padded_inputs+CUDA_QUANTIZE_BLOCK_SIZE-1)/CUDA_QUANTIZE_BLOCK_SIZE,n.rows,1);
        // The original quantizer calls lc before its prologue and sync before
        // reading converted input. The successor dot also has a read-side sync.
        return test_mmvq_launch_pdl(quantize_q8_1<true>,grid,
            dim3(CUDA_QUANTIZE_BLOCK_SIZE,1,1),stream,
            static_cast<const float *>(converted),packed,int64_t(n.inputs),int64_t(n.padded_inputs),
            int64_t(n.rows)*n.padded_inputs,int64_t(n.rows)*n.padded_inputs,
            int64_t(n.padded_inputs),n.rows,init_fastdiv_values(1),static_cast<uint32_t *>(flags));
#else
        return -2;
#endif
    });
}

extern "C" int ferrum_test_upstream_mmvq_pdl_dot_v1(const FerrumUpstreamLinearPlanV1 * p,
        const void * weights, const void * packed, void * output, void * fixup, void * s) noexcept {
    return ferrum_upstream_boundary([&]() -> int {
        TestMmvqPlan n{};
        if (!test_mmvq_pdl_valid(p,n) || fixup) return -1;
        if (!weights || !packed || !output || uintptr_t(weights)%4 ||
                uintptr_t(packed)%4 || uintptr_t(output)%4) return -5;
#if defined(GGML_CUDA_USE_PDL)
        if (n.cc < GGML_CUDA_CC_HOPPER) return -2;
        auto stream = static_cast<cudaStream_t>(s);
        auto result = static_cast<float *>(output);
        switch (n.format) {
            case GGML_TYPE_Q4_K: return test_mmvq_pdl_dispatch<GGML_TYPE_Q4_K>(n,weights,packed,result,stream);
            case GGML_TYPE_Q5_K: return test_mmvq_pdl_dispatch<GGML_TYPE_Q5_K>(n,weights,packed,result,stream);
            case GGML_TYPE_IQ4_XS: return test_mmvq_pdl_dispatch<GGML_TYPE_IQ4_XS>(n,weights,packed,result,stream);
            default: return -1;
        }
#else
        return -2;
#endif
    });
}

// Inspect real capture metadata under the CUDA header ABI instead of mirroring
// cudaGraphEdgeData in Rust. Count the exact quantizer -> selected-dot edges.
extern "C" int ferrum_test_upstream_mmvq_pdl_graph_edges_v1(
        const FerrumUpstreamLinearPlanV1 * p, void * graph_ptr,
        uint64_t * total_edges, uint64_t * programmatic_edges, uint64_t * quant_dot_edges) noexcept {
    return ferrum_upstream_boundary([&]() -> int {
        TestMmvqPlan n{};
        if (!test_mmvq_pdl_valid(p,n) || !graph_ptr || !total_edges ||
                !programmatic_edges || !quant_dot_edges) return -1;
        *total_edges = *programmatic_edges = *quant_dot_edges = 0;
#if defined(GGML_CUDA_USE_PDL) && CUDART_VERSION >= 12030
        const auto graph = static_cast<cudaGraph_t>(graph_ptr);
        size_t count = 0;
        auto status = cudaGraphGetEdges_v2(graph,nullptr,nullptr,nullptr,&count);
        if (status != cudaSuccess) return status;
        std::vector<cudaGraphNode_t> from(count), to(count);
        std::vector<cudaGraphEdgeData> data(count);
        status = cudaGraphGetEdges_v2(graph,from.data(),to.data(),data.data(),&count);
        if (status != cudaSuccess) return status;
        *total_edges = count;
        for (size_t i = 0; i < count; ++i) {
            if (data[i].type != cudaGraphDependencyTypeProgrammatic) continue;
            ++*programmatic_edges;
            cudaGraphNodeType from_type{}, to_type{};
            status = cudaGraphNodeGetType(from[i],&from_type); if (status != cudaSuccess) return status;
            status = cudaGraphNodeGetType(to[i],&to_type); if (status != cudaSuccess) return status;
            if (from_type != cudaGraphNodeTypeKernel || to_type != cudaGraphNodeTypeKernel) continue;
            cudaKernelNodeParams a{}, b{};
            status = cudaGraphKernelNodeGetParams(from[i],&a); if (status != cudaSuccess) return status;
            status = cudaGraphKernelNodeGetParams(to[i],&b); if (status != cudaSuccess) return status;
            if (a.func == reinterpret_cast<void *>(quantize_q8_1<true>) &&
                    b.func == test_mmvq_pdl_kernel(n) &&
                    data[i].from_port == cudaGraphKernelNodePortProgrammatic && data[i].to_port == 0) {
                ++*quant_dot_edges;
            }
        }
        return 0;
#else
        return -2;
#endif
    });
}
