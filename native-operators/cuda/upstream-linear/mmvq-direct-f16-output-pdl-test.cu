// TEST-only replacement TU. Includes the prior direct-input experiment so
// one archive can compare original, output-only, and both-boundary variants.
// Do not link this object together with the normal mmvq object.
#define FERRUM_TEST_MMVQ_F16_OUTPUT
#include "mmvq-direct-f16-pdl-test.cu"

static bool test_mmvq_f16_output_valid(const FerrumUpstreamLinearPlanV1 * p,
        TestMmvqPlan & n) {
    return upstream_mmvq_pdl_valid(p,n) && n.rows == 1 && n.ncols == 1 &&
        n.channels == 1 && n.layout == 0;
}

template <ggml_type T, bool Small>
static const void * test_mmvq_f16_output_kernel() {
    return reinterpret_cast<const void *>(mul_mat_vec_q<T,1,false,Small,false,false,true>);
}

template <ggml_type T>
static const void * test_mmvq_f16_output_kernel(const TestMmvqPlan & n) {
    return n.small_k ? test_mmvq_f16_output_kernel<T,true>()
                     : test_mmvq_f16_output_kernel<T,false>();
}

static const void * test_mmvq_f16_output_kernel(const TestMmvqPlan & n) {
    switch (n.format) {
        case GGML_TYPE_Q4_K: return test_mmvq_f16_output_kernel<GGML_TYPE_Q4_K>(n);
        case GGML_TYPE_Q5_K: return test_mmvq_f16_output_kernel<GGML_TYPE_Q5_K>(n);
        case GGML_TYPE_IQ4_XS: return test_mmvq_f16_output_kernel<GGML_TYPE_IQ4_XS>(n);
        default: return nullptr;
    }
}

extern "C" int ferrum_test_mmvq_f16_output_pdl_support_v1(
        const FerrumUpstreamLinearPlanV1 * p, uint32_t * supported) noexcept {
    return ferrum_upstream_boundary([&]() -> int {
        if (!supported) return -1;
        *supported = 0;
        TestMmvqPlan n{};
        if (!test_mmvq_f16_output_valid(p,n)) return -1;
#if defined(GGML_CUDA_USE_PDL) && CUDART_VERSION >= 12030
        auto status = ferrum_upstream_mmvq_pdl_support_v1(p,0,supported);
        if (status != cudaSuccess || !*supported) return status;
        *supported = 0;
        const void * kernel = test_mmvq_f16_output_kernel(n);
        if (!kernel) return -2;
        cudaFuncAttributes attributes{};
        status = cudaFuncGetAttributes(&attributes,kernel);
        if (status != cudaSuccess) return status;
        *supported = attributes.ptxVersion >= 90;
#endif
        return 0;
    });
}

#if defined(GGML_CUDA_USE_PDL)
template <ggml_type T, bool Small>
static int test_mmvq_f16_output_launch(const TestMmvqPlan & n,
        const void * weights, const void * packed, TestMmvqF16Output output,
        cudaStream_t stream) {
    const dim3 grid(n.padded_outputs/n.rows_per_block,n.channels,1), block(32,n.nwarps,1);
    const auto one = init_fastdiv_values(1);
    const uint32_t qstride = n.padded_inputs/32;
    ggml_cuda_mm_fusion_args_device fusion{};
    // Identical geometry, dot arithmetic and reduction to the original M1
    // specialization. N is the logical store bound, not output.stride.
    // No early completion trigger: successor clears wait for the final store.
    return upstream_mmvq_launch_pdl(mul_mat_vec_q<T,1,false,Small,false,false,true>,
        grid,block,stream,weights,packed,static_cast<const int32_t *>(nullptr),
        fusion,output,n.inputs,make_uint3(0,0,0),n.inputs/256,qstride,n.outputs,
        init_fastdiv_values(n.channels),n.padded_outputs*(n.inputs/256),
        n.rows*qstride,n.rows*n.outputs,one,n.padded_outputs*(n.inputs/256),
        n.rows*qstride,n.rows*n.outputs,uint32_t(0));
}

template <ggml_type T>
static int test_mmvq_f16_output_launch(const TestMmvqPlan & n,
        const void * weights, const void * packed, TestMmvqF16Output output,
        cudaStream_t stream) {
    return n.small_k ? test_mmvq_f16_output_launch<T,true>(n,weights,packed,output,stream)
                     : test_mmvq_f16_output_launch<T,false>(n,weights,packed,output,stream);
}
#endif

extern "C" int ferrum_test_mmvq_f16_output_pdl_dot_v1(
        const FerrumUpstreamLinearPlanV1 * p, const void * weights,
        const void * packed, void * output, uint32_t output_stride,
        const void * row_flags, const void * weight_flag, void * s) noexcept {
    return ferrum_upstream_boundary([&]() -> int {
        TestMmvqPlan n{};
        if (!test_mmvq_f16_output_valid(p,n)) return -1;
        if (!weights || !packed || !output || !row_flags || !weight_flag ||
                output_stride < n.outputs || uint64_t(output_stride)*n.rows > INT_MAX ||
                uintptr_t(weights)%4 || uintptr_t(packed)%4 || uintptr_t(output)%2 ||
                uintptr_t(row_flags)%4 || uintptr_t(weight_flag)%4) return -5;
#if defined(GGML_CUDA_USE_PDL)
        if (n.cc < GGML_CUDA_CC_HOPPER) return -2;
        const TestMmvqF16Output destination{static_cast<half *>(output),
            static_cast<const uint32_t *>(row_flags),
            static_cast<const uint32_t *>(weight_flag),output_stride};
        const auto stream = static_cast<cudaStream_t>(s);
        switch (n.format) {
            case GGML_TYPE_Q4_K: return test_mmvq_f16_output_launch<GGML_TYPE_Q4_K>(n,weights,packed,destination,stream);
            case GGML_TYPE_Q5_K: return test_mmvq_f16_output_launch<GGML_TYPE_Q5_K>(n,weights,packed,destination,stream);
            case GGML_TYPE_IQ4_XS: return test_mmvq_f16_output_launch<GGML_TYPE_IQ4_XS>(n,weights,packed,destination,stream);
            default: return -1;
        }
#else
        return -2;
#endif
    });
}

// Counts: nodes,kernels,other,clear,widen,F32pack,F16pack,normal_dot,
// fused_dot,cast,edges,programmatic_edges,cast->next_clear,
// fused_dot->next_clear,F32pack->fused_dot,F16pack->fused_dot.
extern "C" int ferrum_test_mmvq_f16_output_pdl_graph_v1(
        void * graph_ptr, uint64_t * counts, size_t count) noexcept {
    if (!graph_ptr || !counts || count < 16) return -1;
    for (size_t i=0;i<16;++i) counts[i]=0;
#if defined(GGML_CUDA_USE_PDL) && CUDART_VERSION >= 12030
    return ferrum_upstream_boundary([&]() -> int {
        const auto graph=static_cast<cudaGraph_t>(graph_ptr);
        size_t nn=0,ne=0;
        auto status=cudaGraphGetNodes(graph,nullptr,&nn); if(status) return status;
        status=cudaGraphGetEdges_v2(graph,nullptr,nullptr,nullptr,&ne); if(status) return status;
        std::vector<cudaGraphNode_t> nodes(nn),from(ne),to(ne);
        std::vector<cudaGraphEdgeData> edges(ne);
        status=cudaGraphGetNodes(graph,nodes.data(),&nn); if(status) return status;
        status=cudaGraphGetEdges_v2(graph,from.data(),to.data(),edges.data(),&ne); if(status) return status;
        counts[0]=nn; counts[10]=ne;
        const void * clear=reinterpret_cast<const void *>(upstream_mmvq_pdl_clear);
        const void * widen=reinterpret_cast<const void *>(upstream_mmvq_pdl_widen);
        const void * f32pack=reinterpret_cast<const void *>(quantize_q8_1<true>);
        const void * f16pack=reinterpret_cast<const void *>(quantize_q8_1_f16_direct);
        const void * cast=reinterpret_cast<const void *>(upstream_mmvq_pdl_cast_kernel);
        const auto normal=[](const void * f) {
            return f==upstream_mmvq_pdl_dot_kernel<GGML_TYPE_Q4_K,1,false,false>() ||
                f==upstream_mmvq_pdl_dot_kernel<GGML_TYPE_Q5_K,1,false,false>() ||
                f==upstream_mmvq_pdl_dot_kernel<GGML_TYPE_IQ4_XS,1,false,false>() ||
                f==upstream_mmvq_pdl_dot_kernel<GGML_TYPE_Q4_K,1,true,false>() ||
                f==upstream_mmvq_pdl_dot_kernel<GGML_TYPE_Q5_K,1,true,false>() ||
                f==upstream_mmvq_pdl_dot_kernel<GGML_TYPE_IQ4_XS,1,true,false>();
        };
        const auto fused=[](const void * f) {
            return f==test_mmvq_f16_output_kernel<GGML_TYPE_Q4_K,false>() ||
                f==test_mmvq_f16_output_kernel<GGML_TYPE_Q5_K,false>() ||
                f==test_mmvq_f16_output_kernel<GGML_TYPE_IQ4_XS,false>() ||
                f==test_mmvq_f16_output_kernel<GGML_TYPE_Q4_K,true>() ||
                f==test_mmvq_f16_output_kernel<GGML_TYPE_Q5_K,true>() ||
                f==test_mmvq_f16_output_kernel<GGML_TYPE_IQ4_XS,true>();
        };
        for(auto node:nodes) {
            cudaGraphNodeType type{};
            status=cudaGraphNodeGetType(node,&type); if(status) return status;
            if(type!=cudaGraphNodeTypeKernel) {++counts[2];continue;}
            ++counts[1]; cudaKernelNodeParams p{};
            status=cudaGraphKernelNodeGetParams(node,&p); if(status) return status;
            if(p.func==clear) ++counts[3];
            if(p.func==widen) ++counts[4];
            if(p.func==f32pack) ++counts[5];
            if(p.func==f16pack) ++counts[6];
            if(normal(p.func)) ++counts[7];
            if(fused(p.func)) ++counts[8];
            if(p.func==cast) ++counts[9];
        }
        for(size_t i=0;i<ne;++i) {
            if(edges[i].type!=cudaGraphDependencyTypeProgrammatic) continue;
            ++counts[11];
            if(edges[i].from_port!=cudaGraphKernelNodePortProgrammatic || edges[i].to_port!=0) continue;
            cudaGraphNodeType a_type{},b_type{};
            status=cudaGraphNodeGetType(from[i],&a_type); if(status) return status;
            status=cudaGraphNodeGetType(to[i],&b_type); if(status) return status;
            if(a_type!=cudaGraphNodeTypeKernel || b_type!=cudaGraphNodeTypeKernel) continue;
            cudaKernelNodeParams a{},b{};
            status=cudaGraphKernelNodeGetParams(from[i],&a); if(status) return status;
            status=cudaGraphKernelNodeGetParams(to[i],&b); if(status) return status;
            if(a.func==cast && b.func==clear) ++counts[12];
            if(fused(a.func) && b.func==clear) ++counts[13];
            if(a.func==f32pack && fused(b.func)) ++counts[14];
            if(a.func==f16pack && fused(b.func)) ++counts[15];
        }
        return 0;
    });
#else
    return -2;
#endif
}
