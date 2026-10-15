// TEST-only replacement TU. Normal source/ABI/math remain unchanged.
#define FERRUM_TEST_DIRECT_F16_PACK
#include "mmvq.cu"
#include <vector>

extern "C" int ferrum_test_mmvq_direct_pdl_support_v1(
        const FerrumUpstreamLinearPlanV1 * p, uint32_t * supported) noexcept {
    if (!supported) return -1;
    *supported = 0;
    TestMmvqPlan n{};
    if (!upstream_mmvq_pdl_valid(p,n) || n.rows != 1) return -1;
#if defined(GGML_CUDA_USE_PDL) && CUDART_VERSION >= 12030
    auto status = ferrum_upstream_mmvq_pdl_support_v1(p,0,supported);
    if (status || !*supported) return status;
    *supported = 0;
    cudaFuncAttributes attrs{};
    status = cudaFuncGetAttributes(&attrs,quantize_q8_1_f16_direct);
    if (status != cudaSuccess) return status;
    *supported = attrs.ptxVersion >= 90;
#endif
    return 0;
}

extern "C" int ferrum_test_mmvq_direct_pdl_pack_v1(
        const FerrumUpstreamLinearPlanV1 * p, const void * input, uint32_t stride,
        void * converted, void * packed, void * flags, void * s) noexcept {
    return ferrum_upstream_boundary([&]() -> int {
        TestMmvqPlan n{};
        if (!upstream_mmvq_pdl_valid(p,n) || n.rows != 1) return -1;
        if (!input || !converted || !packed || !flags || stride<n.inputs ||
                uint64_t(stride)*n.rows>INT_MAX || uintptr_t(input)%2 ||
                uintptr_t(converted)%16 || uintptr_t(packed)%4 || uintptr_t(flags)%4) return -5;
#if defined(GGML_CUDA_USE_PDL)
        if (n.cc < GGML_CUDA_CC_HOPPER) return -2;
        const auto stream = static_cast<cudaStream_t>(s);
        auto status = upstream_mmvq_launch_pdl(upstream_mmvq_pdl_clear,
            dim3(1),dim3(128),stream,static_cast<uint32_t *>(flags),n.rows);
        if (status != cudaSuccess) return status;
        const dim3 grid((n.padded_inputs+CUDA_QUANTIZE_BLOCK_SIZE-1)/CUDA_QUANTIZE_BLOCK_SIZE,n.rows,1);
        // Same half loader, reductions, Q8 bytes/scales and MarkerV2 body as
        // the existing direct-F16 experiment. LC/sync are inside that body.
        // Converted is deliberately untouched; clear remains a separate kernel.
        return upstream_mmvq_launch_pdl(quantize_q8_1_f16_direct,grid,
            dim3(CUDA_QUANTIZE_BLOCK_SIZE),stream,static_cast<const half *>(input),
            packed,int64_t(n.inputs),int64_t(stride),int64_t(n.padded_inputs),
            n.rows,init_fastdiv_values(1),static_cast<uint32_t *>(flags));
#else
        return -2;
#endif
    });
}

// Actual node identities and typed PDL edges, including cast->next clear.
// counts: nodes,kernels,other,clear,widen,f32pack,f16pack,cast,edges,
// programmatic-edges,cast->clear,clear->direct, direct->dot.
extern "C" int ferrum_test_mmvq_direct_pdl_graph_v1(
        void * graph_ptr, uint64_t * counts, size_t count) noexcept {
    if (!graph_ptr || !counts || count < 13) return -1;
    for (size_t i=0;i<13;++i) counts[i]=0;
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
        counts[0]=nn; counts[8]=ne;
        const void * clear=reinterpret_cast<const void *>(upstream_mmvq_pdl_clear);
        const void * direct=reinterpret_cast<const void *>(quantize_q8_1_f16_direct);
        const void * cast=reinterpret_cast<const void *>(upstream_mmvq_pdl_cast_kernel);
        const auto is_dot=[](const void * f) {
            return f==upstream_mmvq_pdl_dot_kernel<GGML_TYPE_Q4_K,1,false,false>() ||
                f==upstream_mmvq_pdl_dot_kernel<GGML_TYPE_Q5_K,1,false,false>() ||
                f==upstream_mmvq_pdl_dot_kernel<GGML_TYPE_IQ4_XS,1,false,false>() ||
                f==upstream_mmvq_pdl_dot_kernel<GGML_TYPE_Q4_K,1,true,false>() ||
                f==upstream_mmvq_pdl_dot_kernel<GGML_TYPE_Q5_K,1,true,false>() ||
                f==upstream_mmvq_pdl_dot_kernel<GGML_TYPE_IQ4_XS,1,true,false>();
        };
        for(auto node:nodes) {
            cudaGraphNodeType type{};
            status=cudaGraphNodeGetType(node,&type); if(status) return status;
            if(type!=cudaGraphNodeTypeKernel) {++counts[2];continue;}
            ++counts[1]; cudaKernelNodeParams p{};
            status=cudaGraphKernelNodeGetParams(node,&p); if(status) return status;
            if(p.func==clear) ++counts[3];
            if(p.func==reinterpret_cast<const void *>(upstream_mmvq_pdl_widen)) ++counts[4];
            if(p.func==reinterpret_cast<const void *>(quantize_q8_1<true>)) ++counts[5];
            if(p.func==direct) ++counts[6];
            if(p.func==cast) ++counts[7];
        }
        for(size_t i=0;i<ne;++i) {
            if(edges[i].type!=cudaGraphDependencyTypeProgrammatic) continue;
            ++counts[9];
            if(edges[i].from_port!=cudaGraphKernelNodePortProgrammatic || edges[i].to_port!=0) continue;
            cudaKernelNodeParams a{},b{};
            status=cudaGraphKernelNodeGetParams(from[i],&a); if(status) return status;
            status=cudaGraphKernelNodeGetParams(to[i],&b); if(status) return status;
            if(a.func==cast && b.func==clear) ++counts[10];
            if(a.func==clear && b.func==direct) ++counts[11];
            if(a.func==direct && is_dot(b.func)) ++counts[12];
        }
        return 0;
    });
#else
    return -2;
#endif
}
