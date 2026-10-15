// TEST-only replacement TU. Test entrypoints delegate the normal capability;
// one archive compares original, output-only, and both-boundary variants.
// Do not link this object together with the normal mmvq object.
#include "mmvq-direct-f16-pdl-test.cu"

extern "C" int ferrum_test_mmvq_f16_output_pdl_support_v1(
        const FerrumUpstreamLinearPlanV1 * p, uint32_t * supported) noexcept {
    return ferrum_upstream_mmvq_m1_f16_pdl_support_v1(p,supported);
}

extern "C" int ferrum_test_mmvq_f16_output_pdl_dot_v1(
        const FerrumUpstreamLinearPlanV1 * p, const void * weights,
        const void * packed, void * output, uint32_t output_stride,
        const void * row_flags, const void * weight_flag, void * s) noexcept {
    return ferrum_upstream_mmvq_m1_f16_pdl_dot_v1(
        p,weights,packed,output,output_stride,row_flags,weight_flag,s);
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
            return f==upstream_mmvq_m1_f16_pdl_kernel<GGML_TYPE_Q4_K,false>() ||
                f==upstream_mmvq_m1_f16_pdl_kernel<GGML_TYPE_Q5_K,false>() ||
                f==upstream_mmvq_m1_f16_pdl_kernel<GGML_TYPE_IQ4_XS,false>() ||
                f==upstream_mmvq_m1_f16_pdl_kernel<GGML_TYPE_Q4_K,true>() ||
                f==upstream_mmvq_m1_f16_pdl_kernel<GGML_TYPE_Q5_K,true>() ||
                f==upstream_mmvq_m1_f16_pdl_kernel<GGML_TYPE_IQ4_XS,true>();
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
