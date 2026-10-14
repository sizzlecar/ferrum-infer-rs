// TEST TU: compare the ordered adapter with its real programmatic capability.
#include "mmq.cu"
#include <vector>

extern "C" int ferrum_test_upstream_mmq_pdl_support_v1(const FerrumUpstreamLinearPlanV1 * p, uint32_t * supported) noexcept {
    return ferrum_upstream_mmq_programmatic_support_v1(p,supported);
}

extern "C" int ferrum_test_upstream_mmq_pdl_pack_f16_small_v2(const FerrumUpstreamLinearPlanV1 * p, const void * input, uint32_t stride, void * converted, void * packed, void * flags, void * stream) noexcept {
    return ferrum_upstream_mmq_programmatic_pack_f16_small_v2(p,input,stride,converted,packed,flags,stream);
}

extern "C" int ferrum_test_upstream_mmq_pdl_dot_v1(const FerrumUpstreamLinearPlanV1 * p, const void * weights, const void * packed, void * output, void * fixup, void * stream) noexcept {
    return ferrum_upstream_mmq_programmatic_dot_v1(p,weights,packed,output,fixup,stream);
}

extern "C" int ferrum_test_upstream_mmq_pdl_cast_v2(const FerrumUpstreamLinearPlanV1 * p, const void * input, void * output, uint32_t stride, const void * rows, const void * weights, void * stream) noexcept {
    return ferrum_upstream_mmq_programmatic_cast_v2(p,input,output,stride,rows,weights,stream);
}

// counts: all, programmatic, clear->pack, pack->main, main->fixup,
// fixup->cast, main->cast (no fixup), cast->next-clear. Match actual functions.
extern "C" int ferrum_test_upstream_mmq_pdl_graph_edges_v1(
        const FerrumUpstreamLinearPlanV1 * p, void * graph_ptr,
        uint64_t * counts, size_t counts_len) noexcept {
    return ferrum_upstream_boundary([&]() -> int {
        TestMmqPlan n{};
        if (!upstream_mmq_pdl_valid(p,n) || !graph_ptr || !counts || counts_len < 8) return -1;
        for (size_t i = 0; i < 8; ++i) counts[i] = 0;
#if defined(GGML_CUDA_USE_PDL) && CUDART_VERSION >= 12030
        const auto kernels = upstream_mmq_pdl_kernels(n);
        const void * clear = reinterpret_cast<const void *>(upstream_mmq_pdl_clear);
        const void * cast = reinterpret_cast<const void *>(upstream_mmq_pdl_cast_kernel);
        const void * from_kernels[] = {clear,kernels.pack,kernels.main,kernels.fixup,kernels.main,cast};
        const void * to_kernels[] = {kernels.pack,kernels.main,kernels.fixup,cast,cast,clear};
        const auto graph = static_cast<cudaGraph_t>(graph_ptr);
        size_t count = 0;
        auto status = cudaGraphGetEdges_v2(graph,nullptr,nullptr,nullptr,&count);
        if (status != cudaSuccess) return status;
        std::vector<cudaGraphNode_t> from(count), to(count);
        std::vector<cudaGraphEdgeData> data(count);
        status = cudaGraphGetEdges_v2(graph,from.data(),to.data(),data.data(),&count);
        if (status != cudaSuccess) return status;
        counts[0] = count;
        for (size_t i = 0; i < count; ++i) {
            if (data[i].type != cudaGraphDependencyTypeProgrammatic) continue;
            ++counts[1];
            if (data[i].from_port != cudaGraphKernelNodePortProgrammatic || data[i].to_port != 0) continue;
            bool reserved_zero = true;
            for (const auto byte : data[i].reserved) reserved_zero &= byte == 0;
            if (!reserved_zero) continue;
            cudaGraphNodeType a_type{}, b_type{};
            status = cudaGraphNodeGetType(from[i],&a_type); if (status != cudaSuccess) return status;
            status = cudaGraphNodeGetType(to[i],&b_type); if (status != cudaSuccess) return status;
            if (a_type != cudaGraphNodeTypeKernel || b_type != cudaGraphNodeTypeKernel) continue;
            cudaKernelNodeParams a{}, b{};
            status = cudaGraphKernelNodeGetParams(from[i],&a); if (status != cudaSuccess) return status;
            status = cudaGraphKernelNodeGetParams(to[i],&b); if (status != cudaSuccess) return status;
            for (size_t stage = 0; stage < 6; ++stage) {
                if (a.func == from_kernels[stage] && b.func == to_kernels[stage]) ++counts[2+stage];
            }
        }
        return 0;
#else
        return -2;
#endif
    });
}
