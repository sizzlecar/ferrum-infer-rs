// TEST only. Compile this TU separately with the production RMS flags:
// -O3 --use_fast_math --expt-relaxed-constexpr -std=c++17, not --fmad=false.
// Include the original ferrum-kernels/kernels directory. Do not include GGML's
// common.cuh or compile the retained MMVQ archive with these flags.
#include <cuda_runtime.h>
#include <cstdint>
#include <vector>
#include "rms_norm.cu"

static __global__ void test_rms_master_pdl(const float * input,
        const half * weight, half * output, int width, float epsilon) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    // Every thread triggers before waiting. No dependent read or scratch
    // overwrite precedes the wait, including at the next group's boundary.
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
    vnext_rms_norm_master(input, weight, output, width, epsilon);
}

extern "C" int ferrum_test_rms_mmvq_support_v1(uint32_t * supported) noexcept {
    if (!supported) return -1;
    *supported = 0;
#if CUDART_VERSION >= 12030
    int device = 0, major = 0;
    auto status = cudaGetDevice(&device);
    if (status != cudaSuccess) return status;
    status = cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
    if (status != cudaSuccess) return status;
    if (major < 9) return 0;
    cudaFuncAttributes normal{}, candidate{};
    status = cudaFuncGetAttributes(&normal, vnext_rms_norm_f32_to_f16);
    if (status != cudaSuccess) return status;
    status = cudaFuncGetAttributes(&candidate, test_rms_master_pdl);
    if (status != cudaSuccess) return status;
    *supported = normal.ptxVersion >= 90 && candidate.ptxVersion >= 90 &&
        normal.maxThreadsPerBlock >= 1024 && candidate.maxThreadsPerBlock >= 1024;
#endif
    return 0;
}

extern "C" int ferrum_test_rms_mmvq_launch_v1(const void * input,
        const void * weight, void * output, float epsilon, uint32_t arm,
        void * stream_ptr) noexcept {
    if (!input || !weight || !output || uintptr_t(input)%4 || uintptr_t(weight)%2 ||
            uintptr_t(output)%2 || !(epsilon > 0 && epsilon <= 1) || arm > 1) return -1;
    auto stream = static_cast<cudaStream_t>(stream_ptr);
    const auto x = static_cast<const float *>(input);
    const auto w = static_cast<const half *>(weight);
    const auto y = static_cast<half *>(output);
    if (arm == 0) {
        vnext_rms_norm_f32_to_f16<<<1,1024,0,stream>>>(x,w,y,5120,epsilon);
        return cudaGetLastError();
    }
#if CUDART_VERSION >= 12030
    cudaLaunchAttribute attr{};
    attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr.val.programmaticStreamSerializationAllowed = 1;
    cudaLaunchConfig_t config{};
    config.gridDim = dim3(1);
    config.blockDim = dim3(1024);
    config.stream = stream;
    config.attrs = &attr;
    config.numAttrs = 1;
    return cudaLaunchKernelEx(&config,test_rms_master_pdl,x,w,y,5120,epsilon);
#else
    return -2;
#endif
}

// Header-owned graph ABI. First call queries sizes with null arrays. Nodes:
// [handle, function, node_type, RMS kind(0 other/1 original/2 PDL)]. Edges:
// [from_handle,to_handle,type,from_port,to_port]. No assumed A/B edge difference:
// a producer without explicit LC can still have a programmatic outgoing edge.
extern "C" int ferrum_test_rms_mmvq_graph_v1(void * graph_ptr,
        uint64_t * nodes_out, size_t node_words, uint64_t * edges_out,
        size_t edge_words, uint64_t * counts) noexcept {
    if (!graph_ptr || !counts) return -1;
    counts[0] = counts[1] = 0;
#if CUDART_VERSION >= 12030
    try {
        const auto graph = static_cast<cudaGraph_t>(graph_ptr);
        size_t nodes = 0, edges = 0;
        auto status = cudaGraphGetNodes(graph,nullptr,&nodes);
        if (status != cudaSuccess) return status;
        status = cudaGraphGetEdges_v2(graph,nullptr,nullptr,nullptr,&edges);
        if (status != cudaSuccess) return status;
        counts[0] = nodes; counts[1] = edges;
        counts[2] = cudaGraphDependencyTypeProgrammatic;
        counts[3] = cudaGraphKernelNodePortProgrammatic;
        if (!nodes_out && !edges_out && !node_words && !edge_words) return 0;
        if (!nodes_out || !edges_out || node_words/4 < nodes || edge_words/5 < edges) return -1;
        std::vector<cudaGraphNode_t> handles(nodes), from(edges), to(edges);
        std::vector<cudaGraphEdgeData> data(edges);
        status = cudaGraphGetNodes(graph,handles.data(),&nodes);
        if (status != cudaSuccess) return status;
        status = cudaGraphGetEdges_v2(graph,from.data(),to.data(),data.data(),&edges);
        if (status != cudaSuccess) return status;
        for (size_t i=0; i<nodes; ++i) {
            cudaGraphNodeType type{};
            status = cudaGraphNodeGetType(handles[i],&type);
            if (status != cudaSuccess) return status;
            cudaKernelNodeParams params{};
            if (type == cudaGraphNodeTypeKernel) {
                status = cudaGraphKernelNodeGetParams(handles[i],&params);
                if (status != cudaSuccess) return status;
            }
            nodes_out[4*i] = reinterpret_cast<uintptr_t>(handles[i]);
            nodes_out[4*i+1] = reinterpret_cast<uintptr_t>(params.func);
            nodes_out[4*i+2] = type;
            nodes_out[4*i+3] = params.func == reinterpret_cast<void *>(vnext_rms_norm_f32_to_f16)
                ? 1 : params.func == reinterpret_cast<void *>(test_rms_master_pdl) ? 2 : 0;
        }
        for (size_t i=0; i<edges; ++i) {
            edges_out[5*i] = reinterpret_cast<uintptr_t>(from[i]);
            edges_out[5*i+1] = reinterpret_cast<uintptr_t>(to[i]);
            edges_out[5*i+2] = data[i].type;
            edges_out[5*i+3] = data[i].from_port;
            edges_out[5*i+4] = data[i].to_port;
        }
        return 0;
    } catch (...) { return -3; }
#else
    return -2;
#endif
}
