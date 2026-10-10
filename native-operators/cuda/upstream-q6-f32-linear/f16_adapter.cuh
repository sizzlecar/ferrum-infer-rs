// Explicit diagnostic F16 boundary over the existing Q6 D4/MMQ implementation.
// Included after the private F32 helpers in mmq.cu; no duplicated dot/pack math.
#pragma once

static FerrumUpstreamQ6F32RequestV1 q6_f16_geometry_request(const FerrumUpstreamQ6F16RequestV1 & r) {
    return {1, sizeof(FerrumUpstreamQ6F32RequestV1), r.format, r.layout, r.rows,
        r.inputs, r.outputs, r.cc, r.sm_count, r.reserved, r.shared_limit};
}

static int q6_f16_plan(const FerrumUpstreamQ6F16RequestV1 * r,
        FerrumUpstreamQ6F16PlanV1 & out, Q6MmqGeometry & n) {
    if (!r || r->abi!=2 || r->size!=sizeof(*r)) return -1;
    // Only the private geometry calculation uses an F32-shaped request. The
    // public F32 request and launch validators remain closed to abi2.
    const auto geometry_request=q6_f16_geometry_request(*r);
    FerrumUpstreamQ6F32PlanV1 geometry{};
    const int status=q6_plan(&geometry_request,geometry,n);
    if (status) return status;
    out={}; out.request=*r; out.abi=2; out.size=sizeof(out);
#define FERRUM_Q6_COPY_GEOMETRY(field) out.field=geometry.field
    FERRUM_Q6_COPY_GEOMETRY(algorithm);
    FERRUM_Q6_COPY_GEOMETRY(pack_abi);
    FERRUM_Q6_COPY_GEOMETRY(padded_inputs);
    FERRUM_Q6_COPY_GEOMETRY(padded_outputs);
    FERRUM_Q6_COPY_GEOMETRY(guard_blocks);
    FERRUM_Q6_COPY_GEOMETRY(j);
    FERRUM_Q6_COPY_GEOMETRY(i);
    FERRUM_Q6_COPY_GEOMETRY(nthreads);
    FERRUM_Q6_COPY_GEOMETRY(shared_bytes);
    FERRUM_Q6_COPY_GEOMETRY(blocks);
    FERRUM_Q6_COPY_GEOMETRY(tiles_y);
    FERRUM_Q6_COPY_GEOMETRY(fixup);
    FERRUM_Q6_COPY_GEOMETRY(ncols);
    FERRUM_Q6_COPY_GEOMETRY(channels);
    FERRUM_Q6_COPY_GEOMETRY(nwarps);
    FERRUM_Q6_COPY_GEOMETRY(rows_per_block);
    FERRUM_Q6_COPY_GEOMETRY(small_k);
    FERRUM_Q6_COPY_GEOMETRY(reserved);
    FERRUM_Q6_COPY_GEOMETRY(weight_bytes);
    FERRUM_Q6_COPY_GEOMETRY(converted_bytes);
    FERRUM_Q6_COPY_GEOMETRY(packed_bytes);
    FERRUM_Q6_COPY_GEOMETRY(output_bytes);
    FERRUM_Q6_COPY_GEOMETRY(fixup_bytes);
#undef FERRUM_Q6_COPY_GEOMETRY
    return 0;
}

static bool q6_f16_valid(const FerrumUpstreamQ6F16PlanV1 * p, Q6MmqGeometry & n) {
    FerrumUpstreamQ6F16PlanV1 expected{};
    return p && !q6_f16_plan(&p->request,expected,n) && std::memcmp(p,&expected,sizeof(*p))==0;
}

// Same F16 boundary expressions as upstream-extra-linear. Only live K values
// are read; logical K padding is positive zero. Odd element strides and a
// two-byte aligned sub-leaf base are legal.
static __global__ void q6_f16_convert(const half * input,float * converted,
        int rows,int inputs,int input_stride,int padded_inputs) {
    const uint64_t index=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if (index<uint64_t(rows)*padded_inputs) {
        const uint64_t row=index/padded_inputs, col=index%padded_inputs;
        converted[index]=col<inputs ? __half2float(input[row*input_stride+col]) : 0.0f;
    }
}
static __global__ void q6_f16_cast(const float * input,half * output,int rows,int columns,
        int stride,const uint32_t * row_flags,const uint32_t * weight_flag) {
    const uint64_t index=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if (index>=uint64_t(rows)*columns) return;
    const float value=input[index]; const half rounded=__float2half_rn(value);
    const bool bad=*weight_flag || row_flags[index/columns] ||
        !marker_finite(value) || !marker_half_finite(rounded);
    output[(index/columns)*stride+index%columns]=bad ? __ushort_as_half(0x7e00) : rounded;
}

extern "C" int ferrum_upstream_q6_f16_plan_v1(const FerrumUpstreamQ6F16RequestV1 * r,
        FerrumUpstreamQ6F16PlanV1 * out) noexcept {
    return ferrum_upstream_q6_f32_boundary([&]() -> int {
        if (!out) return -1;
        FerrumUpstreamQ6F16PlanV1 candidate{}; Q6MmqGeometry n{};
        int status=q6_f16_plan(r,candidate,n); if (status) return status;
        status=ferrum_upstream_q6_f32_verify_device(q6_f16_geometry_request(*r));
        if (status) return status;
        status=q6_mmq_type<GGML_TYPE_Q6_K>(n,nullptr,nullptr,nullptr,nullptr,nullptr,true);
        if (!status) *out=candidate;
        return status;
    });
}
extern "C" int ferrum_upstream_q6_f16_pack_v1(const FerrumUpstreamQ6F16PlanV1 * p,
        const void * input,uint32_t stride,void * converted,void * packed,void * rows,void * s) noexcept {
    return ferrum_upstream_q6_f32_boundary([&]() -> int {
        Q6MmqGeometry n{}; if (!q6_f16_valid(p,n)) return -1;
        if (!input||!converted||!packed||!rows||stride<n.inputs||uint64_t(stride)*n.rows>INT_MAX||
                uintptr_t(input)%2||uintptr_t(converted)%16||uintptr_t(packed)%16||uintptr_t(rows)%4) return -5;
        const auto stream=static_cast<cudaStream_t>(s);
        auto status=cudaMemsetAsync(rows,0,n.rows*sizeof(uint32_t),stream);
        if (status!=cudaSuccess) return status;
        if (p->guard_blocks) {
            status=cudaMemsetAsync(static_cast<char *>(packed)+uint64_t(n.rows)*n.padded_inputs*9/8,
                0,p->guard_blocks*144,stream);
            if (status!=cudaSuccess) return status;
        }
        q6_f16_convert<<<(n.rows*n.padded_inputs+255)/256,256,0,stream>>>(static_cast<const half *>(input),
            static_cast<float *>(converted),n.rows,n.inputs,stride,n.padded_inputs);
        status=cudaGetLastError(); if (status!=cudaSuccess) return status;
        const dim3 grid(n.rows,n.padded_inputs/(4*CUDA_QUANTIZE_BLOCK_SIZE_MMQ),1);
        quantize_mmq_q8_1<MMQ_Q8_1_DS_LAYOUT_D4,false,true><<<grid,CUDA_QUANTIZE_BLOCK_SIZE_MMQ,0,stream>>>(
            static_cast<const float *>(converted),nullptr,packed,n.inputs,n.padded_inputs,
            n.rows*n.padded_inputs,n.rows*n.padded_inputs,n.padded_inputs,n.rows,1,0,
            static_cast<uint32_t *>(rows));
        return cudaGetLastError();
    });
}
extern "C" int ferrum_upstream_q6_f16_dot_v1(const FerrumUpstreamQ6F16PlanV1 * p,
        const void * weights,const void * packed,void * raw,void * fixup,void * s) noexcept {
    return ferrum_upstream_q6_f32_boundary([&]() -> int {
        Q6MmqGeometry n{}; if (!q6_f16_valid(p,n)) return -1;
        if (!weights||!packed||!raw||(n.fixup&&!fixup)||uintptr_t(weights)%4||uintptr_t(packed)%16||
                uintptr_t(raw)%4||uintptr_t(fixup)%4) return -5;
        return q6_mmq_type<GGML_TYPE_Q6_K>(n,weights,packed,static_cast<float *>(raw),
            static_cast<float *>(fixup),static_cast<cudaStream_t>(s),false);
    });
}
extern "C" int ferrum_upstream_q6_f16_check_weights_v1(const FerrumUpstreamQ6F16PlanV1 * p,
        const void * weights,void * flag,void * s) noexcept {
    return ferrum_upstream_q6_f32_boundary([&]() -> int {
        Q6MmqGeometry n{}; if (!q6_f16_valid(p,n)) return -1;
        if (!weights||!flag||uintptr_t(weights)%4||uintptr_t(flag)%4) return -5;
        const auto stream=static_cast<cudaStream_t>(s);
        auto status=cudaMemsetAsync(flag,0,sizeof(uint32_t),stream);
        if (status!=cudaSuccess) return status;
        const uint64_t count=uint64_t(n.outputs)*(n.inputs/256);
        q6_weights<<<(count+255)/256,256,0,stream>>>(static_cast<const unsigned char *>(weights),count,
            static_cast<uint32_t *>(flag));
        return cudaGetLastError();
    });
}
extern "C" int ferrum_upstream_q6_f16_cast_v1(const FerrumUpstreamQ6F16PlanV1 * p,
        const void * raw,void * output,uint32_t stride,const void * rows,const void * weight,void * s) noexcept {
    return ferrum_upstream_q6_f32_boundary([&]() -> int {
        Q6MmqGeometry n{}; if (!q6_f16_valid(p,n)) return -1;
        if (!raw||!output||!rows||!weight||stride<n.outputs||uint64_t(stride)*n.rows>INT_MAX||
                uintptr_t(raw)%4||uintptr_t(output)%2||uintptr_t(rows)%4||uintptr_t(weight)%4) return -5;
        q6_f16_cast<<<(n.rows*n.outputs+255)/256,256,0,static_cast<cudaStream_t>(s)>>>(
            static_cast<const float *>(raw),static_cast<half *>(output),n.rows,n.outputs,stride,
            static_cast<const uint32_t *>(rows),static_cast<const uint32_t *>(weight));
        return cudaGetLastError();
    });
}
