// Q5_K cooperative GEMV adapted from llama.cpp's kernel_mul_mv_q5_K_f32_impl
// (ggml/src/ggml-metal/ggml-metal.metal, MIT licensed).

#include <metal_stdlib>
using namespace metal;

#define QK_K 256
#define N_R0 2
#define N_SG 2

struct block_q5_K {
    half d;
    half dmin;
    uchar scales[12];
    uchar qh[QK_K / 8];
    uchar qs[QK_K / 2];
};

struct GemvQ5KBatchParams {
    uint rows;
    uint in_features;
    uint out_features;
    uint output_stride;
    uint output_column_offset;
};

kernel void gemv_f16a_q5kw_v2_batched(
    device const half * src1 [[buffer(0)]],
    device const block_q5_K * src0 [[buffer(1)]],
    device half * dst [[buffer(2)]],
    constant GemvQ5KBatchParams & p [[buffer(3)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    if (tgpig.y >= p.rows) return;

    constexpr ushort kmask1 = 0x3f3f;
    constexpr ushort kmask2 = 0x0f0f;
    constexpr ushort kmask3 = 0xc0c0;

    const int nb = int(p.in_features / QK_K);
    const int first_row = (int(tgpig.x) * N_SG + int(sgitg)) * N_R0;
    if (first_row >= int(p.out_features)) return;
    const int row_count = min(N_R0, int(p.out_features) - first_row);

    device const block_q5_K * x = src0 + first_row * nb;
    device const half * yy = src1 + ulong(tgpig.y) * p.in_features;
    const int row_stride = int(sizeof(block_q5_K)) * nb;

    float sumf[N_R0] = {0.f};
    float yl[16];
    float yh[16];

    const int tid = tiisg / 4;
    const int ix = tiisg % 4;
    const int iq = tid / 4;
    const int ir = tid % 4;
    const int l0 = 8 * ir;
    const int q_offset = 32 * iq + l0;
    const int y_offset = 64 * iq + l0;

    const uchar hm1 = 1u << (2 * iq);
    const uchar hm2 = hm1 << 1;
    const uchar hm3 = hm1 << 4;
    const uchar hm4 = hm2 << 4;

    ushort sc16[4];
    thread const uchar * sc8 = (thread const uchar *)sc16;
    device const half * y1 = yy + ix * QK_K + y_offset;

    for (int block = ix; block < nb; block += 4) {
        device const uchar * q1 = x[block].qs + q_offset;
        device const uchar * qh = x[block].qh + l0;
        device const half * dh = &x[block].d;
        device const ushort * scales = (device const ushort *)x[block].scales + iq;
        device const half * y2 = y1 + 128;

        float4 sumy = {0.f};
        for (short lane = 0; lane < 8; ++lane) {
            yl[lane] = float(y1[lane]);
            sumy[0] += yl[lane];
            yl[lane + 8] = float(y1[lane + 32]);
            sumy[1] += yl[lane + 8];
            yh[lane] = float(y2[lane]);
            sumy[2] += yh[lane];
            yh[lane + 8] = float(y2[lane + 32]);
            sumy[3] += yh[lane + 8];
        }

        for (int row = 0; row < row_count; ++row) {
            device const uchar * q2 = q1 + 64;

            sc16[0] = scales[0] & kmask1;
            sc16[1] = scales[2] & kmask1;
            sc16[2] = ((scales[4] >> 0) & kmask2) | ((scales[0] & kmask3) >> 2);
            sc16[3] = ((scales[4] >> 4) & kmask2) | ((scales[2] & kmask3) >> 2);

            float4 acc1 = {0.f};
            float4 acc2 = {0.f};
            for (short lane = 0; lane < 8; ++lane) {
                const uchar high = qh[lane];
                acc1[0] += yl[lane] * float(q1[lane] & 0x0f);
                acc1[1] += yl[lane + 8] * float(q1[lane] & 0xf0);
                acc1[2] += yh[lane] * float(q2[lane] & 0x0f);
                acc1[3] += yh[lane + 8] * float(q2[lane] & 0xf0);
                acc2[0] += (high & hm1) != 0 ? yl[lane] : 0.f;
                acc2[1] += (high & hm2) != 0 ? yl[lane + 8] : 0.f;
                acc2[2] += (high & hm3) != 0 ? yh[lane] : 0.f;
                acc2[3] += (high & hm4) != 0 ? yh[lane + 8] : 0.f;
            }

            sumf[row] += float(dh[0]) * (
                float(sc8[0]) * (acc1[0] + 16.f * acc2[0])
                + float(sc8[1]) * (acc1[1] / 16.f + 16.f * acc2[1])
                + float(sc8[4]) * (acc1[2] + 16.f * acc2[2])
                + float(sc8[5]) * (acc1[3] / 16.f + 16.f * acc2[3])
            ) - float(dh[1]) * (
                sumy[0] * float(sc8[2]) + sumy[1] * float(sc8[3])
                + sumy[2] * float(sc8[6]) + sumy[3] * float(sc8[7])
            );

            q1 += row_stride;
            qh += row_stride;
            dh += row_stride / 2;
            scales += row_stride / 2;
        }
        y1 += 4 * QK_K;
    }

    for (int row = 0; row < row_count; ++row) {
        const float total = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst[ulong(tgpig.y) * p.output_stride
                + p.output_column_offset + first_row + row] = half(total);
        }
    }
}
