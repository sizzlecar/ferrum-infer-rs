// Small decode batches reuse packed weights across independent activation rows.
// Quantized dot-product order follows q4_k_gemv_v2.metal / q6_k_gemv.metal;
// Input/output retain their declared F16 or F32 type; arithmetic and SIMD
// reduction remain FP32, including the model's F32 vocabulary boundary.
#include <metal_stdlib>
using namespace metal;

#define UNROLL(x) _Pragma("clang loop unroll(full)") for (x)
struct Q4Block { half d; half dmin; uchar scales[12]; uchar qs[128]; };
struct Q6Block { uchar ql[128]; uchar qh[64]; char scales[16]; half d; };
struct Params {
    uint rows;
    uint in_features;
    uint out_features;
    uint output_stride;
    uint output_column_offset;
};

template <ushort B, typename T>
void q4_shared(device const T * input, device const Q4Block * weights,
               device T * output, constant Params & p,
               uint group, ushort lane, ushort simdgroup) {
    const uint first = (group * 2 + simdgroup) * 2;
    if (first >= p.out_features) return;
    const uint blocks = p.in_features / 256;
    const ushort ix = lane / 8;
    const ushort it = lane % 8;
    const ushort iq = it / 4;
    const ushort ir = it % 4;
    float sums[B][2] = {};
    for (uint block = ix; block < blocks; block += 4) {
        UNROLL (ushort row = 0; row < 2; ++row) {
            if (first + row >= p.out_features) continue;
            device const Q4Block & w = weights[(first + row) * blocks + block];
            device const ushort * sc = (device const ushort *)w.scales + iq;
            ushort packed_sc[4];
            packed_sc[0] = sc[0] & 0x3f3f;
            packed_sc[1] = sc[2] & 0x3f3f;
            packed_sc[2] = (sc[4] & 0x0f0f) | ((sc[0] & 0xc0c0) >> 2);
            packed_sc[3] = ((sc[4] >> 4) & 0x0f0f) | ((sc[2] & 0xc0c0) >> 2);
            thread const uchar * scales = (thread const uchar *)packed_sc;
            device const ushort * q = (device const ushort *)w.qs + 16 * iq + 4 * ir;
            ushort qlo[4], qhi[4];
            UNROLL (ushort i = 0; i < 4; ++i) { qlo[i] = q[i]; qhi[i] = q[i + 32]; }
            const float d = w.d;
            const float dmin = w.dmin;
            UNROLL (ushort batch = 0; batch < B; ++batch) {
                device const T * y = input + ulong(batch) * p.in_features
                    + block * 256 + 64 * iq + 8 * ir;
                float yl[16], yh[16];
                float4 sumy = 0.0f;
                UNROLL (ushort i = 0; i < 8; ++i) {
                    yl[i] = y[i]; sumy[0] += yl[i];
                    yl[i + 8] = y[i + 32]; sumy[1] += yl[i + 8];
                    yh[i] = y[i + 128]; sumy[2] += yh[i];
                    yh[i + 8] = y[i + 160]; sumy[3] += yh[i + 8];
                }
                float4 acc1 = 0.0f, acc2 = 0.0f;
                UNROLL (ushort i = 0; i < 4; ++i) {
                    acc1[0] += yl[2*i] * (qlo[i] & 0x000f);
                    acc1[1] += yl[2*i + 1] * (qlo[i] & 0x0f00);
                    acc1[2] += yl[2*i + 8] * (qlo[i] & 0x00f0);
                    acc1[3] += yl[2*i + 9] * (qlo[i] & 0xf000);
                    acc2[0] += yh[2*i] * (qhi[i] & 0x000f);
                    acc2[1] += yh[2*i + 1] * (qhi[i] & 0x0f00);
                    acc2[2] += yh[2*i + 8] * (qhi[i] & 0x00f0);
                    acc2[3] += yh[2*i + 9] * (qhi[i] & 0xf000);
                }
                sums[batch][row] += d * (
                    (acc1[0] + (1.f/256.f) * acc1[1]) * scales[0] +
                    (acc1[2] + (1.f/256.f) * acc1[3]) * scales[1] * (1.f/16.f) +
                    (acc2[0] + (1.f/256.f) * acc2[1]) * scales[4] +
                    (acc2[2] + (1.f/256.f) * acc2[3]) * scales[5] * (1.f/16.f)
                ) - dmin * (sumy[0] * scales[2] + sumy[1] * scales[3] +
                            sumy[2] * scales[6] + sumy[3] * scales[7]);
            }
        }
    }
    UNROLL (ushort batch = 0; batch < B; ++batch) {
        UNROLL (ushort row = 0; row < 2; ++row) {
            if (first + row >= p.out_features) continue;
            const float sum = simd_sum(sums[batch][row]);
            if (lane == 0) output[ulong(batch) * p.output_stride + p.output_column_offset + first + row] = T(sum);
        }
    }
}

template <ushort B, typename T>
void q6_shared(device const T * input, device const Q6Block * weights,
               device T * output, constant Params & p,
               uint group, ushort lane, ushort simdgroup) {
    const uint first = (group * 2 + simdgroup) * 2;
    if (first >= p.out_features) return;
    const uint blocks = p.in_features / 256;
    const ushort tid = lane / 2, ix = lane % 2;
    const ushort ip = tid / 8, il = tid % 8, l0 = 4 * il;
    const ushort scale_index = 8 * ip + l0 / 16;
    float sums[B][2] = {};
    for (uint block = ix; block < blocks; block += 2) {
        UNROLL (ushort row = 0; row < 2; ++row) {
            if (first + row >= p.out_features) continue;
            device const Q6Block & w = weights[(first + row) * blocks + block];
            device const uchar * q1 = w.ql + 64 * ip + l0;
            device const uchar * q2 = q1 + 32;
            device const uchar * qh = w.qh + 32 * ip + l0;
            int4 quants[4];
            UNROLL (ushort l = 0; l < 4; ++l) {
                quants[l] = int4(
                    ((q1[l] & 0x0f) | ((qh[l] & 0x03) << 4)) - 32,
                    ((q2[l] & 0x0f) | ((qh[l] & 0x0c) << 2)) - 32,
                    ((q1[l] >> 4) | (qh[l] & 0x30)) - 32,
                    ((q2[l] >> 4) | ((qh[l] & 0xc0) >> 2)) - 32);
            }
            device const char * sc = w.scales + scale_index;
            const float4 scale = float4(sc[0], sc[2], sc[4], sc[6]);
            const float d = w.d;
            UNROLL (ushort batch = 0; batch < B; ++batch) {
                device const T * y = input + ulong(batch) * p.in_features
                    + block * 256 + 128 * ip + l0;
                float4 partial = 0.0f;
                UNROLL (ushort l = 0; l < 4; ++l) {
                    partial[0] += float(y[l]) * quants[l][0];
                    partial[1] += float(y[l + 32]) * quants[l][1];
                    partial[2] += float(y[l + 64]) * quants[l][2];
                    partial[3] += float(y[l + 96]) * quants[l][3];
                }
                sums[batch][row] += d * (partial[0] * scale[0] + partial[1] * scale[1]
                                      + partial[2] * scale[2] + partial[3] * scale[3]);
            }
        }
    }
    UNROLL (ushort batch = 0; batch < B; ++batch) {
        UNROLL (ushort row = 0; row < 2; ++row) {
            if (first + row >= p.out_features) continue;
            const float sum = simd_sum(sums[batch][row]);
            if (lane == 0) output[ulong(batch) * p.output_stride + p.output_column_offset + first + row] = T(sum);
        }
    }
}

#define ENTRY(FORMAT, B, T, NAME) \
kernel void NAME( \
    device const T * input [[buffer(0)]], \
    device const FORMAT##Block * weights [[buffer(1)]], \
    device T * output [[buffer(2)]], \
    constant Params & p [[buffer(3)]], \
    uint3 group [[threadgroup_position_in_grid]], \
    ushort lane [[thread_index_in_simdgroup]], \
    ushort simdgroup [[simdgroup_index_in_threadgroup]]) { \
    if (p.rows != B) return; \
    FORMAT##_shared<B, T>(input, weights, output, p, group.x, lane, simdgroup); \
}

// Token aliases keep the exported pipeline names lower-case.
typedef Q4Block q4Block;
typedef Q6Block q6Block;
ENTRY(q4, 2, half, q4_shared_b2)
ENTRY(q4, 3, half, q4_shared_b3)
ENTRY(q4, 4, half, q4_shared_b4)
ENTRY(q6, 2, half, q6_shared_b2)
ENTRY(q6, 3, half, q6_shared_b3)
ENTRY(q6, 4, half, q6_shared_b4)
ENTRY(q6, 2, float, q6_shared_f32_b2)
ENTRY(q6, 3, float, q6_shared_f32_b3)
ENTRY(q6, 4, float, q6_shared_f32_b4)
