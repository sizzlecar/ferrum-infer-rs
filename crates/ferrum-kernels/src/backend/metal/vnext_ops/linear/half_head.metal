#include <metal_stdlib>
using namespace metal;

struct HalfHeadQ6Block { uchar ql[128]; uchar qh[64]; char scales[16]; half d; };
struct HalfHeadParams { uint rows; uint in_features; uint out_features; uint output_stride; uint output_column_offset; };

// Half conversion precedes multiplication for both operands. Never factor the
// Q6 scale outside the product or substitute the strict-F32 B4 reduction.
// Reuse each half-rounded input across the two output rows.
template <ushort B>
void half_head(device const float *input, device const HalfHeadQ6Block *weights,
               device float *output, constant HalfHeadParams &p,
               uint group, ushort lane, ushort sg) {
    const uint first = (group * 2 + sg) * 2;
    if (first >= p.out_features) return;
    const uint blocks = p.in_features / 256;
    const ushort tid = lane / 2, ix = lane % 2;
    const ushort ip = tid / 8, il = tid % 8, l0 = 4 * il;
    const ushort scale_index = 8 * ip + l0 / 16;
    float sums[B][2] = {};
    for (uint block = ix; block < blocks; block += 2) {
        half4 cached[B][4];
        for (ushort batch = 0; batch < B; ++batch) {
            device const float *y = input + ulong(batch) * p.in_features + block * 256 + 128 * ip + l0;
            // Host input alignment is 16 bytes, K is divisible by 256 and
            // l0 is divisible by four. Every vector load is fully in bounds.
            for (ushort quarter = 0; quarter < 4; ++quarter) {
                cached[batch][quarter] = half4(*(device const float4 *)(y + 32 * quarter));
            }
        }
        for (ushort row = 0; row < 2; ++row) {
            if (first + row >= p.out_features) continue;
            device const HalfHeadQ6Block &w = weights[(first + row) * blocks + block];
            device const uchar *q1 = w.ql + 64 * ip + l0;
            device const uchar *q2 = q1 + 32;
            device const uchar *qh = w.qh + 32 * ip + l0;
            device const char *sc = w.scales + scale_index;
            const float4 scale = float(w.d) * float4(sc[0], sc[2], sc[4], sc[6]);
            float4 decoded[4];
            for (ushort l = 0; l < 4; ++l) {
                const int4 q = int4(
                    ((q1[l] & 15) | ((qh[l] & 3) << 4)) - 32,
                    ((q2[l] & 15) | ((qh[l] & 12) << 2)) - 32,
                    ((q1[l] >> 4) | (qh[l] & 48)) - 32,
                    ((q2[l] >> 4) | ((qh[l] & 192) >> 2)) - 32);
                decoded[l] = float4(half4(scale * float4(q)));
            }
            for (ushort batch = 0; batch < B; ++batch) {
                float4 partial = 0.0f;
                for (ushort l = 0; l < 4; ++l) {
                    const float4 x = float4(cached[batch][0][l], cached[batch][1][l],
                                            cached[batch][2][l], cached[batch][3][l]);
                    partial += x * decoded[l];
                }
                sums[batch][row] += partial[0] + partial[1] + partial[2] + partial[3];
            }
        }
    }
    for (ushort batch = 0; batch < B; ++batch) {
        for (ushort row = 0; row < 2; ++row) {
            if (first + row >= p.out_features) continue;
            const float sum = simd_sum(sums[batch][row]);
            if (lane == 0) output[ulong(batch) * p.output_stride + p.output_column_offset + first + row] = sum;
        }
    }
}

#define HALF_HEAD_ENTRY(B) \
kernel void q6_half_head_b##B(device const float *input [[buffer(0)]], \
    device const HalfHeadQ6Block *weights [[buffer(1)]], device float *output [[buffer(2)]], \
    constant HalfHeadParams &p [[buffer(3)]], uint3 group [[threadgroup_position_in_grid]], \
    ushort lane [[thread_index_in_simdgroup]], ushort sg [[simdgroup_index_in_threadgroup]]) { \
    if (p.rows != B) return; half_head<B>(input, weights, output, p, group.x, lane, sg); \
}
HALF_HEAD_ENTRY(1)
HALF_HEAD_ENTRY(2)
HALF_HEAD_ENTRY(3)
HALF_HEAD_ENTRY(4)
