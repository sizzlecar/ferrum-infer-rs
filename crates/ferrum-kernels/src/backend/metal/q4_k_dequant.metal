// Q4_K_M dequant kernel — ferrum-native Metal implementation.
//
// Block layout (matches GGML / llama.cpp / candle CPU reference):
//   block_q4_K {
//       half d;                       // super-block scale-of-scales
//       half dmin;                    // super-block scale-of-mins
//       uchar scales[12];             // 8 sub-blocks × (6-bit scale + 6-bit min)
//       uchar qs[128];                // 256 weights × 4-bit (packed 2 per byte)
//   }                                 // total 144 bytes / 256 weights = 4.5 bits/w
//
// One thread per super-block: each thread expands 256 weights into 256
// fp16 outputs. Ports candle_core::quantized::k_quants::BlockQ4K::to_float
// (which itself follows ggml's reference) verbatim — every constant and
// loop bound matches so we get bit-identical output up to fp16 rounding.

#include <metal_stdlib>
using namespace metal;

#define QK_K          256
#define K_SCALE_SIZE  12

struct block_q4_K {
    half  d;
    half  dmin;
    uchar scales[K_SCALE_SIZE];
    uchar qs[QK_K / 2];
};

// 6-bit scale & 6-bit min unpacker. `j` ∈ [0, 8). Matches candle's
// `get_scale_min_k4` (crates/candle-core/src/quantized/utils.rs).
static inline void get_scale_min_k4(
    int j,
    thread const uchar * q,
    thread uchar       & sc,
    thread uchar       & mn
) {
    if (j < 4) {
        sc = q[j]     & 63;
        mn = q[j + 4] & 63;
    } else {
        sc = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4);
        mn = (q[j + 4] >> 4)   | ((q[j]     >> 6) << 4);
    }
}

kernel void dequantize_q4_k_f16(
    device const block_q4_K * blocks [[buffer(0)]],
    device       half       * out    [[buffer(1)]],
    uint tid [[thread_position_in_grid]]
) {
    block_q4_K b = blocks[tid];
    const float d    = float(b.d);
    const float dmin = float(b.dmin);

    device half * y = out + tid * QK_K;
    int is = 0;

    for (int j = 0; j < QK_K; j += 64) {
        thread const uchar * q = b.qs + (j / 2); // 32 packed bytes → 64 weights

        uchar sc, mn;

        get_scale_min_k4(is, b.scales, sc, mn);
        const float d1 = d    * float(sc);
        const float m1 = dmin * float(mn);

        get_scale_min_k4(is + 1, b.scales, sc, mn);
        const float d2 = d    * float(sc);
        const float m2 = dmin * float(mn);

        // Lower 4 bits of each byte → first 32 weights of this 64-block
        for (int l = 0; l < 32; l++) {
            y[j + l] = half(d1 * float(q[l] & 0xF) - m1);
        }
        // Upper 4 bits of each byte → next 32 weights
        for (int l = 0; l < 32; l++) {
            y[j + l + 32] = half(d2 * float(q[l] >> 4) - m2);
        }
        is += 2;
    }
}
