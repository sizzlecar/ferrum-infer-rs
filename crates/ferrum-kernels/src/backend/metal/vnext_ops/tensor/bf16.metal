#include <metal_stdlib>
using namespace metal;
#pragma METAL fp contract(off)

inline float decode_bf16(ushort bits) { return as_type<float>(uint(bits) << 16); }
inline ushort encode_bf16(float x) {
    uint bits = as_type<uint>(x);
    if ((bits & 0x7fffffff) > 0x7f800000) return ushort((bits >> 16) | 0x0040);
    return ushort((bits + 0x7fff + ((bits >> 16) & 1)) >> 16);
}
inline float rounded_bf16(float x) { return decode_bf16(encode_bf16(x)); }
