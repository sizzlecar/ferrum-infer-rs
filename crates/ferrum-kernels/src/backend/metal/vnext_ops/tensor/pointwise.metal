#include <metal_stdlib>
using namespace metal;
#pragma METAL fp contract(off)

struct Params { ulong elements; ulong width; uint kind; uint row_broadcast; };

kernel void ferrum_tensor_pointwise_bf16(
    device const ushort* input [[buffer(0)]],
    device const ushort* parameter [[buffer(1)]],
    device const ushort* update [[buffer(2)]],
    device ushort* output [[buffer(3)]],
    constant Params& p [[buffer(4)]],
    uint index [[thread_position_in_grid]]) {
    if (ulong(index) >= p.elements) return;
    float x = decode_bf16(input[index]);
    float y;
    switch (p.kind) {
        case 0: y = x / (1.0f + exp(-x)); break;
        case 1: y = rounded_bf16(x / (1.0f + exp(-x))) * decode_bf16(parameter[index]); break;
        case 2: y = tanh(x); break;
        case 3: {
            ulong k = p.row_broadcast ? ulong(index) % p.width : ulong(index);
            y = x * rounded_bf16(1.0f + decode_bf16(parameter[k])); break;
        }
        case 4: {
            ulong k = p.row_broadcast ? ulong(index) % p.width : ulong(index);
            y = x + rounded_bf16(decode_bf16(parameter[k]) * decode_bf16(update[index])); break;
        }
        case 5: {
            float cubic = (x * x) * x;
            float inner = 0.7978845608028654f * (x + 0.044715f * cubic);
            y = (0.5f * x) * (1.0f + tanh(inner)); break;
        }
        default: y = x + decode_bf16(parameter[index]); break;
    }
    output[index] = encode_bf16(y);
}
