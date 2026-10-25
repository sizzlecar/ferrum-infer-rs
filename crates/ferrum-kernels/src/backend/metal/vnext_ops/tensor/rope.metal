struct RopeParams { ulong pairs; ulong grid_width; ulong columns; ulong sequence; ulong head_dim; };

kernel void ferrum_tensor_adjacent_rope_bf16(
    device const ushort* input [[buffer(0)]],
    device const float* frequencies [[buffer(1)]],
    device ushort* output [[buffer(2)]],
    constant RopeParams& p [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
    for (ulong pair = tid; pair < p.pairs; pair += p.grid_width) {
        ulong row = pair / (p.columns / 2);
        ulong channel = (pair % (p.columns / 2)) % (p.head_dim / 2);
        ulong frequency = (row % p.sequence) * p.head_dim + 2 * channel;
        float a = decode_bf16(input[2 * pair]);
        float b = decode_bf16(input[2 * pair + 1]);
        float c = frequencies[frequency], s = frequencies[frequency + 1];
        float bs = b * s, as = a * s;
        output[2 * pair] = encode_bf16(fma(a, c, -bs));
        output[2 * pair + 1] = encode_bf16(fma(b, c, as));
    }
}

kernel void ferrum_tensor_half_split_rope_bf16(
    device const ushort* input [[buffer(0)]],
    device const ushort* cosine [[buffer(1)]],
    device const ushort* sine [[buffer(2)]],
    device ushort* output [[buffer(3)]],
    constant RopeParams& p [[buffer(4)]],
    uint tid [[thread_position_in_grid]]) {
    for (ulong pair = tid; pair < p.pairs; pair += p.grid_width) {
        ulong row = pair / (p.columns / 2);
        ulong row_pair = pair % (p.columns / 2);
        ulong half_width = p.head_dim / 2;
        ulong channel = row_pair % half_width;
        ulong left = row * p.columns + (row_pair / half_width) * p.head_dim + channel;
        ulong right = left + half_width;
        ulong frequency = row * p.head_dim + channel;
        float a = decode_bf16(input[left]), b = decode_bf16(input[right]);
        float ac = rounded_bf16(a * decode_bf16(cosine[frequency]));
        float bs = rounded_bf16(-b * decode_bf16(sine[frequency]));
        float bc = rounded_bf16(b * decode_bf16(cosine[frequency + half_width]));
        float as = rounded_bf16(a * decode_bf16(sine[frequency + half_width]));
        output[left] = encode_bf16(ac + bs);
        output[right] = encode_bf16(bc + as);
    }
}
