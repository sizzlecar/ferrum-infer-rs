struct GroupMeanParams {
    ulong elements, grid_threads, input_channels, input_height, input_width;
    ulong output_channels, output_height, output_width, scale_h, scale_w, frames, group_size;
};
inline float spatial_group_value(device const ushort* input, constant GroupMeanParams& p, ulong batch, ulong y, ulong x, ulong channel_index) {
    ulong tile = p.scale_h * p.scale_w;
    ulong frame = (channel_index / tile) % p.frames;
    if (frame + 1 != p.frames) return 0.0f;
    ulong channel = channel_index / (tile * p.frames);
    ulong dy = (channel_index % tile) / p.scale_w, dx = channel_index % p.scale_w;
    return decode_bf16(input[((batch * p.input_channels + channel) * p.input_height + y * p.scale_h + dy) * p.input_width + x * p.scale_w + dx]);
}
kernel void ferrum_tensor_spatial_group_mean_bf16(
    device const ushort* input [[buffer(0)]], device ushort* output [[buffer(1)]],
    constant GroupMeanParams& p [[buffer(2)]], uint tid [[thread_position_in_grid]]) {
    for (ulong i = tid; i < p.elements; i += p.grid_threads) {
        ulong x = i % p.output_width, y = (i / p.output_width) % p.output_height;
        ulong plane = i / (p.output_width * p.output_height), batch = plane / p.output_channels;
        ulong first = (plane % p.output_channels) * p.group_size;
        float sum = spatial_group_value(input,p,batch,y,x,first);
        for (ulong j = 1; j < p.group_size; ++j) sum = sum + spatial_group_value(input,p,batch,y,x,first+j);
        output[i] = encode_bf16(sum / float(p.group_size));
    }
}
