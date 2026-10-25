struct UpsampleParams {
    ulong elements, grid_threads, input_channels, input_height, input_width;
    ulong output_channels, output_height, output_width, scale_h, scale_w, channel_stride, offsets[64];
};
kernel void ferrum_tensor_spatial_upsample_bf16(
    device const ushort* input [[buffer(0)]], device ushort* output [[buffer(1)]],
    constant UpsampleParams& p [[buffer(2)]], uint tid [[thread_position_in_grid]]) {
    for (ulong i = tid; i < p.elements; i += p.grid_threads) {
        ulong x = i % p.output_width, y = (i / p.output_width) % p.output_height;
        ulong plane = i / (p.output_width * p.output_height);
        ulong batch = plane / p.output_channels, channel = plane % p.output_channels;
        ulong source_channel = channel * p.channel_stride + p.offsets[(y % p.scale_h) * p.scale_w + x % p.scale_w];
        output[i] = input[((batch * p.input_channels + source_channel) * p.input_height + y / p.scale_h) * p.input_width + x / p.scale_w];
    }
}
