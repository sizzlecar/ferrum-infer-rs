#include <metal_simdgroup_matrix>
constant uint TENSOR_CONV_KERNEL [[function_constant(102)]];
constant bool TENSOR_CONV_MASKED [[function_constant(103)]];
constant bool direct_square = is_function_constant_defined(TENSOR_CONV_KERNEL);
constant uint direct_kernel = direct_square ? TENSOR_CONV_KERNEL : 1;
constant bool masked_3x3 = is_function_constant_defined(TENSOR_CONV_MASKED) ? TENSOR_CONV_MASKED : false;
struct PadParams {
    ulong elements, grid_threads, channels, height, width, padded_height, padded_width, top, left;
};
kernel void ferrum_tensor_pad_nchw_bf16(
    device const ushort* input [[buffer(0)]], device ushort* output [[buffer(1)]],
    constant PadParams& p [[buffer(2)]], uint tid [[thread_position_in_grid]]) {
    for (ulong i = tid; i < p.elements; i += p.grid_threads) {
        ulong x = i % p.padded_width, y = (i / p.padded_width) % p.padded_height;
        ulong plane = i / (p.padded_width * p.padded_height);
        output[i] = y >= p.top && y - p.top < p.height && x >= p.left && x - p.left < p.width
            ? input[(plane * p.height + y - p.top) * p.width + x - p.left] : ushort(0);
    }
}
struct ConvParams {
    ulong jobs, grid_groups, spatial_tiles, output_tiles, groups;
    ulong input_channels, input_height, input_width, output_channels, output_height, output_width;
    ulong kernel_height, kernel_width, stride_h, stride_w, dilation_h, dilation_w, pad_top, pad_left;
};
// 32 spatial positions x 64 output channels; im2col exists only on chip.
// External +0 padding is loaded here or materialized by a prior managed stage.
kernel void ferrum_tensor_conv2d_bf16(
    device const ushort* input [[buffer(0)]], device const ushort* weight [[buffer(1)]],
    device const ushort* bias [[buffer(2)]], device ushort* output [[buffer(3)]],
    constant ConvParams& p [[buffer(4)]], uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]], uint sid [[simdgroup_index_in_threadgroup]]) {
    threadgroup float workspace[3072];
    threadgroup ulong source_origins[32];
    threadgroup ulong2 padded_origins[32];
    threadgroup ushort valid_taps[32];
    threadgroup float* a = workspace;
    threadgroup float* b = workspace + 1024;
    ulong spatial = p.output_height * p.output_width;
    ulong input_spatial = p.input_height * p.input_width;
    ulong ic = p.input_channels / p.groups, oc = p.output_channels / p.groups;
    ulong kernel_area = direct_square ? direct_kernel * direct_kernel
        : (masked_3x3 ? 9 : p.kernel_height * p.kernel_width);
    ulong inner_columns = ic * kernel_area;
    uint local_row = (sid / 2) * 16, local_col = (sid % 2) * 32;
    for (ulong job = group; job < p.jobs; job += p.grid_groups) {
        ulong row_start = (job % p.spatial_tiles) * 32;
        ulong col_start = ((job / p.spatial_tiles) % p.output_tiles) * 64;
        ulong batch_group = job / (p.spatial_tiles * p.output_tiles);
        ulong batch = batch_group / p.groups, channel_group = batch_group % p.groups;
        // Spatial division is invariant across all channel/kernel tiles. Cache
        // it once on chip; the admitted input makes every selected tap valid.
        if (direct_square) {
            if (tid < 32) {
                ulong position = row_start + tid;
                source_origins[tid] = (position / p.output_width) * p.input_width
                    + position % p.output_width;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        } else if (masked_3x3) {
            // Keep coordinates before subtracting padding: an origin above or
            // left of the input is never represented by unsigned wraparound.
            if (tid < 32) {
                ulong position = row_start + tid;
                ulong2 origin = ulong2(0);
                ushort mask = 0;
                if (position < spatial) {
                    origin = ulong2(position / p.output_width, position % p.output_width);
                    for (uint tap = 0; tap < 9; ++tap) {
                        ulong y = origin.x + tap / 3, x = origin.y + tap % 3;
                        if (y >= p.pad_top && y - p.pad_top < p.input_height
                            && x >= p.pad_left && x - p.pad_left < p.input_width) {
                            mask |= ushort(1u << tap);
                        }
                    }
                }
                padded_origins[tid] = origin;
                valid_taps[tid] = mask;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        simdgroup_float8x8 sums[8];
        for (uint i = 0; i < 8; ++i) sums[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
        for (ulong k = 0; k < inner_columns; k += 32) {
            if (direct_square) {
                ulong inner = k + tid % 32;
                ulong channel = channel_group * ic + inner / kernel_area;
                ulong tap = (inner % kernel_area) / direct_kernel * p.input_width + inner % direct_kernel;
                ulong base = (batch * p.input_channels + channel) * input_spatial + tap;
                for (uint i = tid; i < 1024; i += 128) {
                    ulong position = row_start + i / 32;
                    a[i] = position < spatial && inner < inner_columns
                        ? decode_bf16(input[base + source_origins[i / 32]]) : 0.0f;
                }
            } else if (masked_3x3) {
                ulong inner = k + tid % 32;
                uint tap = uint(inner % 9);
                for (uint i = tid; i < 1024; i += 128) {
                    float x = 0.0f;
                    if (inner < inner_columns && (valid_taps[i / 32] & ushort(1u << tap))) {
                        ulong2 origin = padded_origins[i / 32];
                        ulong y = origin.x + tap / 3 - p.pad_top;
                        ulong xcol = origin.y + tap % 3 - p.pad_left;
                        ulong channel = channel_group * ic + inner / 9;
                        x = decode_bf16(input[(batch * p.input_channels + channel) * input_spatial
                            + y * p.input_width + xcol]);
                    }
                    // Invalid taps still multiply +0 by the original weight;
                    // skipping that arithmetic would change NaN/Inf behavior.
                    a[i] = x;
                }
            } else {
                for (uint i = tid; i < 1024; i += 128) {
                    ulong position = row_start + i / 32, inner = k + i % 32;
                    float x = 0.0f;
                    if (position < spatial && inner < inner_columns) {
                        ulong ky = (inner % kernel_area) / p.kernel_width, kx = inner % p.kernel_width;
                        ulong y = (position / p.output_width) * p.stride_h + ky * p.dilation_h;
                        ulong xcol = (position % p.output_width) * p.stride_w + kx * p.dilation_w;
                        if (y >= p.pad_top && y - p.pad_top < p.input_height && xcol >= p.pad_left && xcol - p.pad_left < p.input_width) {
                            ulong channel = channel_group * ic + inner / kernel_area;
                            x = decode_bf16(input[(batch * p.input_channels + channel) * input_spatial
                                + (y - p.pad_top) * p.input_width + xcol - p.pad_left]);
                        }
                    }
                    a[i] = x;
                }
            }
            for (uint i = tid; i < 2048; i += 128) {
                ulong channel = col_start + i % 64, inner = k + i / 64;
                b[i] = channel < oc && inner < inner_columns
                    ? decode_bf16(weight[(channel_group * oc + channel) * inner_columns + inner]) : 0.0f;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint chunk = 0; chunk < 32; chunk += 8) {
                simdgroup_float8x8 av[2], bv[4];
                for (uint m = 0; m < 2; ++m) simdgroup_load(av[m], a + (local_row + m * 8) * 32 + chunk, 32, 0, false);
                for (uint n = 0; n < 4; ++n) simdgroup_load(bv[n], b + chunk * 64 + local_col + n * 8, 64, 0, false);
                for (uint m = 0; m < 2; ++m) for (uint n = 0; n < 4; ++n)
                    simdgroup_multiply_accumulate(sums[m * 4 + n], av[m], bv[n], sums[m * 4 + n]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        for (uint m = 0; m < 2; ++m) for (uint n = 0; n < 4; ++n)
            simdgroup_store(sums[m * 4 + n], b + (local_row + m * 8) * 64 + local_col + n * 8, 64, 0, false);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < 2048; i += 128) {
            ulong position = row_start + i / 64, channel = col_start + i % 64;
            if (position < spatial && channel < oc) {
                ulong output_channel = channel_group * oc + channel;
                output[(batch * p.output_channels + output_channel) * spatial + position]
                    = encode_bf16(rounded_bf16(b[i]) + decode_bf16(bias[output_channel]));
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
