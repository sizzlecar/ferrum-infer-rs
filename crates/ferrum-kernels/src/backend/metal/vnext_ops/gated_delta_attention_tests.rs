use std::ffi::c_void;
use std::ops::Range;

use half::f16;
use metal::{BufferRef, CommandQueueRef, MTLCommandBufferStatus, MTLResourceOptions};

use super::*;

const TOKENS: usize = 4;
const KEY_HEADS: usize = 16;
const VALUE_HEADS: usize = 32;
const KEY_DIM: usize = 128;
const VALUE_DIM: usize = 128;
const QK_FEATURES: usize = KEY_HEADS * KEY_DIM;
const VALUE_FEATURES: usize = VALUE_HEADS * VALUE_DIM;
const QKV_FEATURES: usize = 2 * QK_FEATURES + VALUE_FEATURES;
const CONV_KERNEL: usize = 4;
const CONV_STATE_WIDTH: usize = CONV_KERNEL - 1;

struct SegmentInputs<'a> {
    mixed_qkv: &'a [f16],
    a_raw: &'a [f16],
    b_raw: &'a [f16],
    z: &'a [f16],
}

struct StaticWeights<'a> {
    conv: &'a BufferRef,
    a_log: &'a BufferRef,
    dt_bias: &'a BufferRef,
    norm: &'a BufferRef,
}

#[test]
fn recurrent_core_matches_cpu_and_preserves_split_decode_state_on_real_metal() {
    let Some(device) = Device::system_default() else {
        eprintln!("no Metal device; skipping gated-delta conformance");
        return;
    };
    let pipelines = MetalGatedDeltaPipelines::new(&device).unwrap();
    let queue = device.new_command_queue();

    let mixed_qkv = half_values(TOKENS * QKV_FEATURES, 0.017, 0.21);
    let a_raw = half_values(TOKENS * VALUE_HEADS, 0.071, 0.18);
    let b_raw = half_values(TOKENS * VALUE_HEADS, 0.053, 0.24);
    let z = half_values(TOKENS * VALUE_FEATURES, 0.029, 0.31);
    let conv_weight = half_values(QKV_FEATURES * CONV_KERNEL, 0.011, 0.16);
    let a_log = (0..VALUE_HEADS)
        .map(|index| -1.7 + index as f32 * 0.07)
        .collect::<Vec<_>>();
    let dt_bias = (0..VALUE_HEADS)
        .map(|index| -0.25 + index as f32 * 0.04)
        .collect::<Vec<_>>();
    let norm = (0..VALUE_DIM)
        .map(|index| 0.82 + index as f32 * 0.013)
        .collect::<Vec<_>>();
    let initial_conv = half_values(QKV_FEATURES * CONV_STATE_WIDTH, 0.023, 0.04);
    let initial_delta = float_values(VALUE_HEADS * VALUE_DIM * KEY_DIM, 0.007, 0.025);

    let conv_weight_buffer = shared_buffer(&device, &conv_weight);
    let a_log_buffer = shared_buffer(&device, &a_log);
    let dt_bias_buffer = shared_buffer(&device, &dt_bias);
    let norm_buffer = shared_buffer(&device, &norm);
    let weights = StaticWeights {
        conv: &conv_weight_buffer,
        a_log: &a_log_buffer,
        dt_bias: &dt_bias_buffer,
        norm: &norm_buffer,
    };

    let full_conv_state = shared_buffer(&device, &initial_conv);
    let full_delta_state = shared_buffer(&device, &initial_delta);
    let full_output = run_segment(
        &device,
        &queue,
        &pipelines,
        SegmentInputs {
            mixed_qkv: &mixed_qkv,
            a_raw: &a_raw,
            b_raw: &b_raw,
            z: &z,
        },
        &weights,
        &full_conv_state,
        &full_delta_state,
    );

    let split_conv_state = shared_buffer(&device, &initial_conv);
    let split_delta_state = shared_buffer(&device, &initial_delta);
    let split_at = 3;
    let mut split_output = run_segment(
        &device,
        &queue,
        &pipelines,
        segment(&mixed_qkv, &a_raw, &b_raw, &z, 0..split_at),
        &weights,
        &split_conv_state,
        &split_delta_state,
    );
    split_output.extend(run_segment(
        &device,
        &queue,
        &pipelines,
        segment(&mixed_qkv, &a_raw, &b_raw, &z, split_at..TOKENS),
        &weights,
        &split_conv_state,
        &split_delta_state,
    ));

    let mut cpu_conv_state = initial_conv.clone();
    let mut cpu_delta_state = initial_delta.clone();
    let cpu_output = cpu_segment(
        SegmentInputs {
            mixed_qkv: &mixed_qkv,
            a_raw: &a_raw,
            b_raw: &b_raw,
            z: &z,
        },
        &conv_weight,
        &a_log,
        &dt_bias,
        &norm,
        &mut cpu_conv_state,
        &mut cpu_delta_state,
    );

    assert!(full_output.iter().any(|value| value.abs() > 1.0e-4));
    assert_close("full/cpu output", &full_output, &cpu_output, 0.012);
    assert_close("full/split output", &full_output, &split_output, 0.001);
    assert_close(
        "full/cpu conv state",
        &read_f16(&full_conv_state, initial_conv.len()),
        &as_f32(&cpu_conv_state),
        0.001,
    );
    assert_close(
        "full/cpu delta state",
        &read_f32(&full_delta_state, initial_delta.len()),
        &cpu_delta_state,
        0.001,
    );
    assert_close(
        "full/split conv state",
        &read_f16(&full_conv_state, initial_conv.len()),
        &read_f16(&split_conv_state, initial_conv.len()),
        0.001,
    );
    assert_close(
        "full/split delta state",
        &read_f32(&full_delta_state, initial_delta.len()),
        &read_f32(&split_delta_state, initial_delta.len()),
        0.001,
    );
}

#[test]
fn launch_extent_validation_rejects_msl_uint_overflow() {
    let shape = AttentionShape {
        hidden_size: 16,
        key_heads: 2,
        value_heads: 8,
        key_dim: 16,
        value_dim: 16,
        qkv_features: QKV_FEATURES as u64,
        value_features: VALUE_FEATURES as u64,
        conv_kernel: CONV_KERNEL as u64,
        conv_state_width: CONV_STATE_WIDTH as u64,
        epsilon: 1.0e-6,
        layer_index: 0,
    };
    let tokens = u64::from(u32::MAX) / shape.qkv_features + 1;
    assert!(shape
        .validate_launch_extents(tokens)
        .unwrap_err()
        .contains("QKV activation elements"));
}

fn segment<'a>(
    mixed_qkv: &'a [f16],
    a_raw: &'a [f16],
    b_raw: &'a [f16],
    z: &'a [f16],
    range: Range<usize>,
) -> SegmentInputs<'a> {
    SegmentInputs {
        mixed_qkv: row_slice(mixed_qkv, QKV_FEATURES, range.clone()),
        a_raw: row_slice(a_raw, VALUE_HEADS, range.clone()),
        b_raw: row_slice(b_raw, VALUE_HEADS, range.clone()),
        z: row_slice(z, VALUE_FEATURES, range),
    }
}

fn row_slice<T>(values: &[T], width: usize, range: Range<usize>) -> &[T] {
    &values[range.start * width..range.end * width]
}

#[allow(clippy::too_many_arguments)]
fn run_segment(
    device: &Device,
    queue: &CommandQueueRef,
    pipelines: &MetalGatedDeltaPipelines,
    inputs: SegmentInputs<'_>,
    weights: &StaticWeights<'_>,
    conv_state: &BufferRef,
    delta_state: &BufferRef,
) -> Vec<f32> {
    let tokens = inputs.mixed_qkv.len() / QKV_FEATURES;
    let params = test_params(tokens);
    let mixed_qkv = shared_buffer(device, inputs.mixed_qkv);
    let a_raw = shared_buffer(device, inputs.a_raw);
    let b_raw = shared_buffer(device, inputs.b_raw);
    let z = shared_buffer(device, inputs.z);
    let query = output_buffer::<f32>(device, tokens * QK_FEATURES);
    let key = output_buffer::<f32>(device, tokens * QK_FEATURES);
    let value = output_buffer::<f32>(device, tokens * VALUE_FEATURES);
    let g = output_buffer::<f32>(device, tokens * VALUE_HEADS);
    let beta = output_buffer::<f32>(device, tokens * VALUE_HEADS);
    let core = output_buffer::<f32>(device, tokens * VALUE_FEATURES);
    let output = output_buffer::<f16>(device, tokens * VALUE_FEATURES);
    let next_conv = output_buffer::<f16>(device, QKV_FEATURES * CONV_STATE_WIDTH);

    let command = queue.new_command_buffer();
    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipelines.prepare_conv);
    for (index, buffer) in [
        &*mixed_qkv,
        weights.conv,
        conv_state,
        &*query,
        &*key,
        &*value,
    ]
    .into_iter()
    .enumerate()
    {
        set_raw(encoder, index as u64, buffer);
    }
    set_params(encoder, 6, &params);
    dispatch_elements(encoder, (tokens * QKV_FEATURES) as u64);

    encoder.set_compute_pipeline_state(&pipelines.prepare_gates);
    for (index, buffer) in [
        &*a_raw,
        &*b_raw,
        weights.a_log,
        weights.dt_bias,
        &*g,
        &*beta,
    ]
    .into_iter()
    .enumerate()
    {
        set_raw(encoder, index as u64, buffer);
    }
    set_params(encoder, 6, &params);
    dispatch_elements(encoder, (tokens * VALUE_HEADS) as u64);

    encoder.set_compute_pipeline_state(&pipelines.collect_conv_state);
    set_raw(encoder, 0, &mixed_qkv);
    set_raw(encoder, 1, conv_state);
    set_raw(encoder, 2, &next_conv);
    set_params(encoder, 3, &params);
    dispatch_elements(encoder, (QKV_FEATURES * CONV_STATE_WIDTH) as u64);

    encoder.set_compute_pipeline_state(&pipelines.copy_f16);
    set_raw(encoder, 0, &next_conv);
    set_raw(encoder, 1, conv_state);
    let conv_elements = (QKV_FEATURES * CONV_STATE_WIDTH) as u32;
    encoder.set_bytes(
        2,
        std::mem::size_of::<u32>() as u64,
        &conv_elements as *const _ as *const c_void,
    );
    dispatch_elements(encoder, u64::from(conv_elements));

    encoder.set_compute_pipeline_state(&pipelines.qk_norm);
    set_raw(encoder, 0, &query);
    set_raw(encoder, 1, &key);
    set_params(encoder, 2, &params);
    encoder.set_threadgroup_memory_length(0, 16 * std::mem::size_of::<f32>() as u64);
    encoder.dispatch_thread_groups(
        MTLSize::new((tokens * KEY_HEADS) as u64, 1, 1),
        MTLSize::new(THREADS_PER_GROUP, 1, 1),
    );

    encoder.set_compute_pipeline_state(&pipelines.delta);
    for (index, buffer) in [&*query, &*key, &*value, &*g, &*beta, delta_state, &*core]
        .into_iter()
        .enumerate()
    {
        set_raw(encoder, index as u64, buffer);
    }
    set_params(encoder, 7, &params);
    encoder.dispatch_thread_groups(
        MTLSize::new(
            (VALUE_DIM as u64).div_ceil(VALUE_TILE),
            VALUE_HEADS as u64,
            1,
        ),
        MTLSize::new(THREADS_PER_GROUP, 1, 1),
    );

    encoder.set_compute_pipeline_state(&pipelines.gated_norm);
    for (index, buffer) in [&*core, &*z, weights.norm, &*output]
        .into_iter()
        .enumerate()
    {
        set_raw(encoder, index as u64, buffer);
    }
    set_params(encoder, 4, &params);
    encoder.set_threadgroup_memory_length(0, 8 * std::mem::size_of::<f32>() as u64);
    encoder.dispatch_thread_groups(
        MTLSize::new((tokens * VALUE_HEADS) as u64, 1, 1),
        MTLSize::new(THREADS_PER_GROUP, 1, 1),
    );

    encoder.end_encoding();
    command.commit();
    command.wait_until_completed();
    assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
    read_f16(&output, tokens * VALUE_FEATURES)
}

#[allow(clippy::too_many_arguments)]
fn cpu_segment(
    inputs: SegmentInputs<'_>,
    conv_weight: &[f16],
    a_log: &[f32],
    dt_bias: &[f32],
    norm: &[f32],
    conv_state: &mut [f16],
    delta_state: &mut [f32],
) -> Vec<f32> {
    let tokens = inputs.mixed_qkv.len() / QKV_FEATURES;
    let previous_conv = conv_state.to_vec();
    let mut query = vec![0.0_f32; tokens * QK_FEATURES];
    let mut key = vec![0.0_f32; tokens * QK_FEATURES];
    let mut value = vec![0.0_f32; tokens * VALUE_FEATURES];
    for token in 0..tokens {
        for channel in 0..QKV_FEATURES {
            let mut sum = 0.0_f32;
            for kernel in 0..CONV_KERNEL {
                let source = token as isize + kernel as isize - CONV_STATE_WIDTH as isize;
                let activation = if source >= 0 {
                    inputs.mixed_qkv[source as usize * QKV_FEATURES + channel].to_f32()
                } else {
                    previous_conv
                        [channel * CONV_STATE_WIDTH + (CONV_STATE_WIDTH as isize + source) as usize]
                        .to_f32()
                };
                sum += activation * conv_weight[channel * CONV_KERNEL + kernel].to_f32();
            }
            let mixed = silu(sum);
            if channel < QK_FEATURES {
                query[token * QK_FEATURES + channel] = mixed;
            } else if channel < 2 * QK_FEATURES {
                key[token * QK_FEATURES + channel - QK_FEATURES] = mixed;
            } else {
                value[token * VALUE_FEATURES + channel - 2 * QK_FEATURES] = mixed;
            }
        }
    }
    for channel in 0..QKV_FEATURES {
        for position in 0..CONV_STATE_WIDTH {
            let source = tokens as isize + position as isize - CONV_STATE_WIDTH as isize;
            conv_state[channel * CONV_STATE_WIDTH + position] = if source >= 0 {
                inputs.mixed_qkv[source as usize * QKV_FEATURES + channel]
            } else {
                previous_conv
                    [channel * CONV_STATE_WIDTH + (CONV_STATE_WIDTH as isize + source) as usize]
            };
        }
    }

    for token in 0..tokens {
        for head in 0..KEY_HEADS {
            let base = (token * KEY_HEADS + head) * KEY_DIM;
            let query_inverse = (query[base..base + KEY_DIM]
                .iter()
                .map(|value| value * value)
                .sum::<f32>()
                + 1.0e-6)
                .sqrt()
                .recip();
            let key_inverse = (key[base..base + KEY_DIM]
                .iter()
                .map(|value| value * value)
                .sum::<f32>()
                + 1.0e-6)
                .sqrt()
                .recip();
            for column in 0..KEY_DIM {
                query[base + column] *= query_inverse;
                key[base + column] *= key_inverse;
            }
        }
    }

    let mut core = vec![0.0_f32; tokens * VALUE_FEATURES];
    let repeat = VALUE_HEADS / KEY_HEADS;
    let scale = (KEY_DIM as f32).sqrt().recip();
    for token in 0..tokens {
        for value_head in 0..VALUE_HEADS {
            let key_head = value_head / repeat;
            let gate_index = token * VALUE_HEADS + value_head;
            let g = -a_log[value_head].exp()
                * softplus(inputs.a_raw[gate_index].to_f32() + dt_bias[value_head]);
            let beta = sigmoid(inputs.b_raw[gate_index].to_f32());
            let qk_base = (token * KEY_HEADS + key_head) * KEY_DIM;
            for value_column in 0..VALUE_DIM {
                let state_base = (value_head * VALUE_DIM + value_column) * KEY_DIM;
                let mut prediction = 0.0_f32;
                for key_column in 0..KEY_DIM {
                    let state_index = state_base + key_column;
                    let decayed = delta_state[state_index] * g.exp();
                    delta_state[state_index] = decayed;
                    prediction += decayed * key[qk_base + key_column];
                }
                let value_index = (token * VALUE_HEADS + value_head) * VALUE_DIM + value_column;
                let delta = (value[value_index] - prediction) * beta;
                let mut result = 0.0_f32;
                for key_column in 0..KEY_DIM {
                    let state_index = state_base + key_column;
                    let updated = delta_state[state_index] + delta * key[qk_base + key_column];
                    delta_state[state_index] = updated;
                    result += updated * query[qk_base + key_column] * scale;
                }
                core[value_index] = result;
            }
        }
    }

    let mut output = vec![0.0_f32; core.len()];
    for row in 0..tokens * VALUE_HEADS {
        let base = row * VALUE_DIM;
        let inverse = (core[base..base + VALUE_DIM]
            .iter()
            .map(|value| value * value)
            .sum::<f32>()
            / VALUE_DIM as f32
            + 1.0e-6)
            .sqrt()
            .recip();
        for column in 0..VALUE_DIM {
            output[base + column] = f16::from_f32(
                core[base + column]
                    * inverse
                    * norm[column]
                    * silu(inputs.z[base + column].to_f32()),
            )
            .to_f32();
        }
    }
    output
}

fn test_params(tokens: usize) -> GatedDeltaParams {
    GatedDeltaParams {
        tokens: tokens as u32,
        hidden_size: 16,
        key_heads: KEY_HEADS as u32,
        value_heads: VALUE_HEADS as u32,
        key_dim: KEY_DIM as u32,
        value_dim: VALUE_DIM as u32,
        qkv_features: QKV_FEATURES as u32,
        value_features: VALUE_FEATURES as u32,
        conv_kernel: CONV_KERNEL as u32,
        epsilon: 1.0e-6,
        scale: (KEY_DIM as f32).sqrt().recip(),
    }
}

fn half_values(elements: usize, frequency: f32, amplitude: f32) -> Vec<f16> {
    (0..elements)
        .map(|index| f16::from_f32((index as f32 * frequency).sin() * amplitude))
        .collect()
}

fn float_values(elements: usize, frequency: f32, amplitude: f32) -> Vec<f32> {
    (0..elements)
        .map(|index| (index as f32 * frequency).sin() * amplitude)
        .collect()
}

fn shared_buffer<T>(device: &Device, values: &[T]) -> metal::Buffer {
    device.new_buffer_with_data(
        values.as_ptr() as *const c_void,
        std::mem::size_of_val(values) as u64,
        MTLResourceOptions::StorageModeShared,
    )
}

fn output_buffer<T>(device: &Device, elements: usize) -> metal::Buffer {
    device.new_buffer(
        (elements * std::mem::size_of::<T>()) as u64,
        MTLResourceOptions::StorageModeShared,
    )
}

fn set_raw(encoder: &ComputeCommandEncoderRef, index: u64, buffer: &BufferRef) {
    encoder.set_buffer(index, Some(buffer), 0);
}

fn read_f16(buffer: &BufferRef, elements: usize) -> Vec<f32> {
    let values: &[f16] =
        unsafe { std::slice::from_raw_parts(buffer.contents() as *const f16, elements) };
    as_f32(values)
}

fn read_f32(buffer: &BufferRef, elements: usize) -> Vec<f32> {
    let values: &[f32] =
        unsafe { std::slice::from_raw_parts(buffer.contents() as *const f32, elements) };
    values.to_vec()
}

fn as_f32(values: &[f16]) -> Vec<f32> {
    values.iter().map(|value| value.to_f32()).collect()
}

fn assert_close(name: &str, actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len(), "{name} length");
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= tolerance,
            "{name}[{index}] {actual} != {expected} (tol={tolerance})"
        );
    }
}

fn sigmoid(value: f32) -> f32 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exp = value.exp();
        exp / (1.0 + exp)
    }
}

fn silu(value: f32) -> f32 {
    value * sigmoid(value)
}

fn softplus(value: f32) -> f32 {
    if value > 20.0 {
        value
    } else if value < -20.0 {
        value.exp()
    } else {
        (1.0 + value.exp()).ln()
    }
}
