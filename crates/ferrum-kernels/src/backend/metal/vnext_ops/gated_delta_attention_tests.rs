use std::ffi::c_void;
use std::ops::Range;

use half::f16;
use metal::{BufferRef, CommandQueueRef, MTLCommandBufferStatus, MTLResourceOptions};

use super::super::numerical_tolerance;
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
// This stricter diagnostic never substitutes for a catalog-bound release comparison.
const SPLIT_CONTINUITY_DIAGNOSTIC_MAX_ABS: f32 = 0.001;
// These core-algorithm diagnostics are not provider-conformance or release tolerances.
const CHUNK_CORE_ORACLE_DIAGNOSTIC_MAX_ABS: f32 = 0.012;
const CHUNK_STATE_ORACLE_DIAGNOSTIC_MAX_ABS: f32 = 0.001;

struct SegmentInputs<'a> {
    mixed_qkv: &'a [f16],
    a_raw: &'a [f16],
    b_raw: &'a [f16],
    z: &'a [f16],
}

struct StaticWeights<'a> {
    conv: &'a BufferRef,
    decay_parameter: &'a BufferRef,
    dt_bias: &'a BufferRef,
    norm: &'a BufferRef,
}

#[derive(Debug, Clone, Copy)]
struct TestSemantics {
    decay_parameterization: GatedDeltaDecayParameterization,
    value_head_mapping: GatedDeltaValueHeadMapping,
}

#[test]
fn recurrent_core_matches_cpu_and_preserves_split_decode_state_on_real_metal() {
    for semantics in [
        TestSemantics {
            decay_parameterization: GatedDeltaDecayParameterization::LogRate,
            value_head_mapping: GatedDeltaValueHeadMapping::GroupedByKeyHead,
        },
        TestSemantics {
            decay_parameterization: GatedDeltaDecayParameterization::NegativeRate,
            value_head_mapping: GatedDeltaValueHeadMapping::InterleavedByKeyHead,
        },
    ] {
        assert_recurrent_conformance(semantics);
    }
}

fn assert_recurrent_conformance(semantics: TestSemantics) {
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
    let log_rates = (0..VALUE_HEADS)
        .map(|index| -1.7 + index as f32 * 0.07)
        .collect::<Vec<_>>();
    let decay_parameters = match semantics.decay_parameterization {
        GatedDeltaDecayParameterization::LogRate => log_rates,
        GatedDeltaDecayParameterization::NegativeRate => {
            log_rates.into_iter().map(|value| -value.exp()).collect()
        }
    };
    let dt_bias = (0..VALUE_HEADS)
        .map(|index| -0.25 + index as f32 * 0.04)
        .collect::<Vec<_>>();
    let norm = (0..VALUE_DIM)
        .map(|index| 0.82 + index as f32 * 0.013)
        .collect::<Vec<_>>();
    let initial_conv = half_values(QKV_FEATURES * CONV_STATE_WIDTH, 0.023, 0.04);
    let initial_delta = float_values(VALUE_HEADS * VALUE_DIM * KEY_DIM, 0.007, 0.025);

    let conv_weight_buffer = shared_buffer(&device, &conv_weight);
    let decay_parameter_buffer = shared_buffer(&device, &decay_parameters);
    let dt_bias_buffer = shared_buffer(&device, &dt_bias);
    let norm_buffer = shared_buffer(&device, &norm);
    let weights = StaticWeights {
        conv: &conv_weight_buffer,
        decay_parameter: &decay_parameter_buffer,
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
        semantics,
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
        semantics,
    );
    split_output.extend(run_segment(
        &device,
        &queue,
        &pipelines,
        segment(&mixed_qkv, &a_raw, &b_raw, &z, split_at..TOKENS),
        &weights,
        &split_conv_state,
        &split_delta_state,
        semantics,
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
        &decay_parameters,
        &dt_bias,
        &norm,
        &mut cpu_conv_state,
        &mut cpu_delta_state,
        semantics,
    );
    assert!(full_output.iter().any(|value| value.abs() > 1.0e-4));
    let (output_tolerance_id, output_tolerance_fingerprint) = cpu_output_tolerance(semantics);
    numerical_tolerance::assert_matches(
        "full/cpu output",
        &full_output,
        &[TOKENS, VALUE_HEADS, VALUE_DIM],
        &cpu_output,
        &[TOKENS, VALUE_HEADS, VALUE_DIM],
        numerical_tolerance::LogicalDtype::Fp16,
        output_tolerance_id,
        output_tolerance_fingerprint,
    )
    .expect("reviewed gated-delta output numerical contract");
    assert_close(
        "full/split output",
        &full_output,
        &split_output,
        SPLIT_CONTINUITY_DIAGNOSTIC_MAX_ABS,
    );
    let full_conv_state_values = read_f16(&full_conv_state, initial_conv.len());
    let cpu_conv_state_values = as_f32(&cpu_conv_state);
    let (conv_tolerance_id, conv_tolerance_fingerprint) = cpu_conv_state_tolerance(semantics);
    numerical_tolerance::assert_matches(
        "full/cpu conv state",
        &full_conv_state_values,
        &[QKV_FEATURES, CONV_STATE_WIDTH],
        &cpu_conv_state_values,
        &[QKV_FEATURES, CONV_STATE_WIDTH],
        numerical_tolerance::LogicalDtype::Fp16,
        conv_tolerance_id,
        conv_tolerance_fingerprint,
    )
    .expect("reviewed gated-delta conv-state numerical contract");
    let full_delta_state_values = read_f32(&full_delta_state, initial_delta.len());
    let (delta_tolerance_id, delta_tolerance_fingerprint) = cpu_delta_state_tolerance(semantics);
    numerical_tolerance::assert_matches(
        "full/cpu delta state",
        &full_delta_state_values,
        &[VALUE_HEADS, VALUE_DIM, KEY_DIM],
        &cpu_delta_state,
        &[VALUE_HEADS, VALUE_DIM, KEY_DIM],
        numerical_tolerance::LogicalDtype::Fp32,
        delta_tolerance_id,
        delta_tolerance_fingerprint,
    )
    .expect("reviewed gated-delta delta-state numerical contract");
    assert_close(
        "full/split conv state",
        &full_conv_state_values,
        &read_f16(&split_conv_state, initial_conv.len()),
        SPLIT_CONTINUITY_DIAGNOSTIC_MAX_ABS,
    );
    assert_close(
        "full/split delta state",
        &full_delta_state_values,
        &read_f32(&split_delta_state, initial_delta.len()),
        SPLIT_CONTINUITY_DIAGNOSTIC_MAX_ABS,
    );
}

fn cpu_output_tolerance(semantics: TestSemantics) -> (&'static str, &'static str) {
    match (
        semantics.decay_parameterization,
        semantics.value_head_mapping,
    ) {
        (
            GatedDeltaDecayParameterization::LogRate,
            GatedDeltaValueHeadMapping::GroupedByKeyHead,
        ) => (
            "runtime-vnext.metal.gated-delta.v4.operation.fp16.none.log-rate-grouped",
            "042cde4824acf50ff0c5fd4d77f0ae4c7e7424bca0ff4a09fcf176e3369c7935",
        ),
        (
            GatedDeltaDecayParameterization::NegativeRate,
            GatedDeltaValueHeadMapping::InterleavedByKeyHead,
        ) => (
            "runtime-vnext.metal.gated-delta.v4.operation.fp16.none.negative-rate-interleaved",
            "2cc1888ca5453ff990ab09e788e49b3d90ffddea6e0c1e2669b81d62ee531f95",
        ),
        _ => panic!("unreviewed gated-delta output tolerance selector"),
    }
}

fn cpu_conv_state_tolerance(semantics: TestSemantics) -> (&'static str, &'static str) {
    match (
        semantics.decay_parameterization,
        semantics.value_head_mapping,
    ) {
        (
            GatedDeltaDecayParameterization::LogRate,
            GatedDeltaValueHeadMapping::GroupedByKeyHead,
        ) => (
            "runtime-vnext.metal.gated-delta.v4.state.conv.fp16.none.log-rate-grouped",
            "be3d2caf3c6b0b7fe6b28e00639a6dd6e3f04e56f9b16e8ed11dae07e314ff98",
        ),
        (
            GatedDeltaDecayParameterization::NegativeRate,
            GatedDeltaValueHeadMapping::InterleavedByKeyHead,
        ) => (
            "runtime-vnext.metal.gated-delta.v4.state.conv.fp16.none.negative-rate-interleaved",
            "fd0a279ca3aea3684f60625e09d130295af6ad0afca771d640eda8567b7a827c",
        ),
        _ => panic!("unreviewed gated-delta conv-state tolerance selector"),
    }
}

fn cpu_delta_state_tolerance(semantics: TestSemantics) -> (&'static str, &'static str) {
    match (
        semantics.decay_parameterization,
        semantics.value_head_mapping,
    ) {
        (
            GatedDeltaDecayParameterization::LogRate,
            GatedDeltaValueHeadMapping::GroupedByKeyHead,
        ) => (
            "runtime-vnext.metal.gated-delta.v4.state.delta.fp32.none.log-rate-grouped",
            "a779e9b4045c63023ac9250463b71cef5993f70933a013a6c749da6ae2c753ab",
        ),
        (
            GatedDeltaDecayParameterization::NegativeRate,
            GatedDeltaValueHeadMapping::InterleavedByKeyHead,
        ) => (
            "runtime-vnext.metal.gated-delta.v4.state.delta.fp32.none.negative-rate-interleaved",
            "0dbe4774a52f662adb015b47792eb4ffb481b5a82c1acb9ee85378fc417c935d",
        ),
        _ => panic!("unreviewed gated-delta delta-state tolerance selector"),
    }
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
        decay_parameterization: GatedDeltaDecayParameterization::LogRate,
        value_head_mapping: GatedDeltaValueHeadMapping::GroupedByKeyHead,
    };
    let tokens = u64::from(u32::MAX) / shape.qkv_features + 1;
    assert!(shape
        .validate_launch_extents(tokens)
        .unwrap_err()
        .contains("QKV activation elements"));
}

#[test]
fn chunked_c64_matches_recurrent_oracle_and_non_aligned_state_continuity() {
    const CHUNK_TOKENS: usize = 145;
    const CHUNK_KEY_HEADS: usize = 2;
    const CHUNK_VALUE_HEADS: usize = 4;
    const STATE_SENTINEL: f32 = 73.25;

    let Some(device) = Device::system_default() else {
        eprintln!("no Metal device; skipping chunked gated-delta conformance");
        return;
    };
    let pipelines = MetalGatedDeltaPipelines::new(&device).unwrap();
    let queue = device.new_command_queue();
    for (key_dim, value_dim) in [(32, 32), (128, 128)] {
        let mut query = float_values(CHUNK_TOKENS * CHUNK_KEY_HEADS * key_dim, 0.019, 0.4);
        let mut key = float_values(CHUNK_TOKENS * CHUNK_KEY_HEADS * key_dim, 0.031, 0.35);
        normalize_rows(&mut query, key_dim);
        normalize_rows(&mut key, key_dim);
        let value = float_values(CHUNK_TOKENS * CHUNK_VALUE_HEADS * value_dim, 0.023, 0.2);
        let g = (0..CHUNK_TOKENS * CHUNK_VALUE_HEADS)
            .map(|index| -0.008 - (index as f32 * 0.017).sin().abs() * 0.025)
            .collect::<Vec<_>>();
        let beta = (0..CHUNK_TOKENS * CHUNK_VALUE_HEADS)
            .map(|index| 0.2 + (index as f32 * 0.013).sin().abs() * 0.6)
            .collect::<Vec<_>>();
        let initial_state = float_values(CHUNK_VALUE_HEADS * value_dim * key_dim, 0.011, 0.025);

        for mapping in [
            GatedDeltaValueHeadMapping::GroupedByKeyHead,
            GatedDeltaValueHeadMapping::InterleavedByKeyHead,
        ] {
            let shape = ChunkTestShape {
                tokens: CHUNK_TOKENS,
                key_heads: CHUNK_KEY_HEADS,
                value_heads: CHUNK_VALUE_HEADS,
                key_dim,
                value_dim,
                mapping,
            };
            let (expected_output, expected_state) =
                recurrent_chunk_oracle(&query, &key, &value, &g, &beta, &initial_state, shape);
            let (full_output, full_state, full_sentinel) = run_chunked_core(
                &device,
                &queue,
                &pipelines,
                &query,
                &key,
                &value,
                &g,
                &beta,
                &initial_state,
                shape,
                STATE_SENTINEL,
            );
            assert_close(
                "chunk/full output",
                &full_output,
                &expected_output,
                CHUNK_CORE_ORACLE_DIAGNOSTIC_MAX_ABS,
            );
            assert_close(
                "chunk/full state",
                &full_state,
                &expected_state,
                CHUNK_STATE_ORACLE_DIAGNOSTIC_MAX_ABS,
            );
            assert_eq!(full_sentinel, [STATE_SENTINEL; 16]);

            let split = 73;
            let first_shape = ChunkTestShape {
                tokens: split,
                ..shape
            };
            let (mut split_output, split_state, first_sentinel) = run_chunked_core(
                &device,
                &queue,
                &pipelines,
                row_slice(&query, CHUNK_KEY_HEADS * key_dim, 0..split),
                row_slice(&key, CHUNK_KEY_HEADS * key_dim, 0..split),
                row_slice(&value, CHUNK_VALUE_HEADS * value_dim, 0..split),
                row_slice(&g, CHUNK_VALUE_HEADS, 0..split),
                row_slice(&beta, CHUNK_VALUE_HEADS, 0..split),
                &initial_state,
                first_shape,
                STATE_SENTINEL,
            );
            assert_eq!(first_sentinel, [STATE_SENTINEL; 16]);
            let second_shape = ChunkTestShape {
                tokens: CHUNK_TOKENS - split,
                ..shape
            };
            let (second_output, split_state, second_sentinel) = run_chunked_core(
                &device,
                &queue,
                &pipelines,
                row_slice(&query, CHUNK_KEY_HEADS * key_dim, split..CHUNK_TOKENS),
                row_slice(&key, CHUNK_KEY_HEADS * key_dim, split..CHUNK_TOKENS),
                row_slice(&value, CHUNK_VALUE_HEADS * value_dim, split..CHUNK_TOKENS),
                row_slice(&g, CHUNK_VALUE_HEADS, split..CHUNK_TOKENS),
                row_slice(&beta, CHUNK_VALUE_HEADS, split..CHUNK_TOKENS),
                &split_state,
                second_shape,
                STATE_SENTINEL,
            );
            split_output.extend(second_output);
            assert_eq!(second_sentinel, [STATE_SENTINEL; 16]);
            assert_close(
                "chunk split output",
                &split_output,
                &full_output,
                CHUNK_CORE_ORACLE_DIAGNOSTIC_MAX_ABS,
            );
            assert_close(
                "chunk split state",
                &split_state,
                &full_state,
                CHUNK_STATE_ORACLE_DIAGNOSTIC_MAX_ABS,
            );
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct ChunkTestShape {
    tokens: usize,
    key_heads: usize,
    value_heads: usize,
    key_dim: usize,
    value_dim: usize,
    mapping: GatedDeltaValueHeadMapping,
}

#[allow(clippy::too_many_arguments)]
fn run_chunked_core(
    device: &Device,
    queue: &CommandQueueRef,
    pipelines: &MetalGatedDeltaPipelines,
    query: &[f32],
    key: &[f32],
    value: &[f32],
    g: &[f32],
    beta: &[f32],
    initial_state: &[f32],
    shape: ChunkTestShape,
    sentinel: f32,
) -> (Vec<f32>, Vec<f32>, [f32; 16]) {
    let params = GatedDeltaParams {
        tokens: shape.tokens as u32,
        hidden_size: 1,
        key_heads: shape.key_heads as u32,
        value_heads: shape.value_heads as u32,
        key_dim: shape.key_dim as u32,
        value_dim: shape.value_dim as u32,
        qkv_features: (2 * shape.key_heads * shape.key_dim + shape.value_heads * shape.value_dim)
            as u32,
        value_features: (shape.value_heads * shape.value_dim) as u32,
        conv_kernel: 2,
        epsilon: 1.0e-6,
        scale: (shape.key_dim as f32).sqrt().recip(),
        decay_parameterization: 0,
        value_head_mapping: match shape.mapping {
            GatedDeltaValueHeadMapping::GroupedByKeyHead => 0,
            GatedDeltaValueHeadMapping::InterleavedByKeyHead => 1,
        },
    };
    let query_buffer = shared_buffer(device, query);
    let key_buffer = shared_buffer(device, key);
    let value_buffer = shared_buffer(device, value);
    let g_buffer = shared_buffer(device, g);
    let beta_buffer = shared_buffer(device, beta);
    let inverse = output_buffer::<f16>(
        device,
        shape.tokens * shape.value_heads * GATED_DELTA_CHUNK_SIZE as usize,
    );
    let uw = output_buffer::<f16>(
        device,
        shape.tokens * shape.value_heads * (shape.value_dim + shape.key_dim),
    );
    let raw_qk = output_buffer::<f16>(
        device,
        shape.tokens * shape.key_heads * GATED_DELTA_CHUNK_SIZE as usize,
    );
    let output = output_buffer::<f32>(device, shape.tokens * shape.value_heads * shape.value_dim);
    let mut guarded_state = initial_state.to_vec();
    guarded_state.extend([sentinel; 16]);
    let state = shared_buffer(device, &guarded_state);

    let command = queue.new_command_buffer();
    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipelines.chunk_kkt_inverse);
    for (index, buffer) in [&*key_buffer, &*g_buffer, &*beta_buffer, &*inverse]
        .into_iter()
        .enumerate()
    {
        set_raw(encoder, index as u64, buffer);
    }
    set_params(encoder, 4, &params);
    encoder.dispatch_thread_groups(
        MTLSize::new(
            (shape.tokens as u64).div_ceil(u64::from(GATED_DELTA_CHUNK_SIZE)),
            shape.value_heads as u64,
            1,
        ),
        MTLSize::new(THREADS_PER_GROUP, 1, 1),
    );

    let uses_specialized_uw = uses_chunk_uw_k128_v128(&params);
    encoder.set_compute_pipeline_state(if uses_specialized_uw {
        &pipelines.chunk_uw_k128_v128
    } else {
        &pipelines.chunk_uw_generic
    });
    for (index, buffer) in [
        &*key_buffer,
        &*value_buffer,
        &*g_buffer,
        &*beta_buffer,
        &*inverse,
        &*uw,
    ]
    .into_iter()
    .enumerate()
    {
        set_raw(encoder, index as u64, buffer);
    }
    set_params(encoder, 6, &params);
    if uses_specialized_uw {
        encoder.dispatch_thread_groups(
            MTLSize::new(shape.tokens as u64, shape.value_heads as u64, 1),
            MTLSize::new(THREADS_PER_GROUP, 1, 1),
        );
    } else {
        dispatch_elements(
            encoder,
            (shape.tokens * shape.value_heads * (shape.value_dim + shape.key_dim)) as u64,
        );
    }

    encoder.set_compute_pipeline_state(&pipelines.chunk_qk);
    for (index, buffer) in [&*query_buffer, &*key_buffer, &*raw_qk]
        .into_iter()
        .enumerate()
    {
        set_raw(encoder, index as u64, buffer);
    }
    set_params(encoder, 3, &params);
    dispatch_elements(
        encoder,
        (shape.tokens * shape.key_heads * GATED_DELTA_CHUNK_SIZE as usize) as u64,
    );

    encoder.set_compute_pipeline_state(
        if shape.key_dim == GATED_DELTA_CHUNK_KEY_DIM_LIMIT as usize {
            &pipelines.chunk_carry_k128
        } else {
            &pipelines.chunk_carry_generic
        },
    );
    for (index, buffer) in [
        &*query_buffer,
        &*key_buffer,
        &*g_buffer,
        &*uw,
        &*state,
        &*output,
    ]
    .into_iter()
    .enumerate()
    {
        set_raw(encoder, index as u64, buffer);
    }
    set_params(encoder, 6, &params);
    encoder.dispatch_thread_groups(
        MTLSize::new(
            (shape.value_dim as u64).div_ceil(VALUE_TILE),
            shape.value_heads as u64,
            1,
        ),
        MTLSize::new(THREADS_PER_GROUP, 1, 1),
    );

    encoder.set_compute_pipeline_state(&pipelines.chunk_output);
    for (index, buffer) in [&*raw_qk, &*g_buffer, &*uw, &*output]
        .into_iter()
        .enumerate()
    {
        set_raw(encoder, index as u64, buffer);
    }
    set_params(encoder, 4, &params);
    dispatch_elements(
        encoder,
        (shape.tokens * shape.value_heads * shape.value_dim) as u64,
    );
    encoder.end_encoding();
    command.commit();
    command.wait_until_completed();
    assert_eq!(command.status(), MTLCommandBufferStatus::Completed);

    let state_elements = shape.value_heads * shape.value_dim * shape.key_dim;
    let guarded = read_f32(&state, state_elements + 16);
    let tail: [f32; 16] = guarded[state_elements..].try_into().unwrap();
    (
        read_f32(&output, shape.tokens * shape.value_heads * shape.value_dim),
        guarded[..state_elements].to_vec(),
        tail,
    )
}

#[allow(clippy::too_many_arguments)]
fn recurrent_chunk_oracle(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    g: &[f32],
    beta: &[f32],
    initial_state: &[f32],
    shape: ChunkTestShape,
) -> (Vec<f32>, Vec<f32>) {
    let mut state = initial_state.to_vec();
    let mut output = vec![0.0_f32; shape.tokens * shape.value_heads * shape.value_dim];
    let repeat = shape.value_heads / shape.key_heads;
    let scale = (shape.key_dim as f32).sqrt().recip();
    for token in 0..shape.tokens {
        for value_head in 0..shape.value_heads {
            let key_head = match shape.mapping {
                GatedDeltaValueHeadMapping::GroupedByKeyHead => value_head / repeat,
                GatedDeltaValueHeadMapping::InterleavedByKeyHead => value_head % shape.key_heads,
            };
            let qk_base = (token * shape.key_heads + key_head) * shape.key_dim;
            let gate_index = token * shape.value_heads + value_head;
            let decay = g[gate_index].exp();
            for value_column in 0..shape.value_dim {
                let state_base = (value_head * shape.value_dim + value_column) * shape.key_dim;
                let mut prediction = 0.0_f32;
                for key_column in 0..shape.key_dim {
                    state[state_base + key_column] *= decay;
                    prediction += state[state_base + key_column] * key[qk_base + key_column];
                }
                let value_index =
                    (token * shape.value_heads + value_head) * shape.value_dim + value_column;
                let delta = (value[value_index] - prediction) * beta[gate_index];
                let mut current = 0.0_f32;
                for key_column in 0..shape.key_dim {
                    state[state_base + key_column] += delta * key[qk_base + key_column];
                    current += state[state_base + key_column] * query[qk_base + key_column] * scale;
                }
                output[value_index] = current;
            }
        }
    }
    (output, state)
}

fn normalize_rows(values: &mut [f32], width: usize) {
    for row in values.chunks_exact_mut(width) {
        let inverse = (row.iter().map(|value| value * value).sum::<f32>() + 1.0e-6)
            .sqrt()
            .recip();
        for value in row {
            *value *= inverse;
        }
    }
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
    semantics: TestSemantics,
) -> Vec<f32> {
    let tokens = inputs.mixed_qkv.len() / QKV_FEATURES;
    let params = test_params(tokens, semantics);
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
        weights.decay_parameter,
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
    decay_parameters: &[f32],
    dt_bias: &[f32],
    norm: &[f32],
    conv_state: &mut [f16],
    delta_state: &mut [f32],
    semantics: TestSemantics,
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
            let key_head = match semantics.value_head_mapping {
                GatedDeltaValueHeadMapping::GroupedByKeyHead => value_head / repeat,
                GatedDeltaValueHeadMapping::InterleavedByKeyHead => value_head % KEY_HEADS,
            };
            let gate_index = token * VALUE_HEADS + value_head;
            let decay_rate = match semantics.decay_parameterization {
                GatedDeltaDecayParameterization::LogRate => -decay_parameters[value_head].exp(),
                GatedDeltaDecayParameterization::NegativeRate => decay_parameters[value_head],
            };
            let g = decay_rate * softplus(inputs.a_raw[gate_index].to_f32() + dt_bias[value_head]);
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

fn test_params(tokens: usize, semantics: TestSemantics) -> GatedDeltaParams {
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
        decay_parameterization: match semantics.decay_parameterization {
            GatedDeltaDecayParameterization::LogRate => 0,
            GatedDeltaDecayParameterization::NegativeRate => 1,
        },
        value_head_mapping: match semantics.value_head_mapping {
            GatedDeltaValueHeadMapping::GroupedByKeyHead => 0,
            GatedDeltaValueHeadMapping::InterleavedByKeyHead => 1,
        },
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
