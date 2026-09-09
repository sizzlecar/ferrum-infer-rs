//! Execute the production preparation launch against scalar F64 semantics.
//! Qwen rotates a prefix; proportional Gemma RoPE pads frequencies before
//! rotating the whole head. These must remain distinct for partial RoPE.
use super::*;
use cudarc::driver::{CudaContext, DevicePtr, DevicePtrMut};
use half::f16;

fn normalized(raw: &[f16], weights: Option<&[f16]>, epsilon: f32) -> Vec<f64> {
    let mean = raw.iter().map(|x| x.to_f64().powi(2)).sum::<f64>() / raw.len() as f64;
    raw.iter()
        .enumerate()
        .map(|(i, x)| {
            x.to_f64() / (mean + epsilon as f64).sqrt()
                * weights.map_or(1.0, |weights| weights[i].to_f64())
        })
        .collect()
}

pub(super) fn reference(
    raw: &[f16],
    weights: &[f16],
    shape: CausalAttentionShape,
    position: usize,
    proportional: bool,
) -> Vec<f64> {
    let mut result = normalized(raw, Some(weights), shape.epsilon);
    // Model definitions apply rotate_half to the sliced rotary prefix (Qwen)
    // or to the whole head with padded inactive frequencies (Gemma).
    let rotated_width = if proportional {
        raw.len()
    } else {
        shape.rope_dim as usize
    };
    let half_width = rotated_width / 2;
    let source = result.clone();
    for frequency in 0..shape.rope_dim as usize / 2 {
        let angle = position as f64
            * (shape.rope_theta as f64)
                .powf(-2.0 * frequency as f64 / shape.rope_frequency_denominator as f64);
        let (sine, cosine) = angle.sin_cos();
        let (a, b) = if shape.rope_interleaved {
            (2 * frequency, 2 * frequency + 1)
        } else {
            (frequency, frequency + half_width)
        };
        result[a] = source[a] * cosine - source[b] * sine;
        result[b] = source[a] * sine + source[b] * cosine;
    }
    result
}

fn close(actual: f16, expected: f64, context: &str, index: usize) {
    // Two F16 relative spacings plus a small F32 elementary-function allowance.
    // This compares the declared F16 store against an unrounded F64 oracle.
    let bound = 2.0 * 0.0009765625 * expected.abs().max(1.0) + 1e-5;
    assert!(
        (actual.to_f64() - expected).abs() <= bound,
        "{context}[{index}]: actual={} expected={expected} bound={bound}",
        actual.to_f32()
    );
}

fn run(shape: CausalAttentionShape, position: usize, proportional: bool) {
    let context = CudaContext::new(0).expect("RoPE conformance requires actual CUDA");
    let module = context
        .load_module(Ptx::from_src(crate::ptx::VNEXT_CAUSAL_ATTENTION))
        .unwrap();
    let function = module.load_function(PREPARE_FUNCTION).unwrap();
    let stream = context.default_stream();
    let tokens = 2;
    let width = shape.head_dim as usize;
    let queries = shape.query_heads as usize;
    let keys = shape.key_value_heads as usize;
    let query_stride = shape.query_projection_features as usize;
    let head_stride = width * if shape.output_gate { 2 } else { 1 };
    let kv_stride = shape.kv_features as usize;
    let data = |count, salt| {
        (0..count)
            .map(|i| f16::from_f32(((i * 13 + salt) % 79) as f32 / 32.0 - 1.2))
            .collect::<Vec<_>>()
    };
    let q = data(tokens * query_stride, 7);
    let k = data(tokens * kv_stride, 19);
    let v = data(tokens * kv_stride, 31);
    let norm = (0..width)
        .map(|i| f16::from_f32(0.5 + (i % 17) as f32 / 32.0))
        .collect::<Vec<_>>();
    let q_gpu = stream.clone_htod(&q).unwrap();
    let k_gpu = stream.clone_htod(&k).unwrap();
    let v_gpu = stream.clone_htod(&v).unwrap();
    let norm_gpu = stream.clone_htod(&norm).unwrap();
    let control = stream
        .clone_htod(&[
            1_i32,
            position as i32,
            tokens as i32,
            (position + tokens) as i32,
        ])
        .unwrap();
    let guard = f16::from_f32(-1234.0);
    let padding = 8;
    let output_elements = tokens * queries * width;
    let page_elements = VNEXT_KV_PAGE_BYTES as usize / 2;
    assert!((position + tokens) * 2 * kv_stride <= page_elements);
    let mut output = stream
        .clone_htod(&vec![guard; output_elements + 2 * padding])
        .unwrap();
    let mut page = stream
        .clone_htod(&vec![guard; page_elements + 2 * padding])
        .unwrap();
    {
        let (q_ptr, _q_guard) = q_gpu.device_ptr(&stream);
        let (k_ptr, _k_guard) = k_gpu.device_ptr(&stream);
        let (v_ptr, _v_guard) = v_gpu.device_ptr(&stream);
        let (norm_ptr, _norm_guard) = norm_gpu.device_ptr(&stream);
        let (control_ptr, _control_guard) = control.device_ptr(&stream);
        let (output_ptr, _output_guard) = output.device_ptr_mut(&stream);
        let (page_ptr, _page_guard) = page.device_ptr_mut(&stream);
        let table = stream
            .clone_htod(&[page_ptr + (2 * padding) as u64])
            .unwrap();
        let (table_ptr, _table_guard) = table.device_ptr(&stream);
        let launch = CausalAttentionLaunch {
            input_region: 0,
            output_region: 1,
            binding_offset: 0,
            packed_token_start: 0,
            packed_query_raw: 0,
            packed_key_raw: 0,
            packed_value_raw: 0,
            packed_query: 0,
            packed_context: 0,
            tokens: tokens as u64,
            tokens_i32: tokens as i32,
            sequence_tokens: (position + tokens) as u64,
            sequence_tokens_i32: (position + tokens) as i32,
            table_entries_i32: 1,
            replay_topology: CausalAttentionReplayTopology::new(
                shape,
                CausalAttentionKernelPath::TokenMajorFallback,
                (position + tokens) as u64,
            )
            .unwrap(),
            path: CausalAttentionKernelPath::TokenMajorFallback,
        };
        // Production launcher owns the exact kernel ABI and grid. All raw
        // pointers are backed by live guarded allocations on this stream.
        launch_prepare(
            &stream,
            &function,
            q_ptr,
            k_ptr,
            v_ptr,
            norm_ptr,
            norm_ptr,
            output_ptr + (2 * padding) as u64,
            control_ptr,
            table_ptr,
            launch,
            shape.cuda_shape().unwrap(),
            0,
            None,
        )
        .unwrap();
    }
    let actual = stream.clone_dtoh(&output).unwrap();
    let actual_page = stream.clone_dtoh(&page).unwrap();
    for token in 0..tokens {
        for head in 0..queries {
            let start = token * query_stride + head * head_stride;
            let expected = reference(
                &q[start..start + width],
                &norm,
                shape,
                position + token,
                proportional,
            );
            for (dim, expected) in expected.into_iter().enumerate() {
                let index = padding + (token * queries + head) * width + dim;
                close(actual[index], expected, "query", index);
            }
        }
        for head in 0..keys {
            let start = token * kv_stride + head * width;
            let expected_key = reference(
                &k[start..start + width],
                &norm,
                shape,
                position + token,
                proportional,
            );
            let expected_value = if shape.value_rms_norm {
                normalized(&v[start..start + width], None, shape.epsilon)
            } else {
                v[start..start + width].iter().map(|x| x.to_f64()).collect()
            };
            for dim in 0..width {
                let index = padding + (position + token) * kv_stride * 2 + head * width + dim;
                close(actual_page[index], expected_key[dim], "key", index);
                close(
                    actual_page[index + kv_stride],
                    expected_value[dim],
                    "value",
                    index + kv_stride,
                );
            }
        }
    }
    for (i, value) in actual.iter().enumerate() {
        if !(padding..padding + output_elements).contains(&i) {
            assert_eq!(value.to_bits(), guard.to_bits());
        }
    }
    let written = padding + position * 2 * kv_stride..padding + (position + tokens) * 2 * kv_stride;
    for (i, value) in actual_page.iter().enumerate() {
        if !written.contains(&i) {
            assert_eq!(value.to_bits(), guard.to_bits(), "KV guard {i}");
        }
    }
}

#[test]
#[ignore = "requires an actual CUDA device"]
fn standard_partial_rope_rotates_only_its_prefix_on_cuda() {
    for interleaved in [false, true] {
        let mut attributes = tests::attributes(true);
        attributes.insert(
            AttributeId::new("rope_interleaved").unwrap(),
            SemanticValue::Bool(interleaved),
        );
        let shape = CausalAttentionShape::from_attributes(&attributes).unwrap();
        for position in [0, 3] {
            run(shape, position, false);
        }
    }
}

#[test]
#[ignore = "requires an actual CUDA device"]
fn proportional_partial_rope_retains_padded_head_pairs_on_cuda() {
    let shape = CausalAttentionShape::from_attributes_for(
        &tests::gemma4_attributes(true),
        CausalAttentionSemantics::Gemma4,
    )
    .unwrap();
    for position in [0, 3] {
        run(shape, position, true);
    }
}
