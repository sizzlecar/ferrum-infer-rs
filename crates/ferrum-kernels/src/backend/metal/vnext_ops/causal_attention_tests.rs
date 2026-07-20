use std::ffi::c_void;

use half::f16;
use metal::{Buffer, BufferRef, CommandQueueRef, MTLCommandBufferStatus, MTLResourceOptions};

use super::super::numerical_tolerance;
use super::*;

const TOKENS: usize = 2;
const QUERY_HEADS: usize = 2;
const KV_HEADS: usize = 1;
const HEAD_DIM: usize = 32;
const ROPE_DIM: usize = 16;
const QUERY_FEATURES: usize = QUERY_HEADS * HEAD_DIM;
const QUERY_PROJECTION_FEATURES: usize = QUERY_FEATURES * 2;
const KV_FEATURES: usize = KV_HEADS * HEAD_DIM;
const TEST_PAGE_ELEMENTS: usize = 2 * KV_FEATURES;
const CPU_OUTPUT_TOLERANCE_ID: &str =
    "runtime-vnext.metal.causal-attention.v2.operation.fp16.none.fixed-page-split";
const CPU_OUTPUT_TOLERANCE_FINGERPRINT: &str =
    "d30006c0535a3b3172ac88db66f75f07df6256e321509188bb0949c7a64a9fdb";
// This stricter diagnostic never substitutes for a catalog-bound release comparison.
const SPLIT_CONTINUITY_DIAGNOSTIC_MAX_ABS: f32 = 0.001;
const CPU_KV_STATE_DIAGNOSTIC_MAX_ABS: f32 = 0.001;

#[test]
fn fixed_page_attention_matches_cpu_and_preserves_split_decode_state_on_real_metal() {
    let Some(device) = Device::system_default() else {
        eprintln!("no Metal device; skipping causal-attention conformance");
        return;
    };
    let pipelines = MetalCausalAttentionPipelines::new(&device).unwrap();
    assert_eq!(pipelines.prepare.thread_execution_width(), SIMD_THREADS);
    assert_eq!(pipelines.attention.thread_execution_width(), SIMD_THREADS);
    let queue = device.new_command_queue();

    let query_raw = half_values(TOKENS * QUERY_PROJECTION_FEATURES, 0.037, 0.31);
    let key_raw = half_values(TOKENS * KV_FEATURES, 0.043, 0.27);
    let value_raw = half_values(TOKENS * KV_FEATURES, 0.029, 0.22);
    let query_norm = half_values(HEAD_DIM, 0.019, 0.94);
    let key_norm = half_values(HEAD_DIM, 0.023, 0.89);

    let full_pages = test_pages(&device, TOKENS);
    assert_ne!(full_pages[0].contents(), full_pages[1].contents());
    let full_output = run_segment(
        &device,
        &queue,
        &pipelines,
        SegmentInputs {
            query_raw: &query_raw,
            key_raw: &key_raw,
            value_raw: &value_raw,
        },
        &query_norm,
        &key_norm,
        &full_pages,
        0,
    );

    let split_pages = test_pages(&device, TOKENS);
    let mut split_output = run_segment(
        &device,
        &queue,
        &pipelines,
        segment(&query_raw, &key_raw, &value_raw, 0),
        &query_norm,
        &key_norm,
        &split_pages[..1],
        0,
    );
    split_output.extend(run_segment(
        &device,
        &queue,
        &pipelines,
        segment(&query_raw, &key_raw, &value_raw, 1),
        &query_norm,
        &key_norm,
        &split_pages,
        1,
    ));

    let cpu_output = cpu_attention(&query_raw, &key_raw, &value_raw, &query_norm, &key_norm);
    let cpu_kv_state = cpu_fixed_page_kv_state(&key_raw, &value_raw, &key_norm);
    let full_kv_state = read_pages(&full_pages);
    let split_kv_state = read_pages(&split_pages);
    assert!(full_output.iter().any(|value| value.abs() > 1.0e-4));
    numerical_tolerance::assert_matches(
        "Metal/CPU causal output",
        &full_output,
        &[TOKENS, QUERY_HEADS, HEAD_DIM],
        &cpu_output,
        &[TOKENS, QUERY_HEADS, HEAD_DIM],
        numerical_tolerance::LogicalDtype::Fp16,
        CPU_OUTPUT_TOLERANCE_ID,
        CPU_OUTPUT_TOLERANCE_FINGERPRINT,
    )
    .expect("reviewed causal-attention numerical contract");
    assert_close(
        "full/cpu causal KV state",
        &full_kv_state,
        &cpu_kv_state,
        CPU_KV_STATE_DIAGNOSTIC_MAX_ABS,
    );
    assert_close(
        "split/cpu causal KV state",
        &split_kv_state,
        &cpu_kv_state,
        CPU_KV_STATE_DIAGNOSTIC_MAX_ABS,
    );
    assert_close(
        "full/split causal output",
        &full_output,
        &split_output,
        SPLIT_CONTINUITY_DIAGNOSTIC_MAX_ABS,
    );
    assert_close(
        "full/split first KV page",
        &read_f16(&full_pages[0], TEST_PAGE_ELEMENTS),
        &read_f16(&split_pages[0], TEST_PAGE_ELEMENTS),
        SPLIT_CONTINUITY_DIAGNOSTIC_MAX_ABS,
    );
    assert_close(
        "full/split second KV page",
        &read_f16(&full_pages[1], TEST_PAGE_ELEMENTS),
        &read_f16(&split_pages[1], TEST_PAGE_ELEMENTS),
        SPLIT_CONTINUITY_DIAGNOSTIC_MAX_ABS,
    );
}

fn read_pages(pages: &[Buffer]) -> Vec<f32> {
    pages
        .iter()
        .flat_map(|page| read_f16(page, TEST_PAGE_ELEMENTS))
        .collect()
}

struct SegmentInputs<'a> {
    query_raw: &'a [f16],
    key_raw: &'a [f16],
    value_raw: &'a [f16],
}

fn segment<'a>(
    query_raw: &'a [f16],
    key_raw: &'a [f16],
    value_raw: &'a [f16],
    token: usize,
) -> SegmentInputs<'a> {
    SegmentInputs {
        query_raw: &query_raw
            [token * QUERY_PROJECTION_FEATURES..(token + 1) * QUERY_PROJECTION_FEATURES],
        key_raw: &key_raw[token * KV_FEATURES..(token + 1) * KV_FEATURES],
        value_raw: &value_raw[token * KV_FEATURES..(token + 1) * KV_FEATURES],
    }
}

#[allow(clippy::too_many_arguments)]
fn run_segment(
    device: &Device,
    queue: &CommandQueueRef,
    pipelines: &MetalCausalAttentionPipelines,
    inputs: SegmentInputs<'_>,
    query_norm_values: &[f16],
    key_norm_values: &[f16],
    pages: &[Buffer],
    position_start: usize,
) -> Vec<f32> {
    let tokens = inputs.query_raw.len() / QUERY_PROJECTION_FEATURES;
    let params = CausalAttentionParams {
        page_elements: TEST_PAGE_ELEMENTS as u32,
        page_count: pages.len() as u32,
        position_start: position_start as u32,
        tokens: tokens as u32,
        query_heads: QUERY_HEADS as u32,
        key_value_heads: KV_HEADS as u32,
        head_dim: HEAD_DIM as u32,
        rope_dim: ROPE_DIM as u32,
        query_projection_stride: QUERY_PROJECTION_FEATURES as u32,
        query_head_stride: (2 * HEAD_DIM) as u32,
        kv_projection_stride: KV_FEATURES as u32,
        output_gate: 1,
        rope_interleaved: 0,
        epsilon: 1.0e-6,
        rope_theta: 10_000.0,
    };
    let query_raw = shared_buffer(device, inputs.query_raw);
    let key_raw = shared_buffer(device, inputs.key_raw);
    let value_raw = shared_buffer(device, inputs.value_raw);
    let query_norm = shared_buffer(device, query_norm_values);
    let key_norm = shared_buffer(device, key_norm_values);
    let query = output_buffer::<f16>(device, tokens * QUERY_FEATURES);
    let output = output_buffer::<f16>(device, tokens * QUERY_FEATURES);
    let argument_buffer = device.new_buffer(
        pipelines.binding_slot_bytes().unwrap(),
        MTLResourceOptions::StorageModeShared,
    );
    let argument_encoder = pipelines.new_binding_encoder();
    argument_encoder.set_argument_buffer(&argument_buffer, 0);
    let page_refs = pages.iter().map(|page| &**page).collect::<Vec<_>>();
    let page_offsets = vec![0; pages.len()];
    argument_encoder.set_buffers(0, &page_refs, &page_offsets);

    let command = queue.new_command_buffer();
    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipelines.prepare);
    for (index, buffer) in [
        &*query_raw,
        &*key_raw,
        &*value_raw,
        &*query_norm,
        &*key_norm,
        &*query,
    ]
    .into_iter()
    .enumerate()
    {
        set_raw(encoder, index as u64, buffer);
    }
    encoder.set_buffer(PREPARE_PAGE_TABLE_INDEX, Some(&argument_buffer), 0);
    set_raw_params(encoder, 7, &params);
    use_raw_pages(encoder, pages);
    encoder.dispatch_thread_groups(
        MTLSize::new(tokens as u64, (QUERY_HEADS + 2 * KV_HEADS) as u64, 1),
        MTLSize::new(SIMD_THREADS, 1, 1),
    );

    encoder.set_compute_pipeline_state(&pipelines.attention);
    set_raw(encoder, 0, &query);
    set_raw(encoder, 1, &query_raw);
    set_raw(encoder, 2, &output);
    encoder.set_buffer(ATTENTION_PAGE_TABLE_INDEX, Some(&argument_buffer), 0);
    set_raw_params(encoder, 4, &params);
    use_raw_pages(encoder, pages);
    encoder.dispatch_thread_groups(
        MTLSize::new(tokens as u64, QUERY_HEADS as u64, 1),
        MTLSize::new(SIMD_THREADS, 1, 1),
    );
    encoder.end_encoding();
    command.commit();
    command.wait_until_completed();
    assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
    read_f16(&output, tokens * QUERY_FEATURES)
}

fn use_raw_pages(encoder: &ComputeCommandEncoderRef, pages: &[Buffer]) {
    for page in pages {
        encoder.use_resource(&**page, MTLResourceUsage::Read | MTLResourceUsage::Write);
    }
}

fn set_raw(encoder: &ComputeCommandEncoderRef, index: u64, buffer: &BufferRef) {
    encoder.set_buffer(index, Some(buffer), 0);
}

fn set_raw_params(encoder: &ComputeCommandEncoderRef, index: u64, params: &CausalAttentionParams) {
    encoder.set_bytes(
        index,
        std::mem::size_of::<CausalAttentionParams>() as u64,
        params as *const _ as *const c_void,
    );
}

fn test_pages(device: &Device, count: usize) -> Vec<Buffer> {
    (0..count)
        .map(|_| output_buffer::<f16>(device, TEST_PAGE_ELEMENTS))
        .collect()
}

fn cpu_attention(
    query_raw: &[f16],
    key_raw: &[f16],
    value_raw: &[f16],
    query_norm: &[f16],
    key_norm: &[f16],
) -> Vec<f32> {
    let mut query = vec![0.0_f32; TOKENS * QUERY_FEATURES];
    let mut key = vec![0.0_f32; TOKENS * KV_FEATURES];
    for token in 0..TOKENS {
        for head in 0..QUERY_HEADS {
            let source = token * QUERY_PROJECTION_FEATURES + head * 2 * HEAD_DIM;
            let destination = token * QUERY_FEATURES + head * HEAD_DIM;
            prepare_head(
                &query_raw[source..source + HEAD_DIM],
                query_norm,
                token,
                &mut query[destination..destination + HEAD_DIM],
            );
        }
        prepare_head(
            &key_raw[token * KV_FEATURES..(token + 1) * KV_FEATURES],
            key_norm,
            token,
            &mut key[token * KV_FEATURES..(token + 1) * KV_FEATURES],
        );
    }

    let mut output = vec![0.0_f32; TOKENS * QUERY_FEATURES];
    for token in 0..TOKENS {
        for head in 0..QUERY_HEADS {
            let query_row = &query[token * QUERY_FEATURES + head * HEAD_DIM
                ..token * QUERY_FEATURES + (head + 1) * HEAD_DIM];
            let mut scores = (0..=token)
                .map(|position| {
                    query_row
                        .iter()
                        .zip(key[position * KV_FEATURES..(position + 1) * KV_FEATURES].iter())
                        .map(|(query, key)| query * key)
                        .sum::<f32>()
                        / (HEAD_DIM as f32).sqrt()
                })
                .collect::<Vec<_>>();
            let maximum = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let denominator = scores
                .iter_mut()
                .map(|score| {
                    *score = (*score - maximum).exp();
                    *score
                })
                .sum::<f32>();
            for dim in 0..HEAD_DIM {
                let context = scores
                    .iter()
                    .enumerate()
                    .map(|(position, score)| {
                        score / denominator * f32::from(value_raw[position * KV_FEATURES + dim])
                    })
                    .sum::<f32>();
                let gate_index =
                    token * QUERY_PROJECTION_FEATURES + head * 2 * HEAD_DIM + HEAD_DIM + dim;
                let gate = 1.0 / (1.0 + (-f32::from(query_raw[gate_index])).exp());
                output[token * QUERY_FEATURES + head * HEAD_DIM + dim] = context * gate;
            }
        }
    }
    output
}

fn cpu_fixed_page_kv_state(key_raw: &[f16], value_raw: &[f16], key_norm: &[f16]) -> Vec<f32> {
    let mut state = vec![0.0_f32; TOKENS * TEST_PAGE_ELEMENTS];
    for token in 0..TOKENS {
        let page = token * TEST_PAGE_ELEMENTS;
        prepare_head(
            &key_raw[token * KV_FEATURES..(token + 1) * KV_FEATURES],
            key_norm,
            token,
            &mut state[page..page + KV_FEATURES],
        );
        for dim in 0..KV_FEATURES {
            state[page + KV_FEATURES + dim] = f32::from(value_raw[token * KV_FEATURES + dim]);
        }
    }
    state
}

fn prepare_head(source: &[f16], weight: &[f16], position: usize, output: &mut [f32]) {
    let sum_squares = source
        .iter()
        .map(|value| f32::from(*value).powi(2))
        .sum::<f32>();
    let scale = 1.0 / (sum_squares / HEAD_DIM as f32 + 1.0e-6).sqrt();
    for pair in 0..ROPE_DIM / 2 {
        let low = pair;
        let high = pair + ROPE_DIM / 2;
        let x0 = f32::from(source[low]) * scale * f32::from(weight[low]);
        let x1 = f32::from(source[high]) * scale * f32::from(weight[high]);
        let angle = position as f32 * 10_000.0_f32.powf(-((2 * pair) as f32) / ROPE_DIM as f32);
        output[low] = x0 * angle.cos() - x1 * angle.sin();
        output[high] = x1 * angle.cos() + x0 * angle.sin();
    }
    for dim in ROPE_DIM..HEAD_DIM {
        output[dim] = f32::from(source[dim]) * scale * f32::from(weight[dim]);
    }
}

fn half_values(length: usize, step: f32, base: f32) -> Vec<f16> {
    (0..length)
        .map(|index| f16::from_f32(base + ((index * 17 + 5) % 41) as f32 * step / 41.0))
        .collect()
}

fn shared_buffer<T>(device: &Device, values: &[T]) -> Buffer {
    device.new_buffer_with_data(
        values.as_ptr().cast(),
        std::mem::size_of_val(values) as u64,
        MTLResourceOptions::StorageModeShared,
    )
}

fn output_buffer<T>(device: &Device, elements: usize) -> Buffer {
    device.new_buffer(
        (elements * std::mem::size_of::<T>()) as u64,
        MTLResourceOptions::StorageModeShared,
    )
}

fn read_f16(buffer: &BufferRef, elements: usize) -> Vec<f32> {
    unsafe { std::slice::from_raw_parts(buffer.contents().cast::<f16>(), elements) }
        .iter()
        .map(|value| f32::from(*value))
        .collect()
}

fn assert_close(label: &str, actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len(), "{label} length");
    let (index, maximum) = actual
        .iter()
        .zip(expected)
        .enumerate()
        .map(|(index, (actual, expected))| (index, (actual - expected).abs()))
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
        .unwrap();
    assert!(
        maximum <= tolerance,
        "{label} maximum absolute error {maximum} at {index}: actual={} expected={} tolerance={tolerance}",
        actual[index],
        expected[index]
    );
}
