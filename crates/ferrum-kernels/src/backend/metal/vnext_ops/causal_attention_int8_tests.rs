//! Real Metal prepare/read/continuation checks for the INT8 + F32-scale ABI.
//! These projected-input fixtures do not replace full-provider/model tests.

use super::*;
use std::mem::size_of;

struct State {
    payload: Vec<Buffer>,
    scales: Vec<Buffer>,
    heads: usize,
    dim: usize,
}

impl State {
    fn new(device: &Device, tokens: usize, heads: usize, dim: usize) -> Self {
        let pages = |bytes: usize| {
            (0..bytes.div_ceil(VNEXT_KV_PAGE_BYTES as usize))
                .map(|_| shared_buffer(device, &vec![0xa5_u8; VNEXT_KV_PAGE_BYTES as usize]))
                .collect()
        };
        Self {
            payload: pages(tokens * 2 * heads * dim),
            scales: pages(tokens * 2 * heads * size_of::<f32>()),
            heads,
            dim,
        }
    }

    fn payload(&self) -> Vec<i8> {
        self.payload
            .iter()
            .flat_map(|page| read::<i8>(page, page.length() as usize))
            .collect()
    }

    fn scales(&self) -> Vec<f32> {
        self.scales
            .iter()
            .flat_map(|page| read::<f32>(page, page.length() as usize / 4))
            .collect()
    }

    fn copy_prefix_to(&self, queue: &CommandQueueRef, target: &Self, tokens: usize) {
        let command = queue.new_command_buffer();
        let blit = command.new_blit_command_encoder();
        for (source, destination, bytes) in [
            (
                &self.payload,
                &target.payload,
                tokens * 2 * self.heads * self.dim,
            ),
            (&self.scales, &target.scales, tokens * 2 * self.heads * 4),
        ] {
            let mut remaining = bytes as u64;
            for (source, destination) in source.iter().zip(destination) {
                let count = remaining.min(VNEXT_KV_PAGE_BYTES);
                if count == 0 {
                    break;
                }
                blit.copy_from_buffer(source, 0, destination, 0, count);
                remaining -= count;
            }
            assert_eq!(remaining, 0);
        }
        blit.end_encoding();
        command.commit();
        command.wait_until_completed();
        assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
    }
}

fn read<T: Copy>(buffer: &BufferRef, elements: usize) -> Vec<T> {
    assert!(elements * size_of::<T>() <= buffer.length() as usize);
    // SAFETY: each caller owns a shared allocation and reads after its exact
    // submission completed. Fixtures use naturally aligned primitive types.
    unsafe { std::slice::from_raw_parts(buffer.contents().cast::<T>(), elements).to_vec() }
}

struct Inputs {
    query: Vec<f16>,
    key: Vec<f16>,
    value: Vec<f16>,
    heads: usize,
    kv_heads: usize,
    dim: usize,
    tokens: usize,
    gate: bool,
}

impl Inputs {
    fn new(tokens: usize, heads: usize, kv_heads: usize, dim: usize, gate: bool) -> Self {
        Self {
            query: half_values(tokens * heads * dim * if gate { 2 } else { 1 }, 0.17, 0.4),
            key: half_values(tokens * kv_heads * dim, 0.31, -0.15),
            value: half_values(tokens * kv_heads * dim, 0.72, -0.36),
            heads,
            kv_heads,
            dim,
            tokens,
            gate,
        }
    }

    fn execute(
        &self,
        device: &Device,
        queue: &CommandQueueRef,
        pipelines: &MetalCausalAttentionPipelines,
        state: &State,
        start: usize,
        attention: bool,
    ) -> Output {
        assert_eq!((self.kv_heads, self.dim), (state.heads, state.dim));
        let params = CausalAttentionParams {
            page_elements: VNEXT_KV_PAGE_BYTES as u32,
            page_count: state.payload.len() as u32,
            position_start: start as u32,
            tokens: self.tokens as u32,
            query_heads: self.heads as u32,
            key_value_heads: self.kv_heads as u32,
            head_dim: self.dim as u32,
            rope_dim: (self.dim / 2) as u32,
            query_projection_stride: (self.heads * self.dim * if self.gate { 2 } else { 1 }) as u32,
            query_head_stride: (self.dim * if self.gate { 2 } else { 1 }) as u32,
            kv_projection_stride: (self.kv_heads * self.dim) as u32,
            output_gate: self.gate as u32,
            rope_interleaved: 1,
            attention_simdgroups: pipelines
                .attention_simdgroups_for_context((start + self.tokens) as u64),
            epsilon: 1e-6,
            rope_theta: 10_000.0,
        };
        let q = shared_buffer(device, &self.query);
        let k = shared_buffer(device, &self.key);
        let v = shared_buffer(device, &self.value);
        let norm = shared_buffer(device, &vec![f16::ONE; self.dim]);
        let prepared = output_buffer::<f16>(device, self.tokens * self.heads * self.dim);
        let output = output_buffer::<f16>(device, self.tokens * self.heads * self.dim);
        let arguments = device.new_buffer(
            pipelines.binding_slot_bytes().unwrap(),
            MTLResourceOptions::StorageModeShared,
        );
        pipelines
            .with_binding_encoder(|encoder| {
                encoder.set_argument_buffer(&arguments, 0);
                encoder.set_buffers(
                    0,
                    &state.payload.iter().map(|page| &**page).collect::<Vec<_>>(),
                    &vec![0; state.payload.len()],
                );
                encoder.set_buffers(
                    MAXIMUM_KV_PAGES,
                    &state.scales.iter().map(|page| &**page).collect::<Vec<_>>(),
                    &vec![0; state.scales.len()],
                );
                Ok(())
            })
            .unwrap();
        let status_offset = pipelines.error_flag_offset().unwrap();
        let command = queue.new_command_buffer();
        let blit = command.new_blit_command_encoder();
        blit.fill_buffer(&arguments, NSRange::new(status_offset, 4), 0);
        blit.end_encoding();
        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&pipelines.prepare);
        for (index, buffer) in [&*q, &*k, &*v, &*norm, &*norm, &*prepared]
            .into_iter()
            .enumerate()
        {
            set_raw(encoder, index as u64, buffer);
        }
        encoder.set_buffer(PREPARE_PAGE_TABLE_INDEX, Some(&arguments), 0);
        encoder.set_buffer(8, Some(&arguments), status_offset);
        set_raw_params(encoder, 7, &params);
        use_raw_pages(encoder, &state.payload);
        use_raw_pages(encoder, &state.scales);
        encoder.dispatch_thread_groups(
            MTLSize::new(
                self.tokens as u64,
                (self.heads + 2 * self.kv_heads) as u64,
                1,
            ),
            MTLSize::new(SIMD_THREADS, 1, 1),
        );
        let dispatch = attention.then(|| pipelines.dispatch_plan(&params));
        if let Some(plan) = dispatch {
            set_raw(encoder, 0, &prepared);
            set_raw(encoder, 1, &q);
            set_raw(encoder, 2, &output);
            encoder.set_buffer(ATTENTION_PAGE_TABLE_INDEX, Some(&arguments), 0);
            set_raw_params(encoder, 4, &params);
            encode_attention_dispatch(pipelines, encoder, plan);
        }
        encoder.end_encoding();
        command.commit();
        command.wait_until_completed();
        assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
        let error =
            read::<u32>(&arguments, status_offset as usize / 4 + 1)[status_offset as usize / 4];
        Output {
            query: read::<f16>(&prepared, self.tokens * self.heads * self.dim),
            attention: if attention {
                read::<f16>(&output, self.tokens * self.heads * self.dim)
            } else {
                Vec::new()
            },
            error,
            dispatch: dispatch.map(|plan| plan.kind),
        }
    }
}

struct Output {
    query: Vec<f16>,
    attention: Vec<f16>,
    error: u32,
    dispatch: Option<AttentionDispatchKind>,
}

fn quantize(values: &[f16]) -> (Vec<i8>, f32) {
    let maximum = values
        .iter()
        .map(|value| value.to_f32().abs())
        .fold(0.0_f32, f32::max);
    let scale = if maximum == 0.0 { 1.0 } else { maximum / 127.0 };
    (
        values
            .iter()
            .map(|value| {
                (value.to_f32() / scale)
                    .round_ties_even()
                    .clamp(-127.0, 127.0) as i8
            })
            .collect(),
        scale,
    )
}

#[test]
fn int8_prepare_matches_reference_rounding_zero_and_subnormal_scales() {
    let device = Device::system_default().expect("INT8 KV conformance requires Metal");
    let queue = device.new_command_queue();
    let pipelines = MetalCausalAttentionPipelines::new_int8(&device).unwrap();
    let mut inputs = Inputs::new(1, 3, 3, 32, false);
    for (head, values) in inputs.value.chunks_exact_mut(32).enumerate() {
        for (dim, value) in values.iter_mut().enumerate() {
            *value = match head {
                0 => f16::ZERO,
                1 => f16::from_f32([127.0, -127.0, 0.5, -0.5, 1.5, -1.5, 2.5, -2.5][dim % 8]),
                _ => f16::from_bits(if dim.is_multiple_of(2) { 1 } else { 0x8001 }),
            };
        }
    }
    let state = State::new(&device, 1, 3, 32);
    let output = inputs.execute(&device, &queue, &pipelines, &state, 0, true);
    assert_eq!(output.error, 0);
    let payload = state.payload();
    let scales = state.scales();
    for head in 0..3 {
        let (expected, scale) = quantize(&inputs.value[head * 32..(head + 1) * 32]);
        assert_eq!(&payload[(3 + head) * 32..(4 + head) * 32], expected);
        assert_eq!(scales[3 + head].to_bits(), scale.to_bits());
    }
    assert!(output.attention.iter().all(|value| value.is_finite()));
    assert!(payload[192..].iter().all(|value| *value == 0xa5_u8 as i8));
}

#[test]
fn int8_prepare_crosses_payload_and_independent_scale_page_frontiers() {
    let device = Device::system_default().expect("INT8 KV conformance requires Metal");
    let queue = device.new_command_queue();
    let pipelines = MetalCausalAttentionPipelines::new_int8(&device).unwrap();
    let inputs = Inputs::new(3, 2, 1, 32, false);
    for start in [1023, 8191] {
        let state = State::new(&device, start + 3, 1, 32);
        let before_scales = state
            .scales()
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>();
        let output = inputs.execute(&device, &queue, &pipelines, &state, start, false);
        assert_eq!(output.error, 0);
        let payload = state.payload();
        let scales = state.scales();
        assert!(payload[..start * 64]
            .iter()
            .all(|value| *value == 0xa5_u8 as i8));
        assert!(payload[(start + 3) * 64..]
            .iter()
            .all(|value| *value == 0xa5_u8 as i8));
        for token in 0..3 {
            let (expected, scale) = quantize(&inputs.value[token * 32..(token + 1) * 32]);
            let base = ((start + token) * 2 + 1) * 32;
            assert_eq!(&payload[base..base + 32], expected);
            assert_eq!(scales[(start + token) * 2 + 1].to_bits(), scale.to_bits());
        }
        assert_eq!(
            scales[..start * 2]
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            before_scales[..start * 2]
        );
        assert_eq!(
            scales[(start + 3) * 2..]
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            before_scales[(start + 3) * 2..]
        );
    }
}

fn attention_reference(inputs: &Inputs, output: &Output, state: &State, start: usize) -> Vec<f32> {
    let payload = state.payload();
    let scales = state.scales();
    let value = |token: usize, kind: usize, head: usize, dim: usize| {
        let index = (token * 2 + kind) * inputs.kv_heads + head;
        f32::from(payload[index * inputs.dim + dim]) * scales[index]
    };
    let mut expected = vec![0.0; output.attention.len()];
    for token in 0..inputs.tokens {
        for head in 0..inputs.heads {
            let kv_head = head / (inputs.heads / inputs.kv_heads);
            let base = (token * inputs.heads + head) * inputs.dim;
            let scores = (0..=start + token)
                .map(|position| {
                    (0..inputs.dim)
                        .map(|dim| {
                            output.query[base + dim].to_f32() * value(position, 0, kv_head, dim)
                        })
                        .sum::<f32>()
                        / (inputs.dim as f32).sqrt()
                })
                .collect::<Vec<_>>();
            let maximum = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let weights = scores
                .iter()
                .map(|score| (score - maximum).exp())
                .collect::<Vec<_>>();
            let sum = weights.iter().sum::<f32>();
            for dim in 0..inputs.dim {
                let mut result = weights
                    .iter()
                    .enumerate()
                    .map(|(position, weight)| weight * value(position, 1, kv_head, dim))
                    .sum::<f32>()
                    / sum;
                if inputs.gate {
                    let gate = inputs.query[token * inputs.heads * inputs.dim * 2
                        + head * inputs.dim * 2
                        + inputs.dim
                        + dim]
                        .to_f32();
                    result *= 1.0 / (1.0 + (-gate).exp());
                }
                expected[base + dim] = result;
            }
        }
    }
    expected
}

#[test]
fn int8_attention_matches_cpu_for_gqa_mqa_and_complete_prefix_restore() {
    let device = Device::system_default().expect("INT8 attention conformance requires Metal");
    let queue = device.new_command_queue();
    let pipelines = MetalCausalAttentionPipelines::new_int8(&device).unwrap();
    for (heads, kv_heads, dim, gate) in [(2, 2, 32, false), (4, 1, 128, true), (4, 2, 256, false)] {
        let source = State::new(&device, 5, kv_heads, dim);
        let checkpoint = State::new(&device, 5, kv_heads, dim);
        let restored = State::new(&device, 5, kv_heads, dim);
        let prefix = Inputs::new(3, heads, kv_heads, dim, gate);
        let prefix_output = prefix.execute(&device, &queue, &pipelines, &source, 0, true);
        assert_eq!(prefix_output.error, 0);
        for (actual, expected) in prefix_output.attention.iter().zip(attention_reference(
            &prefix,
            &prefix_output,
            &source,
            0,
        )) {
            assert!(
                (actual.to_f32() - expected).abs() <= 0.001,
                "INT8 attention differs from scalar reference"
            );
        }
        source.copy_prefix_to(&queue, &checkpoint, 3);
        let saved_payload = checkpoint.payload();
        let saved_scales = checkpoint
            .scales()
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>();
        let suffix = Inputs::new(2, heads, kv_heads, dim, gate);
        let continued = suffix.execute(&device, &queue, &pipelines, &source, 3, true);
        checkpoint.copy_prefix_to(&queue, &restored, 3);
        let resumed = suffix.execute(&device, &queue, &pipelines, &restored, 3, true);
        assert_eq!(continued.error, 0);
        assert_eq!(resumed.error, 0);
        assert_eq!(continued.attention, resumed.attention);
        assert_eq!(source.payload(), restored.payload());
        assert_eq!(checkpoint.payload(), saved_payload);
        assert_eq!(
            checkpoint
                .scales()
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            saved_scales
        );
        for (actual, expected) in resumed
            .attention
            .iter()
            .zip(attention_reference(&suffix, &resumed, &restored, 3))
        {
            assert!(
                (actual.to_f32() - expected).abs() <= 0.001,
                "restored INT8 attention differs from scalar reference"
            );
        }
    }
}

#[test]
fn int8_prepare_marks_nonfinite_input_in_device_status() {
    let device =
        Device::system_default().expect("INT8 numerical failure conformance requires Metal");
    let queue = device.new_command_queue();
    let pipelines = MetalCausalAttentionPipelines::new_int8(&device).unwrap();
    for invalid in [f16::NAN, f16::INFINITY] {
        let mut inputs = Inputs::new(1, 1, 1, 32, false);
        inputs.value[7] = invalid;
        let state = State::new(&device, 1, 1, 32);
        let output = inputs.execute(&device, &queue, &pipelines, &state, 0, false);
        assert_ne!(output.error, 0);
    }
}

#[test]
fn int8_tiled_prefill_and_direct_decode_match_reference_across_payload_page_and_tail() {
    let device =
        Device::system_default().expect("INT8 optimized attention conformance requires Metal");
    let queue = device.new_command_queue();
    let pipelines = MetalCausalAttentionPipelines::new_int8(&device).unwrap();
    for dim in [128, 256] {
        let heads = 4;
        let kv_heads = 2;
        let page_tokens = VNEXT_KV_PAGE_BYTES as usize / (2 * kv_heads * dim);
        let prefix_tokens = page_tokens - 1;
        let state = State::new(&device, prefix_tokens + 10, kv_heads, dim);
        let prefix = Inputs::new(prefix_tokens, heads, kv_heads, dim, true);
        let prefix_output = prefix.execute(&device, &queue, &pipelines, &state, 0, true);
        assert_eq!(prefix_output.error, 0);
        assert_eq!(
            prefix_output.dispatch,
            Some(AttentionDispatchKind::TiledPrefill)
        );
        let tail = Inputs::new(9, heads, kv_heads, dim, true);
        let output = tail.execute(&device, &queue, &pipelines, &state, prefix_tokens, true);
        assert_eq!(output.error, 0);
        assert_eq!(output.dispatch, Some(AttentionDispatchKind::TiledPrefill));
        for (actual, expected) in
            output
                .attention
                .iter()
                .zip(attention_reference(&tail, &output, &state, prefix_tokens))
        {
            assert!(
                (actual.to_f32() - expected).abs() <= 0.001,
                "INT8 tiled prefill differs from scalar reference: {actual} versus {expected}"
            );
        }
        let decode = Inputs::new(1, heads, kv_heads, dim, true);
        let output = decode.execute(&device, &queue, &pipelines, &state, prefix_tokens + 9, true);
        assert_eq!(output.error, 0);
        assert_eq!(output.dispatch, Some(AttentionDispatchKind::DirectDecode));
        for (actual, expected) in output.attention.iter().zip(attention_reference(
            &decode,
            &output,
            &state,
            prefix_tokens + 9,
        )) {
            assert!(
                (actual.to_f32() - expected).abs() <= 0.001,
                "INT8 direct decode differs from scalar reference"
            );
        }
    }
}

#[test]
fn int8_tiled_dispatch_accounts_for_bounded_dequantization_memory() {
    let mut params = dispatch_test_params(8, 256);
    params.page_elements = VNEXT_KV_PAGE_BYTES as u32;
    let plan = int8_attention_dispatch_plan(&params, 32 * 1024);
    assert_eq!(plan.kind, AttentionDispatchKind::TiledPrefill);
    assert!(plan.threadgroup_memory_bytes.iter().sum::<u64>() <= 32 * 1024);
    assert_eq!(
        int8_attention_dispatch_plan(&params, 16 * 1024).kind,
        AttentionDispatchKind::General
    );
    // Gather-based INT8 tiles can cross page boundaries that prevent F16 direct loads.
    params.key_value_heads = 3;
    params.query_heads = 6;
    params.query_projection_stride = params.query_heads * params.query_head_stride;
    params.kv_projection_stride = params.key_value_heads * params.head_dim;
    assert!(!page_holds_whole_token_rows(&params));
    assert_eq!(
        int8_attention_dispatch_plan(&params, 32 * 1024).kind,
        AttentionDispatchKind::TiledPrefill
    );
}
