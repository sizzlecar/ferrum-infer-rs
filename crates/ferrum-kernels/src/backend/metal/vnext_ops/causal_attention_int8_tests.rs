//! Real Metal prepare/read/continuation checks for the INT8 + F32-scale ABI.
//! These projected-input fixtures do not replace full-provider/model tests.

use super::*;
use std::mem::size_of;

#[path = "causal_attention_packed_tests.rs"]
mod packed;

#[path = "causal_attention_int8_timing_tests.rs"]
mod timing;

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
        let params = self.params(pipelines, state, start);
        self.execute_with_params(device, queue, pipelines, state, &params, attention)
    }

    fn params(
        &self,
        pipelines: &MetalCausalAttentionPipelines,
        state: &State,
        start: usize,
    ) -> CausalAttentionParams {
        CausalAttentionParams {
            page_elements: VNEXT_KV_PAGE_BYTES as u32,
            page_count: state.payload.len() as u32,
            position_start: start as u32,
            tokens: self.tokens as u32,
            query_heads: self.heads as u32,
            key_value_heads: self.kv_heads as u32,
            head_dim: self.dim as u32,
            rope_dim: (self.dim / 4 * 2) as u32,
            query_projection_stride: (self.heads * self.dim * if self.gate { 2 } else { 1 }) as u32,
            query_head_stride: (self.dim * if self.gate { 2 } else { 1 }) as u32,
            kv_projection_stride: (self.kv_heads * self.dim) as u32,
            output_gate: self.gate as u32,
            rope_interleaved: 1,
            attention_simdgroups: pipelines
                .attention_simdgroups_for_context((start + self.tokens) as u64),
            epsilon: 1e-6,
            rope_theta: 10_000.0,
        }
    }

    fn execute_with_params(
        &self,
        device: &Device,
        queue: &CommandQueueRef,
        pipelines: &MetalCausalAttentionPipelines,
        state: &State,
        params: &CausalAttentionParams,
        attention: bool,
    ) -> Output {
        let q = shared_buffer(device, &self.query);
        let k = shared_buffer(device, &self.key);
        let v = shared_buffer(device, &self.value);
        let norm = shared_buffer(device, &vec![f16::ONE; self.dim]);
        let prepared = output_buffer::<f16>(device, self.tokens * self.heads * self.dim);
        let output = GuardedAttentionOutput::new(device, params);
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
        set_raw_params(encoder, 7, params);
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
        let dispatch = attention.then(|| pipelines.dispatch_plan(params));
        if let Some(plan) = dispatch {
            set_raw(encoder, 0, &prepared);
            set_raw(encoder, 1, &q);
            set_raw(encoder, 2, &output.buffer);
            encoder.set_buffer(ATTENTION_PAGE_TABLE_INDEX, Some(&arguments), 0);
            set_raw_params(encoder, 4, params);
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
            attention: dispatch.map_or_else(Vec::new, |plan| {
                output
                    .read_after_completion(plan.kind)
                    .into_iter()
                    .map(f16::from_f32)
                    .collect()
            }),
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
        assert_eq!(&payload[(3 + head) * 32..(4 + head) * 32], &expected);
        assert_eq!(scales[3 + head].to_bits(), scale.to_bits());
        // A one-token attention row has probability exactly one. Check the
        // separately compiled reader's conversion too, including F16 subnormals.
        for (actual, quantized) in output.attention[head * 32..(head + 1) * 32]
            .iter()
            .zip(expected)
        {
            assert_eq!(
                actual.to_bits(),
                f16::from_f32(f32::from(quantized) * scale).to_bits()
            );
        }
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
        for operand in ["query", "key", "value"] {
            let mut inputs = Inputs::new(1, 1, 1, 32, false);
            match operand {
                "query" => inputs.query[7] = invalid,
                "key" => inputs.key[7] = invalid,
                "value" => inputs.value[7] = invalid,
                _ => unreachable!(),
            }
            let state = State::new(&device, 1, 1, 32);
            let output = inputs.execute(&device, &queue, &pipelines, &state, 0, false);
            assert_ne!(
                output.error, 0,
                "{operand} must retain non-finite detection"
            );
        }
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
        let tiled_bytes = (8 * dim + 8 * 32 + 32 * dim) * size_of::<f16>()
            + (8 * dim + 8 * 32) * size_of::<f32>();
        let expected_dispatch = if dim == 256
            && pipelines.supports_gqa_tiled_prefill
            && pipelines.maximum_threadgroup_memory_length >= 31 * 1024
        {
            AttentionDispatchKind::GqaTiledPrefill
        } else if pipelines.maximum_threadgroup_memory_length >= tiled_bytes as u64 {
            AttentionDispatchKind::TiledPrefill
        } else {
            AttentionDispatchKind::General
        };
        assert_eq!(prefix_output.dispatch, Some(expected_dispatch));
        let tail = Inputs::new(9, heads, kv_heads, dim, true);
        let output = tail.execute(&device, &queue, &pipelines, &state, prefix_tokens, true);
        assert_eq!(output.error, 0);
        assert_eq!(output.dispatch, Some(expected_dispatch));
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
    let plan = int8_attention_dispatch_plan(&params, 31 * 1024, true);
    assert_eq!(plan.kind, AttentionDispatchKind::GqaTiledPrefill);
    assert_eq!(plan.threadgroup_memory_bytes, [13312, 18432]);
    assert_eq!(plan.threadgroups, [1, 8, 1]);
    assert_eq!(plan.threads_per_threadgroup, [32, 8, 1]);
    assert_eq!(
        int8_attention_dispatch_plan(&params, 31 * 1024 - 1, true).kind,
        AttentionDispatchKind::TiledPrefill
    );
    assert_eq!(
        int8_attention_dispatch_plan(&params, 32 * 1024, false).kind,
        AttentionDispatchKind::TiledPrefill
    );
    assert_eq!(
        int8_attention_dispatch_plan(&params, 16 * 1024, true).kind,
        AttentionDispatchKind::General
    );
    // Gather-based INT8 tiles can cross page boundaries that prevent F16 direct loads.
    params.key_value_heads = 3;
    params.query_heads = 6;
    params.query_projection_stride = params.query_heads * params.query_head_stride;
    params.kv_projection_stride = params.key_value_heads * params.head_dim;
    assert!(!page_holds_whole_token_rows(&params));
    assert_eq!(
        int8_attention_dispatch_plan(&params, 32 * 1024, true).kind,
        AttentionDispatchKind::GqaTiledPrefill
    );
    for (tokens, dim, heads, kv_heads, expected) in [
        (1, 256, 16, 4, AttentionDispatchKind::DirectDecode),
        (7, 256, 16, 4, AttentionDispatchKind::General),
        (8, 128, 16, 4, AttentionDispatchKind::TiledPrefill),
        (8, 256, 4, 4, AttentionDispatchKind::TiledPrefill),
        (8, 256, 6, 2, AttentionDispatchKind::TiledPrefill),
        (8, 34, 2, 1, AttentionDispatchKind::General),
        (9, 256, 8, 1, AttentionDispatchKind::GqaTiledPrefill),
    ] {
        let mut params = dispatch_test_params(tokens, dim);
        params.page_elements = VNEXT_KV_PAGE_BYTES as u32;
        params.query_heads = heads;
        params.key_value_heads = kv_heads;
        assert_eq!(
            int8_attention_dispatch_plan(&params, 32 * 1024, true).kind,
            expected
        );
    }
}

fn check_gqa_prefill_case(
    heads: usize,
    kv_heads: usize,
    prefix_tokens: usize,
    tokens: usize,
    gate: bool,
) {
    let _ = check_gqa_prefill_inputs(
        prefix_tokens,
        Inputs::new(tokens, heads, kv_heads, 256, gate),
    );
}

fn check_gqa_prefill_inputs(prefix_tokens: usize, inputs: Inputs) -> (Vec<f16>, Vec<f16>) {
    let (heads, kv_heads, dim, tokens, gate) = (
        inputs.heads,
        inputs.kv_heads,
        inputs.dim,
        inputs.tokens,
        inputs.gate,
    );
    assert_eq!(dim, 256);
    let device = Device::system_default().expect("INT8 GQA prefill conformance requires Metal");
    let queue = device.new_command_queue();
    let mut pipelines = MetalCausalAttentionPipelines::new_int8(&device).unwrap();
    assert!(
        pipelines.supports_gqa_tiled_prefill
            && pipelines.maximum_threadgroup_memory_length >= 31 * 1024,
        "INT8 GQA reader not covered on this device: requires 32-lane SIMD, 256 threads and 31744 bytes; actual SIMD={}, threads={}, memory={}",
        pipelines.gqa_tiled_prefill_attention.thread_execution_width(),
        pipelines.gqa_tiled_prefill_attention.max_total_threads_per_threadgroup(),
        pipelines.maximum_threadgroup_memory_length,
    );
    let state = State::new(&device, prefix_tokens + tokens, kv_heads, dim);
    // Poison unused scales with NaN, including each page's final invalid rows.
    for scale in &state.scales {
        // SAFETY: no command has been submitted and this fixture exclusively
        // owns each shared scale buffer. The first prepare writes valid heads.
        unsafe {
            std::slice::from_raw_parts_mut(
                scale.contents().cast::<f32>(),
                scale.length() as usize / 4,
            )
            .fill(f32::NAN);
        }
    }
    if prefix_tokens != 0 {
        let prefix = Inputs::new(prefix_tokens, heads, kv_heads, dim, gate);
        assert_eq!(
            prefix
                .execute(&device, &queue, &pipelines, &state, 0, false)
                .error,
            0
        );
    }
    let selected = inputs.execute(&device, &queue, &pipelines, &state, prefix_tokens, true);
    assert_eq!(selected.error, 0);
    assert_eq!(
        selected.dispatch,
        Some(AttentionDispatchKind::GqaTiledPrefill)
    );
    let saved_payload = state.payload();
    let saved_scales = state
        .scales()
        .iter()
        .map(|scale| scale.to_bits())
        .collect::<Vec<_>>();
    pipelines.supports_gqa_tiled_prefill = false;
    let tiled = inputs.execute(&device, &queue, &pipelines, &state, prefix_tokens, true);
    assert_eq!(tiled.error, 0);
    assert_eq!(tiled.dispatch, Some(AttentionDispatchKind::TiledPrefill));
    assert_eq!(state.payload(), saved_payload);
    assert_eq!(
        state
            .scales()
            .iter()
            .map(|scale| scale.to_bits())
            .collect::<Vec<_>>(),
        saved_scales
    );
    let reference = attention_reference(&inputs, &selected, &state, prefix_tokens);
    let selected_values = selected
        .attention
        .iter()
        .map(|value| value.to_f32())
        .collect::<Vec<_>>();
    let tiled_values = tiled
        .attention
        .iter()
        .map(|value| value.to_f32())
        .collect::<Vec<_>>();
    let cpu_error = assert_close(
        "INT8 GQA/quantized CPU",
        &selected_values,
        &reference,
        0.001,
    );
    let tiled_error = assert_close(
        "INT8 GQA/single-head tiled",
        &selected_values,
        &tiled_values,
        0.001,
    );

    // Compare the reader with F16 general attention over the same dequantized
    // values. This isolates the reader from expected INT8 storage error; it is
    // a test-only conversion, never a provider cache or inference workspace.
    let f16_pipelines = MetalCausalAttentionPipelines::new(&device).unwrap();
    let page_elements = VNEXT_KV_PAGE_BYTES as usize / size_of::<f16>();
    let valid_elements = (prefix_tokens + tokens) * 2 * kv_heads * dim;
    let mut values = vec![f16::NAN; valid_elements.div_ceil(page_elements) * page_elements];
    let scales = state.scales();
    for (index, value) in values[..valid_elements].iter_mut().enumerate() {
        *value = f16::from_f32(f32::from(saved_payload[index]) * scales[index / dim]);
    }
    let pages = values
        .chunks_exact(page_elements)
        .map(|page| shared_buffer(&device, page))
        .collect::<Vec<_>>();
    let mut params = inputs.params(&f16_pipelines, &state, prefix_tokens);
    params.page_elements = page_elements as u32;
    params.page_count = pages.len() as u32;
    let f16_values = run_attention_plan(
        &device,
        &queue,
        &f16_pipelines,
        &shared_buffer(&device, &selected.query),
        &shared_buffer(&device, &inputs.query),
        &pages,
        &params,
        general_attention_dispatch_plan(&params),
    );
    let f16_error = assert_close(
        "INT8 GQA/F16 reader on dequantized state",
        &selected_values,
        &f16_values,
        0.001,
    );
    eprintln!(
        "INT8 GQA actual={:?} Hq={heads} Hkv={kv_heads} D={dim} prefix={prefix_tokens} tokens={tokens} gate={gate}; pipeline SIMD={} max_threads={} device_TG_memory={}; max_abs CPU={cpu_error} old_tiled={tiled_error} F16_dequantized={f16_error}",
        selected.dispatch,
        pipelines.gqa_tiled_prefill_attention.thread_execution_width(),
        pipelines.gqa_tiled_prefill_attention.max_total_threads_per_threadgroup(),
        pipelines.maximum_threadgroup_memory_length,
    );
    (selected.attention, tiled.attention)
}

#[test]
fn int8_gqa_prefill_reuses_kv_across_heads_and_matches_tiled_f16_and_cpu() {
    // GQA4 with a payload-page crossing and query/key tails; GQA2 with a
    // token split across payload pages; MQA8 with one exact query/key tile.
    check_gqa_prefill_case(16, 4, 31, 9, true);
    check_gqa_prefill_case(6, 3, 42, 9, false);
    check_gqa_prefill_case(8, 1, 0, 8, false);

    // Distinct adjacent lanes expose a vector load/store permutation. The
    // three V heads exercise ordinary signed values, half subnormals and the
    // canonical zero head/scale; the ninth token retains the scalar key tail.
    let mut inputs = Inputs::new(9, 6, 3, 256, false);
    let ordinary = [-1.0_f32, 0.0, 0.25, 0.75];
    let subnormal = [0x0001, 0x8001, 0x0002, 0x8002];
    for token in 0..inputs.tokens {
        for head in 0..inputs.kv_heads {
            for dim in 0..inputs.dim {
                let index = (token * inputs.kv_heads + head) * inputs.dim + dim;
                inputs.value[index] = match head {
                    0 => f16::from_f32(ordinary[dim % 4] * (1.0 - token as f32 * 0.05)),
                    1 => f16::from_bits(subnormal[dim % 4]),
                    _ => f16::ZERO,
                };
                if head == 2 {
                    inputs.key[index] = f16::ZERO;
                }
            }
        }
    }
    let (selected, tiled) = check_gqa_prefill_inputs(0, inputs);
    assert_eq!(
        selected
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        tiled
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        "vector gather changed scalar-reader output bits",
    );
}

#[test]
fn int8_gqa_prefill_reads_across_an_independent_scale_page() {
    let kv_heads = 4;
    let prefix_tokens = VNEXT_KV_PAGE_BYTES as usize / (2 * kv_heads * size_of::<f32>()) - 1;
    check_gqa_prefill_case(16, kv_heads, prefix_tokens, 9, true);
}

#[test]
fn int8_general_attention_accepts_a_partial_simd_head() {
    let device = Device::system_default().expect("INT8 partial-head conformance requires Metal");
    let queue = device.new_command_queue();
    let pipelines = MetalCausalAttentionPipelines::new_int8(&device).unwrap();
    let inputs = Inputs::new(9, 2, 1, 34, true);
    let state = State::new(&device, inputs.tokens, inputs.kv_heads, inputs.dim);
    let output = inputs.execute(&device, &queue, &pipelines, &state, 0, true);
    assert_eq!(output.error, 0);
    assert_eq!(output.dispatch, Some(AttentionDispatchKind::General));
    for (actual, expected) in output
        .attention
        .iter()
        .zip(attention_reference(&inputs, &output, &state, 0))
    {
        assert!((actual.to_f32() - expected).abs() <= 0.001);
    }
}

#[test]
fn int8_optimized_attention_reads_across_an_independent_scale_page() {
    let device = Device::system_default().expect("INT8 scale-page conformance requires Metal");
    let queue = device.new_command_queue();
    let pipelines = MetalCausalAttentionPipelines::new_int8(&device).unwrap();
    let (heads, kv_heads, dim) = (2, 1, 128);
    let prefix_tokens = VNEXT_KV_PAGE_BYTES as usize / (2 * kv_heads * size_of::<f32>()) - 1;
    let state = State::new(&device, prefix_tokens + 10, kv_heads, dim);
    let prefix = Inputs::new(prefix_tokens, heads, kv_heads, dim, false);
    // Initialize the existing history without quadratic prefix attention work.
    assert_eq!(
        prefix
            .execute(&device, &queue, &pipelines, &state, 0, false)
            .error,
        0
    );
    for (start, tokens, expected_dispatch) in [
        (prefix_tokens, 9, AttentionDispatchKind::TiledPrefill),
        (prefix_tokens + 9, 1, AttentionDispatchKind::DirectDecode),
    ] {
        let inputs = Inputs::new(tokens, heads, kv_heads, dim, false);
        let output = inputs.execute(&device, &queue, &pipelines, &state, start, true);
        assert_eq!(output.error, 0);
        assert_eq!(output.dispatch, Some(expected_dispatch));
        for (actual, expected) in output
            .attention
            .iter()
            .zip(attention_reference(&inputs, &output, &state, start))
        {
            assert!(
                (actual.to_f32() - expected).abs() <= 0.001,
                "{expected_dispatch:?} read across a scale page: {actual} versus {expected}"
            );
        }
    }
}
