//! Exercise production scratch offsets with the contiguous projected rows emitted
//! by packed GEMM. Independent submissions supply the state/prepare oracle;
//! attention is also checked against a scalar Rust implementation.

use super::*;

struct PagedKv {
    payload: Vec<Buffer>,
    scales: Vec<Buffer>,
    dtype: ElementType,
}

impl PagedKv {
    fn new(device: &Device, tokens: usize, inputs: &Inputs, dtype: ElementType) -> Self {
        let pages = |bytes: usize| {
            (0..bytes.div_ceil(VNEXT_KV_PAGE_BYTES as usize))
                .map(|_| shared_buffer(device, &vec![0xa5_u8; VNEXT_KV_PAGE_BYTES as usize]))
                .collect()
        };
        Self {
            payload: pages(tokens * 2 * inputs.kv_heads * inputs.dim * dtype.size_bytes() as usize),
            scales: if dtype == ElementType::I8 {
                pages(tokens * 2 * inputs.kv_heads * size_of::<f32>())
            } else {
                Vec::new()
            },
            dtype,
        }
    }

    fn snapshot(&self) -> Vec<u8> {
        self.payload
            .iter()
            .chain(&self.scales)
            .flat_map(|page| read::<u8>(page, page.length() as usize))
            .collect()
    }

    fn decoded(&self, inputs: &Inputs) -> Vec<f32> {
        if self.dtype == ElementType::F16 {
            return self
                .payload
                .iter()
                .flat_map(|page| read::<f16>(page, page.length() as usize / 2))
                .map(f16::to_f32)
                .collect();
        }
        let scales = self
            .scales
            .iter()
            .flat_map(|page| read::<f32>(page, page.length() as usize / 4))
            .collect::<Vec<_>>();
        self.payload
            .iter()
            .flat_map(|page| read::<i8>(page, page.length() as usize))
            .enumerate()
            .map(|(index, value)| f32::from(value) * scales[index / inputs.dim])
            .collect()
    }
}

fn run_packed_projected(
    device: &Device,
    queue: &CommandQueueRef,
    pipelines: &MetalCausalAttentionPipelines,
    participants: &[(&Inputs, usize, &PagedKv)],
) -> Vec<Output> {
    let first = participants[0].0;
    let shape = CausalAttentionShape {
        hidden_size: 8,
        query_heads: first.heads as u64,
        key_value_heads: first.kv_heads as u64,
        head_dim: first.dim as u64,
        query_features: (first.heads * first.dim) as u64,
        query_projection_features: (first.heads * first.dim * if first.gate { 2 } else { 1 })
            as u64,
        kv_features: (first.kv_heads * first.dim) as u64,
        rope_dim: (first.dim / 4 * 2) as u64,
        maximum_context_tokens: 64,
        epsilon: 1e-6,
        rope_theta: 10_000.0,
        rope_interleaved: true,
        output_gate: first.gate,
    };
    let total_tokens = participants
        .iter()
        .map(|(inputs, _, _)| inputs.tokens as u64)
        .sum();
    let layout =
        ScratchLayout::new_with_storage(shape, total_tokens, participants.len(), pipelines.kv_type)
            .unwrap();
    let mut scratch_values = vec![f16::ZERO; layout.required_bytes as usize / 2];
    let mut packed_start = 0;
    for &(inputs, _, _) in participants {
        for (base, width, values) in [
            (
                layout.query_raw,
                shape.query_projection_features,
                &inputs.query,
            ),
            (layout.key_raw, shape.kv_features, &inputs.key),
            (layout.value_raw, shape.kv_features, &inputs.value),
        ] {
            // These are the contiguous rows written by the shared projection,
            // independently of the per-segment allocation alignment.
            let start = (base / 2 + packed_start * width) as usize;
            scratch_values[start..start + values.len()].copy_from_slice(values);
        }
        packed_start += inputs.tokens as u64;
    }
    let scratch = shared_buffer(device, &scratch_values);
    let norm = shared_buffer(device, &vec![f16::ONE; first.dim]);
    let binding_layout =
        BindingLayout::new(pipelines.binding_slot_bytes().unwrap(), participants.len()).unwrap();
    let arguments = device.new_buffer(
        binding_layout.required_bytes,
        MTLResourceOptions::StorageModeShared,
    );
    for (participant, &(_, _, state)) in participants.iter().enumerate() {
        pipelines
            .with_binding_encoder(|encoder| {
                encoder
                    .set_argument_buffer(&arguments, binding_layout.offset(participant).unwrap());
                encoder.set_buffers(
                    0,
                    &state.payload.iter().map(|page| &**page).collect::<Vec<_>>(),
                    &vec![0; state.payload.len()],
                );
                if state.dtype == ElementType::I8 {
                    encoder.set_buffers(
                        MAXIMUM_KV_PAGES,
                        &state.scales.iter().map(|page| &**page).collect::<Vec<_>>(),
                        &vec![0; state.scales.len()],
                    );
                }
                Ok(())
            })
            .unwrap();
    }
    let command = queue.new_command_buffer();
    if pipelines.kv_type == ElementType::I8 {
        let blit = command.new_blit_command_encoder();
        for participant in 0..participants.len() {
            blit.fill_buffer(
                &arguments,
                NSRange::new(
                    binding_layout.offset(participant).unwrap()
                        + pipelines.error_flag_offset().unwrap(),
                    4,
                ),
                0,
            );
        }
        blit.end_encoding();
    }
    let encoder = command.new_compute_command_encoder();
    packed_start = 0;
    let mut readback = Vec::new();
    for (participant, &(inputs, start, state)) in participants.iter().enumerate() {
        assert_eq!(
            (inputs.heads, inputs.kv_heads, inputs.dim, inputs.gate),
            (first.heads, first.kv_heads, first.dim, first.gate)
        );
        let mut params = shape
            .params(
                inputs.tokens as u64,
                start as u64,
                state.payload.len() as u64,
                pipelines.attention_simdgroups_for_context((start + inputs.tokens) as u64),
            )
            .unwrap();
        params.page_elements = (VNEXT_KV_PAGE_BYTES / state.dtype.size_bytes()) as u32;
        let q_raw = layout
            .token_offset(
                layout.query_raw,
                packed_start,
                shape.query_projection_features,
            )
            .unwrap();
        let k_raw = layout
            .token_offset(layout.key_raw, packed_start, shape.kv_features)
            .unwrap();
        let v_raw = layout
            .token_offset(layout.value_raw, packed_start, shape.kv_features)
            .unwrap();
        let query = layout
            .token_offset(layout.query, packed_start, shape.query_features)
            .unwrap();
        let context = layout
            .token_offset(layout.context, packed_start, shape.query_features)
            .unwrap();
        let binding_offset = binding_layout.offset(participant).unwrap();
        encoder.set_compute_pipeline_state(&pipelines.prepare);
        for (index, offset) in [q_raw, k_raw, v_raw].into_iter().enumerate() {
            encoder.set_buffer(index as u64, Some(&scratch), offset);
        }
        set_raw(encoder, 3, &norm);
        set_raw(encoder, 4, &norm);
        encoder.set_buffer(5, Some(&scratch), query);
        encoder.set_buffer(PREPARE_PAGE_TABLE_INDEX, Some(&arguments), binding_offset);
        set_raw_params(encoder, 7, &params);
        if state.dtype == ElementType::I8 {
            encoder.set_buffer(
                8,
                Some(&arguments),
                binding_offset + pipelines.error_flag_offset().unwrap(),
            );
        }
        use_raw_pages(encoder, &state.payload);
        use_raw_pages(encoder, &state.scales);
        encoder.set_threadgroup_memory_length(0, 0);
        encoder.set_threadgroup_memory_length(1, 0);
        encoder.dispatch_thread_groups(
            MTLSize::new(
                inputs.tokens as u64,
                (inputs.heads + 2 * inputs.kv_heads) as u64,
                1,
            ),
            MTLSize::new(SIMD_THREADS, 1, 1),
        );
        encoder.set_buffer(0, Some(&scratch), query);
        encoder.set_buffer(1, Some(&scratch), q_raw);
        encoder.set_buffer(2, Some(&scratch), context);
        encoder.set_buffer(ATTENTION_PAGE_TABLE_INDEX, Some(&arguments), binding_offset);
        set_raw_params(encoder, 4, &params);
        let plan = pipelines.dispatch_plan(&params);
        assert_eq!(plan.kind, AttentionDispatchKind::General);
        encode_attention_dispatch(pipelines, encoder, plan);
        // The packed output projection consumes contiguous context rows too.
        readback.push((
            query,
            layout.context + packed_start * shape.query_features * 2,
            plan.kind,
        ));
        packed_start += inputs.tokens as u64;
    }
    encoder.end_encoding();
    command.commit();
    command.wait_until_completed();
    assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
    let values = read::<f16>(&scratch, scratch.length() as usize / 2);
    participants
        .iter()
        .zip(readback)
        .enumerate()
        .map(
            |(participant, ((inputs, _, _), (query, context, dispatch)))| {
                let elements = inputs.tokens * inputs.heads * inputs.dim;
                let error = if pipelines.kv_type == ElementType::I8 {
                    let offset = (binding_layout.offset(participant).unwrap()
                        + pipelines.error_flag_offset().unwrap())
                        as usize
                        / 4;
                    read::<u32>(&arguments, offset + 1)[offset]
                } else {
                    0
                };
                Output {
                    query: values[query as usize / 2..query as usize / 2 + elements].to_vec(),
                    attention: values[context as usize / 2..context as usize / 2 + elements]
                        .to_vec(),
                    error,
                    dispatch: Some(dispatch),
                }
            },
        )
        .collect()
}

fn check_reference(
    inputs: &Inputs,
    start: usize,
    state: &PagedKv,
    expected_query: &[f16],
    output: &Output,
) {
    let values = state.decoded(inputs);
    let load = |position, kind, head, dim| {
        values[((position * 2 + kind) * inputs.kv_heads + head) * inputs.dim + dim]
    };
    for token in 0..inputs.tokens {
        for head in 0..inputs.heads {
            let kv_head = head / (inputs.heads / inputs.kv_heads);
            let base = (token * inputs.heads + head) * inputs.dim;
            let scores = (0..=start + token)
                .map(|position| {
                    (0..inputs.dim)
                        .map(|dim| {
                            expected_query[base + dim].to_f32() * load(position, 0, kv_head, dim)
                        })
                        .sum::<f32>()
                        / (inputs.dim as f32).sqrt()
                })
                .collect::<Vec<_>>();
            let maximum = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let probabilities = scores
                .iter()
                .map(|score| (score - maximum).exp())
                .collect::<Vec<_>>();
            let sum = probabilities.iter().sum::<f32>();
            for dim in 0..inputs.dim {
                let mut expected = probabilities
                    .iter()
                    .enumerate()
                    .map(|(position, probability)| probability * load(position, 1, kv_head, dim))
                    .sum::<f32>()
                    / sum;
                if inputs.gate {
                    let gate = inputs.query[token * inputs.heads * inputs.dim * 2
                        + head * inputs.dim * 2
                        + inputs.dim
                        + dim]
                        .to_f32();
                    expected *= 1.0 / (1.0 + (-gate).exp());
                }
                assert!(
                    (output.attention[base + dim].to_f32() - expected).abs() <= 0.001,
                    "packed {:?} attention differs from Rust reference",
                    state.dtype
                );
            }
        }
    }
}

fn packed_partial_head_case(dtype: ElementType) {
    let device = Device::system_default().expect("packed attention conformance requires Metal");
    let queue = device.new_command_queue();
    let pipelines = if dtype == ElementType::I8 {
        MetalCausalAttentionPipelines::new_int8(&device)
    } else {
        MetalCausalAttentionPipelines::new(&device)
    }
    .unwrap();
    let prefixes = [
        Inputs::new(3, 2, 1, 34, true),
        Inputs::new(7, 2, 1, 34, true),
    ];
    let suffixes = [
        Inputs::new(1, 2, 1, 34, true),
        Inputs::new(2, 2, 1, 34, true),
    ];
    let states = prefixes
        .iter()
        .zip(&suffixes)
        .map(|(prefix, suffix)| PagedKv::new(&device, prefix.tokens + suffix.tokens, prefix, dtype))
        .collect::<Vec<_>>();
    let expected_states = prefixes
        .iter()
        .zip(&suffixes)
        .map(|(prefix, suffix)| PagedKv::new(&device, prefix.tokens + suffix.tokens, prefix, dtype))
        .collect::<Vec<_>>();
    let mut expected = Vec::new();
    for index in 0..prefixes.len() {
        for state in [&states[index], &expected_states[index]] {
            run_packed_projected(&device, &queue, &pipelines, &[(&prefixes[index], 0, state)]);
        }
        expected.push(
            run_packed_projected(
                &device,
                &queue,
                &pipelines,
                &[(
                    &suffixes[index],
                    prefixes[index].tokens,
                    &expected_states[index],
                )],
            )
            .remove(0),
        );
    }
    let participants = suffixes
        .iter()
        .zip(&prefixes)
        .zip(&states)
        .map(|((suffix, prefix), state)| (suffix, prefix.tokens, state))
        .collect::<Vec<_>>();
    let actual = run_packed_projected(&device, &queue, &pipelines, &participants);
    for index in 0..participants.len() {
        assert_eq!(actual[index].error, 0);
        assert_eq!(
            actual[index].query, expected[index].query,
            "participant {index} prepared Q must use its own packed row"
        );
        let actual_state = states[index].snapshot();
        let expected_state = expected_states[index].snapshot();
        assert_eq!(actual_state.len(), expected_state.len());
        let mismatch = actual_state
            .iter()
            .zip(&expected_state)
            .position(|(actual, expected)| actual != expected);
        assert!(mismatch.is_none(), "participant {index} must preserve its own absolute KV history and page slack; first differing byte {mismatch:?}");
        check_reference(
            &suffixes[index],
            prefixes[index].tokens,
            &expected_states[index],
            &expected[index].query,
            &actual[index],
        );
    }
}

#[test]
fn packed_partial_head_f16_preserves_contiguous_rows_and_independent_histories() {
    packed_partial_head_case(ElementType::F16);
}

#[test]
fn packed_partial_head_int8_preserves_contiguous_rows_and_independent_histories() {
    packed_partial_head_case(ElementType::I8);
}
