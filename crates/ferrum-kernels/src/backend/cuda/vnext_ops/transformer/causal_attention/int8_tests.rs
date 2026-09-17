//! Exercise the actual portable launchers, including independent payload/scale
//! page boundaries. These tests require CUDA; host geometry checks do not.
use super::super::test_support::Guarded;
use super::*;
use crate::backend::cuda::vnext_ops::cuda_vnext_runtime_config;
use ferrum_interfaces::vnext::DeviceId;
use half::f16;

fn shape(width: u64) -> CausalAttentionShape {
    let mut attributes = tests::attributes(false);
    for (name, value) in [
        ("head_dim", width),
        ("query_heads", 6),
        ("key_value_heads", 2),
        ("hidden_size", 256),
        ("query_features", 6 * width),
        ("query_projection_features", 6 * width),
        ("kv_features", 2 * width),
        ("rope_dim", if width == 6 { 4 } else { width / 2 }),
        ("maximum_context_tokens", 8192),
    ] {
        attributes.insert(
            AttributeId::new(name).unwrap(),
            SemanticValue::Unsigned(value),
        );
    }
    CausalAttentionShape::from_attributes_for(&attributes, CausalAttentionSemantics::StandardInt8)
        .unwrap()
}

fn runtime() -> CudaDeviceRuntime {
    CudaDeviceRuntime::new(
        cuda_vnext_runtime_config(
            0,
            DeviceId::new("device.test.int8-kv").unwrap(),
            AttentionExecutionPolicy::Portable,
        )
        .unwrap(),
    )
    .expect("INT8 KV tests require an actual CUDA device")
}

fn sample(index: usize, salt: usize) -> f16 {
    f16::from_f32((((index * 13 + salt * 7) % 41) as f32 - 20.0) * 0.03125)
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

fn launch(shape: CausalAttentionShape, position: usize, tokens: usize) -> CausalAttentionLaunch {
    CausalAttentionLaunch {
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
        table_entries_i32: shape.table_entries((position + tokens) as u64).unwrap() as i32,
        replay_topology: CausalAttentionReplayTopology::new(
            shape,
            CausalAttentionKernelPath::TokenMajorFallback,
            (position + tokens) as u64,
        )
        .unwrap(),
        path: CausalAttentionKernelPath::TokenMajorFallback,
    }
}

#[test]
fn int8_kv_geometry_counts_independent_scales_and_rejects_native() {
    let contract_ids = |policy| {
        crate::backend::cuda::vnext_ops::cuda_operation_contracts(policy)
            .unwrap()
            .iter()
            .map(|contract| contract.descriptor().id.to_string())
            .collect::<std::collections::BTreeSet<_>>()
    };
    let native = contract_ids(AttentionExecutionPolicy::NativeAdaptive);
    let portable = contract_ids(AttentionExecutionPolicy::Portable);
    assert!(native.is_subset(&portable));
    assert_eq!(
        portable
            .difference(&native)
            .cloned()
            .collect::<std::collections::BTreeSet<_>>(),
        std::collections::BTreeSet::from([
            ferrum_interfaces::vnext::CAUSAL_PAGED_ATTENTION_INT8_KV_OPERATION_ID.to_owned(),
            ferrum_interfaces::vnext::CAUSAL_PAGED_ATTENTION_F32_MASTER_INT8_KV_OPERATION_ID
                .to_owned(),
        ])
    );
    let shape = shape(128);
    assert_eq!(shape.kv_layout().unwrap(), CausalKvLayout::TokenMajorPages);
    assert_eq!(shape.state_bytes_per_token().unwrap(), 512);
    assert_eq!(
        shape.physical_state_bytes(128).unwrap(),
        VNEXT_KV_PAGE_BYTES
    );
    assert_eq!(
        shape.physical_state_bytes(129).unwrap(),
        2 * VNEXT_KV_PAGE_BYTES
    );
    assert_eq!(shape.scale_state_bytes(4096).unwrap(), VNEXT_KV_PAGE_BYTES);
    assert_eq!(
        shape.scale_state_bytes(4097).unwrap(),
        2 * VNEXT_KV_PAGE_BYTES
    );
    assert!(shape.scale_state_bytes(0).is_err());
    assert!(shape.scale_state_bytes(8193).is_err());
    assert!(CausalAttentionKernelPath::select(
        AttentionExecutionPolicy::NativeAdaptive,
        shape,
        1,
        128
    )
    .is_err());
    assert!(
        CausalAttentionKernelPath::select(AttentionExecutionPolicy::Auto, shape, 1, 128).is_err()
    );
    assert_eq!(
        CausalAttentionKernelPath::select(AttentionExecutionPolicy::Portable, shape, 1, 128)
            .unwrap(),
        CausalAttentionKernelPath::TokenMajorFallback
    );
    let bindings = storage_bindings(CausalAttentionSemantics::StandardInt8).unwrap();
    assert_eq!(bindings.len(), 11);
    assert!(
        shape.binding_slot_bytes().unwrap()
            >= BINDING_CONTROL_BYTES + (shape.maximum_pages().unwrap() + 2) * POINTER_BYTES
    );
    let mut overflowing = shape;
    overflowing.key_value_heads = u64::MAX;
    assert!(overflowing.scale_state_bytes(2).is_err());
}

fn exercise(
    runtime: &CudaDeviceRuntime,
    width: usize,
    prefix: usize,
    chunks: &[usize],
) -> (Vec<f16>, Vec<i8>, Vec<f32>) {
    let stream = runtime.context().default_stream();
    let quantized =
        CudaCausalPagedAttentionProvider::new_int8_kv(runtime, AttentionExecutionPolicy::Portable)
            .unwrap();
    let reference =
        CudaCausalPagedAttentionProvider::new(runtime, AttentionExecutionPolicy::Portable).unwrap();
    let shape = shape(width as u64);
    let end = prefix + chunks.iter().sum::<usize>();
    let kv_heads = shape.key_value_heads as usize;
    let kv_stride = kv_heads * width;
    let query_stride = shape.query_features as usize;
    let payload_pages =
        shape.physical_state_bytes(end as u64).unwrap() as usize / VNEXT_KV_PAGE_BYTES as usize;
    let scale_pages =
        shape.scale_state_bytes(end as u64).unwrap() as usize / VNEXT_KV_PAGE_BYTES as usize;
    let f16_page_elements = VNEXT_KV_PAGE_BYTES as usize / 2;
    let f16_pages = (end * 2 * kv_stride).div_ceil(f16_page_elements);
    let mut initial_f16 = vec![f16::from_f32(-70.0); f16_pages * f16_page_elements];
    for (index, value) in initial_f16[..prefix * 2 * kv_stride].iter_mut().enumerate() {
        *value = sample(index, 11);
    }
    let mut initial_quant = vec![77_i8; payload_pages * VNEXT_KV_PAGE_BYTES as usize];
    let mut initial_scales = vec![-71.0_f32; scale_pages * VNEXT_KV_PAGE_BYTES as usize / 4];
    for (row, values) in initial_f16[..prefix * 2 * kv_stride]
        .chunks_exact(width)
        .enumerate()
    {
        let (q, scale) = quantize(values);
        initial_quant[row * width..(row + 1) * width].copy_from_slice(&q);
        initial_scales[row] = scale;
    }
    // Reverse physical allocation order so the logical page table is essential.
    let payload = initial_quant
        .chunks_exact(VNEXT_KV_PAGE_BYTES as usize)
        .rev()
        .map(|page| Guarded::new(&stream, page, -122_i8))
        .collect::<Vec<_>>();
    let scales = initial_scales
        .chunks_exact(VNEXT_KV_PAGE_BYTES as usize / 4)
        .rev()
        .map(|page| Guarded::new(&stream, page, -123.0_f32))
        .collect::<Vec<_>>();
    let fp16 = initial_f16
        .chunks_exact(f16_page_elements)
        .rev()
        .map(|page| Guarded::new(&stream, page, f16::from_f32(-124.0)))
        .collect::<Vec<_>>();
    let tables = payload
        .iter()
        .rev()
        .map(|page| page.pointer(&stream))
        .chain(scales.iter().rev().map(|page| page.pointer(&stream)))
        .collect::<Vec<_>>();
    let tables = Guarded::new(&stream, &tables, 0xDEAD_u64);
    let fp16_tables = Guarded::new(
        &stream,
        &fp16
            .iter()
            .rev()
            .map(|page| page.pointer(&stream))
            .collect::<Vec<_>>(),
        0xBEEF_u64,
    );
    let norm = Guarded::new(&stream, &vec![f16::ONE; width], f16::from_f32(-125.0));
    let mut position = prefix;
    let mut outputs = Vec::new();
    for &tokens in chunks {
        let raw = |stride, salt| {
            (position * stride..(position + tokens) * stride)
                .map(|index| sample(index, salt))
                .collect::<Vec<_>>()
        };
        let q = Guarded::new(&stream, &raw(query_stride, 3), f16::from_f32(-126.0));
        let k = Guarded::new(&stream, &raw(kv_stride, 7), f16::from_f32(-127.0));
        let mut values = raw(kv_stride, 13);
        // Exact ties, all-zero rows, subnormals and large finite values are
        // selected by absolute position so changing the chunking changes no data.
        for (row, values) in values.chunks_exact_mut(width).enumerate() {
            match (position * kv_heads + row) % 4 {
                0 => {
                    values.fill(f16::ZERO);
                }
                1 => {
                    values.fill(f16::from_bits(1));
                }
                2 => {
                    for (d, value) in values.iter_mut().enumerate() {
                        *value = f16::from_f32([127.0, 0.5, 1.5, -0.5, -1.5, -127.0][d % 6]);
                    }
                }
                _ => {
                    values[0] = f16::MAX;
                }
            }
        }
        let v = Guarded::new(&stream, &values, f16::from_f32(-128.0));
        let control = Guarded::new(
            &stream,
            &[
                payload_pages as i32,
                position as i32,
                tokens as i32,
                (position + tokens) as i32,
                0,
                0,
            ],
            -333_i32,
        );
        let fp16_control = Guarded::new(
            &stream,
            &[
                f16_pages as i32,
                position as i32,
                tokens as i32,
                (position + tokens) as i32,
                0,
                0,
            ],
            -334_i32,
        );
        let query = Guarded::new(
            &stream,
            &vec![f16::ZERO; tokens * query_stride],
            f16::from_f32(-129.0),
        );
        let fp16_query = Guarded::new(
            &stream,
            &vec![f16::ZERO; tokens * query_stride],
            f16::from_f32(-130.0),
        );
        let output = Guarded::new(
            &stream,
            &vec![f16::ZERO; tokens * query_stride],
            f16::from_f32(-131.0),
        );
        let selected = launch(shape, position, tokens);
        let mut plain = shape;
        plain.int8_kv = false;
        for (function, target_query, ctrl, table, logical) in [
            (
                &quantized.functions.prepare,
                &query,
                &control,
                &tables,
                shape,
            ),
            (
                &reference.functions.prepare,
                &fp16_query,
                &fp16_control,
                &fp16_tables,
                plain,
            ),
        ] {
            launch_prepare(
                &stream,
                function,
                q.pointer(&stream),
                k.pointer(&stream),
                v.pointer(&stream),
                norm.pointer(&stream),
                norm.pointer(&stream),
                target_query.pointer(&stream),
                ctrl.pointer(&stream),
                table.pointer(&stream),
                selected,
                logical.cuda_shape().unwrap(),
                0,
                None,
            )
            .unwrap();
        }
        let actual_query = query.read(&stream);
        assert_eq!(
            actual_query,
            fp16_query.read(&stream),
            "INT8 changed Q preparation"
        );
        let actual_payload = payload
            .iter()
            .rev()
            .flat_map(|page| page.read(&stream))
            .collect::<Vec<_>>();
        let actual_scales = scales
            .iter()
            .rev()
            .flat_map(|page| page.read(&stream))
            .collect::<Vec<_>>();
        let actual_fp16 = fp16
            .iter()
            .rev()
            .flat_map(|page| page.read(&stream))
            .collect::<Vec<_>>();
        for row in position * 2 * kv_heads..(position + tokens) * 2 * kv_heads {
            let (expected, scale) = quantize(&actual_fp16[row * width..(row + 1) * width]);
            assert_eq!(
                &actual_payload[row * width..(row + 1) * width],
                expected,
                "row {row}"
            );
            assert_eq!(actual_scales[row], scale, "scale row {row}");
        }
        assert_eq!(
            &actual_payload[(position + tokens) * 2 * kv_stride..],
            &initial_quant[(position + tokens) * 2 * kv_stride..]
        );
        assert_eq!(
            &actual_scales[(position + tokens) * 2 * kv_heads..],
            &initial_scales[(position + tokens) * 2 * kv_heads..]
        );
        control.assert_unchanged(&stream);
        launch_fallback_attention(
            &stream,
            &quantized.functions,
            query.pointer(&stream),
            q.pointer(&stream),
            control.pointer(&stream),
            tables.pointer(&stream),
            output.pointer(&stream),
            selected,
            shape.cuda_shape().unwrap(),
            0,
            None,
        )
        .unwrap();
        let actual = output.read(&stream);
        for token in 0..tokens {
            for head in 0..shape.query_heads as usize {
                let kv_head = head / (shape.query_heads as usize / kv_heads);
                let row_start = (token * shape.query_heads as usize + head) * width;
                let logits = (0..=position + token)
                    .map(|history| {
                        let row = history * 2 * kv_heads + kv_head;
                        (0..width)
                            .map(|d| {
                                actual_query[row_start + d].to_f64()
                                    * f64::from(actual_payload[row * width + d])
                                    * f64::from(actual_scales[row])
                            })
                            .sum::<f64>()
                            * f64::from(shape.attention_scale)
                    })
                    .collect::<Vec<_>>();
                let maximum = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let probabilities = logits
                    .iter()
                    .map(|score| (score - maximum).exp())
                    .collect::<Vec<_>>();
                let denominator = probabilities.iter().sum::<f64>();
                for d in 0..width {
                    let expected = probabilities
                        .iter()
                        .enumerate()
                        .map(|(history, p)| {
                            let row = (history * 2 + 1) * kv_heads + kv_head;
                            p * f64::from(actual_payload[row * width + d])
                                * f64::from(actual_scales[row])
                        })
                        .sum::<f64>()
                        / denominator;
                    let observed = actual[row_start + d].to_f64();
                    assert!(
                        (observed - expected).abs() <= expected.abs().max(1.0) * 0.003,
                        "INT8 attention row {row_start} dim {d}: {observed} vs {expected}"
                    );
                }
            }
        }
        outputs.extend(actual);
        position += tokens;
    }
    (
        outputs,
        payload
            .iter()
            .rev()
            .flat_map(|page| page.read(&stream))
            .collect(),
        scales
            .iter()
            .rev()
            .flat_map(|page| page.read(&stream))
            .collect(),
    )
}

#[test]
#[ignore = "requires an actual CUDA device"]
fn int8_kv_prepare_and_attention_cross_independent_pages_on_cuda() {
    let runtime = runtime();
    for (width, prefix) in [(128, 127), (6, 4094)] {
        let split = exercise(&runtime, width, prefix, &[3, 1]);
        let whole = exercise(&runtime, width, prefix, &[4]);
        assert_eq!(
            split, whole,
            "INT8 chunking changed the stored state or continuation"
        );
    }
}

#[test]
#[ignore = "requires an actual CUDA device"]
fn int8_kv_nonfinite_write_sets_step_status_without_a_device_trap() {
    let runtime = runtime();
    let stream = runtime.context().default_stream();
    let provider =
        CudaCausalPagedAttentionProvider::new_int8_kv(&runtime, AttentionExecutionPolicy::Portable)
            .unwrap();
    let shape = shape(6);
    let q = Guarded::new(
        &stream,
        &vec![f16::ONE; shape.query_features as usize],
        f16::ZERO,
    );
    let k = Guarded::new(
        &stream,
        &vec![f16::ONE; shape.kv_features as usize],
        f16::ZERO,
    );
    let mut invalid = vec![f16::ZERO; shape.kv_features as usize];
    invalid[0] = f16::INFINITY;
    let v = Guarded::new(&stream, &invalid, f16::ZERO);
    let norm = Guarded::new(&stream, &vec![f16::ONE; 6], f16::ZERO);
    let payload = Guarded::new(&stream, &vec![0_i8; VNEXT_KV_PAGE_BYTES as usize], -127);
    let scales = Guarded::new(
        &stream,
        &vec![0.0_f32; VNEXT_KV_PAGE_BYTES as usize / 4],
        -123.0,
    );
    let tables = Guarded::new(
        &stream,
        &[payload.pointer(&stream), scales.pointer(&stream)],
        0_u64,
    );
    let control = Guarded::new(&stream, &[1_i32, 0, 1, 1, 0, 0], -123);
    let query = Guarded::new(
        &stream,
        &vec![f16::ZERO; shape.query_features as usize],
        f16::ZERO,
    );
    launch_prepare(
        &stream,
        &provider.functions.prepare,
        q.pointer(&stream),
        k.pointer(&stream),
        v.pointer(&stream),
        norm.pointer(&stream),
        norm.pointer(&stream),
        query.pointer(&stream),
        control.pointer(&stream),
        tables.pointer(&stream),
        launch(shape, 0, 1),
        shape.cuda_shape().unwrap(),
        0,
        None,
    )
    .unwrap();
    let status = control.read(&stream);
    assert_eq!(&status[..5], &[1, 0, 1, 1, 0]);
    assert_eq!(status[5] as u32, INT8_NUMERICAL_FAILURE_MASK);
    // The context is still usable after the numerical failure.
    let later = stream.clone_htod(&[123_u32]).unwrap();
    assert_eq!(stream.clone_dtoh(&later).unwrap(), [123]);
}

fn packed_case(
    runtime: &CudaDeviceRuntime,
    heads: u64,
    kv_heads: u64,
    packed: bool,
) -> (Vec<f16>, Vec<Vec<i8>>, Vec<Vec<f32>>) {
    let stream = runtime.context().default_stream();
    let provider =
        CudaCausalPagedAttentionProvider::new_int8_kv(runtime, AttentionExecutionPolicy::Portable)
            .unwrap();
    let mut shape = shape(128);
    shape.query_heads = heads;
    shape.key_value_heads = kv_heads;
    shape.query_features = heads * shape.head_dim;
    shape.query_projection_features = shape.query_features;
    shape.kv_features = kv_heads * shape.head_dim;
    let rows = [(127_usize, 2_usize, 0_usize), (3, 1, 2)];
    let payload = rows
        .iter()
        .map(|&(start, tokens, _)| {
            (0..shape.physical_state_bytes((start + tokens) as u64).unwrap() / VNEXT_KV_PAGE_BYTES)
                .map(|page| {
                    Guarded::new(
                        &stream,
                        &(0..VNEXT_KV_PAGE_BYTES as usize)
                            .map(|i| ((i + page as usize) % 23) as i8 - 11)
                            .collect::<Vec<_>>(),
                        -127_i8,
                    )
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let scales = rows
        .iter()
        .map(|&(start, tokens, _)| {
            (0..shape.scale_state_bytes((start + tokens) as u64).unwrap() / VNEXT_KV_PAGE_BYTES)
                .map(|_| {
                    Guarded::new(
                        &stream,
                        &vec![0.0125_f32; VNEXT_KV_PAGE_BYTES as usize / 4],
                        -127.0,
                    )
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let slot_bytes = shape.binding_slot_bytes().unwrap() as usize;
    let mut host_bindings = vec![0_u8; slot_bytes * rows.len()];
    for (participant, &(start, tokens, packed_start)) in rows.iter().enumerate() {
        let slot = &mut host_bindings[participant * slot_bytes..(participant + 1) * slot_bytes];
        for (word, value) in [
            payload[participant].len() as i32,
            start as i32,
            tokens as i32,
            (start + tokens) as i32,
            packed_start as i32,
            0,
        ]
        .into_iter()
        .enumerate()
        {
            slot[word * 4..word * 4 + 4].copy_from_slice(&value.to_ne_bytes());
        }
        for (entry, pointer) in payload[participant]
            .iter()
            .map(|page| page.pointer(&stream))
            .chain(scales[participant].iter().map(|page| page.pointer(&stream)))
            .enumerate()
        {
            let offset = BINDING_CONTROL_BYTES as usize + entry * 8;
            slot[offset..offset + 8].copy_from_slice(&pointer.to_ne_bytes());
        }
    }
    let bindings = Guarded::new(&stream, &host_bindings, 0xED_u8);
    let queries = Guarded::new(
        &stream,
        &(0..3 * shape.query_features as usize)
            .map(|i| sample(i, 3))
            .collect::<Vec<_>>(),
        f16::ZERO,
    );
    let keys = Guarded::new(
        &stream,
        &(0..3 * shape.kv_features as usize)
            .map(|i| sample(i, 7))
            .collect::<Vec<_>>(),
        f16::ZERO,
    );
    let values = Guarded::new(
        &stream,
        &(0..3 * shape.kv_features as usize)
            .map(|i| sample(i, 13))
            .collect::<Vec<_>>(),
        f16::ZERO,
    );
    let norm = Guarded::new(&stream, &vec![f16::ONE; shape.head_dim as usize], f16::ZERO);
    let query = Guarded::new(
        &stream,
        &vec![f16::ZERO; 3 * shape.query_features as usize],
        f16::ZERO,
    );
    let output = Guarded::new(
        &stream,
        &vec![f16::ZERO; 3 * shape.query_features as usize],
        f16::ZERO,
    );
    let selected = launch(shape, rows[0].0, rows[0].1);
    if packed {
        let grid = Some(PackedFallbackLaunch {
            token_grid: 2,
            packed_token_grid: 3,
            participant_grid: 2,
            participant_count_i32: 2,
            binding_slot_bytes: slot_bytes as u64,
            path: CausalAttentionKernelPath::TokenMajorFallback,
        });
        launch_prepare(
            &stream,
            &provider.functions.prepare,
            queries.pointer(&stream),
            keys.pointer(&stream),
            values.pointer(&stream),
            norm.pointer(&stream),
            norm.pointer(&stream),
            query.pointer(&stream),
            bindings.pointer(&stream),
            bindings.pointer(&stream) + BINDING_CONTROL_BYTES,
            selected,
            shape.cuda_shape().unwrap(),
            0,
            grid,
        )
        .unwrap();
        launch_fallback_attention(
            &stream,
            &provider.functions,
            query.pointer(&stream),
            queries.pointer(&stream),
            bindings.pointer(&stream),
            bindings.pointer(&stream) + BINDING_CONTROL_BYTES,
            output.pointer(&stream),
            selected,
            shape.cuda_shape().unwrap(),
            0,
            grid,
        )
        .unwrap();
    } else {
        for (participant, &(start, tokens, packed_start)) in rows.iter().enumerate() {
            let q_offset = packed_start as u64 * shape.query_features * 2;
            let kv_offset = packed_start as u64 * shape.kv_features * 2;
            let control = bindings.pointer(&stream) + (participant * slot_bytes) as u64;
            let selected = launch(shape, start, tokens);
            launch_prepare(
                &stream,
                &provider.functions.prepare,
                queries.pointer(&stream) + q_offset,
                keys.pointer(&stream) + kv_offset,
                values.pointer(&stream) + kv_offset,
                norm.pointer(&stream),
                norm.pointer(&stream),
                query.pointer(&stream) + q_offset,
                control,
                control + BINDING_CONTROL_BYTES,
                selected,
                shape.cuda_shape().unwrap(),
                0,
                None,
            )
            .unwrap();
            launch_fallback_attention(
                &stream,
                &provider.functions,
                query.pointer(&stream) + q_offset,
                queries.pointer(&stream) + q_offset,
                control,
                control + BINDING_CONTROL_BYTES,
                output.pointer(&stream) + q_offset,
                selected,
                shape.cuda_shape().unwrap(),
                0,
                None,
            )
            .unwrap();
        }
    }
    let result = output.read(&stream);
    bindings.assert_unchanged(&stream);
    (
        result,
        payload
            .iter()
            .map(|pages| pages.iter().flat_map(|page| page.read(&stream)).collect())
            .collect(),
        scales
            .iter()
            .map(|pages| pages.iter().flat_map(|page| page.read(&stream)).collect())
            .collect(),
    )
}

#[test]
#[ignore = "requires an actual CUDA device"]
fn int8_kv_packed_mha_gqa_mqa_preserve_independent_absolute_positions_on_cuda() {
    let runtime = runtime();
    for (heads, kv_heads) in [(2, 2), (6, 2), (6, 1)] {
        assert_eq!(
            packed_case(&runtime, heads, kv_heads, true),
            packed_case(&runtime, heads, kv_heads, false),
            "packed INT8 request isolation differs for heads={heads}, kv_heads={kv_heads}"
        );
    }
}
