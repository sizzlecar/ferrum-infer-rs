//! Production launchers checked at the F32 hidden/F16 activation and KV boundaries.
use super::super::test_support::Guarded;
use super::*;
use crate::backend::cuda::vnext_ops::cuda_vnext_runtime_config;
use ferrum_interfaces::vnext::DeviceId;
use half::f16;

fn runtime() -> CudaDeviceRuntime {
    CudaDeviceRuntime::new(
        cuda_vnext_runtime_config(
            0,
            DeviceId::new("device.test.causal-master").unwrap(),
            AttentionExecutionPolicy::default(),
        )
        .unwrap(),
    )
    .expect("causal numerical conformance requires actual CUDA")
}

fn sample(index: usize, salt: usize) -> f16 {
    f16::from_f32((((index * 13 + salt * 7) % 41) as f32 - 20.0) * 0.03125)
}

fn close(actual: f16, expected: f64, stage: &str, index: usize) {
    let bound = expected.abs().max(1.0) * 0.001 + 1.0e-5;
    assert!(
        actual.is_finite() && (actual.to_f64() - expected).abs() <= bound,
        "{stage}[{index}]: actual={}, F64 reference={expected}, bound={bound}",
        actual.to_f32()
    );
}

#[test]
#[ignore = "requires an actual CUDA device"]
fn causal_master_preserves_f32_hidden_and_both_residual_alias_modes_on_cuda() {
    let runtime = runtime();
    let provider = CudaCausalPagedAttentionProvider::new_f32_master(
        &runtime,
        runtime.attention_execution_policy(),
    )
    .unwrap();
    assert_eq!(
        provider.descriptor.operation_id().as_str(),
        "operation.causal_paged_attention.f32-master"
    );
    let stream = runtime.context().default_stream();
    for hidden in [3_usize, 33, 256] {
        let rows = 3;
        let input = (0..rows * hidden)
            .map(|i| 1.0001 + sample(i, 3).to_f32())
            .collect::<Vec<_>>();
        let weight = (0..hidden)
            .map(|i| f16::from_f32(1.0 + sample(i, 5).to_f32()))
            .collect::<Vec<_>>();
        let branch = (0..input.len())
            .map(|i| if i % 2 == 0 { f16::ZERO } else { sample(i, 9) })
            .collect::<Vec<_>>();
        let x = Guarded::new(&stream, &input, 123.0_f32);
        let w = Guarded::new(&stream, &weight, f16::from_f32(124.0));
        let b = Guarded::new(&stream, &branch, f16::from_f32(125.0));
        let normalized = Guarded::new(&stream, &vec![f16::ZERO; input.len()], f16::from_f32(126.0));
        launch_rms_norm(
            &stream,
            &provider.functions.rms_norm,
            x.pointer(&stream),
            w.pointer(&stream),
            normalized.pointer(&stream),
            rows as u64,
            hidden as i32,
            1.0e-6,
        )
        .unwrap();
        let actual = normalized.read(&stream);
        for (row, values) in input.chunks_exact(hidden).enumerate() {
            let inv = (values.iter().map(|&x| f64::from(x).powi(2)).sum::<f64>() / hidden as f64
                + 1.0e-6)
                .sqrt()
                .recip();
            for column in 0..hidden {
                close(
                    actual[row * hidden + column],
                    f64::from(values[column]) * inv * weight[column].to_f64(),
                    "input norm",
                    row * hidden + column,
                );
            }
        }
        x.assert_unchanged(&stream);
        let output = Guarded::new(&stream, &vec![0.0_f32; input.len()], 127.0);
        let expected = input
            .iter()
            .zip(&branch)
            .map(|(a, b)| *a + b.to_f32())
            .collect::<Vec<_>>();
        for alias in [false, true] {
            let destination = if alias { &x } else { &output };
            launch_residual(
                &stream,
                &provider.functions.residual_add,
                provider.functions.residual_add_inplace.as_ref(),
                x.pointer(&stream),
                b.pointer(&stream),
                destination.pointer(&stream),
                input.len() as u64,
            )
            .unwrap();
            assert_eq!(
                destination.read(&stream),
                expected,
                "residual alias={alias}"
            );
            if !alias {
                x.assert_unchanged(&stream);
            }
        }
        w.assert_unchanged(&stream);
        b.assert_unchanged(&stream);
    }
}

fn shape(width: u64, gate: bool) -> CausalAttentionShape {
    let mut attributes = tests::attributes(gate);
    for (name, value) in [
        ("head_dim", width),
        ("query_heads", 6),
        ("key_value_heads", 2),
        ("hidden_size", 256),
        ("query_features", 6 * width),
        (
            "query_projection_features",
            6 * width * if gate { 2 } else { 1 },
        ),
        ("kv_features", 2 * width),
        ("rope_dim", if width == 6 { 4 } else { width / 2 }),
    ] {
        attributes.insert(
            AttributeId::new(name).unwrap(),
            SemanticValue::Unsigned(value),
        );
    }
    CausalAttentionShape::from_attributes(&attributes).unwrap()
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
        table_entries_i32: 2,
        replay_topology: CausalAttentionReplayTopology::new(
            shape,
            CausalAttentionKernelPath::TokenMajorFallback,
            (position + tokens) as u64,
        )
        .unwrap(),
        path: CausalAttentionKernelPath::TokenMajorFallback,
    }
}

fn exercise(
    stream: &Arc<CudaStream>,
    functions: &CausalAttentionFunctions,
    shape: CausalAttentionShape,
    prefix: usize,
    batches: &[usize],
) -> (Vec<f16>, Vec<f16>) {
    let width = shape.head_dim as usize;
    let qheads = shape.query_heads as usize;
    let kvheads = shape.key_value_heads as usize;
    let qstride = shape.query_projection_features as usize;
    let kvstride = shape.kv_features as usize;
    let page_elements = VNEXT_KV_PAGE_BYTES as usize / 2;
    let unused = f16::from_f32(-73.0);
    let mut initial = vec![unused; 2 * page_elements];
    for (i, value) in initial[..prefix * 2 * kvstride].iter_mut().enumerate() {
        *value = sample(i, 11);
    }
    // Independent physical pages are deliberately permuted; page 1 remains idle.
    let pages = [
        Guarded::new(stream, &initial[page_elements..], f16::from_f32(81.0)),
        Guarded::new(stream, &vec![unused; page_elements], f16::from_f32(82.0)),
        Guarded::new(stream, &initial[..page_elements], f16::from_f32(83.0)),
    ];
    let table = Guarded::new(
        stream,
        &[pages[2].pointer(stream), pages[0].pointer(stream)],
        0xDEAD_u64,
    );
    let norm = (0..width)
        .map(|i| f16::from_f32(0.75 + sample(i, 5).to_f32() * 0.25))
        .collect::<Vec<_>>();
    let norm_gpu = Guarded::new(stream, &norm, f16::from_f32(84.0));
    let mut position = prefix;
    let mut all_output = Vec::new();
    let mut previous = initial;
    for &tokens in batches {
        let raw = |stride, salt| {
            (position * stride..(position + tokens) * stride)
                .map(|i| sample(i, salt))
                .collect::<Vec<_>>()
        };
        let q = raw(qstride, 3);
        let k = raw(kvstride, 7);
        let v = raw(kvstride, 13);
        let q_gpu = Guarded::new(stream, &q, f16::from_f32(85.0));
        let k_gpu = Guarded::new(stream, &k, f16::from_f32(86.0));
        let v_gpu = Guarded::new(stream, &v, f16::from_f32(87.0));
        let control = Guarded::new(
            stream,
            &[
                2_i32,
                position as i32,
                tokens as i32,
                (position + tokens) as i32,
                0,
                0,
            ],
            -111_i32,
        );
        let query = Guarded::new(
            stream,
            &vec![f16::ZERO; tokens * qheads * width],
            f16::from_f32(88.0),
        );
        let output = Guarded::new(
            stream,
            &vec![f16::ZERO; tokens * qheads * width],
            f16::from_f32(89.0),
        );
        let launch = launch(shape, position, tokens);
        let cuda = shape.cuda_shape().unwrap();
        launch_prepare(
            stream,
            &functions.prepare,
            q_gpu.pointer(stream),
            k_gpu.pointer(stream),
            v_gpu.pointer(stream),
            norm_gpu.pointer(stream),
            norm_gpu.pointer(stream),
            query.pointer(stream),
            control.pointer(stream),
            table.pointer(stream),
            launch,
            cuda,
            0,
            None,
        )
        .unwrap();
        let query_host = query.read(stream);
        let current = [pages[2].read(stream), pages[0].read(stream)].concat();
        let written = position * 2 * kvstride..(position + tokens) * 2 * kvstride;
        for (i, (&actual, &before)) in current.iter().zip(&previous).enumerate() {
            if !written.contains(&i) {
                assert_eq!(actual, before, "unowned KV element {i}");
            }
        }
        for token in 0..tokens {
            for head in 0..qheads {
                let offset = token * qstride + head * width * if shape.output_gate { 2 } else { 1 };
                let expected = rotary_tests::reference(
                    &q[offset..offset + width],
                    &norm,
                    shape,
                    position + token,
                    false,
                );
                for (dim, value) in expected.into_iter().enumerate() {
                    close(
                        query_host[(token * qheads + head) * width + dim],
                        value,
                        "prepared query",
                        dim,
                    );
                }
            }
            for head in 0..kvheads {
                let offset = token * kvstride + head * width;
                let expected = rotary_tests::reference(
                    &k[offset..offset + width],
                    &norm,
                    shape,
                    position + token,
                    false,
                );
                let start = (position + token) * 2 * kvstride + head * width;
                for dim in 0..width {
                    close(current[start + dim], expected[dim], "stored key", dim);
                    assert_eq!(
                        current[start + kvstride + dim],
                        v[offset + dim],
                        "stored value"
                    );
                }
            }
        }
        launch_fallback_attention(
            stream,
            functions,
            query.pointer(stream),
            q_gpu.pointer(stream),
            control.pointer(stream),
            table.pointer(stream),
            output.pointer(stream),
            launch,
            cuda,
            0,
            None,
        )
        .unwrap();
        let attention = output.read(stream);
        // Preparation was independently checked above. Evaluate softmax in F64
        // over the admitted F16 stores to isolate attention's arithmetic error.
        for token in 0..tokens {
            for head in 0..qheads {
                let kvhead = head / (qheads / kvheads);
                let query = &query_host[(token * qheads + head) * width..][..width];
                let logits = (0..=position + token)
                    .map(|key_position| {
                        let key = &current[key_position * 2 * kvstride + kvhead * width..][..width];
                        query
                            .iter()
                            .zip(key)
                            .map(|(q, k)| q.to_f64() * k.to_f64())
                            .sum::<f64>()
                            * f64::from(cuda.attention_scale)
                    })
                    .collect::<Vec<_>>();
                let max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let scores = logits.iter().map(|x| (x - max).exp()).collect::<Vec<_>>();
                let denominator: f64 = scores.iter().sum();
                for dim in 0..width {
                    let expected = scores
                        .iter()
                        .enumerate()
                        .map(|(p, s)| {
                            s * current[p * 2 * kvstride + kvstride + kvhead * width + dim].to_f64()
                        })
                        .sum::<f64>()
                        / denominator;
                    let index = (token * qheads + head) * width + dim;
                    close(attention[index], expected, "attention", index);
                }
            }
        }
        if shape.output_gate {
            launch_attention_gate(
                stream,
                &functions.attention_gate,
                output.pointer(stream),
                q_gpu.pointer(stream),
                launch,
                cuda,
            )
            .unwrap();
            let gated = output.read(stream);
            for token in 0..tokens {
                for head in 0..qheads {
                    for dim in 0..width {
                        let index = (token * qheads + head) * width + dim;
                        let gate = q[token * qstride + head * 2 * width + width + dim].to_f64();
                        close(
                            gated[index],
                            attention[index].to_f64() / (1.0 + (-gate).exp()),
                            "output gate",
                            index,
                        );
                    }
                }
            }
        }
        all_output.extend(output.read(stream));
        for immutable in [&q_gpu, &k_gpu, &v_gpu, &norm_gpu] {
            immutable.assert_unchanged(stream);
        }
        control.assert_unchanged(stream);
        table.assert_unchanged(stream);
        pages[1].assert_unchanged(stream);
        previous = current;
        position += tokens;
    }
    (all_output, previous)
}

#[test]
#[ignore = "requires an actual CUDA device"]
fn causal_kv_carry_crosses_physical_pages_and_matches_f64_attention_on_cuda() {
    let runtime = runtime();
    let stream = runtime.context().default_stream();
    for provider in [
        CudaCausalPagedAttentionProvider::new_f32_master(
            &runtime,
            runtime.attention_execution_policy(),
        )
        .unwrap(),
        CudaCausalPagedAttentionProvider::new(&runtime, runtime.attention_execution_policy())
            .unwrap(),
    ] {
        for (width, prefix) in [(6, 3), (128, 62)] {
            for gate in [false, true] {
                let shape = shape(width, gate);
                let split = exercise(&stream, &provider.functions, shape, prefix, &[3, 1]);
                let whole = exercise(&stream, &provider.functions, shape, prefix, &[4]);
                assert_eq!(
                    split, whole,
                    "prefill/decode outputs and KV state differ: {shape:?}"
                );
            }
        }
    }
}
