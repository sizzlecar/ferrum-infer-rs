//! Candidate kernels versus the unchanged ordinary scan and independent F64 math.
use super::*;

fn bit_eq(a: &[f32], b: &[f32]) {
    assert_eq!(
        a.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        b.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );
}

fn trial(
    stream: &Arc<CudaStream>,
    functions: &AttentionFunctions,
    shape: AttentionShape,
    counts: [usize; 2],
) {
    let width = shape.qkvzba_features as usize;
    let conv_size = shape.conv_state_elements().unwrap() as usize;
    let delta_size = (shape.value_heads * shape.value_head_dim * shape.key_head_dim) as usize;
    let tokens = counts.iter().sum::<usize>();
    let qk = (shape.key_heads * shape.key_head_dim) as usize;
    let values = shape.value_features as usize;
    let heads = shape.value_heads as usize;
    let halves = |n, salt| {
        (0..n)
            .map(|i| f16::from_f32(sample(i, salt, 0.015625)))
            .collect::<Vec<_>>()
    };
    let conv_original = [halves(conv_size, 1), halves(conv_size, 2)];
    let delta_original = std::array::from_fn::<_, 2, _>(|slot| {
        (0..delta_size)
            .map(|i| sample(i, slot + 3, 0.00390625))
            .collect::<Vec<_>>()
    });
    let conv = conv_original
        .iter()
        .map(|x| Guarded::new(stream, x, f16::from_f32(81.0)))
        .collect::<Vec<_>>();
    let delta = delta_original
        .iter()
        .map(|x| Guarded::new(stream, x, 91.0))
        .collect::<Vec<_>>();
    let out_conv = (0..2)
        .map(|_| {
            Guarded::new(
                stream,
                &vec![f16::from_f32(-17.0); 2 * conv_size],
                f16::from_f32(121.0),
            )
        })
        .collect::<Vec<_>>();
    let out_delta = (0..2)
        .map(|_| Guarded::new(stream, &vec![-17.0_f32; 2 * delta_size], 131.0))
        .collect::<Vec<_>>();
    // Deliberately reverse physical owners relative to packed token order.
    let slots = [1, 0];
    let original_bindings = Guarded::new(
        stream,
        &slots
            .iter()
            .flat_map(|&i| [conv[i].pointer(stream), delta[i].pointer(stream)])
            .collect::<Vec<_>>(),
        u64::MAX,
    );
    let candidates = Guarded::new(
        stream,
        &slots
            .iter()
            .flat_map(|&i| [out_conv[i].pointer(stream), out_delta[i].pointer(stream)])
            .collect::<Vec<_>>(),
        u64::MAX,
    );
    let raw = halves(tokens * width, 7);
    let weights = halves(shape.qkv_features as usize * shape.conv_kernel as usize, 9);
    let decay = (0..heads)
        .map(|i| match shape.decay_parameterization {
            GatedDeltaDecayParameterization::LogRate => -0.8 + i as f32 * 0.2,
            GatedDeltaDecayParameterization::NegativeRate => -0.2 - i as f32 * 0.3,
        })
        .collect::<Vec<_>>();
    let bias = (0..heads).map(|i| sample(i, 5, 0.125)).collect::<Vec<_>>();
    let raw_gpu = Guarded::new(stream, &raw, f16::from_f32(41.0));
    let weights_gpu = Guarded::new(stream, &weights, f16::from_f32(71.0));
    let decay_gpu = Guarded::new(stream, &decay, 61.0);
    let bias_gpu = Guarded::new(stream, &bias, 51.0);
    let lengths = Guarded::new(stream, &[0, counts[0] as u32, tokens as u32], u32::MAX);
    let indices = Guarded::new(
        stream,
        &counts
            .iter()
            .enumerate()
            .flat_map(|(i, &n)| std::iter::repeat_n(i as u32, n))
            .collect::<Vec<_>>(),
        u32::MAX,
    );
    let buffer = |n| Guarded::new(stream, &vec![0.0_f32; n], 101.0);
    let q = buffer(tokens * qk);
    let k = buffer(tokens * qk);
    let v = buffer(tokens * values);
    let g = buffer(tokens * heads);
    let beta = buffer(tokens * heads);
    let output = buffer(tokens * values);
    let z = Guarded::new(
        stream,
        &vec![f16::ZERO; tokens * values],
        f16::from_f32(111.0),
    );
    let final_conv = Guarded::new(
        stream,
        &vec![f16::ZERO; 2 * conv_size],
        f16::from_f32(121.0),
    );
    let cuda = shape.cuda_shape().unwrap();
    launch_prepare(AttentionPrepareRequest {
        stream,
        function: functions.prepare_for(shape.decay_parameterization),
        buffers: AttentionPrepareBuffers {
            qkvzba: raw_gpu.pointer(stream),
            conv_weight: weights_gpu.pointer(stream),
            state_binding: original_bindings.pointer(stream),
            a_log: decay_gpu.pointer(stream),
            dt_bias: bias_gpu.pointer(stream),
            cu_seqlens: lengths.pointer(stream),
            token_seq_indices: indices.pointer(stream),
            query: q.pointer(stream),
            key: k.pointer(stream),
            value: v.pointer(stream),
            z: z.pointer(stream),
            g: g.pointer(stream),
            beta: beta.pointer(stream),
            final_conv_state: final_conv.pointer(stream),
        },
        context: AttentionPrepareContext {
            tokens: tokens as u64,
            tokens_i32: tokens as i32,
            batch: 2,
            physical: cuda,
            logical: shape,
        },
    })
    .unwrap();
    candidate_scan::launch_convolution(
        stream,
        &functions.candidates.as_ref().unwrap().convolution,
        raw_gpu.pointer(stream),
        original_bindings.pointer(stream),
        candidates.pointer(stream),
        lengths.pointer(stream),
        2,
        shape,
    )
    .unwrap();
    launch_qk_norm(
        stream,
        &functions.qk_norm,
        q.pointer(stream),
        k.pointer(stream),
        tokens as u64,
        tokens as i32,
        cuda,
    )
    .unwrap();
    launch_delta_with_candidates(
        stream,
        functions,
        q.pointer(stream),
        k.pointer(stream),
        v.pointer(stream),
        g.pointer(stream),
        beta.pointer(stream),
        original_bindings.pointer(stream),
        lengths.pointer(stream),
        output.pointer(stream),
        2,
        tokens as i32,
        cuda,
        Some(candidates.pointer(stream)),
    )
    .unwrap();

    let reference_initial = || Reference {
        conv: slots
            .iter()
            .map(|&i| conv_original[i].iter().map(|x| x.to_f64()).collect())
            .collect(),
        delta: slots
            .iter()
            .map(|&i| delta_original[i].iter().copied().map(f64::from).collect())
            .collect(),
    };
    let mut expected = reference_initial();
    let expected_output = expected.forward(shape, &counts, &raw, &weights, &decay, &bias);
    assert_close(
        &output.read(stream),
        &expected_output.output,
        "candidate F64 hidden",
    );
    let prepared = [
        q.read(stream),
        k.read(stream),
        v.read(stream),
        g.read(stream),
        beta.read(stream),
    ];
    // Re-run only the unchanged recurrence using the SAME prepared operands.
    // Prefix repacking is a test input copy, not an alternative arithmetic path.
    for prefix in [false, true] {
        let scan_counts = counts.map(|n| n - usize::from(prefix));
        let scan_tokens = scan_counts.iter().sum::<usize>();
        let compact = |data: &[f32], columns: usize| {
            let mut cursor = 0;
            let mut result = Vec::new();
            for (&old, &new) in counts.iter().zip(&scan_counts) {
                result.extend_from_slice(&data[cursor * columns..(cursor + new) * columns]);
                cursor += old;
            }
            result
        };
        let operands = prepared
            .iter()
            .zip([qk, qk, values, heads, heads])
            .map(|(data, columns)| Guarded::new(stream, &compact(data, columns), 101.0))
            .collect::<Vec<_>>();
        let ordinary = delta_original
            .iter()
            .map(|x| Guarded::new(stream, x, 91.0))
            .collect::<Vec<_>>();
        let ordinary_bindings = Guarded::new(
            stream,
            &slots
                .iter()
                .flat_map(|&i| [conv[i].pointer(stream), ordinary[i].pointer(stream)])
                .collect::<Vec<_>>(),
            u64::MAX,
        );
        let scan_lengths = Guarded::new(
            stream,
            &[0, scan_counts[0] as u32, scan_tokens as u32],
            u32::MAX,
        );
        let ordinary_output = buffer(scan_tokens * values);
        launch_delta(
            stream,
            functions,
            operands[0].pointer(stream),
            operands[1].pointer(stream),
            operands[2].pointer(stream),
            operands[3].pointer(stream),
            operands[4].pointer(stream),
            ordinary_bindings.pointer(stream),
            scan_lengths.pointer(stream),
            ordinary_output.pointer(stream),
            2,
            scan_tokens as i32,
            cuda,
        )
        .unwrap();
        let slot = usize::from(!prefix);
        if !prefix {
            bit_eq(&output.read(stream), &ordinary_output.read(stream));
        }
        let mut raw_prefix = Vec::new();
        let mut cursor = 0;
        for (&old, &new) in counts.iter().zip(&scan_counts) {
            raw_prefix.extend_from_slice(&raw[cursor * width..(cursor + new) * width]);
            cursor += old;
        }
        let mut ref_prefix = reference_initial();
        ref_prefix.forward(shape, &scan_counts, &raw_prefix, &weights, &decay, &bias);
        for (sequence, &physical) in slots.iter().enumerate() {
            let candidate_delta = out_delta[physical].read(stream);
            let selected = &candidate_delta[slot * delta_size..(slot + 1) * delta_size];
            bit_eq(selected, &ordinary[physical].read(stream));
            assert_close(selected, &ref_prefix.delta[sequence], "candidate F64 state");
            let candidate_conv = out_conv[physical].read(stream);
            assert_eq!(
                &candidate_conv[slot * conv_size..(slot + 1) * conv_size],
                ref_prefix.conv[sequence]
                    .iter()
                    .map(|&v| f16::from_f64(v))
                    .collect::<Vec<_>>()
            );
        }
        ordinary_bindings.assert_unchanged(stream);
        scan_lengths.assert_unchanged(stream);
        for operand in &operands {
            operand.assert_unchanged(stream);
        }
    }
    for item in &conv {
        item.assert_unchanged(stream);
    }
    for item in &delta {
        item.assert_unchanged(stream);
    }
    raw_gpu.assert_unchanged(stream);
    weights_gpu.assert_unchanged(stream);
    decay_gpu.assert_unchanged(stream);
    bias_gpu.assert_unchanged(stream);
    original_bindings.assert_unchanged(stream);
    candidates.assert_unchanged(stream);
    lengths.assert_unchanged(stream);
    indices.assert_unchanged(stream);
    final_conv.read(stream);
    z.read(stream);
}

#[test]
#[ignore = "requires an actual CUDA device"]
fn cuda_candidate_scan_tail_two_matches_original_math_with_b2_and_read_only_state() {
    let runtime = CudaDeviceRuntime::new(
        cuda_vnext_runtime_config(
            0,
            DeviceId::new("device.test.cuda.candidate_scan").unwrap(),
            AttentionExecutionPolicy::Portable,
        )
        .unwrap(),
    )
    .unwrap();
    let provider =
        CudaGatedDeltaRecurrentAttentionProvider::new_candidate(&runtime, false, false).unwrap();
    let stream = runtime.context().default_stream();
    for (kd, vd) in [(3, 5), (32, 17), (128, 128)] {
        for decay in GatedDeltaDecayParameterization::ALL {
            for mapping in GatedDeltaValueHeadMapping::ALL {
                let mut shape = super::super::tests::test_shape();
                shape.key_head_dim = kd;
                shape.value_head_dim = vd;
                shape.value_heads = 6;
                shape.value_features = shape.value_heads * vd;
                shape.qkv_features = 2 * shape.key_heads * kd + shape.value_features;
                shape.qkvz_features = shape.qkv_features + shape.value_features;
                shape.ba_features = 2 * shape.value_heads;
                shape.qkvzba_features = shape.qkvz_features + shape.ba_features;
                shape.decay_parameterization = decay;
                shape.value_head_mapping = mapping;
                for counts in [[1, 2], [2, 5], [5, 1]] {
                    trial(&stream, &provider.functions, shape, counts);
                }
            }
        }
    }
}
