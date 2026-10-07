//! Actual shared input packing used by attention, with independent affine-stage
//! oracles. This leaf gate does not replace full attention/state provider tests.
use super::super::super::q8act_attention::PreparedAttentionProjections;
use super::*;

fn part(ordinal: u32, format: GgufBlockFormat, k: usize, n: usize) -> MatrixPart {
    MatrixPart {
        component_id: id(&format!("component.attention.{ordinal}")),
        format: MatrixFormat::Block(format),
        rows: n as u32,
        columns: k as u32,
        output_offset: 0,
        transform: None,
        signs_region: None,
    }
}

fn causal_plan(
    formats: [GgufBlockFormat; 4],
    k: usize,
    n: usize,
) -> (PreparedAttentionProjections, Vec<MatrixPart>) {
    let parts = formats
        .into_iter()
        .enumerate()
        .map(|(i, format)| part(i as u32 + 2, format, k, n))
        .collect::<Vec<_>>();
    let values = parts
        .iter()
        .enumerate()
        .map(|(i, p)| binding(i as u32 + 2, std::slice::from_ref(p), k, n))
        .collect::<Vec<_>>();
    (
        PreparedAttentionProjections::prepare(Q8ActAttentionProfile::Causal, &values).unwrap(),
        parts,
    )
}

#[test]
fn attention_q8act_pack_groups_follow_retained_projection_eligibility() {
    use GgufBlockFormat::{Iq4Xs, Q4K, Q5K, Q6K};
    for (formats, expected) in [
        ([Q6K; 4], 0),
        ([Q4K, Q6K, Q6K, Q6K], 1),
        ([Q6K, Q5K, Iq4Xs, Q6K], 1),
        ([Q6K, Q6K, Q6K, Q4K], 1),
        ([Q4K, Q5K, Iq4Xs, Q4K], 2),
    ] {
        let (plan, parts) = causal_plan(formats, 256, 17);
        assert_eq!(plan.pack_count().unwrap(), expected);
        for (i, role) in [
            ProjectionRole::CausalQuery,
            ProjectionRole::CausalKey,
            ProjectionRole::CausalValue,
            ProjectionRole::CausalOutput,
        ]
        .into_iter()
        .enumerate()
        {
            plan.validate_parts(role, std::slice::from_ref(&parts[i]))
                .unwrap();
            let mut drift = parts[i].clone();
            drift.output_offset = 1;
            assert!(plan.validate_parts(role, &[drift]).is_err());
        }
    }
}

#[test]
#[ignore = "requires actual CUDA; shared attention pack and G32 projection gate"]
fn attention_q8act_shared_pack_matches_independent_projection_oracles_on_cuda() {
    use GgufBlockFormat::{Iq4Xs, Q4K, Q5K, Q6K};
    let context = CudaContext::new(0).expect("attention Q8act gate requires CUDA");
    let stream = context.default_stream();
    let native = CudaNativeBlockKernels::load(&context).unwrap();
    let q8 = Q8ActKernels::load_attention(&context).unwrap();
    for (k, n, rows) in [
        (256, 17, 1),
        (256, 17, 3),
        (256, 17, 8),
        (256, 17, 9),
        (256, 17, 1025),
        (5120, 49, 8),
        (6144, 17, 9),
    ] {
        for formats in [[Q4K, Q5K, Iq4Xs, Q4K], [Q6K, Q4K, Q5K, Iq4Xs]] {
            let (plan, parts) = causal_plan(formats, k, n);
            let weights = formats
                .into_iter()
                .enumerate()
                .map(|(i, f)| three_format::affine_matrix(MatrixFormat::Block(f), n, k, i))
                .collect::<Vec<_>>();
            let gpu_weights = weights
                .iter()
                .map(|(bytes, _)| {
                    let mut padded = vec![0xBA; 5];
                    padded.extend(bytes);
                    padded.extend([0xBA; 3]);
                    Guarded::new(&stream, &padded, 0xABu8)
                })
                .collect::<Vec<_>>();
            let pointers = gpu_weights
                .iter()
                .map(|w| w.pointer(&stream) + 5)
                .collect::<Vec<_>>();
            let input = (0..rows * k)
                .map(|i| {
                    f16::from_f32(((i * 7 + i / k * 3) % 23) as f32 / 32768.0 - 11.0 / 32768.0)
                })
                .collect::<Vec<_>>();
            let mut padded = vec![f16::from_f32(19.0); 3];
            padded.extend(&input);
            padded.extend([f16::from_f32(19.0); 3]);
            let x = Guarded::new(&stream, &padded, f16::from_f32(73.0));
            let input_ptr = x.pointer(&stream) + 6;
            let workspace_bytes = plan.bytes_per_token() * rows as u64;
            let workspace = Guarded::new(&stream, &vec![0xABu8; workspace_bytes as usize], 0xCD);
            let stride = n + 7;
            let offset = 3;
            let guard = f16::from_f32(79.0);
            let roles = [
                ProjectionRole::CausalQuery,
                ProjectionRole::CausalKey,
                ProjectionRole::CausalValue,
                ProjectionRole::CausalOutput,
            ];
            let mut previous: Option<Vec<Vec<u16>>> = None;
            for _ in 0..2 {
                let outputs = (0..4)
                    .map(|_| Guarded::new(&stream, &vec![guard; rows * stride + offset], guard))
                    .collect::<Vec<_>>();
                q8.with_packed_input(
                    &stream,
                    true,
                    input_ptr,
                    rows as u32,
                    k as u32,
                    workspace.pointer(&stream),
                    workspace_bytes,
                    |packed| {
                        for i in 0..3 {
                            q8.launch_with_packed(
                                &native,
                                &stream,
                                plan.projection(roles[i]).unwrap(),
                                std::slice::from_ref(&parts[i]),
                                std::slice::from_ref(&pointers[i]),
                                input_ptr,
                                outputs[i].pointer(&stream) + offset as u64 * 2,
                                rows as u32,
                                stride as u32,
                                packed,
                                0,
                            )?;
                        }
                        // The same address shifted by one F16 is not the packed
                        // input identity, even when its allocation is still live.
                        assert!(q8
                            .launch_with_packed(
                                &native,
                                &stream,
                                plan.projection(roles[1]).unwrap(),
                                std::slice::from_ref(&parts[1]),
                                std::slice::from_ref(&pointers[1]),
                                input_ptr + 2,
                                outputs[1].pointer(&stream),
                                rows as u32,
                                stride as u32,
                                packed,
                                0
                            )
                            .is_err());
                        Ok(())
                    },
                )
                .unwrap();
                // Distinct output-projection input: its fresh pack must not
                // reuse the normalized-QKV codes left in shared scratch.
                let o_input = input
                    .iter()
                    .map(|x| f16::from_f32(-x.to_f32() * 0.75))
                    .collect::<Vec<_>>();
                let ox = Guarded::new(&stream, &o_input, f16::from_f32(71.0));
                q8.launch(
                    &native,
                    &stream,
                    plan.projection(roles[3]).unwrap(),
                    std::slice::from_ref(&parts[3]),
                    std::slice::from_ref(&pointers[3]),
                    ox.pointer(&stream),
                    outputs[3].pointer(&stream) + offset as u64 * 2,
                    rows as u32,
                    stride as u32,
                    workspace.pointer(&stream),
                    workspace_bytes,
                    0,
                )
                .unwrap();
                let mut all_bits = Vec::new();
                for i in 0..4 {
                    let source = if i == 3 {
                        o_input.as_slice()
                    } else {
                        input.as_slice()
                    };
                    let actual = outputs[i].read(&stream);
                    assert!(actual[..offset].iter().all(|x| *x == guard));
                    for row in 0..rows {
                        assert!(
                            actual[offset + row * stride + n..offset + (row + 1) * stride]
                                .iter()
                                .all(|x| *x == guard)
                        );
                    }
                    let projection = plan.projection(roles[i]).unwrap();
                    if projection.has_staged_leaf() {
                        three_format::assert_staged_projection(
                            source,
                            &weights[i].0,
                            &weights[i].1,
                            &actual[offset..],
                            formats[i],
                            k,
                            n,
                            stride,
                            0,
                        );
                    } else {
                        projection_oracle(
                            source,
                            &weights[i].1,
                            &actual[offset..],
                            k,
                            n,
                            stride,
                            0,
                            false,
                        );
                    }
                    let independent =
                        Guarded::new(&stream, &vec![guard; rows * stride + offset], guard);
                    let source_gpu = if i == 3 { &ox } else { &x };
                    let source_ptr = source_gpu.pointer(&stream) + if i == 3 { 0 } else { 6 };
                    q8.launch(
                        &native,
                        &stream,
                        projection,
                        std::slice::from_ref(&parts[i]),
                        std::slice::from_ref(&pointers[i]),
                        source_ptr,
                        independent.pointer(&stream) + offset as u64 * 2,
                        rows as u32,
                        stride as u32,
                        workspace.pointer(&stream),
                        workspace_bytes,
                        0,
                    )
                    .unwrap();
                    let bits = actual.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
                    assert_eq!(
                        bits,
                        independent
                            .read(&stream)
                            .iter()
                            .map(|x| x.to_bits())
                            .collect::<Vec<_>>()
                    );
                    all_bits.push(bits);
                }
                if let Some(previous) = &previous {
                    assert_eq!(previous, &all_bits);
                }
                previous = Some(all_bits);
                ox.assert_unchanged(&stream);
                workspace.read(&stream);
            }
            x.assert_unchanged(&stream);
            for weight in &gpu_weights {
                weight.assert_unchanged(&stream);
            }
        }
    }
}
