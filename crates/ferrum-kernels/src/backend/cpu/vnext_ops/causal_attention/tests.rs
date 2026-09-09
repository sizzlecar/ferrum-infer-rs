use super::*;
use crate::backend::cpu::vnext_ops::tests::encoded;

fn reference_rotary(shape: CausalShape, position: usize, input: &[u8], norm: &[u8]) -> Vec<f64> {
    let values = (0..shape.head_dim)
        .map(|i| CpuFloat::F16.read(input, i) as f64)
        .collect::<Vec<_>>();
    let scale = (values.iter().map(|v| v * v).sum::<f64>() / shape.head_dim as f64
        + shape.epsilon as f64)
        .sqrt()
        .recip();
    let mut normalized = values
        .iter()
        .enumerate()
        .map(|(i, v)| v * scale * CpuFloat::F16.read(norm, i) as f64)
        .collect::<Vec<_>>();
    let original = normalized.clone();
    for pair in 0..shape.rope_dim / 2 {
        let angle =
            position as f64 / (shape.theta as f64).powf((pair * 2) as f64 / shape.rope_dim as f64);
        let (a, b) = if shape.interleaved {
            (pair * 2, pair * 2 + 1)
        } else {
            (pair, pair + shape.rope_dim / 2)
        };
        normalized[a] = original[a] * angle.cos() - original[b] * angle.sin();
        normalized[b] = original[b] * angle.cos() + original[a] * angle.sin();
    }
    normalized
}

#[test]
fn causal_prefix_matches_f64_attention_and_rotary_with_gqa_and_optional_gate() {
    for interleaved in [false, true] {
        for gated in [false, true] {
            let shape = CausalShape {
                hidden: 8,
                query_heads: 4,
                kv_heads: 2,
                head_dim: 6,
                rope_dim: 4,
                maximum_context: 8,
                epsilon: 1.0e-5,
                theta: 10000.0,
                interleaved,
                gated,
            };
            let norm = encoded(
                &(0..shape.head_dim)
                    .map(|i| 0.75 + i as f32 * 0.0625)
                    .collect::<Vec<_>>(),
                CpuFloat::F16,
            );
            // Future capacity contains NaNs, so accidentally attending beyond
            // the admitted position invalidates the output.
            let mut kv = encoded(
                &vec![f32::NAN; shape.maximum_context * shape.kv() * 2],
                CpuFloat::F16,
            );
            let mut query = vec![0; shape.queries() * 2];
            let mut accumulated = vec![0; shape.queries() * 4];
            let mut context = vec![0; shape.queries() * 2];
            for position in 0..5 {
                let values = |width: usize, phase: f32| {
                    encoded(
                        &(0..width)
                            .map(|i| ((i + position * 7) as f32 * 0.31 + phase).sin() * 0.75)
                            .collect::<Vec<_>>(),
                        CpuFloat::F16,
                    )
                };
                let raw_query = values(shape.query_projection(), 0.0);
                let raw_key = values(shape.kv(), 0.7);
                let raw_value = values(shape.kv(), 1.3);
                step(
                    shape,
                    position,
                    CausalInputs {
                        query: &raw_query,
                        key: &raw_key,
                        value: &raw_value,
                        query_norm: &norm,
                        key_norm: &norm,
                    },
                    &mut kv[..],
                    CausalScratch {
                        query: &mut query,
                        accumulated: &mut accumulated,
                        context: &mut context,
                    },
                )
                .unwrap();
                let stride = shape.head_dim * if gated { 2 } else { 1 };
                for head in 0..shape.query_heads {
                    let expected = reference_rotary(
                        shape,
                        position,
                        &raw_query[head * stride * 2..(head * stride + shape.head_dim) * 2],
                        &norm,
                    );
                    for (column, expected) in expected.iter().enumerate() {
                        let actual =
                            CpuFloat::F16.read(&query, head * shape.head_dim + column) as f64;
                        assert!((actual - expected).abs() < expected.abs() * 0.001 + 1.0e-5);
                    }
                }
                for head in 0..shape.kv_heads {
                    let expected = reference_rotary(
                        shape,
                        position,
                        &raw_key[head * shape.head_dim * 2..(head + 1) * shape.head_dim * 2],
                        &norm,
                    );
                    for (column, expected) in expected.iter().enumerate() {
                        let index = position * shape.kv() * 2 + head * shape.head_dim + column;
                        let actual = CpuFloat::F16.read(&kv, index) as f64;
                        assert!((actual - expected).abs() < expected.abs() * 0.001 + 1.0e-5);
                        assert_eq!(
                            CpuFloat::F16.read(&kv, index + shape.kv()),
                            CpuFloat::F16.read(&raw_value, head * shape.head_dim + column)
                        );
                    }
                }
                for head in 0..shape.query_heads {
                    let kv_head = head / (shape.query_heads / shape.kv_heads);
                    let scores = (0..=position)
                        .map(|token| {
                            (0..shape.head_dim)
                                .map(|column| {
                                    CpuFloat::F16.read(&query, head * shape.head_dim + column)
                                        as f64
                                        * CpuFloat::F16.read(
                                            &kv,
                                            token * shape.kv() * 2
                                                + kv_head * shape.head_dim
                                                + column,
                                        ) as f64
                                })
                                .sum::<f64>()
                                / (shape.head_dim as f64).sqrt()
                        })
                        .collect::<Vec<_>>();
                    let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                    let probabilities = scores
                        .iter()
                        .map(|score| (score - maximum).exp())
                        .collect::<Vec<_>>();
                    let sum = probabilities.iter().sum::<f64>();
                    for column in 0..shape.head_dim {
                        let mut expected = probabilities
                            .iter()
                            .enumerate()
                            .map(|(token, probability)| {
                                probability
                                    * CpuFloat::F16.read(
                                        &kv,
                                        token * shape.kv() * 2
                                            + shape.kv()
                                            + kv_head * shape.head_dim
                                            + column,
                                    ) as f64
                            })
                            .sum::<f64>()
                            / sum;
                        if gated {
                            expected /= 1.0
                                + (-(CpuFloat::F16
                                    .read(&raw_query, head * stride + shape.head_dim + column)
                                    as f64))
                                    .exp();
                        }
                        let actual =
                            CpuFloat::F16.read(&context, head * shape.head_dim + column) as f64;
                        assert!((actual - expected).abs() <= expected.abs() * 0.001 + 1.0e-5, "position={position}, head={head}, column={column}: {actual} != {expected}");
                    }
                }
            }
            let before = kv.clone();
            assert!(step(
                shape,
                shape.maximum_context,
                CausalInputs {
                    query: &[],
                    key: &[],
                    value: &[],
                    query_norm: &norm,
                    key_norm: &norm
                },
                &mut kv[..],
                CausalScratch {
                    query: &mut query,
                    accumulated: &mut accumulated,
                    context: &mut context
                }
            )
            .is_err());
            assert_eq!(kv, before);
        }
    }
}

#[test]
fn invalid_causal_dimensions_fail_before_memory_access() {
    let shape = CausalShape {
        hidden: 8,
        query_heads: 4,
        kv_heads: 2,
        head_dim: 6,
        rope_dim: 4,
        maximum_context: 8,
        epsilon: 1.0e-5,
        theta: 10000.0,
        interleaved: false,
        gated: true,
    };
    for invalid in [
        CausalShape {
            kv_heads: 0,
            ..shape
        },
        CausalShape {
            head_dim: usize::MAX,
            ..shape
        },
        CausalShape {
            rope_dim: 3,
            ..shape
        },
        CausalShape {
            maximum_context: usize::MAX,
            ..shape
        },
        CausalShape {
            query_heads: 3,
            ..shape
        },
        CausalShape {
            theta: 0.0,
            ..shape
        },
    ] {
        assert!(invalid.validate().is_err());
    }
}

#[test]
fn paged_kv_preserves_prefix_across_growth_and_cross_page_heads() {
    let shape = CausalShape {
        hidden: 8,
        query_heads: 4,
        kv_heads: 2,
        head_dim: 6,
        rope_dim: 4,
        maximum_context: 8,
        epsilon: 1.0e-5,
        theta: 10000.0,
        interleaved: false,
        gated: true,
    };
    // A 12-byte head and a 48-byte token both cross these separate allocations.
    let page_bytes = 32;
    let mut pages: Vec<Vec<u8>> = Vec::new();
    let mut contiguous = vec![0; shape.maximum_context * shape.state_bytes_per_token()];
    let norm = encoded(&vec![1.0; shape.head_dim], CpuFloat::F16);
    for position in 0..shape.maximum_context {
        let required = (position + 1) * shape.state_bytes_per_token();
        while pages.len() * page_bytes < required {
            pages.push(vec![0; page_bytes]);
        }
        let values = |width: usize, phase: f32| {
            encoded(
                &(0..width)
                    .map(|i| ((i + position * 7) as f32 * 0.31 + phase).sin())
                    .collect::<Vec<_>>(),
                CpuFloat::F16,
            )
        };
        let raw_query = values(shape.query_projection(), 0.0);
        let raw_key = values(shape.kv(), 0.7);
        let raw_value = values(shape.kv(), 1.3);
        let inputs = || CausalInputs {
            query: &raw_query,
            key: &raw_key,
            value: &raw_value,
            query_norm: &norm,
            key_norm: &norm,
        };
        let mut query = vec![0; shape.queries() * 2];
        let mut accumulated = vec![0; shape.queries() * 4];
        let mut expected = vec![0; shape.queries() * 2];
        step(
            shape,
            position,
            inputs(),
            &mut contiguous[..],
            CausalScratch {
                query: &mut query,
                accumulated: &mut accumulated,
                context: &mut expected,
            },
        )
        .unwrap();
        let mut actual = vec![0; expected.len()];
        let mut refs = pages.iter_mut().map(Vec::as_mut_slice).collect::<Vec<_>>();
        let mut paged = CpuKvPages::new(&mut refs).unwrap();
        step(
            shape,
            position,
            inputs(),
            &mut paged,
            CausalScratch {
                query: &mut query,
                accumulated: &mut accumulated,
                context: &mut actual,
            },
        )
        .unwrap();
        assert_eq!(actual, expected, "position {position}");
        let flattened = pages.iter().flatten().copied().collect::<Vec<_>>();
        assert_eq!(&flattened[..required], &contiguous[..required]);
    }
}

#[test]
fn paged_kv_rejects_empty_odd_or_unequal_pages() {
    assert!(CpuKvPages::new(&mut []).is_err());
    assert!(CpuKvPages::new(&mut [&mut []]).is_err());
    assert!(CpuKvPages::new(&mut [&mut [0; 3]]).is_err());
    assert!(CpuKvPages::new(&mut [&mut [0; 4], &mut [0; 2]]).is_err());
}
