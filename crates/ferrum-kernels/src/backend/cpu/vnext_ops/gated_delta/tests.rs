use super::*;
use crate::backend::cpu::vnext_ops::tests::encoded;

#[test]
fn recurrent_steps_match_f64_with_both_declared_head_and_decay_conventions() {
    for mapping in GatedDeltaValueHeadMapping::ALL {
        for decay in GatedDeltaDecayParameterization::ALL {
            let shape = GatedDeltaShape {
                hidden: 8,
                key_heads: 2,
                value_heads: 4,
                key_dim: 3,
                value_dim: 2,
                convolution: 3,
                epsilon: 1.0e-5,
                decay,
                mapping,
            };
            let conv_values = (0..shape.qkv() * shape.convolution)
                .map(|i| ((i as f32) * 0.31).cos() * 0.2)
                .collect::<Vec<_>>();
            let conv_weight = encoded(&conv_values, CpuFloat::F16);
            let rates = match decay {
                GatedDeltaDecayParameterization::LogRate => [-0.4_f32, 0.1, 0.3, -0.2],
                GatedDeltaDecayParameterization::NegativeRate => {
                    [-0.4_f32, 0.1, 0.3, -0.2].map(|value| -value.exp())
                }
            };
            let rate_weight = encoded(&rates, CpuFloat::F32);
            let bias = encoded(&[0.1, -0.2, 0.3, 0.0], CpuFloat::F32);
            let norm = encoded(&[0.75, 1.125], CpuFloat::F32);
            let mut conv_state = vec![0; shape.qkv() * (shape.convolution - 1) * 2];
            let mut recurrent = vec![0; shape.values() * shape.key_dim * 4];
            let mut qkv = vec![0; shape.qkv() * 4];
            let mut core = vec![0; shape.values() * 4];
            let mut output = vec![0; shape.values() * 2];
            let mut previous = vec![vec![0.0_f64; 2]; shape.qkv()];
            let mut reference_state =
                vec![vec![vec![0.0_f64; shape.key_dim]; shape.value_dim]; shape.value_heads];
            for token in 0..7 {
                let mixed = encoded(
                    &(0..shape.mixed())
                        .map(|i| ((i + token * 13) as f32 * 0.37).sin())
                        .collect::<Vec<_>>(),
                    CpuFloat::F16,
                );
                step(
                    shape,
                    &mixed,
                    GatedDeltaWeights {
                        convolution: &conv_weight,
                        decay: &rate_weight,
                        dt_bias: &bias,
                        norm: &norm,
                    },
                    GatedDeltaState {
                        convolution: &mut conv_state,
                        recurrent: &mut recurrent,
                    },
                    GatedDeltaScratch {
                        qkv: &mut qkv,
                        core: &mut core,
                        output: &mut output,
                    },
                )
                .unwrap();

                let read = |i| CpuFloat::F16.read(&mixed, i) as f64;
                let mut convolved = vec![0.0; shape.qkv()];
                for channel in 0..shape.qkv() {
                    let samples = [previous[channel][0], previous[channel][1], read(channel)];
                    let value = samples
                        .iter()
                        .enumerate()
                        .map(|(k, x)| x * CpuFloat::F16.read(&conv_weight, channel * 3 + k) as f64)
                        .sum::<f64>();
                    convolved[channel] = value / (1.0 + (-value).exp());
                    previous[channel] = vec![samples[1], samples[2]];
                    for k in 0..2 {
                        assert_eq!(
                            CpuFloat::F16.read(&conv_state, channel * 2 + k) as f64,
                            previous[channel][k]
                        );
                    }
                }
                let mut queries = convolved[..shape.qk()].to_vec();
                let mut keys = convolved[shape.qk()..shape.qk() * 2].to_vec();
                for rows in [&mut queries, &mut keys] {
                    for head in rows.chunks_exact_mut(shape.key_dim) {
                        let scale = (head.iter().map(|v| v * v).sum::<f64>() + 1.0e-6)
                            .sqrt()
                            .recip();
                        for value in head {
                            *value *= scale;
                        }
                    }
                }
                for head in 0..shape.value_heads {
                    let key_head = match mapping {
                        GatedDeltaValueHeadMapping::GroupedByKeyHead => head / 2,
                        GatedDeltaValueHeadMapping::InterleavedByKeyHead => head % 2,
                    };
                    let gate_start = shape.qkv() + shape.values();
                    let beta = 1.0 / (1.0 + (-read(gate_start + head)).exp());
                    let a = read(gate_start + shape.value_heads + head)
                        + CpuFloat::F32.read(&bias, head) as f64;
                    let rate = match decay {
                        GatedDeltaDecayParameterization::LogRate => -(rates[head] as f64).exp(),
                        GatedDeltaDecayParameterization::NegativeRate => rates[head] as f64,
                    };
                    let multiplier = (rate * (1.0 + a.exp()).ln()).exp();
                    let mut projected = vec![0.0; shape.value_dim];
                    for column in 0..shape.value_dim {
                        let state = &mut reference_state[head][column];
                        for value in state.iter_mut() {
                            *value *= multiplier;
                        }
                        let prediction = state
                            .iter()
                            .enumerate()
                            .map(|(k, value)| value * keys[key_head * shape.key_dim + k])
                            .sum::<f64>();
                        let residual = beta
                            * (convolved[shape.qk() * 2 + head * shape.value_dim + column]
                                - prediction);
                        for (k, value) in state.iter_mut().enumerate() {
                            *value += residual * keys[key_head * shape.key_dim + k];
                            let index = (head * shape.value_dim + column) * shape.key_dim + k;
                            assert!(
                                (CpuFloat::F32.read(&recurrent, index) as f64 - *value).abs()
                                    < 1.0e-5,
                                "state token={token} head={head} dim={column},{k}"
                            );
                        }
                        projected[column] = state
                            .iter()
                            .enumerate()
                            .map(|(k, value)| value * queries[key_head * shape.key_dim + k])
                            .sum::<f64>()
                            / (shape.key_dim as f64).sqrt();
                    }
                    let norm_scale = (projected.iter().map(|v| v * v).sum::<f64>()
                        / shape.value_dim as f64
                        + shape.epsilon as f64)
                        .sqrt()
                        .recip();
                    for (column, value) in projected.iter().enumerate() {
                        let z = read(shape.qkv() + head * shape.value_dim + column);
                        let expected =
                            value * norm_scale * CpuFloat::F32.read(&norm, column) as f64 * z
                                / (1.0 + (-z).exp());
                        let actual =
                            CpuFloat::F16.read(&output, head * shape.value_dim + column) as f64;
                        // One final F16 conversion; F32 recurrent arithmetic adds
                        // a small absolute allowance near zero.
                        assert!(
                            (actual - expected).abs() <= expected.abs() * 0.001 + 1.0e-5,
                            "output token={token} head={head} dim={column}: {actual} != {expected}"
                        );
                    }
                }
            }
            let old_state = recurrent.clone();
            assert!(step(
                shape,
                &vec![0; shape.mixed() * 2 - 1],
                GatedDeltaWeights {
                    convolution: &conv_weight,
                    decay: &rate_weight,
                    dt_bias: &bias,
                    norm: &norm
                },
                GatedDeltaState {
                    convolution: &mut conv_state,
                    recurrent: &mut recurrent
                },
                GatedDeltaScratch {
                    qkv: &mut qkv,
                    core: &mut core,
                    output: &mut output
                }
            )
            .is_err());
            assert_eq!(recurrent, old_state);
        }
    }
}

#[test]
fn invalid_recurrent_dimensions_fail_before_memory_access() {
    let shape = GatedDeltaShape {
        hidden: 8,
        key_heads: 2,
        value_heads: 4,
        key_dim: 3,
        value_dim: 2,
        convolution: 3,
        epsilon: 1.0e-5,
        decay: GatedDeltaDecayParameterization::LogRate,
        mapping: GatedDeltaValueHeadMapping::GroupedByKeyHead,
    };
    for invalid in [
        GatedDeltaShape {
            key_heads: 0,
            ..shape
        },
        GatedDeltaShape {
            key_dim: usize::MAX,
            ..shape
        },
        GatedDeltaShape {
            convolution: 1,
            ..shape
        },
        GatedDeltaShape {
            epsilon: f32::NAN,
            ..shape
        },
        GatedDeltaShape {
            value_heads: 3,
            ..shape
        },
    ] {
        assert!(invalid.validate().is_err());
    }
}
