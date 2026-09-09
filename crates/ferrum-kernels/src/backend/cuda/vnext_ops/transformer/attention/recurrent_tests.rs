//! Numerical and state-isolation checks through the production recurrent launchers.
use super::*;
use crate::backend::cuda::vnext_ops::cuda_vnext_runtime_config;
use cudarc::driver::{CudaSlice, DevicePtr, DeviceRepr};
use ferrum_interfaces::vnext::DeviceId;
use ferrum_types::AttentionExecutionPolicy;
use half::f16;
use std::fmt::Debug;

struct Guarded<T> {
    device: CudaSlice<T>,
    original: Vec<T>,
}

impl<T: DeviceRepr + Clone + PartialEq + Debug> Guarded<T> {
    fn new(stream: &Arc<CudaStream>, values: &[T], guard: T) -> Self {
        let mut original = vec![guard.clone(); 8];
        original.extend_from_slice(values);
        original.extend(vec![guard; 8]);
        Self {
            device: stream.clone_htod(&original).unwrap(),
            original,
        }
    }

    fn pointer(&self, stream: &Arc<CudaStream>) -> u64 {
        self.device.device_ptr(stream).0 + (8 * std::mem::size_of::<T>()) as u64
    }

    fn read(&self, stream: &Arc<CudaStream>) -> Vec<T> {
        // Raw-pointer launchers retain buffers in this fixture on one stream.
        stream.synchronize().unwrap();
        let result = stream.clone_dtoh(&self.device).unwrap();
        let end = result.len() - 8;
        assert_eq!(&result[..8], &self.original[..8], "leading guard modified");
        assert_eq!(
            &result[end..],
            &self.original[end..],
            "trailing guard modified"
        );
        result[8..end].to_vec()
    }

    fn assert_unchanged(&self, stream: &Arc<CudaStream>) {
        assert_eq!(self.read(stream), self.original[8..self.original.len() - 8]);
    }
}

fn assert_close(actual: &[f32], expected: &[f64], stage: &str) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        let bound = 4.0e-5 * (1.0 + expected.abs());
        assert!(
            actual.is_finite() && (f64::from(actual) - expected).abs() <= bound,
            "{stage}[{index}]: actual {actual}, F64 reference {expected}, bound {bound}"
        );
    }
}

fn sample(index: usize, salt: usize, scale: f32) -> f32 {
    (((index * 13 + salt * 7) % 41) as f32 - 20.0) * scale
}

struct Reference {
    conv: Vec<Vec<f64>>,
    delta: Vec<Vec<f64>>,
}

struct PreparedReference {
    query: Vec<f64>,
    key: Vec<f64>,
    value: Vec<f64>,
    g: Vec<f64>,
    beta: Vec<f64>,
    z: Vec<f16>,
    output: Vec<f64>,
}

impl Reference {
    fn forward(
        &mut self,
        shape: AttentionShape,
        counts: &[usize],
        raw: &[f16],
        conv_weight: &[f16],
        decay: &[f32],
        dt_bias: &[f32],
    ) -> PreparedReference {
        let kh = shape.key_heads as usize;
        let vh = shape.value_heads as usize;
        let kd = shape.key_head_dim as usize;
        let vd = shape.value_head_dim as usize;
        let qk = kh * kd;
        let values = vh * vd;
        let channels = 2 * qk + values;
        let width = shape.qkvzba_features as usize;
        let kernel = shape.conv_kernel as usize;
        let tokens = counts.iter().sum::<usize>();
        let mut result = PreparedReference {
            query: vec![0.0; tokens * qk],
            key: vec![0.0; tokens * qk],
            value: vec![0.0; tokens * values],
            g: vec![0.0; tokens * vh],
            beta: vec![0.0; tokens * vh],
            z: vec![f16::ZERO; tokens * values],
            output: vec![0.0; tokens * values],
        };
        let mut token = 0;
        for (sequence, &count) in counts.iter().enumerate() {
            for _ in 0..count {
                let row = &raw[token * width..][..width];
                let mut convolution = vec![0.0; channels];
                for channel in 0..channels {
                    let history = &mut self.conv[sequence]
                        [channel * (kernel - 1)..(channel + 1) * (kernel - 1)];
                    let weight = &conv_weight[channel * kernel..][..kernel];
                    let sum = history
                        .iter()
                        .zip(weight)
                        .map(|(x, w)| x * w.to_f64())
                        .sum::<f64>()
                        + row[channel].to_f64() * weight[kernel - 1].to_f64();
                    convolution[channel] = sum / (1.0 + (-sum).exp());
                    history.rotate_left(1);
                    history[kernel - 2] = row[channel].to_f64();
                }
                result.query[token * qk..][..qk].copy_from_slice(&convolution[..qk]);
                result.key[token * qk..][..qk].copy_from_slice(&convolution[qk..2 * qk]);
                result.value[token * values..][..values].copy_from_slice(&convolution[2 * qk..]);
                result.z[token * values..][..values]
                    .copy_from_slice(&row[channels..channels + values]);
                for head in 0..vh {
                    let b = row[channels + values + head].to_f64();
                    let a = row[channels + values + vh + head].to_f64() + f64::from(dt_bias[head]);
                    let rate = match shape.decay_parameterization {
                        GatedDeltaDecayParameterization::LogRate => -f64::from(decay[head]).exp(),
                        GatedDeltaDecayParameterization::NegativeRate => f64::from(decay[head]),
                    };
                    result.g[token * vh + head] = rate * a.exp().ln_1p();
                    result.beta[token * vh + head] = 1.0 / (1.0 + (-b).exp());
                    let key_head = match shape.value_head_mapping {
                        GatedDeltaValueHeadMapping::GroupedByKeyHead => head / (vh / kh),
                        GatedDeltaValueHeadMapping::InterleavedByKeyHead => head % kh,
                    };
                    let q = &result.query[token * qk + key_head * kd..][..kd];
                    let k = &result.key[token * qk + key_head * kd..][..kd];
                    let q_norm = (q.iter().map(|x| x * x).sum::<f64>() + 1.0e-6).sqrt();
                    let k_norm = (k.iter().map(|x| x * x).sum::<f64>() + 1.0e-6).sqrt();
                    let discount = result.g[token * vh + head].exp();
                    for column in 0..vd {
                        let state = &mut self.delta[sequence][(head * vd + column) * kd..][..kd];
                        for entry in state.iter_mut() {
                            *entry *= discount;
                        }
                        let remembered = state
                            .iter()
                            .zip(k)
                            .map(|(s, k)| s * k / k_norm)
                            .sum::<f64>();
                        let innovation = (convolution[2 * qk + head * vd + column] - remembered)
                            * result.beta[token * vh + head];
                        for (entry, key) in state.iter_mut().zip(k) {
                            *entry += innovation * key / k_norm;
                        }
                        result.output[token * values + head * vd + column] = state
                            .iter()
                            .zip(q)
                            .map(|(s, q)| s * q / q_norm / (kd as f64).sqrt())
                            .sum();
                    }
                }
                token += 1;
            }
        }
        result
    }
}

fn exercise(
    stream: &Arc<CudaStream>,
    functions: &AttentionFunctions,
    shape: AttentionShape,
    batches: &[[usize; 3]],
) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let channels = shape.qkv_features as usize;
    let width = shape.qkvzba_features as usize;
    let qk = (shape.key_heads * shape.key_head_dim) as usize;
    let values = shape.value_features as usize;
    let heads = shape.value_heads as usize;
    let conv_count = shape.conv_state_elements().unwrap() as usize;
    let state_count = (shape.value_heads * shape.value_head_dim * shape.key_head_dim) as usize;
    let conv_initial = (0..4)
        .map(|slot| {
            (0..conv_count)
                .map(|i| f16::from_f32(sample(i, slot + 1, 0.015625)))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let state_initial = (0..4)
        .map(|slot| {
            (0..state_count)
                .map(|i| sample(i, slot + 3, 0.00390625))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let conv = conv_initial
        .iter()
        .map(|x| Guarded::new(stream, x, f16::from_f32(81.0)))
        .collect::<Vec<_>>();
    let states = state_initial
        .iter()
        .map(|x| Guarded::new(stream, x, 91.0_f32))
        .collect::<Vec<_>>();
    // Deliberately permuted sequence-to-slot bindings; slot 1 stays idle.
    let slots = [2, 0, 3];
    let pointers = slots
        .iter()
        .flat_map(|&slot| [conv[slot].pointer(stream), states[slot].pointer(stream)])
        .collect::<Vec<_>>();
    let bindings = Guarded::new(stream, &pointers, 0xDEAD_u64);
    let conv_weight = (0..channels * shape.conv_kernel as usize)
        .map(|i| f16::from_f32(sample(i, 9, 0.03125)))
        .collect::<Vec<_>>();
    let decay = (0..heads)
        .map(|i| match shape.decay_parameterization {
            GatedDeltaDecayParameterization::LogRate => -0.8 + i as f32 * 0.2,
            GatedDeltaDecayParameterization::NegativeRate => -0.2 - i as f32 * 0.3,
        })
        .collect::<Vec<_>>();
    let bias = (0..heads).map(|i| sample(i, 5, 0.125)).collect::<Vec<_>>();
    let weight_gpu = Guarded::new(stream, &conv_weight, f16::from_f32(71.0));
    let decay_gpu = Guarded::new(stream, &decay, 61.0_f32);
    let bias_gpu = Guarded::new(stream, &bias, 51.0_f32);
    let mut reference = Reference {
        conv: slots
            .iter()
            .map(|&i| conv_initial[i].iter().map(|x| x.to_f64()).collect())
            .collect(),
        delta: slots
            .iter()
            .map(|&i| state_initial[i].iter().copied().map(f64::from).collect())
            .collect(),
    };
    let mut positions = [0; 3];
    let mut outputs = vec![Vec::new(); 3];
    for counts in batches {
        let tokens: usize = counts.iter().sum();
        let mut raw = Vec::new();
        let mut lengths = vec![0_u32];
        let mut token_sequences = Vec::new();
        for (sequence, &count) in counts.iter().enumerate() {
            for local in 0..count {
                raw.extend((0..width).map(|i| {
                    f16::from_f32(sample(
                        i + (positions[sequence] + local) * width,
                        sequence + 7,
                        0.0625,
                    ))
                }));
                token_sequences.push(sequence as u32);
            }
            positions[sequence] += count;
            lengths.push(lengths.last().unwrap() + count as u32);
        }
        let expected = reference.forward(shape, counts, &raw, &conv_weight, &decay, &bias);
        let raw_gpu = Guarded::new(stream, &raw, f16::from_f32(41.0));
        let lengths_gpu = Guarded::new(stream, &lengths, u32::MAX);
        let sequence_gpu = Guarded::new(stream, &token_sequences, u32::MAX);
        let f32_buffer = |length| Guarded::new(stream, &vec![0.0_f32; length], 101.0);
        let query = f32_buffer(tokens * qk);
        let key = f32_buffer(tokens * qk);
        let value = f32_buffer(tokens * values);
        let g = f32_buffer(tokens * heads);
        let beta = f32_buffer(tokens * heads);
        let output = f32_buffer(tokens * values);
        let z = Guarded::new(
            stream,
            &vec![f16::ZERO; tokens * values],
            f16::from_f32(111.0),
        );
        let final_conv = Guarded::new(
            stream,
            &vec![f16::ZERO; 3 * conv_count],
            f16::from_f32(121.0),
        );
        let cuda = shape.cuda_shape().unwrap();
        launch_prepare(AttentionPrepareRequest {
            stream,
            function: functions.prepare_for(shape.decay_parameterization),
            buffers: AttentionPrepareBuffers {
                qkvzba: raw_gpu.pointer(stream),
                conv_weight: weight_gpu.pointer(stream),
                state_binding: bindings.pointer(stream),
                a_log: decay_gpu.pointer(stream),
                dt_bias: bias_gpu.pointer(stream),
                cu_seqlens: lengths_gpu.pointer(stream),
                token_seq_indices: sequence_gpu.pointer(stream),
                query: query.pointer(stream),
                key: key.pointer(stream),
                value: value.pointer(stream),
                z: z.pointer(stream),
                g: g.pointer(stream),
                beta: beta.pointer(stream),
                final_conv_state: final_conv.pointer(stream),
            },
            context: AttentionPrepareContext {
                tokens: tokens as u64,
                tokens_i32: tokens as i32,
                batch: 3,
                physical: cuda,
                logical: shape,
            },
        })
        .unwrap();
        assert_close(&query.read(stream), &expected.query, "convolved query");
        assert_close(&key.read(stream), &expected.key, "convolved key");
        assert_close(&value.read(stream), &expected.value, "convolved value");
        assert_close(&g.read(stream), &expected.g, "decay gate");
        assert_close(&beta.read(stream), &expected.beta, "beta gate");
        assert_eq!(z.read(stream), expected.z);
        launch_conv_state_commit(
            stream,
            &functions.conv_state_commit,
            final_conv.pointer(stream),
            bindings.pointer(stream),
            3,
            conv_count as i32,
        )
        .unwrap();
        launch_qk_norm(
            stream,
            &functions.qk_norm,
            query.pointer(stream),
            key.pointer(stream),
            tokens as u64,
            tokens as i32,
            cuda,
        )
        .unwrap();
        launch_delta(
            stream,
            functions,
            query.pointer(stream),
            key.pointer(stream),
            value.pointer(stream),
            g.pointer(stream),
            beta.pointer(stream),
            bindings.pointer(stream),
            lengths_gpu.pointer(stream),
            output.pointer(stream),
            3,
            tokens as i32,
            cuda,
        )
        .unwrap();
        let actual = output.read(stream);
        assert_close(&actual, &expected.output, "recurrent output");
        let mut cursor = 0;
        for (sequence, &count) in counts.iter().enumerate() {
            outputs[sequence].extend_from_slice(&actual[cursor..cursor + count * values]);
            cursor += count * values;
            assert_close(
                &states[slots[sequence]].read(stream),
                &reference.delta[sequence],
                "recurrent state",
            );
            assert_eq!(
                conv[slots[sequence]].read(stream),
                reference.conv[sequence]
                    .iter()
                    .map(|&x| f16::from_f64(x))
                    .collect::<Vec<_>>()
            );
        }
        final_conv.read(stream);
        query.read(stream);
        key.read(stream);
        raw_gpu.assert_unchanged(stream);
        lengths_gpu.assert_unchanged(stream);
        sequence_gpu.assert_unchanged(stream);
        bindings.assert_unchanged(stream);
        weight_gpu.assert_unchanged(stream);
        decay_gpu.assert_unchanged(stream);
        bias_gpu.assert_unchanged(stream);
        conv[1].assert_unchanged(stream);
        states[1].assert_unchanged(stream);
    }
    (outputs, states.iter().map(|s| s.read(stream)).collect())
}

#[test]
#[ignore = "requires an actual CUDA device"]
fn recurrent_cuda_semantics_preserve_f64_oracle_state_carry_and_isolated_slots() {
    let runtime = CudaDeviceRuntime::new(
        cuda_vnext_runtime_config(
            0,
            DeviceId::new("device.test.recurrent").unwrap(),
            AttentionExecutionPolicy::default(),
        )
        .unwrap(),
    )
    .expect("recurrent conformance requires CUDA");
    let provider = CudaGatedDeltaRecurrentAttentionProvider::new(&runtime).unwrap();
    let stream = runtime.context().default_stream();
    for (key_dim, value_dim) in [(3, 5), (32, 17), (128, 128)] {
        for decay in GatedDeltaDecayParameterization::ALL {
            for mapping in GatedDeltaValueHeadMapping::ALL {
                let mut shape = super::tests::test_shape();
                shape.key_head_dim = key_dim;
                shape.value_head_dim = value_dim;
                shape.value_heads = 6;
                shape.value_features = shape.value_heads * value_dim;
                shape.qkv_features = 2 * shape.key_heads * key_dim + shape.value_features;
                shape.qkvz_features = shape.qkv_features + shape.value_features;
                shape.ba_features = 2 * shape.value_heads;
                shape.qkvzba_features = shape.qkvz_features + shape.ba_features;
                shape.decay_parameterization = decay;
                shape.value_head_mapping = mapping;
                let split = exercise(&stream, &provider.functions, shape, &[[0, 1, 4], [0, 1, 1]]);
                let whole = exercise(&stream, &provider.functions, shape, &[[0, 2, 5]]);
                assert_eq!(
                    split, whole,
                    "prefill/decode state carry differs: {shape:?}"
                );
            }
        }
    }
}
