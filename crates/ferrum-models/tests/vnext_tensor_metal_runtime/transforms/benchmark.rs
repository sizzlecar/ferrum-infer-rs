//! Opt-in managed image operator timings, with analytical full-output checks.
//! Run `cargo test --profile release-thin -p ferrum-models --features metal
//! --test vnext_tensor_metal_runtime managed_image_kernel_microbench --
//! --ignored --nocapture --test-threads=1` after coordinating GPU access.
//! Timings include admission, encoding, device completion, readback and the
//! scoped attention and linear diagnostic logging, but exclude input construction and
//! output checking. They are not kernel-only timestamps, a trained-model
//! benchmark or a performance acceptance threshold.

use super::*;
use std::time::Instant;

enum Case {
    Linear {
        rows: u64,
        inputs: u64,
        outputs: u64,
    },
    Attention {
        sequence: u64,
        heads: u64,
        width: u64,
    },
    NonuniformAttention {
        sequence: u64,
        heads: u64,
        width: u64,
    },
    CrossAttention {
        queries: u64,
        keys: u64,
        heads: u64,
        width: u64,
    },
    NonuniformCrossAttention {
        queries: u64,
        keys: u64,
        heads: u64,
        width: u64,
    },
    Conv {
        channels: u64,
        side: u64,
    },
    PaddedConv {
        channels: u64,
        side: u64,
    },
    ExplicitPaddedConv {
        channels: u64,
        side: u64,
    },
    ChannelL2 {
        channels: u64,
        side: u64,
    },
    SpatialAttention {
        channels: u64,
        side: u64,
    },
}

impl Case {
    fn fixture_for_profile(&self, profile: AttentionBenchmarkProfile) -> Fixture {
        let fixture = self.fixture();
        if profile == AttentionBenchmarkProfile::Reference {
            return fixture;
        }
        let kind = match fixture.kind() {
            MetalTensorOperation::StructuredAttention => {
                MetalTensorOperation::StructuredAttentionFused
            }
            MetalTensorOperation::CrossAttention => MetalTensorOperation::CrossAttentionFused,
            _ => panic!("fused attention benchmark requires an attention fixture"),
        };
        let mut fused = Fixture::new(kind, fixture.inputs, fixture.attributes);
        fused.output = fixture.output;
        fused
    }

    fn fixture(&self) -> Fixture {
        let bf16 = |shape| spec(shape, ElementType::Bf16);
        let (kind, inputs, output, attributes) = match *self {
            Self::Linear {
                rows,
                inputs,
                outputs,
            } => (
                MetalTensorOperation::AffineLinear,
                vec![
                    bf16(vec![rows, inputs]),
                    bf16(vec![outputs, inputs]),
                    bf16(vec![outputs]),
                ],
                bf16(vec![rows, outputs]),
                BTreeMap::from([
                    (id("in_features"), SemanticValue::Unsigned(inputs)),
                    (id("out_features"), SemanticValue::Unsigned(outputs)),
                ]),
            ),
            Self::Attention {
                sequence,
                heads,
                width,
            }
            | Self::NonuniformAttention {
                sequence,
                heads,
                width,
            } => {
                let shape = bf16(vec![sequence, heads * width]);
                (
                    MetalTensorOperation::StructuredAttention,
                    vec![shape.clone(); 3],
                    shape,
                    BTreeMap::from([
                        (id("sequence_length"), SemanticValue::Unsigned(sequence)),
                        (id("head_dim"), SemanticValue::Unsigned(width)),
                        (
                            id("scale"),
                            SemanticValue::Rational(CanonicalRational::new(1, 8).unwrap()),
                        ),
                        (
                            id("segment_lengths"),
                            SemanticValue::Integers(vec![sequence as i64]),
                        ),
                        (
                            id("segment_bidirectional"),
                            SemanticValue::Integers(vec![1]),
                        ),
                        (id("key_run_counts"), SemanticValue::Integers(vec![1])),
                        (
                            id("key_run_lengths"),
                            SemanticValue::Integers(vec![sequence as i64]),
                        ),
                        (id("key_run_valid"), SemanticValue::Integers(vec![1])),
                    ]),
                )
            }
            Self::CrossAttention {
                queries,
                keys,
                heads,
                width,
            }
            | Self::NonuniformCrossAttention {
                queries,
                keys,
                heads,
                width,
            } => (
                MetalTensorOperation::CrossAttention,
                [queries, keys, keys]
                    .into_iter()
                    .map(|rows| bf16(vec![rows, heads * width]))
                    .collect(),
                bf16(vec![queries, heads * width]),
                BTreeMap::from([
                    (id("batch"), SemanticValue::Unsigned(1)),
                    (id("head_dim"), SemanticValue::Unsigned(width)),
                    (
                        id("scale"),
                        SemanticValue::Rational(CanonicalRational::new(1, 8).unwrap()),
                    ),
                    (id("key_run_counts"), SemanticValue::Integers(vec![1])),
                    (
                        id("key_run_lengths"),
                        SemanticValue::Integers(vec![keys as i64]),
                    ),
                    (id("key_run_valid"), SemanticValue::Integers(vec![1])),
                ]),
            ),
            Self::Conv { channels, side } => (
                MetalTensorOperation::Conv2d,
                vec![
                    bf16(vec![1, channels, side + 2, side + 2]),
                    bf16(vec![channels, channels, 3, 3]),
                    bf16(vec![channels]),
                ],
                bf16(vec![1, channels, side, side]),
                BTreeMap::from([
                    (id("padding"), SemanticValue::Integers(vec![0; 4])),
                    (id("stride"), SemanticValue::Integers(vec![1; 2])),
                    (id("dilation"), SemanticValue::Integers(vec![1; 2])),
                    (id("groups"), SemanticValue::Unsigned(1)),
                    (
                        id("padding_mode"),
                        SemanticValue::Text("implicit_zero".into()),
                    ),
                ]),
            ),
            Self::PaddedConv { channels, side } | Self::ExplicitPaddedConv { channels, side } => (
                MetalTensorOperation::Conv2d,
                vec![
                    bf16(vec![1, channels, side, side]),
                    bf16(vec![channels, channels, 3, 3]),
                    bf16(vec![channels]),
                ],
                bf16(vec![1, channels, side, side]),
                BTreeMap::from([
                    (id("padding"), SemanticValue::Integers(vec![1; 4])),
                    (id("stride"), SemanticValue::Integers(vec![1; 2])),
                    (id("dilation"), SemanticValue::Integers(vec![1; 2])),
                    (id("groups"), SemanticValue::Unsigned(1)),
                    (
                        id("padding_mode"),
                        SemanticValue::Text(
                            if matches!(self, Self::ExplicitPaddedConv { .. }) {
                                "explicit_zero"
                            } else {
                                "implicit_zero"
                            }
                            .into(),
                        ),
                    ),
                ]),
            ),
            Self::ChannelL2 { channels, side } => {
                // Match the decoder's effective F32 sqrt(C) scale exactly.
                let scale_bits = ((channels as f64).sqrt() as f32).to_bits();
                let scale = CanonicalRational::new(
                    i64::from((scale_bits & 0x7fffff) | (1 << 23)),
                    1_u64 << (150 - ((scale_bits >> 23) & 255)),
                )
                .unwrap();
                let image = bf16(vec![1, channels, side, side]);
                (
                    MetalTensorOperation::ChannelL2Norm,
                    vec![image.clone(), bf16(vec![channels, 1, 1, 1])],
                    image,
                    BTreeMap::from([
                        (
                            id("epsilon"),
                            SemanticValue::Rational(
                                CanonicalRational::new(1, 1_000_000_000_000).unwrap(),
                            ),
                        ),
                        (id("scale"), SemanticValue::Rational(scale)),
                    ]),
                )
            }
            Self::SpatialAttention { channels, side } => (
                MetalTensorOperation::SpatialAttention,
                vec![bf16(vec![1, 3 * channels, side, side])],
                bf16(vec![1, channels, side, side]),
                BTreeMap::from([(
                    id("scale"),
                    SemanticValue::Rational(CanonicalRational::new(1, 32).unwrap()),
                )]),
            ),
        };
        let mut fixture = Fixture::new(kind, inputs, attributes);
        fixture.output = Some(output);
        fixture
    }

    fn value(&self, port: usize, index: u64) -> f32 {
        match *self {
            Self::Linear { inputs, .. } => match port {
                // Sparse activations give a cheap independent oracle while the
                // selected provider still executes the full dense matrix work.
                0 if index % inputs == ((index / inputs) * 17 + 3) % inputs => {
                    ((index / inputs) % 5 + 1) as f32 / 16.0
                }
                1 => (((index / inputs * 7 + index % inputs * 13) % 31) as i32 - 15) as f32 / 32.0,
                _ => 0.0,
            },
            Self::Attention { heads, width, .. } | Self::CrossAttention { heads, width, .. } => {
                match port {
                    0 => 0.0,
                    1 => ((index % 7) as i32 - 3) as f32 / 16.0,
                    // Each channel is constant over keys: every query must return
                    // the same channel value after the full online softmax/PV.
                    _ => (((index % (heads * width)) % 17) as i32 - 8) as f32 / 16.0,
                }
            }
            Self::NonuniformAttention { heads, width, .. }
            | Self::NonuniformCrossAttention { heads, width, .. } => match port {
                0 | 1 => ((index % 7) as i32 - 3) as f32 / 2.0,
                // Keep the original constant-channel V and exact output oracle.
                _ => (((index % (heads * width)) % 17) as i32 - 8) as f32 / 16.0,
            },
            Self::Conv { channels, .. }
            | Self::PaddedConv { channels, .. }
            | Self::ExplicitPaddedConv { channels, .. } => match port {
                0 => 1.0 / 64.0,
                1 => (((index / (channels * 9)) % 7) as i32 - 3) as f32 / 256.0,
                _ => (index % 3) as f32 / 128.0,
            },
            Self::ChannelL2 { .. } => match port {
                // Signed pixels and varying gamma retain an analytical norm:
                // every squared input is exactly 1/4096, including BF16 input.
                0 => (if index % 2 == 0 { 1.0 } else { -1.0 }) / 64.0,
                _ => ((index % 7) as i32 - 3) as f32 / 8.0,
            },
            Self::SpatialAttention { channels, side } => {
                let channel = index / (side * side);
                if channel < channels {
                    0.0
                } else if channel < 2 * channels {
                    ((index % 7) as i32 - 3) as f32 / 16.0
                } else {
                    (((channel - 2 * channels) % 17) as i32 - 8) as f32 / 16.0
                }
            }
        }
    }

    fn expected(&self, index: u64) -> u16 {
        let value = match *self {
            Self::Linear {
                inputs, outputs, ..
            } => {
                let row = index / outputs;
                let inner = (row * 17 + 3) % inputs;
                self.value(0, row * inputs + inner)
                    * self.value(1, (index % outputs) * inputs + inner)
            }
            Self::Attention { .. }
            | Self::CrossAttention { .. }
            | Self::NonuniformAttention { .. }
            | Self::NonuniformCrossAttention { .. } => self.value(2, index),
            Self::Conv { channels, side } => {
                let channel = index / (side * side);
                rounds(channels as f32 * 9.0 / 64.0 * self.value(1, channel * channels * 9))
                    + self.value(2, channel)
            }
            Self::PaddedConv { channels, side } | Self::ExplicitPaddedConv { channels, side } => {
                let channel = index / (side * side);
                let y = (index / side) % side;
                let x = index % side;
                // Constant input/weights make every output independently
                // checkable, including four-/six-tap corners and edges.
                let rows = 1 + u64::from(y > 0) + u64::from(y + 1 < side);
                let columns = 1 + u64::from(x > 0) + u64::from(x + 1 < side);
                rounds(
                    channels as f32 * (rows * columns) as f32 / 64.0
                        * self.value(1, channel * channels * 9),
                ) + self.value(2, channel)
            }
            Self::ChannelL2 { channels, side } => {
                let norm = (channels as f64 / 4096.0).sqrt() as f32;
                let scale = (channels as f64).sqrt() as f32;
                let normalized = rounds(self.value(0, index) / norm);
                let scaled = rounds(normalized * scale);
                rounds(scaled * self.value(1, index / (side * side))) + 0.0
            }
            Self::SpatialAttention { channels, side } => {
                self.value(0, 2 * channels * side * side + index)
            }
        };
        bf16::from_f32(value).to_bits()
    }
}

#[test]
#[ignore = "real Metal image-shape timing; requires exclusive GPU access"]
fn managed_image_kernel_microbench() {
    run_microbench(&[
        Case::Linear {
            rows: 4096,
            inputs: 4096,
            outputs: 4096,
        },
        Case::Linear {
            rows: 4096,
            inputs: 4096,
            outputs: 12288,
        },
        Case::Attention {
            sequence: 4096,
            heads: 32,
            width: 128,
        },
        Case::Attention {
            sequence: 8192,
            heads: 32,
            width: 128,
        },
        Case::Conv {
            channels: 512,
            side: 128,
        },
        Case::Conv {
            channels: 128,
            side: 512,
        },
    ]);
}

#[test]
#[ignore = "real Metal rectangular image attention timing; requires exclusive GPU access"]
fn managed_cross_attention_microbench() {
    // Cached image editing has fewer query tokens than retained key tokens.
    // Keep output validation independent of either implementation and include
    // a tail plus a square-attention control using the same measured scope.
    run_microbench(&[
        Case::CrossAttention {
            queries: 4096,
            keys: 8192,
            heads: 32,
            width: 128,
        },
        Case::CrossAttention {
            queries: 4096,
            keys: 8228,
            heads: 32,
            width: 128,
        },
        Case::Attention {
            sequence: 4096,
            heads: 32,
            width: 128,
        },
    ]);
}

#[test]
#[ignore = "real Metal nonuniform attention timing; requires exclusive GPU access"]
fn managed_nonuniform_attention_microbench() {
    // Nonzero, exact BF16 half-integer Q/K exercise nonzero exponent arguments.
    // Constant-channel V still gives the original exact full-output oracle.
    // Both implementations must use this same harness and input construction.
    run_microbench(&nonuniform_attention_cases());
}

#[test]
#[ignore = "real Metal paired production attention timing; requires exclusive GPU access"]
fn managed_nonuniform_attention_fused_paired_microbench() {
    // Both profiles use the same compiled binary, runtime budget and inputs.
    // Each A/B/B/A group warms once and measures three executions; validation
    // covers every output, including warmups. The fused timing includes its
    // prepass, dense dispatch and strict fallback dispatch, not just the core.
    run_microbench_schedule(
        &nonuniform_attention_cases(),
        MicrobenchDiagnostics::DispatchSelection,
        MicrobenchSchedule::PairedAttention,
    );
}

#[test]
fn paired_attention_profiles_preserve_fixture_precision_layout_and_data() {
    for case in nonuniform_attention_cases() {
        let reference = case.fixture_for_profile(AttentionBenchmarkProfile::Reference);
        let fused = case.fixture_for_profile(AttentionBenchmarkProfile::FusedDense);
        assert_eq!(reference.inputs, fused.inputs);
        assert_eq!(reference.output, fused.output);
        assert_eq!(reference.attributes, fused.attributes);
        assert!(fused.inputs.iter().all(|input| {
            input.element_type == ElementType::Bf16
                && input.layout == ResolvedTensorLayout::Contiguous
        }));
        assert_eq!(
            fused.kind(),
            match reference.kind() {
                MetalTensorOperation::StructuredAttention => {
                    MetalTensorOperation::StructuredAttentionFused
                }
                MetalTensorOperation::CrossAttention => MetalTensorOperation::CrossAttentionFused,
                _ => unreachable!(),
            }
        );
        for fixture in [reference, fused] {
            TypedFamilyRegistration::new(TransformFamily)
                .prepare_with_profile(
                    &serde_json::to_value(&fixture).unwrap(),
                    &id(fixture.kind().profile()),
                )
                .unwrap();
            for (port, input) in fixture.inputs.iter().enumerate() {
                let elements = input.dimensions.iter().product::<u64>();
                for index in [0, 1, elements / 2, elements - 1] {
                    let value = case.value(port, index);
                    assert_eq!(rounds(value), value);
                }
            }
        }
    }
}

#[test]
#[ignore = "real Metal execution host timing diagnostic; requires exclusive GPU access"]
fn managed_nonuniform_attention_host_timing_diagnostic() {
    // Existing execute_bound wall timers include host encoding/submission and
    // completion wait plus readback. They are not GPU timestamps, and this
    // extra logging must not be mixed into the original benchmark samples.
    run_microbench_with_diagnostics(
        &nonuniform_attention_cases(),
        MicrobenchDiagnostics::ExecutionHost,
    );
}

fn nonuniform_attention_cases() -> [Case; 3] {
    [
        Case::NonuniformAttention {
            sequence: 4096,
            heads: 32,
            width: 128,
        },
        Case::NonuniformAttention {
            sequence: 8192,
            heads: 32,
            width: 128,
        },
        Case::NonuniformCrossAttention {
            queries: 4096,
            keys: 8228,
            heads: 32,
            width: 128,
        },
    ]
}

#[test]
fn nonuniform_attention_inputs_have_distinct_nonzero_scores() {
    let cases = [
        Case::NonuniformAttention {
            sequence: 7,
            heads: 32,
            width: 128,
        },
        Case::NonuniformCrossAttention {
            queries: 3,
            keys: 7,
            heads: 32,
            width: 128,
        },
    ];
    for case in cases {
        let fixture = case.fixture();
        let columns = fixture.inputs[0].dimensions[1];
        let keys = fixture.inputs[1].dimensions[0];
        let width = 128;
        assert!((0..width).any(|channel| case.value(0, channel) != 0.0));
        let scores: Vec<f64> = (0..keys)
            .map(|key| {
                (0..width)
                    .map(|channel| {
                        let q = case.value(0, channel);
                        let k = case.value(1, key * columns + channel);
                        assert_eq!(rounds(q).to_bits(), q.to_bits());
                        assert_eq!(rounds(k).to_bits(), k.to_bits());
                        f64::from(q) * f64::from(k)
                    })
                    .sum::<f64>()
                    / 8.0
            })
            .collect();
        assert!(scores.iter().all(|score| score.is_finite()));
        assert!(scores.iter().any(|score| *score != 0.0));
        assert!(scores.iter().any(|score| *score != scores[0]));
    }
}

#[test]
#[ignore = "real Metal implicit-padding convolution timing; requires exclusive GPU access"]
fn managed_masked_conv_microbench() {
    // True padding=1 selects the bounded masked path when it is supported.
    // Keep the existing six-case control benchmark unchanged.
    run_microbench(&[
        Case::PaddedConv {
            channels: 512,
            side: 128,
        },
        Case::PaddedConv {
            channels: 128,
            side: 512,
        },
    ]);
}

#[test]
#[ignore = "real Metal VAE convolution and channel norm timing; requires exclusive GPU access"]
fn managed_vae_real_shape_microbench() {
    // Decoder base=144 and multipliers=[1,2,4,8,8]. Include both the
    // residual's explicit pad+conv and the upsampler's implicit-pad path.
    // These isolated operators do not replace a complete decoder profile.
    let cases: Vec<_> = [(1152, 128), (144, 1024)]
        .into_iter()
        .flat_map(|(channels, side)| {
            [
                Case::ExplicitPaddedConv { channels, side },
                Case::PaddedConv { channels, side },
                Case::ChannelL2 { channels, side },
            ]
        })
        .collect();
    run_microbench_with_budget(
        &cases,
        MicrobenchDiagnostics::VaeOperators,
        MicrobenchSchedule::Original,
        MicrobenchBudget::VaeRealShape,
    );
}

#[test]
fn vae_channel_norm_analytical_output_matches_independent_oracle() {
    // Use actual channel counts but a tiny spatial domain for the CPU check.
    for channels in [144, 1152] {
        let case = Case::ChannelL2 { channels, side: 3 };
        let fixture = case.fixture();
        TypedFamilyRegistration::new(TransformFamily)
            .prepare_with_profile(
                &serde_json::to_value(&fixture).unwrap(),
                &id(fixture.kind().profile()),
            )
            .unwrap();
        let values: Vec<Vec<f32>> = fixture
            .inputs
            .iter()
            .enumerate()
            .map(|(port, spec)| {
                (0..spec.dimensions.iter().product::<u64>())
                    .map(|index| case.value(port, index))
                    .collect()
            })
            .collect();
        let output = fixture.output.as_ref().unwrap();
        let expected: Vec<_> = (0..output.dimensions.iter().product::<u64>())
            .map(|index| bf16::from_bits(case.expected(index)).to_f32())
            .collect();
        super::super::spatial::check(
            fixture.kind(),
            &fixture.attributes,
            &fixture
                .inputs
                .iter()
                .zip(&values)
                .map(|(spec, values)| (spec, values.as_slice()))
                .collect::<Vec<_>>(),
            output,
            &expected,
        );
    }
}

#[test]
#[ignore = "real Metal matrix-tail timing; requires exclusive GPU access"]
fn managed_linear_tail_microbench() {
    // Row, column and combined tails exercise disjoint interior/edge writes.
    // These are operator diagnostics, not complete image-edit measurements.
    run_microbench(&[
        Case::Linear {
            rows: 4112,
            inputs: 4096,
            outputs: 4096,
        },
        Case::Linear {
            rows: 4096,
            inputs: 4096,
            outputs: 4097,
        },
        Case::Linear {
            rows: 8208,
            inputs: 4096,
            outputs: 12289,
        },
    ]);
}

#[test]
#[ignore = "real Metal spatial attention timing; requires exclusive GPU access"]
fn managed_vae_attention_microbench() {
    // Smaller spatial grid for hotspot diagnosis; this is not a complete VAE
    // or a substitute for full-resolution end-to-end image measurements.
    run_microbench(&[Case::SpatialAttention {
        channels: 512,
        side: 64,
    }]);
}

#[derive(Clone, Copy)]
enum MicrobenchDiagnostics {
    DispatchSelection,
    ExecutionHost,
    VaeOperators,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AttentionBenchmarkProfile {
    Reference,
    FusedDense,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum MicrobenchSchedule {
    Original,
    PairedAttention,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum MicrobenchBudget {
    Original,
    VaeRealShape,
}

impl MicrobenchBudget {
    fn host_class_bytes(self) -> usize {
        match self {
            Self::Original => 256 << 20,
            // C144x1024x1024 BF16 is 288 MiB for input and output each.
            // Input also owns the conv weights/bias. The separate Device
            // allowance remains 1 GiB, covering input, output and pad scratch.
            Self::VaeRealShape => 384 << 20,
        }
    }
}

impl MicrobenchSchedule {
    fn profiles(self) -> &'static [AttentionBenchmarkProfile] {
        use AttentionBenchmarkProfile::*;
        match self {
            Self::Original => &[Reference],
            Self::PairedAttention => &[Reference, FusedDense, FusedDense, Reference],
        }
    }
}

impl MicrobenchDiagnostics {
    fn filter(self) -> &'static str {
        match self {
            Self::VaeOperators => {
                "off,ferrum_kernels::backend::metal::vnext_ops::tensor::spatial::conv=info"
            }
            Self::DispatchSelection => {
                "off,ferrum_kernels::backend::metal::vnext_ops::tensor::attention=debug,ferrum_kernels::backend::metal::vnext_ops::tensor::linear=debug"
            }
            Self::ExecutionHost => {
                "off,ferrum_kernels::backend::metal::vnext_ops::tensor::attention=debug,ferrum_kernels::backend::metal::vnext_ops::tensor::linear=debug,ferrum_execution::tensor=debug"
            }
        }
    }

    fn timing_scope(self) -> &'static str {
        match self {
            Self::VaeOperators => {
                "managed_execute_admission_encode_wait_readback_including_conv_dispatch_logging_not_gpu_timestamps"
            }
            Self::DispatchSelection => {
                "managed_execute_admission_encode_wait_readback_including_attention_and_linear_logging"
            }
            Self::ExecutionHost => {
                "managed_execute_admission_encode_wait_readback_including_attention_linear_and_execution_host_logging_not_gpu_timestamps"
            }
        }
    }
}

fn run_microbench(cases: &[Case]) {
    run_microbench_with_diagnostics(cases, MicrobenchDiagnostics::DispatchSelection);
}

fn run_microbench_with_diagnostics(cases: &[Case], diagnostics: MicrobenchDiagnostics) {
    run_microbench_schedule(cases, diagnostics, MicrobenchSchedule::Original);
}

fn run_microbench_schedule(
    cases: &[Case],
    diagnostics: MicrobenchDiagnostics,
    schedule: MicrobenchSchedule,
) {
    run_microbench_with_budget(cases, diagnostics, schedule, MicrobenchBudget::Original);
}

fn run_microbench_with_budget(
    cases: &[Case],
    diagnostics: MicrobenchDiagnostics,
    schedule: MicrobenchSchedule,
    budget: MicrobenchBudget,
) {
    // This thread-local scope reports attention/linear capability and actual dispatch
    // selection without installing a global subscriber or reading an env filter.
    // Dispatch diagnostics run inside the measured execute call below.
    let subscriber = tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::new(diagnostics.filter()))
        .with_writer(std::io::stderr)
        .with_ansi(false)
        .without_time()
        .finish();
    let _diagnostics = tracing::subscriber::set_default(subscriber);
    const DEVICE_BYTES: u64 = 1 << 30;
    let runtime = Arc::new(
        MetalDeviceRuntime::new_tensor_eager(MetalTensorRuntimeConfig {
            device: metal_tensor_runtime_config(id("device.metal.0")).unwrap(),
            usable_memory_bytes: DEVICE_BYTES,
            bootstrap_memory_bytes: 16 << 20,
            stream_memory_bytes: 1 << 20,
            maximum_streams: 2,
            maximum_in_flight_per_stream: 1,
            maximum_commands_per_submission: 8,
            opaque_memory_profile: "metal.image.microbench.declared.v1".into(),
        })
        .unwrap(),
    );
    let selected = cases.iter().fold(Vec::new(), |mut kinds, case| {
        for &profile in schedule.profiles() {
            let kind = case.fixture_for_profile(profile).kind();
            if !kinds.contains(&kind) {
                kinds.push(kind);
            }
        }
        kinds
    });
    let registry = OperationRuntimeRegistry::new(
        selected
            .iter()
            .map(|kind| Box::new(kind.contract().unwrap()) as Box<dyn OperationContract>)
            .collect(),
        metal_tensor_providers(&runtime, &selected).unwrap(),
    )
    .unwrap();
    let engine = EngineProviderDescriptor::new(
        id("provider.engine.metal.image.microbench"),
        ContractVersion::new(1, 0),
        runtime
            .descriptor()
            .runtime_implementation_fingerprint
            .clone(),
        runtime.descriptor().id.clone(),
        runtime.descriptor().capabilities.clone(),
    )
    .unwrap();
    let catalog = registry
        .capability_catalog(runtime.descriptor().clone(), vec![engine])
        .unwrap();
    let policy = TensorComponentPolicy::new(
        DEVICE_BYTES,
        runtime
            .descriptor()
            .dynamic_storage_profiles
            .iter()
            .copied()
            .collect(),
    )
    .unwrap();
    let host = runtime
        .tensor_host_account(
            HostMemoryLimits::new(
                2 * budget.host_class_bytes(),
                HostMemoryClassLimits {
                    input_preprocess_bytes: budget.host_class_bytes(),
                    output_bytes: budget.host_class_bytes(),
                    transfer_staging_bytes: 0,
                    weight_backing_bytes: 0,
                },
            )
            .unwrap(),
        )
        .unwrap();

    // Image-token projections/FFN, generation/edit attention lengths and VAE
    // channel/spatial regimes. Synthetic data isolates costs, not model quality.
    let runs = cases.iter().enumerate().flat_map(|(number, case)| {
        schedule
            .profiles()
            .iter()
            .copied()
            .enumerate()
            .map(move |(group, profile)| (number, case, group, profile))
    });
    for (number, case, group, profile) in runs {
        let fixture = case.fixture_for_profile(profile);
        let prepared = TypedFamilyRegistration::new(TransformFamily)
            .prepare_with_profile(
                &serde_json::to_value(&fixture).unwrap(),
                &id(fixture.kind().profile()),
            )
            .unwrap();
        let mut options = ProgramPlanCompileOptions::new(fixture.specs()).unwrap();
        options
            .insert_tensor_spec(id(OUTPUT), fixture.output.clone().unwrap())
            .unwrap();
        let component = CompiledTensorComponent::compile(
            prepared,
            &registry,
            &catalog,
            &policy,
            &WeightMaterializerRegistry::identity_only().unwrap(),
            &options,
        )
        .unwrap();
        let scratch_admitted = component.executable().execution_plan().payload().nodes()[0]
            .provider_resources()
            .scratch()
            .is_some();
        if profile == AttentionBenchmarkProfile::FusedDense {
            // Supported finite-domain fixtures must select the production
            // guarded schedule, not silently benchmark an unavailable PSO's
            // no-scratch strict compatibility fallback.
            assert!(scratch_admitted, "production fused schedule is unavailable");
        }
        let mut session = component
            .initialize(
                runtime.clone(),
                host.clone(),
                ComponentWeightSource::None,
                initialization("metal.image.microbench"),
            )
            .unwrap();
        let inputs: Vec<_> = fixture
            .inputs
            .iter()
            .enumerate()
            .map(|(port, shape)| {
                let elements = shape.dimensions.iter().product::<u64>();
                let mut bytes = host
                    .allocate_zeroed(HostMemoryClass::InputPreprocess, (elements * 2) as usize)
                    .unwrap();
                for (index, word) in bytes.as_mut().chunks_exact_mut(2).enumerate() {
                    word.copy_from_slice(
                        &bf16::from_f32(case.value(port, index as u64))
                            .to_bits()
                            .to_le_bytes(),
                    );
                }
                ManagedHostTensor::new(
                    input_id(port),
                    ResolvedTensorSpec::new(
                        shape.dimensions.clone(),
                        shape.element_type,
                        shape.layout.clone(),
                    )
                    .unwrap(),
                    bytes.freeze(),
                )
                .unwrap()
            })
            .collect();
        let mut timings = Vec::new();
        for repetition in 0..4 {
            let input = inputs.clone();
            let suffix = if schedule == MicrobenchSchedule::Original {
                format!("{number}.{repetition}")
            } else {
                format!("paired.{number}.{group}.{repetition}")
            };
            let start = Instant::now();
            let output = session
                .execute(
                    id(format!("run.image.microbench.{suffix}")),
                    id(format!("request.image.microbench.{suffix}")),
                    input,
                )
                .unwrap();
            timings.push(start.elapsed().as_secs_f64() * 1000.0);
            assert_eq!(output.len(), 1);
            let bytes = output[0].bytes().bytes();
            assert_eq!(
                bytes.len() as u64,
                fixture
                    .output
                    .as_ref()
                    .unwrap()
                    .dimensions
                    .iter()
                    .product::<u64>()
                    * 2
            );
            for (index, word) in bytes.chunks_exact(2).enumerate() {
                assert_eq!(
                    u16::from_le_bytes(word.try_into().unwrap()),
                    case.expected(index as u64),
                    "{:?} output {index}",
                    fixture.kind()
                );
            }
        }
        drop(inputs);
        assert!(matches!(
            session
                .close()
                .unwrap_or_else(|e| panic!("close: {:?}", e.failure())),
            PlanRuntimeCloseOutcome::Closed(_)
        ));
        assert_eq!(host.usage().unwrap().committed_bytes(), 0);
        if budget == MicrobenchBudget::VaeRealShape {
            eprintln!(
                "vae_operator_microbench={}",
                serde_json::json!({
                    "operation": format!("{:?}", fixture.kind()),
                    "profile": fixture.kind().profile(),
                    "inputs": fixture.inputs, "output": fixture.output,
                    "attributes": fixture.attributes,
                    "warmup_ms": timings[0], "measured_ms": &timings[1..],
                    "measured_samples": timings.len() - 1,
                    "full_output_checks": timings.len(), "errors": 0,
                    "oracle": "analytical_exact_bf16_full_output",
                    "timing_scope": "managed_execute_admission_encode_all_dispatches_wait_readback_including_conv_dispatch_logging;materialized_explicit_conv_includes_pad_dispatch;not_kernel_gpu_timestamps_or_complete_decoder_profile",
                    "provider_scratch_admitted": scratch_admitted,
                    "declared_device_bytes": DEVICE_BYTES,
                    "declared_host_total_bytes": 2 * budget.host_class_bytes(),
                    "declared_host_input_bytes": budget.host_class_bytes(),
                    "declared_host_output_bytes": budget.host_class_bytes(),
                    "host_committed_bytes_after_close": host.usage().unwrap().committed_bytes(),
                    "runtime_fingerprint": runtime.descriptor().runtime_implementation_fingerprint,
                    "native_memory": runtime.native_memory_snapshot(),
                })
            );
        } else if schedule == MicrobenchSchedule::Original {
            eprintln!(
                "image_kernel_microbench={}",
                serde_json::json!({
                    "operation": format!("{:?}", fixture.kind()), "inputs": fixture.inputs,
                    "output": fixture.output, "warmup_ms": timings[0], "measured_ms": &timings[1..],
                    "timing_scope": diagnostics.timing_scope(),
                    "runtime_fingerprint": runtime.descriptor().runtime_implementation_fingerprint,
                    "native_memory": runtime.native_memory_snapshot(),
                })
            );
        } else {
            eprintln!(
                "image_attention_paired_microbench={}",
                serde_json::json!({
                    "case": number, "group": group,
                    "order": "Reference/FusedDense/FusedDense/Reference",
                    "selection": format!("{profile:?}"),
                    "operation": format!("{:?}", fixture.kind()),
                    "profile": fixture.kind().profile(),
                    "inputs": fixture.inputs, "output": fixture.output,
                    "warmup_ms": timings[0], "measured_ms": &timings[1..],
                    "measured_samples": timings.len() - 1,
                    "full_output_checks": timings.len(), "errors": 0,
                    "oracle": "exact_bf16_constant_channel_value_full_output",
                    "timing_scope": "managed_execute_admission_encode_all_dispatches_wait_readback_including_attention_and_linear_logging;fused_includes_guard_dense_and_strict_fallback;not_kernel_gpu_timestamps",
                    "provider_scratch_admitted": scratch_admitted,
                    "host_committed_bytes_after_close": host.usage().unwrap().committed_bytes(),
                    "runtime_fingerprint": runtime.descriptor().runtime_implementation_fingerprint,
                    "native_memory": runtime.native_memory_snapshot(),
                })
            );
        }
    }
}
