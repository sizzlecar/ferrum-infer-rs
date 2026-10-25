//! Opt-in managed image operator timings, with analytical full-output checks.
//! Run `cargo test --profile release-thin -p ferrum-models --features metal
//! --test vnext_tensor_metal_runtime managed_image_kernel_microbench --
//! --ignored --nocapture --test-threads=1` after coordinating GPU access.
//! Timings include admission, encoding, device completion, readback and the
//! scoped attention diagnostic logging, but exclude input construction and
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
    CrossAttention {
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
    SpatialAttention {
        channels: u64,
        side: u64,
    },
}

impl Case {
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
            Self::PaddedConv { channels, side } => (
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
                        SemanticValue::Text("implicit_zero".into()),
                    ),
                ]),
            ),
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
            Self::Conv { channels, .. } | Self::PaddedConv { channels, .. } => match port {
                0 => 1.0 / 64.0,
                1 => (((index / (channels * 9)) % 7) as i32 - 3) as f32 / 256.0,
                _ => (index % 3) as f32 / 128.0,
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
            Self::Attention { .. } | Self::CrossAttention { .. } => self.value(2, index),
            Self::Conv { channels, side } => {
                let channel = index / (side * side);
                rounds(channels as f32 * 9.0 / 64.0 * self.value(1, channel * channels * 9))
                    + self.value(2, channel)
            }
            Self::PaddedConv { channels, side } => {
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

fn run_microbench(cases: &[Case]) {
    // This thread-local scope reports attention capability and actual dispatch
    // selection without installing a global subscriber or reading an env filter.
    // Dispatch diagnostics run inside the measured execute call below.
    let subscriber = tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::new(
            "off,ferrum_kernels::backend::metal::vnext_ops::tensor::attention=debug",
        ))
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
        let kind = case.fixture().kind();
        if !kinds.contains(&kind) {
            kinds.push(kind);
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
                512 << 20,
                HostMemoryClassLimits {
                    input_preprocess_bytes: 256 << 20,
                    output_bytes: 256 << 20,
                    transfer_staging_bytes: 0,
                    weight_backing_bytes: 0,
                },
            )
            .unwrap(),
        )
        .unwrap();

    // Image-token projections/FFN, generation/edit attention lengths and VAE
    // channel/spatial regimes. Synthetic data isolates costs, not model quality.
    for (number, case) in cases.iter().enumerate() {
        let fixture = case.fixture();
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
            let start = Instant::now();
            let output = session
                .execute(
                    id(format!("run.image.microbench.{number}.{repetition}")),
                    id(format!("request.image.microbench.{number}.{repetition}")),
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
        eprintln!(
            "image_kernel_microbench={}",
            serde_json::json!({
                "operation": format!("{:?}", fixture.kind()), "inputs": fixture.inputs,
                "output": fixture.output, "warmup_ms": timings[0], "measured_ms": &timings[1..],
                "timing_scope": "managed_execute_admission_encode_wait_readback_including_attention_logging",
                "runtime_fingerprint": runtime.descriptor().runtime_implementation_fingerprint,
                "native_memory": runtime.native_memory_snapshot(),
            })
        );
    }
}
