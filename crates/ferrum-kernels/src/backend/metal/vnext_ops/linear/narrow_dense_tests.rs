//! Dense small-output candidates: independent numerical checks and opt-in timing.
use super::*;
use half::f16;
use metal::{Buffer, CommandQueueRef, MTLCommandBufferStatus, MTLResourceOptions};

const INPUT_PREFIX: usize = 8;
const WEIGHT_PREFIX: usize = 9;
const OUTPUT_PREFIX: usize = 8;
const GUARD: f32 = -123.0;

#[derive(Clone, Copy, Debug)]
enum Variant {
    Original,
    Threads128,
    Threads256,
}

const VARIANTS: [Variant; 3] = [Variant::Original, Variant::Threads128, Variant::Threads256];

struct Pipelines([ComputePipelineState; 3]);

impl Pipelines {
    fn new(device: &Device) -> Self {
        // Match the existing dense provider's compiler policy exactly.
        let library = device
            .new_library_with_source(SHADER_SOURCE, &CompileOptions::new())
            .unwrap();
        Self(
            [
                (LINEAR_DENSE_KERNEL, 64),
                ("vnext_linear_dense_narrow_f16_128", 128),
                ("vnext_linear_dense_narrow_f16_256", 256),
            ]
            .map(|(name, threads)| {
                let pipeline = device
                    .new_compute_pipeline_state_with_function(
                        &library.get_function(name, None).unwrap(),
                    )
                    .unwrap();
                assert_eq!(pipeline.thread_execution_width(), 32, "{name} SIMD width");
                assert!(
                    pipeline.max_total_threads_per_threadgroup() >= threads,
                    "{name} threadgroup capacity"
                );
                assert!(
                    pipeline.static_threadgroup_memory_length()
                        <= device.max_threadgroup_memory_length(),
                    "{name} static threadgroup memory"
                );
                pipeline
            }),
        )
    }

    fn get(&self, variant: Variant) -> &ComputePipelineState {
        &self.0[match variant {
            Variant::Original => 0,
            Variant::Threads128 => 1,
            Variant::Threads256 => 2,
        }]
    }
}

fn buffer(device: &Device, values: &[f16]) -> Buffer {
    device.new_buffer_with_data(
        values.as_ptr().cast(),
        std::mem::size_of_val(values) as u64,
        MTLResourceOptions::StorageModeShared,
    )
}

fn read(buffer: &Buffer, length: usize) -> Vec<f16> {
    // SAFETY: All buffers are shared and read only after command completion.
    unsafe { std::slice::from_raw_parts(buffer.contents().cast::<f16>(), length).to_vec() }
}

struct Fixture {
    params: LinearParams,
    input: Buffer,
    weight: Buffer,
    output: Buffer,
    input_values: Vec<f16>,
    weight_values: Vec<f16>,
    output_initial: Vec<f16>,
}

impl Fixture {
    fn new(
        device: &Device,
        rows: u32,
        width: u32,
        outputs: u32,
        inputs: Vec<f16>,
        weights: Vec<f16>,
    ) -> Self {
        assert_eq!(inputs.len(), (rows * width) as usize);
        assert_eq!(weights.len(), (outputs * width) as usize);
        let params = LinearParams {
            rows,
            in_features: width,
            out_features: outputs,
            output_stride: outputs + 5,
            output_column_offset: 3,
        };
        let pad = |prefix, values: Vec<f16>| {
            let mut padded = vec![f16::from_f32(GUARD); prefix];
            padded.extend(values);
            padded.extend([f16::from_f32(GUARD); 8]);
            padded
        };
        let input_values = pad(INPUT_PREFIX, inputs);
        let weight_values = pad(WEIGHT_PREFIX, weights);
        let mut output_initial =
            vec![f16::from_f32(GUARD); OUTPUT_PREFIX + (rows * params.output_stride) as usize + 8];
        for row in 0..rows as usize {
            let start = OUTPUT_PREFIX + row * params.output_stride as usize + 3;
            output_initial[start..start + outputs as usize].fill(f16::NAN);
        }
        Self {
            params,
            input: buffer(device, &input_values),
            weight: buffer(device, &weight_values),
            output: buffer(device, &output_initial),
            input_values,
            weight_values,
            output_initial,
        }
    }

    fn generated(device: &Device, rows: u32, width: u32, outputs: u32, dyadic: bool) -> Self {
        let inputs = (0..rows * width)
            .map(|i| f16::from_f32(((i * 37 % 63) as i32 - 31) as f32 / 512.0))
            .collect();
        let weights = (0..outputs * width)
            .map(|i| {
                let coefficient = if dyadic { 1.0 / 128.0 } else { 0.0171 };
                f16::from_f32(((i * 17 % 71) as i32 - 35) as f32 * coefficient)
            })
            .collect();
        Self::new(device, rows, width, outputs, inputs, weights)
    }

    fn run(
        &self,
        pipelines: &Pipelines,
        queue: &CommandQueueRef,
        variant: Variant,
        repeats: usize,
    ) -> Option<f64> {
        self.run_dispatch(
            pipelines.get(variant),
            queue,
            repeats,
            |encoder| match variant {
                Variant::Original => {
                    dispatch_linear_grid(encoder, self.params, LinearDispatchKind::CooperativeGemv)
                }
                Variant::Threads256 => {
                    dispatch_linear_grid(encoder, self.params, LinearDispatchKind::NarrowDenseGemv)
                }
                Variant::Threads128 => encoder.dispatch_thread_groups(
                    MTLSize::new(
                        u64::from(self.params.out_features),
                        u64::from(self.params.rows),
                        1,
                    ),
                    MTLSize::new(128, 1, 1),
                ),
            },
        )
    }

    fn run_dispatch(
        &self,
        pipeline: &ComputePipelineState,
        queue: &CommandQueueRef,
        repeats: usize,
        dispatch: impl Fn(&ComputeCommandEncoderRef),
    ) -> Option<f64> {
        // SAFETY: The previous invocation has completed; no device access is outstanding.
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.output_initial.as_ptr(),
                self.output.contents().cast::<f16>(),
                self.output_initial.len(),
            );
        }
        let command = queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(&self.input), (INPUT_PREFIX * 2) as u64);
        encoder.set_buffer(1, Some(&self.weight), (WEIGHT_PREFIX * 2) as u64);
        encoder.set_buffer(2, Some(&self.output), (OUTPUT_PREFIX * 2) as u64);
        encoder.set_bytes(
            3,
            std::mem::size_of::<LinearParams>() as u64,
            &self.params as *const _ as *const c_void,
        );
        for _ in 0..repeats {
            dispatch(encoder);
        }
        encoder.end_encoding();
        command.commit();
        command.wait_until_completed();
        assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
        super::microbench::gpu_elapsed_ns(command)
    }

    fn output(&self) -> Vec<f16> {
        read(&self.output, self.output_initial.len())
    }

    fn assert_cpu(&self, exact_bits: Option<&[u16]>) {
        let actual = self.output();
        let width = self.params.in_features as usize;
        let outputs = self.params.out_features as usize;
        let input = &self.input_values[INPUT_PREFIX..];
        let weights = &self.weight_values[WEIGHT_PREFIX..];
        for row in 0..self.params.rows as usize {
            let start = OUTPUT_PREFIX + row * self.params.output_stride as usize + 3;
            for col in 0..outputs {
                let terms = input[row * width..(row + 1) * width]
                    .iter()
                    .zip(&weights[col * width..(col + 1) * width])
                    .map(|(&x, &w)| f64::from(x.to_f32()) * f64::from(w.to_f32()))
                    .collect::<Vec<_>>();
                let reference = terms.iter().sum::<f64>();
                let expected = f16::from_f64(reference);
                if let Some(bits) = exact_bits {
                    assert_eq!(
                        expected.to_bits(),
                        bits[col],
                        "literal CPU oracle column {col}"
                    );
                    assert_eq!(
                        actual[start + col].to_bits(),
                        bits[col],
                        "exact half boundary row {row} column {col}"
                    );
                } else {
                    // Existing linear forward-error allowance; no looser candidate budget.
                    let accumulation = 2.0
                        * width as f64
                        * f64::from(f32::EPSILON)
                        * terms.iter().map(|x| x.abs()).sum::<f64>();
                    let rounding = reference.abs() / 1024.0 + 1.0 / 16777216.0;
                    let observed = f64::from(actual[start + col].to_f32());
                    assert!(
                        observed.is_finite()
                            && (observed - reference).abs() <= accumulation + rounding,
                        "row {row} column {col}: {observed} != {reference}"
                    );
                }
            }
        }
        for (index, initial) in self.output_initial.iter().enumerate() {
            if !initial.is_nan() {
                assert_eq!(
                    actual[index].to_bits(),
                    initial.to_bits(),
                    "output guard {index}"
                );
            }
        }
        for (buffer, expected) in [
            (&self.input, &self.input_values),
            (&self.weight, &self.weight_values),
        ] {
            assert_eq!(
                read(buffer, expected.len())
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
                expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );
        }
    }
}

#[test]
fn narrow_dense_threadgroup_requires_exact_simd_and_device_capacity() {
    assert!(supports_narrow_dense_threadgroup(32, 256, 32, 32));
    assert!(supports_narrow_dense_threadgroup(32, 1024, 32, 32768));
    assert!(!supports_narrow_dense_threadgroup(16, 256, 32, 32768));
    assert!(!supports_narrow_dense_threadgroup(64, 256, 32, 32768));
    assert!(!supports_narrow_dense_threadgroup(32, 255, 32, 32768));
    assert!(!supports_narrow_dense_threadgroup(32, 256, 33, 32));
}

#[test]
fn narrow_dense_production_dispatch_preserves_bounds_and_capability_fallback() {
    let device = Device::system_default().expect("narrow dense dispatch requires Metal");
    let mut pipelines = MetalLinearPipelines::new(&device).unwrap();
    let queue = device.new_command_queue();
    assert!(
        pipelines.dense_narrow.is_some(),
        "test requires the optional pipeline"
    );
    for (rows, width, outputs, expected) in [
        (1, 1023, 48, LinearDispatchKind::CooperativeGemv),
        (1, 1024, 1, LinearDispatchKind::NarrowDenseGemv),
        (1, 1024, 128, LinearDispatchKind::NarrowDenseGemv),
        (1, 5120, 48, LinearDispatchKind::NarrowDenseGemv),
        (1, 5121, 49, LinearDispatchKind::NarrowDenseGemv),
        (1, 5120, 129, LinearDispatchKind::CooperativeGemv),
        (2, 5120, 48, LinearDispatchKind::CooperativeGemv),
    ] {
        let fixture = Fixture::generated(&device, rows, width, outputs, false);
        let (pipeline, kind) = pipelines.plain_linear_dispatch(
            LinearPhysicalFormat::DenseF16,
            ElementType::F16,
            fixture.params,
        );
        assert_eq!(kind, expected);
        let _ = fixture.run_dispatch(pipeline, &queue, 1, |encoder| {
            dispatch_linear_grid(encoder, fixture.params, kind);
        });
        fixture.assert_cpu(None);
        assert_eq!(
            pipelines
                .plain_linear_dispatch(
                    LinearPhysicalFormat::DenseF16,
                    ElementType::F32,
                    fixture.params,
                )
                .1,
            LinearDispatchKind::CooperativeGemv
        );
        for format in [
            LinearPhysicalFormat::Q8_0,
            LinearPhysicalFormat::Native(GgufBlockFormat::Pq2_0),
        ] {
            assert_ne!(
                pipelines
                    .plain_linear_dispatch(format, ElementType::F16, fixture.params)
                    .1,
                LinearDispatchKind::NarrowDenseGemv
            );
        }
    }
    // Exercise the actual fallback, not only a capability predicate.
    pipelines.dense_narrow = None;
    let fixture = Fixture::generated(&device, 1, 5120, 48, false);
    let (pipeline, kind) = pipelines.plain_linear_dispatch(
        LinearPhysicalFormat::DenseF16,
        ElementType::F16,
        fixture.params,
    );
    assert_eq!(kind, LinearDispatchKind::CooperativeGemv);
    let _ = fixture.run_dispatch(pipeline, &queue, 1, |encoder| {
        dispatch_linear_grid(encoder, fixture.params, kind);
    });
    fixture.assert_cpu(None);
}

#[test]
fn narrow_dense_gemv_preserves_arbitrary_k_output_tails_and_offsets() {
    let device = Device::system_default().expect("narrow dense conformance requires Metal");
    let pipelines = Pipelines::new(&device);
    let queue = device.new_command_queue();
    for (rows, width, outputs) in [
        (1, 1, 1),
        (1, 127, 3),
        (1, 128, 48),
        (1, 129, 17),
        (1, 255, 127),
        (1, 257, 128),
        (1, 5120, 48),
        (1, 5121, 49),
        (3, 513, 129),
    ] {
        let fixture = Fixture::generated(&device, rows, width, outputs, false);
        for variant in VARIANTS {
            let _ = fixture.run(&pipelines, &queue, variant, 1);
            fixture.assert_cpu(None);
        }
    }
}

#[test]
fn narrow_dense_gemv_preserves_cancellation_subnormals_and_half_rounding() {
    let device = Device::system_default().expect("narrow dense precision requires Metal");
    let pipelines = Pipelines::new(&device);
    let queue = device.new_command_queue();
    const WIDTH: usize = 1025;
    const OUTPUTS: usize = 8;
    let mut input = vec![f16::ONE; WIDTH];
    input[0] = f16::from_f32(256.0);
    input[128] = f16::from_f32(256.0);
    let mut weights = vec![f16::ZERO; WIDTH * OUTPUTS];
    weights[0] = f16::from_f32(256.0);
    weights[128] = f16::from_f32(-256.0);
    weights[256] = f16::ONE; // +65536 -65536 +1 must not overflow or cancel to zero.
    for (column, entries) in [
        (1, vec![(1, 1.0), (33, 1.0 / 2048.0)]),
        (2, vec![(1, 1.0), (33, 1.0 / 2048.0), (65, 1.0 / 8388608.0)]),
        (3, vec![(1, 1.0), (33, 3.0 / 2048.0)]),
        (4, vec![(1, 1.0 / 16777216.0)]),
        (5, vec![(1, 1.0 / 16384.0), (33, -1.0 / 16777216.0)]),
        (6, vec![(1, 65504.0)]),
        (7, vec![(1, -1.0), (33, -1.0 / 2048.0)]),
    ] {
        for (index, value) in entries {
            weights[column * WIDTH + index] = f16::from_f32(value);
        }
    }
    let fixture = Fixture::new(&device, 1, WIDTH as u32, OUTPUTS as u32, input, weights);
    let expected = [
        0x3c00, 0x3c00, 0x3c01, 0x3c02, 0x0001, 0x03ff, 0x7bff, 0xbc00,
    ];
    for variant in VARIANTS {
        let _ = fixture.run(&pipelines, &queue, variant, 1);
        fixture.assert_cpu(Some(&expected));
    }
    let production = MetalLinearPipelines::new(&device).unwrap();
    let (pipeline, kind) = production.plain_linear_dispatch(
        LinearPhysicalFormat::DenseF16,
        ElementType::F16,
        fixture.params,
    );
    assert_eq!(kind, LinearDispatchKind::NarrowDenseGemv);
    let _ = fixture.run_dispatch(pipeline, &queue, 1, |encoder| {
        dispatch_linear_grid(encoder, fixture.params, kind);
    });
    fixture.assert_cpu(Some(&expected));
}

#[test]
#[ignore = "isolated Metal GPU timestamp comparison; coordinate exclusive GPU access"]
fn narrow_dense_gemv_isolated_gpu_timing() {
    let device = Device::system_default().expect("narrow dense timing requires Metal");
    let pipelines = Pipelines::new(&device);
    let queue = device.new_command_queue();
    for (width, outputs) in [
        (128, 1),
        (129, 17),
        (1024, 8),
        (5120, 48),
        (5120, 128),
        (5121, 49),
    ] {
        let fixture = Fixture::generated(&device, 1, width, outputs, true);
        let mut baseline = None;
        for variant in VARIANTS {
            let _ = fixture.run(&pipelines, &queue, variant, 1);
            fixture.assert_cpu(None);
            let bits = fixture
                .output()
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>();
            if let Some(expected) = &baseline {
                assert_eq!(&bits, expected);
            } else {
                baseline = Some(bits);
            }
        }
        let mut samples = Vec::new();
        for round in 0..8 {
            let order = if round % 2 == 0 {
                VARIANTS
            } else {
                [Variant::Threads256, Variant::Threads128, Variant::Original]
            };
            for variant in order {
                let elapsed = fixture
                    .run(&pipelines, &queue, variant, 8)
                    .expect("GPU timestamps unavailable");
                assert!(elapsed.is_finite() && elapsed > 0.0);
                assert_eq!(
                    fixture
                        .output()
                        .iter()
                        .map(|v| v.to_bits())
                        .collect::<Vec<_>>(),
                    *baseline.as_ref().unwrap()
                );
                samples.push(serde_json::json!({"round":round,"variant":format!("{variant:?}"),"device_ns_per_dispatch":elapsed / 8.0}));
            }
        }
        eprintln!(
            "{}",
            serde_json::json!({"kind":"metal_narrow_dense_gemv_timing","device":device.name(),
            "rows":1,"input_features":width,"output_features":outputs,"dtype":"f16","accumulation":"f32",
            "weight_bytes":u64::from(width)*u64::from(outputs)*2,"dispatches_per_command":8,
            "correctness_warmup_dispatches_per_variant":1,"measured_rounds":8,"samples":samples,
            "scope":"single_hot_dense_weight_buffer; initialization, readback and CPU oracle excluded; not model throughput"})
        );
    }
}
