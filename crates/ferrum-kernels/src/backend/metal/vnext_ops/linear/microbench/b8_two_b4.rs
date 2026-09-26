//! Isolated B8 F16 experiment, using two B4 tiles in ONE physical dispatch.
//! Production selection and all F32 vocabulary routes remain unchanged.
use super::*;
use q4_b4_ffn_cooperative::oracle::metrics;

pub(super) mod ffn;
pub(super) mod gdn;
pub(super) mod oracle;

pub(super) const ROWS: usize = 8;
pub(super) const PREFIX: usize = 3;
pub(super) const SUFFIX: usize = 17;
pub(super) const WEIGHT_PREFIX: usize = 2;
pub(super) const HALF_GUARD: f16 = f16::from_bits(0x57b0);
const BYTE_GUARD: u8 = 0xa7;
const TWO_B4_SOURCE: &str = include_str!("b8_two_b4/two_b4.metal");

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Arm {
    ProductionMma,
    TwoB4,
    M8Mma,
}

pub(super) struct Pipelines {
    pub(super) production: MetalLinearPipelines,
    pub(super) candidate: [ComputePipelineState; 3],
}

impl Pipelines {
    fn new(device: &Device) -> Self {
        let source = format!("{}\n{TWO_B4_SOURCE}", small_batch::SHADER_SOURCE);
        let library = device
            .new_library_with_source(&source, &CompileOptions::new())
            .unwrap();
        Self {
            production: MetalLinearPipelines::new(device).unwrap(),
            candidate: ["q4", "q5", "q6"].map(|format| {
                let function = library
                    .get_function(&format!("{format}_shared_b8_two_b4_experiment"), None)
                    .unwrap();
                let pipeline = device
                    .new_compute_pipeline_state_with_function(&function)
                    .unwrap();
                assert_eq!(pipeline.thread_execution_width(), 32);
                assert!(pipeline.max_total_threads_per_threadgroup() >= 64);
                pipeline
            }),
        }
    }

    pub(super) fn candidate(&self, format: GgufBlockFormat) -> Option<&ComputePipelineState> {
        match format {
            GgufBlockFormat::Q4K => Some(&self.candidate[0]),
            GgufBlockFormat::Q5K => Some(&self.candidate[1]),
            GgufBlockFormat::Q6K => Some(&self.candidate[2]),
            _ => None,
        }
    }
}

pub(super) struct Halves {
    pub(super) buffer: Buffer,
    pub(super) len: usize,
}

impl Halves {
    pub(super) fn new(device: &Device, values: &[f16]) -> Self {
        let mut guarded = vec![HALF_GUARD; PREFIX];
        guarded.extend_from_slice(values);
        guarded.extend([HALF_GUARD; SUFFIX]);
        Self {
            buffer: buffer(device, &guarded),
            len: values.len(),
        }
    }

    pub(super) fn empty(device: &Device, len: usize) -> Self {
        Self::new(device, &vec![f16::NAN; len])
    }

    pub(super) fn values(&self) -> &[f16] {
        // SAFETY: fixture-owned shared buffers, read only after command wait.
        let all = unsafe {
            std::slice::from_raw_parts(
                self.buffer.contents().cast::<f16>(),
                self.len + PREFIX + SUFFIX,
            )
        };
        assert!(all[..PREFIX]
            .iter()
            .chain(&all[PREFIX + self.len..])
            .all(|v| v.to_bits() == HALF_GUARD.to_bits()));
        &all[PREFIX..PREFIX + self.len]
    }

    pub(super) fn reset(&self, value: f16) {
        // SAFETY: no command is in flight at fixture reset points.
        unsafe {
            std::slice::from_raw_parts_mut(
                self.buffer.contents().cast::<f16>().add(PREFIX),
                self.len,
            )
            .fill(value);
        }
    }
}

pub(super) struct Projection {
    pub(super) shape: Shape,
    pub(super) bytes: Vec<u8>,
    pub(super) weight: Buffer,
}

impl Projection {
    pub(super) fn new(device: &Device, shape: Shape, seed: usize) -> Self {
        let bytes = if shape.format == GgufBlockFormat::Q8_0 {
            weights(shape)
        } else {
            q4_b4_ffn_cooperative::oracle::matrix(shape, seed)
        };
        let mut guarded = vec![BYTE_GUARD; WEIGHT_PREFIX];
        guarded.extend_from_slice(&bytes);
        guarded.extend([BYTE_GUARD; SUFFIX]);
        Self {
            shape,
            bytes,
            weight: buffer(device, &guarded),
        }
    }

    pub(super) fn immutable(&self) {
        // SAFETY: fixture owns the complete byte allocation and has waited.
        let bytes = unsafe {
            std::slice::from_raw_parts(
                self.weight.contents().cast::<u8>(),
                self.weight.length() as usize,
            )
        };
        assert!(bytes[..WEIGHT_PREFIX]
            .iter()
            .chain(&bytes[WEIGHT_PREFIX + self.bytes.len()..])
            .all(|b| *b == BYTE_GUARD));
        assert_eq!(
            &bytes[WEIGHT_PREFIX..WEIGHT_PREFIX + self.bytes.len()],
            self.bytes
        );
    }

    pub(super) fn encode(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        pipelines: &Pipelines,
        arm: Arm,
        input: &Halves,
        output: &Halves,
        stride: u32,
        column: u32,
    ) {
        let params = LinearParams {
            rows: ROWS as u32,
            in_features: self.shape.input,
            out_features: self.shape.output,
            output_stride: stride,
            output_column_offset: column,
        };
        let format = match self.shape.format {
            GgufBlockFormat::Q8_0 => LinearPhysicalFormat::Q8_0,
            other => physical(other),
        };
        let candidate = (arm != Arm::ProductionMma)
            .then(|| pipelines.candidate(self.shape.format))
            .flatten();
        // Keep this historical experiment's original M32 reference fixed
        // after production adopts the separately qualified M8 route.
        let baseline = match format {
            LinearPhysicalFormat::Q4K => &pipelines.production.k_quant_gemm.q4_k,
            LinearPhysicalFormat::Q5K => &pipelines.production.k_quant_gemm.q5_k,
            LinearPhysicalFormat::Q6K => &pipelines.production.k_quant_gemm.q6_k,
            LinearPhysicalFormat::Q8_0 => &pipelines.production.k_quant_gemm.q8_0,
            _ => unreachable!("quantized projection fixture"),
        };
        let kind = LinearDispatchKind::TiledGemm;
        encoder.set_compute_pipeline_state(candidate.unwrap_or(baseline));
        encoder.set_buffer(0, Some(&input.buffer), (PREFIX * 2) as u64);
        encoder.set_buffer(1, Some(&self.weight), WEIGHT_PREFIX as u64);
        encoder.set_buffer(2, Some(&output.buffer), (PREFIX * 2) as u64);
        bind_linear_params(encoder, params, format, ElementType::F16);
        if candidate.is_some() && arm == Arm::M8Mma {
            // The independent M8 experiment retains production's 8 KiB grant.
            encoder.set_threadgroup_memory_length(0, 8192);
            encoder.dispatch_thread_groups(
                MTLSize::new(
                    u64::from(params.rows).div_ceil(8),
                    u64::from(params.out_features).div_ceil(64),
                    1,
                ),
                MTLSize::new(128, 1, 1),
            );
        } else if candidate.is_some() {
            encoder.set_threadgroup_memory_length(0, 0);
            encoder.dispatch_thread_groups(
                MTLSize::new(u64::from(params.out_features).div_ceil(4), 2, 1),
                MTLSize::new(32, 2, 1),
            );
        } else {
            dispatch_linear_grid(encoder, params, kind);
        }
    }
}

pub(super) fn run(
    queue: &CommandQueueRef,
    arm: Arm,
    worksets: usize,
    dispatches_per_workset: usize,
    encode: impl FnOnce(&metal::ComputeCommandEncoderRef),
) -> serde_json::Value {
    let started = Instant::now();
    let command = queue.new_command_buffer();
    let encoder = command.new_compute_command_encoder();
    encode(encoder);
    encoder.end_encoding();
    let host_encode_ns = started.elapsed().as_nanos() as u64;
    let submitted = Instant::now();
    command.commit();
    command.wait_until_completed();
    assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
    serde_json::json!({"arm":format!("{arm:?}"),"worksets":worksets,"physical_dispatches":worksets*dispatches_per_workset,
        "host_encode_ns":host_encode_ns,"host_submit_wait_ns":submitted.elapsed().as_nanos() as u64,
        "device_command_ns":gpu_elapsed_ns(command).expect("complete command GPU interval")})
}

pub(super) fn dense_input(width: usize) -> Vec<f16> {
    // The existing dense FFN formula, extended from four to eight owners.
    (0..ROWS * width)
        .map(|index| {
            let phase = (index + 1) as f32 * 0.017 + (index / width) as f32 * 0.113;
            let value = f16::from_f32(phase.sin() * 0.03125 + (phase * 0.37).cos() * 0.0078125);
            if value == f16::ZERO {
                f16::from_f32(0.000_061_035_156)
            } else {
                value
            }
        })
        .collect()
}

pub(super) fn exact_bits(actual: &[f16], expected: &[f16]) {
    assert_eq!(actual.len(), expected.len());
    assert!(actual
        .iter()
        .zip(expected)
        .all(|(a, b)| a.to_bits() == b.to_bits()));
}

#[test]
fn b8_two_b4_grid_covers_every_owner_and_packed_column_once() {
    for width in [1_usize, 3, 4, 5, 65, 12288] {
        let stride = width * 2 + 9;
        for column in [1, width + 4] {
            let mut counts = vec![0; ROWS * stride];
            for y in 0..2 {
                for x in 0..width.div_ceil(4) {
                    for sg in 0..2 {
                        for out in 0..2 {
                            let c = (x * 2 + sg) * 2 + out;
                            if c < width {
                                for row in 0..4 {
                                    counts[(y * 4 + row) * stride + column + c] += 1;
                                }
                            }
                        }
                    }
                }
            }
            for (i, count) in counts.into_iter().enumerate() {
                assert_eq!(
                    count,
                    usize::from((column..column + width).contains(&(i % stride)))
                );
            }
        }
    }
}

#[test]
fn b8_two_b4_preserves_each_original_b4_dot_at_packed_offsets_and_tails() {
    let device = Device::system_default().expect("B8 mapping conformance requires Metal");
    let queue = device.new_command_queue();
    let pipelines = Pipelines::new(&device);
    for format in [
        GgufBlockFormat::Q4K,
        GgufBlockFormat::Q5K,
        GgufBlockFormat::Q6K,
    ] {
        for width in [1_u32, 3, 65, 1025] {
            let shape = Shape {
                name: "two_b4_tail",
                input: 512,
                output: width,
                format,
            };
            let projection = Projection::new(&device, shape, 1);
            let values = dense_input(shape.input as usize);
            let input = Halves::new(&device, &values);
            let stride = width + 9;
            let output = std::array::from_fn::<_, 2, _>(|_| {
                Halves::new(&device, &vec![HALF_GUARD; ROWS * stride as usize])
            });
            run(&queue, Arm::TwoB4, 1, 1, |encoder| {
                projection.encode(
                    encoder,
                    &pipelines,
                    Arm::TwoB4,
                    &input,
                    &output[0],
                    stride,
                    4,
                )
            });
            run(&queue, Arm::ProductionMma, 1, 2, |encoder| {
                let format = physical(shape.format);
                let baseline = pipelines
                    .production
                    .small_batch
                    .pipeline(format, 4)
                    .unwrap();
                let params = LinearParams {
                    rows: 4,
                    in_features: shape.input,
                    out_features: width,
                    output_stride: stride,
                    output_column_offset: 4,
                };
                for group in 0..2 {
                    encoder.set_compute_pipeline_state(baseline);
                    encoder.set_threadgroup_memory_length(0, 0);
                    encoder.set_buffer(
                        0,
                        Some(&input.buffer),
                        ((PREFIX + group * 4 * shape.input as usize) * 2) as u64,
                    );
                    encoder.set_buffer(1, Some(&projection.weight), WEIGHT_PREFIX as u64);
                    encoder.set_buffer(
                        2,
                        Some(&output[1].buffer),
                        ((PREFIX + group * 4 * stride as usize) * 2) as u64,
                    );
                    bind_linear_params(encoder, params, format, ElementType::F16);
                    dispatch_linear_grid(encoder, params, LinearDispatchKind::SharedWeightGemv);
                }
            });
            exact_bits(output[0].values(), output[1].values());
            let expected = oracle::project(&projection, &values, false);
            let actual: Vec<_> = output[0]
                .values()
                .chunks_exact(stride as usize)
                .flat_map(|row| {
                    assert!(row[..4]
                        .iter()
                        .chain(&row[4 + width as usize..])
                        .all(|v| v.to_bits() == HALF_GUARD.to_bits()));
                    row[4..4 + width as usize].iter().copied()
                })
                .collect();
            let measured = metrics(&actual, &expected, width as usize);
            println!(
                "{}",
                serde_json::json!({"kind":"b8_two_b4_mapping","format":shape.format.format_id(),"output_width":width,"b4_bitwise_equal":true,"independent_f64":measured})
            );
            assert_eq!(measured["linear_bound_violations"], 0);
            exact_bits(input.values(), &values);
            projection.immutable();
        }
    }
}
