use super::*;
use half::f16;
use metal::{Buffer, CommandQueueRef, MTLCommandBufferStatus, MTLResourceOptions};

fn buffer<T>(device: &Device, data: &[T]) -> Buffer {
    device.new_buffer_with_data(
        data.as_ptr().cast(),
        std::mem::size_of_val(data) as u64,
        MTLResourceOptions::StorageModeShared,
    )
}

pub(super) struct Fixture {
    params: LinearParams,
    input: Buffer,
    weight: Buffer,
    output: Buffer,
    output_type: ElementType,
    input_values: Vec<f32>,
    weight_bytes: Vec<u8>,
    output_elements: usize,
}

impl Fixture {
    pub(super) fn params(&self) -> LinearParams {
        self.params
    }

    pub(super) fn poison_output(&self) {
        let prefix = 16 / self.output_type.size_bytes() as usize;
        // SAFETY: tests call this only after the previous command completes.
        unsafe {
            match self.output_type {
                ElementType::F32 => {
                    let values = std::slice::from_raw_parts_mut(
                        self.output.contents().cast::<f32>(),
                        self.output_elements,
                    );
                    values.fill(-123.0);
                    for row in 0..self.params.rows as usize {
                        let start = prefix + row * self.params.output_stride as usize + 2;
                        values[start..start + self.params.out_features as usize].fill(f32::NAN);
                    }
                }
                ElementType::F16 => {
                    let values = std::slice::from_raw_parts_mut(
                        self.output.contents().cast::<f16>(),
                        self.output_elements,
                    );
                    values.fill(f16::from_f32(-123.0));
                    for row in 0..self.params.rows as usize {
                        let start = prefix + row * self.params.output_stride as usize + 2;
                        values[start..start + self.params.out_features as usize]
                            .fill(f16::from_f32(f32::NAN));
                    }
                }
                _ => unreachable!(),
            }
        }
    }

    pub(super) fn assert_inputs_unchanged(&self) {
        // SAFETY: StorageModeShared buffers are read after command completion.
        unsafe {
            let input = std::slice::from_raw_parts(
                self.input.contents().cast::<f32>(),
                self.input.length() as usize / 4,
            );
            assert!(input[..4].iter().all(|&value| value == -123.0));
            assert_eq!(
                input[4..].iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                self.input_values
                    .iter()
                    .map(|x| x.to_bits())
                    .collect::<Vec<_>>()
            );
            let weight = std::slice::from_raw_parts(
                self.weight.contents().cast::<u8>(),
                self.weight.length() as usize,
            );
            assert!(weight[..18].iter().all(|&value| value == 0xcc));
            assert_eq!(&weight[18..], &self.weight_bytes);
        }
    }

    pub(super) fn new(
        device: &Device,
        rows: u32,
        width: u32,
        outputs: u32,
        output_type: ElementType,
        subnormal_scale: bool,
    ) -> Self {
        let input_values = (0..rows * width)
            .map(|i| ((i * 37 % 63) as i32 - 31) as f32 / 512.0)
            .collect::<Vec<_>>();
        let mut padded_input = vec![-123.0; 4];
        padded_input.extend_from_slice(&input_values);
        let mut weight_bytes = Vec::new();
        for output in 0..outputs {
            for block in 0..width / 128 {
                let scales = if subnormal_scale {
                    [0x3800_u16, 0xb400, 0, 0x0001]
                } else {
                    [0x3000_u16, 0xb400, 0, 0x2c00]
                };
                weight_bytes
                    .extend_from_slice(&scales[((output + block) % 4) as usize].to_le_bytes());
                for byte in 0..32 {
                    weight_bytes
                        .push([0xe4, 0x1b, 0xff, 0x55][((byte + block + output) % 4) as usize]);
                }
            }
        }
        let mut padded_weights = vec![0xcc_u8; 18];
        padded_weights.extend_from_slice(&weight_bytes);
        let params = LinearParams {
            rows,
            in_features: width,
            out_features: outputs,
            output_stride: outputs + 5,
            output_column_offset: 2,
        };
        let output_elements =
            16 / output_type.size_bytes() as usize + (rows * params.output_stride) as usize + 8;
        let output = match output_type {
            ElementType::F16 => buffer(device, &vec![f16::from_f32(-123.0); output_elements]),
            ElementType::F32 => buffer(device, &vec![-123.0_f32; output_elements]),
            _ => unreachable!(),
        };
        Self {
            params,
            input: buffer(device, &padded_input),
            weight: buffer(device, &padded_weights),
            output,
            output_type,
            input_values,
            weight_bytes,
            output_elements,
        }
    }

    pub(super) fn replace_payload(
        &mut self,
        device: &Device,
        input_values: Vec<f32>,
        weight_bytes: Vec<u8>,
    ) {
        assert_eq!(input_values.len(), self.input_values.len());
        assert_eq!(weight_bytes.len(), self.weight_bytes.len());
        let mut padded_input = vec![-123.0; 4];
        padded_input.extend_from_slice(&input_values);
        let mut padded_weights = vec![0xcc_u8; 18];
        padded_weights.extend_from_slice(&weight_bytes);
        self.input = buffer(device, &padded_input);
        self.weight = buffer(device, &padded_weights);
        self.input_values = input_values;
        self.weight_bytes = weight_bytes;
    }

    pub(super) fn run(
        &self,
        pipelines: &MetalLinearPipelines,
        queue: &CommandQueueRef,
        wide: bool,
        repetitions: usize,
    ) -> f64 {
        let (pipeline, dispatch) = match (wide, self.output_type) {
            (true, ElementType::F32) => (
                &pipelines.native.pq2_linear_f32,
                LinearDispatchKind::Pq2CooperativeGemv,
            ),
            (true, ElementType::F16) => (
                &pipelines.native.pq2_linear_f32_f16,
                LinearDispatchKind::Pq2CooperativeGemv,
            ),
            (false, ElementType::F32) => (
                pipelines.native.linear_f32(GgufBlockFormat::Pq2_0),
                LinearDispatchKind::CooperativeGemv,
            ),
            (false, ElementType::F16) => (
                pipelines.native.linear_f32_f16(GgufBlockFormat::Pq2_0),
                LinearDispatchKind::CooperativeGemv,
            ),
            _ => unreachable!(),
        };
        self.run_pipeline(pipeline, dispatch, queue, repetitions)
    }

    pub(super) fn run_pipeline(
        &self,
        pipeline: &ComputePipelineState,
        dispatch: LinearDispatchKind,
        queue: &CommandQueueRef,
        repetitions: usize,
    ) -> f64 {
        let command = queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(&self.input), 16);
        encoder.set_buffer(1, Some(&self.weight), 18);
        encoder.set_buffer(2, Some(&self.output), 16);
        encoder.set_bytes(
            3,
            std::mem::size_of::<LinearParams>() as u64,
            &self.params as *const _ as *const c_void,
        );
        bind_native_block(encoder, GgufBlockFormat::Pq2_0, 4);
        for _ in 0..repetitions {
            dispatch_linear_grid(encoder, self.params, dispatch);
        }
        encoder.end_encoding();
        command.commit();
        command.wait_until_completed();
        assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
        super::microbench::gpu_elapsed_ns(command).unwrap_or(f64::NAN) / 1e9
    }

    pub(super) fn read(&self) -> Vec<f32> {
        // SAFETY: StorageModeShared buffers are read only after command completion.
        unsafe {
            match self.output_type {
                ElementType::F32 => std::slice::from_raw_parts(
                    self.output.contents().cast::<f32>(),
                    self.output_elements,
                )
                .to_vec(),
                ElementType::F16 => std::slice::from_raw_parts(
                    self.output.contents().cast::<f16>(),
                    self.output_elements,
                )
                .iter()
                .map(|x| x.to_f32())
                .collect(),
                _ => unreachable!(),
            }
        }
    }

    pub(super) fn assert_cpu(&self) {
        let width = self.params.in_features as usize;
        let outputs = self.params.out_features as usize;
        let mut weights = vec![0.0; width * outputs];
        GgufBlockFormat::Pq2_0
            .decode(&self.weight_bytes, &mut weights)
            .unwrap();
        let actual = self.read();
        let prefix = 16 / self.output_type.size_bytes() as usize;
        for row in 0..self.params.rows as usize {
            let start = prefix + row * self.params.output_stride as usize + 2;
            for col in 0..outputs {
                let terms = self.input_values[row * width..(row + 1) * width]
                    .iter()
                    .zip(&weights[col * width..(col + 1) * width])
                    .map(|(&x, &w)| f64::from(x) * f64::from(w))
                    .collect::<Vec<_>>();
                let expected = terms.iter().sum::<f64>() as f32;
                let expected = if self.output_type == ElementType::F16 {
                    f16::from_f32(expected).to_f32()
                } else {
                    expected
                };
                // Same forward-error bound as the Hadamard projection fixture;
                // no half conversion is permitted at the activation input.
                let bound = 2.0
                    * width as f64
                    * f64::from(f32::EPSILON)
                    * terms.iter().map(|x| x.abs()).sum::<f64>()
                    + if self.output_type == ElementType::F16 {
                        f64::from(expected.abs()) / 1024.0 + 1.0 / 16777216.0
                    } else {
                        0.0
                    };
                assert!(
                    (f64::from(actual[start + col]) - f64::from(expected)).abs() <= bound,
                    "{:?} row {row} col {col}: {} != {expected}",
                    self.output_type,
                    actual[start + col]
                );
            }
            assert!(actual[start - 2..start].iter().all(|&x| x == -123.0));
            assert!(actual
                [start + outputs..prefix + (row + 1) * self.params.output_stride as usize]
                .iter()
                .all(|&x| x == -123.0));
        }
        assert!(actual[..prefix]
            .iter()
            .chain(&actual[prefix + (self.params.rows * self.params.output_stride) as usize..])
            .all(|&x| x == -123.0));
    }
}

#[test]
fn pq2_wide_gemv_preserves_f32_input_codes_blocks_offsets_and_output_tails() {
    let device = Device::system_default().expect("PQ2 GEMV conformance requires Metal");
    let pipelines = MetalLinearPipelines::new(&device).unwrap();
    let queue = device.new_command_queue();
    // Sub-SIMD, incomplete four-block group, output-tile tail, and small batches.
    for (rows, width, outputs) in [(1, 128, 1), (3, 384, 17), (31, 640, 31)] {
        for output_type in [ElementType::F16, ElementType::F32] {
            let fixture = Fixture::new(&device, rows, width, outputs, output_type, true);
            fixture.run(&pipelines, &queue, true, 1);
            fixture.assert_cpu();
        }
    }
}

#[test]
#[ignore = "isolated Metal GPU timestamp comparison; run without other GPU or compiler work"]
fn pq2_wide_gemv_isolated_gpu_timing() {
    let device = Device::system_default().expect("PQ2 GEMV timing requires Metal");
    let pipelines = MetalLinearPipelines::new(&device).unwrap();
    let queue = device.new_command_queue();
    for (width, outputs) in [(5120, 17408), (17408, 5120), (6144, 5120)] {
        for output_type in [ElementType::F16, ElementType::F32] {
            let fixture = Fixture::new(&device, 1, width, outputs, output_type, false);
            fixture.run(&pipelines, &queue, false, 1);
            let baseline = fixture.read();
            fixture.run(&pipelines, &queue, true, 1);
            // Bounded dyadic inputs/scales keep these sums exactly representable;
            // compare every output and canary without timing host readback.
            assert_eq!(fixture.read(), baseline);
            let mut old_us = Vec::new();
            let mut new_us = Vec::new();
            for sample in 0..7 {
                for wide in if sample % 2 == 0 {
                    [false, true]
                } else {
                    [true, false]
                } {
                    let micros = fixture.run(&pipelines, &queue, wide, 8) * 1e6 / 8.0;
                    assert!(
                        micros.is_finite() && micros > 0.0,
                        "GPU timestamps unavailable"
                    );
                    if wide {
                        new_us.push(micros);
                    } else {
                        old_us.push(micros);
                    }
                }
            }
            let average = |x: &[f64]| x.iter().sum::<f64>() / x.len() as f64;
            eprintln!("PQ2_GPU_TIMING K={width} N={outputs} output={output_type:?} old_us={old_us:?} new_us={new_us:?} ratio={:.6}", average(&new_us) / average(&old_us));
        }
    }
}
