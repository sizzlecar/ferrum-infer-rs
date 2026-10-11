//! Q4K regression and the shared guarded fixture for format specializations.

use super::*;
use cudarc::driver::{sys::CUevent_flags, CudaSlice, DevicePtr, DevicePtrMut};

const INPUT_PREFIX: usize = 3;
const WEIGHT_PREFIX: usize = 5;
const OUTPUT_PREFIX: usize = 8;
const OUTPUT_COLUMN: usize = 2;

pub(super) struct Fixture<T: Scalar> {
    format: GgufBlockFormat,
    rows: usize,
    inputs: usize,
    outputs: usize,
    stride: usize,
    input: Vec<T>,
    weights: Vec<u8>,
    output_initial: Vec<T>,
    expected: Vec<(f64, f64)>,
    input_gpu: CudaSlice<T>,
    weights_gpu: CudaSlice<u8>,
    output_gpu: CudaSlice<T>,
}

impl<T: Scalar> Fixture<T> {
    fn new(stream: &Arc<CudaStream>, rows: usize, inputs: usize, outputs: usize) -> Self {
        Self::for_format(stream, GgufBlockFormat::Q4K, rows, inputs, outputs, false)
    }

    pub(super) fn for_format(
        stream: &Arc<CudaStream>,
        format: GgufBlockFormat,
        rows: usize,
        inputs: usize,
        outputs: usize,
        dense_input: bool,
    ) -> Self {
        assert_eq!(inputs % format.block_values(), 0);
        let (mut encoded, _) = matrix(format, outputs, inputs / format.block_values());
        // Exercise zero, subnormal, positive and negative scales in the exact
        // source bytes. Keep this construction outside every timed submission.
        let scale_offset = if format == GgufBlockFormat::Q6K {
            208
        } else {
            0
        };
        for (index, block) in encoded.chunks_exact_mut(format.block_bytes()).enumerate() {
            if index % 7 == 0 {
                block[scale_offset..scale_offset + 2].copy_from_slice(&1_u16.to_le_bytes());
                if matches!(format, GgufBlockFormat::Q4K | GgufBlockFormat::Q5K) {
                    block[2..4].copy_from_slice(&0_u16.to_le_bytes());
                }
            } else if index % 7 == 1 {
                block[scale_offset..scale_offset + 2].copy_from_slice(&0_u16.to_le_bytes());
            }
        }
        Self::for_encoded_weights(stream, format, rows, inputs, outputs, dense_input, encoded)
    }

    pub(super) fn for_encoded_weights(
        stream: &Arc<CudaStream>,
        format: GgufBlockFormat,
        rows: usize,
        inputs: usize,
        outputs: usize,
        dense_input: bool,
        encoded: Vec<u8>,
    ) -> Self {
        assert_eq!(inputs % format.block_values(), 0);
        assert_eq!(
            encoded.len(),
            outputs * (inputs / format.block_values()) * format.block_bytes()
        );
        let mut decoded = vec![0.0; inputs * outputs];
        format.decode(&encoded, &mut decoded).unwrap();
        let guard = T::from_f32(-12345.0);
        let mut input = vec![guard; INPUT_PREFIX + rows * inputs + 8];
        input[INPUT_PREFIX..INPUT_PREFIX + rows * inputs].fill(T::from_f32(0.0));
        let mut sparse = Vec::new();
        for row in 0..rows {
            let mut entries = Vec::new();
            for index in 0..if dense_input { inputs } else { 16 } {
                let column = if dense_input {
                    index
                } else {
                    (index * (inputs / 16) + row * 13 + 7) % inputs
                };
                let value = T::from_f32(if dense_input {
                    ((index * 17 + row * 7) % 67) as f32 / 4096.0 - 33.0 / 4096.0
                } else {
                    ((index + row * 3) as f32 * 0.7).sin() * 0.125
                });
                input[INPUT_PREFIX + row * inputs + column] = value;
                entries.push((column, f64::from(value.as_f32())));
            }
            sparse.push(entries);
        }
        let mut expected = Vec::new();
        for entries in &sparse {
            for output in 0..outputs {
                let products = entries
                    .iter()
                    .map(|&(column, value)| value * f64::from(decoded[output * inputs + column]));
                expected.push((products.clone().sum(), products.map(f64::abs).sum()));
            }
        }
        let mut weights = vec![0xcc; WEIGHT_PREFIX];
        weights.extend_from_slice(&encoded);
        weights.extend([0xcc; 8]);
        let stride = outputs + 5;
        let mut output_initial = vec![guard; OUTPUT_PREFIX + rows * stride + 8];
        for row in 0..rows {
            let start = OUTPUT_PREFIX + row * stride + OUTPUT_COLUMN;
            output_initial[start..start + outputs].fill(T::from_f32(f32::NAN));
        }
        Self {
            format,
            rows,
            inputs,
            outputs,
            stride,
            input_gpu: stream.clone_htod(&input).unwrap(),
            weights_gpu: stream.clone_htod(&weights).unwrap(),
            output_gpu: stream.clone_htod(&output_initial).unwrap(),
            input,
            weights,
            output_initial,
            expected,
        }
    }

    pub(super) fn run(
        &mut self,
        stream: &Arc<CudaStream>,
        kernel: &CudaFunction,
        row_tile: u32,
        iterations: u32,
    ) -> (f64, f64) {
        assert!(iterations > 0);
        stream
            .memcpy_htod(&self.output_initial, &mut self.output_gpu)
            .unwrap();
        stream.synchronize().unwrap();
        let wall = std::time::Instant::now();
        let start = stream
            .record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
            .unwrap();
        {
            let input = self
                .input_gpu
                .slice(INPUT_PREFIX..INPUT_PREFIX + self.rows * self.inputs);
            let weights = self
                .weights_gpu
                .slice(WEIGHT_PREFIX..self.weights.len() - 8);
            let mut output = self
                .output_gpu
                .slice_mut(OUTPUT_PREFIX..OUTPUT_PREFIX + self.rows * self.stride);
            let parameters = [
                self.rows as u32,
                self.inputs as u32,
                self.outputs as u32,
                self.stride as u32,
                OUTPUT_COLUMN as u32,
                self.format.ggml_type_id(),
                self.format.block_values() as u32,
                self.format.block_bytes() as u32,
            ];
            for _ in 0..iterations {
                let mut launch = stream.launch_builder(kernel);
                launch.arg(&input).arg(&weights).arg(&mut output);
                for parameter in &parameters {
                    launch.arg(parameter);
                }
                // SAFETY: the typed views retain complete matrices. Both
                // kernels guard partial row tiles and output column groups;
                // the odd source-byte offset requires byte-safe GGUF loads.
                unsafe {
                    launch.launch(LaunchConfig {
                        grid_dim: (
                            (self.outputs as u32).div_ceil(4),
                            (self.rows as u32).div_ceil(row_tile),
                            1,
                        ),
                        block_dim: (128, 1, 1),
                        shared_mem_bytes: 0,
                    })
                }
                .unwrap();
            }
        }
        let end = stream
            .record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
            .unwrap();
        end.synchronize().unwrap();
        let wall_ns = wall.elapsed().as_secs_f64() * 1e9;
        let gpu_ns = f64::from(start.elapsed_ms(&end).unwrap()) * 1e6;
        (wall_ns, gpu_ns)
    }

    pub(super) fn run_dispatch(
        &mut self,
        stream: &Arc<CudaStream>,
        kernels: &CudaNativeBlockKernels,
    ) {
        use ferrum_interfaces::vnext::WeightId;
        stream
            .memcpy_htod(&self.output_initial, &mut self.output_gpu)
            .unwrap();
        let input = self
            .input_gpu
            .slice(INPUT_PREFIX..INPUT_PREFIX + self.rows * self.inputs);
        let weights = self
            .weights_gpu
            .slice(WEIGHT_PREFIX..self.weights.len() - 8);
        let mut output = self
            .output_gpu
            .slice_mut(OUTPUT_PREFIX..OUTPUT_PREFIX + self.rows * self.stride);
        let (xp, _x_guard) = input.device_ptr(stream);
        let (wp, _w_guard) = weights.device_ptr(stream);
        let (yp, _y_guard) = output.device_ptr_mut(stream);
        let part = weights::MatrixPart {
            transform: None,
            signs_region: None,
            component_id: WeightId::new("component.specialization-fixture").unwrap(),
            format: weights::MatrixFormat::Block(self.format),
            rows: self.outputs as u32,
            columns: self.inputs as u32,
            output_offset: OUTPUT_COLUMN as u32,
        };
        kernels
            .linear_with_precision(
                stream,
                xp,
                wp,
                yp,
                &part,
                self.rows as u32,
                self.stride as u32,
                T::ELEMENT_TYPE,
                T::ELEMENT_TYPE,
            )
            .unwrap();
    }

    pub(super) fn validate(&self, stream: &Arc<CudaStream>) -> Vec<u32> {
        assert_eq!(
            stream
                .clone_dtoh(&self.input_gpu)
                .unwrap()
                .iter()
                .map(|x| x.as_f32().to_bits())
                .collect::<Vec<_>>(),
            self.input
                .iter()
                .map(|x| x.as_f32().to_bits())
                .collect::<Vec<_>>(),
            "input mutated"
        );
        assert_eq!(
            stream.clone_dtoh(&self.weights_gpu).unwrap(),
            self.weights,
            "weights mutated"
        );
        self.validate_output(stream)
    }

    pub(super) fn validate_output(&self, stream: &Arc<CudaStream>) -> Vec<u32> {
        let actual = stream.clone_dtoh(&self.output_gpu).unwrap();
        for (index, value) in actual.iter().enumerate() {
            let location = index
                .checked_sub(OUTPUT_PREFIX)
                .filter(|index| *index < self.rows * self.stride)
                .filter(|index| {
                    (OUTPUT_COLUMN..OUTPUT_COLUMN + self.outputs).contains(&(index % self.stride))
                });
            if let Some(index) = location {
                let (expected, sum_abs) = self.expected
                    [index / self.stride * self.outputs + index % self.stride - OUTPUT_COLUMN];
                let bound =
                    (self.inputs as f64 * f64::from(f32::EPSILON) + T::ROUNDING) * sum_abs + 1e-6;
                assert!(
                    value.as_f32().is_finite()
                        && (f64::from(value.as_f32()) - expected).abs() <= bound,
                    "{}x{}x{} index {index}: {} != {expected}, bound={bound}",
                    self.rows,
                    self.inputs,
                    self.outputs,
                    value.as_f32()
                );
            } else {
                assert_eq!(
                    value.as_f32().to_bits(),
                    self.output_initial[index].as_f32().to_bits(),
                    "guard at {index}"
                );
            }
        }
        actual
            .iter()
            .map(|value| value.as_f32().to_bits())
            .collect()
    }
}

#[test]
#[ignore = "requires an actual CUDA device"]
fn q4k_specialization_preserves_generic_bits_and_f64_oracle_on_cuda() {
    let context = CudaContext::new(0).expect("Q4K specialization requires CUDA");
    let kernels = CudaNativeBlockKernels::load(&context).unwrap();
    let stream = context.default_stream();
    for (rows, inputs, outputs) in [
        (1, 256, 7),
        (2, 768, 7),
        (7, 512, 17),
        (8, 768, 17),
        (9, 2560, 7),
        (64, 768, 17),
        (65, 2560, 7),
    ] {
        let mut fixture = Fixture::<f16>::new(&stream, rows, inputs, outputs);
        fixture.run(&stream, &kernels.linear_f16, 1, 1);
        let expected = fixture.validate(&stream);
        for (kernel, row_tile) in [
            (&kernels.linear_q4k_f16, 1),
            (&kernels.linear_q4k_tiled_f16, LINEAR_ROW_TILE),
        ] {
            fixture.run(&stream, kernel, row_tile, 1);
            assert_eq!(
                fixture.validate(&stream),
                expected,
                "specialization changed accumulation bits"
            );
            fixture.run(&stream, kernel, row_tile, 2);
            assert_eq!(
                fixture.validate(&stream),
                expected,
                "repeated specialization changed bits"
            );
        }
    }
}

#[test]
#[ignore = "paired GPU timing; coordinate exclusive CUDA access"]
fn q4k_specialization_dispatch_microbench() {
    const ITERATIONS: u32 = 32;
    let context = CudaContext::new(0).expect("Q4K microbench requires CUDA");
    let kernels = CudaNativeBlockKernels::load(&context).unwrap();
    let stream = context.default_stream();
    // Synthetic source blocks with actual Qwen3.5-4B projection dimensions.
    // Timing is per complete projection, not per token or generated response.
    for (inputs, outputs) in [(2560, 2560), (2560, 9216)] {
        for rows in [1, 4, 64, 65] {
            let mut fixture = Fixture::<f16>::new(&stream, rows, inputs, outputs);
            let (generic, specialized, row_tile) = if rows == 1 {
                (&kernels.linear_f16, &kernels.linear_q4k_f16, 1)
            } else {
                (
                    &kernels.linear_tiled_f16,
                    &kernels.linear_q4k_tiled_f16,
                    LINEAR_ROW_TILE,
                )
            };
            fixture.run(&stream, generic, row_tile, 1);
            let reference = fixture.validate(&stream);
            for round in 0..10 {
                for specialized_route in if round % 2 == 0 {
                    [false, true]
                } else {
                    [true, false]
                } {
                    let kernel = if specialized_route {
                        specialized
                    } else {
                        generic
                    };
                    let (wall_ns, gpu_ns) = fixture.run(&stream, kernel, row_tile, ITERATIONS);
                    assert_eq!(
                        fixture.validate(&stream),
                        reference,
                        "timed variant changed bits"
                    );
                    if round >= 2 {
                        println!(
                            "{}",
                            serde_json::json!({
                                "benchmark": "q4k_format_specialization", "rows": rows, "input": inputs, "output": outputs,
                                "row_tile": row_tile, "specialized": specialized_route, "round": round - 2,
                                "projection_iterations": ITERATIONS,
                                "command_wall_ns": wall_ns, "command_gpu_ns": gpu_ns,
                                "wall_ns": wall_ns / f64::from(ITERATIONS), "gpu_ns": gpu_ns / f64::from(ITERATIONS),
                            })
                        );
                    }
                }
            }
        }
    }
}
