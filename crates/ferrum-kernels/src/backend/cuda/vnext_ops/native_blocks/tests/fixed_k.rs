//! Shared guarded K-quant fixture; retained generic kernels are the bitwise control.

use super::*;
use cudarc::driver::{sys::CUevent_flags, CudaSlice};

const INPUT_PREFIX: usize = 3;
const WEIGHT_PREFIX: usize = 5;
const OUTPUT_PREFIX: usize = 8;
const OUTPUT_COLUMN: usize = 2;

pub(super) struct Fixture {
    format: GgufBlockFormat,
    rows: usize,
    inputs: usize,
    outputs: usize,
    stride: usize,
    input: Vec<f16>,
    weights: Vec<u8>,
    output_initial: Vec<f16>,
    expected: Vec<(f64, f64)>,
    input_gpu: CudaSlice<f16>,
    weights_gpu: CudaSlice<u8>,
    output_gpu: CudaSlice<f16>,
}

impl Fixture {
    pub(super) fn new(
        stream: &Arc<CudaStream>,
        rows: usize,
        inputs: usize,
        outputs: usize,
    ) -> Self {
        Self::with_format(stream, GgufBlockFormat::Q4K, rows, inputs, outputs, false)
    }

    pub(super) fn with_format(
        stream: &Arc<CudaStream>,
        format: GgufBlockFormat,
        rows: usize,
        inputs: usize,
        outputs: usize,
        dense: bool,
    ) -> Self {
        assert!(matches!(
            format,
            GgufBlockFormat::Q4K | GgufBlockFormat::Q5K
        ));
        assert_eq!(inputs % 256, 0);
        let (mut encoded, _) = matrix(format, outputs, inputs / 256);
        // Exercise zero, subnormal, positive and negative scales in the exact
        // source bytes. Keep this construction outside every timed submission.
        for (index, block) in encoded.chunks_exact_mut(format.block_bytes()).enumerate() {
            if index % 7 == 0 {
                block[..2].copy_from_slice(&1_u16.to_le_bytes());
                block[2..4].copy_from_slice(&0_u16.to_le_bytes());
            } else if index % 7 == 1 {
                block[..2].copy_from_slice(&0_u16.to_le_bytes());
            }
        }
        let mut decoded = vec![0.0; inputs * outputs];
        format.decode(&encoded, &mut decoded).unwrap();
        let guard = f16::from_f32(-12345.0);
        let mut input = vec![guard; INPUT_PREFIX + rows * inputs + 8];
        input[INPUT_PREFIX..INPUT_PREFIX + rows * inputs].fill(f16::ZERO);
        let mut sparse = Vec::new();
        for row in 0..rows {
            let mut entries = Vec::new();
            for index in 0..if dense { inputs } else { 16 } {
                let column = if dense {
                    index
                } else {
                    (index * (inputs / 16) + row * 13 + 7) % inputs
                };
                let value = f16::from_f32(((index + row * 3) as f32 * 0.7).sin() * 0.125);
                input[INPUT_PREFIX + row * inputs + column] = value;
                entries.push((column, f64::from(value.to_f32())));
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
            output_initial[start..start + outputs].fill(f16::NAN);
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

    pub(super) fn validate(&self, stream: &Arc<CudaStream>) -> Vec<u16> {
        assert_eq!(
            stream.clone_dtoh(&self.input_gpu).unwrap(),
            self.input,
            "input mutated"
        );
        assert_eq!(
            stream.clone_dtoh(&self.weights_gpu).unwrap(),
            self.weights,
            "weights mutated"
        );
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
                let bound = (self.inputs as f64 * f64::from(f32::EPSILON)
                    + <f16 as Scalar>::ROUNDING)
                    * sum_abs
                    + 1e-6;
                assert!(
                    value.is_finite() && (f64::from(value.to_f32()) - expected).abs() <= bound,
                    "{}x{}x{} index {index}: {} != {expected}, bound={bound}",
                    self.rows,
                    self.inputs,
                    self.outputs,
                    value.to_f32()
                );
            } else {
                assert_eq!(
                    value.to_bits(),
                    self.output_initial[index].to_bits(),
                    "guard at {index}"
                );
            }
        }
        actual.iter().map(|value| value.to_bits()).collect()
    }
    pub(super) fn run_production(
        &mut self,
        stream: &Arc<CudaStream>,
        kernels: &CudaNativeBlockKernels,
        captured: bool,
    ) {
        use cudarc::driver::{sys, DevicePtr, DevicePtrMut};
        use ferrum_interfaces::vnext::{ElementType, WeightId};
        stream
            .memcpy_htod(&self.output_initial, &mut self.output_gpu)
            .unwrap();
        stream.synchronize().unwrap();
        let input = self
            .input_gpu
            .slice(INPUT_PREFIX..INPUT_PREFIX + self.rows * self.inputs);
        let weight = self
            .weights_gpu
            .slice(WEIGHT_PREFIX..self.weights.len() - 8);
        let mut output = self
            .output_gpu
            .slice_mut(OUTPUT_PREFIX..OUTPUT_PREFIX + self.rows * self.stride);
        let (ip, _ig) = input.device_ptr(stream);
        let (wp, _wg) = weight.device_ptr(stream);
        let (op, _og) = output.device_ptr_mut(stream);
        let part = weights::MatrixPart {
            component_id: WeightId::new("fixed-k.conformance").unwrap(),
            format: weights::MatrixFormat::Block(self.format),
            rows: self.outputs as u32,
            columns: self.inputs as u32,
            output_offset: OUTPUT_COLUMN as u32,
            transform: None,
            signs_region: None,
        };
        if captured {
            stream
                .begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
                .unwrap();
        }
        kernels
            .linear(
                stream,
                ip,
                wp,
                op,
                &part,
                self.rows as u32,
                self.stride as u32,
                ElementType::F16,
            )
            .unwrap();
        if captured {
            let graph = stream
                .end_capture(
                    sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
                )
                .unwrap()
                .expect("production projection graph");
            let expected = if self.rows == 1 {
                "vnext_gguf_linear_q5k_f16"
            } else {
                "vnext_gguf_linear_q5k_tiled_f16"
            };
            assert_eq!(super::shared_dispatch::captured_kernel(&graph), expected);
            graph.launch().unwrap();
            stream.synchronize().unwrap();
        }
    }
}
