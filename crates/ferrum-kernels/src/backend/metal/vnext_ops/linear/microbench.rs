//! Opt-in comparison of existing Metal linear dispatches at decode batch widths.
//!
//! Run after coordinating exclusive GPU access:
//! `cargo test --release -p ferrum-kernels --features metal --lib
//! quantized_decode_dispatch_microbench -- --ignored --nocapture --test-threads=1`
//! This measures synthetic weights with real Qwen3.5-9B projection dimensions,
//! not model quality or end-to-end decode. No timing threshold gates correctness.

use super::*;
use crate::gguf_blocks::fixtures::oracle_blocks;
use half::f16;
use metal::objc::runtime::{BOOL, YES};
use metal::objc::{msg_send, sel, sel_impl};
use metal::{
    Buffer, CommandBufferRef, CommandQueueRef, MTLCommandBufferStatus, MTLResourceOptions,
};
use std::time::Instant;

const WARMUP_ROUNDS: usize = 2;
const MEASURED_ROUNDS: usize = 7;
const DISPATCHES_PER_COMMAND: usize = 8;

#[derive(Clone, Copy)]
struct Shape {
    name: &'static str,
    input: u32,
    output: u32,
    format: GgufBlockFormat,
}

// Text config: hidden=4096, intermediate=12288, Q heads=16, KV heads=4,
// head_dim=256 and gated Q projection. The large vocabulary projection is
// intentionally outside this bounded experiment. Quantization labels describe
// the tested encoding, not an assertion that every checkpoint tensor uses it.
const SHAPES: &[Shape] = &[
    Shape {
        name: "ffn_gate_or_up",
        input: 4096,
        output: 12288,
        format: GgufBlockFormat::Q4K,
    },
    Shape {
        name: "ffn_down",
        input: 12288,
        output: 4096,
        format: GgufBlockFormat::Q6K,
    },
    Shape {
        name: "gated_attention_query",
        input: 4096,
        output: 8192,
        format: GgufBlockFormat::Q4K,
    },
    Shape {
        name: "attention_key_or_value",
        input: 4096,
        output: 1024,
        format: GgufBlockFormat::Q4K,
    },
    Shape {
        name: "attention_output",
        input: 4096,
        output: 4096,
        format: GgufBlockFormat::Q4K,
    },
    Shape {
        name: "small_projection",
        input: 4096,
        output: 64,
        format: GgufBlockFormat::Q4K,
    },
];

#[derive(Clone, Copy, Debug)]
enum Dispatch {
    Gemv,
    Gemm,
    SharedWeight,
}

fn buffer<T>(device: &Device, data: &[T]) -> Buffer {
    device.new_buffer_with_data(
        data.as_ptr().cast(),
        std::mem::size_of_val(data) as u64,
        MTLResourceOptions::StorageModeShared,
    )
}

fn physical(format: GgufBlockFormat) -> LinearPhysicalFormat {
    match format {
        GgufBlockFormat::Q4K => LinearPhysicalFormat::Q4K,
        GgufBlockFormat::Q6K => LinearPhysicalFormat::Q6K,
        _ => unreachable!("bounded Q4_K/Q6_K experiment"),
    }
}

fn weights(shape: Shape) -> Vec<u8> {
    let blocks_per_row = shape.input as usize / shape.format.block_values();
    let templates = oracle_blocks(shape.format);
    let block_bytes = shape.format.block_bytes();
    let mut data = vec![0; shape.output as usize * blocks_per_row * block_bytes];
    for (index, block) in data.chunks_exact_mut(block_bytes).enumerate() {
        let row = index / blocks_per_row;
        let source = ((index + row / 3) % 2) * block_bytes;
        block.copy_from_slice(&templates[source..source + block_bytes]);
        let scale_offset = if shape.format == GgufBlockFormat::Q6K {
            208
        } else {
            0
        };
        let scale = if (index + row) % 3 == 0 {
            -1.0 / 512.0
        } else {
            1.0 / 256.0
        };
        block[scale_offset..scale_offset + 2].copy_from_slice(&f16::from_f32(scale).to_le_bytes());
        if shape.format == GgufBlockFormat::Q4K {
            block[2..4].copy_from_slice(&f16::from_f32(1.0 / 1024.0).to_le_bytes());
        }
    }
    data
}

fn inputs(rows: usize, width: usize) -> (Vec<f16>, Vec<Vec<(usize, f32)>>) {
    let mut data = vec![f16::ZERO; rows * width];
    let mut nonzero = Vec::with_capacity(rows);
    for row in 0..rows {
        let mut entries = Vec::new();
        for index in 0..8 {
            // Sparse values permit an independent full-output CPU oracle.
            // Both GPU kernels still execute their normal dense arithmetic.
            let column = (index * (width / 8) + 13 * row + 7) % width;
            let value = f16::from_f32(((index + 3 * row) as f32 * 0.7).sin() * 0.125);
            data[row * width + column] = value;
            entries.push((column, value.to_f32()));
        }
        nonzero.push(entries);
    }
    (data, nonzero)
}

fn oracle(shape: Shape, weights: &[u8], inputs: &[Vec<(usize, f32)>]) -> Vec<f32> {
    let values_per_block = shape.format.block_values();
    let bytes_per_block = shape.format.block_bytes();
    let row_bytes = shape.input as usize / values_per_block * bytes_per_block;
    let mut result = Vec::with_capacity(inputs.len() * shape.output as usize);
    for input in inputs {
        for output in 0..shape.output as usize {
            let mut sum = 0.0;
            for &(column, value) in input {
                let offset = output * row_bytes + column / values_per_block * bytes_per_block;
                sum += value
                    * shape.format.decode_value(
                        &weights[offset..offset + bytes_per_block],
                        column % values_per_block,
                    );
            }
            result.push(sum);
        }
    }
    result
}

#[allow(
    unexpected_cfgs,
    reason = "objc 0.2 macros expand their legacy cargo-clippy feature cfg in the calling crate"
)]
fn gpu_elapsed_ns(command: &CommandBufferRef) -> Option<f64> {
    let start_selector = sel!(GPUStartTime);
    let end_selector = sel!(GPUEndTime);
    // SAFETY: query selector availability on the completed Metal command before
    // invoking documented double-valued command-buffer timestamp properties.
    let (has_start, has_end): (BOOL, BOOL) = unsafe {
        (
            msg_send![command, respondsToSelector: start_selector],
            msg_send![command, respondsToSelector: end_selector],
        )
    };
    if has_start != YES || has_end != YES {
        return None;
    }
    let (start, end): (f64, f64) = unsafe {
        (
            msg_send![command, GPUStartTime],
            msg_send![command, GPUEndTime],
        )
    };
    (start.is_finite() && end.is_finite() && start > 0.0 && end > start)
        .then_some((end - start) * 1e9)
}

struct Case<'a> {
    queue: &'a CommandQueueRef,
    pipelines: &'a MetalLinearPipelines,
    shape: Shape,
    rows: u32,
    input: &'a Buffer,
    weights: &'a [Buffer],
    output: &'a Buffer,
}

impl Case<'_> {
    fn run(&self, dispatch: Dispatch, dispatches: usize) -> serde_json::Value {
        let params = LinearParams {
            rows: self.rows,
            in_features: self.shape.input,
            out_features: self.shape.output,
            output_stride: self.shape.output,
            output_column_offset: 0,
        };
        let (pipeline, kind) = match (dispatch, self.shape.format) {
            (Dispatch::Gemv, GgufBlockFormat::Q4K) => (
                &self.pipelines.q4_k_gemv,
                LinearDispatchKind::CooperativeGemv,
            ),
            (Dispatch::Gemv, GgufBlockFormat::Q6K) => (
                &self.pipelines.q6_k_gemv,
                LinearDispatchKind::CooperativeGemv,
            ),
            (Dispatch::Gemm, GgufBlockFormat::Q4K) => (
                &self.pipelines.k_quant_gemm.q4_k,
                LinearDispatchKind::TiledGemm,
            ),
            (Dispatch::Gemm, GgufBlockFormat::Q6K) => (
                &self.pipelines.k_quant_gemm.q6_k,
                LinearDispatchKind::TiledGemm,
            ),
            (Dispatch::SharedWeight, _) => (
                self.pipelines
                    .small_batch
                    .pipeline(physical(self.shape.format), self.rows)
                    .unwrap(),
                LinearDispatchKind::SharedWeightGemv,
            ),
            _ => unreachable!(),
        };
        let started = Instant::now();
        let command = self.queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        for index in 0..dispatches {
            encoder.set_compute_pipeline_state(pipeline);
            encoder.set_buffer(0, Some(self.input), 0);
            encoder.set_buffer(1, Some(&self.weights[index % self.weights.len()]), 0);
            encoder.set_buffer(2, Some(self.output), 0);
            bind_linear_params(
                encoder,
                params,
                physical(self.shape.format),
                ElementType::F16,
            );
            dispatch_linear_grid(encoder, params, kind);
        }
        encoder.end_encoding();
        let encode_ns = started.elapsed().as_nanos() as u64;
        let submitted = Instant::now();
        command.commit();
        command.wait_until_completed();
        let submit_wait_ns = submitted.elapsed().as_nanos() as u64;
        assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
        serde_json::json!({
            "dispatch": format!("{dispatch:?}"),
            "dispatches": dispatches,
            "host_encode_ns": encode_ns,
            "host_submit_wait_ns": submit_wait_ns,
            "device_command_ns": gpu_elapsed_ns(command),
        })
    }

    fn validate(&self, reference: &[f32]) {
        // SAFETY: shared F16 buffer of the requested length; run() waited for
        // command completion before this host read.
        let values = unsafe {
            std::slice::from_raw_parts(self.output.contents().cast::<f16>(), reference.len())
        };
        for (index, (&actual, &expected)) in values.iter().zip(reference).enumerate() {
            let actual = actual.to_f32();
            assert!(
                actual.is_finite() && (actual - expected).abs() <= 0.002 + 0.003 * expected.abs(),
                "{} rows={} output[{index}] actual={actual} oracle={expected}",
                self.shape.name,
                self.rows
            );
        }
    }
}

#[test]
#[ignore = "GPU performance experiment: coordinate exclusive device access"]
fn quantized_decode_dispatch_microbench() {
    measure_dispatches(&[1, 2, 3, 4, 8], true);
}

#[test]
#[ignore = "GPU performance experiment: coordinate exclusive device access"]
fn quantized_shared_weight_microbench() {
    measure_dispatches(&[2, 3, 4], false);
}

fn measure_dispatches(row_counts: &[usize], include_gemm: bool) {
    let device = Device::system_default().expect("microbench requires Metal");
    let queue = device.new_command_queue();
    let pipelines = MetalLinearPipelines::new(&device).unwrap();
    for &shape in SHAPES {
        let weight_bytes = weights(shape);
        // Four distinct allocations expose sensitivity to repeatedly reusing
        // one matrix. Even rotation is not the full 9B model's cache behavior.
        let weight_buffers = (0..4)
            .map(|_| buffer(&device, &weight_bytes))
            .collect::<Vec<_>>();
        for &rows in row_counts {
            let (input_values, nonzero) = inputs(rows, shape.input as usize);
            let reference = oracle(shape, &weight_bytes, &nonzero);
            let input = buffer(&device, &input_values);
            let output = buffer(&device, &vec![f16::NAN; rows * shape.output as usize]);
            for matrices in [1, 4] {
                let case = Case {
                    queue: &queue,
                    pipelines: &pipelines,
                    shape,
                    rows: rows as u32,
                    input: &input,
                    weights: &weight_buffers[..matrices],
                    output: &output,
                };
                let mut variants = vec![Dispatch::Gemv];
                if include_gemm {
                    variants.push(Dispatch::Gemm);
                }
                if (2..=4).contains(&rows) {
                    variants.push(Dispatch::SharedWeight);
                }
                for &dispatch in &variants {
                    case.run(dispatch, 1);
                    case.validate(&reference);
                }
                for _ in 0..WARMUP_ROUNDS {
                    for &dispatch in &variants {
                        case.run(dispatch, DISPATCHES_PER_COMMAND);
                    }
                }
                let mut samples = Vec::new();
                for round in 0..MEASURED_ROUNDS {
                    let mut order = variants.clone();
                    let shift = round % order.len();
                    order.rotate_left(shift);
                    for dispatch in order {
                        let mut sample = case.run(dispatch, DISPATCHES_PER_COMMAND);
                        sample["round"] = serde_json::json!(round);
                        samples.push(sample);
                    }
                }
                case.validate(&reference);
                println!(
                    "{}",
                    serde_json::json!({
                        "kind": "metal_quantized_decode_dispatch_microbench",
                        "device": device.name(), "shape": shape.name, "rows": rows,
                        "input_features": shape.input, "output_features": shape.output,
                        "weight_format": shape.format.format_id(), "weight_bytes": weight_bytes.len(),
                        "rotating_weight_allocations": matrices, "warmup_rounds": WARMUP_ROUNDS,
                        "production_dispatch": format!("{:?}", pipelines.linear_pipeline(physical(shape.format), rows as u32, shape.output).1),
                        "oracle": "full_output_sparse_input_cpu_block_decode",
                        "scope": "synthetic_projection_command_buffer_not_end_to_end_decode",
                        "samples": samples,
                    })
                );
            }
        }
    }
}

#[test]
fn shared_weight_gemv_preserves_rows_offsets_and_output_guards() {
    shared_weight_conformance(7, false);
}

#[test]
fn production_small_batch_linear_matches_oracle_with_strided_output_tail() {
    shared_weight_conformance(1025, true);
}

fn shared_weight_conformance(output_width: u32, production_dispatch: bool) {
    let device = Device::system_default().expect("small-batch conformance requires Metal");
    let queue = device.new_command_queue();
    let pipelines = MetalLinearPipelines::new(&device).unwrap();
    for format in [GgufBlockFormat::Q4K, GgufBlockFormat::Q6K] {
        let shape = Shape {
            name: "strided_tail",
            input: 1280,
            output: output_width,
            format,
        };
        let weight_values = weights(shape);
        let mut padded_weights = vec![0; 16];
        padded_weights.extend_from_slice(&weight_values);
        let weight = buffer(&device, &padded_weights);
        for rows in [2, 3, 4] {
            let input_values = (0..rows * shape.input as usize)
                .map(|index| f16::from_f32((index as f32 * 0.013).sin() * 0.125))
                .collect::<Vec<_>>();
            let dense_entries = input_values
                .chunks_exact(shape.input as usize)
                .map(|row| {
                    row.iter()
                        .enumerate()
                        .map(|(index, value)| (index, value.to_f32()))
                        .collect()
                })
                .collect::<Vec<Vec<_>>>();
            let reference = oracle(shape, &weight_values, &dense_entries);
            let mut padded_input = vec![f16::from_f32(-123.0); 8];
            padded_input.extend_from_slice(&input_values);
            let input = buffer(&device, &padded_input);
            let stride = output_width as usize + 6;
            let column_offset = 3;
            let prefix = 8;
            let sentinel = f16::from_f32(123.0);
            let elements = prefix + rows * stride + 8;
            let output = buffer(&device, &vec![sentinel; elements]);
            let params = LinearParams {
                rows: rows as u32,
                in_features: shape.input,
                out_features: shape.output,
                output_stride: stride as u32,
                output_column_offset: column_offset as u32,
            };
            let command = queue.new_command_buffer();
            let encoder = command.new_compute_command_encoder();
            let (pipeline, dispatch_kind) = if production_dispatch {
                pipelines.linear_pipeline(physical(format), params.rows, params.out_features)
            } else {
                (
                    pipelines
                        .small_batch
                        .pipeline(physical(format), rows as u32)
                        .unwrap(),
                    LinearDispatchKind::SharedWeightGemv,
                )
            };
            assert_eq!(dispatch_kind, LinearDispatchKind::SharedWeightGemv);
            encoder.set_compute_pipeline_state(pipeline);
            encoder.set_buffer(0, Some(&input), 16);
            encoder.set_buffer(1, Some(&weight), 16);
            encoder.set_buffer(2, Some(&output), 16);
            bind_linear_params(encoder, params, physical(format), ElementType::F16);
            dispatch_linear_grid(encoder, params, dispatch_kind);
            encoder.end_encoding();
            command.commit();
            command.wait_until_completed();
            assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
            // SAFETY: command completion precedes reading this shared allocation.
            let actual =
                unsafe { std::slice::from_raw_parts(output.contents().cast::<f16>(), elements) };
            for (index, &value) in actual.iter().enumerate() {
                let relative = index.saturating_sub(prefix);
                let row = relative / stride;
                let column = relative % stride;
                if index >= prefix
                    && row < rows
                    && column >= column_offset
                    && column < column_offset + shape.output as usize
                {
                    let expected = reference[row * shape.output as usize + column - column_offset];
                    assert!(
                        (value.to_f32() - expected).abs() <= 0.002 + 0.003 * expected.abs(),
                        "{format:?} rows={rows} index={index}: {} vs {expected}",
                        value.to_f32()
                    );
                } else {
                    assert_eq!(
                        value, sentinel,
                        "{format:?} rows={rows} overwrote guard {index}"
                    );
                }
            }
        }
    }
}
