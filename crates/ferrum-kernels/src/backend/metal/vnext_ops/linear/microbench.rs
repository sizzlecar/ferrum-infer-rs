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
    activation_type: ElementType,
    input: &'a Buffer,
    weights: &'a [Buffer],
    output: &'a Buffer,
}

impl Case<'_> {
    fn poison_output(&self) {
        let len = self.rows as usize * self.shape.output as usize;
        // SAFETY: the shared allocation has this declared scalar type and
        // length. Every previous run waits for completion before returning.
        unsafe {
            match self.activation_type {
                ElementType::F16 => {
                    std::slice::from_raw_parts_mut(self.output.contents().cast::<f16>(), len)
                        .fill(f16::NAN)
                }
                ElementType::F32 => {
                    std::slice::from_raw_parts_mut(self.output.contents().cast::<f32>(), len)
                        .fill(f32::NAN)
                }
                _ => unreachable!("linear activation ABI"),
            }
        }
    }

    fn run(&self, dispatch: Dispatch, dispatches: usize) -> serde_json::Value {
        let params = LinearParams {
            rows: self.rows,
            in_features: self.shape.input,
            out_features: self.shape.output,
            output_stride: self.shape.output,
            output_column_offset: 0,
        };
        let (pipeline, kind) = match (dispatch, self.shape.format, self.activation_type) {
            (Dispatch::Gemv, _, ElementType::F32) => (
                self.pipelines
                    .f32_linear_pipeline(physical(self.shape.format))
                    .unwrap(),
                LinearDispatchKind::CooperativeGemv,
            ),
            (Dispatch::Gemv, GgufBlockFormat::Q4K, ElementType::F16) => (
                &self.pipelines.q4_k_gemv,
                LinearDispatchKind::CooperativeGemv,
            ),
            (Dispatch::Gemv, GgufBlockFormat::Q6K, ElementType::F16) => (
                &self.pipelines.q6_k_gemv,
                LinearDispatchKind::CooperativeGemv,
            ),
            (Dispatch::Gemm, GgufBlockFormat::Q4K, ElementType::F16) => (
                &self.pipelines.k_quant_gemm.q4_k,
                LinearDispatchKind::TiledGemm,
            ),
            (Dispatch::Gemm, GgufBlockFormat::Q6K, ElementType::F16) => (
                &self.pipelines.k_quant_gemm.q6_k,
                LinearDispatchKind::TiledGemm,
            ),
            (Dispatch::SharedWeight, _, ElementType::F16) => (
                self.pipelines
                    .small_batch
                    .pipeline(physical(self.shape.format), self.rows)
                    .unwrap(),
                LinearDispatchKind::SharedWeightGemv,
            ),
            (Dispatch::SharedWeight, _, ElementType::F32) => {
                let selected = self
                    .pipelines
                    .f32_linear_dispatch(physical(self.shape.format), self.rows, self.shape.output)
                    .unwrap();
                assert_eq!(selected.1, LinearDispatchKind::SharedWeightGemv);
                selected
            }
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
                self.activation_type,
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
        let values = read_linear_values(self.output, reference.len(), self.activation_type);
        for (index, (&actual, &expected)) in values.iter().zip(reference).enumerate() {
            let tolerance = linear_tolerance(self.activation_type, expected);
            assert!(
                actual.is_finite() && (actual - expected).abs() <= tolerance,
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

#[test]
#[ignore = "GPU performance experiment: coordinate exclusive device access"]
fn quantized_f32_shared_head_microbench() {
    let device = Device::system_default().expect("microbench requires Metal");
    let queue = device.new_command_queue();
    let pipelines = MetalLinearPipelines::new(&device).unwrap();
    // Qwen3.5-9B's vocabulary projection retains F32 activations and logits.
    // The local Q4_K_M model's head is Q6_K. F32 Q4_K retains the cooperative
    // implementation: sharing regressed at three and four rows on M1 Max.
    let shape = Shape {
        name: "vocabulary_head",
        input: 4096,
        output: 248_320,
        format: GgufBlockFormat::Q6K,
    };
    let weight_bytes = weights(shape);
    let weight_buffers = (0..4)
        .map(|_| buffer(&device, &weight_bytes))
        .collect::<Vec<_>>();
    for rows in [2, 3, 4] {
        let (_, mut nonzero) = inputs(rows, shape.input as usize);
        let mut input_values = vec![0.0_f32; rows * shape.input as usize];
        for (row, entries) in nonzero.iter_mut().enumerate() {
            for (column, value) in entries {
                *value += 0.000_123;
                input_values[row * shape.input as usize + *column] = *value;
            }
        }
        let reference = oracle(shape, &weight_bytes, &nonzero);
        let input = buffer(&device, &input_values);
        let output = buffer(&device, &vec![f32::NAN; rows * shape.output as usize]);
        for matrices in [1, 4] {
            let case = Case {
                queue: &queue,
                pipelines: &pipelines,
                shape,
                rows: rows as u32,
                activation_type: ElementType::F32,
                input: &input,
                weights: &weight_buffers[..matrices],
                output: &output,
            };
            let samples =
                measure_case(&case, &[Dispatch::Gemv, Dispatch::SharedWeight], &reference);
            println!(
                "{}",
                serde_json::json!({
                    "kind": "metal_quantized_f32_shared_head_microbench",
                    "device": device.name(), "shape": shape.name, "rows": rows,
                    "input_features": shape.input, "output_features": shape.output,
                    "activation_type": "f32", "weight_format": shape.format.format_id(),
                    "weight_bytes": weight_bytes.len(), "rotating_weight_allocations": matrices,
                    "warmup_rounds": WARMUP_ROUNDS,
                    "oracle": "full_output_sparse_input_cpu_block_decode",
                    "scope": "synthetic_projection_command_buffer_not_end_to_end_decode",
                    "samples": samples,
                })
            );
        }
    }
}

fn measure_case(
    case: &Case<'_>,
    variants: &[Dispatch],
    reference: &[f32],
) -> Vec<serde_json::Value> {
    for &dispatch in variants {
        // A skipped write must not inherit a correct result from the previous
        // variant. Reset outside all measured command-buffer intervals.
        case.poison_output();
        case.run(dispatch, 1);
        case.validate(reference);
    }
    for _ in 0..WARMUP_ROUNDS {
        for &dispatch in variants {
            case.run(dispatch, DISPATCHES_PER_COMMAND);
        }
    }
    let mut samples = Vec::new();
    for round in 0..MEASURED_ROUNDS {
        let mut order = variants.to_vec();
        let shift = round % order.len();
        order.rotate_left(shift);
        for dispatch in order {
            let mut sample = case.run(dispatch, DISPATCHES_PER_COMMAND);
            sample["round"] = serde_json::json!(round);
            samples.push(sample);
        }
    }
    case.validate(reference);
    samples
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
                    activation_type: ElementType::F16,
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
                let samples = measure_case(&case, &variants, &reference);
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
    shared_weight_conformance(7, false, ElementType::F16);
}

#[test]
fn production_small_batch_linear_matches_oracle_with_strided_output_tail() {
    shared_weight_conformance(1025, true, ElementType::F16);
}

#[test]
fn shared_weight_f32_gemv_preserves_precision_rows_offsets_and_output_guards() {
    shared_weight_conformance(7, false, ElementType::F32);
    shared_weight_conformance(1025, true, ElementType::F32);
}

fn read_linear_values(output: &Buffer, len: usize, activation_type: ElementType) -> Vec<f32> {
    // SAFETY: callers allocate the declared scalar type and length and wait for
    // Metal command completion before reading the shared output allocation.
    unsafe {
        match activation_type {
            ElementType::F16 => std::slice::from_raw_parts(output.contents().cast::<f16>(), len)
                .iter()
                .map(|value| value.to_f32())
                .collect(),
            ElementType::F32 => {
                std::slice::from_raw_parts(output.contents().cast::<f32>(), len).to_vec()
            }
            _ => unreachable!("linear activation ABI"),
        }
    }
}

fn linear_tolerance(activation_type: ElementType, expected: f32) -> f32 {
    match activation_type {
        ElementType::F16 => 0.002 + 0.003 * expected.abs(),
        ElementType::F32 => 2.0e-4_f32.max(expected.abs() * 2.0e-4),
        _ => unreachable!("linear activation ABI"),
    }
}

fn linear_values_buffer(device: &Device, values: &[f32], activation_type: ElementType) -> Buffer {
    match activation_type {
        ElementType::F16 => buffer(
            device,
            &values
                .iter()
                .copied()
                .map(f16::from_f32)
                .collect::<Vec<_>>(),
        ),
        ElementType::F32 => buffer(device, values),
        _ => unreachable!("linear activation ABI"),
    }
}

fn shared_weight_conformance(
    output_width: u32,
    production_dispatch: bool,
    activation_type: ElementType,
) {
    let device = Device::system_default().expect("small-batch conformance requires Metal");
    let queue = device.new_command_queue();
    let pipelines = MetalLinearPipelines::new(&device).unwrap();
    let formats: &[GgufBlockFormat] = if activation_type == ElementType::F32 {
        &[GgufBlockFormat::Q6K]
    } else {
        &[GgufBlockFormat::Q4K, GgufBlockFormat::Q6K]
    };
    for &format in formats {
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
                .map(|index| {
                    let value = (index as f32 * 0.013).sin() * 0.125;
                    if activation_type == ElementType::F16 {
                        f16::from_f32(value).to_f32()
                    } else if index % shape.input as usize == 13 {
                        // A finite F32 input outside F16's range makes an
                        // accidental half conversion observable at this ABI.
                        1_048_576.0 * (index / shape.input as usize + 1) as f32
                    } else {
                        value + 0.000_123
                    }
                })
                .collect::<Vec<_>>();
            let dense_entries = input_values
                .chunks_exact(shape.input as usize)
                .map(|row| {
                    row.iter()
                        .enumerate()
                        .map(|(index, value)| (index, *value))
                        .collect()
                })
                .collect::<Vec<Vec<_>>>();
            let reference = oracle(shape, &weight_values, &dense_entries);
            let mut padded_input = vec![-123.0; 8];
            padded_input.extend_from_slice(&input_values);
            let input = linear_values_buffer(&device, &padded_input, activation_type);
            let stride = output_width as usize + 6;
            let column_offset = 3;
            let prefix = 8;
            let sentinel = 123.0;
            let elements = prefix + rows * stride + 8;
            let output = linear_values_buffer(&device, &vec![sentinel; elements], activation_type);
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
                if activation_type == ElementType::F32 {
                    pipelines
                        .f32_linear_dispatch(physical(format), params.rows, params.out_features)
                        .unwrap()
                } else {
                    pipelines.linear_pipeline(physical(format), params.rows, params.out_features)
                }
            } else if activation_type == ElementType::F32 {
                (
                    pipelines
                        .small_batch
                        .f32_pipeline(physical(format), rows as u32)
                        .unwrap(),
                    LinearDispatchKind::SharedWeightGemv,
                )
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
            let scalar_bytes = if activation_type == ElementType::F32 {
                4
            } else {
                2
            };
            encoder.set_buffer(0, Some(&input), 8 * scalar_bytes);
            encoder.set_buffer(1, Some(&weight), 16);
            encoder.set_buffer(2, Some(&output), prefix as u64 * scalar_bytes);
            bind_linear_params(encoder, params, physical(format), activation_type);
            dispatch_linear_grid(encoder, params, dispatch_kind);
            encoder.end_encoding();
            command.commit();
            command.wait_until_completed();
            assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
            let actual = read_linear_values(&output, elements, activation_type);
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
                        value.is_finite()
                            && (value - expected).abs()
                                <= linear_tolerance(activation_type, expected),
                        "{format:?} rows={rows} index={index}: {} vs {expected}",
                        value
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
