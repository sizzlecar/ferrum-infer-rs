//! Paired comparisons of existing B4 kernels and the original full-row route.

use super::*;
use crate::backend::metal::vnext_ops::MetalVNextComposition;
use ferrum_interfaces::vnext::{BufferRequest, BufferUsage, DeviceId, ResourceId};

const INPUT_PREFIX: usize = 4;
const OUTPUT_PREFIX: usize = 8;
const WEIGHT_PREFIX: usize = 16;
const COLUMN_OFFSET: usize = 2;

fn activation_bytes(values: &[f32], dtype: ElementType) -> Vec<u8> {
    match dtype {
        ElementType::F16 => values
            .iter()
            .flat_map(|value| f16::from_f32(*value).to_le_bytes())
            .collect(),
        ElementType::F32 => values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect(),
        _ => unreachable!(),
    }
}

fn retained(
    runtime: &MetalDeviceRuntime,
    name: &str,
    bytes: &[u8],
    dtype: ElementType,
) -> MetalBufferRegion {
    let region = runtime
        .allocate_test_region(
            &BufferRequest::new(
                ResourceId::new(name).unwrap(),
                bytes.len() as u64,
                64,
                BufferUsage::Transfer,
                dtype,
            )
            .unwrap(),
        )
        .unwrap();
    // SAFETY: this newly allocated shared region has exactly this byte length
    // and has not been submitted to the GPU yet.
    unsafe {
        std::ptr::copy_nonoverlapping(
            bytes.as_ptr(),
            region
                .buffer()
                .contents()
                .cast::<u8>()
                .add(region.offset_bytes() as usize),
            bytes.len(),
        );
    }
    region
}

struct Fixture {
    params: LinearParams,
    format: GgufBlockFormat,
    activation_type: ElementType,
    input_values: Vec<f32>,
    weight_bytes: Vec<u8>,
    initial_output: Vec<f32>,
    reference: Vec<f64>,
    input: Buffer,
    weight: Buffer,
    output: Buffer,
    regions: Vec<MetalBufferRegion>,
    launch: LinearLaunch,
}

impl Fixture {
    fn new(
        runtime: &MetalDeviceRuntime,
        rows: u32,
        width: u32,
        outputs: u32,
        format: GgufBlockFormat,
        activation_type: ElementType,
    ) -> Self {
        let shape = Shape {
            name: "four_row_groups",
            input: width,
            output: outputs,
            format,
        };
        let encoded = weights(shape);
        let (_, mut entries) = inputs(rows as usize, width as usize);
        let mut input_values =
            vec![OUTPUT_GUARD; INPUT_PREFIX + rows as usize * width as usize + 8];
        input_values[INPUT_PREFIX..INPUT_PREFIX + rows as usize * width as usize].fill(0.0);
        for (row, entries) in entries.iter_mut().enumerate() {
            for (column, value) in entries {
                // Preserve F32-only mantissa bits at the vocabulary boundary.
                if activation_type == ElementType::F32 {
                    *value += 0.000_123;
                }
                input_values[INPUT_PREFIX + row * width as usize + *column] = *value;
            }
        }
        let row_bytes = width as usize / format.block_values() * format.block_bytes();
        let mut reference = Vec::new();
        for entries in &entries {
            for output in 0..outputs as usize {
                reference.push(
                    entries
                        .iter()
                        .map(|&(column, value)| {
                            let offset = output * row_bytes
                                + column / format.block_values() * format.block_bytes();
                            f64::from(value)
                                * f64::from(format.decode_value(
                                    &encoded[offset..offset + format.block_bytes()],
                                    column % format.block_values(),
                                ))
                        })
                        .sum::<f64>(),
                );
            }
        }
        let mut weight_bytes = vec![0xcc; WEIGHT_PREFIX];
        weight_bytes.extend(encoded);
        weight_bytes.extend([0xcc; 16]);
        let stride = outputs + 5;
        let mut initial_output =
            vec![OUTPUT_GUARD; OUTPUT_PREFIX + rows as usize * stride as usize + 8];
        for row in 0..rows as usize {
            let start = OUTPUT_PREFIX + row * stride as usize + COLUMN_OFFSET;
            initial_output[start..start + outputs as usize].fill(f32::NAN);
        }
        let input_bytes = activation_bytes(&input_values, activation_type);
        let output_bytes = activation_bytes(&initial_output, activation_type);
        let parents = [
            retained(runtime, "groups.input", &input_bytes, activation_type),
            retained(runtime, "groups.weight", &weight_bytes, ElementType::U8),
            retained(runtime, "groups.output", &output_bytes, activation_type),
        ];
        let scalar_bytes = activation_type.size_bytes();
        // Split each prefix between retained-region and local launch offsets.
        let regions = vec![
            parents[0]
                .test_subregion(
                    INPUT_PREFIX as u64 / 2 * scalar_bytes
                        ..input_bytes.len() as u64 - 8 * scalar_bytes,
                )
                .unwrap(),
            parents[1]
                .test_subregion(WEIGHT_PREFIX as u64..weight_bytes.len() as u64 - 16)
                .unwrap(),
            parents[2]
                .test_subregion(
                    OUTPUT_PREFIX as u64 / 2 * scalar_bytes
                        ..output_bytes.len() as u64 - 8 * scalar_bytes,
                )
                .unwrap(),
        ];
        let launch = linear_launch_typed(
            PreparedLinearPart {
                region: 1,
                transform: None,
                format: physical(format),
                output_offset: COLUMN_OFFSET as u32,
                out_features: outputs,
            },
            0,
            2,
            u64::from(rows),
            u64::from(width),
            u64::from(stride),
            INPUT_PREFIX as u64 / 2 * scalar_bytes,
            OUTPUT_PREFIX as u64 / 2 * scalar_bytes,
            activation_type,
        )
        .unwrap();
        validate_launch_regions(&regions, &[launch]).unwrap();
        Self {
            params: launch.params,
            format,
            activation_type,
            input: parents[0].buffer().to_owned(),
            weight: parents[1].buffer().to_owned(),
            output: parents[2].buffer().to_owned(),
            input_values,
            weight_bytes,
            initial_output,
            reference,
            regions,
            launch,
        }
    }

    fn run(
        &self,
        pipelines: &MetalLinearPipelines,
        queue: &CommandQueueRef,
        grouped: bool,
        iterations: u32,
    ) -> (f64, Option<f64>) {
        self.run_mode(pipelines, queue, Some(grouped), iterations)
    }

    fn run_mode(
        &self,
        pipelines: &MetalLinearPipelines,
        queue: &CommandQueueRef,
        grouped: Option<bool>,
        iterations: u32,
    ) -> (f64, Option<f64>) {
        assert!(iterations > 0);
        if grouped == Some(true) {
            assert!(self.params.rows.is_multiple_of(4));
        }
        let initial_bytes = activation_bytes(&self.initial_output, self.activation_type);
        // SAFETY: this fixture owns the initialized shared output; the previous
        // command has completed before every reset and readback.
        unsafe {
            std::ptr::copy_nonoverlapping(
                initial_bytes.as_ptr(),
                self.output.contents().cast::<u8>(),
                initial_bytes.len(),
            );
        }
        let (pipeline, kind) = if grouped == Some(true) {
            (
                match self.activation_type {
                    ElementType::F32 => {
                        pipelines.small_batch.f32_pipeline(physical(self.format), 4)
                    }
                    ElementType::F16 => pipelines.small_batch.pipeline(physical(self.format), 4),
                    _ => unreachable!(),
                }
                .unwrap(),
                LinearDispatchKind::SharedWeightGemv,
            )
        } else {
            pipelines.plain_linear_dispatch(
                physical(self.format),
                self.activation_type,
                self.params,
            )
        };
        let group_rows = if grouped == Some(true) {
            4
        } else {
            self.params.rows
        };
        let groups = self.params.rows / group_rows;
        let params = LinearParams {
            rows: group_rows,
            ..self.params
        };
        let started = Instant::now();
        let command = queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        let scalar_bytes = self.activation_type.size_bytes();
        for _ in 0..iterations {
            if grouped.is_none() {
                dispatch_linear(pipelines, encoder, &self.regions, self.launch);
                continue;
            }
            for group in 0..groups {
                let row = u64::from(group * group_rows);
                encoder.set_compute_pipeline_state(pipeline);
                encoder.set_buffer(
                    0,
                    Some(&self.input),
                    (INPUT_PREFIX as u64 + row * u64::from(params.in_features)) * scalar_bytes,
                );
                encoder.set_buffer(1, Some(&self.weight), WEIGHT_PREFIX as u64);
                encoder.set_buffer(
                    2,
                    Some(&self.output),
                    (OUTPUT_PREFIX as u64 + row * u64::from(params.output_stride)) * scalar_bytes,
                );
                bind_linear_params(encoder, params, physical(self.format), self.activation_type);
                dispatch_linear_grid(encoder, params, kind);
            }
        }
        encoder.end_encoding();
        command.commit();
        command.wait_until_completed();
        let wall_ns = started.elapsed().as_secs_f64() * 1e9;
        assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
        (wall_ns, gpu_elapsed_ns(command))
    }

    fn validate(&self) -> Vec<u32> {
        // SAFETY: all three allocations have the recorded scalar lengths and
        // every submission completed before this CPU access.
        let input = read_linear_values(&self.input, self.input_values.len(), self.activation_type);
        assert_eq!(input, self.input_values);
        let weights = unsafe {
            std::slice::from_raw_parts(self.weight.contents().cast::<u8>(), self.weight_bytes.len())
        };
        assert_eq!(
            &weights[..WEIGHT_PREFIX],
            &self.weight_bytes[..WEIGHT_PREFIX]
        );
        assert_eq!(
            &weights[weights.len() - 16..],
            &self.weight_bytes[weights.len() - 16..]
        );
        let values = read_linear_values(
            &self.output,
            self.initial_output.len(),
            self.activation_type,
        );
        let stride = self.params.output_stride as usize;
        let outputs = self.params.out_features as usize;
        for (index, &actual) in values.iter().enumerate() {
            let location = index
                .checked_sub(OUTPUT_PREFIX)
                .filter(|index| *index < self.params.rows as usize * stride)
                .filter(|index| {
                    (COLUMN_OFFSET..COLUMN_OFFSET + outputs).contains(&(index % stride))
                });
            if let Some(index) = location {
                let expected =
                    self.reference[index / stride * outputs + index % stride - COLUMN_OFFSET];
                // Retain the existing activation ABI tolerance; independently
                // decoded source products are accumulated in F64 here.
                let tolerance = f64::from(linear_tolerance(self.activation_type, expected as f32));
                assert!(
                    actual.is_finite() && (f64::from(actual) - expected).abs() <= tolerance,
                    "rows={} index={index}: {actual} != {expected}",
                    self.params.rows
                );
            } else {
                assert_eq!(
                    actual.to_bits(),
                    self.initial_output[index].to_bits(),
                    "guard {index}"
                );
            }
        }
        values.iter().map(|value| value.to_bits()).collect()
    }
}

#[test]
fn q6_f32_head_four_row_groups_match_cpu_on_metal() {
    let composition =
        MetalVNextComposition::create(DeviceId::new("metal.q6-groups.test").unwrap()).unwrap();
    let runtime = composition.runtime();
    let pipelines = MetalLinearPipelines::new(runtime.device()).unwrap();
    let queue = runtime.device().new_command_queue();
    for rows in [4, 5, 7, 8, 9] {
        let fixture = Fixture::new(
            runtime,
            rows,
            768,
            1025,
            GgufBlockFormat::Q6K,
            ElementType::F32,
        );
        assert_eq!(
            fixture.launch.dispatch_count(),
            if rows == 8 { 2 } else { 1 }
        );
        fixture.run(&pipelines, &queue, false, 1);
        let baseline_bits = fixture.validate();
        for grouped in [Some(false), None] {
            fixture.run_mode(&pipelines, &queue, grouped, 1);
            let reference = fixture.validate();
            assert_eq!(
                reference, baseline_bits,
                "prepared grouping changed F32 bits"
            );
            fixture.run_mode(&pipelines, &queue, grouped, 2);
            assert_eq!(
                fixture.validate(),
                reference,
                "same eager dispatch changed bits"
            );
        }
        let mut truncated = fixture.regions.clone();
        let bytes = truncated[2].length_bytes();
        truncated[2] = truncated[2].test_subregion(0..bytes - 4).unwrap();
        assert!(validate_launch_regions(&truncated, &[fixture.launch]).is_err());
    }
}

#[test]
#[ignore = "Q6 F32 vocabulary candidate timing; coordinate exclusive Metal access"]
fn q6_f32_head_four_row_groups_microbench() {
    const ITERATIONS: u32 = 8;
    let composition =
        MetalVNextComposition::create(DeviceId::new("metal.q6-groups.bench").unwrap()).unwrap();
    let runtime = composition.runtime();
    let pipelines = MetalLinearPipelines::new(runtime.device()).unwrap();
    let queue = runtime.device().new_command_queue();
    for rows in [4, 8] {
        // Actual Qwen3.5-4B vocabulary dimensions; synthetic Q6K source bytes.
        let fixture = Fixture::new(
            runtime,
            rows,
            2560,
            248_320,
            GgufBlockFormat::Q6K,
            ElementType::F32,
        );
        fixture.run(&pipelines, &queue, false, 1);
        let baseline_bits = fixture.validate();
        for round in 0..10 {
            for grouped in if round % 2 == 0 {
                [false, true]
            } else {
                [true, false]
            } {
                let (wall_ns, gpu_ns) = fixture.run(&pipelines, &queue, grouped, ITERATIONS);
                let bits = fixture.validate();
                if round >= 2 {
                    println!(
                        "{}",
                        serde_json::json!({
                            "benchmark": "q6_f32_head_four_row_groups", "rows": rows,
                            "input": 2560, "output": 248320, "activation_type": "f32", "weight_format": "q6_k",
                            "grouped": grouped, "round": round - 2, "projection_iterations": ITERATIONS,
                            "physical_dispatches_per_iteration": if grouped { rows / 4 } else { 1 },
                            "command_wall_ns": wall_ns, "command_gpu_ns": gpu_ns,
                            "wall_ns": wall_ns / f64::from(ITERATIONS), "gpu_ns": gpu_ns.map(|ns| ns / f64::from(ITERATIONS)),
                            "bitwise_equal_to_baseline": bits == baseline_bits,
                        })
                    );
                }
            }
        }
    }
}

#[test]
fn quantized_f16_four_row_groups_match_cpu_on_metal() {
    let composition =
        MetalVNextComposition::create(DeviceId::new("metal.down-groups.test").unwrap()).unwrap();
    let runtime = composition.runtime();
    let pipelines = MetalLinearPipelines::new(runtime.device()).unwrap();
    let queue = runtime.device().new_command_queue();
    for format in [
        GgufBlockFormat::Q4K,
        GgufBlockFormat::Q5K,
        GgufBlockFormat::Q6K,
    ] {
        for width in [768, 1024, 2560, 9216] {
            for rows in [4, 5, 7, 8, 9] {
                let fixture = Fixture::new(runtime, rows, width, 1025, format, ElementType::F16);
                let selected =
                    rows == 8 && width >= 1024 && (format != GgufBlockFormat::Q5K || 1025 < width);
                assert_eq!(
                    fixture.launch.dispatch_count(),
                    if selected { 2 } else { 1 }
                );
                fixture.run(&pipelines, &queue, selected, 1);
                let selected_bits = fixture.validate();
                for route in [Some(false), None] {
                    fixture.run_mode(&pipelines, &queue, route, 1);
                    let bits = fixture.validate();
                    if route.is_none() {
                        assert_eq!(
                            bits, selected_bits,
                            "prepared dispatch differed from selected route"
                        );
                    }
                    fixture.run_mode(&pipelines, &queue, route, 2);
                    assert_eq!(fixture.validate(), bits, "same F16 dispatch changed bits");
                }
            }
        }
    }
}

#[test]
#[ignore = "F16 projection candidate timing; coordinate exclusive Metal access"]
fn quantized_f16_four_row_groups_microbench() {
    const ITERATIONS: u32 = 32;
    let composition =
        MetalVNextComposition::create(DeviceId::new("metal.down-groups.bench").unwrap()).unwrap();
    let runtime = composition.runtime();
    let pipelines = MetalLinearPipelines::new(runtime.device()).unwrap();
    let queue = runtime.device().new_command_queue();
    // Cover long reductions, expanding gate/up projections, and square output
    // projections. The local Q4_K_M uses both Q4K and Q6K matrix parts.
    for (width, outputs) in [(9216, 2560), (2560, 9216), (2560, 2560)] {
        for format in [GgufBlockFormat::Q4K, GgufBlockFormat::Q6K] {
            for rows in [4, 8] {
                let fixture = Fixture::new(runtime, rows, width, outputs, format, ElementType::F16);
                fixture.run(&pipelines, &queue, false, 1);
                let baseline_bits = fixture.validate();
                for round in 0..10 {
                    for grouped in if round % 2 == 0 {
                        [false, true]
                    } else {
                        [true, false]
                    } {
                        let (wall_ns, gpu_ns) =
                            fixture.run(&pipelines, &queue, grouped, ITERATIONS);
                        let bits = fixture.validate();
                        if round >= 2 {
                            println!(
                                "{}",
                                serde_json::json!({
                                    "benchmark": "quantized_f16_four_row_groups", "rows": rows,
                                    "input": width, "output": outputs, "activation_type": "f16", "weight_format": format!("{format:?}"),
                                    "grouped": grouped, "round": round - 2, "projection_iterations": ITERATIONS,
                                    "physical_dispatches_per_iteration": if grouped { rows / 4 } else { 1 },
                                    "command_wall_ns": wall_ns, "command_gpu_ns": gpu_ns,
                                    "wall_ns": wall_ns / f64::from(ITERATIONS), "gpu_ns": gpu_ns.map(|ns| ns / f64::from(ITERATIONS)),
                                    "bitwise_equal_to_baseline": bits == baseline_bits,
                                })
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn q5_f16_gated_delta_four_row_groups_match_cpu_on_metal() {
    let composition =
        MetalVNextComposition::create(DeviceId::new("metal.q5-groups.test").unwrap()).unwrap();
    let runtime = composition.runtime();
    let pipelines = MetalLinearPipelines::new(runtime.device()).unwrap();
    let queue = runtime.device().new_command_queue();
    // Real Q5K input/output projection shapes in the local Q4_K_M model.
    // Guarded offsets, column parts and independent F64 sums are shared with
    // the other format cases; every route must satisfy the same tolerance.
    for (width, outputs) in [(2560, 8192), (4096, 2560)] {
        for rows in [4, 8] {
            let fixture = Fixture::new(
                runtime,
                rows,
                width,
                outputs,
                GgufBlockFormat::Q5K,
                ElementType::F16,
            );
            fixture.run(&pipelines, &queue, true, 1);
            let grouped_bits = fixture.validate();
            fixture.run(&pipelines, &queue, false, 1);
            let baseline_bits = fixture.validate();
            fixture.run_mode(&pipelines, &queue, None, 1);
            assert_eq!(
                fixture.validate(),
                if rows == 8 && outputs < width {
                    grouped_bits
                } else {
                    baseline_bits
                }
            );
        }
    }
}

#[test]
#[ignore = "Q5 F16 projection candidate timing; coordinate exclusive Metal access"]
fn q5_f16_gated_delta_four_row_groups_microbench() {
    const ITERATIONS: u32 = 32;
    let composition =
        MetalVNextComposition::create(DeviceId::new("metal.q5-groups.bench").unwrap()).unwrap();
    let runtime = composition.runtime();
    let pipelines = MetalLinearPipelines::new(runtime.device()).unwrap();
    let queue = runtime.device().new_command_queue();
    for (width, outputs) in [(2560, 8192), (4096, 2560)] {
        for rows in [4, 8] {
            let fixture = Fixture::new(
                runtime,
                rows,
                width,
                outputs,
                GgufBlockFormat::Q5K,
                ElementType::F16,
            );
            fixture.run(&pipelines, &queue, false, 1);
            let baseline_bits = fixture.validate();
            for round in 0..10 {
                for grouped in if round % 2 == 0 {
                    [false, true]
                } else {
                    [true, false]
                } {
                    let (wall_ns, gpu_ns) = fixture.run(&pipelines, &queue, grouped, ITERATIONS);
                    let bits = fixture.validate();
                    if round >= 2 {
                        println!(
                            "{}",
                            serde_json::json!({
                                "benchmark": "q5_f16_gated_delta_four_row_groups", "rows": rows,
                                "input": width, "output": outputs, "activation_type": "f16", "weight_format": "Q5K",
                                "grouped": grouped, "round": round - 2, "projection_iterations": ITERATIONS,
                                "physical_dispatches_per_iteration": if grouped { rows / 4 } else { 1 },
                                "command_wall_ns": wall_ns, "command_gpu_ns": gpu_ns,
                                "wall_ns": wall_ns / f64::from(ITERATIONS), "gpu_ns": gpu_ns.map(|ns| ns / f64::from(ITERATIONS)),
                                "bitwise_equal_to_baseline": bits == baseline_bits,
                            })
                        );
                    }
                }
            }
        }
    }
}
