//! Independent numerical/region checks and an opt-in paired tail microbenchmark.

use super::*;
use crate::backend::metal::vnext_ops::MetalVNextComposition;
use ferrum_interfaces::vnext::{BufferRequest, BufferUsage, DeviceId, ResourceId};
use half::f16;
use metal::{CommandQueueRef, MTLCommandBufferStatus};

const REGION_PREFIX: usize = 8;
const LOCAL_PREFIX: usize = 8;
const GUARD: f32 = -123.0;

fn region<T: Copy>(
    runtime: &MetalDeviceRuntime,
    name: &str,
    values: &[T],
    dtype: ElementType,
) -> MetalBufferRegion {
    let region = runtime
        .allocate_test_region(
            &BufferRequest::new(
                ResourceId::new(name).unwrap(),
                std::mem::size_of_val(values) as u64,
                64,
                BufferUsage::Transfer,
                dtype,
            )
            .unwrap(),
        )
        .unwrap();
    overwrite(&region, values);
    region
}

fn overwrite<T: Copy>(region: &MetalBufferRegion, values: &[T]) {
    assert_eq!(
        region.length_bytes() as usize,
        std::mem::size_of_val(values)
    );
    // SAFETY: the fixture owns the exact retained shared allocation. Every
    // previous submission is complete before reset or readback.
    unsafe {
        std::ptr::copy_nonoverlapping(
            values.as_ptr().cast::<u8>(),
            region
                .buffer()
                .contents()
                .cast::<u8>()
                .add(region.offset_bytes() as usize),
            std::mem::size_of_val(values),
        );
    }
}

fn read<T: Copy>(region: &MetalBufferRegion) -> Vec<T> {
    // SAFETY: callers use the allocated scalar type after command completion.
    unsafe {
        std::slice::from_raw_parts(
            region
                .buffer()
                .contents()
                .cast::<u8>()
                .add(region.offset_bytes() as usize)
                .cast::<T>(),
            region.length_bytes() as usize / std::mem::size_of::<T>(),
        )
        .to_vec()
    }
}

fn launch(rows: u64, width: u64, outputs: u32, stride: u64) -> LinearLaunch {
    linear_launch(
        PreparedLinearPart {
            region: 1,
            transform: None,
            format: LinearPhysicalFormat::Q4K,
            output_offset: 3,
            out_features: outputs,
        },
        0,
        2,
        rows,
        width,
        stride,
        (LOCAL_PREFIX * 2) as u64,
        (LOCAL_PREFIX * 2) as u64,
    )
    .unwrap()
}

#[test]
fn q4k_plain_tail_plan_preserves_shape_abi_and_staged_dispatch_count() {
    for rows in [1, 7, 8, 31, 32, 33, 63, 64, 65, 66, 67, 68, 95, 96, 97, 257] {
        for outputs in [1023, 1024, 1025] {
            let launch = launch(rows, 512, outputs, u64::from(outputs) + 12);
            let split = rows > 32 && rows % 32 == 1 && outputs >= 1024;
            assert_eq!(launch.dispatch_count(), if split { 2 } else { 1 });
            if let Some([head, tail]) = launch.plain_plan.parts(launch) {
                assert_eq!(head.params.rows + tail.params.rows, rows as u32);
                assert_eq!(tail.params.rows, 1);
                assert_eq!(tail.params.output_column_offset, 3);
                assert_eq!(tail.params.output_stride, launch.params.output_stride);
                assert_eq!(tail.weight_region, launch.weight_region);
                assert_eq!(
                    tail.input_offset_bytes,
                    launch.input_offset_bytes + (rows - 1) * 512 * 2
                );
                assert_eq!(
                    tail.output_offset_bytes,
                    launch.output_offset_bytes
                        + (rows - 1) * u64::from(launch.params.output_stride) * 2
                );
            }
        }
    }
    let original = launch(65, 512, 2048, 2060);
    for format in [
        LinearPhysicalFormat::Q5K,
        LinearPhysicalFormat::Q6K,
        LinearPhysicalFormat::Q8_0,
        LinearPhysicalFormat::DenseF16,
    ] {
        assert!(matches!(
            PlainLinearPlan::for_launch(LinearLaunch { format, ..original }),
            PlainLinearPlan::Single
        ));
    }
    assert!(matches!(
        PlainLinearPlan::for_launch(LinearLaunch {
            activation_type: ElementType::F32,
            ..original
        }),
        PlainLinearPlan::Single
    ));
    assert!(matches!(
        PlainLinearPlan::for_launch(LinearLaunch {
            transform: Some(HadamardTransform {
                block_size: 128,
                signs_region: None,
                inverse: false,
                permutation: None
            }),
            ..original
        }),
        PlainLinearPlan::Single
    ));
    assert!(matches!(
        PlainLinearPlan::for_launch(LinearLaunch {
            input_offset_bytes: u64::MAX,
            ..original
        }),
        PlainLinearPlan::Single
    ));
}

#[test]
fn q6_f32_eight_row_plan_preserves_parts_strides_and_partial_group_fallbacks() {
    for rows in [1, 2, 3, 4, 5, 7, 8, 9, 12, 16, 33, 65] {
        for outputs in [1023, 1024, 1025] {
            let launch = linear_launch_typed(
                PreparedLinearPart {
                    region: 7,
                    transform: None,
                    format: LinearPhysicalFormat::Q6K,
                    output_offset: 3,
                    out_features: outputs,
                },
                2,
                5,
                rows,
                768,
                u64::from(outputs) + 12,
                32,
                48,
                ElementType::F32,
            )
            .unwrap();
            let selected = rows == 8 && outputs >= 1024;
            assert_eq!(launch.dispatch_count(), if selected { 2 } else { 1 });
            assert_eq!(launch.plain_plan.parts(launch).is_some(), selected);
            if let Some([first, second]) = launch.plain_plan.parts(launch) {
                for part in [first, second] {
                    assert_eq!(part.params.rows, 4);
                    assert_eq!(part.params.output_stride, outputs + 12);
                    assert_eq!(part.params.output_column_offset, 3);
                    assert_eq!(part.weight_region, 7);
                    assert_eq!(part.input_region, 2);
                    assert_eq!(part.output_region, 5);
                    assert_eq!(part.dispatch_count(), 1);
                }
                assert_eq!(first.input_offset_bytes, 32);
                assert_eq!(first.output_offset_bytes, 48);
                assert_eq!(second.input_offset_bytes, 32 + 4 * 768 * 4);
                assert_eq!(
                    second.output_offset_bytes,
                    48 + 4 * u64::from(outputs + 12) * 4
                );
            }
        }
    }
}

#[test]
fn quantized_f16_eight_row_plan_keeps_short_narrow_and_partial_group_fallbacks() {
    for format in [
        LinearPhysicalFormat::Q4K,
        LinearPhysicalFormat::Q5K,
        LinearPhysicalFormat::Q6K,
    ] {
        for rows in [7, 8, 9] {
            for width in [768, 1024, 2560] {
                for outputs in [1023, 1024, 1025] {
                    let mut launch = LinearLaunch {
                        format,
                        ..launch(rows, width, outputs, u64::from(outputs) + 12)
                    };
                    launch.plain_plan = PlainLinearPlan::for_launch(launch);
                    let selected = rows == 8
                        && width >= 1024
                        && outputs >= 1024
                        && (format != LinearPhysicalFormat::Q5K || u64::from(outputs) < width);
                    assert_eq!(launch.dispatch_count(), if selected { 2 } else { 1 });
                    if let Some([first, second]) = launch.plain_plan.parts(launch) {
                        assert_eq!((first.params.rows, second.params.rows), (4, 4));
                        assert_eq!(
                            second.input_offset_bytes,
                            launch.input_offset_bytes + 4 * width * 2
                        );
                        assert_eq!(
                            second.output_offset_bytes,
                            launch.output_offset_bytes + 4 * u64::from(outputs + 12) * 2
                        );
                        assert_eq!(second.params.output_column_offset, 3);
                    }
                }
            }
        }
    }
}

struct Fixture {
    regions: Vec<MetalBufferRegion>,
    parents: [MetalBufferRegion; 3],
    input: Vec<f16>,
    weights: Vec<u8>,
    initial_output: Vec<f16>,
    expected: Vec<f64>,
    launch: LinearLaunch,
}

impl Fixture {
    fn new(runtime: &MetalDeviceRuntime, rows: u32, width: u32, outputs: u32) -> Self {
        let shape = microbench::Shape {
            name: "q4k_tail",
            input: width,
            output: outputs,
            format: GgufBlockFormat::Q4K,
        };
        let encoded = microbench::weights(shape);
        let prefix = REGION_PREFIX + LOCAL_PREFIX;
        let stride = outputs as usize + 12;
        let mut input = vec![f16::from_f32(GUARD); prefix + rows as usize * width as usize + 8];
        input[prefix..prefix + rows as usize * width as usize].fill(f16::ZERO);
        let mut sparse = Vec::new();
        for row in 0..rows as usize {
            let mut entries = Vec::new();
            // Distinct rows span every block. Sparse inputs keep the full
            // independent CPU oracle cheap; both GPU routes execute dense math.
            for index in 0..16 {
                let column = (index * (width as usize / 16) + row * 13 + 7) % width as usize;
                let value = f16::from_f32(((index + row * 3) as f32 * 0.7).sin() * 0.125);
                input[prefix + row * width as usize + column] = value;
                entries.push((column, f64::from(value.to_f32())));
            }
            sparse.push(entries);
        }
        let format = GgufBlockFormat::Q4K;
        let row_bytes = width as usize / format.block_values() * format.block_bytes();
        let mut expected = Vec::new();
        for entries in &sparse {
            for output in 0..outputs as usize {
                expected.push(
                    entries
                        .iter()
                        .map(|&(column, value)| {
                            let offset = output * row_bytes
                                + column / format.block_values() * format.block_bytes();
                            value
                                * f64::from(format.decode_value(
                                    &encoded[offset..offset + format.block_bytes()],
                                    column % format.block_values(),
                                ))
                        })
                        .sum::<f64>(),
                );
            }
        }
        let mut weights = vec![0xcc; REGION_PREFIX * 2];
        weights.extend_from_slice(&encoded);
        weights.extend([0xcc; 16]);
        let mut initial_output = vec![f16::from_f32(GUARD); prefix + rows as usize * stride + 8];
        for row in 0..rows as usize {
            initial_output[prefix + row * stride + 3..prefix + row * stride + 3 + outputs as usize]
                .fill(f16::NAN);
        }
        let parents = [
            region(runtime, "tail.input", &input, ElementType::F16),
            region(runtime, "tail.weights", &weights, ElementType::U8),
            region(runtime, "tail.output", &initial_output, ElementType::F16),
        ];
        let regions = vec![
            parents[0]
                .test_subregion((REGION_PREFIX * 2) as u64..(input.len() * 2 - 16) as u64)
                .unwrap(),
            parents[1]
                .test_subregion((REGION_PREFIX * 2) as u64..(weights.len() - 16) as u64)
                .unwrap(),
            parents[2]
                .test_subregion((REGION_PREFIX * 2) as u64..(initial_output.len() * 2 - 16) as u64)
                .unwrap(),
        ];
        let launch = launch(u64::from(rows), u64::from(width), outputs, stride as u64);
        validate_launch_regions(&regions, &[launch]).unwrap();
        Self {
            regions,
            parents,
            input,
            weights,
            initial_output,
            expected,
            launch,
        }
    }

    fn with_dense_cancellation_input(mut self) -> Self {
        let width = self.launch.params.in_features as usize;
        assert_eq!(width, 512);
        let rows = self.launch.params.rows as usize;
        let outputs = self.launch.params.out_features as usize;
        let weight_prefix = REGION_PREFIX * 2;
        // Equal source blocks make opposite activation halves cancel across
        // the K reduction. Both blocks remain ordinary, independently decoded
        // Q4K bytes; neither GPU result supplies the reference.
        for output in 0..outputs {
            let first = weight_prefix + output * 288;
            self.weights.copy_within(first..first + 144, first + 144);
        }
        let mut decoded = vec![0.0_f32; outputs * width];
        GgufBlockFormat::Q4K
            .decode(
                &self.weights[weight_prefix..self.weights.len() - 16],
                &mut decoded,
            )
            .unwrap();
        let input_prefix = REGION_PREFIX + LOCAL_PREFIX;
        for row in 0..rows {
            for column in 0..256 {
                // All entries are nonzero normal F16 values; magnitudes span
                // 2^-13 through 2^-2. A one-ULP-sized relative difference leaves
                // a small residual instead of an identically zero answer.
                let exponent = -2 - ((column + row) % 12) as i32;
                let sign = if (column + row / 3) % 2 == 0 {
                    1.0
                } else {
                    -1.0
                };
                let value = sign * 2_f32.powi(exponent);
                self.input[input_prefix + row * width + column] = f16::from_f32(value);
                self.input[input_prefix + row * width + column + 256] =
                    f16::from_f32(-value * (1.0 - 1.0 / 1024.0));
            }
        }
        self.expected.clear();
        for row in 0..rows {
            let input = &self.input[input_prefix + row * width..][..width];
            for output in 0..outputs {
                self.expected.push(
                    input
                        .iter()
                        .zip(&decoded[output * width..][..width])
                        .map(|(x, w)| f64::from(x.to_f32()) * f64::from(*w))
                        .sum::<f64>(),
                );
            }
        }
        overwrite(&self.parents[0], &self.input);
        overwrite(&self.parents[1], &self.weights);
        self
    }

    fn run(
        &self,
        pipelines: &MetalLinearPipelines,
        queue: &CommandQueueRef,
        split: bool,
    ) -> (f64, Option<f64>) {
        self.run_repeated(pipelines, queue, split, 1)
    }

    fn run_repeated(
        &self,
        pipelines: &MetalLinearPipelines,
        queue: &CommandQueueRef,
        split: bool,
        iterations: u32,
    ) -> (f64, Option<f64>) {
        assert!(iterations > 0);
        overwrite(&self.parents[2], &self.initial_output);
        let start = std::time::Instant::now();
        let command = queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        for _ in 0..iterations {
            if split {
                dispatch_linear(pipelines, encoder, &self.regions, self.launch);
            } else {
                dispatch_single_plain_linear(pipelines, encoder, &self.regions, self.launch);
            }
        }
        encoder.end_encoding();
        command.commit();
        command.wait_until_completed();
        let wall_ns = start.elapsed().as_secs_f64() * 1e9;
        assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
        (wall_ns, microbench::gpu_elapsed_ns(command))
    }

    fn validate(&self) -> Vec<u16> {
        assert_eq!(read::<f16>(&self.parents[0]), self.input, "input mutated");
        assert_eq!(read::<u8>(&self.parents[1]), self.weights, "weight mutated");
        let actual = read::<f16>(&self.parents[2]);
        let prefix = REGION_PREFIX + LOCAL_PREFIX;
        let stride = self.launch.params.output_stride as usize;
        let outputs = self.launch.params.out_features as usize;
        for (index, value) in actual.iter().enumerate() {
            let location = index
                .checked_sub(prefix)
                .filter(|index| *index < self.launch.params.rows as usize * stride)
                .filter(|index| (3..3 + outputs).contains(&(index % stride)));
            if let Some(index) = location {
                let expected = self.expected[index / stride * outputs + index % stride - 3];
                // Same F16 projection tolerance used by the existing Q4K
                // GEMM/GEMV conformance. The oracle decodes source blocks on
                // CPU and sums in F64; it does not call either GPU algorithm.
                assert!(
                    value.is_finite()
                        && (f64::from(value.to_f32()) - expected).abs()
                            <= 0.002 + 0.003 * expected.abs(),
                    "rows={} ({},{}): {} != {expected}",
                    self.launch.params.rows,
                    index / stride,
                    index % stride - 3,
                    value.to_f32()
                );
            } else {
                assert_eq!(*value, f16::from_f32(GUARD), "guard changed at {index}");
            }
        }
        actual.iter().map(|value| value.to_bits()).collect()
    }
}

#[test]
fn q4k_plain_tail_matches_cpu_with_regions_strides_and_repeated_execution_on_metal() {
    let composition =
        MetalVNextComposition::create(DeviceId::new("metal.q4k-tail.test").unwrap()).unwrap();
    let runtime = composition.runtime();
    let pipelines = MetalLinearPipelines::new(runtime.device()).unwrap();
    let queue = runtime.device().new_command_queue();
    for (rows, width, outputs) in [
        (32, 256, 1025),
        (33, 256, 1025),
        (64, 512, 1025),
        (65, 512, 1025),
        (66, 256, 1025),
        (97, 256, 1023),
        (97, 256, 1024),
    ] {
        let fixture = Fixture::new(runtime, rows, width, outputs);
        fixture.run(&pipelines, &queue, false);
        fixture.validate();
        fixture.run(&pipelines, &queue, true);
        let first = fixture.validate();
        fixture.run(&pipelines, &queue, true);
        assert_eq!(
            fixture.validate(),
            first,
            "same prepared eager plan changed bits"
        );
        // The parent has guards beyond this view. They cannot authorize a
        // launch whose full logical output extends outside the retained span.
        let mut truncated = fixture.regions.clone();
        let bytes = truncated[2].length_bytes();
        truncated[2] = truncated[2].test_subregion(0..bytes - 2).unwrap();
        assert!(validate_launch_regions(&truncated, &[fixture.launch]).is_err());
    }
}

#[test]
fn q4k_plain_tail_dense_cancellation_matches_f64_on_metal() {
    let composition =
        MetalVNextComposition::create(DeviceId::new("metal.q4k-tail.dense").unwrap()).unwrap();
    let runtime = composition.runtime();
    let pipelines = MetalLinearPipelines::new(runtime.device()).unwrap();
    let queue = runtime.device().new_command_queue();
    let fixture = Fixture::new(runtime, 65, 512, 1024).with_dense_cancellation_input();
    assert!(fixture.launch.plain_plan.parts(fixture.launch).is_some());
    fixture.run(&pipelines, &queue, false);
    fixture.validate();
    fixture.run(&pipelines, &queue, true);
    let first = fixture.validate();
    fixture.run(&pipelines, &queue, true);
    assert_eq!(
        fixture.validate(),
        first,
        "dense eager execution changed bits"
    );
}

#[test]
fn q4k_plain_tail_staging_keeps_two_physical_dispatches_on_metal() {
    let composition =
        MetalVNextComposition::create(DeviceId::new("metal.q4k-tail.staged").unwrap()).unwrap();
    let runtime = composition.runtime();
    let pipelines = MetalLinearPipelines::new(runtime.device()).unwrap();
    let queue = runtime.device().new_command_queue();
    let mut fixture = Fixture::new(runtime, 257, 512, 2048);
    let scratch = vec![0_u8; 512 * 2048 * 2];
    fixture
        .regions
        .push(region(runtime, "tail.staging", &scratch, ElementType::U8));
    let workspace = staged_prefill::Workspace::new(
        &fixture.regions,
        3,
        0,
        scratch.len() as u64,
        [fixture.launch],
    )
    .unwrap();
    assert!(workspace.is_some());
    assert_eq!(fixture.launch.dispatch_count(), 2);
    assert_eq!(staged_prefill::dispatch_count(fixture.launch, workspace), 2);
    let command = queue.new_command_buffer();
    let encoder = command.new_compute_command_encoder();
    staged_prefill::dispatch(
        &pipelines,
        encoder,
        &fixture.regions,
        fixture.launch,
        workspace,
    );
    encoder.end_encoding();
    command.commit();
    command.wait_until_completed();
    assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
    fixture.validate();
}

#[test]
#[ignore = "paired Q4K GPU timing; coordinate exclusive Metal access"]
fn q4k_plain_tail_dispatch_microbench() {
    const PROJECTION_ITERATIONS: u32 = 32;
    let composition =
        MetalVNextComposition::create(DeviceId::new("metal.q4k-tail.bench").unwrap()).unwrap();
    let runtime = composition.runtime();
    let pipelines = MetalLinearPipelines::new(runtime.device()).unwrap();
    let queue = runtime.device().new_command_queue();
    // Real Qwen3.5-4B projection dimensions, synthetic source blocks. This is
    // kernel evidence only; no timing threshold gates numerical correctness.
    for outputs in [2560, 9216] {
        for rows in [64, 65, 66, 97] {
            let fixture = Fixture::new(runtime, rows, 2560, outputs);
            for round in 0..10 {
                for split in if round % 2 == 0 {
                    [false, true]
                } else {
                    [true, false]
                } {
                    // Keep the GPU busy across many projections before CPU
                    // readback/validation. One iteration is a complete logical
                    // projection (one baseline or two split dispatches).
                    let (command_wall_ns, command_gpu_ns) =
                        fixture.run_repeated(&pipelines, &queue, split, PROJECTION_ITERATIONS);
                    fixture.validate();
                    if round >= 2 {
                        let physical_dispatches = if split {
                            fixture.launch.dispatch_count()
                        } else {
                            1
                        };
                        println!(
                            "{}",
                            serde_json::json!({
                                "benchmark": "q4k_plain_tail", "rows": rows, "input": 2560, "output": outputs,
                                "round": round - 2, "split": split,
                                "wall_ns": command_wall_ns / f64::from(PROJECTION_ITERATIONS),
                                "gpu_ns": command_gpu_ns.map(|ns| ns / f64::from(PROJECTION_ITERATIONS)),
                                "command_wall_ns": command_wall_ns, "command_gpu_ns": command_gpu_ns,
                                "projection_iterations": PROJECTION_ITERATIONS,
                                "physical_dispatches_per_iteration": physical_dispatches,
                                "physical_dispatches": physical_dispatches * u64::from(PROJECTION_ITERATIONS),
                            })
                        );
                    }
                }
            }
        }
    }
}
