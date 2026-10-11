//! Test-only M8 scalar versus two existing native B4 shared-weight launches.
//!
//! The production selector and shaders are unchanged. Periodic synthetic block
//! templates keep the independent dense F64 oracle bounded at real FFN shapes;
//! no expanded matrix is uploaded and both routes read the full encoded matrix.

use super::*;
use crate::backend::metal::vnext_ops::MetalVNextComposition;
use ferrum_interfaces::vnext::{BufferRequest, BufferUsage, DeviceId, ResourceId};
use metal::objc::rc::autoreleasepool;

const ROWS: usize = 8;
const GROUP_ROWS: u32 = 4;
const INPUT_PREFIX: usize = 3;
const OUTPUT_PREFIX: usize = 7;
const WEIGHT_PREFIX: usize = 5;
const SUFFIX: usize = 8;
const COLUMN_OFFSET: u32 = 2;
const FORMATS: [GgufBlockFormat; 3] = [
    GgufBlockFormat::Q3K,
    GgufBlockFormat::Iq3S,
    GgufBlockFormat::Iq4Nl,
];

#[derive(Clone, Copy, Debug)]
enum Pattern {
    Dense,
    Cancellation,
}

#[derive(Clone, Copy, Debug)]
enum Route {
    OriginalScalar,
    TwoSharedGroups,
    Production,
}

impl Pattern {
    fn input_variant(self, block: usize) -> usize {
        match self {
            Self::Dense => block % 3,
            Self::Cancellation => block % 2,
        }
    }

    fn template(self, output: usize, block: usize, count: usize) -> usize {
        let block = match self {
            Self::Dense => block,
            // Identical weights multiply opposite activation blocks. The
            // second block has a small residual, exercising cancellation.
            Self::Cancellation => block / 2,
        };
        (output * 7 + output / 3 + block * 5) % count
    }

    fn activation(self, row: usize, variant: usize, lane: usize) -> f16 {
        let base = ((lane * 17 + row * 11) % 67) as f32 / 512.0 - 33.0 / 512.0;
        let value = match self {
            Self::Dense => base + (variant as f32 - 1.0) / 2048.0,
            Self::Cancellation if variant == 1 => {
                -base + if lane == row { 1.0 / 4096.0 } else { 0.0 }
            }
            Self::Cancellation => base,
        };
        f16::from_f32(value)
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
    overwrite(&region, bytes);
    region
}

fn overwrite(region: &MetalBufferRegion, bytes: &[u8]) {
    assert_eq!(region.length_bytes() as usize, bytes.len());
    // SAFETY: retained shared allocation, exact bounds, no in-flight command.
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
}

fn read_bytes(region: &MetalBufferRegion) -> &[u8] {
    // SAFETY: the fixture retains this initialized allocation; every caller
    // waits for its command before reading, and the slice cannot outlive it.
    unsafe {
        std::slice::from_raw_parts(
            region
                .buffer()
                .contents()
                .cast::<u8>()
                .add(region.offset_bytes() as usize),
            region.length_bytes() as usize,
        )
    }
}

fn half_bytes(values: &[f16]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

struct Fixture {
    format: GgufBlockFormat,
    pattern: Pattern,
    parents: [MetalBufferRegion; 3],
    regions: Vec<MetalBufferRegion>,
    initial: [Vec<u8>; 3],
    reference: Vec<(f64, f64)>,
    launch: LinearLaunch,
    groups: [LinearLaunch; 2],
}

impl Fixture {
    fn new(
        runtime: &MetalDeviceRuntime,
        format: GgufBlockFormat,
        width: u32,
        outputs: u32,
        pattern: Pattern,
    ) -> Self {
        let block_values = format.block_values();
        assert_eq!(width as usize % block_values, 0);
        let block_count = width as usize / block_values;
        let mut templates = oracle_blocks(format);
        let scale_offset = if format == GgufBlockFormat::Q3K {
            108
        } else {
            0
        };
        for (index, block) in templates.chunks_exact_mut(format.block_bytes()).enumerate() {
            let scale = if index % 2 == 0 {
                1.0 / 256.0
            } else {
                -1.0 / 512.0
            };
            block[scale_offset..scale_offset + 2]
                .copy_from_slice(&f16::from_f32(scale).to_le_bytes());
        }
        let template_count = templates.len() / format.block_bytes();
        let guard = f16::from_f32(-123.0);
        let mut input = vec![guard; INPUT_PREFIX + ROWS * width as usize + SUFFIX];
        for row in 0..ROWS {
            for block in 0..block_count {
                for lane in 0..block_values {
                    input[INPUT_PREFIX + row * width as usize + block * block_values + lane] =
                        pattern.activation(row, pattern.input_variant(block), lane);
                }
            }
        }
        // Decode each unique block with the CPU implementation. F64 dot
        // products are reused only for byte-identical block/input templates;
        // all blocks and output columns still contribute to the full oracle.
        let mut decoded = vec![0.0_f32; template_count * block_values];
        format.decode(&templates, &mut decoded).unwrap();
        let mut dots = vec![(0.0_f64, 0.0_f64); ROWS * 3 * template_count];
        for row in 0..ROWS {
            for variant in 0..3 {
                for template in 0..template_count {
                    let entry = &mut dots[(row * 3 + variant) * template_count + template];
                    for lane in 0..block_values {
                        let product = f64::from(pattern.activation(row, variant, lane).to_f32())
                            * f64::from(decoded[template * block_values + lane]);
                        entry.0 += product;
                        entry.1 += product.abs();
                    }
                }
            }
        }
        let mut weights = vec![0xcc; WEIGHT_PREFIX];
        let mut reference = vec![(0.0, 0.0); ROWS * outputs as usize];
        for output in 0..outputs as usize {
            for block in 0..block_count {
                let template = pattern.template(output, block, template_count);
                let offset = template * format.block_bytes();
                weights.extend_from_slice(&templates[offset..offset + format.block_bytes()]);
                for row in 0..ROWS {
                    let dot =
                        dots[(row * 3 + pattern.input_variant(block)) * template_count + template];
                    let total = &mut reference[row * outputs as usize + output];
                    total.0 += dot.0;
                    total.1 += dot.1;
                }
            }
        }
        weights.extend([0xcc; SUFFIX]);
        let stride = outputs + 5;
        let mut output = vec![guard; OUTPUT_PREFIX + ROWS * stride as usize + SUFFIX];
        for row in 0..ROWS {
            let start = OUTPUT_PREFIX + row * stride as usize + COLUMN_OFFSET as usize;
            output[start..start + outputs as usize].fill(f16::NAN);
        }
        let initial = [half_bytes(&input), weights, half_bytes(&output)];
        let parents = [
            retained(
                runtime,
                "native-groups.input",
                &initial[0],
                ElementType::F16,
            ),
            retained(
                runtime,
                "native-groups.weight",
                &initial[1],
                ElementType::U8,
            ),
            retained(
                runtime,
                "native-groups.output",
                &initial[2],
                ElementType::F16,
            ),
        ];
        // Retained-region offsets and launch-local offsets both participate.
        let regions = vec![
            parents[0]
                .test_subregion(2..(initial[0].len() - SUFFIX * 2) as u64)
                .unwrap(),
            parents[1]
                .test_subregion(WEIGHT_PREFIX as u64..(initial[1].len() - SUFFIX) as u64)
                .unwrap(),
            parents[2]
                .test_subregion(6..(initial[2].len() - SUFFIX * 2) as u64)
                .unwrap(),
        ];
        let launch = linear_launch(
            PreparedLinearPart {
                region: 1,
                transform: None,
                format: LinearPhysicalFormat::Native(format),
                output_offset: COLUMN_OFFSET,
                out_features: outputs,
            },
            0,
            2,
            ROWS as u64,
            u64::from(width),
            u64::from(stride),
            ((INPUT_PREFIX - 1) * 2) as u64,
            ((OUTPUT_PREFIX - 3) * 2) as u64,
        )
        .unwrap();
        let mut groups = [launch; 2];
        for (index, group) in groups.iter_mut().enumerate() {
            group.params.rows = GROUP_ROWS;
            group.input_offset_bytes += index as u64 * u64::from(GROUP_ROWS) * u64::from(width) * 2;
            group.output_offset_bytes +=
                index as u64 * u64::from(GROUP_ROWS) * u64::from(stride) * 2;
            // This is a test-only launch pair, not a new production plan.
            group.plain_plan = PlainLinearPlan::Single;
        }
        validate_launch_regions(&regions, &[launch]).unwrap();
        validate_launch_regions(&regions, &groups).unwrap();
        Self {
            format,
            pattern,
            parents,
            regions,
            initial,
            reference,
            launch,
            groups,
        }
    }

    fn run(
        &self,
        pipelines: &MetalLinearPipelines,
        queue: &CommandQueueRef,
        route: Route,
        iterations: u32,
    ) -> (f64, Option<f64>) {
        assert!(iterations > 0);
        overwrite(&self.parents[2], &self.initial[2]);
        let started = Instant::now();
        let command = queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        for _ in 0..iterations {
            if matches!(route, Route::Production) {
                dispatch_linear(pipelines, encoder, &self.regions, self.launch);
            } else {
                // Keep the original scalar control independent of future
                // production selector/plan changes. Both PSOs already exist.
                let (pipeline, launches, kind) = match route {
                    Route::OriginalScalar => (
                        pipelines.native.linear_f16(self.format),
                        std::slice::from_ref(&self.launch),
                        LinearDispatchKind::CooperativeGemv,
                    ),
                    Route::TwoSharedGroups => (
                        pipelines
                            .native
                            .shared_linear(self.format, GROUP_ROWS, ElementType::F16)
                            .unwrap(),
                        self.groups.as_slice(),
                        LinearDispatchKind::SharedWeightGemv,
                    ),
                    Route::Production => unreachable!(),
                };
                for launch in launches {
                    encoder.set_compute_pipeline_state(pipeline);
                    set_region_offset(
                        encoder,
                        0,
                        &self.regions[launch.input_region],
                        launch.input_offset_bytes,
                    );
                    set_region_offset(encoder, 1, &self.regions[launch.weight_region], 0);
                    set_region_offset(
                        encoder,
                        2,
                        &self.regions[launch.output_region],
                        launch.output_offset_bytes,
                    );
                    bind_linear_params(
                        encoder,
                        launch.params,
                        launch.format,
                        launch.activation_type,
                    );
                    dispatch_linear_grid(encoder, launch.params, kind);
                }
            }
        }
        encoder.end_encoding();
        command.commit();
        command.wait_until_completed();
        let wall_ns = started.elapsed().as_secs_f64() * 1e9;
        assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
        (wall_ns, gpu_elapsed_ns(command))
    }

    fn validate_inputs(&self) {
        for index in 0..2 {
            assert_eq!(
                read_bytes(&self.parents[index]),
                self.initial[index],
                "input or encoded weights mutated"
            );
        }
    }

    fn validate_output(&self) -> Vec<u16> {
        let params = self.launch.params;
        read_bytes(&self.parents[2])
            .chunks_exact(2)
            .enumerate()
            .map(|(index, bytes)| {
                let bits = u16::from_le_bytes([bytes[0], bytes[1]]);
                let location = index
                    .checked_sub(OUTPUT_PREFIX)
                    .filter(|i| *i < ROWS * params.output_stride as usize)
                    .filter(|i| {
                        (COLUMN_OFFSET as usize
                            ..COLUMN_OFFSET as usize + params.out_features as usize)
                            .contains(&(i % params.output_stride as usize))
                    });
                if let Some(location) = location {
                    let reference_index = location / params.output_stride as usize
                        * params.out_features as usize
                        + location % params.output_stride as usize
                        - COLUMN_OFFSET as usize;
                    let (expected, sum_abs) = self.reference[reference_index];
                    let actual = f64::from(f16::from_bits(bits).to_f32());
                    // Lane sums and SIMD reduction use F32; output is rounded once
                    // to F16. The absolute floor includes a half subnormal ULP.
                    let steps = f64::from(params.in_features) / 32.0 + 8.0;
                    let gamma =
                        steps * f64::from(f32::EPSILON) / (1.0 - steps * f64::from(f32::EPSILON));
                    let bound = gamma * sum_abs + expected.abs() / 1024.0 + 1e-6;
                    assert!(
                        actual.is_finite() && (actual - expected).abs() <= bound,
                        "{:?} {:?} K{} N{} index{index}: {actual} != {expected}, bound={bound}",
                        self.format,
                        self.pattern,
                        params.in_features,
                        params.out_features
                    );
                } else {
                    assert_eq!(
                        &self.initial[2][index * 2..index * 2 + 2],
                        bytes,
                        "output guard {index}"
                    );
                }
                bits
            })
            .collect()
    }

    fn reject_truncated_regions(&self) {
        for index in [0, 2] {
            let mut regions = self.regions.clone();
            let len = regions[index].length_bytes();
            regions[index] = regions[index].test_subregion(0..len - 2).unwrap();
            assert!(validate_launch_regions(&regions, &[self.launch]).is_err());
            assert!(validate_launch_regions(&regions, &self.groups).is_err());
        }
    }
}

#[test]
fn native_eight_rows_four_row_groups_match_f64_on_metal() {
    autoreleasepool(|| {
        let composition =
            MetalVNextComposition::create(DeviceId::new("metal.native-groups.test").unwrap())
                .unwrap();
        let runtime = composition.runtime();
        let pipelines = MetalLinearPipelines::new(runtime.device()).unwrap();
        let queue = runtime.device().new_command_queue();
        for format in FORMATS {
            for (width, outputs) in [
                (format.block_values() as u32, 7),
                (768, 1023),
                (1024, 1024),
                (1280, 1025),
                (5120, 17),
                (17408, 7),
                (5120, 17408),
                (17408, 5120),
            ]
            .into_iter()
            // The generic rule also reaches a native IQ4_NL QKV projection.
            .chain((format == GgufBlockFormat::Iq4Nl).then_some((5120, 10240)))
            {
                for pattern in [Pattern::Dense, Pattern::Cancellation] {
                    autoreleasepool(|| {
                        let fixture = Fixture::new(runtime, format, width, outputs, pattern);
                        fixture.reject_truncated_regions();
                        fixture.run(&pipelines, &queue, Route::OriginalScalar, 1);
                        fixture.validate_inputs();
                        let baseline = fixture.validate_output();
                        for route in [
                            Route::OriginalScalar,
                            Route::TwoSharedGroups,
                            Route::Production,
                        ] {
                            for iterations in [1, 2] {
                                fixture.run(&pipelines, &queue, route, iterations);
                                assert_eq!(
                                    fixture.validate_output(),
                                    baseline,
                                    "{route:?} or repetition changed F16 bits"
                                );
                            }
                        }
                        fixture.validate_inputs();
                    });
                }
            }
        }
    });
}

#[test]
#[ignore = "paired native M8 versus two B4 GPU timing; coordinate exclusive Metal access"]
fn native_eight_rows_four_row_groups_microbench() {
    const ITERATIONS: u32 = 4;
    autoreleasepool(|| {
        let composition =
            MetalVNextComposition::create(DeviceId::new("metal.native-groups.bench").unwrap())
                .unwrap();
        let runtime = composition.runtime();
        let pipelines = MetalLinearPipelines::new(runtime.device()).unwrap();
        let queue = runtime.device().new_command_queue();
        for format in FORMATS {
            // Small K, narrow N and both sides of likely shape boundaries are
            // measured, without presuming grouping benefits every projection.
            for (width, outputs) in [
                (format.block_values() as u32, 7),
                (768, 64),
                (1024, 1023),
                (1024, 1024),
                (1024, 1025),
                (5120, 17408),
                (17408, 5120),
            ]
            .into_iter()
            .chain((format == GgufBlockFormat::Iq4Nl).then_some((5120, 10240)))
            {
                autoreleasepool(|| {
                    let fixture = Fixture::new(runtime, format, width, outputs, Pattern::Dense);
                    fixture.run(&pipelines, &queue, Route::OriginalScalar, 1);
                    fixture.validate_inputs();
                    let baseline = fixture.validate_output();
                    for round in 0..6 {
                        for grouped in if round % 2 == 0 {
                            [false, true]
                        } else {
                            [true, false]
                        } {
                            let (wall_ns, gpu_ns) = fixture.run(
                                &pipelines,
                                &queue,
                                if grouped {
                                    Route::TwoSharedGroups
                                } else {
                                    Route::OriginalScalar
                                },
                                ITERATIONS,
                            );
                            assert_eq!(
                                fixture.validate_output(),
                                baseline,
                                "timed route changed F16 bits"
                            );
                            if round >= 2 {
                                println!(
                                    "{}",
                                    serde_json::json!({
                                        "benchmark":"native_eight_rows_four_row_groups", "format":format!("{format:?}"),
                                        "rows":ROWS, "input":width, "output":outputs, "activation_type":"f16",
                                        "synthetic_weights":true, "input_pattern":"dense_periodic", "grouped":grouped,
                                        "round":round-2, "pair_order":if round % 2 == 0 { "baseline_then_candidate" } else { "candidate_then_baseline" },
                                        "projection_iterations":ITERATIONS, "physical_dispatches_per_iteration":if grouped { 2 } else { 1 },
                                        "baseline_route":"native.linear_f16/CooperativeGemv", "candidate_route":"native.shared_linear(B4) twice",
                                        "command_wall_ns":wall_ns, "command_gpu_ns":gpu_ns,
                                        "wall_ns":wall_ns/f64::from(ITERATIONS), "gpu_ns":gpu_ns.map(|ns| ns/f64::from(ITERATIONS)),
                                        "timing_scope":"whole_command_buffer", "bitwise_equal_to_baseline":true,
                                    })
                                );
                            }
                        }
                    }
                    fixture.validate_inputs();
                });
            }
        }
    });
}
