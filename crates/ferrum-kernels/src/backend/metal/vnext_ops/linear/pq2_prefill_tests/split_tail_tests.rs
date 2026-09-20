//! Production split-prefill planning, numerical guards, and a paired GPU benchmark.
//! The baseline retains the old unsplit dispatch over the same F32 workspace.

use super::vector_input_tests::{read, region};
use super::*;
use crate::backend::metal::vnext_ops::MetalVNextComposition;
use ferrum_interfaces::vnext::DeviceId;

const WORKSPACE_OFFSET: u64 = 16;

impl Fixture {
    fn with_output_layout(mut self, device: &Device, stride: u32, column_offset: u32) -> Self {
        assert!(column_offset + self.params.out_features <= stride);
        self.params.output_stride = stride;
        self.params.output_column_offset = column_offset;
        self.initial_output = vec![
            f16::from_f32(GUARD);
            OUTPUT_PREFIX + self.params.rows as usize * stride as usize + 8
        ];
        for row in 0..self.params.rows as usize {
            let first = OUTPUT_PREFIX + row * stride as usize + column_offset as usize;
            self.initial_output[first..first + self.params.out_features as usize].fill(f16::NAN);
        }
        self.output = buffer(device, &self.initial_output);
        self
    }
}

fn split_transformed_launches(
    pipelines: &MetalLinearPipelines,
    regions: &[MetalBufferRegion],
    launch: LinearLaunch,
) -> Option<[LinearLaunch; 2]> {
    TransformedLinearPlan::for_launch(pipelines, regions, launch).parts(launch)
}

struct BoundFixture {
    oracle: Fixture,
    workspace_allocation: MetalBufferRegion,
    workspace_values: Vec<f32>,
    weight_allocation: MetalBufferRegion,
    original_input: Vec<f16>,
    regions: Vec<MetalBufferRegion>,
    launch: LinearLaunch,
}

impl BoundFixture {
    fn new(
        runtime: &MetalDeviceRuntime,
        pipelines: &MetalLinearPipelines,
        oracle: Fixture,
        region_offset: u64,
    ) -> Self {
        let count = oracle.params.rows as usize * oracle.params.in_features as usize;
        let mut workspace_values = vec![GUARD; ((region_offset + WORKSPACE_OFFSET) / 4) as usize];
        workspace_values
            .extend_from_slice(&oracle.input_values[INPUT_PREFIX..INPUT_PREFIX + count]);
        workspace_values.extend([GUARD; 8]);
        let workspace_allocation = region(
            runtime,
            "split.workspace",
            &workspace_values,
            ElementType::U8,
        );
        let workspace = workspace_allocation
            .test_subregion(region_offset..workspace_allocation.length_bytes())
            .unwrap();
        let weight_allocation = region(
            runtime,
            "split.weights",
            &oracle.weight_bytes,
            ElementType::U8,
        );
        let weights = weight_allocation
            .test_subregion(WEIGHT_PREFIX as u64..oracle.weight_bytes.len() as u64 - 16)
            .unwrap();
        let original_input = vec![f16::ZERO; count];
        let regions = vec![
            workspace,
            weights,
            region(
                runtime,
                "split.output",
                &oracle.initial_output,
                ElementType::F16,
            ),
            region(
                runtime,
                "split.original-input",
                &original_input,
                ElementType::F16,
            ),
        ];
        let mut launch = LinearLaunch {
            input_region: 3,
            weight_region: 1,
            output_region: 2,
            input_offset_bytes: 0,
            output_offset_bytes: (OUTPUT_PREFIX * 2) as u64,
            activation_type: ElementType::F16,
            format: LinearPhysicalFormat::Native(GgufBlockFormat::Pq2_0),
            params: oracle.params,
            transform: Some(HadamardTransform {
                block_size: 128,
                signs_region: None,
                inverse: false,
                permutation: None,
            }),
            transform_workspace: None,
            transformed_plan: TransformedLinearPlan::Single,
        };
        launch
            .bind_hadamard_workspace(pipelines, &regions, 0, WORKSPACE_OFFSET)
            .unwrap();
        validate_launch_regions(&regions, &[launch]).unwrap();
        Self {
            oracle,
            workspace_allocation,
            workspace_values,
            weight_allocation,
            original_input,
            regions,
            launch,
        }
    }

    fn run(
        &self,
        pipelines: &MetalLinearPipelines,
        queue: &CommandQueueRef,
        split: bool,
    ) -> (f64, Option<f64>, u64) {
        // SAFETY: every previous call waited for completion. This retained
        // allocation has exactly the initialized F16 output/guard span.
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.oracle.initial_output.as_ptr(),
                self.regions[2]
                    .buffer()
                    .contents()
                    .cast::<u8>()
                    .add(self.regions[2].offset_bytes() as usize)
                    .cast::<f16>(),
                self.oracle.initial_output.len(),
            );
        }
        let started = std::time::Instant::now();
        let command = queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        let dispatches = if split {
            dispatch_transformed_linear(pipelines, encoder, &self.regions, self.launch);
            self.launch.transformed_plan.projection_dispatch_count()
        } else {
            dispatch_single_transformed_linear(pipelines, encoder, &self.regions, self.launch);
            1
        };
        encoder.end_encoding();
        command.commit();
        command.wait_until_completed();
        let wall_ns = started.elapsed().as_secs_f64() * 1e9;
        assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
        (
            wall_ns,
            super::super::microbench::gpu_elapsed_ns(command),
            dispatches,
        )
    }

    fn validate(&self) -> Vec<u16> {
        assert_eq!(
            read::<f32>(&self.workspace_allocation),
            self.workspace_values
        );
        assert_eq!(
            read::<u8>(&self.weight_allocation),
            self.oracle.weight_bytes
        );
        assert_eq!(read::<f16>(&self.regions[3]), self.original_input);
        self.oracle
            .validate_output_values(&read::<f16>(&self.regions[2]))
    }
}

#[test]
fn pq2_split_tail_preserves_f32_math_boundaries_and_fallbacks_on_metal() {
    let composition =
        MetalVNextComposition::create(DeviceId::new("metal.pq2.split-tail").unwrap()).unwrap();
    let runtime = composition.runtime();
    let mut pipelines = MetalLinearPipelines::new(runtime.device()).unwrap();
    let queue = runtime.device().new_command_queue();
    for (rows, width, outputs, wide_scale, region_offset) in [
        (65, 128, 4096, false, 0),
        (96, 128, 4096, false, 0),
        (127, 128, 4096, false, 0),
        (128, 128, 4096, false, 0),
        (129, 128, 4096, false, 16),
        (191, 384, 4096, true, 0),
        (255, 128, 4096, false, 0),
        (513, 128, 4096, false, 0),
        (544, 128, 4096, false, 0),
        (575, 128, 4096, false, 0),
        (597, 128, 4096, false, 0),
        (63, 128, 4096, false, 0),
        (129, 128, 4097, false, 0),
        (129, 128, 4096, false, 4),
        (129, 128, 4032, false, 0),
    ] {
        let mut oracle = Fixture::new(runtime.device(), rows, width, outputs, wide_scale);
        if rows == 129 && region_offset == 16 {
            oracle.use_distinct_weight_rows(runtime.device());
        }
        let fixture = BoundFixture::new(runtime, &pipelines, oracle, region_offset);
        let expected_split = rows > 128
            && rows % 64 != 0
            && outputs >= 4096
            && outputs % 64 == 0
            && region_offset % 16 == 0;
        let parts = fixture.launch.transformed_plan.parts(fixture.launch);
        assert_eq!(parts.is_some(), expected_split);
        assert_eq!(
            fixture.launch.dispatch_count(),
            if expected_split { 3 } else { 2 }
        );
        if let Some([head, tail]) = parts {
            assert_eq!(head.params.rows + tail.params.rows, rows);
            assert_eq!(
                tail.params.output_column_offset,
                fixture.launch.params.output_column_offset
            );
        }
        fixture.run(&pipelines, &queue, false);
        let baseline = fixture.validate();
        let (_, _, dispatches) = fixture.run(&pipelines, &queue, true);
        assert_eq!(dispatches, if expected_split { 2 } else { 1 });
        assert_eq!(
            fixture.validate(),
            baseline,
            "split changed F32 MMA result bits"
        );
        // A large allocation cannot authorize reading outside its retained view.
        let mut short = fixture.regions.clone();
        short[0] = short[0]
            .test_subregion(0..WORKSPACE_OFFSET + u64::from(rows) * u64::from(width) * 4 - 1)
            .unwrap();
        assert!(split_transformed_launches(&pipelines, &short, fixture.launch).is_none());
    }
    let mut fixture = BoundFixture::new(
        runtime,
        &pipelines,
        Fixture::new(runtime.device(), 129, 128, 4096, false),
        0,
    );
    for (format, activation_type, transform) in [
        (
            fixture.launch.format,
            ElementType::F32,
            fixture.launch.transform,
        ),
        (
            LinearPhysicalFormat::Native(GgufBlockFormat::Iq4Xs),
            ElementType::F16,
            fixture.launch.transform,
        ),
        (fixture.launch.format, ElementType::F16, None),
    ] {
        let mut incompatible = LinearLaunch {
            format,
            activation_type,
            transform,
            ..fixture.launch
        };
        assert!(split_transformed_launches(&pipelines, &fixture.regions, incompatible).is_none());
        if transform.is_none() {
            incompatible
                .bind_hadamard_workspace(&pipelines, &fixture.regions, 0, WORKSPACE_OFFSET)
                .unwrap();
            assert_eq!(incompatible.dispatch_count(), 1);
        }
    }
    let mut incomplete_k = fixture.launch;
    incomplete_k.params.in_features -= 1;
    assert!(split_transformed_launches(&pipelines, &fixture.regions, incomplete_k).is_none());
    let mut absent_workspace = fixture.launch;
    absent_workspace.transform_workspace = None;
    assert!(split_transformed_launches(&pipelines, &fixture.regions, absent_workspace).is_none());
    let mut overflowing = fixture.launch;
    overflowing.output_offset_bytes = u64::MAX;
    assert!(split_transformed_launches(&pipelines, &fixture.regions, overflowing).is_none());
    for missing in 0..3 {
        let saved = match missing {
            0 => pipelines.native.pq2_gemm_input_f32_output_f16_m64.take(),
            1 => pipelines
                .native
                .pq2_gemm_input_f32_output_f16_m64_full_tiles
                .take(),
            _ => pipelines
                .native
                .pq2_gemm_input_f32_output_f16_m64_full_tiles_vector_input
                .take(),
        };
        assert!(split_transformed_launches(&pipelines, &fixture.regions, fixture.launch).is_none());
        // A fresh binding must replace an earlier split decision when any
        // required optional pipeline is absent. Encoding and counts both fall
        // back to the original unsplit projection.
        fixture
            .launch
            .bind_hadamard_workspace(&pipelines, &fixture.regions, 0, WORKSPACE_OFFSET)
            .unwrap();
        assert_eq!(fixture.launch.dispatch_count(), 2);
        fixture.run(&pipelines, &queue, false);
        let baseline = fixture.validate();
        assert_eq!(fixture.run(&pipelines, &queue, true).2, 1);
        assert_eq!(fixture.validate(), baseline);
        match missing {
            0 => pipelines.native.pq2_gemm_input_f32_output_f16_m64 = saved,
            1 => {
                pipelines
                    .native
                    .pq2_gemm_input_f32_output_f16_m64_full_tiles = saved
            }
            _ => {
                pipelines
                    .native
                    .pq2_gemm_input_f32_output_f16_m64_full_tiles_vector_input = saved
            }
        }
        fixture
            .launch
            .bind_hadamard_workspace(&pipelines, &fixture.regions, 0, WORKSPACE_OFFSET)
            .unwrap();
        assert_eq!(fixture.launch.dispatch_count(), 3);
    }
}

#[test]
#[ignore = "GPU timing experiment: coordinate exclusive device access"]
fn pq2_split_tail_paired_gpu_microbench() {
    let composition =
        MetalVNextComposition::create(DeviceId::new("metal.pq2.split-tail.bench").unwrap())
            .unwrap();
    let runtime = composition.runtime();
    let pipelines = MetalLinearPipelines::new(runtime.device()).unwrap();
    let queue = runtime.device().new_command_queue();
    const WARMUPS: usize = 2;
    const REPEATS: usize = 5;
    // Match the production FFN row layouts: a gate slice of packed [gate, up],
    // and a contiguous down projection. Adversarial offsets/padding stay in the
    // correctness test above. Both timings still retain allocation guards.
    for (shape, width, outputs, stride) in [
        ("ffn_gate_or_up", 5120, 17408, 34816),
        ("ffn_down", 17408, 5120, 5120),
    ] {
        // Sweep both prefix size and guarded-tail occupancy to distinguish
        // amortization from merely having any M64 prefix. M597 anchors the
        // previous small-prefix sweep without rerunning that entire cohort.
        for rows in [129, 191, 255, 513, 544, 575, 597] {
            let fixture = BoundFixture::new(
                runtime,
                &pipelines,
                Fixture::new(runtime.device(), rows, width, outputs, false).with_output_layout(
                    runtime.device(),
                    stride,
                    0,
                ),
                0,
            );
            let split = fixture.launch.transformed_plan.parts(fixture.launch);
            let head_rows = split.map(|parts| parts[0].params.rows);
            let tail_rows = split.map(|parts| parts[1].params.rows);
            fixture.run(&pipelines, &queue, false);
            let expected = fixture.validate();
            fixture.run(&pipelines, &queue, true);
            assert_eq!(fixture.validate(), expected);
            for repeat in 0..WARMUPS {
                for candidate in if repeat % 2 == 0 {
                    [false, true]
                } else {
                    [true, false]
                } {
                    fixture.run(&pipelines, &queue, candidate);
                }
            }
            let mut samples = Vec::new();
            for repeat in 0..REPEATS {
                let order = if repeat % 2 == 0 { [0, 1] } else { [1, 0] };
                let mut wall = [0.0; 2];
                let mut gpu = [0.0; 2];
                let mut dispatches = [0; 2];
                for variant in order {
                    let timing = fixture.run(&pipelines, &queue, variant == 1);
                    wall[variant] = timing.0;
                    gpu[variant] = timing
                        .1
                        .expect("completed GPU command must expose timestamps");
                    dispatches[variant] = timing.2;
                }
                samples.push(serde_json::json!({
                    "repeat": repeat, "order": order, "dispatches": dispatches,
                    "production_wall_ns": wall[0], "split_wall_ns": wall[1],
                    "production_gpu_ns": gpu[0], "split_gpu_ns": gpu[1],
                    "split_over_production_wall": wall[1] / wall[0],
                    "split_over_production_gpu": gpu[1] / gpu[0],
                }));
            }
            for split in [false, true] {
                fixture.run(&pipelines, &queue, split);
                assert_eq!(fixture.validate(), expected);
            }
            println!(
                "{}",
                serde_json::json!({
                    "schema_version": 1, "kind": "pq2_split_tail_paired_gpu_microbench",
                    "device": runtime.device().name(), "shape": shape, "rows": rows,
                    "in_features": width, "out_features": outputs,
                    "head_rows": head_rows, "tail_rows": tail_rows,
                    "complete_m64_prefix_tiles": rows / 64,
                    "guarded_m32_tail_tiles": (rows % 64).div_ceil(32),
                    "complete_prefix_row_fraction": f64::from(rows / 64 * 64) / f64::from(rows),
                    "input_offset_bytes": WORKSPACE_OFFSET,
                    "weight_offset_bytes": WEIGHT_PREFIX,
                    "output_offset_bytes": OUTPUT_PREFIX * 2,
                    "output_stride": fixture.launch.params.output_stride,
                    "output_column_offset": fixture.launch.params.output_column_offset,
                    "production_selection_changed": true,
                    "baseline": "previous unsplit post-Hadamard dispatch",
                    "candidate": "bound production plan: vector-input/full-M64 prefix plus guarded M32 tail",
                    "timing": {"warmups_per_variant": WARMUPS, "computes_per_sample": 1,
                        "wall_scope": "encode-submit-synchronous-completion",
                        "gpu_scope": "one command containing all projection dispatches",
                        "excludes": ["Hadamard transform", "allocation", "pipeline-compilation", "output-poison", "readback", "oracle"],
                        "samples": samples},
                    "validation": {"independent_oracle": "literal-pq2-f64", "weight_patterns_per_row": 4,
                        "input_operand_accumulator_dtype": "f32", "output_dtype": "f16",
                        "error_bound": "original-f32-forward-bound-plus-final-f16-store",
                        "output_bitwise_equal": true, "guards_and_immutable_inputs_verified": true,
                        "before_and_after_timing": true, "model_quality_validated": false}
                })
            );
        }
    }
}
