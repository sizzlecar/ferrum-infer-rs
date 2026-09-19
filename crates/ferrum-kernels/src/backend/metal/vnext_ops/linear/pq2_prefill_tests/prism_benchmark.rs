//! Explicit, ignored comparison with a caller-provided Prism release library.
//! Input/weight bytes match; each backend retains its native operand/output ABI.

use super::super::super::native_blocks::{
    pq2_full_tiles_supported, pq2_full_tiles_vector_input_supported,
};
use super::*;

impl Fixture {
    fn literal_half_operand_reference(&self) -> Vec<Vec<(f64, f64)>> {
        let width = self.params.in_features as usize;
        let columns = if self.column_reference.is_some() {
            self.params.out_features as usize
        } else {
            (self.params.out_features as usize).min(4)
        };
        (0..self.params.rows as usize)
            .map(|row| {
                (0..columns)
                    .map(|column| {
                        let mut sum = 0.0;
                        let mut absolute_sum = 0.0;
                        for k in 0..width {
                            let block = k / 128;
                            let offset = WEIGHT_PREFIX + (column * (width / 128) + block) * 34;
                            let scale = f16::from_le_bytes([
                                self.weight_bytes[offset],
                                self.weight_bytes[offset + 1],
                            ])
                            .to_f32();
                            // No production decoder: use the fixture's literal
                            // PQ2 coefficients, then cast each operand to half.
                            let coefficient =
                                COEFFICIENTS[(column + block + (k % 128) / 4) % 4][k % 4];
                            let weight = f16::from_f32(scale * coefficient as f32).to_f32();
                            let input =
                                f16::from_f32(self.input_values[INPUT_PREFIX + row * width + k])
                                    .to_f32();
                            let product = f64::from(input) * f64::from(weight);
                            sum += product;
                            absolute_sum += product.abs();
                        }
                        (sum, absolute_sum)
                    })
                    .collect()
            })
            .collect()
    }

    fn validate_prism_f32(
        &self,
        actual: &[f32],
        half_reference: &[Vec<(f64, f64)>],
    ) -> serde_json::Value {
        let columns = self.params.out_features as usize;
        assert_eq!(actual.len(), self.params.rows as usize * columns);
        let mut maximum_error = 0.0_f64;
        let mut maximum_bound_fraction = 0.0_f64;
        let mut maximum_full_f32_difference = 0.0_f64;
        let mut full_f32_squared_difference = 0.0_f64;
        let mut outside_full_f32_bound = 0_usize;
        for row in 0..self.params.rows as usize {
            for column in 0..columns {
                let value = f64::from(actual[row * columns + column]);
                let (full_expected, full_absolute_sum) = self
                    .column_reference
                    .as_ref()
                    .map_or(self.reference[row][column % 4], |reference| {
                        reference[row][column]
                    });
                let (expected, absolute_sum) =
                    half_reference[row][column % half_reference[row].len()];
                let n_u = (2 * self.params.in_features + 8) as f64 * f64::from(f32::EPSILON) / 2.0;
                // Prism casts operands to half and accumulates/stores F32.
                // Its own oracle does not get Ferrum's F16-store allowance.
                let bound = n_u / (1.0 - n_u) * absolute_sum;
                let error = (value - expected).abs();
                assert!(
                    value.is_finite() && error <= bound,
                    "Prism half-operands/F32 output ({row},{column}) {value} != {expected}, bound={bound}",
                );
                maximum_error = maximum_error.max(error);
                if bound > 0.0 {
                    maximum_bound_fraction = maximum_bound_fraction.max(error / bound);
                }
                let difference = (value - full_expected).abs();
                maximum_full_f32_difference = maximum_full_f32_difference.max(difference);
                full_f32_squared_difference += difference * difference;
                outside_full_f32_bound +=
                    usize::from(difference > n_u / (1.0 - n_u) * full_absolute_sum);
            }
        }
        serde_json::json!({
            "outputs_checked": actual.len(), "non_finite_outputs": 0,
            "operand_dtype": "f16", "accumulator_dtype": "f32", "output_dtype": "f32",
            "max_absolute_error_vs_half_operand_literal_f64": maximum_error,
            "max_fraction_of_f32_forward_bound": maximum_bound_fraction,
            "max_absolute_difference_vs_full_f32_literal": maximum_full_f32_difference,
            "rms_difference_vs_full_f32_literal": (full_f32_squared_difference / actual.len() as f64).sqrt(),
            "outputs_outside_full_f32_operand_bound": outside_full_f32_bound,
            "full_f32_operand_equivalence_claimed": false,
            "f16_rounding_allowance": false,
        })
    }
}

const ALIGNED_PREFIX_BYTES: usize = 64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ControlVariant {
    Production,
    GuardedM64,
    FullTilesM32,
    FullTilesM64,
}

const CONTROL_VARIANTS: [ControlVariant; 4] = [
    ControlVariant::Production,
    ControlVariant::GuardedM64,
    ControlVariant::FullTilesM32,
    ControlVariant::FullTilesM64,
];

fn checked_variant(params: LinearParams, requested: ControlVariant) -> ControlVariant {
    let row_tile = match requested {
        ControlVariant::Production | ControlVariant::GuardedM64 => return requested,
        ControlVariant::FullTilesM32 => 32,
        ControlVariant::FullTilesM64 => 64,
    };
    if pq2_full_tiles_supported(
        row_tile,
        params.rows,
        params.in_features,
        params.out_features,
    ) {
        requested
    } else {
        ControlVariant::Production
    }
}

// Separate benchmark bindings keep the original adversarial-offset fixtures
// intact while comparing naturally aligned, contiguous output matrices.
struct AlignedControl<'a> {
    source: &'a Fixture,
    params: LinearParams,
    input: Buffer,
    input_region_offset_bytes: u64,
    input_offset_bytes: u64,
    weight: Buffer,
    output: Buffer,
    input_values: Vec<f32>,
    weight_bytes: Vec<u8>,
    initial_output: Vec<f16>,
}

impl<'a> AlignedControl<'a> {
    fn new(device: &Device, source: &'a Fixture) -> Self {
        Self::with_input_binding(device, source, ALIGNED_PREFIX_BYTES as u64, 0)
    }

    fn with_input_binding(
        device: &Device,
        source: &'a Fixture,
        input_region_offset_bytes: u64,
        input_offset_bytes: u64,
    ) -> Self {
        let input_start = input_region_offset_bytes
            .checked_add(input_offset_bytes)
            .expect("fixture input binding must not overflow");
        assert!(input_start.is_multiple_of(4));
        let values = source.params.rows as usize * source.params.in_features as usize;
        let mut input_values = vec![GUARD; input_start as usize / 4];
        input_values.extend_from_slice(&source.input_values[INPUT_PREFIX..INPUT_PREFIX + values]);
        input_values.extend([GUARD; ALIGNED_PREFIX_BYTES / 4]);
        let mut weight_bytes = vec![0xcc; ALIGNED_PREFIX_BYTES];
        weight_bytes
            .extend_from_slice(&source.weight_bytes[WEIGHT_PREFIX..source.weight_bytes.len() - 16]);
        weight_bytes.extend([0xcc; ALIGNED_PREFIX_BYTES]);
        let outputs = source.params.rows as usize * source.params.out_features as usize;
        let mut initial_output = vec![f16::from_f32(GUARD); ALIGNED_PREFIX_BYTES / 2];
        initial_output.resize(initial_output.len() + outputs, f16::NAN);
        initial_output.extend([f16::from_f32(GUARD); ALIGNED_PREFIX_BYTES / 2]);
        Self {
            source,
            params: LinearParams {
                output_stride: source.params.out_features,
                output_column_offset: 0,
                ..source.params
            },
            input: buffer(device, &input_values),
            input_region_offset_bytes,
            input_offset_bytes,
            weight: buffer(device, &weight_bytes),
            output: buffer(device, &initial_output),
            input_values,
            weight_bytes,
            initial_output,
        }
    }

    fn run_timed(
        &self,
        pipelines: &MetalLinearPipelines,
        queue: &CommandQueueRef,
        requested: ControlVariant,
    ) -> (f64, Option<f64>) {
        let (pipeline, dispatch) = match checked_variant(self.params, requested) {
            ControlVariant::Production => pipelines.hadamard_native_dispatch(
                GgufBlockFormat::Pq2_0,
                ElementType::F16,
                self.params,
            ),
            // Fixed old control: never route this through the production
            // selector, which may now choose the optimized full-tile PSO.
            ControlVariant::GuardedM64 => (
                pipelines
                    .native
                    .pq2_gemm_input_f32_output_f16_m64
                    .as_ref()
                    .expect("guarded M64 control requires a supported pipeline"),
                LinearDispatchKind::NativeTiledGemmM64,
            ),
            ControlVariant::FullTilesM32 => (
                pipelines
                    .native
                    .pq2_gemm_input_f32_output_f16_full_tiles
                    .as_ref()
                    .expect("full-tile M32 experiment requires a supported pipeline"),
                LinearDispatchKind::NativeTiledGemm,
            ),
            ControlVariant::FullTilesM64 => (
                pipelines
                    .native
                    .pq2_gemm_input_f32_output_f16_m64_full_tiles
                    .as_ref()
                    .expect("full-tile M64 experiment requires a supported pipeline"),
                LinearDispatchKind::NativeTiledGemmM64,
            ),
        };
        self.run_pipeline_timed(pipeline, dispatch, queue)
    }

    fn vector_input_pipeline<'p>(
        &self,
        pipelines: &'p MetalLinearPipelines,
    ) -> Option<&'p metal::ComputePipelineState> {
        if !pq2_full_tiles_vector_input_supported(
            self.params.rows,
            self.params.in_features,
            self.params.out_features,
            self.input_region_offset_bytes,
            self.input_offset_bytes,
            self.input.length() - self.input_region_offset_bytes,
        ) {
            return None;
        }
        pipelines
            .native
            .pq2_gemm_input_f32_output_f16_m64_full_tiles_vector_input
            .as_ref()
    }

    fn run_vector_input_timed(
        &self,
        pipelines: &MetalLinearPipelines,
        queue: &CommandQueueRef,
    ) -> (f64, Option<f64>) {
        if let Some(pipeline) = self.vector_input_pipeline(pipelines) {
            self.run_pipeline_timed(pipeline, LinearDispatchKind::NativeTiledGemmM64, queue)
        } else {
            self.run_timed(pipelines, queue, ControlVariant::Production)
        }
    }

    fn run_pipeline_timed(
        &self,
        pipeline: &metal::ComputePipelineState,
        dispatch: LinearDispatchKind,
        queue: &CommandQueueRef,
    ) -> (f64, Option<f64>) {
        // SAFETY: every prior command completed; this allocation has the
        // recorded F16 length. Output initialization is outside timing.
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.initial_output.as_ptr(),
                self.output.contents().cast::<f16>(),
                self.initial_output.len(),
            );
        }
        let input_start = self
            .input_region_offset_bytes
            .checked_add(self.input_offset_bytes)
            .expect("validated fixture input offset");
        // Match synchronous graph_compute: encode + submit + completion.
        // Pipeline creation, allocation, poisoning and readback are excluded.
        let started = std::time::Instant::now();
        let command = queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(&self.input), input_start);
        encoder.set_buffer(1, Some(&self.weight), ALIGNED_PREFIX_BYTES as u64);
        encoder.set_buffer(2, Some(&self.output), ALIGNED_PREFIX_BYTES as u64);
        encoder.set_bytes(
            3,
            std::mem::size_of::<LinearParams>() as u64,
            &self.params as *const _ as *const c_void,
        );
        bind_native_block(encoder, GgufBlockFormat::Pq2_0, 4);
        dispatch_linear_grid(encoder, self.params, dispatch);
        encoder.end_encoding();
        command.commit();
        command.wait_until_completed();
        let wall_ns = started.elapsed().as_secs_f64() * 1e9;
        assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
        (wall_ns, super::super::microbench::gpu_elapsed_ns(command))
    }

    fn validate(&self) -> Vec<u16> {
        // SAFETY: synchronized runs finish before validation; all allocations
        // have their recorded initialized lengths and element types.
        let actual = unsafe {
            std::slice::from_raw_parts(
                self.output.contents().cast::<f16>(),
                self.initial_output.len(),
            )
        };
        let input = unsafe {
            std::slice::from_raw_parts(self.input.contents().cast::<f32>(), self.input_values.len())
        };
        let weight = unsafe {
            std::slice::from_raw_parts(self.weight.contents().cast::<u8>(), self.weight_bytes.len())
        };
        assert_eq!(input, self.input_values);
        assert_eq!(weight, self.weight_bytes);
        let prefix = ALIGNED_PREFIX_BYTES / 2;
        let count = self.params.rows as usize * self.params.out_features as usize;
        for (index, value) in actual.iter().enumerate() {
            if !(prefix..prefix + count).contains(&index) {
                assert_eq!(*value, f16::from_f32(GUARD), "aligned output guard {index}");
                continue;
            }
            let row = (index - prefix) / self.params.out_features as usize;
            let column = (index - prefix) % self.params.out_features as usize;
            let (expected, absolute_sum) = self
                .source
                .column_reference
                .as_ref()
                .map_or(self.source.reference[row][column % 4], |reference| {
                    reference[row][column]
                });
            // Exactly the original full-F32 fixture bound plus F16 store.
            let n_u = (2 * self.params.in_features + 8) as f64 * f64::from(f32::EPSILON) / 2.0;
            let accumulation = n_u / (1.0 - n_u) * absolute_sum;
            let rounding = (expected.abs() + accumulation) * 2_f64.powi(-11) + 2_f64.powi(-25);
            assert!(
                value.is_finite()
                    && (f64::from(value.to_f32()) - expected).abs() <= accumulation + rounding,
                "Ferrum aligned output ({row},{column}) {} != {expected}, bound={}",
                value.to_f32(),
                accumulation + rounding,
            );
        }
        actual.iter().map(|value| value.to_bits()).collect()
    }
}

#[test]
fn pq2_vector_input_requires_complete_aligned_in_bounds_binding() {
    const MATRIX_BYTES: u64 = 64 * 128 * 4;
    for (region, inner) in [(16, 0), (4, 12), (64, 16)] {
        let end = inner + MATRIX_BYTES;
        assert!(pq2_full_tiles_vector_input_supported(
            64, 128, 64, region, inner, end
        ));
        assert!(!pq2_full_tiles_vector_input_supported(
            64,
            128,
            64,
            region,
            inner,
            end - 1
        ));
    }
    // Eligibility depends on the complete physical binding, not the outer
    // region's alignment or the workspace-local offset by itself.
    for (region, inner) in [(16, 4), (4, 16), (0, 1)] {
        assert!(!pq2_full_tiles_vector_input_supported(
            64,
            128,
            64,
            region,
            inner,
            u64::MAX
        ));
    }
    for (rows, width, outputs) in [
        (0, 128, 64),
        (63, 128, 64),
        (65, 128, 64),
        (64, 0, 64),
        (64, 32, 64),
        (64, 129, 64),
        (64, 128, 0),
        (64, 128, 65),
    ] {
        assert!(!pq2_full_tiles_vector_input_supported(
            rows,
            width,
            outputs,
            16,
            0,
            u64::MAX
        ));
    }
    assert!(!pq2_full_tiles_vector_input_supported(
        64,
        128,
        64,
        u64::MAX,
        1,
        u64::MAX
    ));
    assert!(!pq2_full_tiles_vector_input_supported(
        64,
        128,
        64,
        u64::MAX - 15,
        0,
        u64::MAX
    ));
    // Individually legal tile dimensions, but rows*K*sizeof(float) overflows.
    assert!(!pq2_full_tiles_vector_input_supported(
        u32::MAX - 63,
        u32::MAX - 127,
        64,
        0,
        0,
        u64::MAX
    ));
}

#[test]
fn pq2_vector_input_preserves_f32_oracle_and_unaligned_fallback_on_metal() {
    let device = Device::system_default().expect("PQ2 vector-input conformance requires Metal");
    let mut pipelines = MetalLinearPipelines::new(&device).unwrap();
    let queue = device.new_command_queue();
    for (region, inner, wide_scale, distinct_columns, expected_candidate) in [
        (16, 0, false, false, true),
        (4, 12, false, true, true),
        (16, 4, false, false, false),
        (64, 16, true, false, true),
    ] {
        let mut fixture = Fixture::new(&device, 64, 384, 1024, wide_scale);
        if distinct_columns {
            fixture.use_distinct_weight_rows(&device);
        }
        let control = AlignedControl::with_input_binding(&device, &fixture, region, inner);
        assert_eq!(
            control.vector_input_pipeline(&pipelines).is_some(),
            expected_candidate
        );
        let _ = control.run_timed(&pipelines, &queue, ControlVariant::FullTilesM64);
        let expected = control.validate();
        let _ = control.run_vector_input_timed(&pipelines, &queue);
        assert_eq!(control.validate(), expected);
        let _ = control.run_vector_input_timed(&pipelines, &queue);
        assert_eq!(control.validate(), expected);
    }
    // Losing an optional PSO must submit the original pipeline even when the
    // shape and full physical input offset satisfy every vector-load guard.
    pipelines
        .native
        .pq2_gemm_input_f32_output_f16_m64_full_tiles_vector_input = None;
    let fixture = Fixture::new(&device, 64, 128, 1024, false);
    let control = AlignedControl::new(&device, &fixture);
    assert!(control.vector_input_pipeline(&pipelines).is_none());
    let _ = control.run_timed(&pipelines, &queue, ControlVariant::Production);
    let expected = control.validate();
    let _ = control.run_vector_input_timed(&pipelines, &queue);
    assert_eq!(control.validate(), expected);
}

#[test]
#[ignore = "GPU timing experiment: coordinate exclusive device access"]
fn pq2_vector_input_paired_gpu_microbench() {
    let device = Device::system_default().expect("PQ2 vector-input timing requires Metal");
    let pipelines = MetalLinearPipelines::new(&device).unwrap();
    let queue = device.new_command_queue();
    const WARMUPS: usize = 2;
    const REPEATS: usize = 5;
    for (name, width, outputs) in [
        ("ffn_gate_or_up", 5120, 17408),
        ("ffn_down", 17408, 5120),
        ("gdn_output", 6144, 5120),
    ] {
        let fixture = Fixture::new(&device, 512, width, outputs, false);
        let control = AlignedControl::new(&device, &fixture);
        assert!(control.vector_input_pipeline(&pipelines).is_some());
        let _ = control.run_timed(&pipelines, &queue, ControlVariant::FullTilesM64);
        let expected = control.validate();
        let _ = control.run_vector_input_timed(&pipelines, &queue);
        assert_eq!(control.validate(), expected);
        for repeat in 0..WARMUPS {
            let order = if repeat % 2 == 0 { [0, 1] } else { [1, 0] };
            for variant in order {
                if variant == 0 {
                    let _ = control.run_timed(&pipelines, &queue, ControlVariant::FullTilesM64);
                } else {
                    let _ = control.run_vector_input_timed(&pipelines, &queue);
                }
            }
        }
        let mut samples = Vec::new();
        for repeat in 0..REPEATS {
            let order = if repeat % 2 == 0 { [0, 1] } else { [1, 0] };
            let mut wall = [0.0; 2];
            let mut gpu = [0.0; 2];
            for variant in order {
                let timing = if variant == 0 {
                    control.run_timed(&pipelines, &queue, ControlVariant::FullTilesM64)
                } else {
                    control.run_vector_input_timed(&pipelines, &queue)
                };
                wall[variant] = timing.0;
                gpu[variant] = timing.1.expect("completed GPU command exposes timestamps");
            }
            samples.push(serde_json::json!({
                "repeat": repeat, "order": order,
                "scalar_full_m64_wall_ns": wall[0], "vector_input_wall_ns": wall[1],
                "scalar_full_m64_gpu_ns": gpu[0], "vector_input_gpu_ns": gpu[1],
                "vector_over_scalar_full_m64_wall": wall[1] / wall[0],
                "vector_over_scalar_full_m64_gpu": gpu[1] / gpu[0],
            }));
        }
        let _ = control.run_timed(&pipelines, &queue, ControlVariant::FullTilesM64);
        assert_eq!(control.validate(), expected);
        let _ = control.run_vector_input_timed(&pipelines, &queue);
        assert_eq!(control.validate(), expected);
        let timing = serde_json::json!({
            "variants": ["scalar_full_m64", "vector_input_full_m64"],
            "initial_validation_computes_per_variant": 1,
            "warmups_per_variant": WARMUPS, "computes_per_sample": 1,
            "wall_scope": "encode-submit-synchronous-completion",
            "excludes": ["allocation", "pipeline-compilation", "output-poison", "readback", "oracle"],
            "samples": samples,
        });
        let validation = serde_json::json!({
            "independent_oracle": "literal-pq2-f64", "weight_patterns_per_row": 4,
            "input_operand_accumulator_dtype": "f32", "output_dtype": "f16",
            "error_bound": "original-f32-forward-bound-plus-final-f16-store",
            "output_bitwise_equal": true, "input_weight_and_output_guards_verified": true,
            "before_and_after_timing": true, "model_quality_validated": false,
        });
        println!(
            "{}",
            serde_json::json!({
                "schema_version": 1, "kind": "pq2_vector_input_paired_gpu_microbench",
                "device": device.name(), "shape": name,
                "rows": 512, "in_features": width, "out_features": outputs,
                "weight_format": "quantization.gguf.pq2-0",
                "input_region_offset_bytes": control.input_region_offset_bytes,
                "input_inner_offset_bytes": control.input_offset_bytes,
                "output_stride": control.params.output_stride,
                "timing": timing, "validation": validation,
                "baseline": "fixed original full-M64 PSO; bypasses evolving production selector",
            })
        );
    }
}

#[test]
fn pq2_full_tiles_reject_partial_and_invalid_shapes() {
    for row_tile in [32, 64] {
        for (rows, width, outputs) in [(row_tile, 128, 64), (row_tile * 2, 384, 128)] {
            assert!(pq2_full_tiles_supported(row_tile, rows, width, outputs));
        }
        for (rows, width, outputs) in [
            (0, 128, 64),
            (row_tile - 1, 128, 64),
            (row_tile + 1, 128, 64),
            (row_tile, 0, 64),
            (row_tile, 32, 64),
            (row_tile, 127, 64),
            (row_tile, 129, 64),
            (row_tile, 128, 0),
            (row_tile, 128, 63),
            (row_tile, 128, 65),
        ] {
            assert!(!pq2_full_tiles_supported(row_tile, rows, width, outputs));
            let params = LinearParams {
                rows,
                in_features: width,
                out_features: outputs,
                output_stride: outputs,
                output_column_offset: 0,
            };
            let requested = if row_tile == 32 {
                ControlVariant::FullTilesM32
            } else {
                ControlVariant::FullTilesM64
            };
            assert_eq!(
                checked_variant(params, requested),
                ControlVariant::Production
            );
        }
    }
    assert!(!pq2_full_tiles_supported(0, 64, 128, 64));
    assert!(!pq2_full_tiles_supported(16, 64, 128, 64));
    assert!(!pq2_full_tiles_supported(128, 128, 128, 64));
    assert!(!pq2_full_tiles_supported(64, 32, 128, 64));
}

#[test]
fn pq2_full_tiles_preserve_f32_oracle_and_partial_fallback_on_metal() {
    let device = Device::system_default().expect("PQ2 full-tile conformance requires Metal");
    let pipelines = MetalLinearPipelines::new(&device).unwrap();
    let queue = device.new_command_queue();
    for (rows, width, outputs, wide_scale) in [
        (32, 128, 1024, false),
        (64, 384, 1024, false),
        (64, 384, 1024, true),
        (33, 384, 65, false),
    ] {
        let mut fixture = Fixture::new(&device, rows, width, outputs, wide_scale);
        if !wide_scale {
            fixture.use_distinct_weight_rows(&device);
        }
        let control = AlignedControl::new(&device, &fixture);
        let _ = control.run_timed(&pipelines, &queue, ControlVariant::Production);
        let expected = control.validate();
        for candidate in [ControlVariant::FullTilesM32, ControlVariant::FullTilesM64] {
            if rows == 33 {
                assert_eq!(
                    checked_variant(control.params, candidate),
                    ControlVariant::Production
                );
            }
            let _ = control.run_timed(&pipelines, &queue, candidate);
            assert_eq!(control.validate(), expected);
        }
    }
}

#[test]
#[ignore = "requires FERRUM_GGML_REFERENCE_DIR containing trusted Prism release libraries; exclusive GPU access"]
fn pq2_prism_reference_synchronized_wall_microbench() {
    let requested_dir = std::env::var_os("FERRUM_GGML_REFERENCE_DIR")
        .expect("set FERRUM_GGML_REFERENCE_DIR to the trusted Prism release library directory");
    let directory = std::path::PathBuf::from(requested_dir)
        .canonicalize()
        .expect("FERRUM_GGML_REFERENCE_DIR must name an existing directory");
    assert!(
        directory.is_dir(),
        "FERRUM_GGML_REFERENCE_DIR must be a directory"
    );
    let reference = prism_reference::Reference::load(&directory)
        .expect("failed to load the explicitly supplied Prism reference libraries");
    let device = Device::system_default().expect("PQ2 Prism comparison requires Metal");
    let pipelines = MetalLinearPipelines::new(&device).unwrap();
    let queue = device.new_command_queue();
    const WARMUPS: usize = 2;
    const REPEATS: usize = 5;
    for (name, width, outputs) in [
        ("ffn_gate_or_up", 5120, 17408),
        ("ffn_down", 17408, 5120),
        ("gdn_output", 6144, 5120),
    ] {
        let fixture = Fixture::new(&device, 512, width, outputs, false);
        let half_reference = fixture.literal_half_operand_reference();
        let control = AlignedControl::new(&device, &fixture);
        let weight_payload = &fixture.weight_bytes[WEIGHT_PREFIX..fixture.weight_bytes.len() - 16];
        let input_payload = &fixture.input_values
            [INPUT_PREFIX..INPUT_PREFIX + fixture.params.rows as usize * width as usize];
        let graph = reference
            .matrix(512, width, outputs, weight_payload, input_payload)
            .expect("Prism must support this exact PQ2/F32 single-matmul graph on Metal");
        let production_dispatch = pipelines
            .hadamard_native_dispatch(GgufBlockFormat::Pq2_0, ElementType::F16, control.params)
            .1;
        let _ = control.run_timed(&pipelines, &queue, ControlVariant::Production);
        let ferrum_output_before = control.validate();
        for candidate in CONTROL_VARIANTS.into_iter().skip(1) {
            assert_eq!(checked_variant(control.params, candidate), candidate);
            let _ = control.run_timed(&pipelines, &queue, candidate);
            assert_eq!(control.validate(), ferrum_output_before);
        }
        graph
            .run_timed()
            .expect("initial Prism graph compute failed");
        let prism_output_before = graph.read_output().expect("Prism output readback failed");
        let prism_validation_before =
            fixture.validate_prism_f32(&prism_output_before, &half_reference);
        for repeat in 0..WARMUPS {
            let order = if repeat % 2 == 0 {
                [0, 1, 2, 3, 4]
            } else {
                [4, 3, 2, 1, 0]
            };
            for engine in order {
                if engine < CONTROL_VARIANTS.len() {
                    let _ = control.run_timed(&pipelines, &queue, CONTROL_VARIANTS[engine]);
                } else {
                    graph.run_timed().expect("Prism warmup compute failed");
                }
            }
        }
        let mut samples = Vec::new();
        for repeat in 0..REPEATS {
            let order = if repeat % 2 == 0 {
                [0, 1, 2, 3, 4]
            } else {
                [4, 3, 2, 1, 0]
            };
            let mut ferrum_wall_ns = [0.0; 4];
            let mut ferrum_gpu_ns = [None; 4];
            let mut prism_wall_ns = 0.0;
            for engine in order {
                if engine < CONTROL_VARIANTS.len() {
                    let timing = control.run_timed(&pipelines, &queue, CONTROL_VARIANTS[engine]);
                    ferrum_wall_ns[engine] = timing.0;
                    ferrum_gpu_ns[engine] = timing.1;
                } else {
                    prism_wall_ns = graph
                        .run_timed()
                        .expect("Prism measured compute failed")
                        .as_secs_f64()
                        * 1e9;
                }
            }
            samples.push(serde_json::json!({
                "repeat": repeat, "order": order,
                "ferrum_synchronized_wall_ns": ferrum_wall_ns[0],
                "prism_synchronized_wall_ns": prism_wall_ns,
                "ferrum_gpu_ns": ferrum_gpu_ns[0],
                "ferrum_over_prism_synchronized_wall": ferrum_wall_ns[0] / prism_wall_ns,
                "guarded_m64": {
                    "wall_ns": ferrum_wall_ns[1], "gpu_ns": ferrum_gpu_ns[1],
                    "over_prism_wall": ferrum_wall_ns[1] / prism_wall_ns,
                    "production_over_guarded_wall": ferrum_wall_ns[0] / ferrum_wall_ns[1],
                },
                "full_tiles": {
                    "m32_wall_ns": ferrum_wall_ns[2], "m64_wall_ns": ferrum_wall_ns[3],
                    "m32_gpu_ns": ferrum_gpu_ns[2], "m64_gpu_ns": ferrum_gpu_ns[3],
                    "m32_over_guarded_m64_wall": ferrum_wall_ns[2] / ferrum_wall_ns[1],
                    "m64_over_guarded_m64_wall": ferrum_wall_ns[3] / ferrum_wall_ns[1],
                },

            }));
        }
        for candidate in CONTROL_VARIANTS {
            let _ = control.run_timed(&pipelines, &queue, candidate);
            assert_eq!(control.validate(), ferrum_output_before);
        }
        let prism_output_after = graph.read_output().expect("Prism final readback failed");
        let prism_validation_after =
            fixture.validate_prism_f32(&prism_output_after, &half_reference);
        assert!(
            prism_output_before
                .iter()
                .zip(&prism_output_after)
                .all(|(before, after)| before.to_bits() == after.to_bits()),
            "Prism reference output changed across repeats",
        );
        let reference_metadata = serde_json::json!({
            "library_directory": directory,
            "library_paths": reference.library_paths(),
            "backend": reference.backend_name(), "device": reference.device_description(),
            "version": reference.version(), "commit": reference.commit(),
        });
        let layout = serde_json::json!({
            "rows": 512, "in_features": width, "out_features": outputs,
            "weight_format": "quantization.gguf.pq2-0", "input_dtype": "f32",
            "same_input_and_packed_weight_payload": true,
            "output_abi_difference": "Ferrum writes contiguous F16; Prism writes contiguous F32",
            "ferrum_output_stride": control.params.output_stride,
            "ferrum_input_weight_output_base_alignment_bytes": ALIGNED_PREFIX_BYTES,
            "reference_output_stride": outputs, "synthetic_weight_row_period": 4,
        });
        let numerics = serde_json::json!({
            "ferrum_output_dtype": "f16", "reference_output_dtype": "f32",
            "ferrum_operand_dtype": "f32", "reference_operand_dtype": "f16",
            "reference_operand_casts": "half(input), half(decoded-f32-weight); F32 accumulation",
            "independent_oracle": "literal-pq2-f64-with-each-backends-operand-precision",
            "oracle_weight_patterns_per_row": 4,
            "ferrum_validation": "original-f32-forward-bound-plus-final-f16-rounding-and-guards",
            "reference_validation_before": prism_validation_before,
            "reference_validation_after": prism_validation_after,
            "full_tiles_bitwise_equal_to_production": true,
            "model_quality_validated": false,
        });
        let timing = serde_json::json!({
            "scope": "command-encoding-or-graph-compute-through-synchronous-completion",
            "excludes": ["initialization", "allocation", "upload", "output-poison", "readback", "oracle"],
            "hadamard_in_timed_operation": false,
            "variant_names": ["production", "guarded_m64", "full_tiles_m32", "full_tiles_m64", "prism"],
            "guarded_m64_control": "original guarded PSO bound directly; independent of production selection",
            "production_label_scope": "shape-only scalar fallback; binding-aware vector dispatch is measured separately",
            "initial_validation_computes_per_variant": 1,
            "warmups_per_variant": WARMUPS, "computes_per_sample": 1,
            "samples": samples,
        });
        println!(
            "{}",
            serde_json::json!({
                "schema_version": 1, "kind": "pq2_prism_reference_synchronized_wall_microbench",
                "device": device.name(), "shape": name,
                "ferrum_dispatch": format!("{production_dispatch:?}"),
                "reference": reference_metadata, "layout": layout,
                "numerics": numerics, "timing": timing,
                "control_selection": "shape-only scalar fallback, including full-tile M64 when eligible",
            })
        );
    }
}
