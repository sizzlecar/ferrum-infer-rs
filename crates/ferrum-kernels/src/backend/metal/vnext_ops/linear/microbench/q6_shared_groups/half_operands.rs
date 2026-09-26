//! Explicitly different arithmetic: half operands, F32 accumulation and logits.
//! This measures potential only; it does not change the F32 profile or its tests.

use super::*;
use crate::backend::metal::vnext_ops::linear::half_head::{
    half_dispatch_count, validate_half_launch, HalfHeadPipelines,
};

fn run_half(
    fixture: &Fixture,
    pipeline: &HalfHeadPipelines,
    queue: &CommandQueueRef,
) -> (f64, Option<f64>) {
    validate_half_launch(&fixture.regions, fixture.launch, &[]).unwrap();
    let initial = activation_bytes(&fixture.initial_output, ElementType::F32);
    // SAFETY: the fixture owns this shared allocation; preceding GPU work has
    // completed. Output initialization and all validation are outside timing.
    unsafe {
        std::ptr::copy_nonoverlapping(
            initial.as_ptr(),
            fixture.output.contents().cast::<u8>(),
            initial.len(),
        );
    }
    let started = Instant::now();
    let command = queue.new_command_buffer();
    let encoder = command.new_compute_command_encoder();
    // Exercise the actual opt-in provider's precise compile options, small
    // half-operand kernels, tiled kernel and production dispatch selection.
    pipeline.dispatch(encoder, &fixture.regions, fixture.launch);
    encoder.end_encoding();
    command.commit();
    command.wait_until_completed();
    let wall_ns = started.elapsed().as_secs_f64() * 1e9;
    assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
    (wall_ns, gpu_elapsed_ns(command))
}

struct HalfOracle {
    values: Vec<f64>,
    bounds: Vec<f64>,
    rounded_inputs: usize,
    rounded_weight_products: usize,
}

impl HalfOracle {
    fn new(fixture: &Fixture) -> Self {
        let width = fixture.params.in_features as usize;
        let outputs = fixture.params.out_features as usize;
        let row_bytes = width / 256 * 210;
        let mut values = Vec::with_capacity(fixture.params.rows as usize * outputs);
        let mut bounds = Vec::with_capacity(values.capacity());
        let mut rounded_inputs = 0;
        let mut rounded_weight_products = 0;
        // For these finite normal operands, a half x half product is exact in
        // F32. gamma_K * sum(abs(products)) bounds F32 accumulation roundoff;
        // this is a new-contract oracle, never a relaxed old-profile tolerance.
        let ku = width as f64 * 2.0_f64.powi(-24);
        assert!(ku < 1.0);
        let gamma = (ku / (1.0 - ku)).next_up();
        // This fixture's half inputs have |x| <= 1, and its rounded weights
        // have |w| <= 16 on a 2^-9 lattice. Products lie on a 2^-33 lattice;
        // this bound makes both F64 sums exact, including cancellation.
        assert!(width as f64 * 16.0 * 2.0_f64.powi(33) <= 2.0_f64.powi(53));
        for row in 0..fixture.params.rows as usize {
            let entries: Vec<_> = fixture.input_values
                [INPUT_PREFIX + row * width..INPUT_PREFIX + (row + 1) * width]
                .iter()
                .enumerate()
                .filter(|(_, value)| **value != 0.0)
                .map(|(column, value)| {
                    let rounded = f16::from_f32(*value).to_f32();
                    rounded_inputs += usize::from(rounded.to_bits() != value.to_bits());
                    assert!(rounded.is_finite());
                    assert!(rounded.abs() <= 1.0);
                    assert!(rounded == 0.0 || rounded.abs() >= f16::MIN_POSITIVE.to_f32());
                    (column, rounded)
                })
                .collect();
            for output in 0..outputs {
                let mut sum = 0.0_f64;
                let mut sum_abs = 0.0_f64;
                for &(column, input) in &entries {
                    let offset = WEIGHT_PREFIX + output * row_bytes + column / 256 * 210;
                    let decoded = GgufBlockFormat::Q6K
                        .decode_value(&fixture.weight_bytes[offset..offset + 210], column % 256);
                    let weight = f16::from_f32(decoded).to_f32();
                    rounded_weight_products += usize::from(weight.to_bits() != decoded.to_bits());
                    assert!(weight.is_finite());
                    assert!(weight.abs() <= 16.0 && (weight * 512.0).fract() == 0.0);
                    assert!(weight == 0.0 || weight.abs() >= f16::MIN_POSITIVE.to_f32());
                    let product = f64::from(input) * f64::from(weight);
                    // Avoid a hidden underflow assumption in the error bound.
                    assert!(product == 0.0 || product.abs() >= f64::from(f32::MIN_POSITIVE));
                    assert!(product.abs() <= f64::from(f32::MAX));
                    sum += product;
                    sum_abs += product.abs();
                }
                values.push(sum);
                bounds.push(if sum_abs == 0.0 {
                    0.0
                } else {
                    (gamma * sum_abs).next_up()
                });
            }
        }
        Self {
            values,
            bounds,
            rounded_inputs,
            rounded_weight_products,
        }
    }

    fn validate(&self, fixture: &Fixture, strict_bits: &[u32]) -> (Vec<u32>, serde_json::Value) {
        assert_eq!(fixture.activation_type, ElementType::F32);
        let input =
            read_linear_values(&fixture.input, fixture.input_values.len(), ElementType::F32);
        assert_eq!(input, fixture.input_values);
        // SAFETY: this shared allocation has the recorded length, and the
        // command completed before readback. Quantized data is read-only.
        let weight = unsafe {
            std::slice::from_raw_parts(
                fixture.weight.contents().cast::<u8>(),
                fixture.weight_bytes.len(),
            )
        };
        assert_eq!(
            &weight[..WEIGHT_PREFIX],
            &fixture.weight_bytes[..WEIGHT_PREFIX]
        );
        assert_eq!(
            &weight[weight.len() - 16..],
            &fixture.weight_bytes[weight.len() - 16..]
        );
        let actual = read_linear_values(
            &fixture.output,
            fixture.initial_output.len(),
            ElementType::F32,
        );
        assert_eq!(actual.len(), strict_bits.len());
        let stride = fixture.params.output_stride as usize;
        let outputs = fixture.params.out_features as usize;
        let mut max_oracle_abs = 0.0_f64;
        let mut max_bound_fraction = 0.0_f64;
        let mut max_strict_abs = 0.0_f64;
        let mut max_strict_relative_nonzero = 0.0_f64;
        let mut strict_zero_candidate_nonzero = 0;
        let mut changed_elements = 0;
        let mut outputs_not_representable_as_half = 0;
        for (index, &value) in actual.iter().enumerate() {
            let location = index
                .checked_sub(OUTPUT_PREFIX)
                .filter(|index| *index < fixture.params.rows as usize * stride)
                .filter(|index| {
                    (COLUMN_OFFSET..COLUMN_OFFSET + outputs).contains(&(index % stride))
                });
            if let Some(location) = location {
                let logical = location / stride * outputs + location % stride - COLUMN_OFFSET;
                let expected = self.values[logical];
                let difference = (f64::from(value) - expected).abs();
                let bound = self.bounds[logical];
                assert!(value.is_finite() && difference <= bound,
                    "half operand oracle rows={} logical={logical}: {value} != {expected}, error={difference}, bound={bound}", fixture.params.rows);
                max_oracle_abs = max_oracle_abs.max(difference);
                if bound > 0.0 {
                    max_bound_fraction = max_bound_fraction.max(difference / bound);
                }
                let strict = f32::from_bits(strict_bits[index]);
                assert!(strict.is_finite());
                let strict_difference = (f64::from(value) - f64::from(strict)).abs();
                max_strict_abs = max_strict_abs.max(strict_difference);
                if strict != 0.0 {
                    max_strict_relative_nonzero = max_strict_relative_nonzero
                        .max(strict_difference / f64::from(strict).abs());
                } else if value != 0.0 {
                    strict_zero_candidate_nonzero += 1;
                }
                changed_elements += usize::from(value.to_bits() != strict.to_bits());
                outputs_not_representable_as_half +=
                    usize::from(f16::from_f32(value).to_f32().to_bits() != value.to_bits());
            } else {
                assert_eq!(
                    value.to_bits(),
                    fixture.initial_output[index].to_bits(),
                    "guard {index}"
                );
            }
        }
        (
            actual.iter().map(|value| value.to_bits()).collect(),
            serde_json::json!({
                "oracle": "independent_f64_sum_of_half_rounded_input_and_q6_weights",
                "oracle_bound": "gamma_K_F32_times_sum_abs_products",
                "checked_output_elements": self.values.len(),
                "max_oracle_absolute_error": max_oracle_abs,
                "max_oracle_error_bound_fraction": max_bound_fraction,
                "strict_f32_output_different_elements": changed_elements,
                "max_absolute_difference_from_strict_f32": max_strict_abs,
                "max_relative_difference_from_nonzero_strict_f32": max_strict_relative_nonzero,
                "strict_zero_candidate_nonzero_outputs": strict_zero_candidate_nonzero,
                "outputs_not_representable_as_half": outputs_not_representable_as_half,
                "input_elements_changed_by_half_rounding": self.rounded_inputs,
                "oracle_product_weights_changed_by_half_rounding": self.rounded_weight_products,
            }),
        )
    }
}

fn sample(
    fixture: &Fixture,
    pipelines: &MetalLinearPipelines,
    candidate: &HalfHeadPipelines,
    queue: &CommandQueueRef,
    half: bool,
) -> serde_json::Value {
    let (wall_ns, gpu_ns) = if half {
        run_half(fixture, candidate, queue)
    } else {
        fixture.run_mode(pipelines, queue, None, 1)
    };
    let gpu_ns = gpu_ns.expect("half-operands experiment requires actual GPU timestamps");
    assert!(gpu_ns.is_finite() && gpu_ns > 0.0);
    assert!(wall_ns.is_finite() && wall_ns > 0.0);
    serde_json::json!({
        "mapping": if half { "HalfOperandsF32Logits" } else { "PreparedGroups" },
        "projection_iterations": 1,
        "physical_dispatches_per_iteration": if half { half_dispatch_count(fixture.params.rows) } else { fixture.launch.dispatch_count() },
        "command_encode_submit_wait_wall_ns": wall_ns, "command_gpu_ns": gpu_ns,
        "encode_submit_wait_wall_ns": wall_ns, "gpu_ns": gpu_ns,
    })
}

#[test]
fn q6_half_operands_f32_logits_cpu_offsets_strides_alignment_and_tails() {
    let composition =
        MetalVNextComposition::create(DeviceId::new("metal.q6-half.test").unwrap()).unwrap();
    let runtime = composition.runtime();
    let pipelines = MetalLinearPipelines::new(runtime.device()).unwrap();
    let candidate = HalfHeadPipelines::new(runtime.device()).unwrap();
    let queue = runtime.device().new_command_queue();
    for (rows, width, outputs) in [
        (1, 256, 65),
        (2, 256, 67),
        (3, 512, 129),
        (4, 768, 1025),
        (8, 768, 1025),
        (9, 768, 1025),
        (31, 768, 1025),
        (32, 768, 1025),
        (33, 512, 129),
    ] {
        let fixture = Fixture::with_dense_input(
            runtime,
            rows,
            width,
            outputs,
            GgufBlockFormat::Q6K,
            ElementType::F32,
            true,
        );
        let oracle = HalfOracle::new(&fixture);
        sample(&fixture, &pipelines, &candidate, &queue, false);
        let strict = fixture.validate();
        let mut previous = None;
        for repeat in 0..2 {
            let timing = sample(&fixture, &pipelines, &candidate, &queue, true);
            let (bits, validation) = oracle.validate(&fixture, &strict);
            if let Some(previous) = previous {
                assert_eq!(bits, previous, "same half-operands invocation changed bits");
            }
            previous = Some(bits);
            println!(
                "{}",
                serde_json::json!({
                    "kind": "q6_half_operands_f32_logits_correctness", "rows": rows, "input": width, "output": outputs,
                    "repeat": repeat, "intentionally_different_arithmetic_contract": true,
                    "input_type": "f32", "operands": "f16", "accumulation": "f32", "output_type": "f32",
                    "input_offset_bytes": fixture.regions[0].offset_bytes() + fixture.launch.input_offset_bytes,
                    "weight_offset_bytes": fixture.regions[1].offset_bytes(),
                    "output_offset_bytes": fixture.regions[2].offset_bytes() + fixture.launch.output_offset_bytes,
                    "output_stride": fixture.params.output_stride, "output_column_offset": COLUMN_OFFSET,
                    "timing": timing, "validation": validation,
                })
            );
        }
        let mut misaligned = fixture.launch;
        // Move into the retained prefix so the scalar F32 span still fits;
        // rejection must come from the vector alignment requirement.
        misaligned.input_offset_bytes = misaligned.input_offset_bytes.checked_sub(4).unwrap();
        validate_launch_regions(&fixture.regions, &[misaligned]).unwrap();
        assert!(validate_half_launch(&fixture.regions, misaligned, &[]).is_err());
        for index in [
            fixture.launch.input_region,
            fixture.launch.weight_region,
            fixture.launch.output_region,
        ] {
            let mut truncated = fixture.regions.clone();
            let bytes = truncated[index].length_bytes();
            truncated[index] = truncated[index].test_subregion(0..bytes - 4).unwrap();
            assert!(validate_half_launch(&truncated, fixture.launch, &[]).is_err());
        }
    }
}

#[test]
#[ignore = "different numerical contract: coordinate exclusive Metal access"]
fn q6_half_operands_f32_logits_9b_head_microbench() {
    let composition =
        MetalVNextComposition::create(DeviceId::new("metal.q6-half.bench").unwrap()).unwrap();
    let runtime = composition.runtime();
    let pipelines = MetalLinearPipelines::new(runtime.device()).unwrap();
    let candidate = HalfHeadPipelines::new(runtime.device()).unwrap();
    let queue = runtime.device().new_command_queue();
    for rows in [1, 2, 3, 4, 8, 32] {
        let fixture = Fixture::new(
            runtime,
            rows,
            4096,
            248_320,
            GgufBlockFormat::Q6K,
            ElementType::F32,
        );
        let oracle = HalfOracle::new(&fixture);
        sample(&fixture, &pipelines, &candidate, &queue, false);
        let strict = fixture.validate();
        sample(&fixture, &pipelines, &candidate, &queue, true);
        let (half_baseline, _) = oracle.validate(&fixture, &strict);
        let mut samples = Vec::new();
        for round in 0..8 {
            for half in if round % 2 == 0 {
                [false, true]
            } else {
                [true, false]
            } {
                let mut timing = sample(&fixture, &pipelines, &candidate, &queue, half);
                // Validate every projection before the next output reset. No
                // decode/oracle, memory reset, or readback enters timing.
                timing["validation"] = if half {
                    let (bits, validation) = oracle.validate(&fixture, &strict);
                    assert_eq!(
                        bits, half_baseline,
                        "same half-operands invocation changed bits"
                    );
                    validation
                } else {
                    assert_eq!(
                        fixture.validate(),
                        strict,
                        "current prepared groups changed bits"
                    );
                    serde_json::json!({ "oracle": "unchanged_existing_strict_f32_f64_reference", "checked_output_elements": oracle.values.len(), "same_contract_bitwise_differences": 0 })
                };
                timing["round"] = round.into();
                timing["warmup"] = (round < 2).into();
                samples.push(timing);
            }
        }
        println!(
            "{}",
            serde_json::json!({
                "kind": "q6_half_operands_f32_logits_9b_head_microbench", "device": runtime.device().name(),
                "rows": rows, "input": 4096, "output": 248320, "weight_format": "q6_k",
                "input_type": "f32", "candidate_operands": "f16", "accumulation": "f32", "output_type": "f32",
                "intentionally_different_arithmetic_contract": true, "default_numerical_profile_changed": false,
                "scope": "hot_synthetic_q6_head_projection_not_model_quality_or_throughput",
                "control": "current_production_prepared_dispatch", "candidate": "opt_in_half_head_production_dispatch_and_precise_compile_options",
                "weight_staging": "none_original_q6_buffer_shared_by_both_routes", "candidate_fast_math": false,
                "samples": samples,
            })
        );
    }
}
