//! Isolate 256-thread PQ2 decode in the unchanged prefill math/layout.

use super::*;
use serde_json::json;

pub(super) fn pipeline<'a>(
    pipelines: &'a MetalLinearPipelines,
    fixture: &Fixture,
    weight_threads: bool,
) -> (&'a ComputePipelineState, LinearDispatchKind) {
    if pq2_full_tiles_vector_input_supported(
        fixture.params.rows,
        fixture.params.in_features,
        fixture.params.out_features,
        0,
        fixture.input_offset_bytes,
        fixture.input.length(),
    ) {
        let baseline = pipelines
            .native
            .pq2_gemm_input_f32_output_f16_m64_full_tiles_vector_input_control
            .as_ref();
        let selected = if weight_threads {
            pipelines
                .native
                .pq2_gemm_input_f32_output_f16_m64_full_tiles_vector_input
                .as_ref()
        } else {
            baseline
        };
        if let Some(selected) = selected {
            return (selected, LinearDispatchKind::NativeTiledGemmM64);
        }
    }
    pipelines.hadamard_native_dispatch(GgufBlockFormat::Pq2_0, ElementType::F16, fixture.params)
}

#[test]
fn pq2_weight_threads_cover_every_packed_byte_on_metal() {
    let device = Device::system_default().expect("PQ2 weight-thread conformance requires Metal");
    let pipelines = MetalLinearPipelines::new(&device).unwrap();
    let queue = device.new_command_queue();
    let mut fixture = Fixture::new(&device, 64, 128, 256, false);
    for column in 0..256_usize {
        let offset = WEIGHT_PREFIX + column * 34;
        fixture.weight_bytes[offset..offset + 2].copy_from_slice(&0x3800_u16.to_le_bytes());
        fixture.weight_bytes[offset + 2..offset + 34].fill(column as u8);
    }
    fixture.weight = buffer(&device, &fixture.weight_bytes);
    fixture.column_reference = Some(
        (0..64_usize)
            .map(|row| {
                (0..256_usize)
                    .map(|packed| {
                        let mut sum = 0.0_f64;
                        let mut absolute_sum = 0.0_f64;
                        for k in 0..128_usize {
                            // All byte patterns use an independent F64 dot oracle;
                            // the candidate changes only their thread ownership.
                            let code = [-1.0, 0.0, 1.0, 2.0][(packed >> (2 * (k % 4))) & 3];
                            let x = f64::from(fixture.input_values[INPUT_PREFIX + row * 128 + k]);
                            let product = x * (0.5 * code);
                            sum += product;
                            absolute_sum += product.abs();
                        }
                        (sum, absolute_sum)
                    })
                    .collect()
            })
            .collect(),
    );
    assert!(pipelines
        .native
        .pq2_gemm_input_f32_output_f16_m64_full_tiles_vector_input
        .is_some());
    let _ = fixture.run_tile(&pipelines, &queue, PrefillTile::ScalarWeightControlM64, 1);
    let baseline = fixture.validate();
    let _ = fixture.run_tile(&pipelines, &queue, PrefillTile::WeightThreadsM64, 1);
    assert_eq!(fixture.validate(), baseline);
}

#[test]
fn pq2_weight_threads_preserve_f32_math_layout_and_fallbacks_on_metal() {
    let device = Device::system_default().expect("PQ2 weight-thread conformance requires Metal");
    let mut pipelines = MetalLinearPipelines::new(&device).unwrap();
    let queue = device.new_command_queue();
    let candidate = pipelines
        .native
        .pq2_gemm_input_f32_output_f16_m64_full_tiles_vector_input
        .as_ref()
        .expect("weight-thread conformance requires its optional M64 pipeline");
    for (rows, width, outputs, wide_scale) in [
        (64, 128, 64, false),
        (128, 384, 1024, false),
        (64, 384, 1024, true),
        (192, 256, 128, false),
    ] {
        let mut fixture = Fixture::new(&device, rows, width, outputs, wide_scale);
        if !wide_scale {
            fixture.use_distinct_weight_rows(&device);
        }
        // Original adversarial fixture: packed offset18, K384 row pitch102,
        // strided output with a column offset, and independent row/column data.
        assert!(std::ptr::eq(
            pipeline(&pipelines, &fixture, true).0,
            candidate
        ));
        let _ = fixture.run_tile(&pipelines, &queue, PrefillTile::ScalarWeightControlM64, 1);
        let baseline = fixture.validate();
        let _ = fixture.run_tile(&pipelines, &queue, PrefillTile::WeightThreadsM64, 1);
        assert_eq!(fixture.validate(), baseline, "changed F32 MMA result bits");
    }

    for (rows, outputs, input_offset) in [(65, 1024, 16), (64, 1025, 16), (64, 1024, 20)] {
        let mut fixture = Fixture::new(&device, rows, 384, outputs, false);
        if input_offset == 20 {
            // Move the identical payload by one F32; precomputed oracle values
            // stay unchanged while only the bound input address loses float4 alignment.
            fixture.input_values.insert(INPUT_PREFIX, GUARD);
            fixture.input = buffer(&device, &fixture.input_values);
            fixture.input_offset_bytes = input_offset;
        }
        let expected = pipelines.hadamard_native_dispatch(
            GgufBlockFormat::Pq2_0,
            ElementType::F16,
            fixture.params,
        );
        assert!(std::ptr::eq(
            pipeline(&pipelines, &fixture, true).0,
            expected.0
        ));
        let _ = fixture.run_tile(&pipelines, &queue, PrefillTile::Production, 1);
        let baseline = fixture.validate();
        let _ = fixture.run_tile(&pipelines, &queue, PrefillTile::WeightThreadsM64, 1);
        assert_eq!(fixture.validate(), baseline);
    }

    let fixture = Fixture::new(&device, 64, 384, 1024, false);
    let _ = fixture.run_tile(&pipelines, &queue, PrefillTile::ScalarWeightControlM64, 1);
    let baseline = fixture.validate();
    pipelines
        .native
        .pq2_gemm_input_f32_output_f16_m64_full_tiles_vector_input = None;
    assert!(std::ptr::eq(
        pipeline(&pipelines, &fixture, true).0,
        pipelines
            .hadamard_native_dispatch(GgufBlockFormat::Pq2_0, ElementType::F16, fixture.params)
            .0,
    ));
    let _ = fixture.run_tile(&pipelines, &queue, PrefillTile::WeightThreadsM64, 1);
    assert_eq!(fixture.validate(), baseline);
}

#[test]
#[ignore = "paired PQ2 weight-thread timing requires exclusive Metal GPU access"]
fn pq2_weight_threads_paired_gpu_microbench() {
    let device = Device::system_default().expect("PQ2 weight-thread timing requires Metal");
    let pipelines = MetalLinearPipelines::new(&device).unwrap();
    let queue = device.new_command_queue();
    let candidate = pipelines
        .native
        .pq2_gemm_input_f32_output_f16_m64_full_tiles_vector_input
        .as_ref()
        .expect("weight-thread benchmark requires its optional M64 pipeline");
    let fixed_baseline = pipelines
        .native
        .pq2_gemm_input_f32_output_f16_m64_full_tiles_vector_input_control
        .as_ref()
        .expect("weight-thread benchmark requires the fixed vector-input M64 pipeline");
    const VARIANTS: [PrefillTile; 2] = [
        PrefillTile::ScalarWeightControlM64,
        PrefillTile::WeightThreadsM64,
    ];
    for (name, width, outputs) in [
        ("ffn_gate_or_up", 5120, 17408),
        ("ffn_down", 17408, 5120),
        ("gdn_output", 6144, 5120),
    ] {
        let fixture = Fixture::new(&device, 512, width, outputs, false);
        assert!(std::ptr::eq(
            pipeline(&pipelines, &fixture, false).0,
            fixed_baseline
        ));
        assert!(std::ptr::eq(
            pipeline(&pipelines, &fixture, true).0,
            candidate
        ));
        let _ = fixture.run_tile(&pipelines, &queue, VARIANTS[0], 1);
        let baseline = fixture.validate();
        let _ = fixture.run_tile(&pipelines, &queue, VARIANTS[1], 1);
        assert_eq!(fixture.validate(), baseline);
        for _ in 0..2 {
            for variant in VARIANTS {
                let _ = fixture.run_tile(&pipelines, &queue, variant, 1);
            }
        }
        let mut samples = Vec::new();
        for repeat in 0..5 {
            let order = if repeat % 2 == 0 { [0, 1] } else { [1, 0] };
            let mut gpu_ns = [0.0; 2];
            for index in order {
                gpu_ns[index] = fixture
                    .run_tile(&pipelines, &queue, VARIANTS[index], 1)
                    .expect("paired microbenchmark requires valid Metal GPU timestamps");
            }
            samples.push(json!({
                "repeat": repeat, "order": order,
                "vector_input_control_gpu_ns": gpu_ns[0],
                "weight_threads_gpu_ns": gpu_ns[1],
                "candidate_over_control_gpu": gpu_ns[1] / gpu_ns[0],
            }));
        }
        for variant in VARIANTS {
            let _ = fixture.run_tile(&pipelines, &queue, variant, 1);
            assert_eq!(fixture.validate(), baseline);
        }
        println!(
            "{}",
            json!({
                "kind": "pq2_weight_threads_paired_gpu_microbench",
                "device": device.name(), "shape": name, "rows": 512,
                "in_features": width, "out_features": outputs,
                "baseline": "fixed 41d3faee vector-input full-M64 PSO",
                "candidate": "256 threads each decode eight coefficients; original K-by-N shared layout",
                "precision": {"operands": "f32", "accumulator": "f32", "output": "f16"},
                "timing": {"scope": "GPU command elapsed", "dispatches_per_sample": 1,
                    "warmups_per_variant": 2, "samples": samples},
                "validation": {"independent_oracle": "literal-pq2-f64",
                    "original_f32_bound_plus_final_f16_store": true,
                    "before_and_after_timing": true, "output_bitwise_equal": true,
                    "input_weight_output_guards_verified": true,
                    "synthetic_weight_patterns_per_row": 4,
                    "model_quality_validated": false},
            })
        );
    }
}
