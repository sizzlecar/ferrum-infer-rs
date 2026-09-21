//! Q6_K F32-output grouped dispatch oracle and opt-in large-head comparison.

use super::*;

#[derive(Clone, Copy, Debug)]
enum HeadMapping {
    CurrentGemv,
    PreparedGroups,
}

impl HeadMapping {
    fn run(
        self,
        fixture: &Fixture,
        pipelines: &MetalLinearPipelines,
        queue: &CommandQueueRef,
        iterations: u32,
    ) -> (f64, Option<f64>) {
        match self {
            Self::CurrentGemv => fixture.run(pipelines, queue, false, iterations),
            Self::PreparedGroups => fixture.run_mode(pipelines, queue, None, iterations),
        }
    }
}

#[test]
fn q6_f32_grouped_head_matches_cpu_with_dense_inputs_offsets_and_tails() {
    let composition =
        MetalVNextComposition::create(DeviceId::new("metal.q6-f32-groups.test").unwrap()).unwrap();
    let runtime = composition.runtime();
    let pipelines = MetalLinearPipelines::new(runtime.device()).unwrap();
    let queue = runtime.device().new_command_queue();
    // K spans three Q6 blocks. Row tails and the bounded/narrow fallbacks use
    // the actual production plan. Both PSOs retain the same F64 tolerance;
    // their separately compiled reductions need not produce identical bits.
    // retained regions have nonzero input/weight/output/column offsets.
    for (rows, outputs) in [
        (9, 1025),
        (10, 1025),
        (11, 1025),
        (31, 1025),
        (32, 1025),
        (33, 1025),
        (32, 65),
    ] {
        let fixture = Fixture::with_dense_input(
            runtime,
            rows,
            768,
            outputs,
            GgufBlockFormat::Q6K,
            ElementType::F32,
            true,
        );
        HeadMapping::CurrentGemv.run(&fixture, &pipelines, &queue, 1);
        let baseline = fixture.validate();
        HeadMapping::PreparedGroups.run(&fixture, &pipelines, &queue, 1);
        let candidate = fixture.validate();
        assert_eq!(candidate.len(), baseline.len());
        let first_difference =
            candidate
                .iter()
                .zip(&baseline)
                .enumerate()
                .find_map(|(index, (&actual, &old))| {
                    (actual != old).then_some((index, f32::from_bits(actual), f32::from_bits(old)))
                });
        let maximum_absolute_difference = candidate
            .iter()
            .zip(&baseline)
            .map(|(&actual, &old)| {
                (f64::from(f32::from_bits(actual)) - f64::from(f32::from_bits(old))).abs()
            })
            .fold(0.0_f64, f64::max);
        println!(
            "q6_f32_grouped_head rows={rows} K=768 N={outputs} bitwise_equal={} first_difference={first_difference:?} maximum_absolute_difference={maximum_absolute_difference:e}",
            first_difference.is_none()
        );
        if fixture.launch.dispatch_count() == 1 {
            assert!(
                first_difference.is_none(),
                "unchanged fallback rows={rows} N={outputs}: {first_difference:?}"
            );
        }
        HeadMapping::PreparedGroups.run(&fixture, &pipelines, &queue, 2);
        for (index, (actual, expected)) in fixture.validate().iter().zip(&candidate).enumerate() {
            assert_eq!(
                actual, expected,
                "same prepared route rows={rows} N={outputs} index={index}"
            );
        }
        let mut truncated = fixture.regions.clone();
        let bytes = truncated[2].length_bytes();
        truncated[2] = truncated[2].test_subregion(0..bytes - 4).unwrap();
        assert!(validate_launch_regions(&truncated, &[fixture.launch]).is_err());
    }
}

#[test]
#[ignore = "9B Q6 F32 head diagnosis; exclusive Metal access and memory required"]
fn q6_f32_grouped_head_microbench() {
    const ITERATIONS: u32 = 2;
    let composition =
        MetalVNextComposition::create(DeviceId::new("metal.q6-f32-groups.bench").unwrap()).unwrap();
    let runtime = composition.runtime();
    let pipelines = MetalLinearPipelines::new(runtime.device()).unwrap();
    let queue = runtime.device().new_command_queue();
    // Exact cached 9B head geometry and encoding. Synthetic encoded weights,
    // sparse F32 inputs, independent full-output F64 oracle; no dense weight
    // materialization, model-quality claim or performance acceptance threshold.
    let fixture = Fixture::new(
        runtime,
        32,
        4096,
        248_320,
        GgufBlockFormat::Q6K,
        ElementType::F32,
    );
    let forward = [HeadMapping::CurrentGemv, HeadMapping::PreparedGroups];
    let reverse = [HeadMapping::PreparedGroups, HeadMapping::CurrentGemv];
    HeadMapping::CurrentGemv.run(&fixture, &pipelines, &queue, 1);
    let baseline = fixture.validate();
    // One complete forward/reverse warmup pair, then three forward/reverse
    // measured pairs. Every mapping executes in every order; no selected runs.
    for order in 0..8 {
        for mapping in if order % 2 == 0 { forward } else { reverse } {
            let (wall_ns, gpu_ns) = mapping.run(&fixture, &pipelines, &queue, ITERATIONS);
            let bits = fixture.validate();
            let gpu_ns = gpu_ns.expect("diagnostic requires completed GPU timestamps");
            if order >= 2 {
                println!(
                    "{}",
                    serde_json::json!({
                        "benchmark": "q6_f32_grouped_head", "mapping": format!("{mapping:?}"),
                        "measured_order": order - 2, "pair": (order - 2) / 2,
                        "rows": 32, "input": 4096, "output": 248320,
                        "activation_type": "f32", "weight_format": "q6_k",
                        "output_type": "f32", "projection_iterations": ITERATIONS,
                        "physical_dispatches_per_iteration": if matches!(mapping, HeadMapping::PreparedGroups) { fixture.launch.dispatch_count() } else { 1 },
                        "command_encode_submit_wait_wall_ns": wall_ns,
                        "command_gpu_ns": gpu_ns,
                        "encode_submit_wait_wall_ns": wall_ns / f64::from(ITERATIONS),
                        "gpu_ns": gpu_ns / f64::from(ITERATIONS),
                        "bitwise_equal_to_current_gemv": bits == baseline,
                        "oracle": "independent_f64_same_existing_f32_tolerance",
                    })
                );
            }
        }
    }
}
