//! Production aligned-scale PQ2 GEMV versus the retained byte-load pipeline.
use super::*;
use metal::CommandQueueRef;

fn candidate(pipelines: &MetalLinearPipelines, output: ElementType) -> &ComputePipelineState {
    match output {
        ElementType::F16 => pipelines
            .native
            .pq2_linear_f32_f16_complete_aligned_scale
            .as_ref(),
        ElementType::F32 => pipelines
            .native
            .pq2_linear_f32_complete_aligned_scale
            .as_ref(),
        _ => unreachable!(),
    }
    .expect("test requires the production aligned-scale pipeline")
}

fn control(pipelines: &MetalLinearPipelines, output: ElementType) -> &ComputePipelineState {
    match output {
        ElementType::F16 => pipelines.native.pq2_linear_f32_f16_complete.as_ref(),
        ElementType::F32 => pipelines.native.pq2_linear_f32_complete.as_ref(),
        _ => unreachable!(),
    }
    .expect("test requires the unchanged production complete-eight pipeline")
}

fn run(
    fixture: &Fixture,
    pipeline: &ComputePipelineState,
    queue: &CommandQueueRef,
    repetitions: usize,
) -> f64 {
    // Fixture::run_pipeline binds its packed weight buffer at byte18, not at
    // a 4/16-byte boundary; K384 additionally exercises the 102-byte row pitch.
    let p = fixture.params();
    assert!(pq2_complete_outputs_supported(
        p.rows,
        p.in_features,
        p.out_features
    ));
    fixture.poison_output();
    fixture.run_pipeline(
        pipeline,
        LinearDispatchKind::Pq2CooperativeGemv,
        queue,
        repetitions,
    )
}

#[test]
fn pq2_scale_load_preserves_f32_arithmetic_offsets_and_guards() {
    let device = Device::system_default().expect("scale-load conformance requires Metal");
    let production = MetalLinearPipelines::new(&device).unwrap();
    let reference = ReferencePipelines::new(&device);
    let queue = device.new_command_queue();
    let check = |fixture: &Fixture, output, cpu| {
        run(fixture, control(&production, output), &queue, 1);
        let expected = fixture.read();
        assert!(expected.iter().all(|x| x.is_finite()));
        if cpu {
            fixture.assert_cpu();
        }
        for pipeline in [candidate(&production, output), reference.get(output)] {
            run(fixture, pipeline, &queue, 1);
            assert_same_bits(&fixture.read(), &expected);
            if cpu {
                fixture.assert_cpu();
            }
            fixture.assert_inputs_unchanged();
        }
        expected
    };
    for output in [ElementType::F16, ElementType::F32] {
        for (rows, width, outputs) in [(1, 128, 16), (4, 384, 32), (1, 5120, 16)] {
            check(
                &Fixture::new(&device, rows, width, outputs, output, true),
                output,
                true,
            );
        }
        // Positive/negative normal, subnormal and maximum finite half scales;
        // all packed codes; signed zero. Inputs keep F16 outputs finite.
        for scale in [0x3555, 0xb955, 0x0001, 0x03ff, 0, 0x8000, 0x7bff, 0xfbff] {
            let mut fixture = Fixture::new(&device, 1, 384, 16, output, true);
            let divisor = if scale & 0x7fff == 0x7bff {
                65536.0
            } else {
                1.0
            };
            fixture.replace_payload(
                &device,
                (0..384)
                    .map(|i| (i as f32 * 0.0171).sin() / divisor)
                    .collect(),
                weights(384, 16, scale, |i| i as u8),
            );
            check(&fixture, output, true);
        }
        let mut wide = Fixture::new(&device, 1, 384, 16, output, true);
        wide.replace_payload(
            &device,
            (0..384)
                .map(|i| if i % 2 == 0 { 131008.0 } else { -131008.0 })
                .collect(),
            weights(384, 16, 0x0001, |i| if i % 2 == 0 { 0xff } else { 0xe4 }),
        );
        check(&wide, output, true);
        wide.replace_payload(
            &device,
            (0..384)
                .map(|i| if i % 2 == 0 { 0.0 } else { -0.0 })
                .collect(),
            weights(384, 16, 0xb955, |i| i as u8),
        );
        check(&wide, output, true);
    }
    // No scale hoisting: cancellation must remain finite with the same
    // per-element scale/product and lane-local accumulation order.
    let mut cancellation = Fixture::new(&device, 1, 640, 16, ElementType::F32, true);
    let mut input = vec![0.0; 640];
    input[..16].fill(2.0_f32.powi(108));
    input[512..528].fill(2.0_f32.powi(108));
    cancellation.replace_payload(
        &device,
        input,
        weights(
            640,
            16,
            0x7800,
            |i| if (i / 32) % 5 == 4 { 0xff } else { 0 },
        ),
    );
    let actual = check(&cancellation, ElementType::F32, true);
    assert_eq!(actual[6], 2.0_f32.powi(127));
    let mut subnormal = Fixture::new(&device, 1, 128, 16, ElementType::F32, true);
    subnormal.replace_payload(
        &device,
        (0..128).map(|i| f32::from_bits(1 + i)).collect(),
        weights(128, 16, 0x7bff, |_| 0xff),
    );
    // Match existing Metal subnormal-input behavior, without imposing CPU FTZ.
    check(&subnormal, ElementType::F32, false);
}

#[test]
#[ignore = "isolated Metal paired GPU timing; exclude other GPU/compiler work"]
fn pq2_scale_load_paired_gpu_timing() {
    const DISPATCHES: usize = 64;
    let device = Device::system_default().expect("scale-load timing requires Metal");
    let production = MetalLinearPipelines::new(&device).unwrap();
    let queue = device.new_command_queue();
    for (width, outputs) in [(5120, 17408), (17408, 5120)] {
        for output in [ElementType::F16, ElementType::F32] {
            let fixture = Fixture::new(&device, 1, width, outputs, output, false);
            let pipelines = [control(&production, output), candidate(&production, output)];
            run(&fixture, pipelines[0], &queue, 1);
            let expected = fixture.read();
            fixture.assert_cpu();
            run(&fixture, pipelines[1], &queue, 1);
            assert_same_bits(&fixture.read(), &expected);
            for _ in 0..2 {
                for pipeline in pipelines {
                    run(&fixture, pipeline, &queue, DISPATCHES);
                }
            }
            let mut samples: [Vec<f64>; 2] = Default::default();
            for pair in 0..5 {
                for index in if pair % 2 == 0 { [0, 1] } else { [1, 0] } {
                    let us = run(&fixture, pipelines[index], &queue, DISPATCHES) * 1e6
                        / DISPATCHES as f64;
                    assert!(us.is_finite() && us > 0.0);
                    samples[index].push(us);
                    assert_same_bits(&fixture.read(), &expected);
                }
            }
            fixture.assert_inputs_unchanged();
            let median = |values: &[f64]| {
                let mut values = values.to_vec();
                values.sort_by(f64::total_cmp);
                values[values.len() / 2]
            };
            eprintln!(
                "PQ2_SCALE_LOAD_GPU {}",
                serde_json::json!({
                    "rows": 1, "width": width, "outputs": outputs,
                    "output": format!("{output:?}"), "timing": "gpu",
                    "control": "production_complete8_uchar_scale",
                    "candidate": "complete8_aligned_ushort_scale",
                    "weight_byte_offset": 18, "warmups": 2, "pairs": 5,
                    "dispatches": DISPATCHES,
                    "control_us": samples[0], "candidate_us": samples[1],
                    "ratio_of_medians": median(&samples[1]) / median(&samples[0]),
                })
            );
        }
    }
}
